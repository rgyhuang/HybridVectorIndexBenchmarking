#!/usr/bin/env python3
"""
Dynamic Selectivity-Variant ACORN Benchmark

This benchmark measures performance under dynamic workloads with:
- Inserts, deletes, and searches interleaved
- Multiple selectivity levels and update frequency

Key metrics:
- QPS (queries per second)
- Recall@k
- Insert latency (ms)
- Delete latency (ms)
"""

import numpy as np
import time
import faiss
import os
import sys
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Set
from ACORNIndex import ACORNIndex

# --- Interfaces ---

class HybridIndex(ABC):
    @abstractmethod
    def build(self, vectors: np.ndarray, ids: np.ndarray, attrs: np.ndarray):
        """Build initial index from data."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int, target_attr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors with given attribute."""
        pass
    
    @abstractmethod
    def insert(self, vector: np.ndarray, doc_id: int, attr: int):
        """Insert a single vector with metadata."""
        pass

    @abstractmethod
    def delete(self, doc_id: int):
        """Delete a vector by ID."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Return current number of vectors in index."""
        pass


# --- Implementations ---

class ACORNIndexWrapper(HybridIndex):
    """
    ACORN: Predicate-agnostic hybrid index.
    
    Supports efficient insert/delete/search without pre-indexing predicates.
    This is the only method that can handle arbitrary predicates efficiently.
    """
    
    def __init__(self, dim, M=32, gamma=1, M_beta=64, efSearch=128):
        self.dim = dim
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.efSearch = efSearch
        self.index = None
        self._size = 0
        
    def build(self, vectors, ids, attrs):
        self.index = ACORNIndex(
            dimension=self.dim,
            M=self.M,
            gamma=self.gamma,
            M_beta=self.M_beta,
            efSearch=self.efSearch
        )
        
        n = len(vectors)
        batch_size = 5000
        
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_vecs = vectors[i:end]
            batch_ids = [int(x) for x in ids[i:end]]
            batch_metas = [{'category': int(a)} for a in attrs[i:end]]
            self.index.insert_batch(batch_vecs, batch_metas, batch_ids)
        
        self._size = n

    def search(self, queries, k, target_attr) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search_category_batch(queries, int(target_attr), k)
        return D, I

    def insert(self, vector, doc_id, attr):
        self.index.insert(vector, {'category': int(attr)}, int(doc_id))
        self._size += 1

    def delete(self, doc_id):
        self.index.delete(int(doc_id))
        self._size -= 1
    
    def size(self) -> int:
        return self._size


class PreFilterIndex(HybridIndex):
    """
    Pre-filtering WITHOUT pre-indexed predicates (realistic scenario).
    
    At query time:
    1. Scan vectors to find those matching the predicate
    2. Perform exact search on matching vectors
    
    This is realistic because we can't pre-index arbitrary predicates.
    Insert/delete are O(1) since we just update the data store.
    Search is O(n_matching) where n_matching is vectors matching predicate.
    """
    
    def __init__(self, dim):
        self.dim = dim
        self.vectors: Dict[int, np.ndarray] = {}  # doc_id -> vector
        self.attrs: Dict[int, int] = {}  # doc_id -> attr
        self.ids_by_attr: Dict[int, Set[int]] = {}  # attr -> set of doc_ids
        
    def build(self, vectors, ids, attrs):
        for i in range(len(vectors)):
            doc_id = int(ids[i])
            attr = int(attrs[i])
            self.vectors[doc_id] = vectors[i].copy()
            self.attrs[doc_id] = attr
            
            if attr not in self.ids_by_attr:
                self.ids_by_attr[attr] = set()
            self.ids_by_attr[attr].add(doc_id)
    
    def search(self, queries, k, target_attr) -> Tuple[np.ndarray, np.ndarray]:
        nq = queries.shape[0]
        
        # Get matching doc_ids
        if target_attr not in self.ids_by_attr:
            return np.full((nq, k), float('inf')), np.full((nq, k), -1, dtype='int64')
        
        matching_ids = list(self.ids_by_attr[target_attr])
        n_matching = len(matching_ids)
        
        if n_matching == 0:
            return np.full((nq, k), float('inf')), np.full((nq, k), -1, dtype='int64')
        
        # Build temporary index on matching vectors (this is the query-time cost)
        matching_vecs = np.array([self.vectors[doc_id] for doc_id in matching_ids], dtype='float32')
        
        index = faiss.IndexFlatL2(self.dim)
        index.add(matching_vecs)
        
        actual_k = min(k, n_matching)
        D, I_local = index.search(queries, actual_k)
        
        # Map local indices to global IDs
        I_global = np.full((nq, k), -1, dtype='int64')
        D_out = np.full((nq, k), float('inf'))
        
        for i in range(nq):
            for j in range(actual_k):
                if I_local[i, j] >= 0:
                    I_global[i, j] = matching_ids[I_local[i, j]]
                    D_out[i, j] = D[i, j]
        
        return D_out, I_global

    def insert(self, vector, doc_id, attr):
        doc_id = int(doc_id)
        attr = int(attr)
        
        self.vectors[doc_id] = vector.copy()
        self.attrs[doc_id] = attr
        
        if attr not in self.ids_by_attr:
            self.ids_by_attr[attr] = set()
        self.ids_by_attr[attr].add(doc_id)

    def delete(self, doc_id):
        doc_id = int(doc_id)
        if doc_id in self.attrs:
            attr = self.attrs[doc_id]
            if attr in self.ids_by_attr:
                self.ids_by_attr[attr].discard(doc_id)
            del self.attrs[doc_id]
            del self.vectors[doc_id]
    
    def size(self) -> int:
        return len(self.vectors)


class PostFilterIndex(HybridIndex):
    """
    Post-filtering with lazy HNSW index rebuild.
    
    Approach:
    - Maintain vectors and metadata in memory
    - Rebuild HNSW index lazily when search is called and data has changed
    - Search the global HNSW with expansion, then filter by predicate
    
    This models a realistic scenario where:
    - Updates are batched and index is rebuilt periodically
    - The rebuild cost is amortized over multiple searches
    
    For fair comparison, we rebuild only when needed (lazy rebuild).
    In production, you might rebuild every N updates or on a timer.
    
    Note: rebuild_interval=0 means rebuild on every search when dirty (strict consistency)
          rebuild_interval=N means rebuild every N updates (eventual consistency)
    """
    
    def __init__(self, dim, expansion_factor=100, rebuild_interval=0):
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.rebuild_interval = rebuild_interval
        self.vectors: Dict[int, np.ndarray] = {}  # doc_id -> vector
        self.attrs: Dict[int, int] = {}  # doc_id -> attr
        self._cached_index = None
        self._cached_ids = None
        self._cache_valid = False
        self._updates_since_rebuild = 0
    
    def build(self, vectors, ids, attrs):
        for i in range(len(vectors)):
            doc_id = int(ids[i])
            self.vectors[doc_id] = vectors[i].copy()
            self.attrs[doc_id] = int(attrs[i])
        self._cache_valid = False
        self._updates_since_rebuild = 0
        # Pre-build index after initial load
        self._ensure_index()
    
    def _ensure_index(self):
        """Rebuild HNSW index if cache is invalid or rebuild interval reached."""
        # Check if rebuild is needed
        need_rebuild = not self._cache_valid
        
        if self.rebuild_interval > 0 and self._updates_since_rebuild >= self.rebuild_interval:
            need_rebuild = True
        
        if not need_rebuild and self._cached_index is not None:
            return
        
        if len(self.vectors) == 0:
            self._cached_index = None
            self._cached_ids = None
            return
        
        # Build HNSW on all current vectors
        self._cached_ids = list(self.vectors.keys())
        all_vecs = np.array([self.vectors[doc_id] for doc_id in self._cached_ids], dtype='float32')
        
        self._cached_index = faiss.IndexHNSWFlat(self.dim, 32)
        self._cached_index.hnsw.efConstruction = 40
        self._cached_index.hnsw.efSearch = 128
        self._cached_index.add(all_vecs)
        
        self._cache_valid = True
        self._updates_since_rebuild = 0
    
    def search(self, queries, k, target_attr) -> Tuple[np.ndarray, np.ndarray]:
        nq = queries.shape[0]
        
        self._ensure_index()
        
        if self._cached_index is None or len(self._cached_ids) == 0:
            return np.full((nq, k), float('inf')), np.full((nq, k), -1, dtype='int64')
        
        # Search with expansion
        k_fetch = min(k * self.expansion_factor, len(self._cached_ids))
        D_raw, I_raw = self._cached_index.search(queries, k_fetch)
        
        # Filter results - note: we use cached_ids which may be stale
        # New inserts won't be found, deleted items may return invalid results
        I_out = np.full((nq, k), -1, dtype='int64')
        D_out = np.full((nq, k), float('inf'))
        
        for i in range(nq):
            count = 0
            for j in range(k_fetch):
                local_idx = I_raw[i, j]
                if local_idx >= 0:
                    doc_id = self._cached_ids[local_idx]
                    # Check if doc still exists and matches attr
                    if doc_id in self.attrs and self.attrs.get(doc_id) == target_attr:
                        I_out[i, count] = doc_id
                        D_out[i, count] = D_raw[i, j]
                        count += 1
                        if count >= k:
                            break
        
        return D_out, I_out

    def insert(self, vector, doc_id, attr):
        doc_id = int(doc_id)
        self.vectors[doc_id] = vector.copy()
        self.attrs[doc_id] = int(attr)
        self._updates_since_rebuild += 1
        if self.rebuild_interval == 0:
            self._cache_valid = False  # Strict: invalidate immediately

    def delete(self, doc_id):
        doc_id = int(doc_id)
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            del self.attrs[doc_id]
            self._updates_since_rebuild += 1
            if self.rebuild_interval == 0:
                self._cache_valid = False  # Strict: invalidate immediately
    
    def size(self) -> int:
        return len(self.vectors)


# --- Data Loading ---

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift1M(base_dir):
    print(f"Loading SIFT1M from {base_dir}...")
    xb = fvecs_read(os.path.join(base_dir, "sift_base.fvecs"))
    xq = fvecs_read(os.path.join(base_dir, "sift_query.fvecs"))
    print(f"Loaded: {xb.shape[0]} base vectors, {xq.shape[0]} queries, dim={xb.shape[1]}")
    return xb, xq

def generate_data(n=10000, d=128, n_attrs=100, seed=42):
    """Generate random vectors with attributes."""
    np.random.seed(seed)
    vectors = np.random.rand(n, d).astype('float32')
    ids = np.arange(n).astype('int64')
    attrs = np.random.randint(0, n_attrs, size=n).astype('int32')
    return vectors, ids, attrs
# --- Evaluation ---

def compute_ground_truth(vectors, ids, attrs, queries, target_attrs, k):
    """Compute exact ground truth for hybrid queries."""
    nq = len(queries)
    gt_I = np.full((nq, k), -1, dtype='int64')
    dim = vectors.shape[1]
    
    # Group by target attribute for efficiency
    unique_targets = np.unique(target_attrs)
    
    for t in unique_targets:
        # Data subset
        mask_data = (attrs == t)
        sub_vecs = vectors[mask_data]
        sub_ids = ids[mask_data]
        
        if len(sub_vecs) == 0:
            continue
            
        # Queries for this target
        mask_query = (target_attrs == t)
        sub_queries = queries[mask_query]
        query_indices = np.where(mask_query)[0]
        
        # Exact search using FAISS
        index_flat = faiss.IndexFlatL2(dim)
        index_flat.add(sub_vecs)
        
        actual_k = min(k, len(sub_vecs))
        D, I = index_flat.search(sub_queries, actual_k)
        
        # Map local indices to global IDs
        for i, q_idx in enumerate(query_indices):
            for j in range(actual_k):
                if I[i, j] >= 0:
                    gt_I[q_idx, j] = sub_ids[I[i, j]]
            
    return gt_I

def calculate_recall(pred_I, gt_I):
    """Calculate recall@k."""
    nq = pred_I.shape[0]
    total_recall = 0
    
    for i in range(nq):
        retrieved = set(pred_I[i])
        retrieved.discard(-1)
        
        relevant = set(gt_I[i])
        relevant.discard(-1)
        
        if len(relevant) == 0:
            total_recall += 1.0
        else:
            total_recall += len(retrieved.intersection(relevant)) / len(relevant)
            
    return total_recall / nq


# --- Benchmarks ---

def run_static_selectivity_benchmark(use_sift=True, n_data=100000):
    """
    Static benchmark across multiple selectivity levels.
    Tests search performance without dynamic updates.
    """
    print("=" * 70)
    print("STATIC SELECTIVITY BENCHMARK")
    print("=" * 70)
    
    if use_sift:
        sift_dir = "ACORN/benchs/sift1M"
        if os.path.exists(sift_dir):
            xb, xq = load_sift1M(sift_dir)
            xb = xb[:n_data]
        else:
            print("SIFT1M not found, using random data")
            xb, _, _ = generate_data(n=n_data, d=128)
            xq = np.random.rand(1000, 128).astype('float32')
    else:
        xb, _, _ = generate_data(n=n_data, d=128)
        xq = np.random.rand(1000, 128).astype('float32')
    
    dim = xb.shape[1]
    k = 10
    n_queries = 100
    
    selectivity_levels = [1.0, 2.0, 5.0, 10.0, 20.0]
    results = []
    
    for selectivity_pct in selectivity_levels:
        print(f"\n{'='*60}")
        print(f"Selectivity = {selectivity_pct}%")
        print(f"{'='*60}")
        
        n_categories = max(1, int(100 / selectivity_pct))
        
        np.random.seed(42)
        attrs = np.random.randint(0, n_categories, size=len(xb)).astype('int32')
        ids = np.arange(len(xb)).astype('int64')
        
        np.random.seed(123)
        target_attrs = np.random.randint(0, n_categories, size=n_queries).astype('int32')
        xq_bench = xq[:n_queries]
        
        print("Computing ground truth...")
        gt_I = compute_ground_truth(xb, ids, attrs, xq_bench, target_attrs, k)
        
        indices = {
        "ACORN-1": ACORNIndexWrapper(dim, M=32, gamma=1, M_beta=64, efSearch=128),
        "ACORN-4": ACORNIndexWrapper(dim, M=32, gamma=4, M_beta=64, efSearch=128),
        "ACORN-8": ACORNIndexWrapper(dim, M=32, gamma=8, M_beta=64, efSearch=128),
        # "ACORN-12": ACORNIndexWrapper(dim, M=32, gamma=12, M_beta=64, efSearch=128),
        "Pre-Filter": PreFilterIndex(dim),
        # "Post-Filter-Strict": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=0),
        "Post-Filter-Batch100": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=100),
        }
        
        for name, idx in indices.items():
            print(f"\nBenchmarking {name}...")
            
            try:
                # Build
                t0 = time.time()
                idx.build(xb, ids, attrs)
                build_time = time.time() - t0
                print(f"  Build: {build_time:.2f}s")
                
                # Search
                unique_targets = np.unique(target_attrs)
                final_I = np.full((n_queries, k), -1, dtype='int64')
                
                t0 = time.time()
                for t_attr in unique_targets:
                    mask = (target_attrs == t_attr)
                    batch_queries = xq_bench[mask]
                    query_indices = np.where(mask)[0]
                    
                    if len(batch_queries) == 0:
                        continue
                    
                    D, I = idx.search(batch_queries, k, t_attr)
                    final_I[query_indices] = I
                
                search_time = time.time() - t0
                qps = n_queries / search_time
                recall = calculate_recall(final_I, gt_I)
                
                print(f"  QPS: {qps:.1f}, Recall@{k}: {recall:.4f}")
                
                results.append({
                    'benchmark': 'static',
                    'selectivity_pct': selectivity_pct,
                    'method': name,
                    'build_time': build_time,
                    'qps': qps,
                    'recall': recall,
                    'avg_insert_ms': 0,
                    'avg_delete_ms': 0,
                })
                
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    return results


def run_dynamic_benchmark(use_sift=True, n_init=50000, n_ops=5000, selectivity_pct=5.0):
    """
    Dynamic benchmark with interleaved insert/search operations.
    
    Workflow:
    1. Build initial index with n_init vectors
    2. Interleave n_ops operations: insert then search
    3. Measure insert latency, search latency, and recall
    """
    print("\n" + "=" * 70)
    print(f"DYNAMIC BENCHMARK (Insert + Search) - {selectivity_pct}% selectivity")
    print("=" * 70)
    
    dim = 128
    k = 10
    n_categories = max(1, int(100 / selectivity_pct))
    
    if use_sift:
        sift_dir = "ACORN/benchs/sift1M"
        if os.path.exists(sift_dir):
            xb, xq = load_sift1M(sift_dir)
            # Use subset
            total_needed = n_init + n_ops
            xb = xb[:min(total_needed, len(xb))]
        else:
            xb, _, _ = generate_data(n=n_init + n_ops, d=dim)
            xq = np.random.rand(n_ops, dim).astype('float32')
    else:
        xb, _, _ = generate_data(n=n_init + n_ops, d=dim)
        xq = np.random.rand(n_ops, dim).astype('float32')
    
    # Split into initial and stream
    np.random.seed(42)
    attrs = np.random.randint(0, n_categories, size=len(xb)).astype('int32')
    ids = np.arange(len(xb)).astype('int64')
    
    init_vecs = xb[:n_init]
    init_ids = ids[:n_init]
    init_attrs = attrs[:n_init]
    
    stream_vecs = xb[n_init:n_init + n_ops]
    stream_ids = ids[n_init:n_init + n_ops]
    stream_attrs = attrs[n_init:n_init + n_ops]
    
    # Pre-generate queries
    np.random.seed(123)
    query_vecs = xq[:n_ops] if len(xq) >= n_ops else np.random.rand(n_ops, dim).astype('float32')
    query_attrs = np.random.randint(0, n_categories, size=n_ops).astype('int32')
    
    indices = {
        "ACORN-1": ACORNIndexWrapper(dim, M=32, gamma=1, M_beta=64, efSearch=128),
        "ACORN-4": ACORNIndexWrapper(dim, M=32, gamma=4, M_beta=64, efSearch=128),
        "ACORN-8": ACORNIndexWrapper(dim, M=32, gamma=8, M_beta=64, efSearch=128),
        "ACORN-12": ACORNIndexWrapper(dim, M=32, gamma=12, M_beta=64, efSearch=128),
        "Pre-Filter": PreFilterIndex(dim),
        # "Post-Filter-Strict": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=0),
        "Post-Filter-Batch100": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=100),
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"\nBenchmarking {name}...")
        
        # Oracle for ground truth (exact search)
        oracle = PreFilterIndex(dim)
        oracle.build(init_vecs.copy(), init_ids.copy(), init_attrs.copy())
        
        # Build initial index
        t0_build = time.time()
        idx.build(init_vecs.copy(), init_ids.copy(), init_attrs.copy())
        build_time = time.time() - t0_build
        print(f"  Build: {build_time:.2f}s ({n_init} vectors)")
        
        insert_times = []
        search_times = []
        recalls = []
        
        # Stream operations
        for i in tqdm(range(n_ops), desc=f"  {name}", leave=False):
            # Update oracle first
            oracle.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            
            # Insert
            t_ins = time.time()
            idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            insert_times.append(time.time() - t_ins)
            
            # Search
            q_vec = query_vecs[i:i+1]
            t_attr = query_attrs[i]
            
            # Ground truth from oracle
            _, gt_I = oracle.search(q_vec, k, t_attr)
            
            t_search = time.time()
            _, pred_I = idx.search(q_vec, k, t_attr)
            search_times.append(time.time() - t_search)
            
            recalls.append(calculate_recall(pred_I, gt_I))
        
        avg_insert = np.mean(insert_times) * 1000  # ms
        avg_search = np.mean(search_times) * 1000  # ms
        qps = n_ops / np.sum(search_times)
        avg_recall = np.mean(recalls)
        
        print(f"  Avg Insert: {avg_insert:.2f}ms")
        print(f"  Avg Search: {avg_search:.2f}ms, QPS: {qps:.1f}")
        print(f"  Recall: {avg_recall:.4f}")
        
        results.append({
            'benchmark': 'dynamic_insert',
            'selectivity_pct': selectivity_pct,
            'method': name,
            'build_time': build_time,
            'qps': qps,
            'recall': avg_recall,
            'avg_insert_ms': avg_insert,
            'avg_delete_ms': 0,
        })
    
    return results


def run_update_benchmark(use_sift=True, n_init=50000, n_ops=2000, selectivity_pct=5.0):
    """
    Update benchmark with interleaved insert/delete/search operations.
    
    Workflow:
    1. Build initial index with n_init vectors
    2. For each operation: insert new vector, delete random existing, search
    3. Index size stays roughly constant
    """
    print("\n" + "=" * 70)
    print(f"UPDATE BENCHMARK (Insert + Delete + Search) - {selectivity_pct}% selectivity")
    print("=" * 70)
    
    dim = 128
    k = 10
    n_categories = max(1, int(100 / selectivity_pct))
    
    if use_sift:
        sift_dir = "ACORN/benchs/sift1M"
        if os.path.exists(sift_dir):
            xb, xq = load_sift1M(sift_dir)
            total_needed = n_init + n_ops
            xb = xb[:min(total_needed, len(xb))]
        else:
            xb, _, _ = generate_data(n=n_init + n_ops, d=dim)
            xq = np.random.rand(n_ops, dim).astype('float32')
    else:
        xb, _, _ = generate_data(n=n_init + n_ops, d=dim)
        xq = np.random.rand(n_ops, dim).astype('float32')
    
    np.random.seed(42)
    attrs = np.random.randint(0, n_categories, size=len(xb)).astype('int32')
    ids = np.arange(len(xb)).astype('int64')
    
    init_vecs = xb[:n_init]
    init_ids = ids[:n_init]
    init_attrs = attrs[:n_init]
    
    stream_vecs = xb[n_init:n_init + n_ops]
    stream_ids = ids[n_init:n_init + n_ops]
    stream_attrs = attrs[n_init:n_init + n_ops]
    
    np.random.seed(123)
    query_vecs = xq[:n_ops] if len(xq) >= n_ops else np.random.rand(n_ops, dim).astype('float32')
    query_attrs = np.random.randint(0, n_categories, size=n_ops).astype('int32')
    
    indices = {
        "ACORN-1": ACORNIndexWrapper(dim, M=32, gamma=1, M_beta=64, efSearch=128),
        "ACORN-4": ACORNIndexWrapper(dim, M=32, gamma=4, M_beta=64, efSearch=128),
        "ACORN-8": ACORNIndexWrapper(dim, M=32, gamma=8, M_beta=64, efSearch=128),
        "ACORN-12": ACORNIndexWrapper(dim, M=32, gamma=12, M_beta=64, efSearch=128),
        "Pre-Filter": PreFilterIndex(dim),
        # "Post-Filter-Strict": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=0),
        "Post-Filter-Batch100": PostFilterIndex(dim, expansion_factor=100, rebuild_interval=100),
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"\nBenchmarking {name}...")
        
        # Reset RNG for consistent deletes
        np.random.seed(456)
        
        # Oracle
        oracle = PreFilterIndex(dim)
        oracle.build(init_vecs.copy(), init_ids.copy(), init_attrs.copy())
        
        # Build
        t0_build = time.time()
        idx.build(init_vecs.copy(), init_ids.copy(), init_attrs.copy())
        build_time = time.time() - t0_build
        print(f"  Build: {build_time:.2f}s ({n_init} vectors)")
        
        # Track active IDs for deletion
        active_ids = list(init_ids)
        
        insert_times = []
        delete_times = []
        search_times = []
        recalls = []
        
        for i in tqdm(range(n_ops), desc=f"  {name}", leave=False):
            # Insert new vector
            t_ins = time.time()
            idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            insert_times.append(time.time() - t_ins)
            
            oracle.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            active_ids.append(int(stream_ids[i]))
            
            # Delete random existing vector
            if len(active_ids) > 1:
                del_idx = np.random.randint(0, len(active_ids))
                # Swap with last for O(1) pop
                active_ids[del_idx], active_ids[-1] = active_ids[-1], active_ids[del_idx]
                doc_id_to_del = active_ids.pop()
                
                t_del = time.time()
                idx.delete(doc_id_to_del)
                delete_times.append(time.time() - t_del)
                
                oracle.delete(doc_id_to_del)
            
            # Search
            q_vec = query_vecs[i:i+1]
            t_attr = query_attrs[i]
            
            _, gt_I = oracle.search(q_vec, k, t_attr)
            
            t_search = time.time()
            _, pred_I = idx.search(q_vec, k, t_attr)
            search_times.append(time.time() - t_search)
            
            recalls.append(calculate_recall(pred_I, gt_I))
        
        avg_insert = np.mean(insert_times) * 1000
        avg_delete = np.mean(delete_times) * 1000 if delete_times else 0
        avg_search = np.mean(search_times) * 1000
        qps = n_ops / np.sum(search_times)
        avg_recall = np.mean(recalls)
        
        print(f"  Avg Insert: {avg_insert:.2f}ms")
        print(f"  Avg Delete: {avg_delete:.2f}ms")
        print(f"  Avg Search: {avg_search:.2f}ms, QPS: {qps:.1f}")
        print(f"  Recall: {avg_recall:.4f}")
        
        results.append({
            'benchmark': 'dynamic_update',
            'selectivity_pct': selectivity_pct,
            'n_ops': n_ops,
            'method': name,
            'build_time': build_time,
            'qps': qps,
            'recall': avg_recall,
            'avg_insert_ms': avg_insert,
            'avg_delete_ms': avg_delete,
        })
    
    return results


def run_selectivity_dynamic_benchmark(use_sift=True, n_init=50000, n_ops=1000):
    """
    Run dynamic benchmark across multiple selectivity levels.
    """
    print("\n" + "=" * 70)
    print("DYNAMIC SELECTIVITY BENCHMARK (Insert + Delete + Search)")
    print("=" * 70)
    
    selectivity_levels = [1.0, 5.0, 10.0, 20.0]
    all_results = []
    
    for sel in selectivity_levels:
        results = run_update_benchmark(
            use_sift=use_sift,
            n_init=n_init,
            n_ops=n_ops,
            selectivity_pct=sel
        )
        all_results.extend(results)
    
    return all_results


def print_summary(results, output_file=None):
    """Print summary table of results."""
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Group by benchmark type
    for benchmark in df['benchmark'].unique():
        print(f"\n=== {benchmark.upper()} ===")
        bdf = df[df['benchmark'] == benchmark]
        
        # Pivot for QPS
        if 'selectivity_pct' in bdf.columns:
            print("\nQPS by Method and Selectivity:")
            pivot = bdf.pivot(index='selectivity_pct', columns='method', values='qps')
            print(pivot.to_string())
            
            print("\nRecall by Method and Selectivity:")
            pivot = bdf.pivot(index='selectivity_pct', columns='method', values='recall')
            print(pivot.round(4).to_string())
            
            if bdf['avg_insert_ms'].sum() > 0:
                print("\nAvg Insert Latency (ms):")
                pivot = bdf.pivot(index='selectivity_pct', columns='method', values='avg_insert_ms')
                print(pivot.round(2).to_string())
            
            if bdf['avg_delete_ms'].sum() > 0:
                print("\nAvg Delete Latency (ms):")
                pivot = bdf.pivot(index='selectivity_pct', columns='method', values='avg_delete_ms')
                print(pivot.round(2).to_string())
    
    # Save to CSV
    if output_file:
        # Append mode - check if file exists to determine if header is needed
        import os
        write_header = not os.path.exists(output_file)
        df.to_csv(output_file, index=False, mode='a', header=write_header)
        print(f"\nResults appended to {output_file}")
    else:
        df.to_csv("dynamic_benchmark_results.csv", index=False)
        print(f"\nResults saved to dynamic_benchmark_results.csv")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic ACORN Benchmark")
    parser.add_argument('--static', action='store_true', help='Run static benchmark')
    parser.add_argument('--dynamic', action='store_true', help='Run dynamic insert benchmark')
    parser.add_argument('--update', action='store_true', help='Run update (insert+delete) benchmark')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--n-init', type=int, default=50000, help='Initial index size')
    parser.add_argument('--n-ops', type=int, default=1000, help='Number of dynamic operations')
    parser.add_argument('--selectivity', type=float, default=5.0, help='Selectivity percentage')
    parser.add_argument('--no-sift', action='store_true', help='Use random data instead of SIFT')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (append mode)')
    args = parser.parse_args()
    
    use_sift = not args.no_sift
    all_results = []
    
    if args.all or (not args.static and not args.dynamic and not args.update):
        # Default: run all
        # all_results.extend(run_static_selectivity_benchmark(use_sift=use_sift, n_data=args.n_init))
        all_results.extend(run_selectivity_dynamic_benchmark(use_sift=use_sift, n_init=args.n_init, n_ops=args.n_ops))
    else:
        if args.static:
            all_results.extend(run_static_selectivity_benchmark(use_sift=use_sift, n_data=args.n_init))
        if args.dynamic:
            all_results.extend(run_dynamic_benchmark(use_sift=use_sift, n_init=args.n_init, n_ops=args.n_ops, selectivity_pct=args.selectivity))
        if args.update:
            all_results.extend(run_update_benchmark(use_sift=use_sift, n_init=args.n_init, n_ops=args.n_ops, selectivity_pct=args.selectivity))
    
    print_summary(all_results, output_file=args.output)


if __name__ == "__main__":
    main()