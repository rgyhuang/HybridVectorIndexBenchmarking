import numpy as np
import time
import faiss
import acorn_ext
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
from ACORNIndex import ACORNIndex

# --- Interfaces ---

class HybridIndex(ABC):
    @abstractmethod
    def build(self, vectors: np.ndarray, ids: np.ndarray, attrs: np.ndarray):
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int, target_attr: int) -> tuple:
        pass
    
    @abstractmethod
    def insert(self, vector: np.ndarray, doc_id: int, attr: int):
        pass

    @abstractmethod
    def delete(self, doc_id: int):
        pass

# --- Implementations ---

class AcornIndexWrapper(HybridIndex):
    def __init__(self, dim, M=32, gamma=12, M_beta=32):
        self.dim = dim
        self.index = ACORNIndex(dimension=dim, M=M, gamma=gamma, M_beta=M_beta)

    def build(self, vectors, ids, attrs):
        # Loop insertion
        for i in tqdm(range(len(vectors)), desc="Building ACORN Index"):
            self.index.insert(vectors[i], {'category': int(attrs[i])}, int(ids[i]))

    def search(self, query, k, target_attr):
        nq = query.shape[0]
        all_labels = []
        all_distances = []
        
        # Loop search
        for i in range(nq):
            # Define predicate
            predicate = lambda meta: meta.get('category') == target_attr
            
            latency, ids, dists = self.index.search(query[i], predicate, k)
            
            # Pad if needed
            while len(ids) < k:
                ids.append(-1)
                dists.append(float('inf'))
                
            all_labels.append(ids)
            all_distances.append(dists)
            
        return np.array(all_distances), np.array(all_labels)

    def insert(self, vector, doc_id, attr):
        self.index.insert(vector, {'category': int(attr)}, int(doc_id))

    def delete(self, doc_id):
        self.index.delete(int(doc_id))

class PostFilterIndex(HybridIndex):
    def __init__(self, dim):
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.metadata = {}
        self.dim = dim

    def build(self, vectors, ids, attrs):
        self.index.add(vectors)
        for i, doc_id in enumerate(ids):
            self.metadata[doc_id] = attrs[i]

    def search(self, query, k, target_attr):
        nq = query.shape[0]
        fetch_k = k * 50 
        D, I = self.index.search(query, fetch_k)
        
        final_D = []
        final_I = []
        
        for i in range(nq):
            res_d, res_i = [], []
            for j, doc_id in enumerate(I[i]):
                if doc_id != -1 and self.metadata.get(doc_id) == target_attr:
                    res_d.append(D[i][j])
                    res_i.append(doc_id)
                    if len(res_i) >= k:
                        break
            while len(res_i) < k:
                res_i.append(-1)
                res_d.append(float('inf'))
            final_D.append(res_d)
            final_I.append(res_i)
            
        return np.array(final_D), np.array(final_I)

    def insert(self, vector, doc_id, attr):
        self.index.add(vector.reshape(1, -1))
        self.metadata[doc_id] = attr

    def delete(self, doc_id):
        if doc_id in self.metadata:
            del self.metadata[doc_id]

class PreFilterIndex(HybridIndex):
    def __init__(self, dim):
        self.dim = dim
        self.partitions = {} 
        self.doc_attrs = {}

    def build(self, vectors, ids, attrs):
        unique_attrs = np.unique(attrs)
        for i, doc_id in enumerate(ids):
            self.doc_attrs[doc_id] = attrs[i]

        for val in unique_attrs:
            mask = (attrs == val)
            sub_vecs = vectors[mask]
            sub_ids = ids[mask]
            
            idx = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
            idx.add_with_ids(sub_vecs, sub_ids)
            self.partitions[val] = idx

    def search(self, query, k, target_attr):
        if target_attr not in self.partitions:
            return np.full((query.shape[0], k), float('inf')), np.full((query.shape[0], k), -1)
        return self.partitions[target_attr].search(query, k)

    def insert(self, vector, doc_id, attr):
        if attr not in self.partitions:
            self.partitions[attr] = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        self.partitions[attr].add_with_ids(vector.reshape(1, -1), np.array([doc_id]))
        self.doc_attrs[doc_id] = attr

    def delete(self, doc_id):
        if doc_id in self.doc_attrs:
            attr = self.doc_attrs[doc_id]
            if attr in self.partitions:
                self.partitions[attr].remove_ids(np.array([doc_id], dtype='int64'))
            del self.doc_attrs[doc_id]

# --- Benchmarking Logic ---

def generate_data(n=10000, d=128, n_attrs=100):
    vectors = np.random.rand(n, d).astype('float32')
    ids = np.arange(n).astype('int64')
    attrs = np.random.randint(0, n_attrs, size=n).astype('int32') 
    return vectors, ids, attrs

def run_static_benchmark():
    print("=== Static Workload Benchmark ===")
    n_data = 10000
    dim = 128
    n_queries = 100
    k = 10
    n_attrs = 50 # Selectivity ~ 2%
    
    vecs, ids, attrs = generate_data(n=n_data, d=dim, n_attrs=n_attrs)
    queries = np.random.rand(n_queries, dim).astype('float32')
    target_attrs = np.random.choice(attrs, n_queries)

    print("Computing Ground Truth...")
    gt_I = compute_ground_truth(vecs, ids, attrs, queries, target_attrs, k)

    indices = {
        "ACORN": AcornIndexWrapper(dim),
        "Post-Filter": PostFilterIndex(dim),
        "Pre-Filter": PreFilterIndex(dim)
    }

    results = []

    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        # Build
        t0 = time.time()
        idx.build(vecs, ids, attrs)
        build_time = time.time() - t0
        
        # Search
        t0 = time.time()
        hits = 0
        
        # Batch queries by attribute to optimize ACORN wrapper usage
        unique_targets = np.unique(target_attrs)
        
        # Store results for recall calc
        final_I = np.full((n_queries, k), -1, dtype='int64')

        for t_attr in unique_targets:
            # Find queries with this target attribute
            mask = (target_attrs == t_attr)
            batch_queries = queries[mask]
            query_indices = np.where(mask)[0]
            
            if len(batch_queries) == 0: continue
            
            D, I = idx.search(batch_queries, k, t_attr)
            
            # Store results
            final_I[query_indices] = I

            # Simple hit check (sanity check)
            if np.all(I != -1):
                hits += len(batch_queries)
                
        total_time = time.time() - t0
        qps = n_queries / total_time
        recall = calculate_recall(final_I, gt_I)
        
        print(f"  Build: {build_time:.4f}s, QPS: {qps:.2f}, Recall@{k}: {recall:.4f}")
        results.append({
            "workload": "static",
            "method": name,
            "build_time": build_time,
            "qps": qps,
            "recall": recall
        })
    return results

def run_dynamic_benchmark():
    print("=== Dynamic Workload Benchmark ===")
    # Start with 50% data, then interleave insert/search
    n_total = 10000
    n_init = 5000
    dim = 128
    n_attrs = 50
    k = 10
    
    vecs, ids, attrs = generate_data(n=n_total, d=dim, n_attrs=n_attrs)
    
    init_vecs = vecs[:n_init]
    init_ids = ids[:n_init]
    init_attrs = attrs[:n_init]
    
    stream_vecs = vecs[n_init:]
    stream_ids = ids[n_init:]
    stream_attrs = attrs[n_init:]
    
    n_ops = len(stream_vecs)
    
    # Pre-generate queries to ensure fairness
    query_vecs = np.random.rand(n_ops, dim).astype('float32')
    query_attrs = np.random.choice(init_attrs, n_ops)
    
    indices = {
        "ACORN": AcornIndexWrapper(dim),
        "Post-Filter": PostFilterIndex(dim),
        "Pre-Filter": PreFilterIndex(dim)
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        # Oracle for Ground Truth
        oracle = PreFilterIndex(dim)
        oracle.build(init_vecs, init_ids, init_attrs)
        
        # Initial Build
        t0_build = time.time()
        idx.build(init_vecs, init_ids, init_attrs)
        build_time = time.time() - t0_build
        
        # Stream
        insert_times = []
        search_times = []
        recalls = []
        
        for i in tqdm(range(n_ops)):
            # Update Oracle
            oracle.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            
            # Insert
            t_ins_start = time.time()
            idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            insert_times.append(time.time() - t_ins_start)
            
            # Search
            q_vec = query_vecs[i:i+1]
            t_attr = query_attrs[i]
            
            # Ground Truth
            gt_D, gt_I = oracle.search(q_vec, k, t_attr)
            
            t_search_start = time.time()
            D, I = idx.search(q_vec, k, t_attr)
            search_times.append(time.time() - t_search_start)
            
            recalls.append(calculate_recall(I, gt_I))
            
        avg_insert = np.mean(insert_times)
        avg_search = np.mean(search_times)
        avg_recall = np.mean(recalls)
        qps = n_ops / np.sum(search_times)
        
        print(f"  Build: {build_time:.4f}s, Avg Insert: {avg_insert*1000:.2f}ms, Avg Search: {avg_search*1000:.2f}ms, QPS: {qps:.2f}, Recall: {avg_recall:.4f}")
        results.append({
            "workload": "dynamic",
            "method": name,
            "build_time": build_time,
            "avg_insert_ms": avg_insert * 1000,
            "avg_search_ms": avg_search * 1000,
            "qps": qps,
            "recall": avg_recall
        })
        
    return results

def run_update_benchmark():
    print("=== Update Workload Benchmark (Inserts + Deletes) ===")
    n_total = 10000
    n_init = 5000
    dim = 128
    n_attrs = 50
    k = 10
    
    vecs, ids, attrs = generate_data(n=n_total, d=dim, n_attrs=n_attrs)
    
    # Initial set
    init_vecs = vecs[:n_init]
    init_ids = ids[:n_init]
    init_attrs = attrs[:n_init]
    
    # Stream set (new IDs)
    stream_vecs = vecs[n_init:]
    stream_ids = ids[n_init:]
    stream_attrs = attrs[n_init:]
    
    n_ops = len(stream_vecs)
    
    # Pre-generate queries
    query_vecs = np.random.rand(n_ops, dim).astype('float32')
    query_attrs = np.random.choice(init_attrs, n_ops)
    
    indices = {
        "ACORN": AcornIndexWrapper(dim),
        "Post-Filter": PostFilterIndex(dim),
        "Pre-Filter": PreFilterIndex(dim)
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        # Reset RNG for consistent deletes
        np.random.seed(42)
        
        # Oracle
        oracle = PreFilterIndex(dim)
        oracle.build(init_vecs, init_ids, init_attrs)
        
        # Initial Build
        t0_build = time.time()
        idx.build(init_vecs, init_ids, init_attrs)
        build_time = time.time() - t0_build
        
        # Track active IDs for deletion
        active_ids = list(init_ids)
        
        # Stream
        insert_times = []
        delete_times = []
        search_times = []
        recalls = []
        
        for i in tqdm(range(n_ops)):
            # Insert
            t_ins_start = time.time()
            idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            insert_times.append(time.time() - t_ins_start)
            
            oracle.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            active_ids.append(stream_ids[i])
            
            # Delete (random old ID)
            if len(active_ids) > 0:
                del_idx = np.random.randint(0, len(active_ids))
                # Swap with last to pop O(1)
                active_ids[del_idx], active_ids[-1] = active_ids[-1], active_ids[del_idx]
                doc_id_to_del = active_ids.pop()
                
                t_del_start = time.time()
                idx.delete(doc_id_to_del)
                delete_times.append(time.time() - t_del_start)
                
                oracle.delete(doc_id_to_del)
            
            # Search
            q_vec = query_vecs[i:i+1]
            t_attr = query_attrs[i]
            
            # Ground Truth
            gt_D, gt_I = oracle.search(q_vec, k, t_attr)
            
            t_search_start = time.time()
            D, I = idx.search(q_vec, k, t_attr)
            search_times.append(time.time() - t_search_start)
            
            recalls.append(calculate_recall(I, gt_I))
            
        avg_insert = np.mean(insert_times) if insert_times else 0
        avg_delete = np.mean(delete_times) if delete_times else 0
        avg_search = np.mean(search_times) if search_times else 0
        avg_recall = np.mean(recalls)
        qps = n_ops / np.sum(search_times) if search_times else 0
        
        print(f"  Build: {build_time:.4f}s, Avg Insert: {avg_insert*1000:.2f}ms, Avg Delete: {avg_delete*1000:.2f}ms, Avg Search: {avg_search*1000:.2f}ms, QPS: {qps:.2f}, Recall: {avg_recall:.4f}")
        results.append({
            "workload": "update",
            "method": name,
            "build_time": build_time,
            "avg_insert_ms": avg_insert * 1000,
            "avg_delete_ms": avg_delete * 1000,
            "avg_search_ms": avg_search * 1000,
            "qps": qps,
            "recall": avg_recall
        })
        
    return results

def run_heavy_update_benchmark():
    print("=== Heavy Update Workload Benchmark (Stress Test) ===")
    # Start with small data, then churn significantly
    n_total = 100000 # Total unique vectors available
    n_init = 2000
    dim = 128
    n_attrs = 50
    k = 10
    
    vecs, ids, attrs = generate_data(n=n_total, d=dim, n_attrs=n_attrs)
    
    # Initial set
    init_vecs = vecs[:n_init]
    init_ids = ids[:n_init]
    init_attrs = attrs[:n_init]
    
    # Stream set
    stream_vecs = vecs[n_init:]
    stream_ids = ids[n_init:]
    stream_attrs = attrs[n_init:]
    
    n_ops = 20000 # 20k ops to see degradation
    
    # Pre-generate queries
    query_vecs = np.random.rand(n_ops, dim).astype('float32')
    query_attrs = np.random.choice(init_attrs, n_ops)
    
    # Only test ACORN for stress test to save time, or compare with Post-Filter
    indices = {
        "ACORN": AcornIndexWrapper(dim),
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        # Initial Build
        idx.build(init_vecs, init_ids, init_attrs)
        
        active_ids = list(init_ids)
        
        # Measure in chunks to see degradation
        chunk_size = 1000
        n_chunks = n_ops // chunk_size
        
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = start_idx + chunk_size
            
            t0_chunk = time.time()
            search_times = []
            
            for i in range(start_idx, end_idx):
                # Insert
                idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
                active_ids.append(stream_ids[i])
                
                # Delete (maintain roughly constant size)
                if len(active_ids) > n_init:
                    del_idx = np.random.randint(0, len(active_ids))
                    active_ids[del_idx], active_ids[-1] = active_ids[-1], active_ids[del_idx]
                    doc_id_to_del = active_ids.pop()
                    idx.delete(doc_id_to_del)
                
                # Search
                q_vec = query_vecs[i:i+1]
                t_attr = query_attrs[i]
                
                t_search = time.time()
                idx.search(q_vec, k, t_attr)
                search_times.append(time.time() - t_search)
            
            chunk_time = time.time() - t0_chunk
            avg_search = np.mean(search_times) * 1000
            qps = chunk_size / np.sum(search_times)
            
            print(f"  Chunk {chunk+1}/{n_chunks}: Avg Search {avg_search:.2f}ms, QPS {qps:.2f}, Total Items (C++ view): {len(idx.index.cpp_doc_ids)}")
            
            results.append({
                "workload": "heavy_update",
                "method": name,
                "chunk": chunk,
                "avg_search_ms": avg_search,
                "qps": qps,
                "total_inserted": len(idx.index.cpp_doc_ids)
            })
            
    return results
def compute_ground_truth(vectors, ids, attrs, queries, target_attrs, k):
    nq = len(queries)
    gt_I = np.full((nq, k), -1, dtype='int64')
    
    # Optimize: Group by target attribute to vectorize distance calc per group
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
        
        # Brute force for this batch using Faiss for speed
        index_flat = faiss.IndexFlatL2(vectors.shape[1])
        index_flat.add(sub_vecs)
        D, I = index_flat.search(sub_queries, k)
        
        # Map local indices I to global IDs
        for i, q_idx in enumerate(query_indices):
            # Filter out -1s from Faiss result (if k > n_available)
            valid_mask = I[i] != -1
            valid_indices = I[i][valid_mask]
            
            # Map back to global IDs
            found_ids = sub_ids[valid_indices]
            gt_I[q_idx, :len(found_ids)] = found_ids
            
    return gt_I

def calculate_recall(I, gt_I):
    # I: (nq, k)
    # gt_I: (nq, k)
    nq = len(I)
    total_recall = 0
    
    for i in range(nq):
        # Set of retrieved IDs (exclude -1)
        retrieved = set(I[i])
        retrieved.discard(-1)
        
        # Set of relevant IDs (exclude -1)
        relevant = set(gt_I[i])
        relevant.discard(-1)
        
        if len(relevant) == 0:
            total_recall += 1.0
        else:
            total_recall += len(retrieved.intersection(relevant)) / len(relevant)
            
    return total_recall / nq

if __name__ == "__main__":
    res_static = run_static_benchmark()
    res_dynamic = run_dynamic_benchmark()
    res_update = run_update_benchmark()
    res_heavy_update = run_heavy_update_benchmark()
    
    df = pd.DataFrame(res_static + res_dynamic + res_update + res_heavy_update)
    print("Results:")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)