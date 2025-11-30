import numpy as np
import time
import sys
import os
import pandas as pd
from tqdm import tqdm
import faiss

# Add ACORN/benchs to path to import datasets if needed, 
# but I'll just implement the loader directly to be safe.

# Import wrappers from benchmark.py
from benchmark import AcornIndexWrapper, PostFilterIndex, PreFilterIndex, compute_ground_truth, calculate_recall

class AcornIndexWrapperWithProgress(AcornIndexWrapper):
    def __init__(self, dim, M=32, gamma=12, M_beta=32, expected_build_time=None, efSearch=100):
        super().__init__(dim, M, gamma, M_beta, efSearch)
        self.expected_build_time = expected_build_time

    def build(self, vectors, ids, attrs):
        print(f"Building ACORN index (gamma={self.index.gamma})...")
        t0 = time.time()
        n = len(vectors)
        batch_size = 5000  # Reduced from 10000 to avoid memory issues
        
        # Use tqdm for progress
        for i in tqdm(range(0, n, batch_size), desc="Inserting"):
            end = min(i + batch_size, n)
            batch_vecs = vectors[i:end]
            batch_ids = ids[i:end]
            batch_attrs = attrs[i:end]
            
            # Create metadata dicts
            batch_metas = [{'category': int(a)} for a in batch_attrs]
            batch_doc_ids = [int(id) for id in batch_ids]
            
            try:
                self.index.insert_batch(batch_vecs, batch_metas, batch_doc_ids)
            except Exception as e:
                print(f"\nError at batch {i}-{end}: {e}")
                raise
            
            elapsed = time.time() - t0
            if self.expected_build_time and elapsed > self.expected_build_time * 2.0:
                 raise TimeoutError(f"Build time exceeded expected limit ({self.expected_build_time}s). Elapsed: {elapsed:.2f}s at {end}/{n} items.")

    def search(self, query, k, target_attr):
        nq = query.shape[0]
        all_labels = []
        all_distances = []
        
        # Loop search
        for i in range(nq):
            # Use optimized search
            latency, ids, dists = self.index.search_category(query[i], int(target_attr), k)
            
            # Pad if needed
            while len(ids) < k:
                ids.append(-1)
                dists.append(float('inf'))
                
            all_labels.append(ids)
            all_distances.append(dists)
            
        return np.array(all_distances), np.array(all_labels)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift1M(base_dir):
    print(f"Loading SIFT1M from {base_dir}...", file=sys.stderr)
    xt = fvecs_read(os.path.join(base_dir, "sift_learn.fvecs"))
    xb = fvecs_read(os.path.join(base_dir, "sift_base.fvecs"))
    xq = fvecs_read(os.path.join(base_dir, "sift_query.fvecs"))
    gt = ivecs_read(os.path.join(base_dir, "sift_groundtruth.ivecs"))
    print("Done loading.", file=sys.stderr)
    return xb, xq, xt, gt

def run_sift1m_benchmark():
    print("=== SIFT1M Hybrid Benchmark ===")
    
    # Path to SIFT1M
    sift_dir = "ACORN/benchs/sift1M"
    if not os.path.exists(sift_dir):
        print(f"Error: SIFT1M directory not found at {sift_dir}")
        return

    # Load Data
    xb, xq, xt, gt_pure = load_sift1M(sift_dir)
    
    n_data = xb.shape[0]
    dim = xb.shape[1]
    n_queries = xq.shape[0]
    k = 10
    
    # Generate Metadata
    # Scenario: Use 10 categories for ~10% selectivity (matches paper's experiments better)
    # Paper shows best ACORN performance at 5-10% selectivity range
    n_attrs = 10
    np.random.seed(42)
    attrs = np.random.randint(0, n_attrs, size=n_data).astype('int32')
    ids = np.arange(n_data).astype('int64')
    
    # Generate Query Metadata
    target_attrs = np.random.randint(0, n_attrs, size=n_queries).astype('int32')
    
    print(f"Data: {n_data} vectors, {dim} dimensions")
    print(f"Attributes: {n_attrs} categories (Selectivity ~{100/n_attrs:.1f}%)")
    
    # Compute Ground Truth for Hybrid Queries
    print("Computing Hybrid Ground Truth...")
    
    n_queries_bench = 1000
    xq_bench = xq[:n_queries_bench]
    target_attrs_bench = target_attrs[:n_queries_bench]
    
    gt_I = compute_ground_truth(xb, ids, attrs, xq_bench, target_attrs_bench, k)
    
    # Reordered indices to run Pre/Post filter first
    # Key insight from ACORN test code: default efSearch is 16!
    # Testing gamma=2 which should be optimal for ~25% selectivity neighborhood
    # For 10% selectivity, we want dense graph connections for filtered neighbors
    indices = {
        "Pre-Filter": PreFilterIndex(dim),
        "Post-Filter": PostFilterIndex(dim),
        "ACORN-1 (Build Opt)": AcornIndexWrapperWithProgress(dim, M=32, gamma=1, M_beta=64, expected_build_time=120, efSearch=256),
        "ACORN-gamma (Search Opt)": AcornIndexWrapperWithProgress(dim, M=32, gamma=2, M_beta=64, expected_build_time=150, efSearch=256),
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        try:
            # Build
            t0 = time.time()
            idx.build(xb, ids, attrs)
            build_time = time.time() - t0
            print(f"  Build Time: {build_time:.2f}s")
            
            # Search
            t0 = time.time()
            
            unique_targets = np.unique(target_attrs_bench)
            final_I = np.full((n_queries_bench, k), -1, dtype='int64')
            
            total_search_time = 0
            
            for t_attr in tqdm(unique_targets, desc=f"Queries ({name})"):
                mask = (target_attrs_bench == t_attr)
                batch_queries = xq_bench[mask]
                query_indices = np.where(mask)[0]
                
                if len(batch_queries) == 0: continue
                
                t_batch_start = time.time()
                D, I = idx.search(batch_queries, k, t_attr)
                t_batch_end = time.time()
                total_search_time += (t_batch_end - t_batch_start)
                
                final_I[query_indices] = I
                
            qps = n_queries_bench / total_search_time
            recall = calculate_recall(final_I, gt_I)
            
            print(f"  QPS: {qps:.2f}, Recall@{k}: {recall:.4f}")
            results.append({
                "method": name,
                "build_time": build_time,
                "qps": qps,
                "recall": recall
            })
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            import traceback
            traceback.print_exc()
        
    df = pd.DataFrame(results)
    print("\n=== Results ===")
    print(df)
    df.to_csv("sift1m_benchmark_results.csv", index=False)

if __name__ == "__main__":
    run_sift1m_benchmark()
