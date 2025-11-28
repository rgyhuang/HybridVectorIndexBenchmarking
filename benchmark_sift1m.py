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
    # Scenario: 50 categories (Selectivity ~ 2%)
    n_attrs = 50
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
    indices = {
        "Pre-Filter": PreFilterIndex(dim),
        "Post-Filter": PostFilterIndex(dim),
        "ACORN-1 (Build Opt)": AcornIndexWrapper(dim, M=32, gamma=1, M_beta=32),
        "ACORN-gamma (Search Opt)": AcornIndexWrapper(dim, M=32, gamma=12, M_beta=32),
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
