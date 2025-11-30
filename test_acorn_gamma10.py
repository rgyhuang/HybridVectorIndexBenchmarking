#!/usr/bin/env python3
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/HybridVectorIndexBenchmarking')

from ACORNIndex import ACORNIndex

print("Testing ACORN with gamma=10...")

# Small test
dim = 128
n = 1000
M = 16
gamma = 10
M_beta = 48

print(f"Creating index with M={M}, gamma={gamma}, M_beta={M_beta}")
try:
    index = ACORNIndex(dimension=dim, M=M, gamma=gamma, M_beta=M_beta, efSearch=400)
    print("✓ Index created")
    
    # Add some vectors
    vectors = np.random.rand(n, dim).astype('float32')
    metas = [{'category': i % 10} for i in range(n)]
    ids = list(range(n))
    
    print("Adding vectors...")
    index.insert_batch(vectors, metas, ids)
    print(f"✓ Added {n} vectors")
    
    # Try a search
    query = np.random.rand(dim).astype('float32')
    print("Searching...")
    latency, result_ids, dists = index.search_category(query, 0, 10)
    print(f"✓ Search completed: found {len(result_ids)} results")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
