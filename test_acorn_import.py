import acorn_ext
import numpy as np

def test_acorn():
    print("Testing ACORN import...")
    d = 128
    M = 32
    gamma = 12
    M_beta = 32
    ntotal = 1000
    
    # Generate random data
    xb = np.random.random((ntotal, d)).astype('float32')
    metadata = np.random.randint(0, 10, size=ntotal).astype('int32')
    
    print("Creating IndexACORNFlat with L2...")
    index = acorn_ext.IndexACORNFlat(d, M, gamma, metadata, M_beta, acorn_ext.MetricType.METRIC_L2)
    
    print(f"Index created. d={index.d}, ntotal={index.ntotal}")
    
    print("Adding vectors...")
    index.add(xb)
    
    print(f"Vectors added. ntotal={index.ntotal}")
    
    # Search
    nq = 5
    xq = np.random.random((nq, d)).astype('float32')
    k = 10
    
    # Filter map: (nq, ntotal) of char (int8)
    # Test 1: All allowed
    print("Test 1: All allowed")
    filter_id_map = np.ones((nq, ntotal), dtype='int8')
    distances, labels = index.search(xq, k, filter_id_map)
    print("First query labels:", labels[0])
    
    # Test 2: All blocked
    print("Test 2: All blocked")
    filter_id_map_blocked = np.zeros((nq, ntotal), dtype='int8')
    distances_blocked, labels_blocked = index.search(xq, k, filter_id_map_blocked)
    print("First query labels (should be -1 or similar):", labels_blocked[0])

if __name__ == "__main__":
    test_acorn()
