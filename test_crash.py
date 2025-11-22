import acorn_ext
import numpy as np

def test_crash():
    dim = 128
    M = 32
    gamma = 12
    M_beta = 32
    
    empty_meta = np.array([], dtype='int32')
    index = acorn_ext.IndexACORNFlat(dim, M, gamma, empty_meta, M_beta, acorn_ext.MetricType.METRIC_L2)
    
    print("Index created")
    
    for i in range(10):
        vec = np.random.rand(1, dim).astype('float32')
        meta = np.array([0], dtype='int32')
        print(f"Adding {i}")
        index.add(vec, meta)
        print(f"Added {i}")

if __name__ == "__main__":
    test_crash()
