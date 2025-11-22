import numpy as np
import time
import faiss
import acorn_ext
from abc import ABC, abstractmethod
from tqdm import tqdm

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

# --- Implementations ---

class AcornIndexWrapper(HybridIndex):
    def __init__(self, dim, M=32, gamma=12, M_beta=32):
        self.dim = dim
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.index = None
        self.attrs = None # Keep track of attributes for filter generation
        self.ids = None

    def build(self, vectors, ids, attrs):
        # ACORN requires metadata at construction for the graph structure
        # We assume 'attrs' are the metadata categories (integers)
        self.attrs = attrs.astype('int32')
        self.ids = ids
        
        # Initialize IndexACORNFlat
        # Note: metadata is passed by reference in C++, pybind11 handles conversion
        self.index = acorn_ext.IndexACORNFlat(
            self.dim, 
            self.M, 
            self.gamma, 
            self.attrs, 
            self.M_beta, 
            acorn_ext.MetricType.METRIC_L2
        )
        
        # The initial metadata passed to constructor is just for initialization/sizing if needed,
        # but our binding's add() now takes metadata to append to storage.
        # Actually, the constructor I exposed takes metadata.
        # And add() takes metadata.
        # If I pass metadata to constructor, does it add it?
        # The constructor initializes metadata_storage with it.
        # But it doesn't add vectors.
        # So when I call add(vectors), I should NOT pass metadata again if it's already in storage?
        # Wait, my add() implementation appends to metadata_storage.
        # If constructor already populated metadata_storage, then add() will append MORE.
        # That would be a mismatch between vectors and metadata size.
        
        # Let's check IndexACORNFlat constructor in pybind.
        # .def(py::init<int, int, int, std::vector<int>&, int, faiss::MetricType>())
        # It takes metadata.
        # So if I pass full metadata to constructor, metadata_storage has N items.
        # Then if I call add(vectors, metadata), metadata_storage will have 2N items.
        # This is WRONG.
        
        # FIX: Initialize with EMPTY metadata, then add both vectors and metadata via add().
        # Or, if constructor requires metadata (for some reason?), we must be careful.
        # ACORN constructor uses metadata to build the graph?
        # No, ACORN graph is built when vectors are added.
        # But ACORN struct takes metadata reference.
        
        # I should initialize with empty metadata.
        empty_meta = np.array([], dtype='int32')
        self.index = acorn_ext.IndexACORNFlat(
            self.dim, 
            self.M, 
            self.gamma, 
            empty_meta, 
            self.M_beta, 
            acorn_ext.MetricType.METRIC_L2
        )
        
        self.index.add(vectors, self.attrs)

    def search(self, query, k, target_attr):
        # Generate filter map: 1 if attr matches, 0 otherwise
        # Shape: (nq, ntotal)
        nq = query.shape[0]
        ntotal = self.index.ntotal
        
        # Optimization: If we track attrs in python, we can generate the mask
        # For a single query batch with SAME target_attr, we can broadcast
        # But the interface allows different target_attr per query?
        # The benchmark loop passes one target_attr per query if we loop.
        # If we batch queries with same target_attr, it's faster.
        
        # Here we assume query is (1, d) or (nq, d) but target_attr is single int
        # If target_attr is list, we need to handle it.
        
        if isinstance(target_attr, (int, np.integer)):
             # Broadcast target_attr to all queries
             # mask = (self.attrs == target_attr).astype('int8')
             # filter_map = np.tile(mask, (nq, 1))
             # This is memory intensive for large nq * ntotal.
             # For benchmarking, we process in small batches or 1 by 1 if needed.
             pass
        
        # For this benchmark, let's assume query is (nq, d) and we have one target_attr for the batch
        # or we loop outside.
        
        # Let's implement for batch query with single target_attr for simplicity of the wrapper
        # The benchmark runner will handle batching.
        
        mask = (self.attrs == target_attr).astype('int8')
        filter_map = np.broadcast_to(mask, (nq, ntotal)).copy() # Ensure contiguous if needed, or just broadcast
        
        # acorn_ext expects (nq, ntotal) char array
        distances, labels = self.index.search(query, k, filter_map)
        return distances, labels

    def insert(self, vector, doc_id, attr):
        # vector: (d,)
        # attr: int
        vec_batch = vector.reshape(1, -1).astype('float32')
        attr_batch = np.array([attr], dtype='int32')
        self.index.add(vec_batch, attr_batch)
        
        # Update python side attrs for filter generation
        self.attrs = np.append(self.attrs, attr_batch)

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

class PreFilterIndex(HybridIndex):
    def __init__(self, dim):
        self.dim = dim
        self.partitions = {} 

    def build(self, vectors, ids, attrs):
        unique_attrs = np.unique(attrs)
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
        # (Since our wrapper takes one target_attr per call)
        unique_targets = np.unique(target_attrs)
        
        for t_attr in unique_targets:
            # Find queries with this target attribute
            mask = (target_attrs == t_attr)
            batch_queries = queries[mask]
            
            if len(batch_queries) == 0: continue
            
            D, I = idx.search(batch_queries, k, t_attr)
            
            # Simple hit check (sanity check)
            if np.all(I != -1):
                hits += len(batch_queries)
                
        total_time = time.time() - t0
        qps = n_queries / total_time
        
        print(f"  Build: {build_time:.4f}s, QPS: {qps:.2f}")
        results.append({
            "workload": "static",
            "method": name,
            "build_time": build_time,
            "qps": qps
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
    
    indices = {
        "ACORN": AcornIndexWrapper(dim),
        "Post-Filter": PostFilterIndex(dim),
        "Pre-Filter": PreFilterIndex(dim)
    }
    
    results = []
    
    for name, idx in indices.items():
        print(f"Benchmarking {name}...")
        
        # Initial Build
        idx.build(init_vecs, init_ids, init_attrs)
        
        # Stream
        t0 = time.time()
        n_ops = len(stream_vecs)
        
        # Simulate 1 search per 1 insert
        search_times = []
        insert_times = []
        
        for i in tqdm(range(n_ops)):
            # Insert
            t_ins_start = time.time()
            idx.insert(stream_vecs[i], stream_ids[i], stream_attrs[i])
            insert_times.append(time.time() - t_ins_start)
            
            # Search
            q_vec = np.random.rand(1, dim).astype('float32')
            t_attr = np.random.choice(init_attrs) # Pick a random attr
            
            t_search_start = time.time()
            idx.search(q_vec, k, t_attr)
            search_times.append(time.time() - t_search_start)
            
        total_time = time.time() - t0
        avg_insert = np.mean(insert_times)
        avg_search = np.mean(search_times)
        
        print(f"  Avg Insert: {avg_insert*1000:.2f}ms, Avg Search: {avg_search*1000:.2f}ms")
        results.append({
            "workload": "dynamic",
            "method": name,
            "avg_insert_ms": avg_insert * 1000,
            "avg_search_ms": avg_search * 1000
        })
        
    return results

if __name__ == "__main__":
    res_static = run_static_benchmark()
    res_dynamic = run_dynamic_benchmark()
    
    import pandas as pd
    df = pd.DataFrame(res_static + res_dynamic)
    print("Results:")
    print(df)
    df.to_csv("benchmark_results.csv", index=False)