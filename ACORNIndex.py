import numpy as np
import time
import os
import sys
from typing import Dict, List, Callable, Set, Tuple, Any

# Add library path for libfaiss.so
_lib_path = os.path.join(os.path.dirname(__file__), 'ACORN', 'build', 'faiss')
if os.path.exists(_lib_path):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = _lib_path

print("ℹ ACORNIndex: Added lib path:", _lib_path)
# Try to import C++ accelerated module
HAS_CPP = False
try:
    import acorn_ext as acorn_cpp
    HAS_CPP = True
    print("✓ C++ ACORN extension loaded successfully")
except ImportError as e:
    HAS_CPP = False
    _import_error = str(e)
    print("⚠ C++ ACORN extension not available: ", _import_error)


class ACORNIndex:
    """
    Python wrapper for the ACORN (Performant and Predicate-Agnostic Search Over 
    Vector Embeddings and Structured Data) index from SIGMOD '24.
    
    Automatically uses C++ implementation if available, falls back to Python.
    """
    
    def __init__(self, dimension: int = 128, M: int = 16, gamma: int = 12, M_beta: int = 48, 
                 use_cpp: bool = True, efSearch: int = 64):
        """
        Initialize ACORN index.
        
        Args:
            dimension: Vector dimension
            M: HNSW connectivity parameter (typical: 16-32, paper uses 16)
            gamma: Attribute replication factor (paper: gamma = 1/selectivity)
            M_beta: Compression parameter for diversity (paper: 2-3x M)
            use_cpp: Use C++ implementation if available (10-100x faster)
            efSearch: Search expansion factor (64-400 for high recall)
        """
        self.d = dimension
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.use_cpp = use_cpp and HAS_CPP
        self.efSearch = efSearch
        
        # Metadata tracking (Python side)
        self.current_ids: Set[int] = set()
        self.metadata: Dict[int, Dict] = {}  # doc_id -> metadata dict
        self.vectors: Dict[int, np.ndarray] = {}  # For pure Python fallback
        
        # C++ index state
        self.cpp_index = None
        self.cpp_doc_ids: List[int] = []  # Track insertion order
        self.doc_id_to_internal: Dict[int, int] = {} # doc_id -> internal_id
        self.cpp_metadata_ints: List[int] = []  # Integer metadata for C++
        self._meta_arr_cache = None # Cache for numpy array of metadata
        
        # Initialize C++ index if available
        if self.use_cpp:
            try:
                # Initialize with empty metadata
                empty_meta = np.array([], dtype='int32')
                self.cpp_index = acorn_cpp.IndexACORNFlat(
                    dimension, 
                    M, 
                    gamma, 
                    empty_meta, 
                    M_beta, 
                    acorn_cpp.MetricType.METRIC_L2
                )
                # Set efSearch if available
                if hasattr(self.cpp_index, 'set_efSearch'):
                    self.cpp_index.set_efSearch(self.efSearch)
                    print(f"✓ Set efSearch={self.efSearch}")
                
                print(f"✓ Using C++ ACORN (d={dimension}, M={M}, gamma={gamma}, M_beta={M_beta})")
            except Exception as e:
                print(f"⚠ Failed to create C++ index: {e}")
                print("  Falling back to Python implementation")
                self.use_cpp = False
                self.cpp_index = None
        else:
            if not HAS_CPP:
                print("ℹ Using pure Python ACORN implementation")
                print("  Build C++ extension for 10-100x speedup: python3 setup.py build_ext --inplace")
    
    def _metadata_to_int(self, meta: Dict) -> int:
        """Convert metadata dict to integer for C++ index."""
        # Simple hash: for this demo, we use category as integer
        # In production, you'd want a proper encoding scheme
        if 'category' in meta:
            val = meta['category']
            if isinstance(val, int):
                return val
            cat_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            return cat_map.get(val, 0)
        return 0
    
    def insert(self, vector: np.ndarray, meta: Dict[str, Any], doc_id: int):
        """
        Insert a vector with metadata.
        
        Args:
            vector: Vector to insert (dimension must match self.d)
            meta: Metadata dictionary
            doc_id: External document ID
        """
        self._meta_arr_cache = None  # Invalidate cache on insert
        
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape[0] != self.d:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match index dimension {self.d}")
        
        # Store metadata
        self.metadata[doc_id] = meta.copy()
        self.current_ids.add(doc_id)
        
        if self.use_cpp and self.cpp_index is not None:
            # Add to C++ index
            internal_id = len(self.cpp_doc_ids)
            self.cpp_doc_ids.append(doc_id)
            self.doc_id_to_internal[doc_id] = internal_id
            self.cpp_metadata_ints.append(self._metadata_to_int(meta))
            
            # C++ index expects batched add, so we buffer single vectors
            vectors_batch = vector.reshape(1, -1)
            metadata_batch = np.array([self._metadata_to_int(meta)], dtype=np.int32)
            self.cpp_index.add(vectors_batch, metadata_batch)
        else:
            # Pure Python fallback
            self.vectors[doc_id] = vector.copy()
    
    def insert_batch(self, vectors: np.ndarray, metas: List[Dict[str, Any]], doc_ids: List[int]):
        """
        Insert a batch of vectors.
        """
        self._meta_arr_cache = None # Invalidate cache
        n = len(doc_ids)
        if vectors.shape[0] != n:
             raise ValueError("Mismatch in batch sizes")
             
        # Update Python state
        for i in range(n):
            doc_id = doc_ids[i]
            self.metadata[doc_id] = metas[i].copy()
            self.current_ids.add(doc_id)
            
        if self.use_cpp and self.cpp_index is not None:
            start_internal_id = len(self.cpp_doc_ids)
            self.cpp_doc_ids.extend(doc_ids)
            
            # Update mapping
            for i in range(n):
                self.doc_id_to_internal[doc_ids[i]] = start_internal_id + i
                
            # Prepare metadata ints
            meta_ints = [self._metadata_to_int(m) for m in metas]
            self.cpp_metadata_ints.extend(meta_ints)
            
            # Call C++ add
            metadata_batch = np.array(meta_ints, dtype=np.int32)
            self.cpp_index.add(vectors, metadata_batch)
        else:
            for i in range(n):
                self.vectors[doc_ids[i]] = vectors[i].copy()

    def delete(self, doc_id: int):
        """
        Delete a vector by ID.
        Note: C++ ACORN doesn't support efficient deletes, so we track deletions.
        
        Args:
            doc_id: External document ID to delete
        """
        if doc_id in self.current_ids:
            self.current_ids.remove(doc_id)
            self._meta_arr_cache = None  # Invalidate cache on delete
            
            if doc_id in self.metadata:
                del self.metadata[doc_id]
            
            if not self.use_cpp and doc_id in self.vectors:
                del self.vectors[doc_id]
            
            # Call C++ delete_node if available
            if self.use_cpp and self.cpp_index is not None:
                if doc_id in self.doc_id_to_internal:
                    internal_id = self.doc_id_to_internal[doc_id]
                    try:
                        # Bounds check before calling C++
                        if 0 <= internal_id < self.cpp_index.ntotal:
                            self.cpp_index.delete_node(internal_id)
                    except (AttributeError, RuntimeError) as e:
                        # delete_node might not be exposed yet if extension not rebuilt
                        # or may fail for edge cases
                        pass
    
    def search(self, query_vec: np.ndarray, predicate: Callable[[Dict], bool], k: int = 10) -> Tuple[float, List[int], List[float]]:
        """
        Hybrid search: find k nearest neighbors satisfying the predicate.
        
        Args:
            query_vec: Query vector
            predicate: Function that takes metadata dict and returns True/False
            k: Number of neighbors to return
            
        Returns:
            (latency_ms, list of document IDs, list of distances)
        """
        t0 = time.time()
        query_vec = np.asarray(query_vec, dtype=np.float32)
        
        if len(self.current_ids) == 0:
            return (time.time() - t0) * 1000, [], []
        
        if self.use_cpp and self.cpp_index is not None:
            # C++ implementation
            result_ids, result_dists = self._search_cpp(query_vec, predicate, k)
        else:
            # Pure Python fallback
            result_ids, result_dists = self._search_python(query_vec, predicate, k)
        
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, result_ids, result_dists
    
    def search_category(self, query_vec: np.ndarray, category: int, k: int = 10) -> Tuple[float, List[int], List[float]]:
        """
        Optimized search for category equality.
        """
        t0 = time.time()
        query_vec = np.asarray(query_vec, dtype=np.float32)
        
        if len(self.current_ids) == 0:
            return (time.time() - t0) * 1000, [], []
            
        if self.use_cpp and self.cpp_index is not None:
            # Optimized bitmap generation
            # Note: converting list to array every time might be slow if done repeatedly.
            # But much faster than python loop.
            if self._meta_arr_cache is None:
                self._meta_arr_cache = np.array(self.cpp_metadata_ints, dtype=np.int32)
            meta_arr = self._meta_arr_cache
            
            filter_map = (meta_arr == category).astype(np.uint8).reshape(1, -1)
            
            query_batch = query_vec.reshape(1, -1)
            distances, labels = self.cpp_index.search(query_batch, k, filter_map)
            
            result_ids = []
            result_dists = []
            for i, internal_id in enumerate(labels[0]):
                if 0 <= internal_id < len(self.cpp_doc_ids):
                    doc_id = self.cpp_doc_ids[internal_id]
                    if doc_id in self.current_ids:
                        result_ids.append(doc_id)
                        result_dists.append(distances[0][i])
            
            latency_ms = (time.time() - t0) * 1000
            return latency_ms, result_ids[:k], result_dists[:k]
        else:
            # Fallback
            predicate = lambda meta: self._metadata_to_int(meta) == category
            return self.search(query_vec, predicate, k)
    
    def search_category_batch(self, queries: np.ndarray, category: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched search for category equality - much faster than single-query loop.
        
        Args:
            queries: (nq, d) array of query vectors
            category: Target category to filter by
            k: Number of neighbors per query
            
        Returns:
            (distances, labels) arrays of shape (nq, k)
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        
        if len(self.current_ids) == 0 or not self.use_cpp or self.cpp_index is None:
            return np.full((nq, k), float('inf')), np.full((nq, k), -1, dtype='int64')
        
        # Always rebuild filter map from cpp_metadata_ints to ensure size matches ntotal
        ntotal = self.cpp_index.ntotal
        n_meta = len(self.cpp_metadata_ints)
        
        if n_meta != ntotal:
            # This should not happen, but if it does, we have a sync issue
            print(f"WARNING: metadata sync issue - n_meta={n_meta}, ntotal={ntotal}")
            # Return empty results to avoid segfault
            return np.full((nq, k), float('inf')), np.full((nq, k), -1, dtype='int64')
        
        # Build filter bitmap once for all queries (same category)
        meta_arr = np.array(self.cpp_metadata_ints, dtype=np.int32)
        
        # Single filter map replicated for all queries
        single_filter = (meta_arr == category).astype(np.uint8)
        filter_map = np.tile(single_filter, (nq, 1))
        
        # Ensure contiguous
        filter_map = np.ascontiguousarray(filter_map)
        
        # Batch search
        distances, labels = self.cpp_index.search(queries, k, filter_map)
        
        # Map internal IDs to doc IDs
        labels_out = np.full((nq, k), -1, dtype='int64')
        for i in range(nq):
            for j in range(k):
                internal_id = labels[i, j]
                if 0 <= internal_id < len(self.cpp_doc_ids):
                    labels_out[i, j] = self.cpp_doc_ids[internal_id]
        
        return distances, labels_out

    def _search_cpp(self, query_vec: np.ndarray, predicate: Callable[[Dict], bool], k: int) -> Tuple[List[int], List[float]]:
        """Search using C++ implementation."""
        # Build filter bitmap for all vectors
        n_total = self.cpp_index.ntotal
        filter_map = np.zeros((1, n_total), dtype=np.uint8)
        
        for i, doc_id in enumerate(self.cpp_doc_ids):
            if doc_id in self.current_ids and doc_id in self.metadata:
                if predicate(self.metadata[doc_id]):
                    filter_map[0, i] = 1
        
        # Perform search
        query_batch = query_vec.reshape(1, -1)
        distances, labels = self.cpp_index.search(query_batch, k, filter_map)
        
        # Convert internal IDs to external doc_ids
        result_ids = []
        result_dists = []
        for i, internal_id in enumerate(labels[0]):
            if 0 <= internal_id < len(self.cpp_doc_ids):
                doc_id = self.cpp_doc_ids[internal_id]
                if doc_id in self.current_ids:  # Check not deleted
                    result_ids.append(doc_id)
                    result_dists.append(distances[0][i])
        
        return result_ids[:k], result_dists[:k]
    
    def _search_python(self, query_vec: np.ndarray, predicate: Callable[[Dict], bool], k: int) -> Tuple[List[int], List[float]]:
        """Pure Python brute force search."""
        # Filter vectors by predicate
        filtered_ids = [doc_id for doc_id in self.current_ids if predicate(self.metadata[doc_id])]
        
        if len(filtered_ids) == 0:
            return [], []
        
        # Compute distances for filtered vectors
        distances = []
        for doc_id in filtered_ids:
            vec = self.vectors[doc_id]
            dist = np.sum((query_vec - vec) ** 2)  # L2 distance
            distances.append((dist, doc_id))
        
        # Sort by distance and return top-k
        distances.sort(key=lambda x: x[0])
        return [doc_id for _, doc_id in distances[:k]], [dist for dist, _ in distances[:k]]
    
    def build_index(self):
        """Build/optimize the index after bulk insertions."""
        # C++ ACORN builds during add(), no separate build step needed
        pass
    
    def __len__(self):
        return len(self.current_ids)
    
    def __repr__(self):
        impl = "C++" if self.use_cpp else "Python"
        return f"ACORNIndex({impl}, d={self.d}, M={self.M}, gamma={self.gamma}, n={len(self)})"


# Backwards compatibility alias
class ACORN_Harness(ACORNIndex):
    """Alias for backwards compatibility with existing benchmark code."""
    pass
