#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <faiss/IndexACORN.h>
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/impl/FaissAssert.h>

namespace py = pybind11;

// Helper to access numpy data directly
template <typename T>
T* get_ptr(py::array_t<T> array) {
    return static_cast<T*>(array.request().ptr);
}

PYBIND11_MODULE(acorn_ext, m) {
    m.doc() = "Python bindings for ACORN Hybrid Index";

    py::enum_<faiss::MetricType>(m, "MetricType")
        .value("METRIC_INNER_PRODUCT", faiss::METRIC_INNER_PRODUCT)
        .value("METRIC_L2", faiss::METRIC_L2);

    py::class_<faiss::Index>(m, "Index"); // Base class

    py::class_<faiss::IndexACORN, faiss::Index>(m, "IndexACORN");

    py::class_<faiss::IndexACORNFlat, faiss::IndexACORN>(m, "IndexACORNFlat")
        .def(py::init<int, int, int, std::vector<int>&, int, faiss::MetricType>(),
             py::arg("d"), py::arg("M"), py::arg("gamma"), py::arg("metadata"), py::arg("M_beta"), py::arg("metric") = faiss::METRIC_L2)
        
        .def("add", [](faiss::IndexACORNFlat &index, py::array_t<float> x, py::array_t<int> metadata) {
            py::buffer_info buf_x = x.request();
            py::buffer_info buf_meta = metadata.request();

            if (buf_x.ndim != 2)
                throw std::runtime_error("x must be 2D array");
            if (buf_meta.ndim != 1)
                throw std::runtime_error("metadata must be 1D array");

            int n = buf_x.shape[0];
            int d = buf_x.shape[1];

            if (buf_meta.shape[0] != n)
                throw std::runtime_error("metadata size must match number of vectors");

            // Copy metadata to C++ vector
            std::vector<int> meta_vec(n);
            std::memcpy(meta_vec.data(), buf_meta.ptr, n * sizeof(int));
            
            // Add metadata to storage
            index.add_metadata(meta_vec);

            // Add vectors
            index.add(n, static_cast<float*>(buf_x.ptr));
        })
        
        .def("search", [](faiss::IndexACORNFlat &self, 
                          py::array_t<float> x, 
                          int k,
                          py::array_t<char> filter_id_map) {
            size_t n = x.shape(0);
            size_t d = x.shape(1);
            FAISS_THROW_IF_NOT(d == self.d);
            
            // Validate filter_id_map dimensions
            py::buffer_info filter_buf = filter_id_map.request();
            if (filter_buf.ndim != 2) {
                throw std::runtime_error("filter_id_map must be 2D array (n_queries, ntotal)");
            }
            size_t filter_rows = filter_buf.shape[0];
            size_t filter_cols = filter_buf.shape[1];
            if (filter_rows != n) {
                throw std::runtime_error("filter_id_map rows must match number of queries");
            }
            if (filter_cols != (size_t)self.ntotal) {
                throw std::runtime_error("filter_id_map columns (" + std::to_string(filter_cols) + 
                    ") must match ntotal (" + std::to_string(self.ntotal) + ")");
            }
            
            // Outputs
            py::array_t<float> distances({n, (size_t)k});
            py::array_t<int64_t> labels({n, (size_t)k});
            
            self.search(n, get_ptr(x), k, get_ptr(distances), get_ptr(labels), get_ptr(filter_id_map));
            
            return py::make_tuple(distances, labels);
        })
        .def("delete_node", [](faiss::IndexACORNFlat &self, faiss::idx_t node_id) {
            // Bounds check to prevent segfault
            if (node_id < 0 || node_id >= self.ntotal) {
                throw std::out_of_range("delete_node: node_id out of range");
            }
            self.delete_node(node_id);
        })
        .def("set_efSearch", [](faiss::IndexACORNFlat &self, int efSearch) {
            self.acorn.efSearch = efSearch;
        })
        .def_readwrite("ntotal", &faiss::Index::ntotal)
        .def_readwrite("d", &faiss::Index::d)
        ;
}
