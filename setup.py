import os
import pybind11
from setuptools import setup, Extension

# Point these to your cloned ACORN/FAISS directories
ACORN_ROOT = "/home/ubuntu/HybridVectorIndexBenchmarking/ACORN"

ext_modules = [
    Extension(
        "acorn_ext",
        ["acorn_pybind.cpp"],
        include_dirs=[
            pybind11.get_include(),
            ACORN_ROOT,
            os.path.join(ACORN_ROOT, "faiss"), 
        ],
        library_dirs=[os.path.join(ACORN_ROOT, "build/faiss")],
        libraries=["faiss"], # Link against the compiled ACORN/FAISS libs
        runtime_library_dirs=[os.path.join(ACORN_ROOT, "build/faiss")], # Embed rpath
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-mavx2', '-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
]

setup(name="acorn_ext", ext_modules=ext_modules)
