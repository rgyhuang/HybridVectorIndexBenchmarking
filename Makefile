# --- Configuration ---

# Path to the root of the cloned ACORN repository
ACORN_ROOT ?= ./ACORN

export CPLUS_INCLUDE_PATH=/home/ubuntu/HybridVectorIndexBenchmarking/ACORN:$CPLUS_INCLUDE_PATH


# Compiler (Ensure it supports C++17)
CXX ?= g++

# Python Interpreter (Change if using a specific venv)
PYTHON ?= python3

# --- Automatic Flag Discovery ---

# Get Python includes and extension suffix (e.g., .cpython-310-x86_64-linux-gnu.so)
PYTHON_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXTENSION_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Target file name
TARGET := acorn_ext$(EXTENSION_SUFFIX)

# --- Compilation Flags ---

# Include paths for ACORN and FAISS headers
# ACORN usually relies on FAISS headers located inside its structure or a submodule
INCLUDES := $(PYTHON_INCLUDES) \
            -I$(ACORN_ROOT)/include \
            -I$(ACORN_ROOT)/faiss \
            -I.

# Compiler Flags
# -O3: Max optimization
# -march=native: Optimize for current CPU (AVX2/AVX512)
# -fPIC: Position Independent Code (Required for shared libs)
# -fopenmp: ACORN/FAISS heavily use OpenMP
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC -fopenmp -march=native $(INCLUDES)

# --- Linker Flags ---

# Link against the compiled ACORN and FAISS libraries
# Adjust the -L path if your build output is in a different folder (e.g., build/faiss)
LDFLAGS := -L$(ACORN_ROOT)/build/lib \
           -L$(ACORN_ROOT)/build/faiss \
           -lacorn -lfaiss

# macOS specific linker flags to handle undefined symbols allowed in Python extensions
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -undefined dynamic_lookup
endif

# --- Build Rules ---

.PHONY: all clean test

all: $(TARGET)

$(TARGET): acorn_pybind.cpp
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Simple test to verify import works
test: $(TARGET)
	@echo "Testing import..."
	$(PYTHON) -c "import acorn_ext; print('Success! Acorn extension loaded:', acorn_ext.__doc__)"

clean:
	rm -f $(TARGET)
	rm -f *.so
	rm -f *.o