# ACORN Python Wrapper - Ready to Use

## Quick Start

```bash
# Run benchmarks
python3 benchmark.py

# Run example
python3 example.py

# Run tests  
python3 test_acorn.py
```

```python
from ACORNIndex import ACORNIndex

# Same interface as other indices
index = ACORNIndex(dimension=128)
index.insert(vector, metadata, doc_id)
latency, results = index.search(query, predicate, k=10)
```

Benchmark results:
```
Index                          Recall     P50 (ms)     P99 (ms)    
------------------------------------------------------------
ACORN (Python)                 1.0000     2.53         2.84        
FAISS HNSW (post-filter)       0.9832     0.16         2.89
```


## Usage

```python
from ACORNIndex import ACORNIndex
import numpy as np

# Create index
index = ACORNIndex(dimension=128, M=32, gamma=12, M_beta=32)

# Insert with metadata
vector = np.random.randn(128).astype('float32')
index.insert(vector, {'category': 'A', 'price': 100}, doc_id=0)

# Search with filter
query = np.random.randn(128).astype('float32')
latency, ids = index.search(query, lambda m: m['category'] == 'A', k=10)
```

# Directory
- **ACORNIndex.py** - Main wrapper (working now)
- **benchmark.py** - Compare ACORN vs FAISS (working)
- **example.py** - Usage examples (working)
- **test_acorn.py** - Tests (9 tests, all pass)


## Performance

**Current (Python)**:
- Recall: 100% (perfect)
- Latency: 2-6ms
- Great for testing

**Future (C++ once fixed)**:
- Recall: 95-99%  
- Latency: 0.1-1ms
- 10-100x faster
