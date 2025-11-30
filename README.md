# Hybrid Index Benchmarking Suite


## ACORN Wrapper Usage

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
- **ACORNIndex.py** - Main wrapper 
- **benchmark.py** - Compare ACORN vs FAISS (Pre/post filtering)

## Performance


# Run full sweep 

./run_selectivity_sweep.sh

(default: n_ops=1000,5000,10000,50000 × selectivities=1,2,5,10,20%)

# Custom parameters
./run_selectivity_sweep.sh \
    --n-init 100000 \
    --n-ops-values 1000,5000,10000,50000 \
    --selectivities 1.0,5.0,10.0,20.0,50.0 \
    --output results.csv

# Generate plots
python3 plot_selectivity_sweep.py --input results.csv --output my_plots