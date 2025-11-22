# ACORN Benchmark Results

## Overview
We benchmarked ACORN against two baselines:
1. **Post-Filtering**: HNSW index on all vectors. Search retrieves `k*50` candidates, then filters by metadata.
2. **Pre-Filtering**: Exact search using `IndexIDMap` + `IndexFlatL2` on partitions (one partition per attribute value).

## 1. Static Workload
*Dataset: 10k vectors, 128 dim, 50 attributes. 100 queries.*

| Method | Build Time (s) | QPS | Recall@10 |
| :--- | :--- | :--- | :--- |
| **ACORN** | 23.68 | 289 | **0.77** |
| Post-Filter | 0.42 | 2972 | 0.31 |
| Pre-Filter | 0.01 | 1299 | 1.00 |

**Analysis**:
* **Recall**: ACORN achieves significantly higher recall (0.77) than Post-Filtering (0.31).
* **Speed**: ACORN is slower than baselines due to complex graph maintenance. Pre-filtering is fast on this small dataset because partitions are tiny (~200 items).

## 2. Dynamic Workload (Inserts + Searches)
*Workload: 5000 initial items, then stream 5000 items (1 insert, 1 search).*

| Method | Build Time (s) | Avg Insert (ms) | Avg Search (ms) | QPS | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ACORN** | 6.54 | 4.18 | 13.03 | 77 | **0.82** |
| Post-Filter | 0.22 | 0.73 | 0.47 | 2143 | 0.35 |
| Pre-Filter | 0.005 | 0.01 | 0.02 | 52340 | 1.00 |

**Analysis**:
* **Recall**: ACORN maintains high recall (0.82) during dynamic updates.
* **Latency**: Insert latency is ~4ms. Search latency increases to ~13ms (single query, no batching).

## 3. Update Workload (Inserts + Deletes + Searches)
*Workload: 5000 initial items, then stream 5000 items (1 insert, 1 delete, 1 search).*

| Method | Build Time (s) | Avg Insert (ms) | Avg Delete (ms) | Avg Search (ms) | QPS | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ACORN** | 6.56 | 4.22 | **0.003** | 12.84 | 78 | **0.76** |
| Post-Filter | 0.21 | 0.77 | 0.002 | 0.51 | 1974 | 0.32 |
| Pre-Filter | 0.005 | 0.01 | 0.02 | 0.02 | 54590 | 1.00 |

**Analysis**:
* **Deletes**: ACORN's delete operation is extremely fast (~3 microseconds) as it is a logical delete.
* **Stability**: Performance (QPS/Recall) remains stable even with mixed read/write/delete workloads.

## 4. Heavy Update Workload (Stress Test)
*Workload: 2000 initial items, then stream 20,000 items (1 insert, 1 delete, 1 search). Active set size remains ~2000.*

| Chunk (1k ops) | Total Inserted (C++ View) | Avg Search (ms) | QPS |
| :--- | :--- | :--- | :--- |
| 1 | 3,000 | 1.01 | 994 |
| 5 | 7,000 | 1.47 | 682 |
| 10 | 12,000 | 1.87 | 536 |
| 15 | 17,000 | 2.43 | 411 |
| 20 | 22,000 | 2.91 | 343 |

**Analysis**:
* **Degradation**: We observe a **3x degradation** in search performance (1.01ms -> 2.91ms) as the number of operations increases, even though the active dataset size remains constant.
* **Cause**:
    1. **Logical Deletes**: Deleted items remain in the HNSW graph ("ghost nodes"), increasing the traversal path length to find valid neighbors.
    2. **Filter Construction**: The Python wrapper iterates over all historically inserted IDs to build the filter bitmask, which is an O(N_total) operation.
* **Conclusion**: While ACORN handles updates functionally, it requires periodic "garbage collection" or graph rebuilding to maintain performance in high-churn environments.

## Conclusion
ACORN successfully bridges the gap between Pre-filtering (high recall, hard to scale/manage partitions) and Post-filtering (fast, poor recall). It provides good recall (~0.76-0.82) while supporting dynamic updates and deletes with reasonable latency. However, for long-running high-churn workloads, the current implementation suffers from performance degradation due to the accumulation of deleted nodes.
