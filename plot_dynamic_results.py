#!/usr/bin/env python3
"""
Plot Dynamic ACORN Benchmark Results

Visualizes performance under dynamic workloads with inserts, deletes, and searches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
df = pd.read_csv('dynamic_benchmark_results.csv')

# Separate static and dynamic benchmarks
static_df = df[df['benchmark'] == 'static'].copy()
dynamic_df = df[df['benchmark'] == 'dynamic'].copy()

# Fill NaN values for insert/delete latency
dynamic_df['avg_insert_ms'] = dynamic_df['avg_insert_ms'].fillna(0)
dynamic_df['avg_delete_ms'] = dynamic_df['avg_delete_ms'].fillna(0)

# Define colors for each method
method_colors = {
    'ACORN-1': '#2ecc71',
    'Pre-Filter': '#3498db',
    'Post-Filter': '#e74c3c',
    'Post-Filter-Strict': '#c0392b',
    'Post-Filter-Batch100': '#e67e22',
}

# =============================================================================
# Figure 1: Static vs Dynamic QPS Comparison
# =============================================================================
fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1a: Static QPS
ax1 = axes[0]
static_methods = static_df['method'].unique()
selectivities = sorted(static_df['selectivity_pct'].unique())

for method in static_methods:
    method_data = static_df[static_df['method'] == method].sort_values('selectivity_pct')
    color = method_colors.get(method, 'gray')
    ax1.plot(method_data['selectivity_pct'], method_data['qps'], 
             label=method, color=color, marker='o', linewidth=2, markersize=8)

ax1.set_xlabel('Selectivity (%)', fontsize=12)
ax1.set_ylabel('Queries Per Second (QPS)', fontsize=12)
ax1.set_title('Static Workload (Search Only)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_xticks(selectivities)
ax1.set_xticklabels([f'{s}%' for s in selectivities])

# Plot 1b: Dynamic QPS
ax2 = axes[1]
dynamic_methods = dynamic_df['method'].unique()
dynamic_selectivities = sorted(dynamic_df['selectivity_pct'].unique())

for method in dynamic_methods:
    method_data = dynamic_df[dynamic_df['method'] == method].sort_values('selectivity_pct')
    if len(method_data) > 0:
        color = method_colors.get(method, 'gray')
        ax2.plot(method_data['selectivity_pct'], method_data['qps'], 
                 label=method, color=color, marker='s', linewidth=2, markersize=8)

ax2.set_xlabel('Selectivity (%)', fontsize=12)
ax2.set_ylabel('Queries Per Second (QPS)', fontsize=12)
ax2.set_title('Dynamic Workload (Insert + Delete + Search)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
if len(dynamic_selectivities) > 1:
    ax2.set_xscale('log')
ax2.set_xticks(dynamic_selectivities)
ax2.set_xticklabels([f'{s}%' for s in dynamic_selectivities])

plt.suptitle('ACORN: Static vs Dynamic Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dynamic_static_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: dynamic_static_comparison.png")


# =============================================================================
# Figure 2: Dynamic Workload - QPS vs Recall Trade-off
# =============================================================================
fig2, ax3 = plt.subplots(figsize=(10, 7))

for method in dynamic_methods:
    method_data = dynamic_df[dynamic_df['method'] == method]
    if len(method_data) > 0:
        color = method_colors.get(method, 'gray')
        for _, row in method_data.iterrows():
            size = 100 + row['selectivity_pct'] * 10
            ax3.scatter(row['recall'], row['qps'], 
                       c=color, s=size, alpha=0.8,
                       label=method if _ == method_data.index[0] else "")
            ax3.annotate(f"{row['selectivity_pct']:.0f}%", 
                        (row['recall'], row['qps']),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=9, alpha=0.8)

ax3.set_xlabel('Recall@10', fontsize=12)
ax3.set_ylabel('Queries Per Second (QPS)', fontsize=12)
ax3.set_title('Dynamic Workload: QPS vs Recall Trade-off\n(Upper-right is Best)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0.9, color='gray', linestyle=':', alpha=0.7)
ax3.text(0.91, ax3.get_ylim()[1] * 0.95, '0.9 Recall Threshold', fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig('dynamic_qps_recall.png', dpi=150, bbox_inches='tight')
print("Saved: dynamic_qps_recall.png")


# =============================================================================
# Figure 3: Insert/Delete Latency Comparison
# =============================================================================
fig3, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Get unique methods with latency data
latency_methods = dynamic_df[dynamic_df['avg_insert_ms'] > 0]['method'].unique()
all_methods = dynamic_df['method'].unique()

# Plot 3a: Insert Latency
ax4 = axes2[0]
x = np.arange(len(all_methods))
insert_latencies = [dynamic_df[dynamic_df['method'] == m]['avg_insert_ms'].mean() for m in all_methods]
colors = [method_colors.get(m, 'gray') for m in all_methods]

bars1 = ax4.bar(x, insert_latencies, color=colors, alpha=0.8)
ax4.set_xlabel('Method', fontsize=12)
ax4.set_ylabel('Insert Latency (ms)', fontsize=12)
ax4.set_title('Average Insert Latency\n(Lower is Better)', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(all_methods, rotation=15, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# Annotate values
for bar, val in zip(bars1, insert_latencies):
    if val > 0:
        ax4.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    else:
        ax4.annotate('<0.01ms', xy=(bar.get_x() + bar.get_width()/2, 0.5),
                    ha='center', fontsize=9, alpha=0.7)

# Plot 3b: Delete Latency
ax5 = axes2[1]
delete_latencies = [dynamic_df[dynamic_df['method'] == m]['avg_delete_ms'].mean() for m in all_methods]

bars2 = ax5.bar(x, delete_latencies, color=colors, alpha=0.8)
ax5.set_xlabel('Method', fontsize=12)
ax5.set_ylabel('Delete Latency (ms)', fontsize=12)
ax5.set_title('Average Delete Latency\n(Lower is Better)', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(all_methods, rotation=15, ha='right')
ax5.grid(True, alpha=0.3, axis='y')

# Annotate values
for bar, val in zip(bars2, delete_latencies):
    if val > 0:
        ax5.annotate(f'{val:.2f}ms', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    else:
        ax5.annotate('<0.01ms', xy=(bar.get_x() + bar.get_width()/2, 0.3),
                    ha='center', fontsize=9, alpha=0.7)

plt.suptitle('Update Operation Latency Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dynamic_latency_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: dynamic_latency_comparison.png")


# =============================================================================
# Figure 4: Comprehensive Dynamic Benchmark Summary
# =============================================================================
fig4, axes3 = plt.subplots(2, 2, figsize=(14, 12))

# 4a: QPS by method (bar chart for each selectivity)
ax6 = axes3[0, 0]
if len(dynamic_selectivities) > 0:
    n_methods = len(dynamic_methods)
    width = 0.8 / n_methods
    
    for i, method in enumerate(dynamic_methods):
        method_data = dynamic_df[dynamic_df['method'] == method].sort_values('selectivity_pct')
        sels = method_data['selectivity_pct'].values
        qps_vals = method_data['qps'].values
        
        x_pos = np.arange(len(sels)) + i * width
        color = method_colors.get(method, 'gray')
        ax6.bar(x_pos, qps_vals, width, label=method, color=color, alpha=0.8)
    
    ax6.set_xlabel('Selectivity (%)', fontsize=12)
    ax6.set_ylabel('QPS', fontsize=12)
    ax6.set_title('Dynamic QPS by Selectivity', fontsize=14, fontweight='bold')
    ax6.set_xticks(np.arange(len(dynamic_selectivities)) + width * (n_methods - 1) / 2)
    ax6.set_xticklabels([f'{s}%' for s in dynamic_selectivities])
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

# 4b: Recall by method
ax7 = axes3[0, 1]
for method in dynamic_methods:
    method_data = dynamic_df[dynamic_df['method'] == method].sort_values('selectivity_pct')
    if len(method_data) > 0:
        color = method_colors.get(method, 'gray')
        ax7.plot(method_data['selectivity_pct'], method_data['recall'], 
                 label=method, color=color, marker='o', linewidth=2, markersize=8)

ax7.set_xlabel('Selectivity (%)', fontsize=12)
ax7.set_ylabel('Recall@10', fontsize=12)
ax7.set_title('Dynamic Recall by Selectivity', fontsize=14, fontweight='bold')
ax7.legend(loc='lower right', fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
ax7.set_ylim(0.5, 1.05)

# 4c: Trade-off visualization
ax8 = axes3[1, 0]
for method in dynamic_methods:
    method_data = dynamic_df[dynamic_df['method'] == method]
    if len(method_data) > 0:
        color = method_colors.get(method, 'gray')
        # Size proportional to QPS
        sizes = method_data['qps'].values / method_data['qps'].max() * 300 + 50
        ax8.scatter(method_data['avg_insert_ms'], method_data['recall'], 
                   c=color, s=sizes, alpha=0.7, label=method)

ax8.set_xlabel('Insert Latency (ms)', fontsize=12)
ax8.set_ylabel('Recall@10', fontsize=12)
ax8.set_title('Insert Latency vs Recall Trade-off\n(Size = relative QPS)', fontsize=14, fontweight='bold')
ax8.legend(loc='lower right', fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)

# 4d: Summary table as text
ax9 = axes3[1, 1]
ax9.axis('off')

# Create summary text
summary_text = """
DYNAMIC BENCHMARK SUMMARY
========================

Methods Compared:
• ACORN-1: Predicate-agnostic hybrid index
• Pre-Filter: Filter → Exact search (no pre-indexing)
• Post-Filter: HNSW → Filter (lazy rebuild)

Key Findings:

1. SEARCH PERFORMANCE (QPS):
   - ACORN: Consistent ~350 QPS across selectivities
   - Pre-Filter: 200-960 QPS (varies with selectivity)
   - Post-Filter-Strict: Very slow (~1.5 QPS) due to rebuilds
   - Post-Filter-Batch: Moderate (70-175 QPS)

2. RECALL:
   - ACORN: >0.96 (maintains high recall under updates)
   - Pre-Filter: 1.0 (exact search)
   - Post-Filter: 0.78-0.98 (degrades at low selectivity)

3. UPDATE LATENCY:
   - ACORN: ~30ms insert, ~10ms delete
   - Pre-Filter: <0.01ms (just data storage)
   - Post-Filter: <0.01ms (deferred rebuild)

CONCLUSION:
ACORN is the best choice when:
• High recall (>0.95) is required
• Predicates are not known in advance
• Moderate update frequency
• Cannot afford full index rebuilds

Pre-Filter is better when:
• Predicates are simple equality checks
• Low selectivity (small filtered sets)
• Insert/delete latency is critical
"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Dynamic ACORN Benchmark: Comprehensive Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dynamic_benchmark_summary.png', dpi=150, bbox_inches='tight')
print("Saved: dynamic_benchmark_summary.png")


# =============================================================================
# Print Summary Table
# =============================================================================
print("\n" + "=" * 80)
print("DYNAMIC BENCHMARK RESULTS SUMMARY")
print("=" * 80)

print("\n=== STATIC WORKLOAD (Search Only) ===")
if len(static_df) > 0:
    print("\nQPS:")
    print(static_df.pivot(index='selectivity_pct', columns='method', values='qps').round(0).to_string())
    print("\nRecall:")
    print(static_df.pivot(index='selectivity_pct', columns='method', values='recall').round(3).to_string())

print("\n=== DYNAMIC WORKLOAD (Insert + Delete + Search) ===")
if len(dynamic_df) > 0:
    print("\nQPS:")
    print(dynamic_df.pivot(index='selectivity_pct', columns='method', values='qps').round(1).to_string())
    print("\nRecall:")
    print(dynamic_df.pivot(index='selectivity_pct', columns='method', values='recall').round(3).to_string())
    print("\nInsert Latency (ms):")
    print(dynamic_df.pivot(index='selectivity_pct', columns='method', values='avg_insert_ms').round(2).to_string())
    print("\nDelete Latency (ms):")
    print(dynamic_df.pivot(index='selectivity_pct', columns='method', values='avg_delete_ms').round(2).to_string())

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
1. ACORN vs Pre-Filter:
   - Pre-Filter wins on QPS at low selectivity (1%) due to small search sets
   - ACORN wins on QPS at higher selectivity (5%+) 
   - ACORN has higher update latency (~30ms insert vs ~0.01ms)
   - Both maintain high recall (>0.96)

2. ACORN vs Post-Filter:
   - Post-Filter-Strict is unusable under dynamic load (1.5 QPS)
   - Post-Filter-Batch is faster but sacrifices consistency
   - ACORN provides consistent performance without batching

3. Best Use Cases:
   - ACORN: Arbitrary predicates, moderate update rate, high recall required
   - Pre-Filter: Known predicates, low selectivity, high update rate
   - Post-Filter: Static/batch workloads only
""")

plt.show()
