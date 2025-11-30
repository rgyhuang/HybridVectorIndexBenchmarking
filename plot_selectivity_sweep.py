#!/usr/bin/env python3
"""
Plot selectivity and n_ops sweep benchmark results.

Generates plots for:
- QPS vs Selectivity (by method, for each n_ops)
- Recall vs Selectivity (by method, for each n_ops)
- QPS vs n_ops (by method, for each selectivity)
- Recall vs n_ops (by method)
- Insert/Delete latency comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Define colors for methods
METHOD_COLORS = {
    'ACORN-1': '#1f77b4',
    'ACORN-4': '#ff7f0e', 
    'ACORN-8': '#2ca02c',
    'ACORN-12': '#d62728',
    'Pre-Filter': '#9467bd',
    'Post-Filter-Strict': '#8c564b',
    'Post-Filter-Batch100': '#e377c2',
}

def plot_selectivity_sweep(input_file: str, output_prefix: str = "selectivity_sweep"):
    """Generate plots from selectivity sweep benchmark results."""
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Selectivities: {sorted(df['selectivity_pct'].unique())}")
    
    # Check if n_ops column exists
    has_n_ops = 'n_ops' in df.columns
    if has_n_ops:
        n_ops_values = sorted(df['n_ops'].unique())
        print(f"N_ops values: {n_ops_values}")
    else:
        n_ops_values = [None]
    
    methods = df['method'].unique()
    selectivities = sorted(df['selectivity_pct'].unique())
    
    # If we have n_ops dimension, generate more comprehensive plots
    if has_n_ops and len(n_ops_values) > 1:
        plot_with_n_ops(df, methods, selectivities, n_ops_values, output_prefix)
    else:
        plot_simple(df, methods, selectivities, output_prefix)
    
    print(f"\nAll plots saved with prefix: {output_prefix}")


def plot_with_n_ops(df, methods, selectivities, n_ops_values, output_prefix):
    """Generate plots when n_ops dimension is present."""
    
    # 1. Summary grid: QPS and Recall vs Selectivity for each n_ops
    n_n_ops = len(n_ops_values)
    fig, axes = plt.subplots(2, n_n_ops, figsize=(5 * n_n_ops, 10))
    if n_n_ops == 1:
        axes = axes.reshape(2, 1)
    
    for col, n_ops in enumerate(n_ops_values):
        ndf = df[df['n_ops'] == n_ops]
        
        # QPS vs Selectivity
        ax = axes[0, col]
        for method in methods:
            mdf = ndf[ndf['method'] == method].sort_values('selectivity_pct')
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['qps'], 'o-', label=method, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Selectivity (%)', fontsize=10)
        ax.set_ylabel('QPS', fontsize=10)
        ax.set_title(f'QPS vs Selectivity\n(n_ops={n_ops:,})', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Recall vs Selectivity
        ax = axes[1, col]
        for method in methods:
            mdf = ndf[ndf['method'] == method].sort_values('selectivity_pct')
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['recall'], 'o-', label=method, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Selectivity (%)', fontsize=10)
        ax.set_ylabel('Recall@10', fontsize=10)
        ax.set_title(f'Recall vs Selectivity\n(n_ops={n_ops:,})', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    output_file = f"{output_prefix}_by_nops.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # 2. QPS vs n_ops for each selectivity
    n_sel = len(selectivities)
    fig, axes = plt.subplots(2, n_sel, figsize=(4 * n_sel, 8))
    if n_sel == 1:
        axes = axes.reshape(2, 1)
    
    for col, sel in enumerate(selectivities):
        sdf = df[df['selectivity_pct'] == sel]
        
        # QPS vs n_ops
        ax = axes[0, col]
        for method in methods:
            mdf = sdf[sdf['method'] == method].sort_values('n_ops')
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['n_ops'], mdf['qps'], 'o-', label=method, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Number of Operations', fontsize=10)
        ax.set_ylabel('QPS', fontsize=10)
        ax.set_title(f'QPS vs n_ops\n(selectivity={sel:.0f}%)', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Recall vs n_ops
        ax = axes[1, col]
        for method in methods:
            mdf = sdf[sdf['method'] == method].sort_values('n_ops')
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['n_ops'], mdf['recall'], 'o-', label=method, color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Number of Operations', fontsize=10)
        ax.set_ylabel('Recall@10', fontsize=10)
        ax.set_title(f'Recall vs n_ops\n(selectivity={sel:.0f}%)', fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    output_file = f"{output_prefix}_by_selectivity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # 3. Latency plots (Insert and Delete)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pick middle selectivity and n_ops for representative latency
    mid_sel = selectivities[len(selectivities) // 2]
    mid_n_ops = n_ops_values[len(n_ops_values) // 2]
    
    # Insert latency vs n_ops (at middle selectivity)
    ax = axes[0, 0]
    sdf = df[df['selectivity_pct'] == mid_sel]
    for method in methods:
        mdf = sdf[sdf['method'] == method].sort_values('n_ops')
        if mdf['avg_insert_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['n_ops'], mdf['avg_insert_ms'], 'o-', label=method, color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Number of Operations', fontsize=10)
    ax.set_ylabel('Insert Latency (ms)', fontsize=10)
    ax.set_title(f'Insert Latency vs n_ops\n(selectivity={mid_sel:.0f}%)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Delete latency vs n_ops (at middle selectivity)
    ax = axes[0, 1]
    for method in methods:
        mdf = sdf[sdf['method'] == method].sort_values('n_ops')
        if mdf['avg_delete_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['n_ops'], mdf['avg_delete_ms'], 'o-', label=method, color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Number of Operations', fontsize=10)
    ax.set_ylabel('Delete Latency (ms)', fontsize=10)
    ax.set_title(f'Delete Latency vs n_ops\n(selectivity={mid_sel:.0f}%)', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Insert latency vs selectivity (at middle n_ops)
    ax = axes[1, 0]
    ndf = df[df['n_ops'] == mid_n_ops]
    for method in methods:
        mdf = ndf[ndf['method'] == method].sort_values('selectivity_pct')
        if mdf['avg_insert_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['avg_insert_ms'], 'o-', label=method, color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Selectivity (%)', fontsize=10)
    ax.set_ylabel('Insert Latency (ms)', fontsize=10)
    ax.set_title(f'Insert Latency vs Selectivity\n(n_ops={mid_n_ops:,})', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Delete latency vs selectivity (at middle n_ops)
    ax = axes[1, 1]
    for method in methods:
        mdf = ndf[ndf['method'] == method].sort_values('selectivity_pct')
        if mdf['avg_delete_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['avg_delete_ms'], 'o-', label=method, color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Selectivity (%)', fontsize=10)
    ax.set_ylabel('Delete Latency (ms)', fontsize=10)
    ax.set_title(f'Delete Latency vs Selectivity\n(n_ops={mid_n_ops:,})', fontsize=11)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    output_file = f"{output_prefix}_latency.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # 4. Heatmap of Recall by method/selectivity/n_ops
    for method in methods:
        mdf = df[df['method'] == method]
        if len(mdf) > 0:
            pivot = mdf.pivot_table(index='n_ops', columns='selectivity_pct', values='recall', aggfunc='mean')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
            
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{s:.0f}%' for s in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'{n:,}' for n in pivot.index])
            
            ax.set_xlabel('Selectivity (%)', fontsize=12)
            ax.set_ylabel('Number of Operations', fontsize=12)
            ax.set_title(f'Recall Heatmap: {method}', fontsize=14)
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    color = 'white' if val < 0.75 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            
            plt.colorbar(im, ax=ax, label='Recall@10')
            plt.tight_layout()
            
            safe_method = method.replace('-', '_').replace(' ', '_')
            output_file = f"{output_prefix}_heatmap_{safe_method}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
    
    # 5. Summary comparison at largest n_ops
    max_n_ops = max(n_ops_values)
    plot_simple(df[df['n_ops'] == max_n_ops], methods, selectivities, f"{output_prefix}_max_nops")


def plot_simple(df, methods, selectivities, output_prefix):
    """Generate simple plots without n_ops dimension."""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. QPS vs Selectivity
    ax = axes[0, 0]
    for method in methods:
        mdf = df[df['method'] == method].sort_values('selectivity_pct')
        color = METHOD_COLORS.get(method, None)
        ax.plot(mdf['selectivity_pct'], mdf['qps'], 'o-', label=method, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('QPS', fontsize=12)
    ax.set_title('Throughput vs Selectivity', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Recall vs Selectivity
    ax = axes[0, 1]
    for method in methods:
        mdf = df[df['method'] == method].sort_values('selectivity_pct')
        color = METHOD_COLORS.get(method, None)
        ax.plot(mdf['selectivity_pct'], mdf['recall'], 'o-', label=method, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('Recall vs Selectivity', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim([0.0, 1.05])
    
    # 3. Insert Latency vs Selectivity
    ax = axes[1, 0]
    for method in methods:
        mdf = df[df['method'] == method].sort_values('selectivity_pct')
        if mdf['avg_insert_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['avg_insert_ms'], 'o-', label=method, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('Insert Latency (ms)', fontsize=12)
    ax.set_title('Insert Latency vs Selectivity', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 4. Delete Latency vs Selectivity
    ax = axes[1, 1]
    for method in methods:
        mdf = df[df['method'] == method].sort_values('selectivity_pct')
        if mdf['avg_delete_ms'].sum() > 0:
            color = METHOD_COLORS.get(method, None)
            ax.plot(mdf['selectivity_pct'], mdf['avg_delete_ms'], 'o-', label=method, color=color, linewidth=2, markersize=8)
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('Delete Latency (ms)', fontsize=12)
    ax.set_title('Delete Latency vs Selectivity', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    output_file = f"{output_prefix}_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # QPS-Recall tradeoff plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for method in methods:
        mdf = df[df['method'] == method].sort_values('selectivity_pct')
        color = METHOD_COLORS.get(method, None)
        
        ax.scatter(mdf['qps'], mdf['recall'], s=100, label=method, color=color)
        ax.plot(mdf['qps'], mdf['recall'], '--', alpha=0.5, color=color)
        
        for _, row in mdf.iterrows():
            ax.annotate(f"{row['selectivity_pct']:.0f}%", 
                       (row['qps'], row['recall']),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('QPS (Throughput)', fontsize=12)
    ax.set_ylabel('Recall@10', fontsize=12)
    ax.set_title('QPS vs Recall Tradeoff (labels show selectivity %)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    output_file = f"{output_prefix}_qps_recall.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot selectivity sweep benchmark results")
    parser.add_argument('--input', type=str, default='selectivity_sweep_results.csv',
                       help='Input CSV file (default: selectivity_sweep_results.csv)')
    parser.add_argument('--output', type=str, default='selectivity_sweep',
                       help='Output file prefix (default: selectivity_sweep)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    plot_selectivity_sweep(args.input, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
