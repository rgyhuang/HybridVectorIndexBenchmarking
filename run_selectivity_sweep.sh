#!/bin/bash
#
# Run benchmark across multiple selectivity levels and n_ops values
# All results are collected into a single CSV file for plotting
#

set -e  # Exit on error

# Configuration
N_INIT=100000
OUTPUT_FILE="selectivity_sweep_results.csv"
SELECTIVITIES=(1.0 5.0 10.0 20.0 50.0)
N_OPS_VALUES=(1000 5000 10000 50000)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-init)
            N_INIT="$2"
            shift 2
            ;;
        --n-ops-values)
            IFS=',' read -ra N_OPS_VALUES <<< "$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --selectivities)
            IFS=',' read -ra SELECTIVITIES <<< "$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-init N           Initial index size (default: 100000)"
            echo "  --n-ops-values N     Comma-separated n_ops values (default: 1000,5000,10000,50000)"
            echo "  --output FILE        Output CSV file (default: selectivity_sweep_results.csv)"
            echo "  --selectivities S    Comma-separated selectivity values (default: 1.0,2.0,5.0,10.0,20.0)"
            echo ""
            echo "Example:"
            echo "  $0 --n-init 50000 --n-ops-values 1000,5000 --selectivities 1.0,5.0,10.0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Remove existing output file to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing $OUTPUT_FILE"
    rm "$OUTPUT_FILE"
fi

# Calculate total runs
TOTAL_RUNS=$((${#SELECTIVITIES[@]} * ${#N_OPS_VALUES[@]}))
CURRENT_RUN=0

echo "========================================"
echo "Selectivity & N_Ops Sweep Benchmark"
echo "========================================"
echo "N_INIT: $N_INIT"
echo "N_OPS values: ${N_OPS_VALUES[*]}"
echo "Selectivities: ${SELECTIVITIES[*]}"
echo "Total runs: $TOTAL_RUNS"
echo "Output: $OUTPUT_FILE"
echo "========================================"
echo ""

# Run benchmark for each combination of selectivity and n_ops
for N_OPS in "${N_OPS_VALUES[@]}"; do
    for SEL in "${SELECTIVITIES[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo ""
        echo "========================================"
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: selectivity=${SEL}%, n_ops=${N_OPS}"
        echo "========================================"
        echo ""
        
        python3 benchmark.py \
            --update \
            --n-init "$N_INIT" \
            --n-ops "$N_OPS" \
            --selectivity "$SEL" \
            --output "$OUTPUT_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✓ Completed selectivity=${SEL}%, n_ops=${N_OPS}"
        else
            echo "✗ Failed at selectivity=${SEL}%, n_ops=${N_OPS}"
            exit 1
        fi
    done
done

echo ""
echo "========================================"
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================"

# Print summary
echo ""
echo "CSV Preview (first 20 lines):"
head -20 "$OUTPUT_FILE"

echo ""
echo "To plot results, run:"
echo "  python3 plot_selectivity_sweep.py --input $OUTPUT_FILE"
