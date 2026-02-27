#!/bin/bash

# Run the node-level noise-noise experiment across all scenarios and GNN layers.
# Mirrors the style of big_iteration_fixed_train.sh.
#
# Usage:
#   bash run_node_classification.sh              # defaults
#   bash run_node_classification.sh -n 11 -r 5  # custom noise levels / repeats

scenarios=("easy" "feature" "structure" "coupled")
layers=("gcn" "gin" "gat")

# Defaults
n_noise_levels=11
n_repeats=5
epochs=50
batch_size=128
num_samples=2000

# Parse optional flags
while getopts "n:r:" opt; do
    case $opt in
        n) n_noise_levels=$OPTARG ;;
        r) n_repeats=$OPTARG ;;
        *) echo "Usage: $0 [-n noise_levels] [-r repeats]" >&2; exit 1 ;;
    esac
done
shift "$((OPTIND - 1))"

for layer in "${layers[@]}"; do
    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario with layer: $layer"
        python node-level.py \
            --scenario "$scenario" \
            --layer_type "$layer" \
            --n_noise_levels "$n_noise_levels" \
            --n_repeats "$n_repeats" \
            --epochs "$epochs" \
            --batch_size "$batch_size" \
            --num_samples "$num_samples"
    done
done
