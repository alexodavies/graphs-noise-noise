#!/bin/bash

# Define the list of OGB graph-level datasets
datasets=(
    "synth-coupled"
    "synth-easy"
    "synth-feature"
    "synth-structure"
)

# Define the list of GNN layers
layers=("gcn" "gin" "gat" "gps")

# Define the structure options
structures=("True" "False")

# Set default values for the arguments
n_noise_levels=10
n_repeats=5
use_linear_flag="False"  # Hardcoded to False

# Parse command-line arguments
while getopts "n:r:" opt; do
    case $opt in
        n) n_noise_levels=$OPTARG ;;          # Number of noise levels
        r) n_repeats=$OPTARG ;;               # Number of repeats
        *) echo "Usage: $0 [-n noise_levels] [-r repeats]" >&2; exit 1 ;;
    esac
done

# Shift the parsed options out of the positional arguments
shift "$((OPTIND - 1))"


# Iterate over each dataset, layer, and structure flag
for dataset in "${datasets[@]}"; do
    for layer in "${layers[@]}"; do
        # for structure in "${structures[@]}"; do
        echo "Evaluating dataset: $dataset with layer: $layer and structure: $structure"
        python graph-level.py \
            --dataset "$dataset" \
            --n_noise_levels "$n_noise_levels" \
            --n_repeats "$n_repeats" \
            --use_linear False \
            --layer "$layer"
        # done
    done
done

# Iterate over each dataset, layer, and structure flag
for dataset in "${datasets[@]}"; do
    for layer in "${layers[@]}"; do
        # for structure in "${structures[@]}"; do
        echo "Evaluating dataset: $dataset with layer: $layer and structure: $structure"
        python graph-level.py \
            --dataset "$dataset" \
            --n_noise_levels "$n_noise_levels" \
            --n_repeats "$n_repeats" \
            --use_linear False \
            --structure True \
            --layer "$layer"
        # done
    done
done
