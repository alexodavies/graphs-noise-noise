#!/bin/bash

# Define the list of OGB graph-level datasets
datasets=(
    synth-easy
    synth-feature
    synth-structure
    synth-coupled
    TUDataset:PROTEINS
    TUDataset:ENZYMES
    ogbg-molbace
    ogbg-molbbbp
    ogbg-molclintox
    ogbg-molesol
    ogbg-molfreesolv
    ogbg-molhiv
    ogbg-mollipo
    ogbg-molsider
    ogbg-moltox21
)

# Define the list of GNN layers
layers=("graphormer")

# Define the structure options
structures=("False" "True")

# Set default values for the arguments
n_noise_levels=10
n_repeats=5
epochs=50
use_linear_flag="False"  # Hardcoded to False
batch_size=128

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
for layer in "${layers[@]}"; do
    for dataset in "${datasets[@]}"; do
        # for structure in "${structures[@]}"; do
        echo "Evaluating dataset: $dataset with layer: $layer"
        python graph-level.py \
            --dataset "$dataset" \
            --n_noise_levels "$n_noise_levels" \
            --n_repeats "$n_repeats" \
            --layer "$layer" \
            --epochs "$epochs"\
            --batch_size "$batch_size"\
            --fixed-train "True"
        # done
    done
done