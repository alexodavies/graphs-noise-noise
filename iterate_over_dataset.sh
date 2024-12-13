#!/bin/bash

# Define the list of OGB graph-level datasets
datasets=(
    "TUDataset:PROTEINS"
    "TUDataset:IMDB-BINARY"
    "TUDataset:COLLAB"
    "TUDataset:REDDIT-BINARY"
    "TUDataset:ENZYMES"
    "TUDataset:MUTAG"
    "ogbg-molesol"
    "ogbg-molbbbp"
    "ogbg-molclintox"
    "ogbg-mollipo"
    "ogbg-molfreesolv"
    "ogbg-molbace"
    "ogbg-molmuv"
    "ogbg-molsider"
    "ogbg-molhiv"
    "ogbg-moltox21"
    "ogbg-molpcba"
    "ogbg-moltoxcast"
)

# Set default values for the arguments
n_noise_levels=10
n_repeats=5
use_linear_flag=""  # Empty by default (not passed to Python)
layer="gcn"

# Parse command-line arguments
while getopts "n:r:l:u" opt; do
    case $opt in
        n) n_noise_levels=$OPTARG ;;  # Number of noise levels
        r) n_repeats=$OPTARG ;;       # Number of repeats
        l) layer=$OPTARG ;;           # Type of GNN layer
        u) use_linear_flag="--use_linear" ;;  # Add the --use_linear flag if specified
        *) echo "Usage: $0 [-n noise_levels] [-r repeats] [-l layer] [-u]" >&2; exit 1 ;;
    esac
done

# Iterate over each dataset and run the evaluation script
for dataset in "${datasets[@]}"; do
    echo "Evaluating dataset: $dataset"
    python graph-level.py --dataset "$dataset" --n_noise_levels "$n_noise_levels" --n_repeats "$n_repeats" $use_linear_flag --layer "$layer"
done