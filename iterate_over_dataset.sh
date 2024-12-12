#!/bin/bash

# Define the list of OGB graph-level datasets
datasets=(
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

# Set default values for noise levels and repeats
n_noise_levels=10
n_repeats=5

# Iterate over each dataset and run the evaluation script
for dataset in "${datasets[@]}"; do
    echo "Evaluating dataset: $dataset"
    python ogb-graph-level.py --dataset "$dataset" --n_noise_levels "$n_noise_levels" --n_repeats "$n_repeats"
done
