#!/bin/bash

# Define the list of OGB graph-level datasets
datasets=(
    "ogbg-molhiv"
    "ogbg-molpcba"
    "ogbg-moltox21"
    "ogbg-moltoxcast"
    "ogbg-molbace"
    "ogbg-molbbbp"
    "ogbg-molclintox"
    "ogbg-molmuv"
    "ogbg-molsider"
    "ogbg-mollipo"
    "ogbg-molfreesolv"
    "ogbg-molesol"
)

# Set default values for noise levels and repeats
n_noise_levels=10
n_repeats=5

# Iterate over each dataset and run the evaluation script
for dataset in "${datasets[@]}"; do
    echo "Evaluating dataset: $dataset"
    python your_evaluation_script.py --dataset "$dataset" --n_noise_levels "$n_noise_levels" --n_repeats "$n_repeats"
done
