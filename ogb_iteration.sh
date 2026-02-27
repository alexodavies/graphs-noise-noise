#!/bin/bash

configs=("ogbg-molbace.yaml"
        "ogbg-molbbbp.yaml"
        "ogbg-molclintox.yaml"
        "ogbg-molesol.yaml"
        "ogbg-molfreesolv.yaml"
        "ogbg-molhiv.yaml"
        "ogbg-mollipo.yaml"
        "ogbg-molmuv.yaml"
        "ogbg-molpcba.yaml"
        "ogbg-molsider.yaml"
        "ogbg-moltox21.yaml"
        "ogbg-moltoxcast.yaml")

# Set default values for the arguments
n_noise_levels=10
n_repeats=10

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
for config in "${configs[@]}"; do
    echo "Evaluating dataset: $config"
    python graph-level.py \
        --config "$config" \
        --n_noise_levels "$n_noise_levels" \
        --n_repeats "$n_repeats"
done
