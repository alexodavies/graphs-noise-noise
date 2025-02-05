#!/bin/bash
# Loop through each YAML file and run the wandb sweep command
for file in *.yaml; do
    echo "Starting sweep for $file"
    wandb sweep "$file"
done
