import os
import yaml

# List of datasets
datasets = [
    "ogbg-molesol",
    "ogbg-molbbbp",
    "ogbg-molclintox",
    "ogbg-mollipo",
    "ogbg-molfreesolv",
    "ogbg-molbace",
    "ogbg-molmuv",
    "ogbg-molsider",
    "ogbg-molhiv",
    "ogbg-moltox21",
    "ogbg-molpcba",
    "ogbg-moltoxcast"
]

# Base sweep configuration
def generate_yaml_config(dataset_name):
    return {
        "name": dataset_name,
        "program": "basic_performance.py",  # Updated script name
        "project": "performance-ogbg",
        "entity": "hierarchical-diffusion",
        "method": "bayes",
        "metric": {
            "name": "Performance",  # Replace with the actual metric name
            "goal": "minimize"
        },
        "parameters": {
            "dataset": {"value": dataset_name},
            "layer_type": {
                "values": ["gcn", "gin", "gat", "gps"]
            },
            "hidden_dim": {
                "min": 16, "max": 512
            },
            "num_layers": {
                "min": 1, "max": 10
            },
            "batch_size": {
                "min": 16, "max": 1024
            },
            "epochs": {
                "min": 10, "max": 50
            },
            "lr": {
                "min": 0.0001, "max": 0.01
            }
        }
    }

# Ensure the sweeps directory exists
os.makedirs("sweeps", exist_ok=True)

# Generate YAML files for each dataset
for dataset in datasets:
    config = generate_yaml_config(dataset)
    yaml_path = os.path.join("sweeps", f"{dataset}.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

print("YAML files generated successfully in the sweeps/ directory.")
