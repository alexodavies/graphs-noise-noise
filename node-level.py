"""
Node-level noise-noise experiment.

Generates synthetic graphs with controlled feature/degree signals (4 scenarios)
and runs the NNRD noise sweep at node-classification level.

Usage
-----
  # Single scenario
  python node-level.py --scenario feature --layer_type gin --epochs 30

  # All scenarios
  python node-level.py --scenario all --layer_type gin

  # Quick smoke test
  python node-level.py --scenario feature --epochs 5 --num_samples 200 \
      --n_noise_levels 5 --n_repeats 2
"""

import os
import json
import warnings
import argparse

import numpy as np
from tqdm import tqdm
import wandb

from metrics import plot_results, nnd
from node_classification.evaluation import NodeClassificationConfig, run_node_noise_sweep

warnings.simplefilter(action='ignore', category=FutureWarning)

ALL_SCENARIOS = ("easy", "feature", "structure", "coupled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_node_result(result_dict: dict, save_dir: str) -> str:
    """Save result dict as JSON and return the file path."""
    os.makedirs(save_dir, exist_ok=True)
    scenario = result_dict["dataset"].replace("node-", "")
    layer = result_dict["layer"]
    path = os.path.join(save_dir, f"{scenario}_{layer}.json")
    with open(path, "w") as f:
        json.dump(result_dict, f, indent=2)
    return path


def run_scenario(scenario: str, args: argparse.Namespace) -> dict:
    """Run the full noise sweep for one scenario, log to WandB, and save results."""

    config = NodeClassificationConfig(
        layer_type=args.layer_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        n_noise_levels=args.n_noise_levels,
        n_repeats=args.n_repeats,
        num_samples=args.num_samples,
    )

    run_name = f"node-{scenario}-{args.layer_type}-ER-swapping"
    wandb.init(
        project="noise-ToP",
        entity="hierarchical-diffusion",
        name=run_name,
        config=vars(args),
        reinit=True,
    )
    wandb.log({"noise_type": "ER-Swapping", "scenario": scenario})

    result_dict = run_node_noise_sweep(scenario, config, verbose=True)

    # Log per-noise-level metrics
    struct_perfs = result_dict["structure"]
    feat_perfs = result_dict["feature"]
    for t_str in struct_perfs:
        t = float(t_str)
        wandb.log({
            "noise_level": t,
            "structure_performance": np.mean([float(v) for v in struct_perfs[t_str]]),
            "feature_performance": np.mean([float(v) for v in feat_perfs[t_str]]),
        })

    nnrd_val = nnd(result_dict)
    wandb.log({"NND": nnrd_val})

    # Plot
    os.makedirs("figures", exist_ok=True)
    image_path = plot_results(
        result_dict,
        extra_save_string=args.layer_type,
        return_path=True,
    )
    wandb.log({"Media/Result-Image": wandb.Image(image_path)})

    # Save JSON and upload as artifact
    json_path = save_node_result(result_dict, args.save_dir)
    artifact = wandb.Artifact(name="result_dict", type="result_output")
    artifact.add_file(local_path=json_path, name="results.json")
    artifact.save()

    wandb.finish()

    print(f"[{scenario}] NNRD = {nnrd_val:.4f}  →  {json_path}")
    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Node-level noise-noise experiment"
    )
    parser.add_argument(
        "--scenario", type=str, default="all",
        help="Scenario to run: easy|feature|structure|coupled|all"
    )
    parser.add_argument(
        "--layer_type", type=str, default="gin",
        help="GNN layer type: gin|gcn|gat"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="Hidden layer dimension (default: 64)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3,
        help="Number of GNN layers (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs per noise level repeat (default: 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--n_noise_levels", type=int, default=11,
        help="Number of noise levels in [0,1] (default: 11)"
    )
    parser.add_argument(
        "--n_repeats", type=int, default=5,
        help="Repeats per noise level (default: 5)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000,
        help="Number of graphs in dataset (default: 2000)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results/node_classification",
        help="Directory to save JSON results (default: results/node_classification)"
    )

    args = parser.parse_args()
    print(args)

    scenarios_to_run = (
        ALL_SCENARIOS if args.scenario == "all" else (args.scenario,)
    )

    for scenario in tqdm(scenarios_to_run, desc="Scenarios"):
        run_scenario(scenario, args)
