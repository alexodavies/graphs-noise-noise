"""
Sensitivity evaluation module for NNRD metric consistency and invariance testing.

This module provides functions to evaluate NNRD stability across:
- Feature dimensionality changes
- Graph size (node count) changes
- Graph density variations
"""

import numpy as np
import torch
import copy
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader
from .synthetic_generators import create_sensitivity_dataset
from noisenoise import add_noise_to_dataset
from metrics import nnd
from model import FlexibleGNN
from supervised_functions import train, evaluate, infer_task_type


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    # Model parameters
    layer_type: str = "gin"
    hidden_dim: int = 100
    num_layers: int = 3
    batch_size: int = 256
    epochs: int = 25
    lr: float = 0.001

    # Evaluation parameters
    n_noise_levels: int = 11
    n_repeats: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_noise_sweep(
    dataset,
    config: SensitivityConfig,
    verbose: bool = False
) -> Dict:
    """
    Run a complete noise sweep evaluation on a dataset.

    Returns a result dict compatible with the nnd() function.
    """
    ts = np.linspace(0, 1, config.n_noise_levels)
    structure_performances = {}
    feature_performances = {}

    task_level, task_type = infer_task_type(dataset)

    # Split dataset
    n_samples = len(dataset)
    split_props = (0.7, 0.2, 0.1)
    split_ns = [int(prop * n_samples) for prop in split_props]
    train_dataset = dataset[:split_ns[0]]
    test_dataset = dataset[split_ns[0] + split_ns[1]:]

    device = torch.device(config.device)

    # Model kwargs
    gin_kwargs = {"eps": 0, "train_eps": True}
    gcn_kwargs = {"add_self_loops": True, "normalize": True}
    gat_kwargs = {"heads": 4, "concat": True, "negative_slope": 0.2, "dropout": 0.6}
    gps_kwargs = {"heads": 4, "attn_type": "multihead", "attn_kwargs": {"dropout": 0.5}}
    kwarg_lookup = {"gin": gin_kwargs, "gcn": gcn_kwargs, "gat": gat_kwargs, "gps": gps_kwargs}

    node_in_dim = dataset.num_node_features
    edge_in_dim = dataset.num_edge_features if hasattr(dataset, "num_edge_features") else 0
    num_classes = dataset[0].y.shape[-1] if len(dataset[0].y.shape) > 0 else 1

    iterator = tqdm(range(config.n_noise_levels), desc="Noise levels") if verbose else range(config.n_noise_levels)

    for ti in iterator:
        ti_perfs_structure = []
        ti_perfs_feature = []

        for _ in range(config.n_repeats):
            if ti == 0:
                # No noise baseline
                perf = _train_and_eval_single(
                    train_dataset, test_dataset, config, device,
                    kwarg_lookup, node_in_dim, edge_in_dim, num_classes,
                    task_level, task_type, t_structure=0, t_feature=0
                )
                ti_perfs_structure.append(perf)
                ti_perfs_feature.append(perf)
            else:
                # Structure noise
                perf_struct = _train_and_eval_single(
                    train_dataset, test_dataset, config, device,
                    kwarg_lookup, node_in_dim, edge_in_dim, num_classes,
                    task_level, task_type, t_structure=ts[ti], t_feature=0
                )
                ti_perfs_structure.append(perf_struct)

                # Feature noise
                perf_feat = _train_and_eval_single(
                    train_dataset, test_dataset, config, device,
                    kwarg_lookup, node_in_dim, edge_in_dim, num_classes,
                    task_level, task_type, t_structure=0, t_feature=ts[ti]
                )
                ti_perfs_feature.append(perf_feat)

        structure_performances[str(ts[ti])] = [str(s) for s in ti_perfs_structure]
        feature_performances[str(ts[ti])] = [str(f) for f in ti_perfs_feature]

    return {
        "structure": structure_performances,
        "feature": feature_performances,
        "task_type": task_type,
        "dataset": "sensitivity_test"
    }


def _train_and_eval_single(
    train_dataset, test_dataset, config, device,
    kwarg_lookup, node_in_dim, edge_in_dim, num_classes,
    task_level, task_type, t_structure, t_feature
) -> float:
    """Train and evaluate a single model with given noise levels."""

    # Apply noise
    noisy_train = add_noise_to_dataset(copy.deepcopy(train_dataset), t_structure, t_feature)
    noisy_test = add_noise_to_dataset(copy.deepcopy(test_dataset), t_structure, t_feature)

    train_loader = DataLoader(noisy_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(noisy_test, batch_size=config.batch_size, shuffle=False)

    model = FlexibleGNN(
        layer_type=config.layer_type,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_layers,
        task_type=task_level,
        model_kwargs=kwarg_lookup.get(config.layer_type, {}),
        pe_dim=0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for _ in range(config.epochs):
        train(model, optimizer, train_loader, device, task_type)

    return evaluate(model, test_loader, device, task_type)


def evaluate_feature_dimensionality(
    feature_dims: List[int],
    label_type: str,
    config: SensitivityConfig,
    num_samples: int = 1600,
    n_repeats: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Evaluate NNRD consistency across feature dimensionalities.

    Parameters:
        feature_dims: List of feature dimensions to test.
        label_type: "structure" or "feature".
        config: Evaluation configuration.
        num_samples: Samples per dataset.
        n_repeats: Repeats per dimension.
        verbose: Show progress.

    Returns:
        Dict with results per dimension.
    """
    results = {}

    for n_feat in tqdm(feature_dims, desc="Feature dims", disable=not verbose):
        dim_results = {"nnrd_scores": [], "result_dicts": []}

        for rep in range(n_repeats):
            dataset = create_sensitivity_dataset(
                label_type=label_type,
                n_features=n_feat,
                num_samples=num_samples,
                root=f"data/sensitivity/feat_dim_{n_feat}_rep{rep}"
            )

            result_dict = run_noise_sweep(dataset, config, verbose=False)
            nnrd = nnd(result_dict)

            dim_results["nnrd_scores"].append(nnrd)
            dim_results["result_dicts"].append(result_dict)

        results[n_feat] = {
            "nnrd_mean": np.mean(dim_results["nnrd_scores"]),
            "nnrd_std": np.std(dim_results["nnrd_scores"]),
            "nnrd_scores": dim_results["nnrd_scores"],
            "n_features": n_feat
        }

    return results


def evaluate_graph_size(
    size_ranges: List[Tuple[int, int]],
    label_type: str,
    config: SensitivityConfig,
    num_samples: int = 1600,
    n_repeats: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Evaluate NNRD consistency across graph sizes.

    Parameters:
        size_ranges: List of (min_nodes, max_nodes) tuples.
        label_type: "structure" or "feature".
        config: Evaluation configuration.
        num_samples: Samples per dataset.
        n_repeats: Repeats per size range.
        verbose: Show progress.

    Returns:
        Dict with results per size range.
    """
    results = {}

    for min_n, max_n in tqdm(size_ranges, desc="Graph sizes", disable=not verbose):
        size_key = f"{min_n}-{max_n}"
        size_results = {"nnrd_scores": [], "result_dicts": []}

        for rep in range(n_repeats):
            dataset = create_sensitivity_dataset(
                label_type=label_type,
                min_nodes=min_n,
                max_nodes=max_n,
                num_samples=num_samples,
                root=f"data/sensitivity/size_{min_n}_{max_n}_rep{rep}"
            )

            result_dict = run_noise_sweep(dataset, config, verbose=False)
            nnrd = nnd(result_dict)

            size_results["nnrd_scores"].append(nnrd)
            size_results["result_dicts"].append(result_dict)

        results[size_key] = {
            "nnrd_mean": np.mean(size_results["nnrd_scores"]),
            "nnrd_std": np.std(size_results["nnrd_scores"]),
            "nnrd_scores": size_results["nnrd_scores"],
            "min_nodes": min_n,
            "max_nodes": max_n
        }

    return results


def evaluate_graph_density(
    densities: List[float],
    label_type: str,
    config: SensitivityConfig,
    num_samples: int = 1600,
    n_repeats: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Evaluate NNRD consistency across graph densities.

    Parameters:
        densities: List of density values (0 to 1).
        label_type: "structure" or "feature".
        config: Evaluation configuration.
        num_samples: Samples per dataset.
        n_repeats: Repeats per density.
        verbose: Show progress.

    Returns:
        Dict with results per density.
    """
    results = {}

    for density in tqdm(densities, desc="Densities", disable=not verbose):
        density_results = {"nnrd_scores": [], "result_dicts": []}

        for rep in range(n_repeats):
            dataset = create_sensitivity_dataset(
                label_type=label_type,
                density=density,
                num_samples=num_samples,
                root=f"data/sensitivity/density_{density:.2f}_rep{rep}"
            )

            result_dict = run_noise_sweep(dataset, config, verbose=False)
            nnrd = nnd(result_dict)

            density_results["nnrd_scores"].append(nnrd)
            density_results["result_dicts"].append(result_dict)

        results[density] = {
            "nnrd_mean": np.mean(density_results["nnrd_scores"]),
            "nnrd_std": np.std(density_results["nnrd_scores"]),
            "nnrd_scores": density_results["nnrd_scores"],
            "density": density
        }

    return results


def save_sensitivity_results(results: Dict, output_path: str):
    """Save sensitivity results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_sensitivity_results(input_path: str) -> Dict:
    """Load sensitivity results from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)
