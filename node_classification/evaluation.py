"""
Noise-sweep evaluation for node classification experiments.

Mirrors the pattern in sensitivity/sensitivity_evaluation.py but uses
NodeClassificationDataset and FlexibleGNN with task_type='node'.
"""

import copy
import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from node_classification.datasets import NodeClassificationDataset
from noisenoise import add_noise_to_dataset
from model import FlexibleGNN
from supervised_functions import train, evaluate


# Layer-specific kwargs (mirrors sensitivity_evaluation.py)
_GNN_KWARGS = {
    "gin": {"eps": 0, "train_eps": True},
    "gcn": {"add_self_loops": True, "normalize": True},
    "gat": {"heads": 4, "concat": True, "negative_slope": 0.2, "dropout": 0.6},
    "gps": {"heads": 4, "attn_type": "multihead", "attn_kwargs": {"dropout": 0.5}},
}


@dataclass
class NodeClassificationConfig:
    """Hyperparameters for the node classification noise-sweep experiment."""
    layer_type: str = "gin"
    hidden_dim: int = 64
    num_layers: int = 3
    batch_size: int = 64
    epochs: int = 30
    lr: float = 0.001
    n_noise_levels: int = 11
    n_repeats: int = 5
    num_samples: int = 2000
    min_nodes: int = 20
    max_nodes: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_node_noise_sweep(
    scenario: str,
    config: NodeClassificationConfig,
    verbose: bool = True,
) -> Dict:
    """
    Build a NodeClassificationDataset for `scenario` and run a full
    structure-noise / feature-noise sweep.

    Protocol: train once on clean train data, then evaluate the frozen model
    across noise levels applied only to the test set (n_repeats independent
    noise draws per level).

    Returns a result dict compatible with nnd() and plot_results():
      {
        "dataset":    "node-<scenario>",
        "structure":  {str(t): [str(perf), ...]},
        "feature":    {str(t): [str(perf), ...]},
        "task_type":  "classification",
        "linear":     False,
        "layer":      config.layer_type,
        "pos":        False,
      }
    """
    dataset = NodeClassificationDataset(
        root=f"data/node_classification/{scenario}",
        scenario=scenario,
        num_samples=config.num_samples,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
    )

    n = len(dataset)
    train_n = int(0.7 * n)
    val_n = int(0.2 * n)
    train_dataset = dataset[:train_n]
    test_dataset = dataset[train_n + val_n:]

    device = torch.device(config.device)

    # --- Train once on clean data ---
    if verbose:
        print(f"[{scenario}] Training on clean data...")
    model = _build_and_train_model(train_dataset, config, device)

    ts = np.linspace(0, 1, config.n_noise_levels)

    structure_performances: Dict[str, list] = {}
    feature_performances: Dict[str, list] = {}

    noise_iter = (
        tqdm(range(config.n_noise_levels), desc=f"Noise sweep [{scenario}]")
        if verbose
        else range(config.n_noise_levels)
    )

    for ti in noise_iter:
        ti_struct = []
        ti_feat = []

        for _ in range(config.n_repeats):
            if ti == 0:
                # No-noise baseline: same evaluation counts for both curves
                perf = _eval_with_noise(model, test_dataset, config, device, 0.0, 0.0)
                ti_struct.append(perf)
                ti_feat.append(perf)
            else:
                ti_struct.append(
                    _eval_with_noise(model, test_dataset, config, device, ts[ti], 0.0)
                )
                ti_feat.append(
                    _eval_with_noise(model, test_dataset, config, device, 0.0, ts[ti])
                )

        structure_performances[str(ts[ti])] = [str(s) for s in ti_struct]
        feature_performances[str(ts[ti])] = [str(f) for f in ti_feat]

    return {
        "dataset": f"node-{scenario}",
        "structure": structure_performances,
        "feature": feature_performances,
        "task_type": "classification",
        "linear": False,
        "layer": config.layer_type,
        "pos": False,
    }


def _build_and_train_model(
    train_dataset,
    config: NodeClassificationConfig,
    device: torch.device,
) -> "FlexibleGNN":
    """Build and train a fresh model on clean train data. Returns the trained model."""
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = FlexibleGNN(
        layer_type=config.layer_type,
        node_in_dim=1,
        edge_in_dim=1,
        hidden_dim=config.hidden_dim,
        num_classes=1,
        num_layers=config.num_layers,
        task_type="node",
        model_kwargs=_GNN_KWARGS.get(config.layer_type, {}),
        pe_dim=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for _ in range(config.epochs):
        train(model, optimizer, train_loader, device, "classification")

    return model


def _eval_with_noise(
    model,
    test_dataset,
    config: NodeClassificationConfig,
    device: torch.device,
    t_structure: float,
    t_feature: float,
) -> float:
    """Apply noise to a fresh copy of the test set and evaluate the frozen model."""
    noisy_test = add_noise_to_dataset(
        copy.deepcopy(test_dataset), t_structure, t_feature
    )
    test_loader = DataLoader(noisy_test, batch_size=config.batch_size, shuffle=False)
    return evaluate(model, test_loader, device, "classification")
