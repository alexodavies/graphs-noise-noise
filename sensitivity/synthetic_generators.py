"""
Configurable synthetic graph generators for sensitivity analysis.

This module provides parameterized generators for creating synthetic datasets
with controlled:
- Feature dimensionality
- Graph size (node count)
- Graph density
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import erdos_renyi_graph
from tqdm import tqdm
from random import random
import os


def generate_circular_ladder_graph(num_nodes: int) -> Data:
    """
    Generates a circular ladder graph with specified node count.

    Parameters:
        num_nodes (int): Number of nodes (must be even, minimum 4).

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    if num_nodes < 4 or num_nodes % 2 != 0:
        raise ValueError("Circular ladder requires even num_nodes >= 4")

    num_rungs = num_nodes // 2
    edges = []

    for i in range(num_rungs):
        top = i
        bottom = i + num_rungs

        # Vertical rungs
        edges.append((top, bottom))
        # Horizontal rails (top and bottom rings)
        edges.append((top, (i + 1) % num_rungs))
        edges.append((bottom, ((i + 1) % num_rungs) + num_rungs))

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def generate_erdos_renyi_graph(num_nodes: int, density: float) -> Data:
    """
    Generates an Erdős-Rényi random graph with specified density.

    Parameters:
        num_nodes (int): Number of nodes.
        density (float): Edge probability (0 to 1).

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=density)
    return Data(edge_index=edge_index, num_nodes=num_nodes)


def generate_density_controlled_graph(num_nodes: int, target_density: float,
                                      base_structure: str = "ladder") -> Data:
    """
    Generates a graph with controlled density.

    For ladder base: starts with ladder structure, adds/removes edges to match density.
    For random base: uses Erdős-Rényi with target density.

    Parameters:
        num_nodes (int): Number of nodes.
        target_density (float): Target edge density (0 to 1).
        base_structure (str): "ladder" or "random".

    Returns:
        Data: A PyTorch Geometric Data object.
    """
    max_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = int(target_density * max_edges)

    if base_structure == "random":
        return generate_erdos_renyi_graph(num_nodes, target_density)

    # For ladder, we need to adjust
    if num_nodes % 2 != 0:
        num_nodes = num_nodes + 1

    data = generate_circular_ladder_graph(num_nodes)
    current_edges = data.edge_index.shape[1] // 2  # Undirected

    if target_edges > current_edges:
        # Add random edges
        existing = set()
        for i in range(data.edge_index.shape[1]):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            existing.add((min(u, v), max(u, v)))

        edges_to_add = target_edges - current_edges
        new_edges = []
        attempts = 0
        while len(new_edges) < edges_to_add and attempts < edges_to_add * 10:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and (min(u, v), max(u, v)) not in existing:
                new_edges.append([u, v])
                new_edges.append([v, u])
                existing.add((min(u, v), max(u, v)))
            attempts += 1

        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).T
            data.edge_index = torch.cat([data.edge_index, new_edge_tensor], dim=1)

    return data


def add_node_features(data: Data, n_features: int, mean: float = 0.0) -> Data:
    """Add constant-valued node features."""
    data.x = torch.ones(data.num_nodes, n_features) * mean
    return data


def add_edge_features(data: Data, n_features: int, mean: float = 0.0) -> Data:
    """Add constant-valued edge features."""
    n_edges = data.edge_index.shape[1]
    data.edge_attr = torch.ones(n_edges, n_features) * mean
    return data


def erdos_renyi_from_data(data: Data) -> Data:
    """
    Replace graph structure with Erdős-Rényi random graph
    preserving node/edge count.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1] // 2  # Undirected edge count

    edges = set()
    attempts = 0
    max_attempts = num_edges * 100

    while len(edges) < num_edges and attempts < max_attempts:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (min(u, v), max(u, v)) not in edges:
            edges.add((min(u, v), max(u, v)))
        attempts += 1

    edge_list = []
    for u, v in edges:
        edge_list.append([u, v])
        edge_list.append([v, u])

    new_edge_index = torch.tensor(edge_list, dtype=torch.long).T
    data.edge_index = new_edge_index
    return data


class ConfigurableSyntheticDataset(InMemoryDataset):
    """
    Synthetic dataset with configurable parameters for sensitivity analysis.

    Parameters:
        root (str): Root directory for data storage.
        label_type (str): "structure" or "feature" - determines label source.
        num_samples (int): Number of graphs to generate.
        n_features (int): Number of node/edge features.
        min_nodes (int): Minimum nodes per graph.
        max_nodes (int): Maximum nodes per graph.
        density (float or None): Target edge density. None uses default structure.
        transform: Optional transform.
        pre_transform: Optional pre-transform.
    """

    def __init__(self, root: str, label_type: str, num_samples: int = 3200,
                 n_features: int = 5, min_nodes: int = 48, max_nodes: int = 256,
                 density: float = None, transform=None, pre_transform=None):
        self.label_type = label_type
        self.num_samples = num_samples
        self.n_features = n_features
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.density = density

        # Create unique identifier for this configuration
        self._config_id = f"{label_type}_n{num_samples}_f{n_features}_nodes{min_nodes}-{max_nodes}"
        if density is not None:
            self._config_id += f"_d{density:.2f}"

        super().__init__(root, transform, pre_transform)

        os.makedirs(self.processed_dir, exist_ok=True)

        if not os.path.exists(self.processed_paths[0]):
            self.process()

        self.task_type = "classification"
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self._config_id}.pt']

    def process(self):
        self._generate_data()

    def _generate_data(self):
        data_list = []
        is_feature_label = self.label_type.endswith("feature")

        for _ in tqdm(range(self.num_samples), desc=f"Generating {self._config_id}"):
            # Random node count (must be even for ladder)
            num_nodes = np.random.randint(self.min_nodes // 2, self.max_nodes // 2) * 2

            # Generate labels
            is_random_structure = random() > 0.5
            structure_label = 1 if is_random_structure else 0
            feature_label = 1 if random() > 0.5 else 0

            # Generate base structure
            if self.density is not None:
                data = generate_density_controlled_graph(
                    num_nodes, self.density,
                    base_structure="random" if is_random_structure else "ladder"
                )
            else:
                data = generate_circular_ladder_graph(num_nodes)
                if is_random_structure:
                    data = erdos_renyi_from_data(data)

            # Add features
            feature_mean = 2 * (feature_label - 0.5)  # -1 or +1
            data = add_node_features(data, self.n_features, mean=feature_mean)
            data = add_edge_features(data, self.n_features, mean=feature_mean)

            # Set label
            label = feature_label if is_feature_label else structure_label
            data.y = torch.tensor([label], dtype=torch.long)

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return f"ConfigurableSyntheticDataset({self._config_id})"


def create_sensitivity_dataset(
    label_type: str,
    n_features: int = 5,
    min_nodes: int = 48,
    max_nodes: int = 256,
    density: float = None,
    num_samples: int = 3200,
    root: str = "data/sensitivity"
) -> ConfigurableSyntheticDataset:
    """
    Factory function to create sensitivity analysis datasets.

    Parameters:
        label_type (str): "structure" or "feature".
        n_features (int): Feature dimensionality.
        min_nodes (int): Minimum graph size.
        max_nodes (int): Maximum graph size.
        density (float): Edge density (None for default).
        num_samples (int): Number of samples.
        root (str): Data root directory.

    Returns:
        ConfigurableSyntheticDataset
    """
    return ConfigurableSyntheticDataset(
        root=root,
        label_type=label_type,
        num_samples=num_samples,
        n_features=n_features,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        density=density
    )
