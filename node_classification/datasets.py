"""
Synthetic node classification datasets with degree-controlled graph construction.

Four scenarios:
  easy      - both x_i=y_i and degree encode the label
  feature   - only x_i=y_i encodes the label; degrees are random from {1,4}
  structure - only degree encodes the label; x_i is random from {0,1}
  coupled   - label = XOR(a,b); x_i = a; degree determined by b (1 if b=0, 4 if b=1)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def build_degree_sequence_graph(degrees: list) -> Data:
    """
    Build an undirected graph where node i has the specified degree.

    Uses a greedy configuration model: creates one 'stub' per degree unit,
    then repeatedly shuffles and pairs stubs into edges, rejecting self-loops
    and multi-edges. Up to 20 passes are made; any remaining unpaired stubs
    are silently dropped (in practice rare for balanced degree-1/4 mixes).
    """
    n = len(degrees)
    stubs = [node_id for node_id, d in enumerate(degrees) for _ in range(d)]

    edges: set = set()
    for _ in range(20):
        if len(stubs) < 2:
            break
        np.random.shuffle(stubs)
        remaining = []
        i = 0
        while i + 1 < len(stubs):
            u, v = stubs[i], stubs[i + 1]
            e = (min(u, v), max(u, v))
            if u != v and e not in edges:
                edges.add(e)
                i += 2
            else:
                remaining.append(stubs[i])
                i += 1
        if i < len(stubs):
            remaining.append(stubs[i])
        stubs = remaining

    edge_list = []
    for u, v in edges:
        edge_list.extend([[u, v], [v, u]])

    if not edge_list:
        # Degenerate fallback: at least one edge
        edge_list = [[0, 1], [1, 0]]

    return Data(
        edge_index=torch.tensor(edge_list, dtype=torch.long).T,
        num_nodes=n,
    )


def _fix_degree_sum(degrees: list) -> list:
    """
    Ensure the sum of degrees is even (required by the handshaking lemma).
    If the sum is odd, flip one randomly-chosen node's degree between 1 and 4.
    """
    if sum(degrees) % 2 != 0:
        idx = np.random.randint(len(degrees))
        degrees[idx] = 4 if degrees[idx] == 1 else 1
    return degrees


class NodeClassificationDataset(InMemoryDataset):
    """
    Synthetic node classification dataset.

    Each Data object is a single graph where every node carries:
      - x:         [num_nodes, 1] float tensor — binary feature in {0.0, 1.0}
      - y:         [num_nodes, 1] float tensor — binary label in {0.0, 1.0}
      - edge_attr: [num_edges, 1] float tensor — dummy (all ones, for GIN/GPS compat.)
      - edge_index: [2, num_edges] — undirected edges built to match target degrees

    Parameters
    ----------
    root : str
        Directory for caching processed data.
    scenario : str
        One of 'easy', 'feature', 'structure', 'coupled'.
    num_samples : int
        Number of graphs to generate.
    min_nodes, max_nodes : int
        Range for number of nodes per graph (will be rounded to nearest even number).
    """

    SCENARIOS = ("easy", "feature", "structure", "coupled")

    def __init__(self, root, scenario, num_samples=2000,
                 min_nodes=20, max_nodes=100, transform=None, pre_transform=None):
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"scenario must be one of {self.SCENARIOS}, got '{scenario}'"
            )
        self.scenario = scenario
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self._config_id = (
            f"{scenario}_n{num_samples}_nodes{min_nodes}-{max_nodes}"
        )
        super().__init__(root, transform, pre_transform)

        os.makedirs(self.processed_dir, exist_ok=True)
        if not os.path.exists(self.processed_paths[0]):
            self.process()

        self.task_type = "classification"
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{self._config_id}.pt"]

    def process(self):
        self._generate_data()

    def _assign_node_attributes(self, num_nodes: int):
        """
        Return (x, y, degrees) arrays for `num_nodes` nodes according to the scenario.

        Scenario rules
        --------------
        easy:      y_i ~ Bern(0.5);  x_i = y_i;   degree = 1 if y_i=0 else 4
        feature:   y_i ~ Bern(0.5);  x_i = y_i;   degree ~ Uniform({1, 4})
        structure: y_i ~ Bern(0.5);  x_i ~ Bern(0.5);  degree = 1 if y_i=0 else 4
        coupled:   a,b ~ Bern(0.5);  y_i = 1[a==b];  x_i = a;  degree = 1 if b=0 else 4
        """
        if self.scenario == "easy":
            y = np.random.randint(0, 2, size=num_nodes)
            x = y.copy()
            degrees = [1 if yi == 0 else 4 for yi in y]

        elif self.scenario == "feature":
            y = np.random.randint(0, 2, size=num_nodes)
            x = y.copy()
            degrees = list(np.random.choice([1, 4], size=num_nodes))

        elif self.scenario == "structure":
            y = np.random.randint(0, 2, size=num_nodes)
            x = np.random.randint(0, 2, size=num_nodes)
            degrees = [1 if yi == 0 else 4 for yi in y]

        elif self.scenario == "coupled":
            a = np.random.randint(0, 2, size=num_nodes)
            b = np.random.randint(0, 2, size=num_nodes)
            y = (a == b).astype(int)
            x = a.copy()
            degrees = [1 if bi == 0 else 4 for bi in b]

        degrees = _fix_degree_sum(list(degrees))
        return x, y, degrees

    def _generate_data(self):
        data_list = []
        for _ in tqdm(range(self.num_samples), desc=f"Generating {self._config_id}"):
            # Sample an even node count so degree sums stay manageable
            num_nodes = (
                np.random.randint(self.min_nodes // 2, self.max_nodes // 2 + 1) * 2
            )

            x, y, degrees = self._assign_node_attributes(num_nodes)

            data = build_degree_sequence_graph(degrees)
            num_edges = data.edge_index.shape[1]

            data.x = torch.tensor(x, dtype=torch.float).reshape(-1, 1)
            data.y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
            # Dummy edge features (all ones) for GIN/GPS compatibility
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float)

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return f"NodeClassificationDataset({self._config_id})"
