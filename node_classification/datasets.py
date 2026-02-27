"""
Synthetic node classification datasets with degree-controlled graph construction.

Four scenarios:
  easy      - both x_i=y_i and degree encode the label
  feature   - only x_i=y_i encodes the label; degrees are random from {1,4}
  structure - only degree encodes the label; x_i is random from {0,1}
  coupled   - label = XOR(a,b); x_i = a; degree determined by b (1 if b=0, 4 if b=1)

The dataset is stored as a plain list of Data objects in a single .pt file.
On first use it is generated and saved; every subsequent load is a fast torch.load.
Slicing returns a plain list, which makes copy.deepcopy (used during noise
application) much faster than InMemoryDataset slices.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

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
        edge_list = [[0, 1], [1, 0]]

    return Data(
        edge_index=torch.tensor(edge_list, dtype=torch.long).T,
        num_nodes=n,
    )


def _fix_degree_sum(degrees: list) -> list:
    """
    Ensure the sum of degrees is even (handshaking lemma).
    If odd, flip one randomly-chosen node's degree between 1 and 4.
    """
    if sum(degrees) % 2 != 0:
        idx = np.random.randint(len(degrees))
        degrees[idx] = 4 if degrees[idx] == 1 else 1
    return degrees


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NodeClassificationDataset:
    """
    Synthetic node classification dataset backed by a plain list of Data objects.

    On first construction the graphs are generated and saved to
    ``{root}/{scenario}_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt``.
    Subsequent instantiations with the same parameters skip generation and
    load directly from that file (fast torch.load).

    Slicing returns a plain Python list, which makes copy.deepcopy cheap
    during noise application.

    Parameters
    ----------
    root : str
        Directory for caching.  Created if it doesn't exist.
    scenario : str
        One of 'easy', 'feature', 'structure', 'coupled'.
    num_samples : int
        Number of graphs to generate.
    min_nodes, max_nodes : int
        Range of nodes per graph (sampled uniformly, rounded to even).
    """

    SCENARIOS = ("easy", "feature", "structure", "coupled")

    def __init__(self, root="data/node_classification", scenario="easy",
                 num_samples=2000, min_nodes=20, max_nodes=100):
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"scenario must be one of {self.SCENARIOS}, got '{scenario}'"
            )
        self.scenario = scenario
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.task_type = "classification"
        self.num_node_features = 1
        self.num_edge_features = 1

        os.makedirs(root, exist_ok=True)
        fname = f"{scenario}_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt"
        self._cache_path = os.path.join(root, fname)

        if os.path.exists(self._cache_path):
            self._data = torch.load(self._cache_path, weights_only=False)
        else:
            self._data = self._generate()
            torch.save(self._data, self._cache_path)

    # ------------------------------------------------------------------
    # Sequence protocol — slicing returns a plain list (cheap to deepcopy)
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _assign_node_attributes(self, num_nodes: int):
        """
        Return (x, y, degrees) arrays for each node.

        Scenario rules
        --------------
        easy:      y_i ~ Bern(0.5);  x_i = y_i;            degree = 1 if y=0 else 4
        feature:   y_i ~ Bern(0.5);  x_i = y_i;            degree ~ Uniform({1,4})
        structure: y_i ~ Bern(0.5);  x_i ~ Bern(0.5);      degree = 1 if y=0 else 4
        coupled:   a,b ~ Bern(0.5);  y = 1[a==b];  x = a;  degree = 1 if b=0 else 4
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

    def _generate(self) -> list:
        desc = f"Generating {self.scenario} (n={self.num_samples})"
        data_list = []
        for _ in tqdm(range(self.num_samples), desc=desc):
            num_nodes = (
                np.random.randint(self.min_nodes // 2, self.max_nodes // 2 + 1) * 2
            )
            x, y, degrees = self._assign_node_attributes(num_nodes)

            data = build_degree_sequence_graph(degrees)
            num_edges = data.edge_index.shape[1]

            data.x = torch.tensor(x, dtype=torch.float).reshape(-1, 1)
            data.y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float)

            data_list.append(data)
        return data_list

    def __repr__(self):
        return (
            f"NodeClassificationDataset(scenario={self.scenario}, "
            f"n={self.num_samples}, nodes={self.min_nodes}-{self.max_nodes}, "
            f"cached={os.path.exists(self._cache_path)})"
        )
