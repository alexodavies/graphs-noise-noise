"""
Synthetic node classification datasets with degree-controlled graph construction.

Structural signal: node degree ∈ {2, 4}
Feature signal:    node feature x_i ∈ {0.5, 0.25}

Design property: x_i × degree_i = 0.5×2 = 0.25×4 = 1 for every class-matched
node. This means naive sum-aggregation cannot trivially combine both signals
into a constant, making neither channel a pure free-ride shortcut.

Four scenarios:
  easy      - x = 0.5/0.25 and degree = 2/4 both encode y_i
  feature   - x = 0.5/0.25 encodes y_i; degree random from {2, 4}
  structure - x random from {0.5, 0.25}; degree = 2/4 encodes y_i
  coupled   - a,b ~ Bern(0.5); y = 1[a==b]; x = 0.5 if a=0 else 0.25;
              degree = 2 if b=0 else 4

Note: with degrees {2, 4} the degree-sum is always even (both values are even),
so no parity-fixing step is required.

The dataset is stored as a plain list of Data objects in a single .pt file.
On first use it is generated and saved; every subsequent load is a fast
torch.load.  Slicing returns a plain list, making copy.deepcopy cheap during
noise application.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# Feature value assigned to each binary signal state
_FEAT = {0: 0.5, 1: 0.25}   # class-0 → 0.5, class-1 → 0.25
_DEG  = {0: 2,   1: 4}       # class-0 → degree 2, class-1 → degree 4


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_degree_sequence_graph(degrees: list) -> Data:
    """
    Build an undirected graph where node i has the specified degree.

    Uses a greedy configuration model: creates one 'stub' per degree unit,
    then repeatedly shuffles and pairs stubs into edges, rejecting self-loops
    and multi-edges. Up to 20 passes are made; any remaining unpaired stubs
    are silently dropped (rare for balanced degree-2/4 mixes).
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NodeClassificationDataset:
    """
    Synthetic node classification dataset backed by a plain list of Data objects.

    Each graph has nodes with:
      x         [num_nodes, 1]  float — 0.5 (class 0) or 0.25 (class 1) when signal present
      y         [num_nodes, 1]  float — binary label {0, 1}
      edge_attr [num_edges, 1]  float — dummy all-ones (for GIN/GPS compatibility)
      edge_index               — built to match target degrees {2, 4}

    On first construction the graphs are generated and saved to
    ``{root}/{scenario}_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt``.
    Subsequent instantiations with the same parameters load from that file.

    Parameters
    ----------
    root : str
        Directory for caching.  Created if it doesn't exist.
    scenario : str
        One of 'easy', 'feature', 'structure', 'coupled'.
    num_samples : int
        Number of graphs to generate.
    min_nodes, max_nodes : int
        Range of nodes per graph (sampled uniformly, always even).
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
        Return (x_vals, y, degrees) for each node under the chosen scenario.

        Signal encoding
        ---------------
        Feature signal: x = 0.5 if class-0, 0.25 if class-1
        Degree signal:  degree = 2 if class-0, 4 if class-1

        Scenarios
        ---------
        easy:      y ~ Bern(0.5);  x encodes y;  degree encodes y
        feature:   y ~ Bern(0.5);  x encodes y;  degree random from {2, 4}
        structure: y ~ Bern(0.5);  x random from {0.5, 0.25};  degree encodes y
        coupled:   a,b ~ Bern(0.5); y = 1[a==b]; x encodes a; degree encodes b
        """
        if self.scenario == "easy":
            y = np.random.randint(0, 2, size=num_nodes)
            x_vals = np.array([_FEAT[yi] for yi in y])
            degrees = [_DEG[yi] for yi in y]

        elif self.scenario == "feature":
            y = np.random.randint(0, 2, size=num_nodes)
            x_vals = np.array([_FEAT[yi] for yi in y])
            degrees = list(np.random.choice([2, 4], size=num_nodes))

        elif self.scenario == "structure":
            y = np.random.randint(0, 2, size=num_nodes)
            x_signal = np.random.randint(0, 2, size=num_nodes)   # random, indep. of y
            x_vals = np.array([_FEAT[xi] for xi in x_signal])
            degrees = [_DEG[yi] for yi in y]

        elif self.scenario == "coupled":
            a = np.random.randint(0, 2, size=num_nodes)
            b = np.random.randint(0, 2, size=num_nodes)
            y = (a == b).astype(int)
            x_vals = np.array([_FEAT[ai] for ai in a])
            degrees = [_DEG[bi] for bi in b]

        # Degrees {2, 4} are both even → sum is always even; no parity fix needed.
        return x_vals, y, degrees

    def _generate(self) -> list:
        data_list = []
        for _ in tqdm(range(self.num_samples),
                      desc=f"Generating {self.scenario} (n={self.num_samples})"):
            num_nodes = (
                np.random.randint(self.min_nodes // 2, self.max_nodes // 2 + 1) * 2
            )
            x_vals, y, degrees = self._assign_node_attributes(num_nodes)

            data = build_degree_sequence_graph(degrees)
            num_edges = data.edge_index.shape[1]

            data.x = torch.tensor(x_vals, dtype=torch.float).reshape(-1, 1)
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
