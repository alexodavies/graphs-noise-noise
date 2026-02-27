"""
Synthetic node classification datasets with triangle-based structural signal.

Structural signal: whether a node is part of a triangle (3-clique).
  Class 1 nodes → in a triangle
  Class 0 nodes → in a cycle (no triangles)

All nodes have degree exactly 2, so degree carries NO class information.

Feature signal: x_i ∈ {0.5, 0.25}  (same values used in all scenarios)
  0.5  encodes "signal state 0"
  0.25 encodes "signal state 1"

Four scenarios:
  easy      - x encodes y AND triangle membership encodes y
  feature   - x encodes y; triangle membership is random (independent of y)
  structure - x is random; triangle membership encodes y
  coupled   - a,b ~ Bern(0.5) balanced; y = 1[a==b];
              x = _FEAT[a]; triangle membership = (b==1)

Construction guarantees
-----------------------
* num_nodes is sampled as a multiple of 6:
    - n/2 nodes always go into triangles  (n/2 divisible by 3 → exact groups)
    - n/2 nodes always go into one cycle  (length n/2 ≥ 4 → no self-triangles)
* Classes (and structural assignments) are BALANCED: exactly n/2 per side.
* a and b in the "coupled" scenario are independent balanced shuffles
  of [0]*n/2 + [1]*n/2, guaranteeing sum(b==1)=n/2 divisible by 3.

Cache
-----
Stored as a plain list of Data objects at
  {root}/{scenario}_tri_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt
"_tri" distinguishes from older degree-based caches.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# Feature value for each binary signal state
_FEAT = {0: 0.5, 1: 0.25}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_triangle_and_cycle_graph(in_triangle: np.ndarray) -> Data:
    """
    Build a graph where:
      - Nodes with in_triangle[i]=1 are grouped into triangles (3-cliques).
      - Nodes with in_triangle[i]=0 form a single cycle.

    All nodes end up with degree exactly 2.

    Parameters
    ----------
    in_triangle : array of 0/1, length n
        1 → node goes into a triangle group; 0 → node goes into the cycle.
        Requires: sum(in_triangle) divisible by 3, sum(1-in_triangle) >= 4.
    """
    triangle_nodes = np.where(in_triangle == 1)[0]
    cycle_nodes    = np.where(in_triangle == 0)[0]

    n_t = len(triangle_nodes)
    n_c = len(cycle_nodes)

    assert n_t % 3 == 0,  f"Triangle node count must be divisible by 3, got {n_t}"
    assert n_c >= 4,       f"Cycle node count must be >= 4 to avoid self-triangles, got {n_c}"

    edges = []

    # Class-1 / triangle nodes: form k = n_t//3 triangles
    for k in range(n_t // 3):
        a, b, c = triangle_nodes[3*k], triangle_nodes[3*k+1], triangle_nodes[3*k+2]
        for u, v in [(a, b), (b, c), (a, c)]:
            edges.extend([[int(u), int(v)], [int(v), int(u)]])

    # Class-0 / cycle nodes: form one cycle of length n_c
    for k in range(n_c):
        u = cycle_nodes[k]
        v = cycle_nodes[(k + 1) % n_c]
        edges.extend([[int(u), int(v)], [int(v), int(u)]])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return Data(edge_index=edge_index, num_nodes=len(in_triangle))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NodeClassificationDataset:
    """
    Synthetic node classification dataset backed by a plain list of Data objects.

    Each graph:
      x         [num_nodes, 1]  float — 0.5 or 0.25
      y         [num_nodes, 1]  float — binary label {0.0, 1.0}
      edge_attr [num_edges, 1]  float — all-ones dummy (GIN/GPS compat.)
      edge_index               — triangle + cycle structure (all degrees = 2)

    Saves to / loads from:
      {root}/{scenario}_tri_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt

    Parameters
    ----------
    root        Directory for caching (created if needed).
    scenario    'easy' | 'feature' | 'structure' | 'coupled'
    num_samples Number of graphs to generate.
    min_nodes, max_nodes
                Sampling range for graph size. Actual num_nodes is the
                nearest multiple of 6 in [min_nodes, max_nodes].
    """

    SCENARIOS = ("easy", "feature", "structure", "coupled")

    def __init__(self, root="data/node_classification", scenario="easy",
                 num_samples=2000, min_nodes=20, max_nodes=100):
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"scenario must be one of {self.SCENARIOS}, got '{scenario}'"
            )
        self.scenario    = scenario
        self.num_samples = num_samples
        self.min_nodes   = min_nodes
        self.max_nodes   = max_nodes
        self.task_type          = "classification"
        self.num_node_features  = 1
        self.num_edge_features  = 1

        os.makedirs(root, exist_ok=True)
        fname = f"{scenario}_tri_n{num_samples}_nodes{min_nodes}-{max_nodes}.pt"
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
    # Generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _balanced_shuffle(n):
        """Return a randomly shuffled array of exactly n/2 zeros and n/2 ones."""
        arr = np.array([0] * (n // 2) + [1] * (n // 2))
        np.random.shuffle(arr)
        return arr

    def _assign_attributes(self, num_nodes: int):
        """
        Return (x_vals, y, in_triangle) for num_nodes nodes.

        in_triangle[i] = 1  → node i is placed in a triangle (class-1 structure)
        in_triangle[i] = 0  → node i is placed in the cycle  (class-0 structure)

        Scenario rules
        --------------
        easy:      y balanced-shuffled; x = _FEAT[y]; in_triangle = y
        feature:   y balanced-shuffled; x = _FEAT[y]; in_triangle random (indep. of y)
        structure: y balanced-shuffled; x = _FEAT[random, indep. of y]; in_triangle = y
        coupled:   a, b independent balanced-shuffles; y = 1[a==b];
                   x = _FEAT[a]; in_triangle = b
        """
        if self.scenario == "easy":
            y           = self._balanced_shuffle(num_nodes)
            in_triangle = y.copy()
            x_vals      = np.array([_FEAT[yi] for yi in y])

        elif self.scenario == "feature":
            y           = self._balanced_shuffle(num_nodes)
            in_triangle = self._balanced_shuffle(num_nodes)   # independent of y
            x_vals      = np.array([_FEAT[yi] for yi in y])

        elif self.scenario == "structure":
            y           = self._balanced_shuffle(num_nodes)
            in_triangle = y.copy()
            x_signal    = self._balanced_shuffle(num_nodes)   # independent of y
            x_vals      = np.array([_FEAT[xi] for xi in x_signal])

        elif self.scenario == "coupled":
            a           = self._balanced_shuffle(num_nodes)
            b           = self._balanced_shuffle(num_nodes)
            y           = (a == b).astype(int)
            x_vals      = np.array([_FEAT[ai] for ai in a])
            in_triangle = b.copy()

        return x_vals, y, in_triangle

    def _generate(self) -> list:
        # Smallest/largest multiple of 6 within [min_nodes, max_nodes]
        lo = (self.min_nodes + 5) // 6   # ceil(min_nodes / 6)
        hi = self.max_nodes // 6          # floor(max_nodes / 6)

        data_list = []
        for _ in tqdm(range(self.num_samples),
                      desc=f"Generating {self.scenario} (n={self.num_samples})"):
            num_nodes = np.random.randint(lo, hi + 1) * 6

            x_vals, y, in_triangle = self._assign_attributes(num_nodes)

            data = build_triangle_and_cycle_graph(in_triangle)
            num_edges = data.edge_index.shape[1]

            data.x         = torch.tensor(x_vals, dtype=torch.float).reshape(-1, 1)
            data.y         = torch.tensor(y,      dtype=torch.float).reshape(-1, 1)
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float)

            data_list.append(data)
        return data_list

    def __repr__(self):
        return (
            f"NodeClassificationDataset(scenario={self.scenario}, "
            f"n={self.num_samples}, nodes={self.min_nodes}-{self.max_nodes} "
            f"[multiples of 6], cached={os.path.exists(self._cache_path)})"
        )
