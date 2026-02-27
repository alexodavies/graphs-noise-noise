"""
visualise_examples.py

Publication-quality 4×4 figure.

  Rows  (top → bottom): easy  |  feature  |  structure  |  coupled
  Cols:
    0  Node-level — same graph, nodes coloured by attribute x
    1  Node-level — same graph, nodes coloured by label y
    2  Graph-level class 0 (ER)     — nodes coloured by attribute
    3  Graph-level class 1 (Ladder) — nodes coloured by attribute

  Graph-level attribute assignment per scenario:
    easy      — ER and Ladder have DIFFERENT attributes (both signals align)
    feature   — ER and Ladder have DIFFERENT attributes (feature encodes class)
    structure — ER and Ladder have the SAME attribute  (feature uninformative)
    coupled   — ER and Ladder have DIFFERENT attributes (complex coupling)

Run:
  python visualise_examples.py

Saves:
  figures/dataset_examples.png  (300 dpi)
  figures/dataset_examples.pdf
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
from torch_geometric.utils import to_networkx

from node_classification.datasets import build_triangle_and_cycle_graph
from synthetic_datasets import generate_circular_ladder_graph

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Class / label palette  (blue / red)
C0 = "#4878CF"
C1 = "#D65F5F"

# Feature / attribute palette  (yellow / purple)
# Deliberately distinct from the blue/red label palette to avoid confusion.
# Corresponds to x=0.5 (feature-state 0) and x=0.25 (feature-state 1)
FA = "#F0C040"   # yellow → feature state 0  (x = 0.5)
FB = "#9B59B6"   # purple → feature state 1  (x = 0.25)

_FEAT = {0: 0.5, 1: 0.25}          # matches datasets.py
_FEAT_COL = {0.5: FA, 0.25: FB}    # x-value → colour

EDGE_COL = "#aaaaaa"
NODE_SZ  = 40

SCENARIOS  = ["easy", "feature", "structure", "coupled"]
ROW_LABELS = ["Easy", "Feature", "Structure", "Coupled"]
COL_TITLES = ["Node: attribute  x", "Node: label  y",
              "Graph: class 0", "Graph: class 1"]

# Graph-level attribute assignment (FA/FB) per scenario:
#   each entry = (er_colour, ladder_colour)
#   "different" scenarios → ER and Ladder get opposite colours
#   "structure" scenario  → both get same colour (feature uninformative)
_GRAPH_ATTR = {
    "easy":      (FA, FB),   # features differ; both signals encode class
    "feature":   (FA, FB),   # features differ; only feature encodes class
    "structure": (FA, FA),   # features same  ; only structure encodes class
    "coupled":   (FB, FA),   # features differ; neither alone encodes class
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _balanced_shuffle(n: int, rng) -> np.ndarray:
    arr = np.array([0] * (n // 2) + [1] * (n // 2))
    rng.shuffle(arr)
    return arr


def _assign(scenario: str, n: int, rng) -> tuple:
    """Return (y, in_triangle, x_vals) arrays for `n` nodes."""
    if scenario == "easy":
        y           = _balanced_shuffle(n, rng)
        in_triangle = y.copy()
        x_vals      = np.array([_FEAT[yi] for yi in y])
    elif scenario == "feature":
        y           = _balanced_shuffle(n, rng)
        in_triangle = _balanced_shuffle(n, rng)
        x_vals      = np.array([_FEAT[yi] for yi in y])
    elif scenario == "structure":
        y           = _balanced_shuffle(n, rng)
        in_triangle = y.copy()
        x_signal    = _balanced_shuffle(n, rng)
        x_vals      = np.array([_FEAT[xi] for xi in x_signal])
    elif scenario == "coupled":
        # Explicitly balanced 4-quadrant construction so the visualisation
        # clearly shows that neither feature alone nor structure alone predicts y.
        # Each of the four (a, b) combinations gets exactly n/4 nodes:
        #   (a=0,b=0) → cycle,    orange feat, y=1  (red)
        #   (a=0,b=1) → triangle, orange feat, y=0  (blue)
        #   (a=1,b=0) → cycle,    teal feat,   y=0  (blue)
        #   (a=1,b=1) → triangle, teal feat,   y=1  (red)
        # Within triangles: teal→red, orange→blue.
        # Within cycle:     orange→red, teal→blue.  ← structure inverts the mapping
        assert n % 4 == 0, "n must be divisible by 4 for balanced coupled vis"
        q           = n // 4
        a           = np.array([0]*q + [0]*q + [1]*q + [1]*q)
        b           = np.array([0]*q + [1]*q + [0]*q + [1]*q)
        perm        = rng.permutation(n)
        a, b        = a[perm], b[perm]
        y           = (a == b).astype(int)
        x_vals      = np.array([_FEAT[ai] for ai in a])
        in_triangle = b.copy()
    return y, in_triangle, x_vals


def _node_layout(in_triangle: np.ndarray) -> dict:
    """Cycle nodes on inner ring; triangle clusters on outer ring."""
    n           = len(in_triangle)
    cycle_nodes = [i for i in range(n) if in_triangle[i] == 0]
    tri_nodes   = [i for i in range(n) if in_triangle[i] == 1]
    pos = {}

    r_cycle = 1.0
    for k, node in enumerate(cycle_nodes):
        angle = 2 * np.pi * k / len(cycle_nodes)
        pos[node] = np.array([r_cycle * np.cos(angle), r_cycle * np.sin(angle)])

    n_cliques = len(tri_nodes) // 3
    r_tri = 1.6
    for c in range(n_cliques):
        angle_c = 2 * np.pi * (c + 0.5) / n_cliques
        cx, cy  = r_tri * np.cos(angle_c), r_tri * np.sin(angle_c)
        for j in range(3):
            la = 2 * np.pi * j / 3
            d  = 0.17
            pos[tri_nodes[3 * c + j]] = np.array([cx + d * np.cos(la),
                                                   cy + d * np.sin(la)])
    return pos


def _make_node_example(scenario: str, n: int, seed: int):
    """
    Build one node-level graph.
    Returns (G, pos, feat_colors, label_colors) — both colour lists share the
    same graph instance so the two columns are directly comparable.
    """
    rng              = np.random.default_rng(seed)
    y, in_tri, x_vals = _assign(scenario, n, rng)
    data             = build_triangle_and_cycle_graph(in_tri)
    G                = to_networkx(data, to_undirected=True)
    pos              = _node_layout(in_tri)

    feat_colors  = [_FEAT_COL[x_vals[nd]] for nd in range(n)]
    label_colors = [C1 if y[nd] == 1 else C0 for nd in range(n)]
    return G, pos, feat_colors, label_colors


def _make_graph_examples(n_rungs: int, seed: int, scenario: str):
    """
    Build (G_er, pos_er, er_col, G_ladder, pos_ladder, lad_col).
    Node colours reflect per-scenario attribute assignment.
    """
    rng = np.random.default_rng(seed)

    # Circular ladder
    data_ladder = generate_circular_ladder_graph(num_edges=2 * n_rungs)
    G_ladder    = to_networkx(data_ladder, to_undirected=True)
    pos_ladder  = {}
    for i in range(n_rungs):
        angle = 2 * np.pi * i / n_rungs
        pos_ladder[i]           = np.array([0.55 * np.cos(angle), 0.55 * np.sin(angle)])
        pos_ladder[i + n_rungs] = np.array([1.0  * np.cos(angle), 1.0  * np.sin(angle)])

    # ER with matched density
    n_er    = 2 * n_rungs
    n_e     = G_ladder.number_of_edges()
    p_er    = 2 * n_e / (n_er * (n_er - 1))
    er_seed = int(rng.integers(0, 9999))
    G_er    = nx.erdos_renyi_graph(n_er, p_er, seed=er_seed)
    if not nx.is_connected(G_er):
        for u, v in nx.minimum_spanning_edges(nx.complete_graph(n_er), data=False):
            if not G_er.has_edge(u, v):
                G_er.add_edge(u, v)
    pos_er = nx.kamada_kawai_layout(G_er)

    er_col, lad_col = _GRAPH_ATTR[scenario]
    er_colors  = [er_col]  * G_er.number_of_nodes()
    lad_colors = [lad_col] * G_ladder.number_of_nodes()

    return G_er, pos_er, er_colors, G_ladder, pos_ladder, lad_colors


# ── Draw ──────────────────────────────────────────────────────────────────────

def _draw_panel(ax, G, pos, colors):
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=EDGE_COL, width=0.8, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors,
                           node_size=NODE_SZ,
                           linewidths=0.4,
                           edgecolors="white")
    ax.set_axis_off()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N       = 24   # nodes per node-level graph (multiple of 6)
    N_RUNGS = 8    # rungs per ladder → 16-node graph-level examples

    node_data  = {}
    graph_data = {}
    for row, scenario in enumerate(SCENARIOS):
        node_data[scenario]  = _make_node_example(scenario, N, seed=row * 10 + 1)
        graph_data[scenario] = _make_graph_examples(N_RUNGS, seed=row * 7 + 3,
                                                    scenario=scenario)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.0))
    fig.subplots_adjust(left=0.10, right=0.99,
                        top=0.90, bottom=0.06,
                        hspace=0.08, wspace=0.05)

    for row, scenario in enumerate(SCENARIOS):
        G, pos, feat_col, lbl_col = node_data[scenario]
        G_er, pos_er, er_col, G_lad, pos_lad, lad_col = graph_data[scenario]

        _draw_panel(axes[row, 0], G,     pos,     feat_col)
        _draw_panel(axes[row, 1], G,     pos,     lbl_col)
        _draw_panel(axes[row, 2], G_er,  pos_er,  er_col)
        _draw_panel(axes[row, 3], G_lad, pos_lad, lad_col)

    # ── Column titles ──────────────────────────────────────────────────────────
    for col, title in enumerate(COL_TITLES):
        axes[0, col].set_title(title, fontsize=7.5, pad=4)

    # ── Row labels ────────────────────────────────────────────────────────────
    fig.canvas.draw()
    for row, label in enumerate(ROW_LABELS):
        bbox  = axes[row, 0].get_position()
        y_mid = (bbox.y0 + bbox.y1) / 2
        fig.text(0.005, y_mid, label, ha="left", va="center",
                 fontsize=8, fontweight="bold", rotation=90)

    # ── Section group headers ─────────────────────────────────────────────────
    def _mid_x(a, b):
        return (a.get_position().x0 + b.get_position().x1) / 2

    x_node  = _mid_x(axes[0, 0], axes[0, 1])
    x_graph = _mid_x(axes[0, 2], axes[0, 3])
    y_hdr   = axes[0, 0].get_position().y1 + 0.025

    fig.text(x_node,  y_hdr, "Node classification",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.text(x_graph, y_hdr, "Graph classification",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Vertical separator
    x_sep = (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0) / 2
    sep = mlines.Line2D([x_sep, x_sep],
                        [axes[3, 0].get_position().y0,
                         axes[0, 0].get_position().y1 + 0.02],
                        transform=fig.transFigure,
                        color="#cccccc", linewidth=0.8, linestyle="--")
    fig.add_artist(sep)

    # ── Legend — two rows: class encoding and feature encoding ────────────────
    class_handles = [
        mpatches.Patch(facecolor=C0, edgecolor="none", label="Class 0"),
        mpatches.Patch(facecolor=C1, edgecolor="none", label="Class 1"),
    ]
    feat_handles = [
        mpatches.Patch(facecolor=FA, edgecolor="none", label=r"Attr. state 0  ($x{=}0.5$)"),
        mpatches.Patch(facecolor=FB, edgecolor="none", label=r"Attr. state 1  ($x{=}0.25$)"),
    ]
    # Place class legend under node columns, feature legend under graph columns
    node_x  = _mid_x(axes[0, 0], axes[0, 1])
    graph_x = _mid_x(axes[0, 2], axes[0, 3])
    y_leg   = axes[3, 0].get_position().y0 - 0.045

    fig.legend(handles=class_handles, loc="lower left",
               ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(node_x - 0.13, y_leg),
               bbox_transform=fig.transFigure,
               handlelength=1.1, columnspacing=0.8)
    fig.legend(handles=feat_handles, loc="lower left",
               ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(graph_x - 0.16, y_leg),
               bbox_transform=fig.transFigure,
               handlelength=1.1, columnspacing=0.8)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/dataset_examples.png", dpi=300, bbox_inches="tight")
    fig.savefig("figures/dataset_examples.pdf", bbox_inches="tight")
    print("Saved figures/dataset_examples.png and figures/dataset_examples.pdf")


if __name__ == "__main__":
    main()
