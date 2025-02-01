import torch

import random

def add_structure_noise_degree_preserving(data, t):
    """
    Add structure noise via degree-preserving double edge swaps while moving
    the corresponding edge attributes along with each edge.

    Parameters:
    - data: PyTorch Geometric data object with attributes:
          - edge_index: LongTensor of shape [2, E] containing unique edges 
            (each undirected edge appears once as (u,v) with u < v).
          - (optionally) edge_attr: Tensor of edge attributes.
          - num_nodes: Number of nodes in the graph.
    - t: Noise level (float in [0, 1]). The number of attempted swaps is
         proportional to t * (number of unique edges).

    Returns:
    - data: The data object with a rewired edge_index (and updated edge_attr).
    """

    # Extract existing edges and attributes (if present)
    edge_index = data.edge_index
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    num_nodes = data.num_nodes

    # Build a list of unique edges (as (u,v) with u < v) and a parallel list of attributes.
    edges = []
    edge_attr_list = [] if edge_attr is not None else None

    num_edges = edge_index.size(1)
    for i in range(num_edges):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        # Ensure a canonical ordering (u < v)
        if u > v:
            u, v = v, u
        edges.append((u, v))
        if edge_attr is not None:
            edge_attr_list.append(edge_attr[i])

    # Determine the number of swaps to attempt.
    num_swaps = int(t * len(edges))
    swaps_done = 0
    attempts = 0
    max_attempts = num_swaps * 10  # safeguard against infinite loops

    while swaps_done < num_swaps and attempts < max_attempts:
        attempts += 1
        # Randomly select two distinct edges.
        idx1, idx2 = random.sample(range(len(edges)), 2)
        u, v = edges[idx1]
        w, x = edges[idx2]

        # To preserve degrees, ensure the four nodes are distinct.
        if len({u, v, w, x}) != 4:
            continue

        # Option 1: new edges (u, w) and (v, x)
        new_edge1_opt1 = (min(u, w), max(u, w))
        new_edge2_opt1 = (min(v, x), max(v, x))
        if new_edge1_opt1 in edges or new_edge2_opt1 in edges:
            # Option 2: try (u, x) and (v, w)
            new_edge1_opt2 = (min(u, x), max(u, x))
            new_edge2_opt2 = (min(v, w), max(v, w))
            if new_edge1_opt2 in edges or new_edge2_opt2 in edges:
                continue  # both options create duplicate edges; skip this swap.
            else:
                new_edge1, new_edge2 = new_edge1_opt2, new_edge2_opt2
        else:
            new_edge1, new_edge2 = new_edge1_opt1, new_edge2_opt1

        # Valid swap found: perform the swap.
        # The edge attribute associated with the edge stays with it.
        edges[idx1] = new_edge1  # carries over the original attribute at idx1
        edges[idx2] = new_edge2  # carries over the original attribute at idx2
        swaps_done += 1

    # Rebuild the full (symmetric) edge_index.
    # For each unique undirected edge, add both (u,v) and (v,u).
    row, col = [], []
    new_edge_attr_list = [] if edge_attr is not None else None

    for i, (u, v) in enumerate(edges):
        row.extend([u, v])
        col.extend([v, u])
        if edge_attr is not None:
            # Duplicate the attribute so that both (u,v) and (v,u) have the same attribute.
            attr = edge_attr_list[i]
            new_edge_attr_list.extend([attr, attr])

    new_edge_index = torch.tensor([row, col], dtype=torch.long)
    data.edge_index = new_edge_index

    if edge_attr is not None:
        # Stack the list of edge attributes into a tensor.
        data.edge_attr = torch.stack(new_edge_attr_list, dim=0)

    return data

def add_structure_noise(data, t):
    """
    Add edge noise in a diffusion-like process: edges are randomly dropped and added based on the diffusion time step.
    
    Parameters:
    - data: PyTorch Geometric data object containing edge_index, edge_attr, and num_nodes.
    - t: Current time step in the diffusion process, normalized by total steps, i.e., range [0, 1].
    
    Returns:
    - Noisy data object with modified edge_index and updated edge_attr.
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_nodes = data.num_nodes
    
    # Scale probabilities based on the diffusion time step
    drop_prob = t
    add_prob = t

    # Original edges
    row, col = edge_index

    # Randomly drop edges
    mask = torch.rand(row.size(0)) > drop_prob
    kept_row, kept_col = row[mask], col[mask]
    removed_row, removed_col = row[~mask], col[~mask]

    # Retain attributes for kept edges
    kept_edge_attr = edge_attr[mask]
    
    # Save attributes of removed edges
    removed_edge_attr = edge_attr[~mask]

    # Number of new edges to add equals the number of removed edges
    num_new_edges = removed_row.size(0)

    # Generate new edges
    new_row = torch.randint(0, num_nodes, (num_new_edges,))
    new_col = torch.randint(0, num_nodes, (num_new_edges,))

    # Use attributes of removed edges for new edges
    new_edge_attr = removed_edge_attr

    # Combine the original kept edges with new edges
    row = torch.cat([kept_row, new_row], dim=0)
    col = torch.cat([kept_col, new_col], dim=0)
    edge_attr = torch.cat([kept_edge_attr, new_edge_attr], dim=0)

    # Update edge_index and edge_attr in the data object
    data.edge_index = torch.stack([row, col], dim=0)
    data.edge_attr = edge_attr
    
    return data
