import torch

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
