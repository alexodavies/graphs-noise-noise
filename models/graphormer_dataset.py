import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Batch


def compute_shortest_path_distance(edge_index, num_nodes, max_distance=5):
    """Compute shortest path distance using Floyd-Warshall algorithm."""
    device = edge_index.device
    
    # Handle empty graphs
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), device=device).long()
    
    # Make graph undirected
    edge_index = to_undirected(edge_index)
    
    # Initialize distance matrix
    dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
    dist[edge_index[0], edge_index[1]] = 1.0
    dist.fill_diagonal_(0)
    
    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    
    # Clip distances to max_distance
    dist = torch.clamp(dist, 0, max_distance)
    
    return dist.long()


def collate_with_paths(data_list, max_distance=5):
    """
    Custom collate function that computes shortest paths during batching.
    
    Args:
        data_list: List of PyG Data objects
        max_distance: Maximum path distance to consider
        
    Returns:
        batch: Regular PyG Batch with added shortest_paths_list
    """
    # First, use standard PyG batching
    batch = Batch.from_data_list(data_list)
    print("In collate")
    
    # Compute shortest paths for each graph
    paths_list = []
    
    # Process each graph separately
    batch_idx = batch.batch
    num_graphs = int(batch_idx.max()) + 1 if batch_idx.numel() > 0 else 0
    
    for i in range(num_graphs):
        # Extract nodes for this graph
        mask = (batch_idx == i)
        node_indices = torch.where(mask)[0]
        
        if node_indices.numel() == 0:
            continue
            
        # Extract subgraph
        num_nodes = node_indices.size(0)
        
        # Find edges for this graph
        edge_mask = torch.isin(batch.edge_index[0], node_indices) & torch.isin(batch.edge_index[1], node_indices)
        sub_edge_index = batch.edge_index[:, edge_mask].clone()
        
        # Renumber nodes to be 0-indexed within this subgraph
        node_mapper = -torch.ones(batch.num_nodes, dtype=torch.long, device=batch.edge_index.device)
        node_mapper[node_indices] = torch.arange(num_nodes, device=batch.edge_index.device)
        sub_edge_index = node_mapper[sub_edge_index]
        
        # Compute shortest paths
        paths = compute_shortest_path_distance(sub_edge_index, num_nodes, max_distance)
        paths_list.append(paths)
    
    # Add paths list to batch
    # batch.shortest_paths_list = paths_list
    setattr(batch, 'shortest_paths_list', paths_list)
    
    return batch


def create_dataloader_with_paths(dataset, batch_size=32, shuffle=True, max_distance=5, **kwargs):
    """Create a dataloader that computes shortest paths during batching."""
    from torch_geometric.loader import DataLoader
    
    # Create custom collate function with the specified max_distance
    # collate_fn = lambda data_list: collate_with_paths(data_list, max_distance=max_distance)

     # Create custom collate function with the specified max_distance
    def custom_collate(data_list):
        print("Custom collate function is being called!")  # Debug print
        return collate_with_paths(data_list, max_distance=max_distance)

    
    # Create dataloader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        **kwargs
    )


    return loader