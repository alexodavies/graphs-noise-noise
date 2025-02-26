import torch
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import copy


def compute_shortest_path_distance(edge_index, num_nodes, max_distance=5):
    """
    Compute the shortest path distance between all pairs of nodes.
    
    Args:
        edge_index: Graph connectivity in COO format with shape [2, num_edges]
        num_nodes: Number of nodes in the graph
        max_distance: Maximum distance to consider (paths longer than this will be clipped)
        
    Returns:
        spatial_pos: Tensor of shape [num_nodes, num_nodes] containing shortest path distances
    """
    # Handle empty graphs
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    
    # Make sure the graph is undirected for distance calculation
    edge_index = to_undirected(edge_index)
    
    # Initialize distance matrix: 0 for self-loops, 1 for direct neighbors, inf for others
    dist = torch.full((num_nodes, num_nodes), float('inf'), device=edge_index.device)
    dist[edge_index[0], edge_index[1]] = 1.0
    
    # Set diagonal to 0 (distance to self is 0)
    dist.fill_diagonal_(0)
    
    # Floyd-Warshall algorithm for all-pairs shortest paths
    for k in range(num_nodes):
        # Update distances: dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        dist_ik = dist[:, k:k+1]  # Shape: [num_nodes, 1]
        dist_kj = dist[k:k+1, :]  # Shape: [1, num_nodes]
        dist = torch.minimum(dist, dist_ik + dist_kj)
    
    # Clip distances to max_distance
    dist = torch.clamp(dist, 0, max_distance)
    
    return dist.long()


def prepare_graphormer_batch(batch_data, max_distance=5):
    """
    Prepare a batch of variable-sized graphs for Graphormer processing.
    Attaches spatial positional encodings as a 'pe' attribute to the data object.
    
    Args:
        batch_data: A PyG Batch object or a list of PyG Data objects
        max_distance: Maximum path distance to consider
        
    Returns:
        batch: A modified PyG Batch object with additional attributes:
            - pe: Padded spatial position matrices [B, max_nodes, max_nodes]
            - attn_mask: Attention mask for padded areas [B, max_nodes, max_nodes]
            - graph_sizes: Original sizes of each graph in the batch
    """
    device = batch_data.x.device if isinstance(batch_data, (Data, Batch)) else batch_data[0].x.device
    
    # Convert list to batch if necessary
    if isinstance(batch_data, list):
        batch = Batch.from_data_list(batch_data)
    else:
        batch = copy.deepcopy(batch_data)  # Avoid modifying the original data
    
    # Get batch information
    if hasattr(batch, 'batch'):
        batch_idx = batch.batch
        num_graphs = int(batch_idx.max()) + 1
        graph_sizes = torch.bincount(batch_idx).tolist()
        max_nodes = max(graph_sizes)
        total_nodes = batch.x.size(0)
    else:
        # Handle single graph case
        num_graphs = 1
        graph_sizes = [batch.x.size(0)]
        max_nodes = graph_sizes[0]
        total_nodes = max_nodes
        batch_idx = torch.zeros(total_nodes, dtype=torch.long, device=device)
    
    # Initialize positional encoding and attention mask tensors
    pe = torch.zeros(num_graphs, max_nodes, max_nodes, dtype=torch.long, device=device)
    attn_mask = torch.ones(num_graphs, max_nodes, max_nodes, dtype=torch.bool, device=device)
    
    # Process each graph in the batch
    ptr = 0
    for i in range(num_graphs):
        # Get nodes for this graph
        if num_graphs > 1:
            mask = (batch_idx == i)
            nodes_idx = torch.where(mask)[0]
            n_i = len(nodes_idx)
        else:
            nodes_idx = torch.arange(total_nodes, device=device)
            n_i = total_nodes
        
        if n_i == 0:
            continue
        
        # Extract subgraph
        edge_index_i = None
        node_offset = ptr
        
        # Find edges for this graph
        if num_graphs > 1:
            edge_mask = torch.isin(batch.edge_index[0], nodes_idx) & torch.isin(batch.edge_index[1], nodes_idx)
            edge_index_i = batch.edge_index[:, edge_mask]
            
            # Adjust indices to be 0-based for this subgraph
            mapping = -torch.ones(total_nodes, dtype=torch.long, device=device)
            mapping[nodes_idx] = torch.arange(n_i, device=device)
            edge_index_i = mapping[edge_index_i]
        else:
            edge_index_i = batch.edge_index
        
        # Compute shortest paths for this graph
        sp_i = compute_shortest_path_distance(edge_index_i, n_i, max_distance)
        
        # Add to positional encoding tensor
        pe[i, :n_i, :n_i] = sp_i
        
        # Create attention mask (False where nodes exist, True for padding)
        attn_mask[i, :n_i, :n_i] = False
        
        ptr += n_i
    
    # Attach computed attributes to the batch
    batch.pe = pe  # Positional encoding (shortest path distances)
    batch.attn_mask = attn_mask  # Attention mask for padding
    batch.graph_sizes = graph_sizes  # Original sizes of graphs
    batch.max_nodes = max_nodes  # Maximum number of nodes in batch
    
    return batch


def prepare_single_graph(data, max_distance=5):
    """
    Prepare a single graph for Graphormer processing.
    
    Args:
        data: A PyG Data object
        max_distance: Maximum path distance to consider
        
    Returns:
        data: The same PyG Data object with additional 'pe' attribute
    """
    # Make a copy to avoid modifying the original
    data_new = copy.deepcopy(data)
    
    # Compute shortest path distances
    sp = compute_shortest_path_distance(data.edge_index, data.num_nodes, max_distance)
    
    # Add as positional encoding
    data_new.pe = sp.unsqueeze(0)  # Add batch dimension [1, num_nodes, num_nodes]
    
    # Create attention mask (all nodes are valid)
    data_new.attn_mask = torch.zeros(1, data.num_nodes, data.num_nodes, 
                                    dtype=torch.bool, device=data.x.device)
    
    # Add metadata
    data_new.graph_sizes = [data.num_nodes]
    data_new.max_nodes = data.num_nodes
    
    return data_new


# Usage example:
def example_usage():
    # For a single graph
    data = Data(
        x=torch.randn(10, 32),  # 10 nodes with 32 features each
        edge_index=torch.randint(0, 10, (2, 15))  # 15 random edges
    )
    prepared_data = prepare_single_graph(data)
    
    # For a batch of graphs
    data_list = [
        Data(x=torch.randn(5, 32), edge_index=torch.randint(0, 5, (2, 8))),
        Data(x=torch.randn(8, 32), edge_index=torch.randint(0, 8, (2, 12))),
        Data(x=torch.randn(3, 32), edge_index=torch.randint(0, 3, (2, 4)))
    ]
    prepared_batch = prepare_graphormer_batch(data_list)
    
    # Access the new attributes
    pe = prepared_batch.pe  # Shape: [3, 8, 8] (batch_size, max_nodes, max_nodes)
    attn_mask = prepared_batch.attn_mask
    graph_sizes = prepared_batch.graph_sizes  # [5, 8, 3]
    
    return prepared_data, prepared_batch