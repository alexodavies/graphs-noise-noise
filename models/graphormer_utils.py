import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import to_undirected
import time
import os
import copy


def compute_shortest_path_distance(edge_index, num_nodes, max_distance=5):
    """
    Compute shortest path distances using the Floyd-Warshall algorithm.
    
    Args:
        edge_index: Graph connectivity [2, num_edges]
        num_nodes: Number of nodes in the graph
        max_distance: Maximum distance to consider
        
    Returns:
        dist: Shortest path distances [num_nodes, num_nodes]
    """
    device = edge_index.device
    
    # Handle empty graphs
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), device=device).long()
    
    # Make the graph undirected for distance calculation
    edge_index = to_undirected(edge_index)
    
    # Initialize distance matrix
    dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
    dist[edge_index[0], edge_index[1]] = 1.0
    dist.fill_diagonal_(0)
    
    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        # Update distances: dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    
    # Clip distances to max_distance
    dist = torch.clamp(dist, 0, max_distance)
    
    return dist.long()


def add_shortest_paths_to_data(data, max_distance=5):
    """
    Compute and add shortest path distances to a PyG Data object.
    
    Args:
        data: PyG Data object
        max_distance: Maximum path distance to consider
        
    Returns:
        data: The same Data object with additional 'shortest_paths' attribute
    """
    # Make a copy to avoid modifying the original data
    data_with_paths = copy.deepcopy(data)
    
    # Compute shortest paths
    shortest_paths = compute_shortest_path_distance(
        data.edge_index, data.num_nodes, max_distance
    )
    
    # Add to data object
    data_with_paths.shortest_paths = shortest_paths
    
    return data_with_paths


def precompute_shortest_paths_dataset(dataset, max_distance=5, batch_size=1000, 
                                     cache_dir=None, force_recompute=False):
    """
    Precompute shortest path distances for all graphs in a dataset.
    Processes in batches to reduce memory pressure.
    Optionally caches computations to disk.
    
    Args:
        dataset: PyG dataset
        max_distance: Maximum path distance
        batch_size: Number of graphs to process at once
        cache_dir: Directory to cache computations (None for no caching)
        force_recompute: Whether to recompute paths even if cached
        
    Returns:
        Processed dataset with shortest_paths attributes
    """
    # Set up progress tracking
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("Install tqdm for progress bars")
    
    # Create cache directory if needed
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        
    # Process dataset in batches
    total_graphs = len(dataset)
    batches = (total_graphs + batch_size - 1) // batch_size
    
    # Set up progress bar
    if has_tqdm:
        pbar = tqdm(total=total_graphs, desc="Computing shortest paths")
    
    for batch_idx in range(batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_graphs)
        
        # Process each graph in the batch
        for i in range(start_idx, end_idx):
            # Check cache
            cache_hit = False
            if cache_dir is not None:
                cache_file = os.path.join(cache_dir, f"graph_{i}_paths.pt")
                if os.path.exists(cache_file) and not force_recompute:
                    try:
                        dataset[i].shortest_paths = torch.load(cache_file)
                        cache_hit = True
                    except Exception as e:
                        print(f"Error loading cache for graph {i}: {e}")
            
            # Compute if not cached
            if not cache_hit:
                # Get the data
                data = dataset[i]
                
                # Skip if already has shortest_paths
                if hasattr(data, 'shortest_paths') and not force_recompute:
                    if has_tqdm:
                        pbar.update(1)
                    continue
                
                # Compute shortest paths
                shortest_paths = compute_shortest_path_distance(
                    data.edge_index, data.num_nodes, max_distance
                )
                
                # Add to data object
                dataset[i].shortest_paths = shortest_paths
                
                # Save to cache if requested
                if cache_dir is not None:
                    torch.save(shortest_paths, cache_file)
            
            # Update progress
            if has_tqdm:
                pbar.update(1)
    
    if has_tqdm:
        pbar.close()
    
    return dataset


def add_shortest_paths_to_dataset(dataset, max_distance=5, in_place=True):
    """
    Simple function to add shortest paths to all graphs in a dataset.
    
    Args:
        dataset: PyG dataset
        max_distance: Maximum path distance
        in_place: Whether to modify the dataset in place
        
    Returns:
        Processed dataset with shortest_paths attributes
    """
    # Create copy if not modifying in place
    if not in_place:
        dataset = copy.deepcopy(dataset)
    
    # Set up progress tracking
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(dataset)), desc="Computing shortest paths")
    except ImportError:
        iterator = range(len(dataset))
        print("Computing shortest paths...")
    
    # Process each graph
    for i in iterator:
        data = dataset[i]
        
        # Skip if already computed
        if hasattr(data, 'shortest_paths'):
            continue
        
        # Compute and add shortest paths
        shortest_paths = compute_shortest_path_distance(
            data.edge_index, data.num_nodes, max_distance
        )
        data.shortest_paths = shortest_paths
        dataset[i] = data
    
    return dataset


def precompute_and_add_shortest_paths(dataset, max_distance=5):
    """
    Precompute shortest paths for each graph and add them directly to the data objects.
    
    Args:
        dataset: PyG dataset
        max_distance: Maximum path distance to consider
        
    Returns:
        dataset: Modified dataset with shortest_paths attributes
    """
    from torch_geometric.utils import to_undirected
    
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(dataset)), desc="Computing shortest paths")
    except ImportError:
        iterator = range(len(dataset))
        print("Computing shortest paths...")
    
    for i in iterator:
        data = dataset[i]
        
        # Skip if already computed
        if hasattr(data, 'shortest_paths'):
            continue
        
        # Get graph info
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Compute shortest paths using Floyd-Warshall
        shortest_paths = compute_shortest_path_distance(edge_index, num_nodes, max_distance)
        
        # Add to data object
        data.shortest_paths = shortest_paths
        dataset[i] = data
    
    return dataset


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





# Example usage
if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    # Load a dataset
    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    
    # Precompute shortest paths
    dataset = add_shortest_paths_to_dataset(dataset, max_distance=5)
    
    # Create dataloader with custom collate function
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=custom_collate_with_paths
    )
    
    # Verify that the batches contain shortest_paths_list
    for batch in loader:
        assert hasattr(batch, 'shortest_paths_list')
        print(f"Batch with {batch.num_graphs} graphs and {len(batch.shortest_paths_list)} path matrices")
        break