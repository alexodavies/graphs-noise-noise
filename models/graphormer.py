import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


class FeedForwardNetwork(nn.Module):
    """
    Standard MLP Feed-Forward Network used in Transformer models.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class GraphormerAttention(nn.Module):
    """
    Graphormer attention mechanism with spatial bias and edge encoding.
    Carefully designed to handle tensor dimensions correctly.
    """
    def __init__(self, 
                dim, 
                num_heads=8, 
                qkv_bias=False, 
                attn_drop=0., 
                proj_drop=0., 
                max_path_distance=5,
                use_edge_encoding=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_edge_encoding = use_edge_encoding
        self.max_path_distance = max_path_distance

        # Check that dimensions align
        assert self.head_dim * num_heads == dim, "Embed dimension must be divisible by num_heads"

        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Spatial encoding - encodes shortest path distances
        self.spatial_pos_encoder = nn.Embedding(max_path_distance + 1, num_heads)
        
        # Edge encoding - for incorporating edge features if available
        if use_edge_encoding:
            self.edge_encoder = nn.Embedding(2, num_heads)  # For basic edge existence
            
        # Virtual [CLS] token for graph-level representation
        self.graph_token = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, node_features, path_distances, edge_attr=None, attn_mask=None):
        """
        Args:
            node_features: Node features [batch_size, num_nodes, dim]
            path_distances: Shortest path distances [batch_size, num_nodes, num_nodes]
            edge_attr: Edge features or existence [batch_size, num_nodes, num_nodes] (optional)
            attn_mask: Attention mask [batch_size, num_nodes, num_nodes] (optional)
            
        Returns:
            Node representations and graph representation
        """
        # Get batch size, number of nodes, and feature dimension
        B, N, C = node_features.shape
        
        # Add virtual [CLS] token
        graph_token = self.graph_token.expand(B, 1, C)
        x = torch.cat([graph_token, node_features], dim=1)  # [B, N+1, C]
        
        # Create positional encoding with virtual node
        pe_with_cls = torch.zeros(B, N+1, N+1, dtype=torch.long, device=path_distances.device)
        
        # Set virtual node distances (0 to itself, 1 to all other nodes)
        pe_with_cls[:, 0, 0] = 0
        pe_with_cls[:, 0, 1:] = 1
        pe_with_cls[:, 1:, 0] = 1
        
        # Copy the original graph distances
        pe_with_cls[:, 1:, 1:] = path_distances
        
        # Project input to query, key, value tensors
        qkv = self.qkv(x)  # [B, N+1, 3*C]
        
        # Split and reshape qkv into separate q, k, v tensors
        qkv = qkv.reshape(B, N+1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N+1, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, num_heads, N+1, head_dim]
        
        # Compute raw attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N+1, N+1]
        
        # Apply spatial bias based on shortest path distances
        spatial_pos_bias = self.spatial_pos_encoder(pe_with_cls)  # [B, N+1, N+1, num_heads]
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)  # [B, num_heads, N+1, N+1]
        attn = attn + spatial_pos_bias
        
        # Apply edge encoding if available and requested
        if self.use_edge_encoding and edge_attr is not None:
            # Create edge encoding tensor with virtual node
            edge_attr_with_cls = torch.zeros(B, N+1, N+1, dtype=torch.long, device=edge_attr.device)
            edge_attr_with_cls[:, 1:, 1:] = edge_attr
            
            # Encode and add to attention
            edge_bias = self.edge_encoder(edge_attr_with_cls)  # [B, N+1, N+1, num_heads]
            edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, num_heads, N+1, N+1]
            attn = attn + edge_bias
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # Create attention mask with virtual node
            mask_with_cls = torch.ones(B, N+1, N+1, dtype=torch.bool, device=attn_mask.device)
            mask_with_cls[:, 0, :] = False  # Virtual node attends to all nodes
            mask_with_cls[:, :, 0] = False  # All nodes attend to virtual node
            mask_with_cls[:, 1:, 1:] = attn_mask
            
            # Apply mask
            attn = attn.masked_fill(mask_with_cls.unsqueeze(1), float('-inf'))
            
        # Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Compute weighted values
        out = attn @ v  # [B, num_heads, N+1, head_dim]
        out = out.transpose(1, 2)  # [B, N+1, num_heads, head_dim]
        out = out.reshape(B, N+1, C)  # [B, N+1, C]
        
        # Project output
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Split virtual node and rest of nodes
        node_out = out[:, 1:]  # [B, N, C]
        graph_out = out[:, 0]  # [B, C]
        
        return node_out, graph_out


class GraphormerLayer(nn.Module):
    """
    Complete Graphormer Layer with pre-normalization, attention, and FFN.
    """
    def __init__(self, 
                dim, 
                num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                dropout=0., 
                attn_dropout=0.,
                max_path_distance=5,
                use_edge_encoding=True):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_graph = nn.LayerNorm(dim)
        
        # Multi-head attention
        self.attn = GraphormerAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
            max_path_distance=max_path_distance,
            use_edge_encoding=use_edge_encoding
        )
        
        # Feed-forward networks
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForwardNetwork(dim, mlp_hidden_dim, dropout)
        self.graph_ffn = FeedForwardNetwork(dim, mlp_hidden_dim, dropout)
        
    def forward(self, x, path_distances, edge_attr=None, attn_mask=None):
        """
        Forward pass for the Graphormer layer.
        
        Args:
            x: Node features [batch_size, num_nodes, dim]
            path_distances: Shortest path distances [batch_size, num_nodes, num_nodes]
            edge_attr: Edge features [batch_size, num_nodes, num_nodes] (optional)
            attn_mask: Attention mask [batch_size, num_nodes, num_nodes] (optional)
            
        Returns:
            Updated node features and graph representation
        """
        # Attention block with residual connection
        x_norm = self.norm1(x)
        attn_out, graph_out = self.attn(x_norm, path_distances, edge_attr, attn_mask)
        x = x + attn_out
        
        # FFN block with residual connection
        x = x + self.ffn(self.norm2(x))
        
        # Process graph token separately
        graph_out = graph_out + self.graph_ffn(self.norm_graph(graph_out))
        
        return x, graph_out


class Graphormer(nn.Module):
    """
    Complete Graphormer model with internal preprocessing.
    """
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers=6, 
                 num_heads=8,
                 dropout=0.1,
                 attn_dropout=0.1,
                 max_path_distance=5,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 use_edge_encoding=True,
                 pooling="cls"):  # 'cls' or 'mean'
        super(Graphormer, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.max_path_distance = max_path_distance
        self.pooling = pooling
        
        # Node feature embedding
        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        
        # Graphormer layers
        self.layers = nn.ModuleList([
            GraphormerLayer(
                dim=hidden_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                max_path_distance=max_path_distance,
                use_edge_encoding=use_edge_encoding
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm and projection
        self.norm = nn.LayerNorm(hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def compute_shortest_path_distance(self, edge_index, num_nodes):
        """
        Compute shortest path distances using Floyd-Warshall algorithm.
        
        Args:
            edge_index: Graph connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph
            
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
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        # Clip distances to max_path_distance
        dist = torch.clamp(dist, 0, self.max_path_distance)
        
        return dist.long()
    
    def prepare_batch_from_pyg(self, data):
        """
        Process a PyG Batch object into the format needed for Graphormer.
        
        Args:
            data: PyG Batch object
            
        Returns:
            node_features: Node features [batch_size, max_nodes, hidden_dim]
            path_distances: Positional encodings [batch_size, max_nodes, max_nodes]
            attn_mask: Attention mask [batch_size, max_nodes, max_nodes]
            batch_indices: List of (start_idx, end_idx) for each graph in batch
        """
        device = data.x.device

        # Get batch information
        if hasattr(data, 'batch') and data.batch.size(0) > 0:
            batch_idx = data.batch
            num_graphs = int(batch_idx.max()) + 1  # batch is 0-indexed
            graph_sizes = torch.bincount(batch_idx).tolist()
            max_nodes = max(graph_sizes)
            batch_size = num_graphs
        else:
            # Single graph
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            graph_sizes = [data.x.size(0)]
            max_nodes = graph_sizes[0]
            batch_size = 1
            num_graphs = 1
        
        # Create tensors with the right shape
        node_features = torch.zeros(batch_size, max_nodes, self.hidden_channels, device=device)
        path_distances = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)
        attn_mask = torch.ones(batch_size, max_nodes, max_nodes, dtype=torch.bool, device=device)
        
        # Process each graph
        batch_indices = []
        start_idx = 0
        
        for i in range(num_graphs):
            # Get number of nodes for this graph
            if i < len(graph_sizes):
                size = graph_sizes[i]
                if size > 0:
                    # Process node features
                    node_feats = self.node_encoder(data.x[start_idx:start_idx + size])
                    node_features[i, :size] = node_feats
                    
                    # Get edges for this graph
                    if hasattr(data, 'edge_index'):
                        # Find edges for this graph
                        edge_mask = torch.logical_and(
                            data.edge_index[0] >= start_idx, 
                            data.edge_index[0] < start_idx + size
                        )
                        edge_mask = torch.logical_and(
                            edge_mask,
                            torch.logical_and(
                                data.edge_index[1] >= start_idx,
                                data.edge_index[1] < start_idx + size
                            )
                        )
                        
                        if edge_mask.sum() > 0:
                            edges = data.edge_index[:, edge_mask].clone()
                            
                            # Adjust edge indices to be 0-based for this graph
                            edges[0] -= start_idx
                            edges[1] -= start_idx
                            
                            # Compute shortest paths for this graph
                            sp = self.compute_shortest_path_distance(edges, size)
                            path_distances[i, :size, :size] = sp
                        
                    # Set attention mask (False = attend, True = mask out)
                    attn_mask[i, :size, :size] = False
                    
                    # Track indices for this graph
                    batch_indices.append((start_idx, start_idx + size))
                    start_idx += size
        
        return node_features, path_distances, attn_mask, batch_indices
        
    def forward(self, data):
        """
        Forward pass with internal preprocessing for PyG data.
        
        Args:
            data: PyG Data or Batch object
            
        Returns:
            out: Model output
        """
        # Preprocess the PyG data
        node_features, path_distances, attn_mask, batch_indices = self.prepare_batch_from_pyg(data)
        
        # Apply Graphormer layers
        last_graph_output = None
        
        for layer in self.layers:
            node_features, graph_output = layer(
                node_features, 
                path_distances,
                attn_mask=attn_mask
            )
            last_graph_output = graph_output
        
        # Apply final normalization to node features
        node_features = self.norm(node_features)
        
        # For graph-level tasks, use appropriate pooling
        if self.pooling == 'cls':
            # Use the graph token from the last layer
            pooled = last_graph_output
        else:  # 'mean' pooling
            # Mean pooling of node features for each graph
            batch_size = len(batch_indices)
            pooled = []
            
            for i in range(batch_size):
                if i < len(batch_indices):
                    size = batch_indices[i][1] - batch_indices[i][0]
                    if size > 0:
                        # Average the non-padded node features
                        graph_nodes = node_features[i, :size]
                        pooled.append(graph_nodes.mean(dim=0))
            
            pooled = torch.stack(pooled) if pooled else torch.zeros(batch_size, self.hidden_channels, device=node_features.device)
        
        # Final projection
        out = self.out_proj(pooled)
        
        return out