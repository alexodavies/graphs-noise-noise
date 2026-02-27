import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch


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
    Graphormer attention mechanism optimized to use precomputed path distances.
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
        
    def forward(self, x, path_distances, edge_attr=None, attn_mask=None):
        """
        Args:
            x: Node features [batch_size, num_nodes, dim]
            path_distances: Shortest path distances [batch_size, num_nodes, num_nodes]
            edge_attr: Edge features or existence [batch_size, num_nodes, num_nodes] (optional)
            attn_mask: Attention mask [batch_size, num_nodes, num_nodes] (optional)
            
        Returns:
            Node representations and graph representation
        """
        # Get batch size, number of nodes, and feature dimension
        B, N, C = x.shape
        
        # Add virtual [CLS] token
        graph_token = self.graph_token.expand(B, 1, C)
        x = torch.cat([graph_token, x], dim=1)  # [B, N+1, C]
        
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
    Graphormer model that uses precomputed shortest paths.
    This is much faster than computing paths during the forward pass.
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
    
    def prepare_batch(self, data):
        """
        Convert PyG batch to dense tensors for Graphormer processing.
        Uses precomputed shortest paths for efficiency.
        
        Args:
            data: PyG Batch object with shortest_paths_list
            
        Returns:
            node_features: [batch_size, max_nodes, hidden_dim]
            path_distances: [batch_size, max_nodes, max_nodes]
            attn_mask: [batch_size, max_nodes, max_nodes]
            graph_indices: List of (batch_idx, num_nodes) tuples
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
        
        # Initialize dense tensors
        node_features = torch.zeros(batch_size, max_nodes, self.hidden_channels, device=device)
        path_distances = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)
        attn_mask = torch.ones(batch_size, max_nodes, max_nodes, dtype=torch.bool, device=device)
        
        # Track graph indices in the batch
        graph_indices = []
        
        # Process each graph
        ptr = 0
        for i in range(num_graphs):
            if i >= len(graph_sizes):
                continue
                
            size = graph_sizes[i]
            if size > 0:
                # Encode node features
                node_feats = self.node_encoder(data.x[ptr:ptr+size])
                node_features[i, :size] = node_feats
                
                # Use precomputed shortest paths if available
                if hasattr(data, 'shortest_paths_list') and i < len(data.shortest_paths_list):
                    paths = data.shortest_paths_list[i]
                    if paths is not None:
                        path_distances[i, :size, :size] = paths
                
                # Set attention mask (False = attend, True = mask out)
                attn_mask[i, :size, :size] = False
                
                # Store graph indices
                graph_indices.append((i, size))
                
                # Update pointer
                ptr += size
        
        return node_features, path_distances, attn_mask, graph_indices
    
    def forward(self, data):
        """
        Forward pass with precomputed shortest paths.
        
        Args:
            data: PyG Batch object with shortest_paths_list attribute
            
        Returns:
            out: Model output
        """
        # Convert batch to dense format
        node_features, path_distances, attn_mask, graph_indices = self.prepare_batch(data)
        
        # Process through Graphormer layers
        last_graph_output = None
        
        for layer in self.layers:
            node_features, graph_output = layer(
                node_features, 
                path_distances,
                attn_mask=attn_mask
            )
            last_graph_output = graph_output
        
        # Apply final normalization
        node_features = self.norm(node_features)
        
        # Pool node features for graph representation
        if self.pooling == 'cls':
            # Use the graph token
            pooled = last_graph_output
        else:  # 'mean' pooling
            # Average valid node representations for each graph
            pooled = []
            for batch_idx, size in graph_indices:
                if size > 0:
                    graph_nodes = node_features[batch_idx, :size]
                    pooled.append(graph_nodes.mean(dim=0))
            
            pooled = torch.stack(pooled) if pooled else torch.zeros(
                len(graph_indices), self.hidden_channels, device=node_features.device
            )
        
        # Final projection
        out = self.out_proj(pooled)
        
        return out