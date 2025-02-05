import torch
from torch.nn import Linear, ReLU, Sequential, ModuleList, Embedding, BatchNorm1d
from torch_geometric.nn import GCNConv, GINEConv, GATConv, GPSConv, global_add_pool, global_mean_pool

class FlexibleGNN(torch.nn.Module):
    VALID_LAYERS = {
        "gcn": GCNConv,      # GCN does not use edge attributes
        "gin": GINEConv,     # GIN supports edge attributes
        "gat": GATConv,      # GAT supports attention mechanism
        "gps": GPSConv,      # GPS combines local and global attention
    }

    def __init__(self, layer_type, node_in_dim, edge_in_dim, hidden_dim, num_classes, num_layers, task_type="graph", model_kwargs=None, pe_dim=0):
        """
        Args:
            layer_type (str): Type of GNN layer to use ("gcn", "gin", "gat", "gps").
            node_in_dim (int): Input dimension of node features.
            edge_in_dim (int): Input dimension of edge features.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            num_layers (int): Number of GNN layers.
            task_type (str): Task type ("graph" or "node").
            model_kwargs (dict): Additional keyword arguments for the selected GNN layer.
            pe_dim (int): Dimension of positional encoding (0 if not used).
        """
        super(FlexibleGNN, self).__init__()

        # Ensure the layer type is valid
        if layer_type not in self.VALID_LAYERS:
            raise ValueError(f"Invalid layer type '{layer_type}'. Valid options are: {list(self.VALID_LAYERS.keys())}")
        self.layer_type = layer_type

        # Select the layer type
        gnn_layer_type = self.VALID_LAYERS[layer_type]

        # Check if the GNN layer supports edge attributes
        self.use_edge_attr = layer_type in ["gin", "gps"]

        # Default model kwargs if not provided
        if model_kwargs is None:
            model_kwargs = {}

        self.use_pe = pe_dim != 0 # (layer_type == "gps")
        self.pe_dim = pe_dim

        # Node, edge, and positional embedding layers
        # self.node_embedding = Embedding(node_in_dim, hidden_dim - pe_dim) if self.use_pe else Linear(node_in_dim, hidden_dim)
        # self.edge_embedding = Embedding(edge_in_dim, hidden_dim) if self.use_edge_attr else None

        self.node_embedding = Linear(node_in_dim, hidden_dim - pe_dim)
        self.edge_embedding = Linear(edge_in_dim, hidden_dim) if self.use_edge_attr else None

        self.pe_lin = Linear(pe_dim, pe_dim) if self.use_pe else None
        self.pe_norm = BatchNorm1d(pe_dim) if self.use_pe else None

        # GNN layers
        self.gnn_layers = ModuleList()
        for i in range(num_layers):
            if layer_type == "gin":
                nn = Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
                self.gnn_layers.append(gnn_layer_type(nn, **model_kwargs))
            elif layer_type == "gat":
                num_heads = model_kwargs.get("heads", 1)
                in_dim = hidden_dim * num_heads if i > 0 else hidden_dim
                self.gnn_layers.append(
                    gnn_layer_type(in_dim, hidden_dim, **model_kwargs)  # Keep hidden_dim consistent
                )

            elif layer_type == "gps":
                num_heads = model_kwargs.get("heads", 1)

                # Ensure hidden_dim is divisible by num_heads
                if hidden_dim % num_heads != 0:
                    new_hidden_dim = hidden_dim + (num_heads - hidden_dim % num_heads)
                    print(f"Adjusting hidden_dim from {hidden_dim} to {new_hidden_dim} to be divisible by num_heads")
                    hidden_dim = new_hidden_dim  # Adjust hidden_dim to nearest valid value

                local_gnn = GINEConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        ReLU(),
                        Linear(hidden_dim, hidden_dim)
                    )
                )
                self.gnn_layers.append(gnn_layer_type(channels=hidden_dim, conv=local_gnn, **model_kwargs))

                # self.gnn_layers.append(gnn_layer_type(channels=hidden_dim, conv=local_gnn, **model_kwargs))
            else:  # For GCN and other layers
                self.gnn_layers.append(gnn_layer_type(hidden_dim, hidden_dim, **model_kwargs))

        # Post-GNN classifier

        output_dim = hidden_dim * model_kwargs.get("heads", 1) if layer_type in ["gat"] else hidden_dim
        self.post_gnn = Linear(output_dim, num_classes)

        # Task type: "graph" or "node"
        self.task_type = task_type

    def forward(self, data): #, x, edge_index, edge_attr=None, batch=None, pe=None):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        pe = data.pe if self.use_pe else None
        """
        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, node_in_dim].
            edge_index (torch.LongTensor): Edge index [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix [num_edges, edge_in_dim].
            batch (torch.Tensor): Batch assignment for nodes [num_nodes] (only needed for graph tasks).
            pe (torch.Tensor): Positional encoding matrix [num_nodes, pe_dim] (only needed for GPS).
        """
        if self.use_pe:
            # Normalize positional encodings and concatenate with node embeddings
            pe = self.pe_norm(pe) if self.pe_norm else pe
            node_embedding = self.node_embedding(x)
            pe_embedding = self.pe_lin(pe)
            x = torch.cat((node_embedding, pe_embedding), dim=-1)
            # x = torch.cat((self.node_embedding(x.squeeze(-1)), self.pe_lin(pe)), dim=-1)
        else:
            # Standard node embedding
            x = self.node_embedding(x)

        # Embed edge features if the GNN layer supports edge attributes
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:

            if self.layer_type == "gps": # originally gps
                x = gnn_layer(x, edge_index, edge_attr=edge_attr, batch=batch)
            elif self.layer_type == "gin": # originally gin
                x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            else: # originally gcn
                x = gnn_layer(x, edge_index)

            # if self.use_edge_attr and self.use_pe: # originally gps
            #     x = gnn_layer(x, edge_index, edge_attr=edge_attr, batch=batch)
            # elif self.use_edge_attr: # originally gin
            #     x = gnn_layer(x, edge_index, edge_attr=edge_attr)
            # else: # originally gcn, gat
            #     x = gnn_layer(x, edge_index)

            x = ReLU()(x)  # Apply non-linearity

        if self.task_type == "graph":
            # Global pooling for graph classification
            x = global_mean_pool(x, batch)

        # Final classifier
        
        x = self.post_gnn(x)
        return x


class FeatureExtractorGNN(torch.nn.Module):
    VALID_LAYERS = {
        "gcn": GCNConv,      # GCN does not use edge attributes
        "gin": GINEConv,     # GIN supports edge attributes
    }

    def __init__(self, layer_type, node_in_dim, edge_in_dim, hidden_dim, num_classes, num_layers, task_type="graph"):
        """
        Args:
            layer_type (str): Type of GNN layer to use ("gcn", "gin").
            node_in_dim (int): Input dimension of node features.
            edge_in_dim (int): Input dimension of edge features.
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Number of output classes.
            num_layers (int): Number of GNN layers.
            task_type (str): Task type ("graph" or "node").
        """
        super(FeatureExtractorGNN, self).__init__()

        # Ensure the layer type is valid
        if layer_type not in self.VALID_LAYERS:
            raise ValueError(f"Invalid layer type '{layer_type}'. Valid options are: {list(self.VALID_LAYERS.keys())}")

        # Select the layer type
        gnn_layer_type = self.VALID_LAYERS[layer_type]

        # Check if the GNN layer supports edge attributes
        self.use_edge_attr = layer_type == "gin"

        # Node and edge feature embedding layers
        self.node_embedding = Linear(node_in_dim, hidden_dim)
        self.edge_embedding = Linear(edge_in_dim, hidden_dim) if self.use_edge_attr else None

        # GNN layers
        self.gnn_layers = ModuleList([
            gnn_layer_type(
                hidden_dim,
                hidden_dim
            ) if layer_type != "gin" else gnn_layer_type(
                Sequential(
                    Linear(hidden_dim, hidden_dim),
                    ReLU(),
                    Linear(hidden_dim, hidden_dim)
                )
            )
            for _ in range(num_layers)
        ])

        # Post-GNN classifier
        # self.post_gnn = Linear(hidden_dim, num_classes)

        # Task type: "graph" or "node"
        self.task_type = task_type

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, node_in_dim].
            edge_index (torch.LongTensor): Edge index [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix [num_edges, edge_in_dim].
            batch (torch.Tensor): Batch assignment for nodes [num_nodes] (only needed for graph tasks).
        """
        # Embed node features
        x = self.node_embedding(x)

        # Embed edge features if the GNN layer supports edge attributes
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_embedding(edge_attr)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            if self.use_edge_attr:
                x = gnn_layer(x, edge_index, edge_attr)
            else:
                x = gnn_layer(x, edge_index)  # For layers like GCNConv
            x = ReLU()(x)  # Apply non-linearity

        if self.task_type == "graph":
            # Global pooling for graph classification
            x = global_mean_pool(x, batch)

        # Final classifier
        # x = self.post_gnn(x)
        return x