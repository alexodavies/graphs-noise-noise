import torch
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool


class FlexibleGNN(torch.nn.Module):
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
        super(FlexibleGNN, self).__init__()

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
        self.post_gnn = Linear(hidden_dim, num_classes)

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
        x = self.post_gnn(x)
        return x
