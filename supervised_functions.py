import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import numpy as np
import torch.nn.functional as F
import copy
from time import time
from tqdm import tqdm

from model import FlexibleGNN, FeatureExtractorGNN  # Import FlexibleGNN from a separate file
from noisenoise import add_noise_to_dataset
from synthetic_datasets import SyntheticDataset, SyntheticDouble
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

tu_classes_lookup: dict = {"ENZYMES":6,
                           "MUTAG":2,
                           "PROTEINS":2,
                           "COLLAB":3,
                           "IMDB-BINARY":2,
                           "REDDIT-BINARY":2}

    


def add_pe_to_dataset(dataset, pe_dim, walk_length=20, attr_name='pe'):
    """
    Adds positional encodings to all graphs in a PyTorch Geometric dataset.

    Args:
        dataset (Dataset): PyTorch Geometric dataset containing the graphs.
        pe_dim (int): Dimension of the positional encoding.
        walk_length (int): Length of the random walks to compute the positional encoding.
        attr_name (str): Attribute name to store the positional encodings in the Data object.

    Returns:
        Dataset: The updated PyTorch Geometric dataset with positional encodings added to all graphs.
    """
    # LaplacianPE = AddLaplacianEigenvectorPE(pe_dim, attr_name = attr_name, is_undirected=True)
    RWPE = AddRandomWalkPE(walk_length, attr_name = attr_name)
    # Add positional encodings to all graphs in the dataset
    for idata, data in enumerate(dataset):
        dataset[idata] = RWPE.forward(data) # , pe_dim, walk_length, attr_name)

    return dataset

def infer_task_type(dataset):
    """Infer task type and task level from dataset."""
    # Task level: node or graph
    is_graph_level = hasattr(dataset[0], "y") and dataset[0].y.dim() > 0
    task_level = "graph" if is_graph_level else "node"

    # Task type: classification or regression
    try:
        task_type = dataset.task_type
    except:
        return task_level, "multiclass-classification"
        
    if "classification" in task_type:
        return task_level, "classification"
    elif "regression" in task_type:
        return task_level, "regression"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def train(model, optimizer, loader, device, task_type):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)

        # Ensure features and labels are float
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()

        optimizer.zero_grad()
        out = model(data) # data.x, data.edge_index, data.edge_attr, data.batch)

        task_losses = []
        if len(data.y.shape) == 1:  # Single task
            data.y = data.y.reshape(-1,1)  # Add task dimension


        if task_type == "multiclass-classification":
            task_loss = F.cross_entropy(out, data.y.argmax(dim=-1))
            task_losses.append(task_loss)

        else:
            for task_idx in range(data.y.shape[1]):  # Loop over tasks
                valid_mask = ~torch.isnan(data.y[:, task_idx])  # Mask for valid labels in this task
                if valid_mask.sum() > 0:  # Only compute loss if there are valid labels
                    if task_type == "classification":
                        task_loss = F.binary_cross_entropy_with_logits(
                            out[valid_mask, task_idx], data.y[valid_mask, task_idx]
                        )

                    elif task_type == "regression":
                        task_loss = F.mse_loss(
                            out[valid_mask, task_idx], data.y[valid_mask, task_idx]
                        )
                    task_losses.append(task_loss)

        # Combine task-wise losses (mean across tasks)
        if len(task_losses) > 0:
            loss = torch.stack(task_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)



def evaluate(model, loader, device, task_type):
    """Evaluate the model on the dataset and return the performance metric."""
    model.eval()
    task_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Ensure features and labels are float
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.y = data.y.float()
            if len(data.y.shape) == 1:  # Single task
                data.y = data.y.reshape(-1,1)  # Add task dimension
            out = model(data) # data.x, data.edge_index, data.edge_attr, data.batch)

            if task_type == "multiclass-classification":
                preds = F.softmax(out, dim = -1).cpu().numpy() #torch.argmax(out, dim=-1).cpu().numpy()
                if preds.shape[1] == 2:
                    preds = preds[:,1]
                labels = torch.argmax(data.y, dim=-1).cpu().numpy()
                score = roc_auc_score(labels, preds, multi_class="ovo")
                task_scores.append(score)
                task_scores.append(score)
            else:
                for task_idx in range(data.y.shape[1]):  # Loop over tasks
                    valid_mask = ~torch.isnan(data.y[:, task_idx])  # Mask for valid labels in this task
                    if valid_mask.sum() > 0:  # Only compute metric if there are valid labels
                        if task_type == "classification":
                            preds = torch.sigmoid(out[valid_mask, task_idx]).cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            if np.unique(labels).size > 1:  # Avoid invalid ROC-AUC computation
                                score = roc_auc_score(labels, preds)
                                task_scores.append(score)
                        elif task_type == "regression":
                            preds = out[valid_mask, task_idx].cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            score = root_mean_squared_error(labels, preds)  # RMSE
                            task_scores.append(score)
    mean_score = np.mean(task_scores)
    # Average metric across tasks
    return mean_score




def train_and_evaluate(dataset, test_dataset, layer_type, hidden_dim, num_layers, batch_size, epochs, lr, t_structure, t_feature, device, pos_encodings = False):
    """
    Train and evaluate the FlexibleGNN on the given dataset with added noise.

    Args:
        dataset: PyTorch Geometric dataset for training.
        test_dataset: PyTorch Geometric dataset for evaluation.
        layer_type (str): GNN layer type to use ("gcn", "gin").
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of GNN layers.
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        t_structure (float): Noise level for graph structure.
        t_feature (float): Noise level for node features.
        device (torch.device): Device to train on.

    Returns:
        Final test performance.
    """
    # Infer task level and type
    task_level, task_type = infer_task_type(dataset)
    pe_original_dim = 20

    # Create noisy copies of datasets
    noisy_train_dataset = add_noise_to_dataset(copy.deepcopy(dataset), t_structure, t_feature)
    if pos_encodings:
        pe_dim = int(0.2 * hidden_dim)
        noisy_train_dataset = add_pe_to_dataset(noisy_train_dataset, pe_original_dim, attr_name='pe')
    else:
        pe_dim = 0

    noisy_train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)

    noisy_test_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), t_structure, t_feature)
    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False)
    if pos_encodings:
        noisy_test_dataset = add_pe_to_dataset(noisy_test_dataset, pe_original_dim, attr_name='pe')

    # Get dataset dimensions
    node_in_dim = dataset.num_node_features
    edge_in_dim = dataset.num_edge_features if hasattr(dataset, "num_edge_features") else 0
    num_classes = dataset[0].y.shape[-1] if task_type == "classification" or task_type == "multiclass-classification" else 1

    gin_model_kwargs = {
    "eps": 0,  # Initial epsilon value for the learnable scalar.
    "train_eps": True,  # Allow epsilon to be learnable.
    }
    gcn_model_kwargs = {
    "add_self_loops": True,  # Whether to add self-loops to the graph.
    "normalize": True,       # Whether to apply symmetric normalization.
    }
    gat_model_kwargs = {
        "heads": 4,             # Number of attention heads.
        "concat": True,         # Whether to concatenate outputs of all heads.
        "negative_slope": 0.2,  # LeakyReLU angle of the negative slope.
        "dropout": 0.6,         # Dropout probability on attention weights.
    }
    gps_model_kwargs = {
        "heads": 4,                  # Number of attention heads.
        "attn_type": "multihead",    # Type of attention ("multihead" or "performer").
        "attn_kwargs": {
            "dropout": 0.5          # Dropout for attention.
        },
        "dropout": 0.2,              # Dropout in message-passing layers.
        "act": "relu",               # Activation function for GPS layers.
        "norm": "batch_norm",        # Normalization method for GPS layers.
    }

    kwarg_lookup = {"gin": gin_model_kwargs, "gcn": gcn_model_kwargs, "gat": gat_model_kwargs, "gps": gps_model_kwargs}

    # Initialize model
    # model = FlexibleGNN(
    #     layer_type=layer_type,
    #     node_in_dim=node_in_dim,
    #     edge_in_dim=edge_in_dim,
    #     hidden_dim=hidden_dim,
    #     num_layers=num_layers,
    #     num_classes=num_classes,
    #     task_type=task_level
    # ).to(device)

    model = FlexibleGNN(
        layer_type=layer_type,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        task_type=task_level,
        model_kwargs=kwarg_lookup[layer_type],
        pe_dim=pe_dim
    ).to(device)

    # model = FlexibleGNN(
    #     layer_type="gps",
    #     node_in_dim=28,
    #     edge_in_dim=4,
    #     hidden_dim=64,
    #     num_classes=1,
    #     num_layers=10,
    #     task_type="graph",
    #     model_kwargs={
    #         "heads": 4,
    #         "attn_type": "multihead",
    #         "attn_kwargs": {"dropout": 0.5}
    #     },
    #     pe_dim=20
    # .to(device))


    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(1, epochs + 1), leave = False):
        train_loss = train(model, optimizer, noisy_train_loader, device, task_type)
    # Evaluate on the final epoch
    test_performance = evaluate(model, noisy_test_loader, device, task_type)

    metric = "ROC-AUC" if task_type == "classification" else "RMSE"

    return test_performance, task_type


def train_and_evaluate_linear(
    dataset,
    test_dataset,
    layer_type,
    hidden_dim,
    num_layers,
    batch_size,
    epochs,
    lr,
    t_structure,
    t_feature,
    device,
):
    """
    Train and evaluate a linear model on embeddings from the FeatureExtractorGNN.

    Args:
        dataset: PyTorch Geometric dataset for training.
        test_dataset: PyTorch Geometric dataset for evaluation.
        layer_type (str): GNN layer type to use ("gcn", "gin").
        hidden_dim (int): Hidden layer dimension.
        num_layers (int): Number of GNN layers.
        batch_size (int): Batch size (not used for linear model).
        epochs (int): Number of training epochs (not used for linear model).
        lr (float): Learning rate (not used for linear model).
        t_structure (float): Noise level for graph structure.
        t_feature (float): Noise level for node features.
        device (torch.device): Device to train on.

    Returns:
        Final test performance (mean across tasks) and task type.
    """
    # Infer task level and type
    task_level, task_type = infer_task_type(dataset)

    # Add noise to datasets
    noisy_train_dataset = add_noise_to_dataset(copy.deepcopy(dataset), t_structure, t_feature)
    noisy_test_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), t_structure, t_feature)

    # Initialize the feature extraction model
    node_in_dim = dataset.num_node_features
    edge_in_dim = dataset.num_edge_features if hasattr(dataset, "num_edge_features") else 0
    model = FeatureExtractorGNN(
        layer_type=layer_type,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=hidden_dim,  # Outputs embeddings
        task_type=task_level,
    ).to(device)
    model.eval()
    with torch.no_grad():
        # Collect train embeddings
        train_embeddings = []
        train_targets = []
        for data in noisy_train_dataset:
            data = data.to(device)
            out = model(data) # data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            train_embeddings.append(out.cpu().numpy())
            train_targets.append(data.y.cpu().numpy())
        train_embeddings = np.vstack(train_embeddings)
        train_targets = np.vstack(train_targets)

        # Collect test embeddings
        test_embeddings = []
        test_targets = []
        for data in noisy_test_dataset:
            data = data.to(device)
            out = model(data) # data.x.float(), data.edge_index, data.edge_attr.float(), data.batch)
            test_embeddings.append(out.cpu().numpy())
            test_targets.append(data.y.cpu().numpy())
        test_embeddings = np.vstack(test_embeddings)
        test_targets = np.vstack(test_targets)

    # Train and evaluate per task
    task_scores = []
    for task_idx in range(train_targets.shape[1]):  # Loop over tasks
        valid_train_mask = ~np.isnan(train_targets[:, task_idx])
        valid_test_mask = ~np.isnan(test_targets[:, task_idx])

        if valid_train_mask.sum() > 0 and valid_test_mask.sum() > 0:
            if task_type == "classification":
                lin_model = LogisticRegression(max_iter=5000)
                lin_model.fit(
                    train_embeddings[valid_train_mask],
                    train_targets[valid_train_mask, task_idx],
                )
                preds = lin_model.predict_proba(test_embeddings[valid_test_mask])[:, 1]
                labels = test_targets[valid_test_mask, task_idx]
                if np.unique(labels).size > 1:  # Avoid invalid ROC-AUC computation
                    score = roc_auc_score(labels, preds)
                    task_scores.append(score)
            elif task_type == "regression":
                lin_model = LinearRegression()
                lin_model.fit(
                    train_embeddings[valid_train_mask],
                    train_targets[valid_train_mask, task_idx],
                )
                preds = lin_model.predict(test_embeddings[valid_test_mask])
                labels = test_targets[valid_test_mask, task_idx]
                score = root_mean_squared_error(labels, preds)
                task_scores.append(score)

    # Return the mean score across tasks
    mean_score = np.mean(task_scores) if len(task_scores) > 0 else 0.0
    return mean_score, task_type


class UnifyTUFormat:
    def __init__(self, num_classes=2):
        """
        Initialize the transform with optional parameters.

        Args:
            param: Optional parameter to control the transform behavior.
        """
        self.num_classes = num_classes

    def __call__(self, data):
        """
        Apply the transform to the input data.

        Args:
            data (torch_geometric.data.Data): The input graph data object.

        Returns:
            torch_geometric.data.Data: The transformed graph data object.
        """
        # Ensure the input is a PyTorch Geometric Data object
        assert isinstance(data, Data), "Input must be a torch_geometric.data.Data object"
        # Example modification: Add a feature column to node features
        if data.x is not None:
            pass
        else:
            data.x = torch.ones(data.num_nodes, 1)  # Create features if none exist

        # Example modification: Add a feature column to node features
        if data.edge_attr is not None:
            pass
        else:
            data.edge_attr = torch.ones(data.num_edges, 1)  # Create features if none exist

        data.y = F.one_hot(data.y, num_classes=self.num_classes)

        return data

def load_tu_dataset(dataset_name):
    """
    Load a graph dataset by name.
    Supports TUDataset and GNNBenchmarkDataset as examples.
    """
    if dataset_name.startswith("TUDataset"):
        dataset = TUDataset(root=f"./data/{dataset_name}", name=dataset_name.split(":")[1],
                             use_edge_attr=True, use_node_attr=True, transform=UnifyTUFormat(num_classes=tu_classes_lookup[dataset_name.split(":")[1]]))
        dataset = dataset.shuffle()
        

    elif dataset_name.startswith("GNNBenchmark"):
        dataset = GNNBenchmarkDataset(root=f"./data/{dataset_name}", name=dataset_name.split(":")[1])
    else:
        dataset = None
    return dataset


def evaluate_main(dataset="ogbg-molclintox",
         layer_type="gin", 
         hidden_dim=100,
         num_layers=3,
         batch_size=128,
         epochs=25,
         lr=0.001, 
         t_structure=0., 
         t_feature=0.,
         linear = False,
         pos_encodings = False):

    # Load dataset
    if dataset.startswith("ogbn"):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=dataset)


        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        test_dataset = dataset[split_idx["test"]]
    elif dataset.startswith("ogbg"):
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=dataset)
    

        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        test_dataset = dataset[split_idx["test"]]

    elif dataset.startswith("TUDataset") or dataset.startswith("GNNBenchmark"):
        dataset = load_tu_dataset(dataset)
        print(f"Loaded: {dataset}")

        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]


    elif dataset.startswith("synth"):
        if dataset.endswith("feature") or dataset.endswith("structure"):
            dataset = SyntheticDataset(root="data/synthetic", label_type = dataset)
        else:
            dataset = SyntheticDouble(root="data/synthetic", label_type=dataset)

        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


    # Train and evaluate
    if torch.cuda.is_available():
        dev_string = "cuda"
    elif torch.backends.mps.is_available():
        dev_string = "mps"
    else:
        dev_string = "cpu"

    device = torch.device(dev_string)
    if not linear:
        print(f"Running {dataset} with noise levels: {t_structure}, {t_feature}")
        score, task_type = train_and_evaluate(
            dataset=train_dataset,
            test_dataset=test_dataset,
            layer_type=layer_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            t_structure=t_structure,
            t_feature=t_feature,
            device=device,
            pos_encodings=pos_encodings
        )
    else:
        print("Training linear model")
        score, task_type = train_and_evaluate_linear(
            dataset=train_dataset,
            test_dataset=test_dataset,
            layer_type=layer_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            t_structure=t_structure,
            t_feature=t_feature,
            device=device
        )

    return score, task_type


if __name__ == "__main__":
    evaluate_main()
