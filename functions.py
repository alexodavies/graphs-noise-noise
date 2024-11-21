import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import numpy as np
import torch.nn.functional as F
import copy
from time import time

from model import FlexibleGNN  # Import FlexibleGNN from a separate file
from noisenoise import add_noise_to_dataset


def infer_task_type(dataset):
    """Infer task type and task level from dataset."""
    # Task level: node or graph
    is_graph_level = hasattr(dataset[0], "y") and dataset[0].y.dim() > 0
    task_level = "graph" if is_graph_level else "node"

    # Task type: classification or regression
    task_type = dataset.task_type
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
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        task_losses = []
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

            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

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




def train_and_evaluate(dataset, test_dataset, layer_type, hidden_dim, num_layers, batch_size, epochs, lr, t_structure, t_feature, device):
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

    # Create noisy copies of datasets
    noisy_train_dataset = add_noise_to_dataset(copy.deepcopy(dataset), t_structure, t_feature)
    noisy_train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)

    noisy_test_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), t_structure, t_feature)
    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False)

    # Get dataset dimensions
    node_in_dim = dataset.num_node_features
    edge_in_dim = dataset.num_edge_features if hasattr(dataset, "num_edge_features") else 0
    num_classes = dataset[0].y.shape[-1] if task_type == "classification" else 1

    # Initialize model
    model = FlexibleGNN(
        layer_type=layer_type,
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        task_type=task_level
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, noisy_train_loader, device, task_type)

    # Evaluate on the final epoch
    test_performance = evaluate(model, noisy_test_loader, device, task_type)

    metric = "ROC-AUC" if task_type == "classification" else "RMSE"

    return test_performance


def evaluate_main(dataset="ogbg-molclintox",
         layer_type="gin", 
         hidden_dim=100,
         num_layers=3,
         batch_size=128,
         epochs=25,
         lr=0.001, 
         t_structure=0., 
         t_feature=0.):

    # Load dataset
    if dataset.startswith("ogbn"):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=dataset)
    elif dataset.startswith("ogbg"):
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=dataset)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    test_dataset = dataset[split_idx["test"]]

    # Train and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = train_and_evaluate(
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

    return score


if __name__ == "__main__":
    evaluate_main()
