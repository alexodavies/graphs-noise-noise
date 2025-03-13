import argparse
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import numpy as np
import torch.nn.functional as F
import copy
import os
from time import time
from tqdm import tqdm
from scipy.special import softmax as scipy_softmax
import wandb

# Import FlexibleGNN from a separate file
from model import FlexibleGNN, FeatureExtractorGNN
from noisenoise import add_noise_to_dataset
from synthetic_datasets import SyntheticDataset, SyntheticDouble
from sklearn.linear_model import LogisticRegression, LinearRegression
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

from models.graphormer import Graphormer
from models.graphormer_dataset import create_dataloader_with_paths

tu_classes_lookup: dict = {"ENZYMES": 6,
                           "MUTAG": 2,
                           "PROTEINS": 2,
                           "COLLAB": 3,
                           "IMDB-BINARY": 2,
                           "REDDIT-BINARY": 2}


def add_pe_to_dataset(dataset, pe_dim, walk_length=20, attr_name='pe'):
    """
    Adds positional encodings to all graphs in a PyTorch Geometric dataset.
    
    This function handles different types of datasets, including those that don't
    support item assignment (like PygGraphPropPredDataset).
    
    Args:
        dataset (Dataset): PyTorch Geometric dataset containing the graphs.
        pe_dim (int): Dimension of the positional encoding.
        walk_length (int): Length of the random walks to compute the positional encoding.
        attr_name (str): Attribute name to store the positional encodings in the Data object.
    
    Returns:
        list: A list of Data objects with added positional encodings.
    """
    from torch_geometric.data import Data
    RWPE = AddRandomWalkPE(walk_length, attr_name=attr_name)
    
    # Create a new list with processed data
    new_data_list = []
    
    print(f"Adding positional encodings to dataset of type {type(dataset).__name__}")
    for i, data in enumerate(dataset):
        if i == 0 or i % 1000 == 0:
            print(f"Processing graph {i}...")
        
        # Create a copy to avoid modifying the original
        if isinstance(data, Data):
            # For Data objects, we can directly apply the transform
            new_data = RWPE.forward(data)
        else:
            # For other types of data, try to convert to Data first
            try:
                data_obj = Data()
                for key, value in data.__dict__.items():
                    if not key.startswith('_'):
                        data_obj[key] = value
                new_data = RWPE.forward(data_obj)
            except Exception as e:
                print(f"Warning: Could not process graph {i}: {str(e)}")
                # If we can't process, just add the original
                new_data = data
        
        new_data_list.append(new_data)
    
    print(f"Added positional encodings to {len(new_data_list)} graphs")
    return new_data_list


def infer_task_type(dataset):
    """Infer task type and task level from dataset."""
    # Task level: node or graph
    is_graph_level = hasattr(dataset[0], "y") and dataset[0].y.dim() > 0
    task_level = "graph" if is_graph_level else "node"
    
    # Task type: classification or regression
    try:
        task_type = dataset.task_type
    except:
        if len(dataset[0].y.shape) == 2:
            return task_level, "classification"
        else:
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
        # data.x, data.edge_index, data.edge_attr, data.batch)
        out = model(data)

        task_losses = []
        if len(data.y.shape) == 1:  # Single task
            data.y = data.y.reshape(-1, 1)  # Add task dimension

        if task_type == "multiclass-classification":
            task_loss = F.cross_entropy(out, data.y.argmax(dim=-1))
            task_losses.append(task_loss)

        else:
            for task_idx in range(data.y.shape[1]):  # Loop over tasks
                # Mask for valid labels in this task
                valid_mask = ~torch.isnan(data.y[:, task_idx])
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # Gradient clipping
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
                data.y = data.y.reshape(-1, 1)  # Add task dimension
            # data.x, data.edge_index, data.edge_attr, data.batch)
            out = model(data)

            if task_type == "multiclass-classification":
                # torch.argmax(out, dim=-1).cpu().numpy()
                preds = F.softmax(out, dim=-1).cpu().numpy()
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
                labels = torch.argmax(data.y, dim=-1).cpu().numpy()

                unique_labels = np.unique(labels)
                num_classes = preds.shape[1]
                # Check if all classes are present
                if len(unique_labels) < num_classes:
                    # Filter out columns of `preds` that don't have corresponding labels
                    preds = preds[:, unique_labels]
                    preds = scipy_softmax(preds, axis = -1)
                    
                    # Convert labels to indices within the batch's unique labels
                    label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    labels = np.array([label_map[label] for label in labels])


                score = roc_auc_score(labels, preds, multi_class="ovo")
                task_scores.append(score)
                task_scores.append(score)
            else:
                for task_idx in range(data.y.shape[1]):  # Loop over tasks
                    # Mask for valid labels in this task
                    valid_mask = ~torch.isnan(data.y[:, task_idx])
                    if valid_mask.sum() > 0:  # Only compute metric if there are valid labels
                        if task_type == "classification":
                            preds = torch.sigmoid(
                                out[valid_mask, task_idx]).cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            # Avoid invalid ROC-AUC computation
                            if np.unique(labels).size > 1:
                                score = roc_auc_score(labels, preds)
                                task_scores.append(score)
                        elif task_type == "regression":
                            preds = out[valid_mask, task_idx].cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            score = root_mean_squared_error(
                                labels, preds)  # RMSE
                            task_scores.append(score)
    mean_score = np.mean(task_scores)
    # Average metric across tasks
    return mean_score


def get_model_save_path(dataset_name, layer_type, hidden_dim, num_layers, pos_encodings):
    """Generate a unique path for saving/loading models based on parameters"""
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # For datasets that might contain slashes or special characters, clean the name
    clean_dataset_name = str(dataset_name).replace('/', '_').replace(':', '_')
    
    pe_suffix = "_pe" if pos_encodings else ""
    return f"saved_models/{clean_dataset_name}_{layer_type}_{hidden_dim}_{num_layers}{pe_suffix}.pt"


def train_and_save_model(dataset,
                         model_save_path,
                         layer_type,
                         hidden_dim,
                         num_layers,
                         batch_size,
                         epochs,
                         lr,
                         device,
                         pos_encodings=False,
                         pos_dim=20):
    """
    Train a model on clean data and save it.
    
    Args:
        dataset: PyTorch Geometric dataset for training.
        model_save_path: Path to save the trained model.
        layer_type, hidden_dim, num_layers, batch_size, epochs, lr: Model parameters.
        device: Device to train on.
        pos_encodings: Whether to use positional encodings.
        pos_dim: Dimension of positional encodings.
        
    Returns:
        Trained model and task type.
    """
    # Infer task level and type
    task_level, task_type = infer_task_type(dataset)
    pe_original_dim = 20
    
    # Use clean train dataset
    train_dataset = copy.deepcopy(dataset)
    
    # Handle positional encodings
    if pos_encodings:
        pe_dim = int(0.2 * hidden_dim)
        print(f"Adding positional encodings with dimension {pe_dim}")
        # Get list of data objects with positional encodings
        train_data_list = add_pe_to_dataset(
            train_dataset, pe_original_dim, attr_name='pe')
        
        # Create loader from the list
        if layer_type == "graphormer":
            print("Creating graphormer loader")
            train_loader = create_dataloader_with_paths(train_data_list, batch_size=batch_size)
        else:
            train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    else:
        pe_dim = 0
        # Use original dataset
        if layer_type == "graphormer":
            print("Creating graphormer loader")
            train_loader = create_dataloader_with_paths(train_dataset, batch_size=batch_size)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get dataset dimensions
    node_in_dim = dataset.num_node_features
    edge_in_dim = dataset.num_edge_features if hasattr(
        dataset, "num_edge_features") else 0
    num_classes = dataset[0].y.shape[-1] if task_type == "classification" or task_type == "multiclass-classification" else 1

    if layer_type != "graphormer":
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
            # Type of attention ("multihead" or "performer").
            "attn_type": "multihead",
            "attn_kwargs": {
                "dropout": 0.5          # Dropout for attention.
            },
            "dropout": 0.2,              # Dropout in message-passing layers.
            "act": "relu",               # Activation function for GPS layers.
            "norm": "batch_norm",        # Normalization method for GPS layers.
        }

        kwarg_lookup = {"gin": gin_model_kwargs, "gcn": gcn_model_kwargs,
                        "gat": gat_model_kwargs, "gps": gps_model_kwargs}

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

    else:
        num_heads = 8
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            new_hidden_dim = hidden_dim + (num_heads - hidden_dim % num_heads)
            hidden_dim = new_hidden_dim  # Adjust hidden_dim to nearest valid value

        model = Graphormer(
            in_channels = node_in_dim,
            hidden_channels=hidden_dim,
            out_channels=num_classes,
            num_layers=num_layers,
            num_heads = 8
        ).to(device)

    wandb.watch(model)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(1, epochs + 1), leave=False):
        train_loss = train(model, optimizer, train_loader, device, task_type)
        wandb.log({"Train Loss": train_loss})
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'task_type': task_type,
        'task_level': task_level,
        'node_in_dim': node_in_dim,
        'edge_in_dim': edge_in_dim,
        'num_classes': num_classes,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'pos_encodings': pos_encodings,
        'pe_dim': pe_dim
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    
    return model, task_type


def load_model(model_save_path, layer_type, device):
    """
    Load a previously trained model from disk.
    
    Args:
        model_save_path: Path to the saved model.
        layer_type: The type of GNN layer.
        device: Device to load the model on.
        
    Returns:
        Loaded model and task type.
    """
    try:
        print(f"Attempting to load model from {model_save_path}")
        checkpoint = torch.load(model_save_path, map_location=device)
        
        task_type = checkpoint['task_type']
        task_level = checkpoint['task_level']
        node_in_dim = checkpoint['node_in_dim']
        edge_in_dim = checkpoint['edge_in_dim']
        num_classes = checkpoint['num_classes']
        hidden_dim = checkpoint['hidden_dim']
        num_layers = checkpoint.get('num_layers', 3)  # Default to 3 if not in checkpoint
        pos_encodings = checkpoint.get('pos_encodings', False)
        pe_dim = checkpoint.get('pe_dim', 0)
        
        print(f"Loaded model parameters: hidden_dim={hidden_dim}, num_layers={num_layers}, "
              f"pos_encodings={pos_encodings}, pe_dim={pe_dim}, task_type={task_type}")
        
        if layer_type != "graphormer":
            gin_model_kwargs = {
                "eps": 0,
                "train_eps": True,
            }
            gcn_model_kwargs = {
                "add_self_loops": True,
                "normalize": True,
            }
            gat_model_kwargs = {
                "heads": 4,
                "concat": True,
                "negative_slope": 0.2,
                "dropout": 0.6,
            }
            gps_model_kwargs = {
                "heads": 4,
                "attn_type": "multihead",
                "attn_kwargs": {
                    "dropout": 0.5
                },
                "dropout": 0.2,
                "act": "relu",
                "norm": "batch_norm",
            }

            kwarg_lookup = {"gin": gin_model_kwargs, "gcn": gcn_model_kwargs,
                            "gat": gat_model_kwargs, "gps": gps_model_kwargs}

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

        else:
            num_heads = 8
            model = Graphormer(
                in_channels=node_in_dim,
                hidden_channels=hidden_dim,
                out_channels=num_classes,
                num_layers=num_layers,
                num_heads=8
            ).to(device)
        
        # Check if our model structure matches the saved state dict
        model_state = model.state_dict()
        checkpoint_state = checkpoint['model_state_dict']
        
        # Make sure the keys match
        missing_keys = set(model_state.keys()) - set(checkpoint_state.keys())
        extra_keys = set(checkpoint_state.keys()) - set(model_state.keys())
        
        if missing_keys:
            print(f"Warning: Model is missing keys that are in the checkpoint: {missing_keys}")
        if extra_keys:
            print(f"Warning: Checkpoint has extra keys not in the model: {extra_keys}")
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model state loaded successfully")
        
        return model, task_type
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise e


def train_and_evaluate_with_noise_levels(dataset,
                                        test_dataset,
                                        layer_type,
                                        hidden_dim,
                                        num_layers,
                                        batch_size,
                                        epochs,
                                        lr,
                                        noise_levels,
                                        device,
                                        pos_encodings=False,
                                        pos_dim=20):
    """
    Train a model on clean data and evaluate it on different noise levels.
    
    Args:
        dataset: Training dataset.
        test_dataset: Test dataset.
        layer_type, hidden_dim, num_layers, batch_size, epochs, lr: Model parameters.
        noise_levels: List of (t_structure, t_feature) pairs to evaluate on.
        device: Device to train on.
        pos_encodings: Whether to use positional encodings.
        pos_dim: Dimension of positional encodings.
        
    Returns:
        Dictionary mapping noise levels to performance scores.
    """
    # Get a consistent name for the dataset
    dataset_name = getattr(dataset, 'name', 'dataset')
    
    # Generate the model save path
    model_save_path = get_model_save_path(dataset_name,
                                         layer_type, hidden_dim, num_layers, pos_encodings)
    
    print(f"Looking for model at path: {model_save_path}")
    
    # Check if model already exists
    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}")
        try:
            model, task_type = load_model(model_save_path, layer_type, device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Training new model instead")
            # Train and save model on clean data
            model, task_type = train_and_save_model(
                dataset=dataset,
                model_save_path=model_save_path,
                layer_type=layer_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                pos_encodings=pos_encodings,
                pos_dim=pos_dim
            )
    else:
        print(f"Model not found at {model_save_path}, training new model")
        # Train and save model on clean data
        model, task_type = train_and_save_model(
            dataset=dataset,
            model_save_path=model_save_path,
            layer_type=layer_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            pos_encodings=pos_encodings,
            pos_dim=pos_dim
        )
    
    # Evaluate on different noise levels
    results = {}
    for t_structure, t_feature in noise_levels:
        test_performance = evaluate_model_with_noise(
            model=model,
            test_dataset=test_dataset,
            t_structure=t_structure,
            t_feature=t_feature,
            batch_size=batch_size,
            device=device,
            task_type=task_type,
            layer_type=layer_type,
            pos_encodings=pos_encodings,
            pe_dim=pos_dim
        )
        
        noise_key = f"s{t_structure}_f{t_feature}"
        results[noise_key] = test_performance
        metric = "ROC-AUC" if task_type == "classification" or task_type == "multiclass-classification" else "RMSE"
        print(f"Noise level (s={t_structure}, f={t_feature}): {metric}={test_performance:.4f}")
        
        # Log to wandb
        wandb.log({
            "Structure Noise": t_structure,
            "Feature Noise": t_feature,
            f"Test {metric}": test_performance
        })
    
    return results, task_type


def evaluate_model_with_noise(model, 
                             test_dataset,
                             t_structure, 
                             t_feature,
                             batch_size,
                             device,
                             task_type,
                             layer_type,
                             pos_encodings=False,
                             pe_dim=20):
    """
    Evaluate a model on a test dataset with added noise.
    
    Args:
        model: The trained model to evaluate.
        test_dataset: The clean test dataset to add noise to.
        t_structure, t_feature: Noise levels.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.
        task_type: Task type (classification, regression, etc.).
        layer_type: Type of GNN layer.
        pos_encodings: Whether to use positional encodings.
        pe_dim: Dimension of positional encodings.
        
    Returns:
        Evaluation score.
    """
    # Add noise to test dataset
    noisy_test_dataset = add_noise_to_dataset(
        copy.deepcopy(test_dataset), t_structure, t_feature)
    
    # Handle positional encodings
    if pos_encodings:
        pe_original_dim = 20
        print(f"Adding positional encodings to test dataset")
        # Get list of data objects with positional encodings
        noisy_test_data_list = add_pe_to_dataset(
            noisy_test_dataset, pe_original_dim, attr_name='pe')
        
        # Create loader from the list
        if layer_type == "graphormer":
            noisy_test_loader = create_dataloader_with_paths(noisy_test_data_list, batch_size=batch_size)
        else:
            noisy_test_loader = DataLoader(noisy_test_data_list, batch_size=batch_size, shuffle=False)
    else:
        # Use original noisy dataset without positional encodings
        if layer_type == "graphormer":
            noisy_test_loader = create_dataloader_with_paths(noisy_test_dataset, batch_size=batch_size)
        else:
            noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    test_performance = evaluate(model, noisy_test_loader, device, task_type)
    
    return test_performance


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
        assert isinstance(
            data, Data), "Input must be a torch_geometric.data.Data object"
        # Example modification: Add a feature column to node features
        if data.x is not None:
            pass
        else:
            # Create features if none exist
            data.x = torch.ones(data.num_nodes, 1)

        # Example modification: Add a feature column to node features
        if data.edge_attr is not None:
            pass
        else:
            # Create features if none exist
            data.edge_attr = torch.ones(data.num_edges, 1)

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
        dataset = GNNBenchmarkDataset(
            root=f"./data/{dataset_name}", name=dataset_name.split(":")[1])
    else:
        dataset = None
    return dataset


def evaluate_main_fixed_train(args,
                           t_feature=0,
                           t_structure=0,
                           eval_on_val=False):
    """
    Main function to train on clean data and evaluate with specified noise levels.
    Maintains the same argument structure as the original evaluate_main function.
    
    Args:
        args: Command-line arguments.
        t_feature: Feature noise level.
        t_structure: Structure noise level.
        eval_on_val: Whether to evaluate on validation set instead of test set.
    """
    dataset = args.dataset
    layer_type = args.layer_type
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    linear = args.use_linear
    pos_encodings = args.structure
    pos_dim = args.pos_dim
    avoid_cuda = args.no_cuda
    fixed_test = args.fixed_test
    
    # Load dataset
    if dataset.startswith("ogbn"):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=dataset)

        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["val"]]
        test_dataset = dataset[split_idx["test"]]

    elif dataset.startswith("ogbg"):
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=dataset)

        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]

    elif dataset.startswith("TUDataset") or dataset.startswith("GNNBenchmark"):
        dataset = load_tu_dataset(dataset)

        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]

    elif dataset.startswith("synth"):
        if dataset.endswith("feature") or dataset.endswith("structure"):
            dataset = SyntheticDataset(
                root="data/synthetic", label_type=dataset)
        else:
            dataset = SyntheticDouble(
                root="data/synthetic", label_type=dataset)

        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Set device
    if torch.cuda.is_available() and not avoid_cuda:
        dev_string = "cuda"
    else:
        dev_string = "cpu"

    device = torch.device(dev_string)
    
    # Set a deterministic dataset name for model save path
    dataset_name = dataset
    if hasattr(dataset, 'name'):
        dataset_name = dataset.name
    
    # Get a consistent model save path regardless of where it's called from
    model_save_path = get_model_save_path(
        dataset_name,
        layer_type, 
        hidden_dim, 
        num_layers, 
        pos_encodings
    )
    
    print(f"Looking for model at path: {model_save_path}")
    
    if not linear:
        if os.path.exists(model_save_path):
            print(f"Loading existing model from {model_save_path}")
            try:
                model, task_type = load_model(model_save_path, layer_type, device)
                print(f"Successfully loaded model from {model_save_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Training new model instead")
                model, task_type = train_and_save_model(
                    dataset=train_dataset,
                    model_save_path=model_save_path,
                    layer_type=layer_type,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    epochs=epochs,
                    lr=lr,
                    device=device,
                    pos_encodings=pos_encodings,
                    pos_dim=pos_dim
                )
        else:
            print(f"Model not found at {model_save_path}, training new model")
            # Train and save model on clean data
            model, task_type = train_and_save_model(
                dataset=train_dataset,
                model_save_path=model_save_path,
                layer_type=layer_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                pos_encodings=pos_encodings,
                pos_dim=pos_dim
            )
        
        # Evaluate with specified noise levels
        evaluation_dataset = val_dataset if eval_on_val else test_dataset
        
        score = evaluate_model_with_noise(
            model=model,
            test_dataset=evaluation_dataset,
            t_structure=t_structure,
            t_feature=t_feature,
            batch_size=batch_size,
            device=device,
            task_type=task_type,
            layer_type=layer_type,
            pos_encodings=pos_encodings,
            pe_dim=pos_dim
        )
        
        # Log to wandb
        wandb.log({
            "Structure Noise": t_structure,
            "Feature Noise": t_feature,
            "Test Score": score
        })
    else:
        print("Linear model evaluation is not implemented yet. Please use the original linear model code.")
        return 0.0, "classification"
    
    return score, task_type