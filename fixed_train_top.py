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

from top.encoder import Encoder, FeaturedTransferModel
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


def create_model(node_in_dim, edge_in_dim, num_classes, device):
    """Create the standard FeaturedTransferModel with hardcoded encoder parameters."""
    # Encoder is completely hardcoded - same for all tasks
    encoder = Encoder(
        emb_dim=300, 
        num_gc_layers=6, 
        drop_ratio=0.2,
        pooling_type="standard", 
        convolution="gin"
    )
    
    # Only the FeaturedTransferModel wrapper adapts to dataset dimensions
    return FeaturedTransferModel(
        encoder=encoder,
        proj_hidden_dim=300, 
        output_dim=num_classes, 
        features=True,
        node_feature_dim=node_in_dim, 
        edge_feature_dim=edge_in_dim,
    ).to(device)


def load_pretrained_weights(model, pretrained_path, device):
    """Load pretrained weights into model if path exists and is not 'untrained'."""
    if not pretrained_path or "untrained" in pretrained_path:
        print("Training from scratch (no pretrained weights)")
        return False
    
    if not os.path.exists(pretrained_path):
        print(f"Pretrained model not found at {pretrained_path}, training from scratch")
        return False
    
    try:
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        
        # Try different checkpoint formats
        if 'encoder_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print("Pretrained weights loaded successfully")
        return True
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {str(e)}")
        print("Training from scratch...")
        return False


def add_pe_to_dataset(dataset, pe_dim, walk_length=20, attr_name='pe'):
    """Adds positional encodings to all graphs in a PyTorch Geometric dataset."""
    from torch_geometric.data import Data
    RWPE = AddRandomWalkPE(walk_length, attr_name=attr_name)
    
    new_data_list = []
    print(f"Adding positional encodings to dataset of type {type(dataset).__name__}")
    
    for i, data in enumerate(dataset):
        if i == 0 or i % 1000 == 0:
            print(f"Processing graph {i}...")
        
        if isinstance(data, Data):
            new_data = RWPE.forward(data)
        else:
            try:
                data_obj = Data()
                for key, value in data.__dict__.items():
                    if not key.startswith('_'):
                        data_obj[key] = value
                new_data = RWPE.forward(data_obj)
            except Exception as e:
                print(f"Warning: Could not process graph {i}: {str(e)}")
                new_data = data
        
        new_data_list.append(new_data)
    
    print(f"Added positional encodings to {len(new_data_list)} graphs")
    return new_data_list


def infer_task_type(dataset):
    """Infer task type and task level from dataset."""
    is_graph_level = hasattr(dataset[0], "y") and dataset[0].y.dim() > 0
    task_level = "graph" if is_graph_level else "node"
    
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
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = data.y.float()

        optimizer.zero_grad()
        out = model(data)[0]

        task_losses = []
        if len(data.y.shape) == 1:
            data.y = data.y.reshape(-1, 1)

        if task_type == "multiclass-classification":
            task_loss = F.cross_entropy(out, data.y.argmax(dim=-1))
            task_losses.append(task_loss)
        else:
            for task_idx in range(data.y.shape[1]):
                valid_mask = ~torch.isnan(data.y[:, task_idx])
                if valid_mask.sum() > 0:
                    if task_type == "classification":
                        task_loss = F.binary_cross_entropy_with_logits(
                            out[valid_mask, task_idx], data.y[valid_mask, task_idx]
                        )
                    elif task_type == "regression":
                        task_loss = F.mse_loss(
                            out[valid_mask, task_idx], data.y[valid_mask, task_idx]
                        )
                    task_losses.append(task_loss)

        if len(task_losses) > 0:
            loss = torch.stack(task_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.y = data.y.float()
            
            if len(data.y.shape) == 1:
                data.y = data.y.reshape(-1, 1)
            
            out = model(data)[0]

            if task_type == "multiclass-classification":
                preds = F.softmax(out, dim=-1).cpu().numpy()
                if preds.shape[1] == 2:
                    preds = preds[:, 1]
                labels = torch.argmax(data.y, dim=-1).cpu().numpy()

                unique_labels = np.unique(labels)
                num_classes = preds.shape[1]
                
                if len(unique_labels) < num_classes:
                    preds = preds[:, unique_labels]
                    preds = scipy_softmax(preds, axis=-1)
                    label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    labels = np.array([label_map[label] for label in labels])

                score = roc_auc_score(labels, preds, multi_class="ovo")
                task_scores.append(score)
            else:
                for task_idx in range(data.y.shape[1]):
                    valid_mask = ~torch.isnan(data.y[:, task_idx])
                    if valid_mask.sum() > 0:
                        if task_type == "classification":
                            preds = torch.sigmoid(out[valid_mask, task_idx]).cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            if np.unique(labels).size > 1:
                                score = roc_auc_score(labels, preds)
                                task_scores.append(score)
                        elif task_type == "regression":
                            preds = out[valid_mask, task_idx].cpu().numpy()
                            labels = data.y[valid_mask, task_idx].cpu().numpy()
                            score = root_mean_squared_error(labels, preds)
                            task_scores.append(score)
    
    return np.mean(task_scores)


def get_model_save_path(dataset_name, top_model, pos_encodings, is_fine_tuned=False):
    """Generate a unique path for saving/loading models based on parameters"""
    os.makedirs("saved_models", exist_ok=True)
    clean_dataset_name = str(dataset_name).replace('/', '_').replace(':', '_')
    pe_suffix = "_pe" if pos_encodings else ""
    fine_tuned_suffix = "_fine_tuned" if is_fine_tuned else ""
    return f"saved_models/{clean_dataset_name}_{top_model}_{pe_suffix}.pt"


def prepare_dataloader(dataset, batch_size, pos_encodings, layer_type, shuffle=True):
    """Prepare dataloader with optional positional encodings."""
    if pos_encodings:
        pe_original_dim = 20
        print(f"Adding positional encodings")
        data_list = add_pe_to_dataset(dataset, pe_original_dim, attr_name='pe')
        
        if layer_type == "graphormer":
            print("Creating graphormer loader")
            for data in data_list:
                data.x = torch.hstack((data.x, data.pe))
            return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
        else:
            return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)
    else:
        if layer_type == "graphormer":
            print("Creating graphormer loader")
            return create_dataloader_with_paths(dataset, batch_size=batch_size)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_and_save_model(dataset, model_save_path, layer_type, hidden_dim, num_layers,
                         batch_size, epochs, lr, device, pos_encodings=False, pos_dim=20,
                         top_model=None):
    """Train a model and save it, always using top_model for initialization if available."""
    task_level, task_type = infer_task_type(dataset)
    
    # Prepare data
    train_loader = prepare_dataloader(dataset, batch_size, pos_encodings, layer_type)
    
    # Get dimensions
    node_in_dim = dataset.num_node_features
    if layer_type == "graphormer" and pos_encodings:
        node_in_dim += int(0.2 * hidden_dim)
    
    edge_in_dim = getattr(dataset, "num_edge_features", 0)
    num_classes = dataset[0].y.shape[-1] if task_type in ["classification", "multiclass-classification"] else 1

    # Create model
    model = create_model(node_in_dim, edge_in_dim, num_classes, device)
    
    # Always try to load top_model weights (unless "untrained" specified)
    if top_model and "untrained" not in top_model:
        top_model_path = f"saved_models/{top_model}.pt"
        load_pretrained_weights(model, top_model_path, device)
    else:
        print("Training from scratch (no top_model specified or 'untrained' flag)")
    
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(1, epochs + 1), leave=False):
        train_loss = train(model, optimizer, train_loader, device, task_type)
        wandb.log({"Train Loss": train_loss})
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'encoder_state_dict': model.state_dict(),
        'model_state_dict': model.state_dict(),
        'task_type': task_type,
        'task_level': task_level,
        'node_in_dim': node_in_dim,
        'edge_in_dim': edge_in_dim,
        'num_classes': num_classes,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'pos_encodings': pos_encodings,
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    return model, task_type


def load_model(model_save_path, device):
    """Load a previously trained model from disk."""
    print(f"Loading model from {model_save_path}")
    checkpoint = torch.load(model_save_path, map_location=device)
    
    # Extract parameters
    task_type = checkpoint.get('task_type', 'classification')
    node_in_dim = checkpoint.get('node_in_dim', 1)
    edge_in_dim = checkpoint.get('edge_in_dim', 1)
    num_classes = checkpoint.get('num_classes', 1)
    
    # Create model
    model = create_model(node_in_dim, edge_in_dim, num_classes, device)
    
    # Load weights
    if 'encoder_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("Model loaded successfully")
    return model, task_type


def evaluate_model_with_noise(model, test_dataset, t_structure, t_feature, batch_size,
                             device, task_type, layer_type, pos_encodings=False, pe_dim=20):
    """Evaluate a model on a test dataset with added noise."""
    # Add noise to test dataset
    noisy_test_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), t_structure, t_feature)
    
    # Prepare dataloader
    noisy_test_loader = prepare_dataloader(noisy_test_dataset, batch_size, pos_encodings, layer_type, shuffle=False)
    
    # Evaluate
    return evaluate(model, noisy_test_loader, device, task_type)


class UnifyTUFormat:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, data):
        assert isinstance(data, Data), "Input must be a torch_geometric.data.Data object"
        
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        if data.edge_attr is None:
            data.edge_attr = torch.ones(data.num_edges, 1)
        
        data.y = F.one_hot(data.y, num_classes=self.num_classes)
        return data


def load_tu_dataset(dataset_name):
    """Load a graph dataset by name."""
    if dataset_name.startswith("TUDataset"):
        name = dataset_name.split(":")[1]
        dataset = TUDataset(
            root=f"./data/{dataset_name}", 
            name=name,
            use_edge_attr=True, 
            use_node_attr=True, 
            transform=UnifyTUFormat(num_classes=tu_classes_lookup[name])
        )
        return dataset.shuffle()
    elif dataset_name.startswith("GNNBenchmark"):
        return GNNBenchmarkDataset(root=f"./data/{dataset_name}", name=dataset_name.split(":")[1])
    else:
        return None


def evaluate_main_top(args, t_feature=0, t_structure=0, eval_on_val=False):
    """Main function to train on clean data and evaluate with specified noise levels."""
    
    # Load dataset
    if args.dataset.startswith("ogbn"):
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=args.dataset)
        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["val"]]
        test_dataset = dataset[split_idx["test"]]
    
    elif args.dataset.startswith("ogbg"):
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name=args.dataset)
        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]
    
    elif args.dataset.startswith("TUDataset") or args.dataset.startswith("GNNBenchmark"):
        dataset = load_tu_dataset(args.dataset)
        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]
    
    elif args.dataset.startswith("synth"):
        if args.dataset.endswith("feature") or args.dataset.endswith("structure"):
            dataset = SyntheticDataset(root="data/synthetic", label_type=args.dataset)
        else:
            dataset = SyntheticDouble(root="data/synthetic", label_type=args.dataset)
        
        split_props = 0.7, 0.2, 0.1
        split_ns = [int(prop * len(dataset)) for prop in split_props]
        train_dataset = dataset[:split_ns[0]]
        val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
        test_dataset = dataset[split_ns[0] + split_ns[1]:]
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Get dataset name for saving
    dataset_name = getattr(dataset, 'name', args.dataset)
    
    # Generate fine-tuned model path
    fine_tuned_model_save_path = get_model_save_path(
        dataset_name, args.top_model,
        args.structure, is_fine_tuned=True
    )
    
    # Use args.top_model directly - no need for separate pretrained_model_path
    
    if not args.use_linear:
        # Check if fine-tuned model exists
        if os.path.exists(fine_tuned_model_save_path):
            print(f"Loading existing fine-tuned model from {fine_tuned_model_save_path}")
            try:
                model, task_type = load_model(fine_tuned_model_save_path, device)
            except Exception as e:
                print(f"Error loading fine-tuned model: {str(e)}")
                print("Training new model instead")
                model, task_type = train_and_save_model(
                    dataset=train_dataset,
                    model_save_path=fine_tuned_model_save_path,
                    layer_type=args.layer_type,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                    pos_encodings=args.structure,
                    pos_dim=args.pos_dim,
                    top_model=args.top_model  # Always pass top_model
                )
        else:
            print(f"Fine-tuned model not found, training new model")
            model, task_type = train_and_save_model(
                dataset=train_dataset,
                model_save_path=fine_tuned_model_save_path,
                layer_type=args.layer_type,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                pos_encodings=args.structure,
                pos_dim=args.pos_dim,
                top_model=args.top_model  # Always pass top_model
            )
        
        # Evaluate with specified noise levels
        evaluation_dataset = val_dataset if eval_on_val else test_dataset
        
        score = evaluate_model_with_noise(
            model=model,
            test_dataset=evaluation_dataset,
            t_structure=t_structure,
            t_feature=t_feature,
            batch_size=args.batch_size,
            device=device,
            task_type=task_type,
            layer_type=args.layer_type,
            pos_encodings=args.structure,
            pe_dim=args.pos_dim
        )
        
        # Log to wandb
        wandb.log({
            "Structure Noise": t_structure,
            "Feature Noise": t_feature,
            "Test Score": score
        })
    else:
        print("Linear model evaluation is not implemented yet.")
        return 0.0, "classification"
    
    return score, task_type