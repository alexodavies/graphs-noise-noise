import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from torch_geometric.loader import DataLoader
import warnings

# Disable wandb for this analysis
os.environ["WANDB_MODE"] = "disabled"

# Ignore future warnings from torch geometric
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import necessary functions from existing codebase
from noisenoise import add_noise_to_dataset
from fixed_train_set import load_model, add_pe_to_dataset, get_model_save_path
from synthetic_datasets import SyntheticDataset, SyntheticDouble

# Custom plot function provided
def plot_results(result_dict, extra_save_string="", return_path=False, default_xticks=False):
    fig, ax = plt.subplots(figsize=(2.75, 2))
    dataset = result_dict["dataset"]
    task_type = result_dict["task_type"]
    ax.set_xlabel("Noise Level")
    perf_string = "Accuracy"  # Changed to use accuracy instead of RMSE/ROC-AUC
    ax.set_ylabel(perf_string)
    # ax.set_title(dataset)
    strucs = result_dict["structure"]
    feats = result_dict["feature"]
    ts = list(strucs.keys())
    # Ensure the x-axis values are numeric and sorted
    ts = sorted([float(t) for t in ts])  # Convert keys to floats and sort them
    struc_means = [np.mean([float(val) for val in strucs[str(t)]]) for t in ts]
    struc_devs = [np.std([float(val) for val in strucs[str(t)]]) for t in ts]
    feat_means = [np.mean([float(val) for val in feats[str(t)]]) for t in ts]
    feat_devs = [np.std([float(val) for val in feats[str(t)]]) for t in ts]
    ax.fill_between(ts, struc_means, feat_means, color="gray", alpha=0.4)
    ax.errorbar(ts, struc_means, yerr=struc_devs, label="Structure", c="black")
    ax.errorbar(ts, feat_means, yerr=feat_devs, label="Feature", c="blue", linestyle="dashed")
    nnrde_y_min = min(struc_means[-1], feat_means[-1])
    nnrde_y_max = max(struc_means[-1], feat_means[-1])
    final_gap = np.abs(struc_means[-1] - feat_means[-1])
    nnrd_y_adjusted = nnrde_y_min + 0.35 * final_gap
    nnrd_root = min(np.min(np.array(struc_means) - np.array(struc_devs)),
                   np.min(np.array(feat_means) - np.array(feat_devs)))
    # ax.text(1.05, nnrd_root, f"$NNRD$:\n{np.around(nnd(result_dict), decimals = 3)}",
    # bbox=dict(facecolor='white', edgecolor='green', boxstyle='round'))
    if not default_xticks:
        # Format x-axis ticks
        t_ticks = np.linspace(0, 1, 6).tolist()
        ax.set_xticks(t_ticks)  # Ensure all unique noise levels are shown
        ax.set_xticklabels([f"{t:.1f}" for t in t_ticks])  # Format as two decimal places
    else:
        ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlim([ts[0] - 0.025, ts[-1] + 0.025])
    extra_string = "" if extra_save_string == "" else f"{extra_save_string}"
    extra_string += "-pos" if result_dict["pos"] else ""
    ax.legend()
    plt.tight_layout()
    
    # Make sure the directory exists
    os.makedirs(f"figures/analysis/{dataset}", exist_ok=True)
    
    plt.savefig(f"figures/analysis/{dataset}/{extra_string}.png", dpi=600)
    plt.close()
    if return_path:
        return f"figures/analysis/{dataset}{extra_string}.png"

def evaluate_sigmoid_values(model, loader, device, task_type):
    """Evaluate the model and return raw sigmoid outputs and true labels."""
    sigmoid_values = []
    true_labels = []
    
    with torch.no_grad():
        model.eval()
        for data in loader:
            data = data.to(device)

            # Ensure features and labels are float
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()
            data.y = data.y.float()
            
            if len(data.y.shape) == 1:  # Single task
                data.y = data.y.reshape(-1, 1)  # Add task dimension
                
            out = model(data)
            
            if task_type == "classification":
                # For binary classification, get raw sigmoid values
                raw_preds = torch.sigmoid(out).cpu().numpy()
                raw_labels = data.y.cpu().numpy()
                
                sigmoid_values.append(raw_preds)
                true_labels.append(raw_labels)
                
    # Concatenate results from all batches
    if sigmoid_values:
        sigmoid_values = np.vstack(sigmoid_values)
        true_labels = np.vstack(true_labels)
        
    return sigmoid_values, true_labels

def calculate_accuracy(sigmoid_values, true_labels):
    """Calculate classification accuracy from sigmoid values."""
    predicted_labels = (sigmoid_values > 0.5).astype(int)
    return np.mean(predicted_labels == true_labels)

def run_analysis():
    """Main function to run the analysis"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Track sigmoid values at different noise levels')
    parser.add_argument('--dataset', type=str, default='synth-structure',
                        help='Dataset to use (default: synth-structure)')
    parser.add_argument('--layer_type', type=str, default='gcn',
                        help='GNN layer type (default: gcn)')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='Hidden dimension (default: 100)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers (default: 3)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--structure', action='store_true',
                        help='Use positional encodings')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--output_dir', type=str, default='sigmoid_analysis',
                        help='Directory to save output files (default: sigmoid_analysis)')
    parser.add_argument('--noise_levels', type=int, default=100,
                        help='Number of noise levels to test (default: 100)')
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of repeats for each noise level (default: 5)')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("figures/analysis", exist_ok=True)

    # Set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset.endswith("feature") or args.dataset.endswith("structure"):
        dataset = SyntheticDataset(root="data/synthetic", label_type=args.dataset)
    else:
        dataset = SyntheticDouble(root="data/synthetic", label_type=args.dataset)

    # Split dataset
    split_props = 0.7, 0.2, 0.1
    split_ns = [int(prop * len(dataset)) for prop in split_props]
    train_dataset = dataset[:split_ns[0]]
    val_dataset = dataset[split_ns[0]:split_ns[0] + split_ns[1]]
    test_dataset = dataset[split_ns[0] + split_ns[1]:]

    # Get model path
    dataset_name = dataset
    if hasattr(dataset, 'name'):
        dataset_name = dataset.name

    model_save_path = get_model_save_path(
        dataset_name,
        args.layer_type, 
        args.hidden_dim, 
        args.num_layers, 
        args.structure
    )

    # Check if model exists
    if not os.path.exists(model_save_path):
        print(f"Model not found at {model_save_path}")
        print("Please train the model first using fixed_train_set.py")
        exit(1)

    # Load model
    print(f"Loading model from {model_save_path}")
    model, task_type = load_model(model_save_path, args.layer_type, device)

    # Define noise levels to test
    feature_noise_levels = np.linspace(0.0, 1.0, args.noise_levels)
    
    # Prepare data structure for results
    result_dict = {
        "dataset": args.dataset,
        "task_type": task_type,  # We'll keep this for consistency even though we're using accuracy
        "pos": args.structure,
        "layer": args.layer_type,
        "structure": {},
        "feature": {}
    }
    
    for t in feature_noise_levels:
        result_dict["structure"][str(t)] = []
        result_dict["feature"][str(t)] = []
    
    # Repeat the experiment multiple times
    for repeat in range(args.repeats):
        print(f"Running repeat {repeat+1}/{args.repeats}")
        
        # For t = 0 (no noise), evaluate once and use for both structure and feature
        t = 0.0
        # Make a copy of test dataset
        clean_test_dataset = copy.deepcopy(test_dataset)
        # Create DataLoader
        if args.structure:
            # Add positional encodings
            pe_original_dim = 20
            test_data_list = add_pe_to_dataset(clean_test_dataset, pe_original_dim, attr_name='pe')
            # Stack positional encodings with features
            for data in test_data_list:
                data.x = torch.hstack((data.x, data.pe))
            test_loader = DataLoader(test_data_list, batch_size=args.batch_size, shuffle=False)
        else:
            test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size, shuffle=False)
            
        # Evaluate
        sigmoid_values, true_labels = evaluate_sigmoid_values(model, test_loader, device, task_type)
        accuracy = calculate_accuracy(sigmoid_values.reshape(-1), true_labels.reshape(-1))
        result_dict["structure"][str(t)].append(str(accuracy))
        result_dict["feature"][str(t)].append(str(accuracy))
        
        # Evaluate for other noise levels
        for t in tqdm(feature_noise_levels[1:], desc="Evaluating noise levels"):
            # Structure noise (t_structure = t, t_feature = 0)
            noisy_structure_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), t, 0.0)
            if args.structure:
                # Add positional encodings
                pe_original_dim = 20
                noisy_structure_data_list = add_pe_to_dataset(noisy_structure_dataset, pe_original_dim, attr_name='pe')
                # Stack positional encodings with features
                for data in noisy_structure_data_list:
                    data.x = torch.hstack((data.x, data.pe))
                noisy_structure_loader = DataLoader(noisy_structure_data_list, batch_size=args.batch_size, shuffle=False)
            else:
                noisy_structure_loader = DataLoader(noisy_structure_dataset, batch_size=args.batch_size, shuffle=False)
                
            # Evaluate
            sigmoid_values, true_labels = evaluate_sigmoid_values(model, noisy_structure_loader, device, task_type)
            accuracy = calculate_accuracy(sigmoid_values.reshape(-1), true_labels.reshape(-1))
            result_dict["structure"][str(t)].append(str(accuracy))
            
            # Feature noise (t_structure = 0, t_feature = t)
            noisy_feature_dataset = add_noise_to_dataset(copy.deepcopy(test_dataset), 0.0, t)
            if args.structure:
                # Add positional encodings
                pe_original_dim = 20
                noisy_feature_data_list = add_pe_to_dataset(noisy_feature_dataset, pe_original_dim, attr_name='pe')
                # Stack positional encodings with features
                for data in noisy_feature_data_list:
                    data.x = torch.hstack((data.x, data.pe))
                noisy_feature_loader = DataLoader(noisy_feature_data_list, batch_size=args.batch_size, shuffle=False)
            else:
                noisy_feature_loader = DataLoader(noisy_feature_dataset, batch_size=args.batch_size, shuffle=False)
                
            # Evaluate
            sigmoid_values, true_labels = evaluate_sigmoid_values(model, noisy_feature_loader, device, task_type)
            accuracy = calculate_accuracy(sigmoid_values.reshape(-1), true_labels.reshape(-1))
            result_dict["feature"][str(t)].append(str(accuracy))
    
    # Save results
    import json
    with open(os.path.join(args.output_dir, f"{args.layer_type}_accuracy_results.json"), 'w') as f:
        json.dump(result_dict, f)
    
    # Plot results
    print("Generating plot...")
    plot_path = plot_results(result_dict, extra_save_string=f"{args.layer_type}_accuracy", return_path=True)
    print(f"Plot saved to {plot_path}")
    
    # Copy plot to output directory
    import shutil
    if os.path.exists(plot_path):
        shutil.copy(plot_path, os.path.join(args.output_dir, f"{args.layer_type}_accuracy_plot.png"))
    else:
        print(f"Warning: Could not find plot at {plot_path}")

if __name__ == "__main__":
    run_analysis()