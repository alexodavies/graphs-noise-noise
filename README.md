# Noise-Noise Analysis for Graph Neural Networks

This repository implements **noise-noise analysis**, a method for evaluating how Graph Neural Networks (GNNs) balance information from graph structure versus node/edge features. The implementation is based on the research paper "A Method and a Metric for GNN Reliance on Information from Features and Structure".

## Overview

Noise-noise analysis helps answer a fundamental question in graph learning: *Does my model rely more on graph structure or node features?* By systematically adding noise to either structure or features independently, we can measure this balance using the **Noise-Noise Ratio Difference (NNRD)** metric.

### Key Concepts

- **Graph-less learning**: Model ignores graph structure, relies only on features
- **Feature-less learning**: Model ignores node/edge features, relies only on structure  
- **NNRD**: Bounded metric comparing performance degradation under feature vs. structure noise
  - Negative NNRD → Feature-biased model
  - Positive NNRD → Structure-biased model
  - NNRD ≈ 0 → Balanced reliance

## Installation

```bash
# Clone repository and install dependencies
pip install torch torch-geometric
pip install numpy tqdm wandb pyyaml matplotlib
pip install ogb  # For molecular datasets
```

## Quick Start

### Basic Usage

```bash
python graph-level.py --dataset ogbg-molhiv --layer_type gin --n_noise_levels 10 --n_repeats 5
```

### With Configuration File

```bash
python graph-level.py --config my_config.yaml --dataset ogbg-molclintox
```

## Command Line Arguments

### Dataset and Model
- `--dataset`: Dataset name (e.g., 'ogbg-molhiv', 'TU-Enzymes')
- `--layer_type`: GNN layer ('gcn', 'gin', 'gat', 'gps', 'graphormer')
- `--hidden_dim`: Hidden dimension size (default: 100)
- `--num_layers`: Number of GNN layers (default: 3)

### Training Parameters
- `--epochs`: Training epochs (default: 25)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 256)

### Noise Analysis Settings
- `--n_noise_levels`: Number of noise levels to test (default: 10)
- `--n_repeats`: Repetitions per noise level (default: 5)
- `--fixed-train`: Fix training set, only noise test set
- `--fixed-test`: Fix test set, only noise training set

### Structural Information
- `--structure`: Include positional encodings
- `--pos_dim`: Positional encoding dimension (default: 16)

### Specialized Modes
- `--top-model`: Use pre-trained ToP model
- `--random-noise-pe`: Add random noise as extra features
- `--use_linear`: Use linear models instead of neural networks

## Configuration Files

Create YAML configuration files in the `configs/` directory:

```yaml
# configs/molecular_analysis.yaml
dataset: "ogbg-molhiv"
layer_type: "gin"
hidden_dim: 128
num_layers: 4
epochs: 50
lr: 0.001
n_noise_levels: 15
n_repeats: 10
structure: true
pos_dim: 20
fixed_train: true
```

## Understanding Results

### NNRD Interpretation

```python
# Example NNRD values and their meanings:
NNRD = -0.3  # Strong feature bias - model relies heavily on node features
NNRD = -0.1  # Moderate feature bias
NNRD =  0.0  # Balanced - equal reliance on features and structure
NNRD = +0.1  # Moderate structure bias  
NNRD = +0.3  # Strong structure bias - model relies heavily on graph structure
```

### Output Files

The script generates:
- **WandB logs**: Real-time training metrics and noise analysis results
- **Result plots**: Performance curves showing degradation under noise
- **JSON results**: Detailed numerical results for further analysis

### Performance Curves

Look for these patterns in the output plots:

1. **Feature-biased model**: Performance drops sharply with feature noise, stable with structure noise
2. **Structure-biased model**: Performance drops sharply with structure noise, stable with feature noise  
3. **Balanced model**: Similar performance drops for both noise types

## Supported Datasets

### Molecular Datasets (OGB)
- `ogbg-molhiv`: HIV replication inhibition
- `ogbg-molbace`: BACE enzyme inhibition
- `ogbg-molbbbp`: Blood-brain barrier penetration
- `ogbg-molclintox`: Clinical toxicity
- `ogbg-molsider`: Side effects
- `ogbg-moltox21`: Toxicity across 21 targets

### Graph Classification (TU Datasets)
- `TU-Enzymes`: Enzyme classification
- `TU-Proteins`: Protein classification

### Synthetic Datasets
The code includes synthetic datasets for controlled experiments:
- **Easy**: Both features and structure provide same information
- **Feature**: Only features are informative
- **Structure**: Only structure is informative  
- **Coupled**: Both features and structure needed together

## Example Workflows

### 1. Analyze Model Bias on Molecular Data

```bash
# Test if your model is feature-biased on molecular tasks
python graph-level.py --dataset ogbg-molhiv --layer_type gin --n_noise_levels 15 --n_repeats 8 --structure false

# Then test with positional encodings to see if structure helps
python graph-level.py --dataset ogbg-molhiv --layer_type gin --n_noise_levels 15 --n_repeats 8 --structure true
```

### 2. Compare Different GNN Architectures

```bash
# Compare GCN, GAT, and GIN on the same dataset
for layer in gcn gat gin; do
    python graph-level.py --dataset ogbg-molclintox --layer_type $layer --n_noise_levels 10 --n_repeats 5
done
```

### 3. Validate on Synthetic Data

```bash
# Test on synthetic datasets where ground truth is known
python graph-level.py --dataset synth-feature --layer_type gin --n_noise_levels 10 --n_repeats 5
python graph-level.py --dataset synth-structure --layer_type gin --n_noise_levels 10 --n_repeats 5
```

## Key Research Findings

Based on the original paper, this analysis reveals:

1. **All GNN layers can do graph-less learning** when features are sufficient
2. **Only GIN can do feature-less learning** due to its post-aggregation parametrization
3. **GCN and GAT become feature-less capable** when positional encodings are added
4. **Molecular tasks tend to be feature-biased** across all tested architectures
5. **Positional encodings increase structure reliance** but don't always improve performance

## Extending the Analysis

### Custom Noise Functions

The current implementation uses:
- **Structure noise**: Replace graphs with Erdős-Rényi random graphs
- **Feature noise**: Replace features with random values from same distribution

You can modify these in the respective evaluation functions.

### New Datasets

To add new datasets:
1. Ensure they follow PyTorch Geometric format
2. Add dataset loading logic to the evaluation functions
3. Configure appropriate train/val/test splits

### Different Architectures

The framework supports any PyTorch Geometric model. Add new architectures by:
1. Implementing the model class
2. Adding it to the layer type options
3. Ensuring it follows the standard forward pass interface

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--hidden_dim`
2. **Poor performance**: Check learning rate and number of epochs
3. **Inconsistent results**: Increase `--n_repeats` for more stable statistics

### Performance Tips

- Use `--fixed-train` for faster analysis (only noise test set)
- Reduce `--n_noise_levels` for quicker experiments
- Use smaller models for initial exploration
