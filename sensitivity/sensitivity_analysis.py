"""
Sensitivity Analysis for NNRD Metric

This script evaluates the consistency and invariance of the NNRD metric across:
- Feature dimensionality changes
- Graph size (node count) changes
- Graph density variations

Usage:
    python -m sensitivity.sensitivity_analysis --test feature_dim --label_type structure
    python -m sensitivity.sensitivity_analysis --test graph_size --label_type feature
    python -m sensitivity.sensitivity_analysis --test density --label_type structure
    python -m sensitivity.sensitivity_analysis --test all --label_type both
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity.sensitivity_evaluation import (
    SensitivityConfig,
    evaluate_feature_dimensionality,
    evaluate_graph_size,
    evaluate_graph_density,
    save_sensitivity_results,
)
from sensitivity.sensitivity_plotting import (
    plot_feature_dim_results,
    plot_graph_size_results,
    plot_density_results,
    plot_combined_summary,
    generate_sensitivity_report,
)


# Default test parameters
DEFAULT_FEATURE_DIMS = [2, 5, 10, 25, 50]
DEFAULT_SIZE_RANGES = [
    (20, 50),
    (50, 100),
    (100, 200),
    (200, 400),
]
DEFAULT_DENSITIES = [0.05, 0.1, 0.2, 0.3, 0.5]


def _load_existing_results(output_path: str):
    """Load existing results if available."""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            return json.load(f)
    return None


def run_feature_dim_analysis(
    label_type: str,
    config: SensitivityConfig,
    feature_dims: list = None,
    num_samples: int = 1600,
    n_repeats: int = 3,
    output_dir: str = "results/sensitivity"
):
    """Run feature dimensionality sensitivity analysis."""
    if feature_dims is None:
        feature_dims = DEFAULT_FEATURE_DIMS

    output_path = os.path.join(output_dir, f"feature_dim_{label_type}.json")

    # Check for existing results
    existing = _load_existing_results(output_path)
    if existing is not None:
        print(f"\n{'='*60}")
        print(f"SKIPPING Feature Dimensionality Analysis (label_type={label_type})")
        print(f"Results already exist at: {output_path}")
        print(f"{'='*60}\n")
        # Re-generate plot with existing data
        plot_path = os.path.join(output_dir, f"feature_dim_{label_type}.png")
        plot_feature_dim_results(existing, save_path=plot_path, label_type=label_type)
        return existing

    print(f"\n{'='*60}")
    print(f"Running Feature Dimensionality Analysis")
    print(f"Label type: {label_type}")
    print(f"Feature dimensions: {feature_dims}")
    print(f"{'='*60}\n")

    results = evaluate_feature_dimensionality(
        feature_dims=feature_dims,
        label_type=label_type,
        config=config,
        num_samples=num_samples,
        n_repeats=n_repeats,
        verbose=True
    )

    # Save results
    save_sensitivity_results(results, output_path)
    print(f"Results saved to: {output_path}")

    # Plot results
    plot_path = os.path.join(output_dir, f"feature_dim_{label_type}.png")
    plot_feature_dim_results(results, save_path=plot_path, label_type=label_type)

    return results


def run_graph_size_analysis(
    label_type: str,
    config: SensitivityConfig,
    size_ranges: list = None,
    num_samples: int = 1600,
    n_repeats: int = 3,
    output_dir: str = "results/sensitivity"
):
    """Run graph size sensitivity analysis."""
    if size_ranges is None:
        size_ranges = DEFAULT_SIZE_RANGES

    output_path = os.path.join(output_dir, f"graph_size_{label_type}.json")

    # Check for existing results
    existing = _load_existing_results(output_path)
    if existing is not None:
        print(f"\n{'='*60}")
        print(f"SKIPPING Graph Size Analysis (label_type={label_type})")
        print(f"Results already exist at: {output_path}")
        print(f"{'='*60}\n")
        # Re-generate plot with existing data
        plot_path = os.path.join(output_dir, f"graph_size_{label_type}.png")
        plot_graph_size_results(existing, save_path=plot_path, label_type=label_type)
        return existing

    print(f"\n{'='*60}")
    print(f"Running Graph Size Analysis")
    print(f"Label type: {label_type}")
    print(f"Size ranges: {size_ranges}")
    print(f"{'='*60}\n")

    results = evaluate_graph_size(
        size_ranges=size_ranges,
        label_type=label_type,
        config=config,
        num_samples=num_samples,
        n_repeats=n_repeats,
        verbose=True
    )

    # Save results
    save_sensitivity_results(results, output_path)
    print(f"Results saved to: {output_path}")

    # Plot results
    plot_path = os.path.join(output_dir, f"graph_size_{label_type}.png")
    plot_graph_size_results(results, save_path=plot_path, label_type=label_type)

    return results


def run_density_analysis(
    label_type: str,
    config: SensitivityConfig,
    densities: list = None,
    num_samples: int = 1600,
    n_repeats: int = 3,
    output_dir: str = "results/sensitivity"
):
    """Run graph density sensitivity analysis."""
    if densities is None:
        densities = DEFAULT_DENSITIES

    output_path = os.path.join(output_dir, f"density_{label_type}.json")

    # Check for existing results
    existing = _load_existing_results(output_path)
    if existing is not None:
        print(f"\n{'='*60}")
        print(f"SKIPPING Graph Density Analysis (label_type={label_type})")
        print(f"Results already exist at: {output_path}")
        print(f"{'='*60}\n")
        # Re-generate plot with existing data
        plot_path = os.path.join(output_dir, f"density_{label_type}.png")
        plot_density_results(existing, save_path=plot_path, label_type=label_type)
        return existing

    print(f"\n{'='*60}")
    print(f"Running Graph Density Analysis")
    print(f"Label type: {label_type}")
    print(f"Densities: {densities}")
    print(f"{'='*60}\n")

    results = evaluate_graph_density(
        densities=densities,
        label_type=label_type,
        config=config,
        num_samples=num_samples,
        n_repeats=n_repeats,
        verbose=True
    )

    # Save results
    save_sensitivity_results(results, output_path)
    print(f"Results saved to: {output_path}")

    # Plot results
    plot_path = os.path.join(output_dir, f"density_{label_type}.png")
    plot_density_results(results, save_path=plot_path, label_type=label_type)

    return results


def run_full_analysis(
    config: SensitivityConfig,
    num_samples: int = 1600,
    n_repeats: int = 3,
    output_dir: str = "results/sensitivity"
):
    """Run complete sensitivity analysis for both label types."""
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for label_type in ["structure", "feature"]:
        all_results[label_type] = {}

        # Feature dimensionality
        all_results[label_type]["feature_dim"] = run_feature_dim_analysis(
            label_type=label_type,
            config=config,
            num_samples=num_samples,
            n_repeats=n_repeats,
            output_dir=output_dir
        )

        # Graph size
        all_results[label_type]["graph_size"] = run_graph_size_analysis(
            label_type=label_type,
            config=config,
            num_samples=num_samples,
            n_repeats=n_repeats,
            output_dir=output_dir
        )

        # Density (only for feature-labeled data)
        if label_type == "feature":
            all_results[label_type]["density"] = run_density_analysis(
                label_type=label_type,
                config=config,
                num_samples=num_samples,
                n_repeats=n_repeats,
                output_dir=output_dir
            )

    # Generate combined summary plot
    summary_path = os.path.join(output_dir, "sensitivity_summary.png")
    plot_combined_summary(all_results, save_path=summary_path)

    # Generate report
    report_path = os.path.join(output_dir, "sensitivity_report.txt")
    generate_sensitivity_report(all_results, save_path=report_path)

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "layer_type": config.layer_type,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "epochs": config.epochs,
            "n_noise_levels": config.n_noise_levels,
            "n_repeats": config.n_repeats
        },
        "num_samples": num_samples,
        "analysis_repeats": n_repeats
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="NNRD Metric Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test feature dimensionality for structure-labeled data
    python -m sensitivity.sensitivity_analysis --test feature_dim --label_type structure

    # Test graph size for feature-labeled data
    python -m sensitivity.sensitivity_analysis --test graph_size --label_type feature

    # Test density for structure-labeled data
    python -m sensitivity.sensitivity_analysis --test density --label_type structure

    # Run all tests for both label types
    python -m sensitivity.sensitivity_analysis --test all --label_type both

    # Custom parameters
    python -m sensitivity.sensitivity_analysis --test feature_dim --label_type structure \\
        --feature_dims 2 5 10 20 --num_samples 800 --n_repeats 5
        """
    )

    parser.add_argument(
        "--test",
        type=str,
        choices=["feature_dim", "graph_size", "density", "all"],
        default="all",
        help="Which sensitivity test to run"
    )
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["structure", "feature", "both"],
        default="both",
        help="Label type for synthetic data"
    )

    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=1600,
                        help="Number of samples per dataset")
    parser.add_argument("--n_repeats", type=int, default=3,
                        help="Number of repeats per configuration")

    # Test-specific parameters
    parser.add_argument("--feature_dims", type=int, nargs="+",
                        default=DEFAULT_FEATURE_DIMS,
                        help="Feature dimensions to test")
    parser.add_argument("--densities", type=float, nargs="+",
                        default=DEFAULT_DENSITIES,
                        help="Densities to test")

    # Model parameters
    parser.add_argument("--layer_type", type=str, default="gin",
                        help="GNN layer type")
    parser.add_argument("--hidden_dim", type=int, default=100,
                        help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")

    # Evaluation parameters
    parser.add_argument("--n_noise_levels", type=int, default=11,
                        help="Number of noise levels")
    parser.add_argument("--noise_repeats", type=int, default=5,
                        help="Repeats per noise level")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/sensitivity",
                        help="Output directory")

    args = parser.parse_args()

    # Create config
    config = SensitivityConfig(
        layer_type=args.layer_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        n_noise_levels=args.n_noise_levels,
        n_repeats=args.noise_repeats
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine label types to test
    if args.label_type == "both":
        label_types = ["structure", "feature"]
    else:
        label_types = [args.label_type]

    # Run requested tests
    if args.test == "all":
        run_full_analysis(
            config=config,
            num_samples=args.num_samples,
            n_repeats=args.n_repeats,
            output_dir=args.output_dir
        )
    else:
        for label_type in label_types:
            if args.test == "feature_dim":
                run_feature_dim_analysis(
                    label_type=label_type,
                    config=config,
                    feature_dims=args.feature_dims,
                    num_samples=args.num_samples,
                    n_repeats=args.n_repeats,
                    output_dir=args.output_dir
                )
            elif args.test == "graph_size":
                run_graph_size_analysis(
                    label_type=label_type,
                    config=config,
                    num_samples=args.num_samples,
                    n_repeats=args.n_repeats,
                    output_dir=args.output_dir
                )
            elif args.test == "density":
                if label_type == "structure":
                    print(f"Skipping density test for label_type='structure' "
                          f"(density changes destroy structural signal)")
                    continue
                run_density_analysis(
                    label_type=label_type,
                    config=config,
                    densities=args.densities,
                    num_samples=args.num_samples,
                    n_repeats=args.n_repeats,
                    output_dir=args.output_dir
                )


if __name__ == "__main__":
    main()
