"""
Sensitivity analysis module for NNRD metric consistency and invariance testing.

This module provides tools to evaluate how the NNRD metric behaves across:
- Feature dimensionality changes
- Graph size (node count) changes
- Graph density variations

Usage:
    python -m sensitivity.sensitivity_analysis --test all --label_type both
"""

from .synthetic_generators import (
    ConfigurableSyntheticDataset,
    create_sensitivity_dataset,
)
from .sensitivity_evaluation import (
    SensitivityConfig,
    run_noise_sweep,
    evaluate_feature_dimensionality,
    evaluate_graph_size,
    evaluate_graph_density,
)
from .sensitivity_plotting import (
    plot_feature_dim_results,
    plot_graph_size_results,
    plot_density_results,
    plot_combined_summary,
    generate_sensitivity_report,
)
