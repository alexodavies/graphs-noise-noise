"""
Visualization and reporting module for NNRD sensitivity analysis.

Provides plotting functions and statistical report generation for
consistency and invariance testing results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional
import os


def plot_feature_dim_results(
    results: Dict,
    save_path: Optional[str] = None,
    label_type: str = "unknown"
):
    """
    Plot NNRD vs feature dimensionality.

    Parameters:
        results: Dict from evaluate_feature_dimensionality().
        save_path: Path to save figure.
        label_type: Label type for title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Keys may be strings (from JSON) or ints
    dims = sorted([int(k) for k in results.keys()])
    means = [results[str(d)]["nnrd_mean"] for d in dims]
    stds = [results[str(d)]["nnrd_std"] for d in dims]

    ax.errorbar(dims, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='steelblue')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Feature Dimensionality", fontsize=12)
    ax.set_ylabel("NNRD", fontsize=12)
    ax.set_title(f"NNRD vs Feature Dimensionality\n(label_type={label_type})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add consistency annotation
    cv = np.std(means) / np.abs(np.mean(means)) if np.mean(means) != 0 else np.inf
    ax.text(0.95, 0.05, f"CV = {cv:.3f}", transform=ax.transAxes,
            ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_graph_size_results(
    results: Dict,
    save_path: Optional[str] = None,
    label_type: str = "unknown"
):
    """
    Plot NNRD vs graph size.

    Parameters:
        results: Dict from evaluate_graph_size().
        save_path: Path to save figure.
        label_type: Label type for title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Sort by midpoint of range
    size_keys = sorted(results.keys(), key=lambda x: (results[x]["min_nodes"] + results[x]["max_nodes"]) / 2)

    x_labels = [f"{results[k]['min_nodes']}-{results[k]['max_nodes']}" for k in size_keys]
    x_pos = range(len(size_keys))
    means = [results[k]["nnrd_mean"] for k in size_keys]
    stds = [results[k]["nnrd_std"] for k in size_keys]

    ax.errorbar(x_pos, means, yerr=stds, marker='s', capsize=5,
                linewidth=2, markersize=8, color='darkorange')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel("Node Count Range", fontsize=12)
    ax.set_ylabel("NNRD", fontsize=12)
    ax.set_title(f"NNRD vs Graph Size\n(label_type={label_type})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add consistency annotation
    cv = np.std(means) / np.abs(np.mean(means)) if np.mean(means) != 0 else np.inf
    ax.text(0.95, 0.05, f"CV = {cv:.3f}", transform=ax.transAxes,
            ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_density_results(
    results: Dict,
    save_path: Optional[str] = None,
    label_type: str = "unknown"
):
    """
    Plot NNRD vs graph density.

    Parameters:
        results: Dict from evaluate_graph_density().
        save_path: Path to save figure.
        label_type: Label type for title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Keys may be strings (from JSON) or floats
    densities = sorted([float(k) for k in results.keys()])
    means = [results[str(d) if str(d) in results else d]["nnrd_mean"] for d in densities]
    stds = [results[str(d) if str(d) in results else d]["nnrd_std"] for d in densities]

    ax.errorbar(densities, means, yerr=stds, marker='^', capsize=5,
                linewidth=2, markersize=8, color='seagreen')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Edge Density", fontsize=12)
    ax.set_ylabel("NNRD", fontsize=12)
    ax.set_title(f"NNRD vs Graph Density\n(label_type={label_type})", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add consistency annotation
    cv = np.std(means) / np.abs(np.mean(means)) if np.mean(means) != 0 else np.inf
    ax.text(0.95, 0.05, f"CV = {cv:.3f}", transform=ax.transAxes,
            ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def plot_combined_summary(
    all_results: Dict,
    save_path: Optional[str] = None
):
    """
    Create a combined summary plot for all sensitivity tests.

    Parameters:
        all_results: Dict with structure {label_type: {test_type: results}}.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    colors = {"structure": "steelblue", "feature": "darkorange"}
    markers = {"structure": "o", "feature": "s"}

    for row, label_type in enumerate(["structure", "feature"]):
        if label_type not in all_results:
            continue

        results = all_results[label_type]

        # Feature dimensionality
        if "feature_dim" in results:
            ax = axes[row, 0]
            dims = sorted([int(k) for k in results["feature_dim"].keys()])
            means = [results["feature_dim"][str(d)]["nnrd_mean"] for d in dims]
            stds = [results["feature_dim"][str(d)]["nnrd_std"] for d in dims]

            ax.errorbar(dims, means, yerr=stds, marker=markers[label_type],
                        capsize=4, linewidth=2, markersize=6, color=colors[label_type])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Feature Dim")
            ax.set_ylabel("NNRD")
            ax.set_title(f"Feature Dim ({label_type})")
            ax.grid(True, alpha=0.3)

        # Graph size
        if "graph_size" in results:
            ax = axes[row, 1]
            size_keys = sorted(results["graph_size"].keys(),
                               key=lambda x: results["graph_size"][x]["min_nodes"])
            x_labels = [k for k in size_keys]
            x_pos = range(len(size_keys))
            means = [results["graph_size"][k]["nnrd_mean"] for k in size_keys]
            stds = [results["graph_size"][k]["nnrd_std"] for k in size_keys]

            ax.errorbar(x_pos, means, yerr=stds, marker=markers[label_type],
                        capsize=4, linewidth=2, markersize=6, color=colors[label_type])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel("Node Range")
            ax.set_ylabel("NNRD")
            ax.set_title(f"Graph Size ({label_type})")
            ax.grid(True, alpha=0.3)

        # Density
        if "density" in results:
            ax = axes[row, 2]
            density_data = results["density"]
            densities = sorted([float(k) for k in density_data.keys()])
            means = [density_data[str(d) if str(d) in density_data else d]["nnrd_mean"] for d in densities]
            stds = [density_data[str(d) if str(d) in density_data else d]["nnrd_std"] for d in densities]

            ax.errorbar(densities, means, yerr=stds, marker=markers[label_type],
                        capsize=4, linewidth=2, markersize=6, color=colors[label_type])
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Density")
            ax.set_ylabel("NNRD")
            ax.set_title(f"Density ({label_type})")
            ax.grid(True, alpha=0.3)

    plt.suptitle("NNRD Sensitivity Analysis Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def compute_consistency_metrics(results: Dict, variable_name: str) -> Dict:
    """
    Compute consistency metrics for a sensitivity test.

    Returns:
        Dict with CV, range, sign_consistency, correlation statistics,
        and formal hypothesis test results.
    """
    if not results:
        return {}

    keys = list(results.keys())
    means = [results[k]["nnrd_mean"] for k in keys]
    stds = [results[k]["nnrd_std"] for k in keys]

    # Collect all individual NNRD scores across all parameter values
    all_scores = []
    scores_by_param = []
    for k in keys:
        scores = results[k]["nnrd_scores"]
        all_scores.extend(scores)
        scores_by_param.append(scores)

    # Coefficient of variation (across different parameter values)
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    cv = overall_std / np.abs(overall_mean) if overall_mean != 0 else np.inf

    # Range
    nnrd_range = np.max(means) - np.min(means)

    # Sign consistency (all same sign?)
    signs = [np.sign(m) for m in means if m != 0]
    sign_consistent = len(set(signs)) <= 1 if signs else True

    # Correlation with parameter (for numeric keys)
    try:
        numeric_keys = [float(k) if not isinstance(k, (int, float)) else k for k in keys]
        if all(isinstance(k, (int, float)) for k in numeric_keys):
            correlation, corr_p_value = stats.spearmanr(numeric_keys, means)
        else:
            correlation, corr_p_value = None, None
    except:
        correlation, corr_p_value = None, None

    # === FORMAL STATISTICAL TESTS ===

    # 1. One-sample t-test: Is NNRD significantly different from zero?
    #    H0: mean(NNRD) = 0
    #    This tests whether the metric detects a real bias
    if len(all_scores) >= 2:
        t_stat_zero, p_value_zero = stats.ttest_1samp(all_scores, 0)
        significant_nonzero = p_value_zero < 0.05
    else:
        t_stat_zero, p_value_zero, significant_nonzero = None, None, None

    # 2. One-way ANOVA / Kruskal-Wallis: Does NNRD vary with parameter?
    #    H0: All parameter groups have the same mean NNRD
    #    Rejecting H0 means NNRD is sensitive to the nuisance parameter (bad)
    if len(scores_by_param) >= 2 and all(len(s) >= 2 for s in scores_by_param):
        # Use Kruskal-Wallis (non-parametric, fewer assumptions)
        h_stat, p_value_kw = stats.kruskal(*scores_by_param)
        significant_variation = p_value_kw < 0.05
    else:
        h_stat, p_value_kw, significant_variation = None, None, None

    # 3. Levene's test: Is variance homogeneous across parameter values?
    #    H0: All groups have equal variance
    if len(scores_by_param) >= 2 and all(len(s) >= 2 for s in scores_by_param):
        levene_stat, p_value_levene = stats.levene(*scores_by_param)
        homogeneous_variance = p_value_levene >= 0.05
    else:
        levene_stat, p_value_levene, homogeneous_variance = None, None, None

    return {
        "variable": variable_name,
        "n_values_tested": len(keys),
        "n_total_samples": len(all_scores),
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "cv": cv,
        "range": nnrd_range,
        "min_nnrd": np.min(means),
        "max_nnrd": np.max(means),
        "sign_consistent": sign_consistent,
        "avg_within_std": np.mean(stds),
        "correlation": correlation,
        "correlation_p_value": corr_p_value,
        # Formal tests
        "t_stat_vs_zero": t_stat_zero,
        "p_value_vs_zero": p_value_zero,
        "significant_nonzero": significant_nonzero,
        "kruskal_wallis_h": h_stat,
        "p_value_kruskal_wallis": p_value_kw,
        "significant_variation": significant_variation,
        "levene_stat": levene_stat,
        "p_value_levene": p_value_levene,
        "homogeneous_variance": homogeneous_variance,
    }


def generate_sensitivity_report(
    all_results: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a text report summarizing sensitivity analysis results.

    Parameters:
        all_results: Dict with structure {label_type: {test_type: results}}.
        save_path: Path to save report.

    Returns:
        Report as string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("NNRD METRIC SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append("")

    # Compute metrics for each test
    summary_data = []

    for label_type in ["structure", "feature"]:
        if label_type not in all_results:
            continue

        results = all_results[label_type]
        lines.append(f"Label Type: {label_type.upper()}")
        lines.append("")

        for test_name, test_results in results.items():
            metrics = compute_consistency_metrics(test_results, test_name)
            summary_data.append((label_type, test_name, metrics))

            lines.append(f"  {test_name}:")
            lines.append(f"    Values tested: {metrics['n_values_tested']}")
            lines.append(f"    Total samples: {metrics['n_total_samples']}")
            lines.append(f"    Mean NNRD: {metrics['overall_mean']:.4f} ± {metrics['overall_std']:.4f}")
            lines.append(f"    CV (consistency): {metrics['cv']:.4f}")
            lines.append(f"    Range: {metrics['range']:.4f}")
            lines.append(f"    Sign consistent: {'Yes' if metrics['sign_consistent'] else 'NO'}")
            if metrics['correlation'] is not None:
                lines.append(f"    Correlation with param: {metrics['correlation']:.3f} (p={metrics['correlation_p_value']:.4f})")
            lines.append("")

        lines.append("")

    # Statistical tests section
    lines.append("=" * 80)
    lines.append("STATISTICAL HYPOTHESIS TESTS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Test 1: One-sample t-test against zero")
    lines.append("  H0: NNRD = 0 (metric detects no bias)")
    lines.append("  Rejecting H0 confirms the metric detects a real structure/feature bias")
    lines.append("")

    lines.append("Test 2: Kruskal-Wallis test across parameter values")
    lines.append("  H0: NNRD is the same across all parameter values")
    lines.append("  Rejecting H0 means NNRD varies with the nuisance parameter (undesirable)")
    lines.append("")

    lines.append("Test 3: Levene's test for homogeneity of variance")
    lines.append("  H0: Variance is equal across parameter values")
    lines.append("  Rejecting H0 means variance changes with parameter (undesirable)")
    lines.append("")

    for label_type, test_name, metrics in summary_data:
        lines.append(f"{label_type}/{test_name}:")

        # Test vs zero
        if metrics['p_value_vs_zero'] is not None:
            sig_zero = "***" if metrics['p_value_vs_zero'] < 0.001 else \
                       "**" if metrics['p_value_vs_zero'] < 0.01 else \
                       "*" if metrics['p_value_vs_zero'] < 0.05 else ""
            lines.append(f"  t-test vs zero: t={metrics['t_stat_vs_zero']:.3f}, p={metrics['p_value_vs_zero']:.4f} {sig_zero}")
            lines.append(f"    -> NNRD is {'SIGNIFICANTLY' if metrics['significant_nonzero'] else 'NOT significantly'} different from zero")

        # Kruskal-Wallis
        if metrics['p_value_kruskal_wallis'] is not None:
            sig_kw = "***" if metrics['p_value_kruskal_wallis'] < 0.001 else \
                     "**" if metrics['p_value_kruskal_wallis'] < 0.01 else \
                     "*" if metrics['p_value_kruskal_wallis'] < 0.05 else ""
            lines.append(f"  Kruskal-Wallis: H={metrics['kruskal_wallis_h']:.3f}, p={metrics['p_value_kruskal_wallis']:.4f} {sig_kw}")
            if metrics['significant_variation']:
                lines.append(f"    -> NNRD VARIES significantly with {test_name} (metric is sensitive)")
            else:
                lines.append(f"    -> NNRD does NOT vary significantly with {test_name} (metric is robust)")

        # Levene's test
        if metrics['p_value_levene'] is not None:
            sig_lev = "***" if metrics['p_value_levene'] < 0.001 else \
                      "**" if metrics['p_value_levene'] < 0.01 else \
                      "*" if metrics['p_value_levene'] < 0.05 else ""
            lines.append(f"  Levene's test: W={metrics['levene_stat']:.3f}, p={metrics['p_value_levene']:.4f} {sig_lev}")
            lines.append(f"    -> Variance is {'HOMOGENEOUS' if metrics['homogeneous_variance'] else 'HETEROGENEOUS'} across parameter values")

        lines.append("")

    # Invariance assessment
    lines.append("=" * 80)
    lines.append("INVARIANCE ASSESSMENT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Criteria for INVARIANCE (sign preservation):")
    lines.append("  - NNRD sign should remain consistent across parameter changes")
    lines.append("  - Structure-labelled data: expect NNRD > 0")
    lines.append("  - Feature-labelled data: expect NNRD < 0")
    lines.append("")

    lines.append("Criteria for CONSISTENCY (magnitude stability):")
    lines.append("  - CV < 0.3: High consistency")
    lines.append("  - CV 0.3-0.5: Moderate consistency")
    lines.append("  - CV > 0.5: Low consistency")
    lines.append("")

    for label_type, test_name, metrics in summary_data:
        expected_sign = "+" if label_type == "structure" else "-"
        actual_sign = "+" if metrics['overall_mean'] > 0 else "-" if metrics['overall_mean'] < 0 else "0"
        sign_ok = (label_type == "structure" and metrics['overall_mean'] > 0) or \
                  (label_type == "feature" and metrics['overall_mean'] < 0)

        cv = metrics['cv']
        if cv < 0.3:
            consistency = "HIGH"
        elif cv < 0.5:
            consistency = "MODERATE"
        else:
            consistency = "LOW"

        invariance = "PASS" if metrics['sign_consistent'] and sign_ok else "FAIL"

        # Combine with statistical evidence
        stat_support = ""
        if metrics['significant_nonzero'] and not metrics['significant_variation']:
            stat_support = " [statistically supported]"
        elif metrics['significant_variation']:
            stat_support = " [WARNING: significant parameter sensitivity]"

        lines.append(f"{label_type}/{test_name}:")
        lines.append(f"  Invariance: {invariance} (expected={expected_sign}, got={actual_sign}, sign_consistent={metrics['sign_consistent']})")
        lines.append(f"  Consistency: {consistency} (CV={cv:.3f}){stat_support}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("A robust NNRD metric should show:")
    lines.append("  1. Significant difference from zero (t-test p < 0.05)")
    lines.append("  2. NO significant variation with nuisance parameters (Kruskal-Wallis p >= 0.05)")
    lines.append("  3. Homogeneous variance across conditions (Levene p >= 0.05)")
    lines.append("  4. Correct sign for the label type (+ve for structure, -ve for feature)")
    lines.append("  5. Low coefficient of variation (CV < 0.3)")
    lines.append("")
    lines.append("- A robust metric should show:")
    lines.append("  1. Sign invariance: correct direction for all parameter values")
    lines.append("  2. Magnitude consistency: low CV across parameter changes")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")

    return report
