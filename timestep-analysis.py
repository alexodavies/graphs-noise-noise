import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

class TimestepAnalyzer:
    def __init__(self, data_dir="timestep_analysis/useful/"):
        self.data_dir = data_dir
        self.all_data = {}
        self.summary_df = None
        
    def load_all_files(self):
        """Load all JSON files from the specified directory"""
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.data_dir}")
            return
            
        print(f"Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract key info for filename
                filename = Path(file_path).stem
                dataset = data['metadata'].get('dataset', 'unknown')
                layer_type = data['metadata'].get('layer_type', 'unknown')
                structure = data['metadata'].get('structure', False)
                
                key = f"{dataset}_{layer_type}_struct{structure}"
                self.all_data[key] = data
                print(f"Loaded: {filename} -> {key}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def create_summary_dataframe(self):
        """Create a summary dataframe with all results"""
        rows = []
        
        for key, data in self.all_data.items():
            dataset = data['metadata'].get('dataset', 'unknown')
            layer_type = data['metadata'].get('layer_type', 'unknown')
            structure = data['metadata'].get('structure', False)
            
            for timestep, results in data['results'].items():
                if results['n_successful'] > 0:  # Only include successful runs
                    row = {
                        'dataset': dataset,
                        'layer_type': layer_type,
                        'structure': structure,
                        'experiment': key,
                        'timestep': int(timestep),
                        'mean_nnrd': results['mean'],
                        'std_nnrd': results['std'],
                        'median_nnrd': results['median'],
                        'q25_nnrd': results['q25'],
                        'q75_nnrd': results['q75'],
                        'min_nnrd': results['min'],
                        'max_nnrd': results['max'],
                        'n_successful': results['n_successful'],
                        'individual_scores': results['nnrd_scores']
                    }
                    rows.append(row)
        
        self.summary_df = pd.DataFrame(rows)
        return self.summary_df
    
    def test_correlation_hypothesis(self, alpha=0.05):
        """
        Test hypothesis for correlation between timestep size and NNRD values.
        
        H0: There is NO significant correlation between timestep size and NNRD (rho = 0)
        H1: There IS a significant correlation between timestep size and NNRD (rho != 0)
        
        This is the standard correlation test.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05)
            
        Returns:
        --------
        dict: Results of correlation hypothesis tests for each experiment
        """
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        correlation_results = {}
        
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            timesteps = exp_data['timestep'].values
            mean_nnrd = exp_data['mean_nnrd'].values
            
            # Linear regression to get slope
            slope, intercept, r_value, p_value_regression, std_err = stats.linregress(timesteps, mean_nnrd)
            
            # Pearson correlation test
            pearson_r, pearson_p = stats.pearsonr(timesteps, mean_nnrd)
            
            # Spearman correlation test (non-parametric)
            spearman_r, spearman_p = stats.spearmanr(timesteps, mean_nnrd)
            
            # Kendall's tau (another non-parametric measure)
            kendall_tau, kendall_p = stats.kendalltau(timesteps, mean_nnrd)
            
            # Standard interpretation: reject H0 if p-value <= alpha (significant correlation found)
            
            correlation_results[experiment] = {
                'n_points': len(timesteps),
                'timestep_range': (timesteps.min(), timesteps.max()),
                'nnrd_range': (mean_nnrd.min(), mean_nnrd.max()),
                
                # Linear regression results
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'slope_std_err': std_err,
                'regression_p_value': p_value_regression,
                
                # Pearson correlation
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'pearson_significant': pearson_p <= alpha,  # True if correlation is significant
                'pearson_reject_h0': pearson_p <= alpha,    # True if we reject H0 (correlation found)
                
                # Spearman correlation  
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'spearman_significant': spearman_p <= alpha,
                'spearman_reject_h0': spearman_p <= alpha,
                
                # Kendall's tau
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'kendall_significant': kendall_p <= alpha,
                'kendall_reject_h0': kendall_p <= alpha,
                
                # Effect size interpretation
                'pearson_effect_size': self._interpret_correlation_magnitude(abs(pearson_r)),
                'spearman_effect_size': self._interpret_correlation_magnitude(abs(spearman_r)),
                
                # Confidence intervals for Pearson r (Fisher transformation)
                'pearson_ci_lower': None,
                'pearson_ci_upper': None,
            }
            
            # Calculate confidence interval for Pearson correlation
            if len(timesteps) > 3:  # Need at least 4 points for CI
                ci_lower, ci_upper = self._pearson_confidence_interval(pearson_r, len(timesteps), alpha)
                correlation_results[experiment]['pearson_ci_lower'] = ci_lower
                correlation_results[experiment]['pearson_ci_upper'] = ci_upper
        
        return correlation_results
        
    def test_critical_correlation_hypothesis(self, alpha=0.05):
        """
        Test hypothesis for correlation between timestep size and coefficient of variation.
        
        H0: There is NO significant correlation between timestep size and CV(NNRD) (rho = 0)
        H1: There IS a significant correlation between timestep size and CV(NNRD) (rho != 0)
        
        Where CV = std_nnrd / mean_nnrd (coefficient of variation)
        
        Parameters:
        -----------
        alpha : float
            Significance level (default 0.05)
            
        Returns:
        --------
        dict: Results of correlation hypothesis tests for each experiment
        """
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        correlation_results = {}
        
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            timesteps = exp_data['timestep'].values
            mean_nnrd = exp_data['mean_nnrd'].values
            std_nnrd = exp_data['std_nnrd'].values
            
            # Calculate coefficient of variation (CV = std/mean)
            # Handle division by zero cases
            cv_nnrd = np.where(mean_nnrd != 0, std_nnrd / mean_nnrd, np.nan)
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(cv_nnrd)
            timesteps_clean = timesteps[valid_mask]
            cv_nnrd_clean = cv_nnrd[valid_mask]
            
            if len(cv_nnrd_clean) < 3:  # Need at least 3 points for meaningful correlation
                correlation_results[experiment] = {
                    'error': 'Insufficient valid data points',
                    'n_points': len(cv_nnrd_clean)
                }
                continue
            
            # Linear regression to get slope
            slope, intercept, r_value, p_value_regression, std_err = stats.linregress(timesteps_clean, cv_nnrd_clean)
            
            # Pearson correlation test
            pearson_r, pearson_p = stats.pearsonr(timesteps_clean, cv_nnrd_clean)
            
            # Spearman correlation test (non-parametric)
            spearman_r, spearman_p = stats.spearmanr(timesteps_clean, cv_nnrd_clean)
            
            # Kendall's tau (another non-parametric measure)
            kendall_tau, kendall_p = stats.kendalltau(timesteps_clean, cv_nnrd_clean)
            
            # Standard interpretation: reject H0 if p-value <= alpha (significant correlation found)
            
            correlation_results[experiment] = {
                'n_points': len(timesteps_clean),
                'timestep_range': (timesteps_clean.min(), timesteps_clean.max()),
                'cv_range': (cv_nnrd_clean.min(), cv_nnrd_clean.max()),
                
                # Linear regression results
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'slope_std_err': std_err,
                'regression_p_value': p_value_regression,
                
                # Pearson correlation
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'pearson_significant': pearson_p <= alpha,  # True if correlation is significant
                'pearson_reject_h0': pearson_p <= alpha,    # True if we reject H0 (correlation found)
                
                # Spearman correlation  
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'spearman_significant': spearman_p <= alpha,
                'spearman_reject_h0': spearman_p <= alpha,
                
                # Kendall's tau
                'kendall_tau': kendall_tau,
                'kendall_p': kendall_p,
                'kendall_significant': kendall_p <= alpha,
                'kendall_reject_h0': kendall_p <= alpha,
                
                # Effect size interpretation
                'pearson_effect_size': self._interpret_correlation_magnitude(abs(pearson_r)),
                'spearman_effect_size': self._interpret_correlation_magnitude(abs(spearman_r)),
                
                # Confidence intervals for Pearson r (Fisher transformation)
                'pearson_ci_lower': None,
                'pearson_ci_upper': None,
            }
            
            # Calculate confidence interval for Pearson correlation
            if len(timesteps_clean) > 3:  # Need at least 4 points for CI
                ci_lower, ci_upper = self._pearson_confidence_interval(pearson_r, len(timesteps_clean), alpha)
                correlation_results[experiment]['pearson_ci_lower'] = ci_lower
                correlation_results[experiment]['pearson_ci_upper'] = ci_upper
        
        return correlation_results

    def print_critical_correlation_test_report(self, alpha=0.05, save_path=None):
        """Print comprehensive correlation hypothesis test report for coefficient of variation"""
        correlation_results = self.test_critical_correlation_hypothesis(alpha)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CRITICAL CORRELATION HYPOTHESIS TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Significance Level (α): {alpha}")
        report_lines.append("")
        report_lines.append("HYPOTHESIS:")
        report_lines.append("H₀: There is NO significant correlation between timestep size and CV(NNRD) (ρ = 0)")
        report_lines.append("H₁: There IS a significant correlation between timestep size and CV(NNRD) (ρ ≠ 0)")
        report_lines.append("")
        report_lines.append("WHERE CV(NNRD) = Standard Deviation / Mean (Coefficient of Variation)")
        report_lines.append("")
        report_lines.append("INTERPRETATION:")
        report_lines.append("- If p-value ≤ α: REJECT H₀ (significant correlation found)")
        report_lines.append("- If p-value > α: FAIL TO REJECT H₀ (no significant correlation)")
        report_lines.append("- CV measures relative variability (stability)")
        report_lines.append("- Lower CV = more stable/consistent performance")
        report_lines.append("")
        
        # Summary statistics
        valid_experiments = [k for k, v in correlation_results.items() if 'error' not in v]
        error_experiments = [k for k, v in correlation_results.items() if 'error' in v]
        
        if valid_experiments:
            significant_count = sum(1 for exp in valid_experiments 
                                if correlation_results[exp]['pearson_significant'])
            total_valid = len(valid_experiments)
            
            report_lines.append(f"SUMMARY:")
            report_lines.append(f"- Total experiments: {len(correlation_results)}")
            report_lines.append(f"- Valid experiments (sufficient data): {total_valid}")
            report_lines.append(f"- Experiments with errors: {len(error_experiments)}")
            report_lines.append(f"- Experiments with significant CV correlation: {significant_count}")
            report_lines.append(f"- Percentage with significant correlation: {100*significant_count/total_valid:.1f}%")
        else:
            report_lines.append(f"SUMMARY:")
            report_lines.append(f"- No valid experiments found")
        
        report_lines.append("")
        
        # Error experiments
        if error_experiments:
            report_lines.append("EXPERIMENTS WITH ERRORS:")
            report_lines.append("-" * 40)
            for exp in error_experiments:
                error_info = correlation_results[exp]
                report_lines.append(f"{exp}: {error_info['error']} (n={error_info.get('n_points', 0)})")
            report_lines.append("")
        
        # Detailed results for valid experiments
        if valid_experiments:
            report_lines.append("DETAILED RESULTS:")
            report_lines.append("-" * 80)
            
            for experiment in valid_experiments:
                results = correlation_results[experiment]
                report_lines.append(f"\n{experiment}:")
                report_lines.append(f"  Data Points: {results['n_points']}")
                report_lines.append(f"  Timestep Range: {results['timestep_range']}")
                report_lines.append(f"  CV Range: ({results['cv_range'][0]:.6f}, {results['cv_range'][1]:.6f})")
                
                # Linear regression results
                report_lines.append(f"  ")
                report_lines.append(f"  LINEAR REGRESSION:")
                report_lines.append(f"    Slope = {results['slope']:.8f} ± {results['slope_std_err']:.8f}")
                report_lines.append(f"    Intercept = {results['intercept']:.6f}")
                report_lines.append(f"    R² = {results['r_squared']:.4f}")
                report_lines.append(f"    Regression p-value = {results['regression_p_value']:.6f}")
                
                # Interpret slope direction and magnitude
                if results['slope'] > 0:
                    slope_direction = "POSITIVE (CV increases with timestep size → less stable)"
                elif results['slope'] < 0:
                    slope_direction = "NEGATIVE (CV decreases with timestep size → more stable)"
                else:
                    slope_direction = "ZERO (no linear trend)"
                report_lines.append(f"    Slope Direction: {slope_direction}")
                
                # Pearson correlation
                report_lines.append(f"  ")
                report_lines.append(f"  PEARSON CORRELATION:")
                report_lines.append(f"    r = {results['pearson_r']:.4f}")
                report_lines.append(f"    p-value = {results['pearson_p']:.6f}")
                report_lines.append(f"    Effect size: {results['pearson_effect_size']}")
                
                if results['pearson_ci_lower'] is not None:
                    report_lines.append(f"    95% CI: [{results['pearson_ci_lower']:.4f}, {results['pearson_ci_upper']:.4f}]")
                
                if results['pearson_significant']:
                    report_lines.append(f"    → REJECT H₀: Significant correlation found")
                else:
                    report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
                
                # Spearman correlation
                report_lines.append(f"  ")
                report_lines.append(f"  SPEARMAN CORRELATION (rank-based):")
                report_lines.append(f"    ρ = {results['spearman_r']:.4f}")
                report_lines.append(f"    p-value = {results['spearman_p']:.6f}")
                report_lines.append(f"    Effect size: {results['spearman_effect_size']}")
                
                if results['spearman_significant']:
                    report_lines.append(f"    → REJECT H₀: Significant correlation found")
                else:
                    report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
                
                # Kendall's tau
                report_lines.append(f"  ")
                report_lines.append(f"  KENDALL'S TAU (rank-based):")
                report_lines.append(f"    τ = {results['kendall_tau']:.4f}")
                report_lines.append(f"    p-value = {results['kendall_p']:.6f}")
                
                if results['kendall_significant']:
                    report_lines.append(f"    → REJECT H₀: Significant correlation found")
                else:
                    report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
                
                report_lines.append(f"  {'-'*50}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("NOTES:")
        report_lines.append("- CV = Coefficient of Variation = std_nnrd / mean_nnrd")
        report_lines.append("- Lower CV indicates more stable/consistent performance")
        report_lines.append("- Positive slope: CV increases with timestep (less stable)")
        report_lines.append("- Negative slope: CV decreases with timestep (more stable)")
        report_lines.append("- Pearson r measures linear correlation")
        report_lines.append("- Spearman ρ measures monotonic correlation (rank-based)")  
        report_lines.append("- Kendall τ is another rank-based correlation measure")
        report_lines.append("- Effect sizes: negligible (<0.1), small (0.1-0.3), medium (0.3-0.5),")
        report_lines.append("  large (0.5-0.7), very large (>0.7)")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nSaved critical correlation test report to: {save_path}")
        
        return report_text

    def plot_critical_correlation_analysis(self, figsize=(15, 12), save_path=None):
        """Plot critical correlation analysis (CV vs timestep) with regression lines and statistical annotations"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        correlation_results = self.test_critical_correlation_hypothesis()
        
        # Filter out experiments with errors
        valid_experiments = [exp for exp, results in correlation_results.items() 
                            if 'error' not in results]
        
        if not valid_experiments:
            print("No valid experiments found for critical correlation analysis.")
            return
        
        # Calculate grid size
        n_exp = len(valid_experiments)
        ncols = 3 if n_exp > 6 else 2 if n_exp > 2 else 1
        nrows = (n_exp + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, experiment in enumerate(valid_experiments):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            results = correlation_results[experiment]
            
            # Calculate CV
            cv_nnrd = exp_data['std_nnrd'] / exp_data['mean_nnrd']
            
            # Scatter plot
            ax.scatter(exp_data['timestep'], cv_nnrd, 
                    alpha=0.7, s=60, color='darkorange', edgecolors='black', linewidth=0.5)
            
            # Add regression line
            z = np.polyfit(exp_data['timestep'], cv_nnrd, 1)
            p = np.poly1d(z)
            ax.plot(exp_data['timestep'], p(exp_data['timestep']), 
                "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
            
            # Statistical annotations
            pearson_r = results['pearson_r']
            pearson_p = results['pearson_p']
            spearman_r = results['spearman_r']
            slope = results['slope']
            
            # Color code based on significance
            if results['pearson_significant']:
                title_color = 'green'
                sig_text = "Significant Correlation"
            else:
                title_color = 'red'
                sig_text = "No Significant Correlation"
            
            ax.set_title(f"{experiment}\n{sig_text}", color=title_color, fontweight='bold')
            ax.set_xlabel('Timestep Size')
            ax.set_ylabel('Coefficient of Variation (CV)')
            
            # Add statistics text box
            stats_text = f'Slope = {slope:.8f}\n'
            stats_text += f'Pearson r = {pearson_r:.3f}\np = {pearson_p:.4f}\n'
            stats_text += f'Spearman ρ = {spearman_r:.3f}\n'
            stats_text += f'Effect: {results["pearson_effect_size"]}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=9, family='monospace')
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(valid_experiments), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved critical correlation analysis plot to: {save_path}")
        
        plt.show()


    
    def _interpret_correlation_magnitude(self, r):
        """Interpret the magnitude of correlation coefficient"""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small" 
        elif r < 0.5:
            return "medium"
        elif r < 0.7:
            return "large"
        else:
            return "very large"
    
    def _pearson_confidence_interval(self, r, n, alpha=0.05):
        """Calculate confidence interval for Pearson correlation using Fisher transformation"""
        if abs(r) >= 1.0:
            return None, None
            
        # Fisher transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval for z_r
        ci_lower_z = z_r - z_alpha * se
        ci_upper_z = z_r + z_alpha * se
        
        # Transform back to correlation scale
        ci_lower = (np.exp(2 * ci_lower_z) - 1) / (np.exp(2 * ci_lower_z) + 1)
        ci_upper = (np.exp(2 * ci_upper_z) - 1) / (np.exp(2 * ci_upper_z) + 1)
        
        return ci_lower, ci_upper
    
    def print_correlation_test_report(self, alpha=0.05, save_path=None):
        """Print comprehensive correlation hypothesis test report"""
        correlation_results = self.test_correlation_hypothesis(alpha)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CORRELATION HYPOTHESIS TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Significance Level (α): {alpha}")
        report_lines.append("")
        report_lines.append("HYPOTHESIS:")
        report_lines.append("H₀: There is NO significant correlation between timestep size and NNRD (ρ = 0)")
        report_lines.append("H₁: There IS a significant correlation between timestep size and NNRD (ρ ≠ 0)")
        report_lines.append("")
        report_lines.append("INTERPRETATION:")
        report_lines.append("- If p-value ≤ α: REJECT H₀ (significant correlation found)")
        report_lines.append("- If p-value > α: FAIL TO REJECT H₀ (no significant correlation)")
        report_lines.append("")
        
        # Summary statistics
        significant_count = sum(1 for result in correlation_results.values() 
                              if result['pearson_significant'])
        total_experiments = len(correlation_results)
        
        report_lines.append(f"SUMMARY:")
        report_lines.append(f"- Total experiments: {total_experiments}")
        report_lines.append(f"- Experiments with significant correlation: {significant_count}")
        report_lines.append(f"- Percentage with significant correlation: {100*significant_count/total_experiments:.1f}%")
        report_lines.append("")
        
        # Detailed results for each experiment
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 80)
        
        for experiment, results in correlation_results.items():
            report_lines.append(f"\n{experiment}:")
            report_lines.append(f"  Data Points: {results['n_points']}")
            report_lines.append(f"  Timestep Range: {results['timestep_range']}")
            report_lines.append(f"  NNRD Range: ({results['nnrd_range'][0]:.4f}, {results['nnrd_range'][1]:.4f})")
            
            # Linear regression results
            report_lines.append(f"  ")
            report_lines.append(f"  LINEAR REGRESSION:")
            report_lines.append(f"    Slope = {results['slope']:.6f} ± {results['slope_std_err']:.6f}")
            report_lines.append(f"    Intercept = {results['intercept']:.4f}")
            report_lines.append(f"    R² = {results['r_squared']:.4f}")
            report_lines.append(f"    Regression p-value = {results['regression_p_value']:.6f}")
            
            # Interpret slope direction and magnitude
            if results['slope'] > 0:
                slope_direction = "POSITIVE (NNRD increases with timestep size)"
            elif results['slope'] < 0:
                slope_direction = "NEGATIVE (NNRD decreases with timestep size)"
            else:
                slope_direction = "ZERO (no linear trend)"
            report_lines.append(f"    Slope Direction: {slope_direction}")
            
            # Pearson correlation
            report_lines.append(f"  ")
            report_lines.append(f"  PEARSON CORRELATION:")
            report_lines.append(f"    r = {results['pearson_r']:.4f}")
            report_lines.append(f"    p-value = {results['pearson_p']:.6f}")
            report_lines.append(f"    Effect size: {results['pearson_effect_size']}")
            
            if results['pearson_ci_lower'] is not None:
                report_lines.append(f"    95% CI: [{results['pearson_ci_lower']:.4f}, {results['pearson_ci_upper']:.4f}]")
            
            if results['pearson_significant']:
                report_lines.append(f"    → REJECT H₀: Significant correlation found")
            else:
                report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
            
            # Spearman correlation
            report_lines.append(f"  ")
            report_lines.append(f"  SPEARMAN CORRELATION (rank-based):")
            report_lines.append(f"    ρ = {results['spearman_r']:.4f}")
            report_lines.append(f"    p-value = {results['spearman_p']:.6f}")
            report_lines.append(f"    Effect size: {results['spearman_effect_size']}")
            
            if results['spearman_significant']:
                report_lines.append(f"    → REJECT H₀: Significant correlation found")
            else:
                report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
            
            # Kendall's tau
            report_lines.append(f"  ")
            report_lines.append(f"  KENDALL'S TAU (rank-based):")
            report_lines.append(f"    τ = {results['kendall_tau']:.4f}")
            report_lines.append(f"    p-value = {results['kendall_p']:.6f}")
            
            if results['kendall_significant']:
                report_lines.append(f"    → REJECT H₀: Significant correlation found")
            else:
                report_lines.append(f"    → FAIL TO REJECT H₀: No significant correlation")
            
            report_lines.append(f"  {'-'*50}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("NOTES:")
        report_lines.append("- Slope units: NNRD change per unit timestep size")
        report_lines.append("- Positive slope: NNRD increases as timestep size increases")
        report_lines.append("- Negative slope: NNRD decreases as timestep size increases")
        report_lines.append("- Pearson r measures linear correlation")
        report_lines.append("- Spearman ρ measures monotonic correlation (rank-based)")  
        report_lines.append("- Kendall τ is another rank-based correlation measure")
        report_lines.append("- Effect sizes: negligible (<0.1), small (0.1-0.3), medium (0.3-0.5),")
        report_lines.append("  large (0.5-0.7), very large (>0.7)")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nSaved correlation test report to: {save_path}")
        
        return report_text
    
    def plot_correlation_analysis(self, figsize=(15, 12), save_path=None):
        """Plot correlation analysis with regression lines and statistical annotations"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        correlation_results = self.test_correlation_hypothesis()
        
        # Calculate grid size
        experiments = list(correlation_results.keys())
        n_exp = len(experiments)
        ncols = 3 if n_exp > 6 else 2 if n_exp > 2 else 1
        nrows = (n_exp + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for idx, experiment in enumerate(experiments):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            results = correlation_results[experiment]
            
            # Scatter plot
            ax.scatter(exp_data['timestep'], exp_data['mean_nnrd'], 
                      alpha=0.7, s=60, color='steelblue', edgecolors='black', linewidth=0.5)
            
            # Add regression line
            z = np.polyfit(exp_data['timestep'], exp_data['mean_nnrd'], 1)
            p = np.poly1d(z)
            ax.plot(exp_data['timestep'], p(exp_data['timestep']), 
                   "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
            
            # Add error bars
            ax.errorbar(exp_data['timestep'], exp_data['mean_nnrd'], 
                       yerr=exp_data['std_nnrd'], fmt='none', 
                       color='gray', alpha=0.5, capsize=3)
            
            # Statistical annotations
            pearson_r = results['pearson_r']
            pearson_p = results['pearson_p']
            spearman_r = results['spearman_r']
            slope = results['slope']
            
            # Color code based on significance
            if results['pearson_significant']:
                title_color = 'green'
                sig_text = "Significant Correlation"
            else:
                title_color = 'red'
                sig_text = "No Significant Correlation"
            
            ax.set_title(f"{experiment}\n{sig_text}", color=title_color, fontweight='bold')
            ax.set_xlabel('Timestep Size')
            ax.set_ylabel('Mean NNRD Score')
            
            # Add statistics text box
            stats_text = f'Slope = {slope:.6f}\n'
            stats_text += f'Pearson r = {pearson_r:.3f}\np = {pearson_p:.4f}\n'
            stats_text += f'Spearman ρ = {spearman_r:.3f}\n'
            stats_text += f'Effect: {results["pearson_effect_size"]}'
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9, family='monospace')
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(experiments), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation analysis plot to: {save_path}")
        
        plt.show()
    
    def plot_mean_curves(self, figsize=(12, 8), save_path=None):
        """Plot mean NNRD scores vs timesteps for all experiments"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        plt.figure(figsize=figsize)
        
        # Plot each experiment
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            plt.plot(exp_data['timestep'], exp_data['mean_nnrd'], 
                    marker='o', label=experiment, linewidth=2, markersize=6)
            
            # Add error bars
            plt.fill_between(exp_data['timestep'], 
                           exp_data['mean_nnrd'] - exp_data['std_nnrd'],
                           exp_data['mean_nnrd'] + exp_data['std_nnrd'],
                           alpha=0.2)
        
        plt.xlabel('Timestep Size (Number of Bins)')
        plt.ylabel('Mean NNRD Score')
        plt.title('NNRD Score vs Timestep Size Across Experiments')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved mean curves plot to: {save_path}")
        
        plt.show()
    
    def plot_stability_metrics(self, figsize=(15, 10), save_path=None):
        """Plot various stability metrics"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Standard deviation vs timestep
        ax1 = axes[0, 0]
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            ax1.plot(exp_data['timestep'], exp_data['std_nnrd'], 
                    marker='o', label=experiment)
        ax1.set_xlabel('Timestep Size')
        ax1.set_ylabel('Standard Deviation')
        ax1.set_title('Standard Deviation vs Timestep Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Coefficient of variation vs timestep
        ax2 = axes[0, 1]
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            cv = exp_data['std_nnrd'] / exp_data['mean_nnrd']
            ax2.plot(exp_data['timestep'], cv, marker='o', label=experiment)
        ax2.set_xlabel('Timestep Size')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Coefficient of Variation vs Timestep Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Range (max - min) vs timestep
        ax3 = axes[1, 0]
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            range_val = exp_data['max_nnrd'] - exp_data['min_nnrd']
            ax3.plot(exp_data['timestep'], range_val, marker='o', label=experiment)
        ax3.set_xlabel('Timestep Size')
        ax3.set_ylabel('Range (Max - Min)')
        ax3.set_title('Range vs Timestep Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Interquartile range vs timestep
        ax4 = axes[1, 1]
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            iqr = exp_data['q75_nnrd'] - exp_data['q25_nnrd']
            ax4.plot(exp_data['timestep'], iqr, marker='o', label=experiment)
        ax4.set_xlabel('Timestep Size')
        ax4.set_ylabel('Interquartile Range')
        ax4.set_title('IQR vs Timestep Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved stability metrics plot to: {save_path}")
        
        plt.show()
    
    def calculate_stability_metrics(self):
        """Calculate comprehensive stability metrics for each experiment"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        stability_results = {}
        
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            # Basic statistics
            mean_scores = exp_data['mean_nnrd'].values
            std_scores = exp_data['std_nnrd'].values
            timesteps = exp_data['timestep'].values
            
            # 1. Overall variability in mean scores
            overall_std = np.std(mean_scores)
            overall_cv = overall_std / np.mean(mean_scores)
            
            # 2. Trend analysis (linear regression)
            slope, intercept, r_value, p_value, std_err = stats.linregress(timesteps, mean_scores)
            
            # 3. Average within-timestep variability
            avg_within_std = np.mean(std_scores)
            
            # 4. Relative stability (inverse of coefficient of variation)
            relative_stability = 1 / overall_cv if overall_cv > 0 else np.inf
            
            # 5. Monotonicity test (Spearman correlation)
            spearman_corr, spearman_p = stats.spearmanr(timesteps, mean_scores)
            
            # 6. Plateau detection (consecutive differences)
            diffs = np.diff(mean_scores)
            small_changes = np.abs(diffs) < 0.01  # Threshold for "small" change
            max_plateau_length = self._find_longest_plateau(small_changes)
            
            stability_results[experiment] = {
                'overall_std': overall_std,
                'overall_cv': overall_cv,
                'relative_stability': relative_stability,
                'linear_slope': slope,
                'linear_r_squared': r_value**2,
                'linear_p_value': p_value,
                'avg_within_std': avg_within_std,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'max_plateau_length': max_plateau_length,
                'n_timesteps': len(timesteps),
                'score_range': np.max(mean_scores) - np.min(mean_scores)
            }
        
        return stability_results
    
    def _find_longest_plateau(self, boolean_array):
        """Find the longest consecutive sequence of True values"""
        if len(boolean_array) == 0:
            return 0
        
        max_length = 0
        current_length = 0
        
        for val in boolean_array:
            if val:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length
    
    def print_stability_report(self, save_path=None):
        """Print a comprehensive stability report"""
        stability_metrics = self.calculate_stability_metrics()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STABILITY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        
        # Sort experiments by overall stability
        sorted_experiments = sorted(stability_metrics.items(), 
                                  key=lambda x: x[1]['overall_cv'])
        
        for experiment, metrics in sorted_experiments:
            report_lines.append(f"\n{experiment}:")
            report_lines.append(f"  Overall Stability (CV): {metrics['overall_cv']:.6f}")
            report_lines.append(f"  Score Range: {metrics['score_range']:.6f}")
            report_lines.append(f"  Linear Trend (slope): {metrics['linear_slope']:.6f}")
            report_lines.append(f"  Linear R²: {metrics['linear_r_squared']:.3f}")
            report_lines.append(f"  Spearman Correlation: {metrics['spearman_correlation']:.3f}")
            report_lines.append(f"  Average Within-Timestep Std: {metrics['avg_within_std']:.6f}")
            report_lines.append(f"  Max Plateau Length: {metrics['max_plateau_length']}")
            report_lines.append(f"  Timesteps Analyzed: {metrics['n_timesteps']}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("INTERPRETATION:")
        report_lines.append("- Lower CV (Coefficient of Variation) = More stable")
        report_lines.append("- Lower |slope| = Less trend/more stable across timesteps")
        report_lines.append("- Higher Max Plateau Length = More stable regions")
        report_lines.append("- Lower Average Within-Timestep Std = More reproducible")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nSaved stability report to: {save_path}")
        
        return report_text
    
    def get_optimal_timestep_recommendations(self):
        """Recommend optimal timestep sizes based on stability metrics"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        recommendations = {}
        
        for experiment in self.summary_df['experiment'].unique():
            exp_data = self.summary_df[self.summary_df['experiment'] == experiment]
            exp_data = exp_data.sort_values('timestep')
            
            # Find timestep with minimum CV
            exp_data['cv'] = exp_data['std_nnrd'] / exp_data['mean_nnrd']
            min_cv_idx = exp_data['cv'].idxmin()
            optimal_timestep = exp_data.loc[min_cv_idx]
            
            # Find plateau regions (low variability in consecutive timesteps)
            diffs = np.abs(np.diff(exp_data['mean_nnrd'].values))
            plateau_threshold = np.std(diffs) * 0.5  # Adaptive threshold
            plateau_regions = diffs < plateau_threshold
            
            recommendations[experiment] = {
                'optimal_timestep': optimal_timestep['timestep'],
                'optimal_cv': optimal_timestep['cv'],
                'optimal_mean_nnrd': optimal_timestep['mean_nnrd'],
                'plateau_regions': plateau_regions,
                'suggested_range': self._get_suggested_range(exp_data, plateau_regions)
            }
        
        return recommendations
    
    def _get_suggested_range(self, exp_data, plateau_regions):
        """Get suggested timestep range based on plateau analysis"""
        if len(plateau_regions) == 0:
            return None
        
        # Find the longest plateau
        plateau_starts = []
        plateau_lengths = []
        
        i = 0
        while i < len(plateau_regions):
            if plateau_regions[i]:
                start = i
                length = 1
                while i + 1 < len(plateau_regions) and plateau_regions[i + 1]:
                    length += 1
                    i += 1
                plateau_starts.append(start)
                plateau_lengths.append(length)
            i += 1
        
        if plateau_lengths:
            max_length_idx = np.argmax(plateau_lengths)
            start_idx = plateau_starts[max_length_idx]
            end_idx = start_idx + plateau_lengths[max_length_idx]
            
            timesteps = exp_data['timestep'].values
            return (timesteps[start_idx], timesteps[min(end_idx, len(timesteps)-1)])
        
        return None
    
    def save_summary_data(self, output_dir=None):
        """Save processed data and results to files"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        # Use same directory as input data if no output_dir specified
        if output_dir is None:
            output_dir = self.data_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary DataFrame
        csv_path = os.path.join(output_dir, "timestep_analysis_summary.csv")
        self.summary_df.to_csv(csv_path, index=False)
        print(f"Saved summary data to: {csv_path}")
        
        # Save stability metrics
        stability_metrics = self.calculate_stability_metrics()
        stability_df = pd.DataFrame(stability_metrics).T
        stability_csv_path = os.path.join(output_dir, "stability_metrics.csv")
        stability_df.to_csv(stability_csv_path)
        print(f"Saved stability metrics to: {stability_csv_path}")
        
        # Save correlation test results
        correlation_results = self.test_correlation_hypothesis()
        correlation_df = pd.DataFrame(correlation_results).T
        correlation_csv_path = os.path.join(output_dir, "correlation_test_results.csv")
        correlation_df.to_csv(correlation_csv_path)
        print(f"Saved correlation test results to: {correlation_csv_path}")
        
        # Save recommendations
        recommendations = self.get_optimal_timestep_recommendations()
        rec_data = []
        for exp, rec in recommendations.items():
            rec_data.append({
                'experiment': exp,
                'optimal_timestep': rec['optimal_timestep'],
                'optimal_cv': rec['optimal_cv'],
                'optimal_mean_nnrd': rec['optimal_mean_nnrd'],
                'suggested_range': str(rec['suggested_range'])
            })
        
        rec_df = pd.DataFrame(rec_data)
        rec_csv_path = os.path.join(output_dir, "timestep_recommendations.csv")
        rec_df.to_csv(rec_csv_path, index=False)
        print(f"Saved recommendations to: {rec_csv_path}")
        
        return {
            'summary_data': csv_path,
            'stability_metrics': stability_csv_path,
            'correlation_tests': correlation_csv_path,
            'recommendations': rec_csv_path
        }
    
    # Updated run_full_analysis method to include critical correlation tests
    def run_full_analysis(self, output_dir=None, alpha=0.05):
        """Run complete analysis including both regular and critical correlation hypothesis tests"""
        print("Running full timestep analysis with correlation hypothesis testing...")
        
        # Use same directory as input data if no output_dir specified
        if output_dir is None:
            output_dir = self.data_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_all_files()
        df = self.create_summary_dataframe()
        print(f"Loaded data for {len(df)} experiments")
        
        # Save plots
        mean_curves_path = os.path.join(output_dir, "mean_curves.png")
        stability_plot_path = os.path.join(output_dir, "stability_metrics.png")
        correlation_plot_path = os.path.join(output_dir, "correlation_analysis.png")
        critical_correlation_plot_path = os.path.join(output_dir, "critical_correlation_analysis.png")
        
        self.plot_mean_curves(save_path=mean_curves_path)
        self.plot_stability_metrics(save_path=stability_plot_path)
        self.plot_correlation_analysis(save_path=correlation_plot_path)
        self.plot_critical_correlation_analysis(save_path=critical_correlation_plot_path)
        
        # Save reports
        report_path = os.path.join(output_dir, "stability_report.txt")
        correlation_report_path = os.path.join(output_dir, "correlation_hypothesis_test_report.txt")
        critical_correlation_report_path = os.path.join(output_dir, "critical_correlation_hypothesis_test_report.txt")
        
        self.print_stability_report(save_path=report_path)
        self.print_correlation_test_report(alpha=alpha, save_path=correlation_report_path)
        self.print_critical_correlation_test_report(alpha=alpha, save_path=critical_correlation_report_path)
        
        # Save data files including critical correlation results
        data_paths = self.save_summary_data(output_dir)
        
        # Save critical correlation test results
        critical_correlation_results = self.test_critical_correlation_hypothesis(alpha)
        critical_correlation_df = pd.DataFrame({k: v for k, v in critical_correlation_results.items() 
                                            if 'error' not in v}).T
        critical_correlation_csv_path = os.path.join(output_dir, "critical_correlation_test_results.csv")
        critical_correlation_df.to_csv(critical_correlation_csv_path)
        print(f"Saved critical correlation test results to: {critical_correlation_csv_path}")
        
        # Print recommendations
        recommendations = self.get_optimal_timestep_recommendations()
        print("\nOPTIMAL TIMESTEP RECOMMENDATIONS:")
        for exp, rec in recommendations.items():
            print(f"{exp}: Timestep {rec['optimal_timestep']} "
                f"(CV={rec['optimal_cv']:.4f}, NNRD={rec['optimal_mean_nnrd']:.4f})")
            if rec['suggested_range']:
                print(f"  Suggested stable range: {rec['suggested_range'][0]}-{rec['suggested_range'][1]}")
        
        # Print correlation test summaries
        correlation_results = self.test_correlation_hypothesis(alpha)
        significant_correlations = sum(1 for result in correlation_results.values() 
                                    if result['pearson_significant'])
        total_experiments = len(correlation_results)
        
        print(f"\nCORRELATION HYPOTHESIS TEST SUMMARY:")
        print(f"- Experiments with significant timestep-NNRD correlation: {significant_correlations}/{total_experiments}")
        print(f"- Percentage with significant correlation: {100*significant_correlations/total_experiments:.1f}%")
        
        # Critical correlation summary
        valid_critical_results = {k: v for k, v in critical_correlation_results.items() if 'error' not in v}
        if valid_critical_results:
            significant_critical_correlations = sum(1 for result in valid_critical_results.values() 
                                                if result['pearson_significant'])
            total_valid_critical = len(valid_critical_results)
            
            print(f"\nCRITICAL CORRELATION HYPOTHESIS TEST SUMMARY:")
            print(f"- Experiments with significant timestep-CV correlation: {significant_critical_correlations}/{total_valid_critical}")
            print(f"- Percentage with significant CV correlation: {100*significant_critical_correlations/total_valid_critical:.1f}%")
        
        print(f"\nAll results saved to: {output_dir}")
        return {
            'plots': [mean_curves_path, stability_plot_path, correlation_plot_path, critical_correlation_plot_path],
            'reports': [report_path, correlation_report_path, critical_correlation_report_path],
            'critical_correlation_results': critical_correlation_csv_path,
            **data_paths
        }

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TimestepAnalyzer("timestep_analysis/useful/")
    
    # Run complete analysis with correlation hypothesis testing
    # Significance level can be adjusted (default α = 0.05)
    results = analyzer.run_full_analysis(alpha=0.05)
    
    # Or run just the correlation hypothesis test
    # analyzer.load_all_files()
    # analyzer.create_summary_dataframe()
    # correlation_results = analyzer.test_correlation_hypothesis(alpha=0.05)
    # analyzer.print_correlation_test_report(alpha=0.05)
    # analyzer.plot_correlation_analysis()
    
    # Or specify a different directory if needed:
    # results = analyzer.run_full_analysis("custom_output_dir/", alpha=0.01)