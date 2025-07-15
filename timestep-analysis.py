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
            'recommendations': rec_csv_path
        }
    
    def run_full_analysis(self, output_dir=None):
        """Run complete analysis and save all results"""
        print("Running full timestep analysis...")
        
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
        
        self.plot_mean_curves(save_path=mean_curves_path)
        self.plot_stability_metrics(save_path=stability_plot_path)
        
        # Save report
        report_path = os.path.join(output_dir, "stability_report.txt")
        self.print_stability_report(save_path=report_path)
        
        # Save data files
        data_paths = self.save_summary_data(output_dir)
        
        # Print recommendations
        recommendations = self.get_optimal_timestep_recommendations()
        print("\nOPTIMAL TIMESTEP RECOMMENDATIONS:")
        for exp, rec in recommendations.items():
            print(f"{exp}: Timestep {rec['optimal_timestep']} "
                  f"(CV={rec['optimal_cv']:.4f}, NNRD={rec['optimal_mean_nnrd']:.4f})")
            if rec['suggested_range']:
                print(f"  Suggested stable range: {rec['suggested_range'][0]}-{rec['suggested_range'][1]}")
        
        print(f"\nAll results saved to: {output_dir}")
        return {
            'plots': [mean_curves_path, stability_plot_path],
            'report': report_path,
            **data_paths
        }


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TimestepAnalyzer("timestep_analysis/useful/")
    
    # Run complete analysis and save all results in the same directory as JSONs
    results = analyzer.run_full_analysis()  # Will save to "timestep_analysis/useful/"
    
    # Or specify a different directory if needed:
    # results = analyzer.run_full_analysis("custom_output_dir/")