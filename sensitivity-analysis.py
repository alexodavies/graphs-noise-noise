#!/usr/bin/env python3
"""
Sensitivity Analysis for Timestep Size in Noise-Noise Analysis

This script evaluates how the choice of timestep size (n_noise_levels) affects
the stability and reliability of NNRD (Noise-Noise Ratio Difference) calculations.
It runs multiple evaluations with different timestep sizes and provides error bars.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import json
import os
from datetime import datetime
import warnings
import yaml
import wandb
from fixed_train_set import evaluate_main_fixed_train
from metrics import nnd
warnings.filterwarnings('ignore')


def evaluate_dataset(args, retrain = False):

    dataset = args.dataset
    n_noise_levels = args.n_noise_levels
    n_repeats = args.n_repeats
    use_linear = args.use_linear

    # if "synth" in dataset:
    #     project = "noise-synthetics-benchmarks-fixed-nnd"
    # elif "ogbg" in dataset:
    #     project = "noise-ogbg"
    # elif "TU" in dataset:
    #     project = "noise-TUDatasets"

    project = "noise-sensitivity"

    use_linear = False  # TODO: fix code - currently being set to true by bash script
    pos_included_string = "-pos" if args.structure else ""
    run_name = args.layer_type + '-' + dataset + pos_included_string + "-ER-swapping"
    if args.top_model is not None:
        run_name = f"ToP-{args.top_model}-{dataset}" + pos_included_string
    wandb.init(project= project, # "noise-synthetics-benchmarks",  # + "-linear" if use_linear else "",
               entity="hierarchical-diffusion",
               name=run_name,
               config=args)
    
    wandb.log({"Noise type":"ER-Swapping"})

    result_dict = {"dataset": dataset, "pos":args.structure}
    structure_performances = dict()
    feature_performances = dict()
    ts = np.linspace(0, 1, n_noise_levels)

    eval_fn = evaluate_main_fixed_train

    # eval_fn = evaluate_main if not args.fixed_train else evaluate_main_fixed_train

    for ti in tqdm(range(n_noise_levels), desc=f"Running {dataset}"):
        ti_performances_structure = []
        ti_performances_feature = []
        repeat_pbar = tqdm(
            range(n_repeats), desc="Running repeats", leave=False)
        for i_repeat in repeat_pbar:
            
            # Same data for t = 0
            if ti == 0:
                # struc, tt = evaluate_main(dataset=dataset, t_structure=ts[ti],
                #                 linear=use_linear, layer_type=args.layer, pos_encodings=args.structure)

                # Do we need to retrain the model?
                force_train = True if retrain and i_repeat == 0 else False
                struc, tt = eval_fn(args, force_retrain = force_train)
                ti_performances_structure.append(struc)
                ti_performances_feature.append(struc)

                pbar_string = f"Struc: {struc}, feat: {struc}"
                repeat_pbar.set_postfix_str(pbar_string)

                continue
            else:
                struc, tt = eval_fn(args, t_structure = ts[ti])
                ti_performances_structure.append(struc)



                # feat, tt = evaluate_main(dataset=dataset, t_feature=ts[ti],
                #                          linear=use_linear, layer_type=args.layer, pos_encodings=args.structure)
                feat, tt = eval_fn(args, t_feature = ts[ti])
                ti_performances_feature.append(feat)

                pbar_string = f"Struc: {struc}, feat: {feat}"
                repeat_pbar.set_postfix_str(pbar_string)

        # Log intermediate results to wandb
        wandb.log({
            "noise_level": ts[ti],
            "structure_performance": np.mean(ti_performances_structure),
            "feature_performance": np.mean(ti_performances_feature),
        })

        structure_performances[str(ts[ti])] = [str(s)
                                               for s in ti_performances_structure]
        feature_performances[str(ts[ti])] = [str(f)
                                             for f in ti_performances_feature]

    result_dict["structure"] = structure_performances
    result_dict["feature"] = feature_performances
    result_dict["task_type"] = tt
    result_dict["linear"] = use_linear
    result_dict["layer"] = args.layer_type

    return nnd(result_dict)

    # image_path = plot_results(
    #     result_dict, extra_save_string=args.layer_type+pos_included_string, return_path=True)

    # wandb.log({"Media/Result-Image": wandb.Image(image_path)})

    # save_run(result_dict)
    # wandb.finish()

class TimestepSensitivityAnalyzer:
    def __init__(self, base_args, timestep_sizes=None, n_bootstrap=10):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            base_args: Base arguments for the noise analysis
            timestep_sizes: List of timestep sizes to test
            n_bootstrap: Number of bootstrap samples for error estimation
        """
        self.base_args = base_args
        self.timestep_sizes = timestep_sizes or [5, 7, 10, 15, 20, 25, 30]
        self.n_bootstrap = n_bootstrap
        self.results = {}
        
    def run_sensitivity_analysis(self):
        """Run sensitivity analysis across different timestep sizes."""
        print(f"Running timestep sensitivity analysis...")
        print(f"Timestep sizes to test: {self.timestep_sizes}")
        print(f"Bootstrap samples per size: {self.n_bootstrap}")
        print(f"Dataset: {self.base_args.dataset}")
        print(f"Layer type: {self.base_args.layer_type}")
        print("-" * 60)
        
        for timestep_size in tqdm(self.timestep_sizes, desc="Testing timestep sizes"):
            print(f"\nTesting timestep size: {timestep_size}")
            
            # Store NNRD scores for this timestep size
            nnrd_scores = []
            
            # Run multiple evaluations for bootstrap error estimation
            for bootstrap_idx in tqdm(range(self.n_bootstrap), 
                                    desc=f"Bootstrap runs (n={timestep_size})", 
                                    leave=False):
                
                # Create modified args for this run
                args_copy = self._copy_args(self.base_args)
                args_copy.n_noise_levels = timestep_size
                
                # Run evaluation and get NNRD score
                try:
                    nnrd_score = evaluate_dataset(args_copy, retrain = True)
                    nnrd_scores.append(nnrd_score)
                    
                    print(f"  Bootstrap {bootstrap_idx+1}/{self.n_bootstrap}: NNRD = {nnrd_score:.4f}")
                    
                except Exception as e:
                    print(f"  Bootstrap {bootstrap_idx+1} failed: {e}")
                    continue
            
            # Store results for this timestep size
            if nnrd_scores:
                self.results[timestep_size] = {
                    'nnrd_scores': nnrd_scores,
                    'mean': np.mean(nnrd_scores),
                    'std': np.std(nnrd_scores),
                    'median': np.median(nnrd_scores),
                    'q25': np.percentile(nnrd_scores, 25),
                    'q75': np.percentile(nnrd_scores, 75),
                    'min': np.min(nnrd_scores),
                    'max': np.max(nnrd_scores),
                    'n_successful': len(nnrd_scores)
                }
                
                print(f"  Results for n={timestep_size}:")
                print(f"    Mean NNRD: {self.results[timestep_size]['mean']:.4f} ± {self.results[timestep_size]['std']:.4f}")
                print(f"    Median NNRD: {self.results[timestep_size]['median']:.4f}")
                print(f"    Range: [{self.results[timestep_size]['min']:.4f}, {self.results[timestep_size]['max']:.4f}]")
            else:
                print(f"  No successful runs for timestep size {timestep_size}")
    
    def _copy_args(self, args):
        """Create a copy of arguments object."""
        import copy
        return copy.deepcopy(args)
    
    def plot_results(self, save_path=None):
        """Create visualization of sensitivity analysis results."""
        if not self.results:
            print("No results to plot. Run sensitivity analysis first.")
            return
        
        # Prepare data for plotting
        timesteps = []
        means = []
        stds = []
        medians = []
        q25s = []
        q75s = []
        
        for timestep in sorted(self.results.keys()):
            timesteps.append(timestep)
            means.append(self.results[timestep]['mean'])
            stds.append(self.results[timestep]['std'])
            medians.append(self.results[timestep]['median'])
            q25s.append(self.results[timestep]['q25'])
            q75s.append(self.results[timestep]['q75'])
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Mean NNRD with error bars
        ax1.errorbar(timesteps, means, yerr=stds, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_xlabel('Number of Timesteps')
        ax1.set_ylabel('NNRD Score')
        ax1.set_title('Mean NNRD vs Timestep Size\n(Error bars show ±1 std)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Balanced (NNRD=0)')
        ax1.legend()
        
        # Plot 2: Median with quartiles
        ax2.plot(timesteps, medians, 'o-', linewidth=2, label='Median')
        ax2.fill_between(timesteps, q25s, q75s, alpha=0.3, label='IQR (25%-75%)')
        ax2.set_xlabel('Number of Timesteps')
        ax2.set_ylabel('NNRD Score')
        ax2.set_title('Median NNRD vs Timestep Size\n(Shaded area shows IQR)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.legend()
        
        # Plot 3: Standard deviation (precision)
        ax3.plot(timesteps, stds, 'o-', color='orange', linewidth=2)
        ax3.set_xlabel('Number of Timesteps')
        ax3.set_ylabel('Standard Deviation of NNRD')
        ax3.set_title('NNRD Precision vs Timestep Size\n(Lower is more stable)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot for all timestep sizes
        box_data = []
        box_labels = []
        for timestep in sorted(self.results.keys()):
            box_data.append(self.results[timestep]['nnrd_scores'])
            box_labels.append(f'n={timestep}')
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        ax4.set_xlabel('Number of Timesteps')
        ax4.set_ylabel('NNRD Score')
        ax4.set_title('Distribution of NNRD Scores by Timestep Size')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        # plt.show()
        
        return fig
    
    def generate_report(self, save_path=None):
        """Generate a detailed report of the sensitivity analysis."""
        if not self.results:
            print("No results to report. Run sensitivity analysis first.")
            return
        
        report = []
        report.append("=" * 80)
        report.append("TIMESTEP SENSITIVITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset: {self.base_args.dataset}")
        report.append(f"Layer Type: {self.base_args.layer_type}")
        report.append(f"Structure (Positional Encodings): {self.base_args.structure}")
        report.append(f"Bootstrap Samples: {self.n_bootstrap}")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        all_means = [self.results[ts]['mean'] for ts in sorted(self.results.keys())]
        all_stds = [self.results[ts]['std'] for ts in sorted(self.results.keys())]
        
        report.append(f"Overall NNRD range: [{min(all_means):.4f}, {max(all_means):.4f}]")
        report.append(f"Mean standard deviation: {np.mean(all_stds):.4f}")
        report.append(f"Stability (std of means): {np.std(all_means):.4f}")
        
        # Identify most stable timestep size
        most_stable_idx = np.argmin(all_stds)
        most_stable_timestep = sorted(self.results.keys())[most_stable_idx]
        report.append(f"Most stable timestep size: {most_stable_timestep} (std = {all_stds[most_stable_idx]:.4f})")
        
        report.append("")
        
        # Detailed results for each timestep size
        report.append("DETAILED RESULTS BY TIMESTEP SIZE")
        report.append("-" * 40)
        
        for timestep in sorted(self.results.keys()):
            result = self.results[timestep]
            report.append(f"Timestep Size: {timestep}")
            report.append(f"  Successful runs: {result['n_successful']}/{self.n_bootstrap}")
            report.append(f"  Mean NNRD: {result['mean']:.6f}")
            report.append(f"  Std Dev: {result['std']:.6f}")
            report.append(f"  Median: {result['median']:.6f}")
            report.append(f"  IQR: [{result['q25']:.6f}, {result['q75']:.6f}]")
            report.append(f"  Range: [{result['min']:.6f}, {result['max']:.6f}]")
            
            # Interpretation
            if abs(result['mean']) < 0.05:
                bias_type = "Balanced"
            elif result['mean'] < 0:
                bias_type = "Feature-biased"
            else:
                bias_type = "Structure-biased"
            report.append(f"  Interpretation: {bias_type}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find timestep with good balance of precision and computational cost
        precision_scores = [(ts, self.results[ts]['std']) for ts in sorted(self.results.keys())]
        precision_scores.sort(key=lambda x: x[1])  # Sort by std (lower is better)
        
        report.append("Top 3 most precise timestep sizes:")
        for i, (ts, std) in enumerate(precision_scores[:3]):
            report.append(f"  {i+1}. n={ts} (std = {std:.6f})")
        
        # Cost-effectiveness analysis
        report.append("")
        report.append("Cost-effectiveness analysis:")
        for ts in [5, 10, 15, 20]:
            if ts in self.results:
                std = self.results[ts]['std']
                efficiency = 1.0 / (ts * std)  # Higher is better
                report.append(f"  n={ts}: precision={std:.6f}, efficiency={efficiency:.4f}")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        print(report_text)
        return report_text
    
    def save_results(self, save_path):
        """Save raw results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for timestep, result in self.results.items():
            json_results[str(timestep)] = {
                'nnrd_scores': result['nnrd_scores'],
                'mean': float(result['mean']),
                'std': float(result['std']),
                'median': float(result['median']),
                'q25': float(result['q25']),
                'q75': float(result['q75']),
                'min': float(result['min']),
                'max': float(result['max']),
                'n_successful': int(result['n_successful'])
            }
        
        # Add metadata
        output_data = {
            'metadata': {
                'dataset': self.base_args.dataset,
                'layer_type': self.base_args.layer_type,
                'structure': self.base_args.structure,
                'n_bootstrap': self.n_bootstrap,
                'timestep_sizes_tested': self.timestep_sizes,
                'analysis_date': datetime.now().isoformat()
            },
            'results': json_results
        }
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {save_path}")


def create_base_args():
    """Create base arguments for the analysis."""
    parser = argparse.ArgumentParser()
    
    # Add all the arguments from the original script
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--layer_type", type=str, default="gin")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--use_linear", type=bool, default=False)
    parser.add_argument("--structure", type=bool, default=False)
    parser.add_argument("--pos_dim", type=int, default=16)
    parser.add_argument("--n_noise_levels", type=int, default=10)  # Will be overridden
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--fixed_test", type=bool, default=False)
    parser.add_argument("--fixed_train", type=bool, default=False)
    parser.add_argument("--random_noise_pe", type=bool, default=False)
    parser.add_argument("--top_model", type=str, default=None)
    parser.add_argument("--config", type=str, default="default")
    
    return parser


def main():
    parser = argparse.ArgumentParser(description="Timestep Sensitivity Analysis for Noise-Noise Analysis")
    
    # Analysis-specific arguments
    parser.add_argument("--timestep_sizes", nargs="+", type=int, 
                       default=[i for i in range(4,100)], 
                       help="List of timestep sizes to test")
    parser.add_argument("--n_bootstrap", type=int, default=3,
                       help="Number of bootstrap samples per timestep size")
    parser.add_argument("--output_dir", type=str, default="timestep_analysis",
                       help="Output directory for results")
    
    # Base experiment arguments
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--layer_type", type=str, default="gin")
    parser.add_argument("--structure", action="store_true", help="Include positional encodings")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--n_repeats", type=int, default=5)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create base arguments for the noise analysis
    base_parser = create_base_args()
    base_args = base_parser.parse_args([])
    
    # Override with user values
    base_args.dataset = args.dataset
    base_args.layer_type = args.layer_type
    base_args.structure = args.structure
    base_args.epochs = args.epochs
    base_args.n_repeats = args.n_repeats
    
    # Run analysis
    analyzer = TimestepSensitivityAnalyzer(
        base_args=base_args,
        timestep_sizes=args.timestep_sizes,
        n_bootstrap=args.n_bootstrap
    )
    
    analyzer.run_sensitivity_analysis()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"timestep_analysis_{args.dataset}_{args.layer_type}_{timestamp}"
    
    results_path = os.path.join(args.output_dir, f"{base_filename}.json")
    analyzer.save_results(results_path)
    
    plot_path = os.path.join(args.output_dir, f"{base_filename}.png")
    analyzer.plot_results(save_path=plot_path)
    
    report_path = os.path.join(args.output_dir, f"{base_filename}_report.txt")
    analyzer.generate_report(save_path=report_path)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()