import os
import warnings
import argparse
import json
import numpy as np
from tqdm import tqdm
from supervised_functions import evaluate_main  # Adapted to use the provided supervised code
import wandb

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to save experiment results
def save_run(performance_dict):
    if "results" not in os.listdir():
        os.mkdir("results")
    if "results-linear" not in os.listdir():
        os.mkdir("results-linear")

    linear = '-linear' if performance_dict['linear'] else ""
    json_path = f"results{linear}/{performance_dict['dataset']}{linear}-portional-shuffling.json"
    with open(json_path, "w") as f:
        json.dump(performance_dict, f)

# Main evaluation function
def evaluate_dataset(dataset, n_noise_levels=10, n_repeats=5, use_linear=False):
    wandb.init(
        project="graph-level-evaluation" + ("-linear" if use_linear else ""),
        name=dataset,
        config={
            "dataset": dataset,
            "n_noise_levels": n_noise_levels,
            "n_repeats": n_repeats,
        }
    )

    result_dict = {"dataset": dataset}
    structure_performances = {}
    feature_performances = {}
    ts = np.linspace(0, 1, n_noise_levels)

    for ti in tqdm(range(n_noise_levels), desc=f"Evaluating {dataset}"):
        structure_scores = []
        feature_scores = []
        repeat_pbar = tqdm(range(n_repeats), desc="Running repeats", leave=False)
        for i_repeat in repeat_pbar:
            # Evaluate with structure noise
            struc_score, task_type = evaluate_main(
                dataset=dataset,
                t_structure=ts[ti],
                t_feature=0.0,  # No feature noise in this run
                linear=use_linear,
            )
            structure_scores.append(struc_score)

            # Evaluate with feature noise
            feat_score, _ = evaluate_main(
                dataset=dataset,
                t_structure=0.0,  # No structure noise in this run
                t_feature=ts[ti],
                linear=use_linear,
            )
            feature_scores.append(feat_score)

            # Update progress bar
            repeat_pbar.set_postfix_str(f"Struc: {struc_score:.4f}, Feat: {feat_score:.4f}")

        # Log intermediate results to wandb
        wandb.log({
            "noise_level": ts[ti],
            "structure_performance_mean": np.mean(structure_scores),
            "feature_performance_mean": np.mean(feature_scores),
        })

        structure_performances[ts[ti]] = structure_scores
        feature_performances[ts[ti]] = feature_scores

    result_dict["structure"] = structure_performances
    result_dict["feature"] = feature_performances
    result_dict["task_type"] = task_type
    result_dict["linear"] = use_linear

    # Save results
    save_run(result_dict)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate dataset with noise and repeats")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to evaluate (e.g., 'ogbg-molclintox')"
    )
    parser.add_argument(
        "--n_noise_levels",
        type=int,
        default=10,
        help="The number of noise levels to evaluate (default: 10)"
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="The number of repeats for each noise level (default: 5)"
    )
    parser.add_argument(
        "--use_linear",
        type=bool,
        default=False,
        help="Use the feature projector for linear models instead of supervised training"
    )
    args = parser.parse_args()

    evaluate_dataset(
        dataset=args.dataset,
        n_noise_levels=args.n_noise_levels,
        n_repeats=args.n_repeats,
        use_linear=args.use_linear,
    )
