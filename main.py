import os
import warnings
import argparse
import json
import numpy as np
from tqdm import tqdm
from functions import evaluate_main
import wandb

# Torch geometric produces future warnings with current version of OGB
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to save experiment runs
def save_run(performance_dict):
    if "results" not in os.listdir():
        os.mkdir("results")

    # Save locally
    json_path = f"results/{performance_dict['dataset']}.json"
    with open(json_path, "w") as f:
        json.dump(performance_dict, f)

    # # Log to wandb
    # wandb.save(json_path)
    # wandb.log({"performance_json": wandb.Artifact(
    #     f"{performance_dict['dataset']}_results", type="dataset",
    #     description="Results JSON", metadata=performance_dict
    # )})

def evaluate_dataset(dataset, n_noise_levels=10, n_repeats=5):
    wandb.init(project="graph-level-evaluation",
               name=dataset,
                 config={
        "dataset": dataset,
        "n_noise_levels": n_noise_levels,
        "n_repeats": n_repeats,
    })

    result_dict = {"dataset": dataset}
    structure_performances = dict()
    feature_performances = dict()
    ts = np.linspace(0, 1, n_noise_levels)

    for ti in tqdm(range(n_noise_levels), desc=f"Running {dataset}"):
        ti_performances_structure = []
        ti_performances_feature = []
        repeat_pbar = tqdm(range(n_repeats), desc="Running repeats", leave=False)
        for i_repeat in repeat_pbar:
            struc = evaluate_main(dataset=dataset, t_structure=ts[ti])
            ti_performances_structure.append(struc)
            feat = evaluate_main(dataset=dataset, t_feature=ts[ti])
            ti_performances_feature.append(feat)
            pbar_string = f"Struc: {struc}, feat: {feat}"
            repeat_pbar.set_postfix_str(pbar_string)

        # Log intermediate results to wandb
        wandb.log({
            "noise_level": ts[ti],
            "structure_performance": np.mean(ti_performances_structure),
            "feature_performance": np.mean(ti_performances_feature),
        })

        structure_performances[ts[ti]] = [str(s) for s in ti_performances_structure]
        feature_performances[ts[ti]] = [str(f) for f in ti_performances_feature]



    result_dict["structure"] = structure_performances
    result_dict["feature"] = feature_performances

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
    args = parser.parse_args()

    evaluate_dataset(dataset=args.dataset, n_noise_levels=args.n_noise_levels, n_repeats=args.n_repeats)
