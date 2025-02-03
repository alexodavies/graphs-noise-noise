import warnings
import argparse
import numpy as np
from tqdm import tqdm
from supervised_functions import evaluate_main
import wandb
from utils import save_run
from metrics import plot_results
# Torch geometric produces future warnings with current version of OGB
warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate_dataset(dataset, n_noise_levels=10, n_repeats=5, use_linear = False):
    use_linear = False # TODO: fix code - currently being set to true by bash script
    pos_included_string = "-pos" if args.structure else ""
    wandb.init(project="noise-synthetics-benchmarks", # + "-linear" if use_linear else "",
               entity="hierarchical-diffusion",
               name="squared-" + args.layer + '-' + dataset + pos_included_string,
                 config={
        "dataset": dataset,
        "n_noise_levels": n_noise_levels,
        "n_repeats": n_repeats,
        "layer_type":args.layer,
        "pos_encodings": args.structure
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
            struc, tt = evaluate_main(dataset=dataset, t_structure=ts[ti],
                                       linear = use_linear, layer_type=args.layer, pos_encodings = args.structure)
            ti_performances_structure.append(struc)
            feat, tt = evaluate_main(dataset=dataset, t_feature=ts[ti],
                                      linear = use_linear, layer_type=args.layer, pos_encodings = args.structure)
            ti_performances_feature.append(feat)
            pbar_string = f"Struc: {struc}, feat: {feat}"
            repeat_pbar.set_postfix_str(pbar_string)

        # Log intermediate results to wandb
        wandb.log({
            "noise_level": ts[ti],
            "structure_performance": np.mean(ti_performances_structure),
            "feature_performance": np.mean(ti_performances_feature),
        })

        structure_performances[str(ts[ti])] = [str(s) for s in ti_performances_structure]
        feature_performances[str(ts[ti])] = [str(f) for f in ti_performances_feature]



    result_dict["structure"] = structure_performances
    result_dict["feature"] = feature_performances
    result_dict["task_type"] = tt
    result_dict["linear"] = use_linear
    result_dict["layer"] = args.layer

    image_path = plot_results(result_dict, extra_save_string=args.layer, return_path=True)

    wandb.log({"Media/Result-Image": wandb.Image(image_path)})

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
        '--use_linear',
        type = bool,
        default = False,
        help = "Whether to use the neural network as a feature projector for linear models instead of normal supervised training"
    )

    parser.add_argument(
        '--structure',
        type = bool,
        default = False,
        help = "Whether to include positional encoding in the model"
    )

    parser.add_argument(
        '--layer',
        type = str,
        default = "gcn",
        help = "The type of GNN layer to use (gcn, gin, gat, gps)"
    )
    args = parser.parse_args()

    evaluate_dataset(dataset=args.dataset, n_noise_levels=args.n_noise_levels, n_repeats=args.n_repeats, use_linear=args.use_linear)
