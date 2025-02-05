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


def evaluate_dataset(args):

    dataset = args.dataset

    if "synth" in dataset:
        project = "performance-synthetics-benchmarks"
    elif "ogbg" in dataset:
        project = "performance-ogbg"
    elif "TU" in dataset:
        project = "performance-TUDatasets"

    use_linear = False  # TODO: fix code - currently being set to true by bash script
    pos_included_string = "-pos" if args.structure else ""
    wandb.init(project= project, # "noise-synthetics-benchmarks",  # + "-linear" if use_linear else "",
               entity="hierarchical-diffusion",
               name=args.layer_type + '-' + dataset + pos_included_string,
               config=args)

    # result_dict = {"dataset": dataset}
    # structure_performances = dict()
    # feature_performances = dict()
    # ts = np.linspace(0, 1, n_noise_levels)

    performance, tt = evaluate_main(args, eval_on_val=True)

    # For sweeps, its more convenient is less is always better
    if "classification" in tt:
        performance = 1-performance

    wandb.log({"Performance":performance})



# # dataset: str = "ogbg-molclintox",
# #                   layer_type: str = "gin",
# #                   hidden_dim: int = 100,
# #                   num_layers: int = 3,
# #                   batch_size: int = 512,
# #                   epochs: int = 25,
# #                   lr: float = 0.001,
# #                   t_structure: float = 0.,
# #                   t_feature: float = 0.,
# #                   linear: bool = False,
# #                   pos_encodings: bool = False,
# #                   pos_dim: int = 20,
# #                   avoid_cuda: bool = True

#     for ti in tqdm(range(n_noise_levels), desc=f"Running {dataset}"):
#         ti_performances_structure = []
#         ti_performances_feature = []
#         repeat_pbar = tqdm(
#             range(n_repeats), desc="Running repeats", leave=False)
#         for i_repeat in repeat_pbar:
            
#             # Same data for t = 0
#             if ti == 0:
#                 # struc, tt = evaluate_main(dataset=dataset, t_structure=ts[ti],
#                 #                 linear=use_linear, layer_type=args.layer, pos_encodings=args.structure)
#                 struc, tt = evaluate_main(args)
#                 ti_performances_structure.append(struc)
#                 ti_performances_feature.append(struc)

#                 pbar_string = f"Struc: {struc}, feat: {struc}"
#                 repeat_pbar.set_postfix_str(pbar_string)

#                 continue

#             struc, tt = evaluate_main(args, t_structure = ts[ti])
#             ti_performances_structure.append(struc)



#             # feat, tt = evaluate_main(dataset=dataset, t_feature=ts[ti],
#             #                          linear=use_linear, layer_type=args.layer, pos_encodings=args.structure)
#             feat, tt = evaluate_main(args, t_feature = ts[ti])
#             ti_performances_feature.append(feat)

#             pbar_string = f"Struc: {struc}, feat: {feat}"
#             repeat_pbar.set_postfix_str(pbar_string)

#         # Log intermediate results to wandb
#         wandb.log({
#             "noise_level": ts[ti],
#             "structure_performance": np.mean(ti_performances_structure),
#             "feature_performance": np.mean(ti_performances_feature),
#         })

#         structure_performances[str(ts[ti])] = [str(s)
#                                                for s in ti_performances_structure]
#         feature_performances[str(ts[ti])] = [str(f)
#                                              for f in ti_performances_feature]

#     result_dict["structure"] = structure_performances
#     result_dict["feature"] = feature_performances
#     result_dict["task_type"] = tt
#     result_dict["linear"] = use_linear
#     result_dict["layer"] = args.layer

#     image_path = plot_results(
#         result_dict, extra_save_string=args.layer, return_path=True)

#     wandb.log({"Media/Result-Image": wandb.Image(image_path)})

#     save_run(result_dict)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate dataset with noise and repeats")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molclintox",
        help="The name of the dataset to evaluate (e.g., 'ogbg-molclintox')"
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="gin",
        help="The type of GNN layer to use (e.g., 'gcn', 'gin', 'gat', 'gps')"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=100,
        help="The hidden dimension size of the model (default: 100)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="The number of layers in the model (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training (default: 512)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        '--use_linear',
        type=bool,
        default=False,
        help="Whether to use the neural network as a feature projector for linear models instead of normal supervised training"
    )
    parser.add_argument(
        '--structure',
        type=bool,
        default=False,
        help="Whether to include positional encoding in the model"
    )
    parser.add_argument(
        "--pos_dim",
        type=int,
        default=16,
        help="Dimension of the positional encodings (default: 16)"
    )
    parser.add_argument(
        '--no_cuda',
        type=bool,
        default=False,
        help="Whether to avoid using the GPU"
    )

    args = parser.parse_args()
    print(args)
    # quit()

    evaluate_dataset(args)
