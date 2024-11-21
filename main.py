import os
import warnings
 # Torch geometric produces future warnings with current version of OGB
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import json
import numpy as np
from functions import evaluate_main


# Function to save experiment runs
def save_run(performance_dict):
    # performance dict should have keys:
    # - dataset
    # - structure: {0:[1, 2, 3, ... , n_repeats], 1:[], 2:[],...}
    # - feature: {0:[], 1:[], 2:[],...}

    if "results" not in os.listdir():
        os.mkdir("results")

    with open(f"results/{performance_dict['dataset']}.json","w") as f:
        json.dump(performance_dict,f)


def evaluate_dataset(dataset, 
                     n_noise_levels = 10,
                     n_repeats = 5):
    result_dict = {"dataset":dataset}

    structure_performances = dict()
    feature_performances   = dict()
    ts = np.linspace(0, 1, n_noise_levels)

    for ti in range(n_noise_levels):
        ti_performances_structure = []
        ti_performances_feature   = []

        for i_repeat in tqdm(range(n_repeats), desc="Running repeats"):
            ti_performances_structure.append(evaluate_main(dataset = dataset,
                                                           t_structure = ts[ti]))
            ti_performances_feature.append(evaluate_main(dataset = dataset,
                                                t_feature = ts[ti]))
            
        structure_performances[ts[ti]] = ti_performances_structure
        feature_performances[ts[ti]]   = ti_performances_feature

    result_dict["structure"] = structure_performances
    result_dict["feature"]   = feature_performances
    save_run(result_dict)

if __name__ == "__main__":
    evaluate_dataset("ogbg-molclintox")