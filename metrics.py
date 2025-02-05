import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def plot_results(result_dict, extra_save_string="", return_path = False):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    dataset = result_dict["dataset"]
    task_type = result_dict["task_type"]

    ax.set_xlabel("Noise Level")
    perf_string = "RMSE" if task_type == "regression" else "ROC-AUC"
    ax.set_ylabel(perf_string)
    
    # ax.set_title(dataset)

    strucs = result_dict["structure"]
    feats = result_dict["feature"]
    ts = list(strucs.keys())

    # Ensure the x-axis values are numeric and sorted
    ts = sorted([float(t) for t in ts])  # Convert keys to floats and sort them

    struc_means = [np.mean([float(val) for val in strucs[str(t)]]) for t in ts]
    struc_devs = [np.std([float(val) for val in strucs[str(t)]]) for t in ts]
    feat_means = [np.mean([float(val) for val in feats[str(t)]]) for t in ts]
    feat_devs = [np.std([float(val) for val in feats[str(t)]]) for t in ts]

    ax.fill_between(ts, struc_means, feat_means, color = "gray", alpha = 0.4)

    ax.errorbar(ts, struc_means, yerr=struc_devs, label="Structure", c = "black")
    ax.errorbar(ts, feat_means, yerr=feat_devs, label="Feature", c = "blue", linestyle = "dashed")

    nnrde_y_min = min(struc_means[-1], feat_means[-1])
    nnrde_y_max = max(struc_means[-1], feat_means[-1])
    final_gap = np.abs(struc_means[-1] - feat_means[-1])
    nnrd_y_adjusted = nnrde_y_min + 0.35*final_gap

    head_width = final_gap / 10
    head_length = final_gap / 10

    # print(nnrd_y_adjusted)
    ax.arrow(1.025, nnrde_y_min, 0, final_gap,
             length_includes_head = True, color = "black",
             head_width = head_width, head_length = head_length)
    ax.arrow(1.025, nnrde_y_max, 0, -final_gap,
             length_includes_head = True, color = "black",
             head_width = head_length, head_length = head_length)

    # ax.annotate(f"$NNRD_e$:\n{np.around(nnd(result_dict, extremis=True), decimals = 3)}",
    #              xy = (1.05, nnrd_y_adjusted),
    #              color = "black", fontsize = 8)
    
    nnrd_root = min(np.min(np.array(struc_means) - np.array(struc_devs)), np.min(np.array(feat_means)-np.array(feat_devs)))

    ax.text(1.05, nnrd_root, f"$NNRD$:\n{np.around(nnd(result_dict), decimals = 3)}", 
        bbox=dict(facecolor='white', edgecolor='green', boxstyle='round'))

    # Format x-axis ticks
    ax.set_xticks(ts)  # Ensure all unique noise levels are shown
    ax.set_xticklabels([f"{t:.1f}" for t in ts])  # Format as two decimal places

    ax.set_xlim([ts[0], 1.225])

    extra_string = "" if extra_save_string == "" else f"-{extra_save_string}"
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}{extra_string}.png", dpi=600)
    plt.close()


    if return_path:
        return f"figures/{dataset}{extra_string}.png"

def nncr(result_dict):
    dataset = result_dict["dataset"]
    strucs = result_dict["structure"]
    feats = result_dict["feature"]
    ts = list(strucs.keys())
    struc_means = [np.mean([float(val) for val in strucs[str(t)]]) for t in ts]
    struc_devs = [np.std([float(val) for val in strucs[str(t)]]) for t in ts]
    feat_means = [np.mean([float(val) for val in feats[str(t)]]) for t in ts]
    feat_devs = [np.std([float(val) for val in feats[str(t)]]) for t in ts]

    r_structure = np.abs(spearmanr(ts, struc_means)[0])
    r_feature = np.abs(spearmanr(ts, feat_means)[0])
    
    nncr = r_feature  / r_structure

    # print(f"r feature: {r_feature}\n r structure: {r_structure} nncr: {nncr}")

    return np.log10(nncr)

def nnd(result_dict, extremis = False):

    dataset = result_dict["dataset"]
    task = result_dict["task_type"]
    strucs = result_dict["structure"]
    feats = result_dict["feature"]
    ts = list(strucs.keys())
    
    struc_means = [np.mean([float(val) for val in strucs[str(t)]]) for t in ts]
    feat_means = [np.mean([float(val) for val in feats[str(t)]]) for t in ts]

    if extremis:
        struc_means = [struc_means[-1]]
        feat_means = [feat_means[-1]]
    if task == "classification":
        difference = np.array(feat_means) / np.mean(struc_means)
    else:
        difference = np.mean(struc_means) / np.array(feat_means)

    return np.log(np.sum(difference) / difference.shape[0])

def minmax_performance_structure(result_dict):
    dataset = result_dict["dataset"]
    task = result_dict["task_type"]
    strucs = result_dict["structure"]
    ts = list(strucs.keys())
    struc_means = [np.mean([float(val) for val in strucs[str(t)]]) for t in ts]

    # if task == "classification":
    return struc_means[0], struc_means[-1] # np.min(struc_means), np.max(struc_means)
    # else:
        # return np.max(struc_means), np.min(struc_means)

def minmax_performance_feature(result_dict):
    dataset = result_dict["dataset"]
    task = result_dict["task_type"]
    feats = result_dict["feature"]
    ts = list(feats.keys())
    feat_means = [np.mean([float(val) for val in feats[str(t)]]) for t in ts]

    # if task == "classification":
    return feat_means[0], feat_means[-1] # np.min(feat_means), np.max(feat_means)
    # else:
        # return np.max(feat_means), np.min(feat_means)