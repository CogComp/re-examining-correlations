import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from typing import List, Tuple

from syslevel.util import (
    COLOR_MAP,
    GROUND_TRUTH,
    SMALL_METRICS,
    load_matrices,
    get_dataset_title,
)


def get_num_inputs_list(M: int, scale: str) -> Tuple[List[int], List[int]]:
    if scale == "normal":
        sizes = list(range(10, M)) + [M]
        return sizes, sizes
    else:
        sizes = []
        labels = []
        max_log = np.log10(M)
        for i in np.arange(2, max_log, 0.05):
            sizes.append(int(10**i))
            labels.append(i)
        if sizes[-1] != M:
            sizes.append(M)
            labels.append(max_log)
        return sizes, labels


def sample_self_correlation(
    X: np.ndarray,
    num_inputs_list: List[int],
    num_iterations: int,
) -> np.ndarray:
    N, M = X.shape
    correlations = np.empty((len(num_inputs_list), num_iterations))
    for i, m in enumerate(num_inputs_list):
        for j in range(num_iterations):
            cols1 = np.random.choice(M, m, replace=True)
            cols2 = np.random.choice(M, m, replace=True)
            X_s1 = X[:, cols1]
            X_s2 = X[:, cols2]

            X_s1_mean = np.nanmean(X_s1, axis=1)
            X_s2_mean = np.nanmean(X_s2, axis=1)
            correlations[i, j] = kendalltau(X_s1_mean, X_s2_mean)[0]

    return correlations


def plot(
    ax,
    X: np.ndarray,
    metric: str,
    num_inputs_list: List[int],
    xvalues: List[int],
    num_iterations: int,
):
    correlations = sample_self_correlation(X, num_inputs_list, num_iterations)
    mean = np.mean(correlations, axis=1)
    std = np.std(correlations, axis=1)

    (line,) = ax.plot(xvalues, mean, label=metric, color=COLOR_MAP[metric])
    ax.fill_between(xvalues, mean - std, mean + std, color=COLOR_MAP[metric], alpha=0.2)
    return line


def main(args):
    num_iterations = 1000
    fontsize = 16

    plt.rcParams.update({"font.size": fontsize})

    Xs_all, _ = load_matrices(args.all_metrics_jsonl, False, SMALL_METRICS)
    Xs_judged, _ = load_matrices(
        args.judged_metrics_jsonl, True, [GROUND_TRUTH] + SMALL_METRICS
    )

    N, M_judged = Xs_judged[0].shape
    num_inputs_judged, xlabels_judged = get_num_inputs_list(M_judged, "normal")

    M_all = Xs_all[0].shape[1]
    num_inputs_list_all, xlabels_all = get_num_inputs_list(M_all, "log")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 4.5))

    lines = []
    for metric, X_jud in zip([GROUND_TRUTH] + SMALL_METRICS, Xs_judged):
        line = plot(
            ax1, X_jud, metric, num_inputs_judged, xlabels_judged, num_iterations
        )
        lines.append(line)

    for metric, X_all in zip(SMALL_METRICS, Xs_all):
        plot(ax2, X_all, metric, num_inputs_list_all, xlabels_all, num_iterations)

    ax1.grid()
    ax1.set_xticks([20, 40, 60, 80, 100])
    ax1.set_ylim(0.2, 1.0)
    ax1.set_xlabel("#Instances", fontsize=fontsize)
    ax1.set_ylabel(f"System-Level Correlation", fontsize=fontsize)

    labels = ["Human Judgment"] + SMALL_METRICS
    ax2.legend(lines, labels)
    ax2.grid()
    ax2.set_ylim(0.2, 1.0)
    ax2.set_xticks([2, 2.5, 3, 3.5, 4])
    ax2.set_xlabel("log(#Instances)", fontsize=fontsize)
    ax2.yaxis.set_tick_params(labelleft=False)

    title = get_dataset_title(args.dataset)
    plt.suptitle(title, y=0.95, fontsize=fontsize)
    plt.tight_layout(pad=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/{args.dataset}.pdf"
    print(f"Saving plot to {output_file}")
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--all-metrics-jsonl", required=True)
    argp.add_argument("--judged-metrics-jsonl", required=True)
    argp.add_argument("--dataset", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)
