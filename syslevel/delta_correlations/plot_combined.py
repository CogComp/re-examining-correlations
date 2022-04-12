import argparse
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from typing import Dict, List

from syslevel.delta_correlations.plot_heatmaps import (
    get_all_pairs,
    get_percentile_deltas,
    calculate_tau,
)
from syslevel.util import (
    COLOR_MAP,
    GROUND_TRUTH,
    SMALL_METRICS,
    ROUGE_METRICS,
    load_matrices,
    get_dataset_title,
)


def run_05_delta(input_jsonl: str) -> None:
    (X, Z), _ = load_matrices(input_jsonl, False, ["ROUGE-1", GROUND_TRUTH])

    x = np.nanmean(X, axis=1)
    z = np.nanmean(Z, axis=1)
    pairs = get_all_pairs(x, z)

    bucket = [pair for pair in pairs if abs(pair[2]) <= 0.5]
    print(calculate_tau(bucket))


def load_data(input_jsonl: str, metrics: List[str]) -> Dict:
    Xs, _ = load_matrices(input_jsonl, False, [GROUND_TRUTH] + metrics)
    Z = Xs[0]
    Xs = Xs[1:]

    metric_to_data = {}

    for X, metric in zip(Xs, metrics):
        x = np.nanmean(X, axis=1)
        z = np.nanmean(Z, axis=1)
        pairs = get_all_pairs(x, z)

        _, max_deltas = get_percentile_deltas(pairs)
        correlations = []

        for max_delta in max_deltas:
            bucket = [pair for pair in pairs if abs(pair[2]) <= max_delta]
            correlations.append(calculate_tau(bucket))

        metric_to_data[metric] = {
            "correlations": correlations,
            "max_deltas": max_deltas,
        }
    return metric_to_data


def main(args):
    fontsize = 8
    show_u_values = True

    if args.rouge_only:
        metrics = ROUGE_METRICS
    else:
        metrics = SMALL_METRICS

    plt.rcParams.update({"font.size": fontsize})

    if False:
        run_05_delta(args.input_fabbri_jsonl)
        run_05_delta(args.input_bhandari_jsonl)
        exit()

    data = {
        "fabbri": load_data(args.input_fabbri_jsonl, metrics),
        "bhandari": load_data(args.input_bhandari_jsonl, metrics),
    }

    fig, axes = plt.subplots(2, len(metrics), sharey=True, figsize=(6.3, 2.2))

    for i, dataset in enumerate(["fabbri", "bhandari"]):
        for j, metric in enumerate(metrics):
            first_row = data[dataset][metric]["correlations"]
            x = [10 * (k + 1) for k in range(len(first_row))]
            axes[i, j].plot(
                x, first_row, color=COLOR_MAP[metric], label=metric
            )  # , linewidth=2.5)

            print(dataset, metric, x, [f"{value:.2f}" for value in first_row])

            axes[i, j].set_xticks([0, 20, 40, 60, 80, 100])
            # axes[i, j].set_xticklabels(x, fontsize=fontsize)
            if args.rouge_only:
                axes[i, j].set_yticks([-0.25, 0, 0.25, 0.50, 0.75])
            else:
                axes[i, j].set_yticks([0, 0.25, 0.50, 0.75])
            if i == 1:
                axes[i, j].legend(loc="lower right")  # , fontsize=fontsize)
            axes[i, j].grid()

            if show_u_values:
                twin = axes[i, j].twiny()
                twin.set_xlim(axes[i, j].get_xlim())
                twin.set_xticks([20, 40, 60, 80, 100])
                max_deltas = data[dataset][metric]["max_deltas"]
                us = [
                    max_deltas[1],
                    max_deltas[3],
                    max_deltas[5],
                    max_deltas[7],
                    max_deltas[9],
                ]
                twin.set_xticklabels([f"{u:.1f}" for u in us])

    fig.text(0.52, 0.0, "Percent of System Pairs Used", ha="center")
    fig.text(0.0, 0.5, "System-Level $\\tau$", va="center", rotation="vertical")

    if show_u_values:
        fig.text(0.078, 0.435, "$u=$")
        fig.text(0.078, 0.908, "$u=$")

        fig.text(0.39, 0.435, "$u=$")
        fig.text(0.39, 0.908, "$u=$")

        fig.text(0.7, 0.435, "$u=$")
        fig.text(0.7, 0.908, "$u=$")

        fig.text(0.98, 0.73, "SummEval", va="center", rotation="vertical")
        fig.text(0.98, 0.26, "REALSumm", va="center", rotation="vertical")
    else:
        fig.text(0.98, 0.77, "SummEval", va="center", rotation="vertical")
        fig.text(0.98, 0.30, "REALSumm", va="center", rotation="vertical")

    # fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel("Percent of System Pairs Used")
    # plt.ylabel("System-Level $\\tau$")

    plt.tight_layout()
    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    print(args.output_file)
    plt.savefig(args.output_file)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-fabbri-jsonl", required=True)
    argp.add_argument("--input-bhandari-jsonl", required=True)
    argp.add_argument("--rouge-only", action="store_true")
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
