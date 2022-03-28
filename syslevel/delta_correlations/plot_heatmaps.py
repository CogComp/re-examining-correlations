import argparse
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from typing import Dict, List

from syslevel.util import (
    COLOR_MAP,
    GROUND_TRUTH,
    METRICS,
    SMALL_METRICS,
    load_matrices,
    get_dataset_title,
)


def get_all_pairs(x: np.ndarray, z: np.ndarray) -> List:
    pairs = []
    N = x.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j, x[i] - x[j], z[i] - z[j]))
    return pairs


def calculate_tau(pairs: List):
    #   tau = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
    # where P is the number of concordant pairs, Q the number of discordant
    # pairs, T the number of ties only in `x`, and U the number of ties only in
    # `y`.  If a tie occurs for the same pair in both `x` and `y`, it is not
    # added to either T or U.
    P, Q, T, U = 0, 0, 0, 0
    for _, _, delta_auto, delta_human in pairs:
        if (delta_auto > 0 and delta_human > 0) or (delta_auto < 0 and delta_human < 0):
            P += 1
        elif (delta_auto > 0 and delta_human < 0) or (
            delta_auto < 0 and delta_human > 0
        ):
            Q += 1
        elif delta_auto == 0 and delta_human != 0:
            T += 1
        elif delta_auto != 0 and delta_human == 0:
            U += 1
    if P + Q + T == 0:
        return 0
    tau = (P - Q) / np.sqrt((P + Q + T) * (P + Q + U))
    return tau


def get_percentile_deltas(pairs: List):
    percentile = 0.1
    num_pairs_per_bucket = int(math.ceil(len(pairs) * percentile))
    num_buckets = int(math.ceil(len(pairs) / num_pairs_per_bucket))

    min_deltas = [0.0]
    max_deltas = []
    pairs = sorted(pairs, key=lambda t: abs(t[2]))
    for i in range(num_buckets):
        pair = pairs[min((i + 1) * num_pairs_per_bucket - 1, len(pairs) - 1)]
        delta = pair[2]
        min_deltas.append(abs(delta))
        max_deltas.append(abs(delta))
    min_deltas.pop()
    return min_deltas, max_deltas


def get_colors(metric: str) -> str:
    if metric == "ROUGE-1":
        return "Reds"
    elif metric == "ROUGE-2":
        return "Oranges"
    elif metric == "ROUGE-L":
        return "Purples"
    elif metric == "BERTScore":
        return "Blues"
    elif metric == "QAEval":
        return "Greens"
    raise Exception(f"Unknown metric: {metric}")


def main(args):
    min_pairs = 5
    fontsize = 14
    metrics = METRICS

    Xs, _ = load_matrices(args.input_jsonl, False, [GROUND_TRUTH] + metrics)
    Z = Xs[0]
    Xs = Xs[1:]

    metric_to_data = OrderedDict()

    for X, metric in zip(Xs, metrics):
        x = np.nanmean(X, axis=1)
        z = np.nanmean(Z, axis=1)
        pairs = get_all_pairs(x, z)

        min_deltas, max_deltas = get_percentile_deltas(pairs)
        correlations = np.zeros((len(min_deltas), len(max_deltas)))
        num_pairs = np.zeros((len(min_deltas), len(max_deltas)))
        mask = np.tril(np.ones(correlations.shape), k=-1)

        for i, min_delta in enumerate(min_deltas):
            for j, max_delta in enumerate(max_deltas):
                if min_delta >= max_delta:
                    continue
                bucket = [
                    pair for pair in pairs if min_delta <= abs(pair[2]) <= max_delta
                ]
                num_pairs[i, j] = len(bucket)
                if len(bucket) < min_pairs:
                    mask[i, j] = 1
                else:
                    correlations[i, j] = calculate_tau(bucket)

        metric_to_data[metric] = {
            "correlations": correlations,
            "min_deltas": min_deltas,
            "max_deltas": max_deltas,
        }

        yticklabels = ["{:.1f}".format(value) for value in min_deltas]
        xticklabels = ["{:.1f}".format(value) for value in max_deltas]

        plt.figure()
        ax = sns.heatmap(
            correlations,
            cmap=get_colors(metric),
            annot=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            mask=mask,
            fmt=".2f",
            cbar_kws={"label": "System-Level Correlation (Kendall's $\\tau$)"},
            vmin=-1,
            vmax=1,
        )
        plt.xlabel(f"$u$, Maximum {metric} $\\Delta$", fontsize=fontsize)
        plt.ylabel(f"$\\ell$, Minimum {metric} $\\Delta$", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)
        ax.figure.axes[-1].yaxis.label.set_size(fontsize)
        # ax.collections[0].colorbar.ax.tick_params(labelsize=14)

        title = get_dataset_title(args.dataset)
        plt.title(title, fontsize=fontsize)
        plt.tight_layout(pad=0)

        metric_dir = f"{args.output_dir}/{metric}"
        os.makedirs(metric_dir, exist_ok=True)
        plt.savefig(f"{metric_dir}/correlations.pdf")
        plt.close()

        plt.figure()
        mask = np.tril(np.ones(correlations.shape), k=-1)
        sns.heatmap(
            num_pairs,
            cmap=get_colors(metric),
            annot=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            mask=mask,
            cbar=False,
            fmt=".0f",
        )
        plt.xlabel(f"Maximum {metric} $\\Delta$", fontsize=fontsize)
        plt.ylabel(f"Minimum {metric} $\\Delta$", fontsize=fontsize)
        title = get_dataset_title(args.dataset)
        plt.title(title, fontsize=fontsize)
        plt.tight_layout(pad=0)
        plt.savefig(f"{metric_dir}/num-pairs.pdf")
        plt.close()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonl", required=True)
    argp.add_argument("--dataset", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)
