import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from typing import List, Tuple

from syslevel.util import METRICS, load_matrices, bootstrap_system_scores


def align_samples(
    samples_all: np.ndarray,
    samples_judged: np.ndarray,
    row_labels_all: List[str],
    row_labels_judged: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    means = np.mean(samples_judged, axis=1)
    means = [(mean, label) for mean, label in zip(means, row_labels_judged)]
    means.sort(key=lambda t: t[0])

    new_all = np.empty(samples_all.shape)
    new_judged = np.empty(samples_judged.shape)
    for i, (_, label) in enumerate(means):
        index = row_labels_all.index(label)
        new_all[i] = samples_all[index]

        index = row_labels_judged.index(label)
        new_judged[i] = samples_judged[index]

    return new_all, new_judged


def calculate_average_variance(samples: np.ndarray) -> float:
    return np.mean(np.var(samples, axis=1))


def convert_to_confidence_interval(samples: np.ndarray) -> List[List[float]]:
    cis = []
    for i in range(samples.shape[0]):
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)
        cis.append([value for value in samples[i] if lower <= value <= upper])
    return cis


def main(args):
    num_iterations = 1000
    fontsize = 16

    Xs_all, row_labels_all = load_matrices(args.all_metrics_jsonl, False, METRICS)
    Xs_judged, row_labels_judged = load_matrices(args.judged_metrics_jsonl, True, METRICS)

    assert sorted(row_labels_all) == sorted(row_labels_judged)
    num_systems = len(row_labels_all)

    for X_all, X_judged, metric in zip(Xs_all, Xs_judged, METRICS):
        samples_all = bootstrap_system_scores(X_all, num_iterations)
        samples_judged = bootstrap_system_scores(X_judged, num_iterations)

        samples_all, samples_judged = align_samples(samples_all, samples_judged, row_labels_all, row_labels_judged)

        variance_all = calculate_average_variance(samples_all)
        variance_judged = calculate_average_variance(samples_judged)
        reduction = (variance_judged - variance_all) / variance_judged * 100
        print(f"{metric}: {reduction:.2f}%")

        samples_all = convert_to_confidence_interval(samples_all)
        samples_judged = convert_to_confidence_interval(samples_judged)

        plt.figure(figsize=(8, 3))
        plt.violinplot(samples_judged, positions=[i for i in range(num_systems)])
        plt.violinplot(samples_all, positions=[i + 0.35 for i in range(num_systems)])

        patches = [
            Patch(facecolor="tab:blue", edgecolor="tab:blue", label="Judged Instances"),
            Patch(facecolor="tab:orange", edgecolor="tab:orange", label="All Test Instances"),
        ]
        plt.legend(handles=patches, loc="lower right", fontsize=fontsize)

        plt.xticks([])
        plt.yticks(fontsize=fontsize)
        plt.xlabel("Systems", fontsize=fontsize)
        plt.ylabel(metric, fontsize=fontsize)
        plt.tight_layout()

        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(f"{args.output_dir}/{metric}.pdf")
        plt.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--all-metrics-jsonl", required=True)
    argp.add_argument("--judged-metrics-jsonl", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)