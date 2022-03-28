import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from syslevel.util import COLOR_MAP, GROUND_TRUTH, SMALL_METRICS, load_matrices


def main(args):
    fontsize = 14

    plt.rcParams.update({"font.size": fontsize})

    metrics = [GROUND_TRUTH] + SMALL_METRICS
    Xs, _ = load_matrices(args.metrics_jsonl, False, metrics)

    fig, axes = plt.subplots(len(metrics), 1)
    for i, (ax, metric, X) in enumerate(zip(axes, metrics, Xs)):
        name = metric if metric != GROUND_TRUTH else "Human Judgment"

        scores = np.nanmean(X, axis=1)

        ax.scatter(scores, [1] * len(scores), label=name, color=COLOR_MAP[metric])
        ax.text(
            0.01,
            0.01,
            name,
            verticalalignment="bottom",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=fontsize,
        )

        ax.set_yticks([])
        ax.set_yticklabels([])

    axes[-1].set_xlabel("Metric Value")

    plt.tight_layout(pad=0, h_pad=1.5)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    plt.savefig(args.output_file)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--metrics-jsonl", required=True)
    argp.add_argument("--output-file", required=True)
    args = argp.parse_args()
    main(args)
