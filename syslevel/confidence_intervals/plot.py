import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from glob import glob
from matplotlib.patches import Patch

COEFS = ["kendall"]
ALL_COLOR = "#1f77b4"
JUDGED_COLOR = "#ff7f0e"


def load_confidence_intervals(input_dir: str):
    samples_dict = defaultdict(lambda: defaultdict(dict))
    metrics = set()
    for metric_dir in glob(f"{input_dir}/*"):
        path = metric_dir.split("/")
        metric = path[-1]
        if metric == "QAEval":
            metric = "QAEval-F$_1$"
        metrics.add(metric)

        for coef in COEFS:
            samples = list(
                map(float, open(f"{metric_dir}/{coef}.txt", "r").read().splitlines())
            )
            lower = np.percentile(samples, 2.5)
            upper = np.percentile(samples, 97.5)
            samples = np.array(list(filter(lambda x: lower <= x <= upper, samples)))
            samples_dict[metric][coef] = samples
    return samples_dict


def set_colors(parts):
    # Color hacks
    for i, pc in enumerate(parts["bodies"]):
        if i % 2 == 1:
            pc.set_color(JUDGED_COLOR)
    parts["cbars"].set_color([ALL_COLOR, JUDGED_COLOR])
    parts["cmaxes"].set_color([ALL_COLOR, JUDGED_COLOR])
    parts["cmins"].set_color([ALL_COLOR, JUDGED_COLOR])


def main(args):
    fontsize = 12
    plt.rcParams.update({"font.size": fontsize})

    fab_all = load_confidence_intervals(args.summeval_all)
    fab_judged = load_confidence_intervals(args.summeval_judged)
    bha_all = load_confidence_intervals(args.realsumm_all)
    bha_judged = load_confidence_intervals(args.realsumm_judged)

    metrics = sorted(bha_all.keys())
    os.makedirs(args.output_dir, exist_ok=True)
    for coef in COEFS:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(6, 6))

        data1, data2 = [], []
        positions = []
        ticks = []
        labels = []
        fab_differences = []
        bha_differences = []
        for i, metric in enumerate(reversed(metrics)):
            data1.append(fab_all[metric][coef])
            data1.append(fab_judged[metric][coef])
            data2.append(bha_all[metric][coef])
            data2.append(bha_judged[metric][coef])
            positions.append(i * 2 + 0.15)
            positions.append(i * 2 + 0.85)
            ticks.append(i * 2 + 0.5)
            labels.append(f"{metric}")

            fab_judged_width = max(fab_judged[metric][coef]) - min(
                fab_judged[metric][coef]
            )
            fab_all_width = max(fab_all[metric][coef]) - min(fab_all[metric][coef])
            bha_judged_width = max(bha_judged[metric][coef]) - min(
                bha_judged[metric][coef]
            )
            bha_all_width = max(bha_all[metric][coef]) - min(bha_all[metric][coef])

            fab_diff = (fab_judged_width - fab_all_width) / fab_judged_width
            bha_diff = (bha_judged_width - bha_all_width) / bha_judged_width

            print(metric)
            print(f"Fabbri: {fab_judged_width} -> {fab_all_width} = {fab_diff}")
            print(f"Bhandari: {bha_judged_width} -> {bha_all_width} = {bha_diff}")

            fab_differences.append(fab_diff)
            bha_differences.append(bha_diff)

        print(coef)
        print("Fabbri width difference", fab_differences, np.mean(fab_differences))
        print("Bhandari width difference", bha_differences, np.mean(bha_differences))

        parts1 = ax1.violinplot(data1, positions=positions, vert=False)
        parts2 = ax2.violinplot(data2, positions=positions, vert=False)
        set_colors(parts1)
        set_colors(parts2)

        ax1.set_title("SummEval")
        ax2.set_title("REALSumm")

        ax1.set_yticks(ticks)
        ax1.set_yticklabels(labels)

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)

        coef_name = coef[0].upper() + coef[1:]
        plt.xlabel(f"System-Level Correlation")

        legend = [
            Patch(facecolor=JUDGED_COLOR, label="$M_\\mathrm{jud}$ Inputs"),
            Patch(facecolor=ALL_COLOR, label="$M_\\mathrm{test}$ Inputs"),
        ]
        ax1.legend(handles=legend, loc="upper left")

        plt.tight_layout()
        fig.savefig(f"{args.output_dir}/{coef}.pdf", bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--summeval-all", required=True)
    argp.add_argument("--summeval-judged", required=True)
    argp.add_argument("--realsumm-all", required=True)
    argp.add_argument("--realsumm-judged", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)
