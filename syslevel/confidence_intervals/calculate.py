import argparse
import os
from nlpstats.correlations import bootstrap

from syslevel.util import GROUND_TRUTH, load_matrices


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    paired_inputs = args.paired_inputs.lower() == "true"
    (X, Z), _ = load_matrices(args.input_file, paired_inputs, [args.metric, GROUND_TRUTH])

    for coefficient in ["pearson", "spearman", "kendall"]:
        result = bootstrap(X, Z, "system", coefficient, args.resampling_method, paired_inputs=paired_inputs)

        with open(f"{args.output_dir}/{coefficient}.txt", "w") as out:
            for sample in result.samples:
                out.write(str(sample) + "\n")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-file", required=True)
    argp.add_argument("--metric", required=True)
    argp.add_argument("--resampling-method", required=True)
    argp.add_argument("--paired-inputs", required=True)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)