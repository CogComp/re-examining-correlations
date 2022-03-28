import argparse
from sacrerouge.io import JsonlReader, JsonlWriter


def main(args):
    with JsonlWriter(args.output_jsonl) as out:
        with JsonlReader(args.input_jsonl) as f:
            for instance in f:
                if "expert" in instance["metrics"]:
                    # SummEval
                    score = instance["metrics"]["expert"]["relevance"]
                elif "litepyramid" in instance["metrics"]:
                    # REALSumm
                    score = instance["metrics"]["litepyramid"]["recall"]
                else:
                    raise Exception()

                del instance["metrics"]
                instance["metrics"] = {"ground-truth": score}
                out.write(instance)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonl", required=True)
    argp.add_argument("--output-jsonl", required=True)
    args = argp.parse_args()
    main(args)
