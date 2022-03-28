import argparse
from collections import defaultdict

from sacrerouge.io import JsonlReader, JsonlWriter


def main(args):
    metrics_dict = defaultdict(dict)

    for input_jsonl in args.input_jsonls:
        with JsonlReader(input_jsonl) as f:
            for instance in f:
                instance_id = instance["instance_id"]
                summarizer_id = instance["summarizer_id"]
                key = (instance_id, summarizer_id)

                if args.dataset == "summeval":
                    if "rouge-1" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-1"] = instance["metrics"]["rouge-1"][
                            "f1"
                        ]
                    if "rouge-2" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-2"] = instance["metrics"]["rouge-2"][
                            "f1"
                        ]
                    if "rouge-l" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-L"] = instance["metrics"]["rouge-l"][
                            "f1"
                        ]
                    if "bertscore" in instance["metrics"]:
                        metrics_dict[key]["BERTScore"] = (
                            instance["metrics"]["bertscore"]["recall"] * 100
                        )
                    if "qa-eval" in instance["metrics"]:
                        metrics_dict[key]["QAEval"] = (
                            instance["metrics"]["qa-eval"]["f1"] * 100
                        )
                    if "ground-truth" in instance["metrics"]:
                        metrics_dict[key]["ground-truth"] = instance["metrics"][
                            "ground-truth"
                        ]

                elif args.dataset == "realsumm":
                    if "rouge-1" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-1"] = instance["metrics"]["rouge-1"][
                            "recall"
                        ]
                    if "rouge-2" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-2"] = instance["metrics"]["rouge-2"][
                            "recall"
                        ]
                    if "rouge-l" in instance["metrics"]:
                        metrics_dict[key]["ROUGE-L"] = instance["metrics"]["rouge-l"][
                            "recall"
                        ]
                    if "bertscore" in instance["metrics"]:
                        metrics_dict[key]["BERTScore"] = (
                            instance["metrics"]["bertscore"]["recall"] * 100
                        )
                    if "qa-eval" in instance["metrics"]:
                        metrics_dict[key]["QAEval"] = (
                            instance["metrics"]["qa-eval"]["f1"] * 100
                        )
                    if "ground-truth" in instance["metrics"]:
                        metrics_dict[key]["ground-truth"] = instance["metrics"][
                            "ground-truth"
                        ]
                else:
                    raise Exception()

    with JsonlWriter(args.output_jsonl) as out:
        for (instance_id, summarizer_id), metrics in metrics_dict.items():
            out.write(
                {
                    "instance_id": instance_id,
                    "summarizer_id": summarizer_id,
                    "summarizer_type": "peer",
                    "metrics": metrics,
                }
            )


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonls", required=True, nargs="+")
    argp.add_argument("--dataset", required=True)
    argp.add_argument("--output-jsonl", required=True)
    args = argp.parse_args()
    main(args)
