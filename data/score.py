import argparse
import gzip
import json
import os
from repro.models.deutsch2021 import QAEval
from repro.models.lin2004 import ROUGE
from repro.models.zhang2020 import BERTScore
from typing import List


def save(inputs: List, micro: List, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as out:
        for inp, metrics in zip(inputs, micro):
            out.write(json.dumps({
                "instance_id": inp["instance_id"],
                "summarizer_id": inp["summarizer_id"],
                "metrics": metrics
            }) + "\n")


def main(args):
    inputs = []
    with gzip.open(args.input_jsonl, "r") as f:
        for line in f:
            instance = json.loads(line.decode())

            instance_id = instance["instance_id"]
            summarizer_id = instance["summarizer_id"]

            candidate = instance["summary"]["text"]
            if "reference" in instance:
                reference = instance["reference"]["text"]
            else:
                reference = instance["references"][0]["text"]

            inputs.append({
                "instance_id": instance_id,
                "summarizer_id": summarizer_id,
                "candidate": candidate,
                "references": [reference]
            })

    metric = ROUGE()
    _, micro = metric.predict_batch(inputs)
    save(inputs, micro, f"{args.output_dir}/rouge.jsonl")

    metric = BERTScore(device=args.device)
    _, micro = metric.predict_batch(inputs)
    save(inputs, micro, f"{args.output_dir}/bertscore.jsonl")

    metric = QAEval(device=args.device)
    _, micro = metric.predict_batch(inputs)
    save(inputs, micro, f"{args.output_dir}/qaeval.jsonl")


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("--input-jsonl", required=True)
    argp.add_argument("--device", required=True, type=int)
    argp.add_argument("--output-dir", required=True)
    args = argp.parse_args()
    main(args)