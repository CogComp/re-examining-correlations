import gzip
import json
import numpy as np
from typing import List, Tuple

METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "QAEval"]
SMALL_METRICS = ["ROUGE-1", "BERTScore", "QAEval"]
ROUGE_METRICS = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
GROUND_TRUTH = "ground-truth"

COLOR_MAP = {
    'ROUGE-1': '#ed5564',
    'ROUGE-2': '#ff965a',
    'ROUGE-L': '#ad92eb',
    'BERTScore': '#50c1e7',
    'QAEval': '#a0d569',
    GROUND_TRUTH: '#ad92eb'
}


def load_matrices(input_file: str, require_parallel: bool, metrics: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    instance_ids = set()
    summarizer_ids = set()
    metrics_dict = {}

    with gzip.open(input_file, "r") as f:
        for line in f:
            instance = json.loads(line.decode())

            instance_id = instance["instance_id"]
            summarizer_id = instance["summarizer_id"]
            key = (instance_id, summarizer_id)

            instance_ids.add(instance_id)
            summarizer_ids.add(summarizer_id)

            metrics_dict[key] = instance["metrics"]

    instance_ids = sorted(instance_ids)
    summarizer_ids = sorted(summarizer_ids)

    m = len(summarizer_ids)
    n = len(instance_ids)
    matrices = []
    for metric in metrics:
        matrix = np.empty((m, n))
        for i, summarizer_id in enumerate(summarizer_ids):
            for j, instance_id in enumerate(instance_ids):
                key = (instance_id, summarizer_id)
                if require_parallel:
                    matrix[i, j] = metrics_dict[key][metric]
                else:
                    if key not in metrics_dict or metric not in metrics_dict[key]:
                        matrix[i, j] = np.nan
                    else:
                        matrix[i, j] = metrics_dict[key][metric]
        matrices.append(matrix)

    return matrices, summarizer_ids


def bootstrap_system_scores(X: np.ndarray, num_iterations: int) -> np.ndarray:
    N = X.shape[0]
    samples = np.empty((N, num_iterations))

    for i in range(N):
        # Take just the non-nan scores
        scores = X[i, ~np.isnan(X[i])]
        M = len(scores)

        for j in range(num_iterations):
            columns = np.random.choice(M, M, replace=True)
            samples[i, j] = np.mean(scores[columns])

    return samples


def get_dataset_title(dataset: str) -> str:
    if dataset == "summeval":
        return "SummEval (Fabbri et al., 2021)"
    elif dataset == "bhandari2020":
        return "realsumm (Bhandari et al., 2020)"
    else:
        raise Exception()