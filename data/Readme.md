This directory contains the pre-computed metric scores for each of the summaries in SummEval and REALSumm:

- `data/{summeval,realsumm}/{all,judged}/metrics.jsonl.gz`: The ROUGE, BERTScore, and QAEval scores for the summaries in the respective datasets.
The `all` directory contains the scores for all of the system summaries on the test set.
The `judged` directory contains only the scores for the summaries which were also judged by humans for their quality.
These `metrics.jsonl.gz` files also contain the ground-truth quality score.

The scripts here can also be used to re-create these files.
Doing so requires installing `sacrerouge` and `repro`, which requires Docker.

From the root of the repository, run:
```shell script
sh data/setup.sh
```
This will create the raw datasets using SacreROUGE then score each summary with ROUGE, BERTScore, and QAEval.
The `NUM_DEVICES` variable in the script configures how many GPUs will be used to score the summaries in parallel.