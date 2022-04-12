# Re-Examining System-Level Correlations of Automatic Summarization Evaluation Metrics
This repository contains the code for the paper "Re-Examining System-Level Correlations of Automatic Summarization Evaluation Metrics."

## Note
If you want to run bootstrapping or permutation tests using the non-paired inputs (e.g., from Section 3 of the paper), the easiest way to do so is through the [`nlpstats`](https://github.com/danieldeutsch/nlpstats) library.

## Python Environment
The code has few dependencies, which are contained in the `requirements.txt`.
You can create a Conda environment for this code by:
```shell script
conda create -n re-examining python=3.7
conda activate re-examining
pip install -r requirements.txt
```

## Data
The data required to produce the results in the experiment is included in the repository.
See the [Readme](data/Readme.md) in the `data` directory for instructions for re-creating the data.

## Experiments
Run the following scripts from the root of the repository to re-create the plots from the paper.
The plots will be saved under the `experiments/<name>/output` directory.

- Figure 2 (the system-level score variances): `sh experiments/variance.sh`
- Figure 3 (the ranking stabilities): `sh experiments/ranking-stability/run.sh`
- Figures 4, 7 and 8 (the confidence intervals): `sh experiments/confidence-intervals/run.sh`
- Figure 5 (the system-level score distributions): `sh experiments/score-distribution/run.sh`
- Figures 6 and 9 (the delta correlations): `sh experiments/delta-correlations/run.sh`
- Figures 10 and 11 (the delta correlations heatmaps): `sh experiments/delta-correlations/run.sh`

## Reproducibility Track
The Docker image created by `Dockerfile` is our submission to the [NAACL 2022 Reproducibility Track](https://naacl2022-reproducibility-track.github.io/).
It will reproduce the results that were plotted in Figure 6.

It can be built and run using the following command:
```shell script
docker build -t re-examining .
docker run -it re-examining
```