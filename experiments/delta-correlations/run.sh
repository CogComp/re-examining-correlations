DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

datasets=( 'summeval' 'realsumm' )

for dataset in "${datasets[@]}"; do
  python -m syslevel.delta_correlations.plot_heatmaps \
    --input-jsonl data/${dataset}/all/metrics.jsonl.gz \
    --dataset ${dataset} \
    --output-dir ${DIR}/output/${dataset}
done

python -m syslevel.delta_correlations.plot_combined \
  --input-fabbri-jsonl data/summeval/all/metrics.jsonl.gz \
  --input-bhandari-jsonl data/realsumm/all/metrics.jsonl.gz \
  --output-file ${DIR}/output/combined.pdf

python -m syslevel.delta_correlations.plot_combined \
  --input-fabbri-jsonl data/summeval/all/metrics.jsonl.gz \
  --input-bhandari-jsonl data/realsumm/all/metrics.jsonl.gz \
  --rouge-only \
  --output-file ${DIR}/output/rouge-combined.pdf