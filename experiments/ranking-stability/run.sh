DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

datasets=( 'summeval' 'realsumm' )

for dataset in "${datasets[@]}"; do
  python -m syslevel.ranking_stability.run \
    --all-metrics-jsonl data/${dataset}/all/metrics.jsonl.gz \
    --judged-metrics-jsonl data/${dataset}/judged/metrics.jsonl.gz \
    --dataset ${dataset} \
    --output-dir ${DIR}/output
done
