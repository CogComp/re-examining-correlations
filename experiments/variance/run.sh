DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

datasets=( 'summeval' 'realsumm' )

for dataset in "${datasets[@]}"; do
  python -m syslevel.variance.run \
    --all-metrics-jsonl data/${dataset}/all/metrics.jsonl.gz \
    --judged-metrics-jsonl data/${dataset}/judged/metrics.jsonl.gz \
    --output-dir ${DIR}/output/${dataset}
done