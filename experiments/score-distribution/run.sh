DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

datasets=( 'summeval' 'realsumm' )

for dataset in "${datasets[@]}"; do
  python -m syslevel.score_distribution.run \
    --metrics-jsonl data/${dataset}/all/metrics.jsonl.gz \
    --output-file ${DIR}/output/${dataset}.pdf
done