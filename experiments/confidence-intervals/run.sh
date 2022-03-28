DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e

datasets=( 'summeval' 'realsumm' )
metrics=( 'BERTScore' 'ROUGE-1' 'ROUGE-2' 'ROUGE-L' 'QAEval' )
methods=( 'systems' 'inputs' 'both' )

for dataset in "${datasets[@]}"; do
  for metric in "${metrics[@]}"; do
    for method in "${methods[@]}"; do
      python -m syslevel.confidence_intervals.calculate \
        --input-file data/${dataset}/judged/metrics.jsonl.gz \
        --metric ${metric} \
        --resampling-method ${method} \
        --paired-inputs true \
        --output-dir ${DIR}/output/${dataset}/judged/correlations/${method}/${metric} \
      &

      python -m syslevel.confidence_intervals.calculate \
        --input-file data/${dataset}/all/metrics.jsonl.gz \
        --metric ${metric} \
        --resampling-method ${method} \
        --paired-inputs false \
        --output-dir ${DIR}/output/${dataset}/all/correlations/${method}/${metric} \
      &
    done
  done
  wait
done

for method in "${methods[@]}"; do
  python -m syslevel.confidence_intervals.plot \
    --summeval-all ${DIR}/output/summeval/all/correlations/${method} \
    --summeval-judged ${DIR}/output/summeval/judged/correlations/${method} \
    --realsumm-all ${DIR}/output/realsumm/all/correlations/${method} \
    --realsumm-judged ${DIR}/output/realsumm/judged/correlations/${method} \
    --output-dir ${DIR}/output/plots/${method}
done