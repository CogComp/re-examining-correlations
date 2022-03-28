set -e

sacrerouge setup-dataset fabbri2020 temp/summeval
sacrerouge setup-dataset bhandari2020 temp/realsumm

mkdir -p data/summeval/judged data/summeval/all
mkdir -p data/realsumm/judged data/realsumm/all

cat temp/summeval/summaries.jsonl | gzip > data/summeval/judged/summaries.jsonl.gz
cp temp/summeval/all-summaries-orig-refs.jsonl.gz data/summeval/all/summaries.jsonl.gz

python data/select_ground_truth.py \
  --input-jsonl temp/summeval/metrics.jsonl \
  --output-jsonl data/summeval/judged/ground-truth.jsonl

cat temp/realsumm/summaries-mix.jsonl | gzip > data/realsumm/judged/summaries.jsonl.gz
cp temp/realsumm/all-summaries-mix.jsonl.gz data/realsumm/all/summaries.jsonl.gz

python data/select_ground_truth.py \
  --input-jsonl temp/realsumm/metrics-mix.jsonl \
  --output-jsonl data/realsumm/judged/ground-truth.jsonl

rm -r temp

NUM_DEVICES=8

for dataset in summeval realsumm; do
  for split in judged all; do
    python data/split_into_batches.py \
      --input-jsonl data/${dataset}/${split}/summaries.jsonl.gz \
      --num-batches ${NUM_DEVICES} \
      --output-dir data/${dataset}/${split}/temp/summaries

    for ((device=0;device<${NUM_DEVICES};device++)); do
      python data/score.py \
        --input-jsonl data/${dataset}/${split}/temp/summaries/${device}.jsonl.gz \
        --device ${device} \
        --output-dir data/${dataset}/${split}/temp/metrics/${device} \
      &
    done
    wait
  done

  python data/merge.py \
    --input-jsonls \
        data/${dataset}/judged/ground-truth.jsonl \
        data/${dataset}/judged/temp/metrics/*/*.jsonl \
    --dataset ${dataset} \
    --output-jsonl data/${dataset}/judged/metrics.jsonl.gz

  python data/merge.py \
    --input-jsonls \
        data/${dataset}/judged/ground-truth.jsonl \
        data/${dataset}/all/rouge.jsonl \
    --dataset ${dataset} \
    --output-jsonl data/${dataset}/all/metrics.jsonl.gz
done