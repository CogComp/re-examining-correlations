python -m syslevel.delta_correlations.plot_combined \
  --input-fabbri-jsonl data/summeval/all/metrics.jsonl.gz \
  --input-bhandari-jsonl data/realsumm/all/metrics.jsonl.gz \
  --output-file ${DIR}/output/combined.pdf
