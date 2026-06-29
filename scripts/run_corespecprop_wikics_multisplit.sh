#!/usr/bin/env bash
set -euo pipefail

SPLIT_INDICES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
METHODS=("autopropcat" "corespecprop")

for split_index in "${SPLIT_INDICES[@]}"; do
  for method in "${METHODS[@]}"; do
    python -m homogcl.train \
      --dataset WikiCS \
      --split public \
      --split-index "${split_index}" \
      --method "${method}" \
      --max-prop-steps 10 \
      --autoprop-plateau-ratio 0.75 \
      --specprop-high-concentration 0.34 \
      --probe sklogreg \
      --logreg-c-grid 0.25,1,4,16 \
      --output-dir results/corespecprop_wikics_multisplit
  done
done

python -m homogcl.compare --input-dirs results/corespecprop_wikics_multisplit --baseline autopropcat --candidate corespecprop --output-csv results/corespecprop_wikics_multisplit_paired.csv
python -m homogcl.diagnose --input-dirs results/corespecprop_wikics_multisplit --baseline autopropcat --candidate corespecprop --output-csv results/corespecprop_wikics_multisplit_diagnostics.csv
python -m homogcl.summarize --input-dir results/corespecprop_wikics_multisplit --output-csv results/corespecprop_wikics_multisplit_summary.csv
