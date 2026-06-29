#!/usr/bin/env bash
set -euo pipefail

DATASETS=("PubMed" "Photo")
SPLIT_SEEDS=(0 1 2 3 4 5 6 7 8 9)
METHODS=("autopropcat" "corespecprop")

for dataset in "${DATASETS[@]}"; do
  for split_seed in "${SPLIT_SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      python -m homogcl.train \
        --dataset "${dataset}" \
        --split class-random \
        --split-seed "${split_seed}" \
        --method "${method}" \
        --max-prop-steps 10 \
        --autoprop-plateau-ratio 0.75 \
        --specprop-high-concentration 0.34 \
        --probe sklogreg \
        --logreg-c-grid 0.25,1,4,16 \
        --output-dir results/corespecprop_key_multisplit
    done
  done
done

python -m homogcl.compare --input-dirs results/corespecprop_key_multisplit --baseline autopropcat --candidate corespecprop --output-csv results/corespecprop_key_multisplit_paired.csv
python -m homogcl.diagnose --input-dirs results/corespecprop_key_multisplit --baseline autopropcat --candidate corespecprop --output-csv results/corespecprop_key_multisplit_diagnostics.csv
python -m homogcl.summarize --input-dir results/corespecprop_key_multisplit --output-csv results/corespecprop_key_multisplit_summary.csv
