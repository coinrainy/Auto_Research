#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Cora" "CiteSeer" "PubMed")
SPLIT_SEEDS=(0 1 2)
METHODS=("autopropcat" "specprop")

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
        --probe sklogreg \
        --logreg-c-grid 0.25,1,4,16 \
        --output-dir results/specprop_strict_multisplit
    done
  done
done

python -m homogcl.summarize --input-dir results/specprop_strict_multisplit --output-csv results/specprop_strict_multisplit_summary.csv
python -m homogcl.select --input-dir results/specprop_strict_multisplit --output-csv results/specprop_strict_multisplit_selected.csv --mode run
python -m homogcl.compare --input-dirs results/specprop_strict_multisplit --baseline autopropcat --candidate specprop --output-csv results/specprop_strict_multisplit_paired.csv
