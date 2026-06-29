#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Photo" "Computers")
METHODS=("autopropcat" "specprop")

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    python -m homogcl.train \
      --dataset "${dataset}" \
      --split class-random \
      --split-seed 0 \
      --method "${method}" \
      --max-prop-steps 10 \
      --autoprop-plateau-ratio 0.75 \
      --probe sklogreg \
      --logreg-c-grid 0.25,1,4,16 \
      --output-dir results/specprop_amazon_smoke
  done
done

python -m homogcl.compare --input-dirs results/specprop_amazon_smoke --baseline autopropcat --candidate specprop --output-csv results/specprop_amazon_smoke_paired.csv
