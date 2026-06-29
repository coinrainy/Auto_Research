#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Cora" "CiteSeer" "PubMed" "Photo" "Computers")
SPLIT_SEEDS=(0 1 2)

for dataset in "${DATASETS[@]}"; do
  for split_seed in "${SPLIT_SEEDS[@]}"; do
    python -m homogcl.train \
      --dataset "${dataset}" \
      --split class-random \
      --split-seed "${split_seed}" \
      --method tierspecprop \
      --max-prop-steps 10 \
      --autoprop-plateau-ratio 0.75 \
      --specprop-high-concentration 0.34 \
      --tierspecprop-wide-concentration 0.36 \
      --tierspecprop-narrow-rank 16 \
      --tierspecprop-wide-rank 32 \
      --probe sklogreg \
      --logreg-c-grid 0.25,1,4,16 \
      --output-dir results/tierspecprop_multisplit
  done
done

python -m homogcl.compare --input-dirs results/corespecprop_multisplit,results/tierspecprop_multisplit --baseline autopropcat --candidate tierspecprop --output-csv results/tierspecprop_multisplit_paired.csv
python -m homogcl.diagnose --input-dirs results/corespecprop_multisplit,results/tierspecprop_multisplit --baseline autopropcat --candidate tierspecprop --output-csv results/tierspecprop_multisplit_diagnostics.csv
