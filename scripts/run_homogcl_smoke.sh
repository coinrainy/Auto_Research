#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Cora" "CiteSeer" "PubMed")
METHODS=("raw" "prop" "propcat" "propcca" "propccat" "ccassg" "ccacat" "grace" "gracecat" "homogcl" "horp" "horpgcl")
SEEDS=(0 1)

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python -m homogcl.train \
        --dataset "${dataset}" \
        --method "${method}" \
        --seed "${seed}" \
        --eval-seed "${seed}" \
        --probe sklogreg \
        --epochs 150 \
        --output-dir results/smoke
    done
  done
done

python -m homogcl.summarize --input-dir results/smoke --output-csv results/smoke_summary.csv
