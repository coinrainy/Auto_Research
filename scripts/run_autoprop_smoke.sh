#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Cora" "CiteSeer" "PubMed")

for dataset in "${DATASETS[@]}"; do
  python -m homogcl.train \
    --dataset "${dataset}" \
    --method autopropcat \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --probe sklogreg \
    --logreg-c-grid 0.25,1,4,16 \
    --output-dir results/autoprop
done

python -m homogcl.summarize --input-dir results/autoprop --output-csv results/autoprop_summary.csv
