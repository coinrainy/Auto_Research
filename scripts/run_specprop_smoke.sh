#!/usr/bin/env bash
set -euo pipefail

DATASETS=("Cora" "CiteSeer" "PubMed")

for dataset in "${DATASETS[@]}"; do
  python -m homogcl.train \
    --dataset "${dataset}" \
    --method specprop \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --probe sklogreg \
    --output-dir results/specprop_fullgrid
done

python -m homogcl.summarize --input-dir results/specprop_fullgrid --output-csv results/specprop_fullgrid_summary.csv
python -m homogcl.select --input-dir results/specprop_fullgrid --output-csv results/specprop_fullgrid_selected.csv
