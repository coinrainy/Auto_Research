#!/usr/bin/env bash
set -euo pipefail

SPLIT_INDICES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for split_index in "${SPLIT_INDICES[@]}"; do
  python -m homogcl.train \
    --dataset WikiCS \
    --split public \
    --split-index "${split_index}" \
    --method tierspecprop \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --specprop-high-concentration 0.34 \
    --tierspecprop-wide-concentration 0.36 \
    --tierspecprop-narrow-rank 16 \
    --tierspecprop-wide-rank 32 \
    --probe sklogreg \
    --logreg-c-grid 0.25,1,4,16 \
    --output-dir results/tierspecprop_wikics_multisplit
done

python -m homogcl.compare --input-dirs results/corespecprop_wikics_rank_ablation/autoprop,results/tierspecprop_wikics_multisplit --baseline autopropcat --candidate tierspecprop --output-csv results/tierspecprop_wikics_multisplit_paired.csv
python -m homogcl.diagnose --input-dirs results/corespecprop_wikics_rank_ablation/autoprop,results/tierspecprop_wikics_multisplit --baseline autopropcat --candidate tierspecprop --output-csv results/tierspecprop_wikics_multisplit_diagnostics.csv
