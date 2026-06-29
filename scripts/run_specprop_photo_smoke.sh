#!/usr/bin/env bash
set -euo pipefail

for method in autopropcat specprop; do
  python -m homogcl.train \
    --dataset Photo \
    --split class-random \
    --split-seed 0 \
    --method "${method}" \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --probe sklogreg \
    --logreg-c-grid 0.25,1,4,16 \
    --output-dir results/specprop_photo_smoke
done

python -m homogcl.compare --input-dirs results/specprop_photo_smoke --baseline autopropcat --candidate specprop --output-csv results/specprop_photo_smoke_paired.csv
