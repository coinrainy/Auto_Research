#!/usr/bin/env bash
set -euo pipefail

SPLIT_INDICES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for split_index in "${SPLIT_INDICES[@]}"; do
  python -m homogcl.train \
    --dataset WikiCS \
    --split public \
    --split-index "${split_index}" \
    --method autopropcat \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --probe sklogreg \
    --logreg-c-grid 0.25,1,4,16 \
    --output-dir results/corespecprop_wikics_rank_ablation/autoprop

  for low_rank in 16 32; do
    python -m homogcl.train \
      --dataset WikiCS \
      --split public \
      --split-index "${split_index}" \
      --method specprop \
      --max-prop-steps 10 \
      --autoprop-plateau-ratio 0.75 \
      --specprop-high-concentration 0.34 \
      --specprop-mid-concentration 0.34 \
      --specprop-low-rank "${low_rank}" \
      --probe sklogreg \
      --logreg-c-grid 0.25,1,4,16 \
      --output-dir "results/corespecprop_wikics_rank_ablation/specprop_rank${low_rank}"
  done

  python -m homogcl.train \
    --dataset WikiCS \
    --split public \
    --split-index "${split_index}" \
    --method corespecprop \
    --max-prop-steps 10 \
    --autoprop-plateau-ratio 0.75 \
    --specprop-high-concentration 0.34 \
    --probe sklogreg \
    --logreg-c-grid 0.25,1,4,16 \
    --output-dir results/corespecprop_wikics_rank_ablation/corespecprop
done

python -m homogcl.summarize --input-dir results/corespecprop_wikics_rank_ablation/autoprop --output-csv results/corespecprop_wikics_rank_ablation_autoprop_summary.csv
python -m homogcl.summarize --input-dir results/corespecprop_wikics_rank_ablation/specprop_rank16 --output-csv results/corespecprop_wikics_rank_ablation_specprop_rank16_summary.csv
python -m homogcl.summarize --input-dir results/corespecprop_wikics_rank_ablation/specprop_rank32 --output-csv results/corespecprop_wikics_rank_ablation_specprop_rank32_summary.csv
python -m homogcl.summarize --input-dir results/corespecprop_wikics_rank_ablation/corespecprop --output-csv results/corespecprop_wikics_rank_ablation_corespecprop_summary.csv
python -m homogcl.compare --input-dirs results/corespecprop_wikics_rank_ablation/autoprop,results/corespecprop_wikics_rank_ablation/specprop_rank16 --baseline autopropcat --candidate specprop --output-csv results/corespecprop_wikics_rank16_vs_autoprop_paired.csv
python -m homogcl.compare --input-dirs results/corespecprop_wikics_rank_ablation/autoprop,results/corespecprop_wikics_rank_ablation/specprop_rank32 --baseline autopropcat --candidate specprop --output-csv results/corespecprop_wikics_rank32_vs_autoprop_paired.csv
python -m homogcl.compare --input-dirs results/corespecprop_wikics_rank_ablation/autoprop,results/corespecprop_wikics_rank_ablation/corespecprop --baseline autopropcat --candidate corespecprop --output-csv results/corespecprop_wikics_rankcore_vs_autoprop_paired.csv
