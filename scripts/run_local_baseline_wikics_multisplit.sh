#!/usr/bin/env bash
set -euo pipefail

SPLIT_INDICES_STR="${SPLIT_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
METHODS_STR="${METHODS:-autopropcat tierspecprop propccat ccacat gracecat}"
OUT_DIR="${OUT_DIR:-results/local_baseline_wikics_multisplit}"
EPOCHS="${EPOCHS:-200}"
PROP_STEPS="${PROP_STEPS:-10}"
LOGREG_C_GRID="${LOGREG_C_GRID:-0.25,1,4,16}"

read -r -a SPLIT_INDICES <<< "${SPLIT_INDICES_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

for split_index in "${SPLIT_INDICES[@]}"; do
  for method in "${METHODS[@]}"; do
    python -m homogcl.train \
      --dataset WikiCS \
      --split public \
      --split-index "${split_index}" \
      --method "${method}" \
      --epochs "${EPOCHS}" \
      --prop-steps "${PROP_STEPS}" \
      --max-prop-steps 10 \
      --autoprop-plateau-ratio 0.75 \
      --specprop-high-concentration 0.34 \
      --tierspecprop-wide-concentration 0.36 \
      --tierspecprop-narrow-rank 16 \
      --tierspecprop-wide-rank 32 \
      --probe sklogreg \
      --logreg-c-grid "${LOGREG_C_GRID}" \
      --output-dir "${OUT_DIR}"
  done
done

python -m homogcl.summarize --input-dir "${OUT_DIR}" --output-csv "${OUT_DIR}_summary.csv"

for baseline in autopropcat propccat ccacat gracecat; do
  if [[ " ${METHODS_STR} " == *" ${baseline} "* && " ${METHODS_STR} " == *" tierspecprop "* ]]; then
    python -m homogcl.compare \
      --input-dirs "${OUT_DIR}" \
      --baseline "${baseline}" \
      --candidate tierspecprop \
      --output-csv "${OUT_DIR}_tierspecprop_vs_${baseline}.csv"
  fi
done
