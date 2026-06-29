#!/usr/bin/env bash
set -euo pipefail

DATASETS_STR="${DATASETS:-PubMed Photo}"
SPLIT_SEEDS_STR="${SPLIT_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
METHODS_STR="${METHODS:-autopropcat tierspecprop tierccacat propccat ccacat gracecat}"
OUT_DIR="${OUT_DIR:-results/local_baseline_key_multisplit}"
EPOCHS="${EPOCHS:-200}"
PROP_STEPS="${PROP_STEPS:-10}"
LOGREG_C_GRID="${LOGREG_C_GRID:-0.25,1,4,16}"

read -r -a DATASETS <<< "${DATASETS_STR}"
read -r -a SPLIT_SEEDS <<< "${SPLIT_SEEDS_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

for dataset in "${DATASETS[@]}"; do
  case "${dataset}" in
    CS|Physics|Coauthor-CS|Coauthor-Physics)
      echo "Skip ${dataset}: Coauthor CS/Physics is intentionally disabled for this stage." >&2
      continue
      ;;
  esac

  for split_seed in "${SPLIT_SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      python -m homogcl.train \
        --dataset "${dataset}" \
        --split class-random \
        --split-seed "${split_seed}" \
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
done

python -m homogcl.summarize --input-dir "${OUT_DIR}" --output-csv "${OUT_DIR}_summary.csv"

for baseline in autopropcat tierspecprop propccat ccacat gracecat; do
  if [[ "${baseline}" == "tierspecprop" && " ${METHODS_STR} " != *" tierccacat "* ]]; then
    continue
  fi
  if [[ " ${METHODS_STR} " == *" ${baseline} "* && " ${METHODS_STR} " == *" tierccacat "* ]]; then
    python -m homogcl.compare \
      --input-dirs "${OUT_DIR}" \
      --baseline "${baseline}" \
      --candidate tierccacat \
      --output-csv "${OUT_DIR}_tierccacat_vs_${baseline}.csv"
  fi
done
