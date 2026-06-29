#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-results/tierccacat_multisplit}"
KEY_DATASETS_STR="${KEY_DATASETS:-PubMed Photo}"
KEY_SPLIT_SEEDS_STR="${KEY_SPLIT_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
WIKICS_SPLIT_INDICES_STR="${WIKICS_SPLIT_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
METHODS_STR="${METHODS:-autopropcat tierspecprop tierccacat}"
EPOCHS="${EPOCHS:-200}"
LOGREG_C_GRID="${LOGREG_C_GRID:-0.25,1,4,16}"

read -r -a KEY_DATASETS <<< "${KEY_DATASETS_STR}"
read -r -a KEY_SPLIT_SEEDS <<< "${KEY_SPLIT_SEEDS_STR}"
read -r -a WIKICS_SPLIT_INDICES <<< "${WIKICS_SPLIT_INDICES_STR}"
read -r -a METHODS <<< "${METHODS_STR}"

COMMON_ARGS=(
  --epochs "${EPOCHS}"
  --prop-steps 10
  --max-prop-steps 10
  --autoprop-plateau-ratio 0.75
  --specprop-high-concentration 0.34
  --tierspecprop-wide-concentration 0.36
  --tierspecprop-narrow-rank 16
  --tierspecprop-wide-rank 32
  --probe sklogreg
  --logreg-c-grid "${LOGREG_C_GRID}"
  --output-dir "${OUT_DIR}"
)

for dataset in "${KEY_DATASETS[@]}"; do
  case "${dataset}" in
    CS|Physics|Coauthor-CS|Coauthor-Physics)
      echo "Skip ${dataset}: Coauthor CS/Physics is intentionally disabled for this stage." >&2
      continue
      ;;
  esac

  for split_seed in "${KEY_SPLIT_SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      python -m homogcl.train \
        --dataset "${dataset}" \
        --split class-random \
        --split-seed "${split_seed}" \
        --method "${method}" \
        "${COMMON_ARGS[@]}"
    done
  done
done

for split_index in "${WIKICS_SPLIT_INDICES[@]}"; do
  for method in "${METHODS[@]}"; do
    python -m homogcl.train \
      --dataset WikiCS \
      --split public \
      --split-index "${split_index}" \
      --method "${method}" \
      "${COMMON_ARGS[@]}"
  done
done

python -m homogcl.summarize --input-dir "${OUT_DIR}" --output-csv "${OUT_DIR}_summary.csv"

if [[ " ${METHODS_STR} " == *" tierccacat "* ]]; then
  for baseline in autopropcat tierspecprop; do
    if [[ " ${METHODS_STR} " == *" ${baseline} "* ]]; then
      python -m homogcl.compare \
        --input-dirs "${OUT_DIR}" \
        --baseline "${baseline}" \
        --candidate tierccacat \
        --output-csv "${OUT_DIR}_tierccacat_vs_${baseline}.csv"
      python -m homogcl.diagnose \
        --input-dirs "${OUT_DIR}" \
        --baseline "${baseline}" \
        --candidate tierccacat \
        --output-csv "${OUT_DIR}_tierccacat_vs_${baseline}_diagnostics.csv"
    fi
  done
fi
