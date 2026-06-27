#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Texas Wisconsin}"
SEEDS="${SEEDS:-0 1 2}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/methods/grace.yaml}"
METHOD_NAME="${METHOD_NAME:-grace}"
EPOCHS="${EPOCHS:-70}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUNS_PATH="${RUNS_PATH:-${RESULTS_DIR}/diagnostics/${METHOD_NAME}_runs.csv}"

mkdir -p "$(dirname "$RUNS_PATH")"
if [[ ! -f "$RUNS_PATH" ]]; then
  printf 'dataset,seed,%s_run_id\n' "$METHOD_NAME" > "$RUNS_PATH"
fi

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    log_file="$(mktemp)"
    echo "[run] dataset=${dataset} seed=${seed} method=${METHOD_NAME}" >&2
    python train.py \
      --config "$METHOD_CONFIG" \
      --dataset "$dataset" \
      --seed "$seed" \
      --mode execute \
      --device "$DEVICE" \
      --epochs "$EPOCHS" \
      --eval-epochs "$EVAL_EPOCHS" \
      --results-dir "$RESULTS_DIR" | tee "$log_file" >&2

    run_id="$(awk -F= '/^run_id=/ {print $2}' "$log_file" | tail -n 1)"
    rm -f "$log_file"
    if [[ -z "$run_id" ]]; then
      echo "Could not parse run_id for dataset=${dataset} seed=${seed} method=${METHOD_NAME}" >&2
      exit 1
    fi
    printf '%s,%s,%s\n' "$dataset" "$seed" "$run_id" >> "$RUNS_PATH"
  done
done

python eval.py \
  --metrics "${RESULTS_DIR}/metrics/main_results.csv" \
  --out "${RESULTS_DIR}/metrics/summary.csv"
