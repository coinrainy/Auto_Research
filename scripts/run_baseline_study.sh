#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Texas Wisconsin}"
SPLITS="${SPLITS:-0}"
SEEDS="${SEEDS:-0 1 2}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/methods/grace.yaml}"
METHOD_NAME="${METHOD_NAME:-grace}"
EPOCHS="${EPOCHS:-70}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-results}"
RUNS_PATH="${RUNS_PATH:-${RESULTS_DIR}/diagnostics/${METHOD_NAME}_runs.csv}"

mkdir -p "$(dirname "$RUNS_PATH")"
EXPECTED_HEADER="dataset,split_index,seed,model_seed,${METHOD_NAME}_run_id"
if [[ ! -f "$RUNS_PATH" ]]; then
  printf '%s\n' "$EXPECTED_HEADER" > "$RUNS_PATH"
elif [[ "$(head -n 1 "$RUNS_PATH")" != "$EXPECTED_HEADER" ]]; then
  echo "Existing RUNS_PATH has a legacy header. Set RUNS_PATH to a new file or migrate it first: $RUNS_PATH" >&2
  exit 1
fi

for dataset in $DATASETS; do
  for split_index in $SPLITS; do
    for seed in $SEEDS; do
      log_file="$(mktemp)"
      echo "[run] dataset=${dataset} split_index=${split_index} model_seed=${seed} method=${METHOD_NAME}" >&2
      python train.py \
        --config "$METHOD_CONFIG" \
        --dataset "$dataset" \
        --seed "$seed" \
        --split-index "$split_index" \
        --mode execute \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --eval-epochs "$EVAL_EPOCHS" \
        --results-dir "$RESULTS_DIR" | tee "$log_file" >&2

      run_id="$(awk -F= '/^run_id=/ {print $2}' "$log_file" | tail -n 1)"
      rm -f "$log_file"
      if [[ -z "$run_id" ]]; then
        echo "Could not parse run_id for dataset=${dataset} split_index=${split_index} model_seed=${seed} method=${METHOD_NAME}" >&2
        exit 1
      fi
      printf '%s,%s,%s,%s,%s\n' "$dataset" "$split_index" "$seed" "$seed" "$run_id" >> "$RUNS_PATH"
    done
  done
done

python eval.py \
  --metrics "${RESULTS_DIR}/metrics/main_results.csv" \
  --out "${RESULTS_DIR}/metrics/summary.csv"
