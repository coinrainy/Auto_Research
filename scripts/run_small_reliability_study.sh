#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Texas Wisconsin}"
SPLITS="${SPLITS:-0}"
SEEDS="${SEEDS:-0 1 2}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/methods/rw_gcl_two_stage.yaml}"
METHOD_NAME="${METHOD_NAME:-rw_gcl_two_stage}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
PAIRS_PATH="${PAIRS_PATH:-${RESULTS_DIR}/diagnostics/reliability_pair_runs.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/diagnostics/reliability_pair_summary.csv}"

mkdir -p "$(dirname "$PAIRS_PATH")"
EXPECTED_HEADER='dataset,split_index,seed,model_seed,normal_run_id,shuffled_run_id'
if [[ ! -f "$PAIRS_PATH" ]]; then
  printf '%s\n' "$EXPECTED_HEADER" > "$PAIRS_PATH"
elif [[ "$(head -n 1 "$PAIRS_PATH")" != "$EXPECTED_HEADER" ]]; then
  echo "Existing PAIRS_PATH has a legacy header. Set PAIRS_PATH to a new file or migrate it first: $PAIRS_PATH" >&2
  exit 1
fi

run_train() {
  local dataset="$1"
  local seed="$2"
  local split_index="$3"
  local label="$4"
  local shuffled="$5"
  local log_file
  log_file="$(mktemp)"
  local extra_args=()
  if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
    read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
  fi
  local cmd=(
    python train.py
    --config "$METHOD_CONFIG"
    --dataset "$dataset"
    --seed "$seed"
    --split-index "$split_index"
    --mode execute
    --device "$DEVICE"
    --warmup-epochs "$WARMUP_EPOCHS"
    --stage2-epochs "$STAGE2_EPOCHS"
    --eval-epochs "$EVAL_EPOCHS"
    --results-dir "$RESULTS_DIR"
  )
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  echo "[run] dataset=${dataset} split_index=${split_index} model_seed=${seed} method=${METHOD_NAME} config=${METHOD_CONFIG} reliability=${label}" >&2
  if [[ "$shuffled" == "true" ]]; then
    cmd+=(--shuffled-reliability)
  fi
  "${cmd[@]}" | tee "$log_file" >&2
  local run_id
  run_id="$(awk -F= '/^run_id=/ {print $2}' "$log_file" | tail -n 1)"
  rm -f "$log_file"
  if [[ -z "$run_id" ]]; then
    echo "Could not parse run_id for dataset=${dataset} split_index=${split_index} model_seed=${seed} reliability=${label}" >&2
    exit 1
  fi
  printf '%s\n' "$run_id"
}

for dataset in $DATASETS; do
  for split_index in $SPLITS; do
    for seed in $SEEDS; do
      normal_run_id="$(run_train "$dataset" "$seed" "$split_index" "normal" "false")"
      shuffled_run_id="$(run_train "$dataset" "$seed" "$split_index" "shuffled" "true")"
      printf '%s,%s,%s,%s,%s,%s\n' "$dataset" "$split_index" "$seed" "$seed" "$normal_run_id" "$shuffled_run_id" >> "$PAIRS_PATH"

      pair_out_dir="${RESULTS_DIR}/diagnostics/${normal_run_id}_vs_${shuffled_run_id}"
      python diagnose.py \
        --run_id "$normal_run_id" \
        --compare-run-id "$shuffled_run_id" \
        --results-dir "$RESULTS_DIR" \
        --out-dir "$pair_out_dir" \
        --diagnostics reliability_summary shuffled_reliability false_negative_mass view_consistency downstream_error
    done
  done
done

python summarize_reliability_pairs.py \
  --pairs "$PAIRS_PATH" \
  --results-dir "$RESULTS_DIR" \
  --out "$SUMMARY_PATH"
python eval.py \
  --metrics "${RESULTS_DIR}/metrics/main_results.csv" \
  --out "${RESULTS_DIR}/metrics/summary.csv"
