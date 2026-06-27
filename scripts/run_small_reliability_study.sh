#!/usr/bin/env bash
set -euo pipefail

DATASETS="${DATASETS:-Texas Wisconsin}"
SEEDS="${SEEDS:-0 1 2}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
EVAL_EPOCHS="${EVAL_EPOCHS:-50}"
DEVICE="${DEVICE:-auto}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PAIRS_PATH="${PAIRS_PATH:-${RESULTS_DIR}/diagnostics/reliability_pair_runs.csv}"
SUMMARY_PATH="${SUMMARY_PATH:-${RESULTS_DIR}/diagnostics/reliability_pair_summary.csv}"

mkdir -p "$(dirname "$PAIRS_PATH")"
if [[ ! -f "$PAIRS_PATH" ]]; then
  printf 'dataset,seed,normal_run_id,shuffled_run_id\n' > "$PAIRS_PATH"
fi

run_train() {
  local dataset="$1"
  local seed="$2"
  local label="$3"
  local shuffled="$4"
  local log_file
  log_file="$(mktemp)"
  echo "[run] dataset=${dataset} seed=${seed} reliability=${label}" >&2
  if [[ "$shuffled" == "true" ]]; then
    python train.py \
      --config configs/methods/rw_gcl_two_stage.yaml \
      --dataset "$dataset" \
      --seed "$seed" \
      --mode execute \
      --device "$DEVICE" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --stage2-epochs "$STAGE2_EPOCHS" \
      --eval-epochs "$EVAL_EPOCHS" \
      --results-dir "$RESULTS_DIR" \
      --shuffled-reliability | tee "$log_file" >&2
  else
    python train.py \
      --config configs/methods/rw_gcl_two_stage.yaml \
      --dataset "$dataset" \
      --seed "$seed" \
      --mode execute \
      --device "$DEVICE" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --stage2-epochs "$STAGE2_EPOCHS" \
      --eval-epochs "$EVAL_EPOCHS" \
      --results-dir "$RESULTS_DIR" | tee "$log_file" >&2
  fi
  local run_id
  run_id="$(awk -F= '/^run_id=/ {print $2}' "$log_file" | tail -n 1)"
  rm -f "$log_file"
  if [[ -z "$run_id" ]]; then
    echo "Could not parse run_id for dataset=${dataset} seed=${seed} reliability=${label}" >&2
    exit 1
  fi
  printf '%s\n' "$run_id"
}

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    normal_run_id="$(run_train "$dataset" "$seed" "normal" "false")"
    shuffled_run_id="$(run_train "$dataset" "$seed" "shuffled" "true")"
    printf '%s,%s,%s,%s\n' "$dataset" "$seed" "$normal_run_id" "$shuffled_run_id" >> "$PAIRS_PATH"

    pair_out_dir="${RESULTS_DIR}/diagnostics/${normal_run_id}_vs_${shuffled_run_id}"
    python diagnose.py \
      --run_id "$normal_run_id" \
      --compare-run-id "$shuffled_run_id" \
      --results-dir "$RESULTS_DIR" \
      --out-dir "$pair_out_dir" \
      --diagnostics reliability_summary shuffled_reliability false_negative_mass view_consistency
  done
done

python summarize_reliability_pairs.py \
  --pairs "$PAIRS_PATH" \
  --results-dir "$RESULTS_DIR" \
  --out "$SUMMARY_PATH"
python eval.py \
  --metrics "${RESULTS_DIR}/metrics/main_results.csv" \
  --out "${RESULTS_DIR}/metrics/summary.csv"
