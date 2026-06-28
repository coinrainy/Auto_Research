#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-Texas Chameleon Squirrel Actor}"
METHODS="${METHODS:-grace gcn_mlp_gcl}"
SPLITS="${SPLITS:-0}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
GPU_ID="${GPU_ID:-0}"
RUNS_DIR="${RUNS_DIR:-runs/split_study}"
CONFIG="${CONFIG:-configs/default.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$RUNS_DIR"

echo "[split-study] datasets=$DATASETS"
echo "[split-study] methods=$METHODS"
echo "[split-study] splits=$SPLITS seeds=$SEEDS epochs=$EPOCHS runs_dir=$RUNS_DIR"

for dataset in $DATASETS; do
  for split in $SPLITS; do
    for seed in $SEEDS; do
      for method in $METHODS; do
        run_name="${dataset}_${method}_seed${seed}_split${split}_e${EPOCHS}"
        cmd=(
          python train.py
          --config "$CONFIG"
          --dataset "$dataset"
          --method "$method"
          --epochs "$EPOCHS"
          --seed "$seed"
          --split-index "$split"
          --gpu-id "$GPU_ID"
          --runs-dir "$RUNS_DIR"
          --run-name "$run_name"
        )
        if [[ "$OVERWRITE" == "1" ]]; then
          cmd+=(--overwrite)
        fi
        if [[ -n "$EXTRA_ARGS" ]]; then
          # shellcheck disable=SC2206
          extra=($EXTRA_ARGS)
          cmd+=("${extra[@]}")
        fi
        echo "[split-study] ${cmd[*]}"
        "${cmd[@]}"
      done
    done
  done
done

python summarize_split_study.py \
  --runs-dir "$RUNS_DIR" \
  --out "$RUNS_DIR/split_study_runs.csv" \
  --aggregate-out "$RUNS_DIR/split_study_aggregate.csv"

