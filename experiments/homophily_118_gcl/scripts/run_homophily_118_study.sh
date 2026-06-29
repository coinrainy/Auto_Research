#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-Cora CiteSeer PubMed}"
METHODS="${METHODS:-raw_features grace hpfs_gcl}"
SPLITS="${SPLITS:-0}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
GPU_ID="${GPU_ID:-0}"
RUNS_DIR="${RUNS_DIR:-runs/homophily_118}"
CONFIG="${CONFIG:-configs/default.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_TAG="${RUN_TAG:-}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$RUNS_DIR"

echo "[homophily-118] datasets=$DATASETS methods=$METHODS splits=$SPLITS seeds=$SEEDS epochs=$EPOCHS"

for dataset in $DATASETS; do
  for split in $SPLITS; do
    for seed in $SEEDS; do
      for method in $METHODS; do
        run_name="${dataset}_${method}_seed${seed}_split${split}_e${EPOCHS}"
        if [[ -n "$RUN_TAG" ]]; then
          run_name="${run_name}_${RUN_TAG}"
        fi
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
        echo "[homophily-118] ${cmd[*]}"
        "${cmd[@]}"
      done
    done
  done
done

python summarize.py \
  --runs-dir "$RUNS_DIR" \
  --out "$RUNS_DIR/runs_vs_grace.csv" \
  --aggregate-out "$RUNS_DIR/aggregate_vs_grace.csv"
