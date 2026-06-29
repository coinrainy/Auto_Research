#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUNS_DIR="${RUNS_DIR:-runs/smoke}"
OVERWRITE="${OVERWRITE:-1}"
GPU_ID="${GPU_ID:-0}"

python train.py --dataset Cora --method raw_features --epochs 2 --split-index 0 --seed 0 --gpu-id "$GPU_ID" --runs-dir "$RUNS_DIR" --run-name Cora_raw_smoke ${OVERWRITE:+--overwrite}
python train.py --dataset Cora --method grace --epochs 2 --split-index 0 --seed 0 --gpu-id "$GPU_ID" --runs-dir "$RUNS_DIR" --run-name Cora_grace_smoke ${OVERWRITE:+--overwrite}
python train.py --dataset Cora --method hpfs_gcl --epochs 2 --split-index 0 --seed 0 --gpu-id "$GPU_ID" --runs-dir "$RUNS_DIR" --run-name Cora_hpfs_smoke ${OVERWRITE:+--overwrite}
python summarize.py --runs-dir "$RUNS_DIR"
