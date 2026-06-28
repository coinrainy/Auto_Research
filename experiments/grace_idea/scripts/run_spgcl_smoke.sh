#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SPGCL_DIR="${ROOT_DIR}/third_party_baselines/SPGCL"

cd "${ROOT_DIR}/experiments/grace_idea"
python export_spgcl_geom_data.py \
  --datasets Chameleon Squirrel \
  --pyg-root ../../data/WikipediaNetwork \
  --out-root ../../third_party_baselines/SPGCL

cd "${SPGCL_DIR}"
mkdir -p results/logs saved_models
PYTHONPATH="${SPGCL_DIR}" python src/main.py \
  --dataset chameleon \
  --neg_selection random \
  --load_params 1 \
  --log_type '' \
  --save_folder logs \
  --reset_epochs 1 \
  --linear_epochs 10 \
  --reset_hidden 64 \
  --reset_seed_num 4 \
  --reset_max_size 64 \
  --reset_subg_num_hops 2
