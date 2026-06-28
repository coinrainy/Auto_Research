#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py --dataset Cora --method grace --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_cora_grace
python train.py --dataset Cora --method energy_spgcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_cora_energy_spgcl
python train.py --dataset Cora --method gcn_mlp_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_cora_gcn_mlp
python train.py --dataset Cora --method er_residual_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_cora_er_residual
python train.py --dataset Cora --method er_cache_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_cora_er
python train.py --dataset Texas --method energy_spgcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_texas_energy_spgcl
python train.py --dataset Texas --method er_residual_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_texas_er_residual
python train.py --dataset Texas --method er_cache_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --run-name smoke_texas_er
python train.py --dataset Texas --method er_cache_gcl --epochs 2 --seed 0 --split-index 0 --skip-eval --shuffle-cache --run-name smoke_texas_er_shuffled
