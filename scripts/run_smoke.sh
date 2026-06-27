#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/methods/grace.yaml --dataset Cora --seed 0
python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0
python eval.py
