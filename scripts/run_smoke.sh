#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/methods/grace.yaml --dataset Cora --describe-data
python train.py --config configs/methods/grace.yaml --dataset Texas --describe-data
python train.py --config configs/methods/grace.yaml --dataset Cora --seed 0 --mode execute --epochs 5 --eval-epochs 20
python train.py --config configs/methods/grace.yaml --dataset Texas --seed 0 --mode execute --epochs 5 --eval-epochs 20
python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0 --mode execute --warmup-epochs 3 --stage2-epochs 3 --eval-epochs 20
python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0 --mode execute --warmup-epochs 3 --stage2-epochs 3 --eval-epochs 20 --shuffled-reliability
python eval.py
