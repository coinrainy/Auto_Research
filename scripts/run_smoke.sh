#!/usr/bin/env bash
set -euo pipefail

python train.py --config configs/methods/grace.yaml --dataset Cora --describe-data
python train.py --config configs/methods/grace.yaml --dataset Texas --describe-data
python train.py --config configs/methods/grace.yaml --dataset Cora --seed 0 --mode execute --epochs 5 --eval-epochs 20
python train.py --config configs/methods/grace.yaml --dataset Texas --seed 0 --mode execute --epochs 5 --eval-epochs 20
rw_output=$(python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0 --mode execute --warmup-epochs 3 --stage2-epochs 3 --eval-epochs 20)
echo "$rw_output"
rw_run_id=$(printf '%s\n' "$rw_output" | awk -F= '/^run_id=/ {print $2}')
shuffled_output=$(python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0 --mode execute --warmup-epochs 3 --stage2-epochs 3 --eval-epochs 20 --shuffled-reliability)
echo "$shuffled_output"
shuffled_run_id=$(printf '%s\n' "$shuffled_output" | awk -F= '/^run_id=/ {print $2}')
python eval.py
python diagnose.py --run_id "$rw_run_id" --compare-run-id "$shuffled_run_id" --diagnostics reliability_summary shuffled_reliability false_negative_mass view_consistency
