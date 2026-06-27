#!/usr/bin/env bash
set -euo pipefail

for dataset in Cora CiteSeer PubMed Texas Wisconsin Cornell Actor Chameleon Squirrel; do
  python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset "${dataset}" --seed 0
done
