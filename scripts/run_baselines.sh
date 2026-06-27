#!/usr/bin/env bash
set -euo pipefail

for dataset in Cora CiteSeer PubMed Texas Wisconsin Cornell Actor Chameleon Squirrel; do
  python train.py --config configs/methods/grace.yaml --dataset "${dataset}" --seed 0
  python train.py --config configs/methods/bgrl.yaml --dataset "${dataset}" --seed 0
done
