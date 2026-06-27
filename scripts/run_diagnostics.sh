#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>" >&2
  exit 2
fi

python diagnose.py --run_id "$1" --diagnostics shuffled_reliability false_negative_mass view_consistency
