#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [compare_run_id]" >&2
  exit 2
fi

if [[ $# -ge 2 ]]; then
  python diagnose.py --run_id "$1" --compare-run-id "$2" --diagnostics shuffled_reliability false_negative_mass view_consistency
else
  python diagnose.py --run_id "$1" --diagnostics shuffled_reliability false_negative_mass view_consistency
fi
