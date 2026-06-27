#!/usr/bin/env python3
"""Evaluation result summarizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.evaluation import group_metric_rows, read_metric_rows
from rwgcl.logging_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RW-GCL metrics.")
    parser.add_argument("--metrics", default="results/metrics/main_results.csv", help="Metrics CSV path.")
    parser.add_argument("--out", default="results/metrics/summary.csv", help="Summary CSV path.")
    parser.add_argument("--group-by", nargs="+", default=["method", "dataset"], help="Columns to group by.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_metric_rows(args.metrics)
    if not rows:
        print(f"No metric rows found at {args.metrics}")
        return 0
    summary = group_metric_rows(rows, args.group_by)
    fieldnames = list(args.group_by) + ["runs", "statuses"]
    write_csv(args.out, summary, fieldnames)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
