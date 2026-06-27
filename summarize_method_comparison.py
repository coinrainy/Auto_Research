#!/usr/bin/env python3
"""Summarize RW-GCL normal/shuffled runs against a baseline method."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.logging_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align RW-GCL normal/shuffled runs with baseline runs and aggregate by dataset."
    )
    parser.add_argument(
        "--rw-summary",
        default="results/diagnostics/reliability_pair_summary.csv",
        help="CSV produced by summarize_reliability_pairs.py.",
    )
    parser.add_argument(
        "--baseline-runs",
        default="results/diagnostics/grace_runs.csv",
        help="CSV with dataset, seed, and baseline run id columns.",
    )
    parser.add_argument("--baseline-method", default="grace", help="Baseline method label.")
    parser.add_argument(
        "--baseline-run-column",
        default=None,
        help="Run id column in --baseline-runs. Defaults to <baseline-method>_run_id.",
    )
    parser.add_argument("--results-dir", default="results", help="Experiment results directory.")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics CSV path. Defaults to <results-dir>/metrics/main_results.csv.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Per-seed output CSV. Defaults to results/diagnostics/rw_gcl_vs_<baseline>.csv.",
    )
    parser.add_argument(
        "--aggregate-out",
        default=None,
        help="Aggregate output CSV. Defaults to results/diagnostics/rw_gcl_vs_<baseline>_aggregate.csv.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def read_rows(path: str | Path) -> list[dict]:
    csv_path = resolve_path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_key(method: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", method.strip().lower()).strip("_")
    return key or "baseline"


def as_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def metric_index(path: str | Path) -> dict[str, dict]:
    return {row.get("run_id", ""): row for row in read_rows(path)}


def metric_value(metrics: dict[str, dict], run_id: str) -> float | None:
    return as_float(metrics.get(run_id, {}).get("value"))


def value_from_summary_or_metrics(summary_row: dict, key: str, metrics: dict[str, dict], run_id_key: str) -> float | None:
    value = as_float(summary_row.get(key))
    if value is not None:
        return value
    return metric_value(metrics, summary_row.get(run_id_key, ""))


def baseline_index(rows: list[dict], run_column: str) -> dict[tuple[str, str], str]:
    indexed: dict[tuple[str, str], str] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        seed = row.get("seed", "")
        run_id = row.get(run_column, "") or row.get("baseline_run_id", "") or row.get("run_id", "")
        if dataset and seed != "" and run_id:
            indexed[(dataset, seed)] = run_id
    return indexed


def compare_rows(
    rw_rows: list[dict],
    baseline_runs: dict[tuple[str, str], str],
    metrics: dict[str, dict],
    baseline: str,
) -> list[dict]:
    rows = []
    baseline_run_col = f"{baseline}_run_id"
    baseline_accuracy_col = f"{baseline}_accuracy"
    normal_minus_baseline_col = f"rw_normal_minus_{baseline}"
    shuffled_minus_baseline_col = f"rw_shuffled_minus_{baseline}"

    for rw_row in rw_rows:
        dataset = rw_row.get("dataset", "")
        seed = rw_row.get("seed", "")
        normal_run_id = rw_row.get("normal_run_id", "")
        shuffled_run_id = rw_row.get("shuffled_run_id", "")
        baseline_run_id = baseline_runs.get((dataset, seed), "")

        normal_accuracy = value_from_summary_or_metrics(rw_row, "normal_accuracy", metrics, "normal_run_id")
        shuffled_accuracy = value_from_summary_or_metrics(rw_row, "shuffled_accuracy", metrics, "shuffled_run_id")
        baseline_accuracy = metric_value(metrics, baseline_run_id)
        normal_minus_shuffled = as_float(rw_row.get("accuracy_delta"))
        if normal_minus_shuffled is None and normal_accuracy is not None and shuffled_accuracy is not None:
            normal_minus_shuffled = normal_accuracy - shuffled_accuracy

        normal_minus_baseline = None
        shuffled_minus_baseline = None
        if normal_accuracy is not None and baseline_accuracy is not None:
            normal_minus_baseline = normal_accuracy - baseline_accuracy
        if shuffled_accuracy is not None and baseline_accuracy is not None:
            shuffled_minus_baseline = shuffled_accuracy - baseline_accuracy

        status = "computed"
        missing = []
        if not baseline_run_id:
            missing.append("baseline_run_id")
        if normal_accuracy is None:
            missing.append("normal_accuracy")
        if shuffled_accuracy is None:
            missing.append("shuffled_accuracy")
        if baseline_accuracy is None:
            missing.append(f"{baseline}_accuracy")
        if missing:
            status = "missing_" + "|".join(missing)

        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "rw_normal_run_id": normal_run_id,
                "rw_shuffled_run_id": shuffled_run_id,
                baseline_run_col: baseline_run_id,
                "rw_normal_accuracy": format_float(normal_accuracy),
                "rw_shuffled_accuracy": format_float(shuffled_accuracy),
                baseline_accuracy_col: format_float(baseline_accuracy),
                normal_minus_baseline_col: format_float(normal_minus_baseline),
                shuffled_minus_baseline_col: format_float(shuffled_minus_baseline),
                "rw_normal_minus_shuffled": format_float(normal_minus_shuffled),
                "view_consistency_gap": rw_row.get("normal_view_consistency_gap", ""),
                "status": status,
            }
        )
    return rows


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pstdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def count_signs(values: list[float]) -> tuple[int, int, int]:
    positive = sum(1 for value in values if value > 0.0)
    zero = sum(1 for value in values if value == 0.0)
    negative = sum(1 for value in values if value < 0.0)
    return positive, zero, negative


def floats_from_rows(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def aggregate_rows(rows: list[dict], baseline: str) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "computed":
            grouped.setdefault(row.get("dataset", ""), []).append(row)

    normal_minus_baseline_col = f"rw_normal_minus_{baseline}"
    baseline_accuracy_col = f"{baseline}_accuracy"
    aggregate = []
    for dataset in sorted(grouped):
        group = grouped[dataset]
        normal_acc = floats_from_rows(group, "rw_normal_accuracy")
        shuffled_acc = floats_from_rows(group, "rw_shuffled_accuracy")
        baseline_acc = floats_from_rows(group, baseline_accuracy_col)
        normal_minus_baseline = floats_from_rows(group, normal_minus_baseline_col)
        normal_minus_shuffled = floats_from_rows(group, "rw_normal_minus_shuffled")
        view_gaps = floats_from_rows(group, "view_consistency_gap")
        pos, zero, neg = count_signs(normal_minus_baseline)
        aggregate.append(
            {
                "dataset": dataset,
                "runs": str(len(group)),
                "rw_normal_accuracy_mean": format_float(mean(normal_acc)),
                "rw_shuffled_accuracy_mean": format_float(mean(shuffled_acc)),
                baseline_accuracy_col + "_mean": format_float(mean(baseline_acc)),
                normal_minus_baseline_col + "_mean": format_float(mean(normal_minus_baseline)),
                normal_minus_baseline_col + "_std": format_float(pstdev(normal_minus_baseline)),
                normal_minus_baseline_col + "_positive_count": str(pos),
                normal_minus_baseline_col + "_zero_count": str(zero),
                normal_minus_baseline_col + "_negative_count": str(neg),
                "rw_normal_minus_shuffled_mean": format_float(mean(normal_minus_shuffled)),
                "rw_normal_minus_shuffled_std": format_float(pstdev(normal_minus_shuffled)),
                "view_consistency_gap_mean": format_float(mean(view_gaps)),
                "view_consistency_gap_min": format_float(min(view_gaps) if view_gaps else None),
                "view_consistency_gap_max": format_float(max(view_gaps) if view_gaps else None),
            }
        )
    return aggregate


def main() -> int:
    args = parse_args()
    baseline = method_key(args.baseline_method)
    metrics_path = args.metrics or str(Path(args.results_dir) / "metrics" / "main_results.csv")
    out = args.out or f"results/diagnostics/rw_gcl_vs_{baseline}.csv"
    aggregate_out = args.aggregate_out or f"results/diagnostics/rw_gcl_vs_{baseline}_aggregate.csv"
    baseline_run_column = args.baseline_run_column or f"{baseline}_run_id"

    rw_rows = read_rows(args.rw_summary)
    baseline_rows = read_rows(args.baseline_runs)
    if not rw_rows:
        print(f"No RW-GCL summary rows found at {args.rw_summary}")
        return 0
    if not baseline_rows:
        print(f"No baseline rows found at {args.baseline_runs}")
        return 0

    metrics = metric_index(metrics_path)
    rows = compare_rows(
        rw_rows=rw_rows,
        baseline_runs=baseline_index(baseline_rows, baseline_run_column),
        metrics=metrics,
        baseline=baseline,
    )
    aggregate = aggregate_rows(rows, baseline)

    row_fieldnames = [
        "dataset",
        "seed",
        "rw_normal_run_id",
        "rw_shuffled_run_id",
        f"{baseline}_run_id",
        "rw_normal_accuracy",
        "rw_shuffled_accuracy",
        f"{baseline}_accuracy",
        f"rw_normal_minus_{baseline}",
        f"rw_shuffled_minus_{baseline}",
        "rw_normal_minus_shuffled",
        "view_consistency_gap",
        "status",
    ]
    aggregate_fieldnames = [
        "dataset",
        "runs",
        "rw_normal_accuracy_mean",
        "rw_shuffled_accuracy_mean",
        f"{baseline}_accuracy_mean",
        f"rw_normal_minus_{baseline}_mean",
        f"rw_normal_minus_{baseline}_std",
        f"rw_normal_minus_{baseline}_positive_count",
        f"rw_normal_minus_{baseline}_zero_count",
        f"rw_normal_minus_{baseline}_negative_count",
        "rw_normal_minus_shuffled_mean",
        "rw_normal_minus_shuffled_std",
        "view_consistency_gap_mean",
        "view_consistency_gap_min",
        "view_consistency_gap_max",
    ]
    write_csv(out, rows, row_fieldnames)
    write_csv(aggregate_out, aggregate, aggregate_fieldnames)
    print(f"wrote {out}")
    print(f"wrote {aggregate_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
