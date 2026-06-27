#!/usr/bin/env python3
"""Summarize paired normal-vs-shuffled RW-GCL reliability runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.logging_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired RW-GCL reliability runs.")
    parser.add_argument(
        "--pairs",
        default="results/diagnostics/reliability_pair_runs.csv",
        help="CSV with dataset, seed, normal_run_id, and shuffled_run_id columns.",
    )
    parser.add_argument("--results-dir", default="results", help="Experiment results directory.")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Metrics CSV path. Defaults to <results-dir>/metrics/main_results.csv.",
    )
    parser.add_argument(
        "--out",
        default="results/diagnostics/reliability_pair_summary.csv",
        help="Output summary CSV path.",
    )
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict]:
    csv_path = PROJECT_ROOT / path
    if Path(path).is_absolute():
        csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_index(path: str | Path) -> dict[str, dict]:
    return {row.get("run_id", ""): row for row in read_rows(path)}


def run_dir(results_dir: str | Path, run_id: str) -> Path:
    return PROJECT_ROOT / results_dir / "raw" / run_id


def load_metadata(results_dir: str | Path, run_id: str) -> dict:
    path = run_dir(results_dir, run_id) / "run_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_payload(results_dir: str | Path, run_id: str) -> dict:
    path = run_dir(results_dir, run_id) / "embeddings.pt"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=True)


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


def summary_value(metadata: dict, key: str) -> str:
    value = metadata.get("last_reliability_summary", {}).get(key, "")
    if value == "":
        return ""
    return f"{float(value):.6f}"


def mean_at(values: torch.Tensor, idx: torch.Tensor) -> float | None:
    if idx.numel() == 0:
        return None
    return float(values[idx].float().mean().item())


def view_consistency_gaps(payload: dict) -> dict[str, str]:
    required = {"positive_reliability", "embedding_stability", "prediction_consistency"}
    if not required.issubset(payload.keys()):
        return {
            "view_reliability_gap": "",
            "view_stability_gap": "",
            "view_consistency_gap": "",
        }
    reliability = payload["positive_reliability"].float()
    stability = payload["embedding_stability"].float()
    consistency = payload["prediction_consistency"].float()
    chunks = torch.chunk(torch.argsort(reliability), 3)
    if len(chunks) < 3:
        return {
            "view_reliability_gap": "",
            "view_stability_gap": "",
            "view_consistency_gap": "",
        }
    low_idx, high_idx = chunks[0], chunks[-1]
    low_reliability = mean_at(reliability, low_idx)
    high_reliability = mean_at(reliability, high_idx)
    low_stability = mean_at(stability, low_idx)
    high_stability = mean_at(stability, high_idx)
    low_consistency = mean_at(consistency, low_idx)
    high_consistency = mean_at(consistency, high_idx)
    return {
        "view_reliability_gap": format_float(
            None if low_reliability is None or high_reliability is None else high_reliability - low_reliability
        ),
        "view_stability_gap": format_float(
            None if low_stability is None or high_stability is None else high_stability - low_stability
        ),
        "view_consistency_gap": format_float(
            None if low_consistency is None or high_consistency is None else high_consistency - low_consistency
        ),
    }


def summarize_pair(pair: dict, metrics: dict[str, dict], results_dir: str | Path) -> dict:
    normal_run_id = pair.get("normal_run_id", "")
    shuffled_run_id = pair.get("shuffled_run_id", "")
    normal_metric = metrics.get(normal_run_id, {})
    shuffled_metric = metrics.get(shuffled_run_id, {})
    normal_accuracy = as_float(normal_metric.get("value"))
    shuffled_accuracy = as_float(shuffled_metric.get("value"))
    normal_meta = load_metadata(results_dir, normal_run_id)
    shuffled_meta = load_metadata(results_dir, shuffled_run_id)
    normal_payload = load_payload(results_dir, normal_run_id)
    gaps = view_consistency_gaps(normal_payload)
    delta = None
    if normal_accuracy is not None and shuffled_accuracy is not None:
        delta = normal_accuracy - shuffled_accuracy
    return {
        "dataset": pair.get("dataset", ""),
        "seed": pair.get("seed", ""),
        "normal_run_id": normal_run_id,
        "shuffled_run_id": shuffled_run_id,
        "normal_accuracy": format_float(normal_accuracy),
        "shuffled_accuracy": format_float(shuffled_accuracy),
        "accuracy_delta": format_float(delta),
        "normal_reliability_mean": summary_value(normal_meta, "reliability_mean"),
        "normal_reliability_std": summary_value(normal_meta, "reliability_std"),
        "shuffled_reliability_mean": summary_value(shuffled_meta, "reliability_mean"),
        "shuffled_reliability_std": summary_value(shuffled_meta, "reliability_std"),
        "normal_embedding_stability_mean": summary_value(normal_meta, "embedding_stability_mean"),
        "normal_prediction_consistency_mean": summary_value(normal_meta, "prediction_consistency_mean"),
        "normal_view_reliability_gap": gaps["view_reliability_gap"],
        "normal_view_stability_gap": gaps["view_stability_gap"],
        "normal_view_consistency_gap": gaps["view_consistency_gap"],
        "normal_status": normal_metric.get("status", ""),
        "shuffled_status": shuffled_metric.get("status", ""),
        "notes": "positive reliability only; accuracy_delta = normal - shuffled",
    }


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics or str(Path(args.results_dir) / "metrics" / "main_results.csv")
    pairs = read_rows(args.pairs)
    if not pairs:
        print(f"No pair rows found at {args.pairs}")
        return 0
    metrics = metric_index(metrics_path)
    rows = [summarize_pair(pair, metrics, args.results_dir) for pair in pairs]
    fieldnames = [
        "dataset",
        "seed",
        "normal_run_id",
        "shuffled_run_id",
        "normal_accuracy",
        "shuffled_accuracy",
        "accuracy_delta",
        "normal_reliability_mean",
        "normal_reliability_std",
        "shuffled_reliability_mean",
        "shuffled_reliability_std",
        "normal_embedding_stability_mean",
        "normal_prediction_consistency_mean",
        "normal_view_reliability_gap",
        "normal_view_stability_gap",
        "normal_view_consistency_gap",
        "normal_status",
        "shuffled_status",
        "notes",
    ]
    write_csv(args.out, rows, fieldnames)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
