"""Diagnostic writers for RW-GCL runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .logging_utils import ensure_dir, write_csv


AVAILABLE_DIAGNOSTICS = {
    "shuffled_reliability",
    "false_negative_mass",
    "view_consistency",
}


def run_dir(results_dir: str | Path, run_id: str) -> Path:
    return PROJECT_ROOT / results_dir / "raw" / run_id


def load_run_payload(results_dir: str | Path, run_id: str) -> dict:
    path = run_dir(results_dir, run_id) / "embeddings.pt"
    if not path.exists():
        raise FileNotFoundError(f"Run artifact not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_run_metadata(results_dir: str | Path, run_id: str) -> dict:
    path = run_dir(results_dir, run_id) / "run_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(results_dir: str | Path, run_id: str) -> dict:
    path = PROJECT_ROOT / results_dir / "metrics" / "main_results.csv"
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_id") == run_id:
                return row
    return {}


def write_view_consistency(output_dir: str | Path, results_dir: str | Path, run_id: str) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / "view_consistency.csv"
    payload = load_run_payload(results_dir, run_id)
    required = {"positive_reliability", "embedding_stability", "prediction_consistency"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        rows = [
            {
                "run_id": run_id,
                "bucket": "not_available",
                "count": 0,
                "reliability_mean": "",
                "embedding_stability_mean": "",
                "prediction_consistency_mean": "",
                "status": "missing_artifact",
                "notes": f"missing={','.join(missing)}",
            }
        ]
    else:
        reliability = payload["positive_reliability"].float()
        stability = payload["embedding_stability"].float()
        consistency = payload["prediction_consistency"].float()
        order = torch.argsort(reliability)
        chunks = torch.chunk(order, 3)
        labels = ["low", "mid", "high"]
        rows = []
        for label, idx in zip(labels, chunks):
            rows.append(
                {
                    "run_id": run_id,
                    "bucket": label,
                    "count": int(idx.numel()),
                    "reliability_mean": f"{float(reliability[idx].mean().item()):.6f}",
                    "embedding_stability_mean": f"{float(stability[idx].mean().item()):.6f}",
                    "prediction_consistency_mean": f"{float(consistency[idx].mean().item()):.6f}",
                    "status": "computed",
                    "notes": "bucketed by positive_reliability",
                }
            )
    write_csv(
        out_path,
        rows,
        [
            "run_id",
            "bucket",
            "count",
            "reliability_mean",
            "embedding_stability_mean",
            "prediction_consistency_mean",
            "status",
            "notes",
        ],
    )
    return out_path


def write_shuffled_reliability(
    output_dir: str | Path,
    results_dir: str | Path,
    run_id: str,
    compare_run_id: str | None,
) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / "shuffled_reliability.csv"
    if not compare_run_id:
        rows = [
            {
                "run_id": run_id,
                "compare_run_id": "",
                "run_accuracy": "",
                "compare_accuracy": "",
                "accuracy_delta": "",
                "run_reliability_mean": "",
                "compare_reliability_mean": "",
                "status": "needs_compare_run_id",
                "notes": "Pass --compare-run-id with the shuffled or normal counterpart.",
            }
        ]
    else:
        meta_a = load_run_metadata(results_dir, run_id)
        meta_b = load_run_metadata(results_dir, compare_run_id)
        row_a = metric_row(results_dir, run_id)
        row_b = metric_row(results_dir, compare_run_id)
        acc_a = float(row_a.get("value") or "nan")
        acc_b = float(row_b.get("value") or "nan")
        rel_a = meta_a.get("last_reliability_summary", {}).get("reliability_mean", "")
        rel_b = meta_b.get("last_reliability_summary", {}).get("reliability_mean", "")
        rows = [
            {
                "run_id": run_id,
                "compare_run_id": compare_run_id,
                "run_accuracy": f"{acc_a:.6f}",
                "compare_accuracy": f"{acc_b:.6f}",
                "accuracy_delta": f"{acc_a - acc_b:.6f}",
                "run_reliability_mean": rel_a,
                "compare_reliability_mean": rel_b,
                "status": "computed",
                "notes": (
                    f"run_shuffled={meta_a.get('shuffled_reliability')}; "
                    f"compare_shuffled={meta_b.get('shuffled_reliability')}"
                ),
            }
        ]
    write_csv(
        out_path,
        rows,
        [
            "run_id",
            "compare_run_id",
            "run_accuracy",
            "compare_accuracy",
            "accuracy_delta",
            "run_reliability_mean",
            "compare_reliability_mean",
            "status",
            "notes",
        ],
    )
    return out_path


def write_false_negative_mass(output_dir: str | Path, run_id: str) -> Path:
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / "false_negative_mass.csv"
    rows = [
        {
            "run_id": run_id,
            "status": "not_applicable_positive_only",
            "weighted_false_negative_mass": "",
            "notes": "Negative weighting is not implemented yet, so false-negative mass is not meaningful.",
        }
    ]
    write_csv(out_path, rows, ["run_id", "status", "weighted_false_negative_mass", "notes"])
    return out_path


def run_diagnostic(
    output_dir: str | Path,
    results_dir: str | Path,
    run_id: str,
    diagnostic: str,
    compare_run_id: str | None = None,
) -> Path:
    if diagnostic not in AVAILABLE_DIAGNOSTICS:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")
    if diagnostic == "view_consistency":
        return write_view_consistency(output_dir, results_dir, run_id)
    if diagnostic == "shuffled_reliability":
        return write_shuffled_reliability(output_dir, results_dir, run_id, compare_run_id)
    if diagnostic == "false_negative_mass":
        return write_false_negative_mass(output_dir, run_id)
    raise AssertionError(f"Unhandled diagnostic: {diagnostic}")
