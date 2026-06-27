#!/usr/bin/env python3
"""Estimate label-based false-negative pressure in learned embeddings."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.logging_utils import write_csv


BUCKET_LABELS = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute label-based same-class negative softmax mass for RW-GCL normal runs."
    )
    parser.add_argument(
        "--comparison",
        required=True,
        help="Per-seed method comparison CSV from summarize_method_comparison.py.",
    )
    parser.add_argument("--results-dir", default="results", help="Experiment results directory.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for similarity mass.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Anchor chunk size for pairwise similarity.")
    parser.add_argument(
        "--run-out",
        required=True,
        help="Output CSV with one row per run.",
    )
    parser.add_argument(
        "--bucket-out",
        required=True,
        help="Output CSV with one row per run/reliability bucket.",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        help="Output CSV with one row per dataset.",
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


def run_payload(results_dir: str | Path, run_id: str) -> dict:
    path = resolve_path(results_dir) / "raw" / run_id / "embeddings.pt"
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
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def pstdev(values: list[float]) -> float | None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    mu = sum(clean) / len(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / len(clean))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return None
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))
    if x_denom == 0.0 or y_denom == 0.0:
        return None
    return numerator / (x_denom * y_denom)


def tensor_mean(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    weights = weights.float().clamp_min(0.0)
    denom = weights.sum()
    if float(denom.item()) <= 0.0:
        return None
    return float((values.float() * weights).sum().item() / denom.item())


def bucket_indices(reliability: torch.Tensor) -> list[torch.Tensor]:
    order = torch.argsort(reliability.float())
    return list(torch.chunk(order, len(BUCKET_LABELS)))


def false_negative_pressure(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    z = F.normalize(embeddings.float(), dim=1)
    labels = labels.long()
    n = int(z.size(0))
    out = torch.zeros(n, dtype=torch.float)
    all_indices = torch.arange(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        idx = all_indices[start:end]
        logits = z[start:end] @ z.t()
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values
        mass = logits.exp()
        mass[:, start:end].diagonal().zero_()
        same_label = labels[start:end, None] == labels[None, :]
        same_label[:, start:end].fill_diagonal_(False)
        denom = mass.sum(dim=1).clamp_min(1e-12)
        numerator = (mass * same_label.float()).sum(dim=1)
        out[idx] = (numerator / denom).float()
    return out


def summarize_run(row: dict, results_dir: str | Path, temperature: float, chunk_size: int) -> tuple[dict, list[dict]]:
    run_id = row.get("rw_normal_run_id", "")
    payload = run_payload(results_dir, run_id)
    required = {"embeddings", "labels", "positive_reliability"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        run_row = {
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "run_id": run_id,
            "status": "missing_artifact",
            "notes": f"missing={','.join(missing)}",
        }
        return run_row, []

    labels = payload["labels"].long()
    reliability = payload["positive_reliability"].float()
    pressure = false_negative_pressure(
        embeddings=payload["embeddings"],
        labels=labels,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    same_class_candidates = torch.bincount(labels, minlength=int(labels.max().item()) + 1)[labels] - 1
    has_false_negative_candidates = same_class_candidates > 0
    low_idx, _, high_idx = bucket_indices(reliability)
    high_low_pressure_gap = tensor_mean(pressure[high_idx])
    low_pressure = tensor_mean(pressure[low_idx])
    if high_low_pressure_gap is not None and low_pressure is not None:
        high_low_pressure_gap = high_low_pressure_gap - low_pressure
    else:
        high_low_pressure_gap = None

    run_row = {
        "dataset": row.get("dataset", ""),
        "seed": row.get("seed", ""),
        "run_id": run_id,
        "rw_normal_minus_grace": row.get("rw_normal_minus_grace", ""),
        "rw_normal_minus_shuffled": row.get("rw_normal_minus_shuffled", ""),
        "false_negative_pressure_mean": format_float(tensor_mean(pressure)),
        "false_negative_pressure_weighted_mean": format_float(weighted_mean(pressure, reliability)),
        "false_negative_pressure_high_low_gap": format_float(high_low_pressure_gap),
        "reliability_pressure_corr": format_float(
            pearson(reliability.tolist(), pressure.tolist())
        ),
        "same_class_candidate_fraction": format_float(float(has_false_negative_candidates.float().mean().item())),
        "temperature": format_float(temperature),
        "status": "computed",
        "notes": "label-based diagnostic only; same-label non-self softmax mass in embedding space",
    }

    bucket_rows = []
    for bucket, idx in zip(BUCKET_LABELS, bucket_indices(reliability)):
        bucket_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "run_id": run_id,
                "bucket": bucket,
                "count": str(int(idx.numel())),
                "rw_normal_minus_grace": row.get("rw_normal_minus_grace", ""),
                "false_negative_pressure_mean": format_float(tensor_mean(pressure[idx])),
                "reliability_mean": format_float(tensor_mean(reliability[idx])),
                "same_class_candidate_fraction": format_float(
                    float(has_false_negative_candidates[idx].float().mean().item())
                ),
                "status": "computed",
                "notes": "bucketed by positive_reliability",
            }
        )
    return run_row, bucket_rows


def values(rows: list[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def group_by_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("dataset", ""), []).append(row)
    return grouped


def sign_counts(vals: list[float]) -> tuple[int, int, int]:
    return (
        sum(1 for value in vals if value > 0),
        sum(1 for value in vals if value == 0),
        sum(1 for value in vals if value < 0),
    )


def summary_rows(run_rows: list[dict]) -> list[dict]:
    summary = []
    for dataset, rows in sorted(group_by_dataset([row for row in run_rows if row.get("status") == "computed"]).items()):
        delta = values(rows, "rw_normal_minus_grace")
        pressure = values(rows, "false_negative_pressure_mean")
        weighted_pressure = values(rows, "false_negative_pressure_weighted_mean")
        pressure_gap = values(rows, "false_negative_pressure_high_low_gap")
        corr = values(rows, "reliability_pressure_corr")
        pos, zero, neg = sign_counts(delta)
        summary.append(
            {
                "dataset": dataset,
                "runs": str(len(rows)),
                "rw_normal_minus_grace_mean": format_float(mean(delta)),
                "rw_normal_minus_grace_positive_count": str(pos),
                "rw_normal_minus_grace_zero_count": str(zero),
                "rw_normal_minus_grace_negative_count": str(neg),
                "false_negative_pressure_mean": format_float(mean(pressure)),
                "false_negative_pressure_weighted_mean": format_float(mean(weighted_pressure)),
                "false_negative_pressure_weighted_minus_unweighted": format_float(
                    None if mean(weighted_pressure) is None or mean(pressure) is None else mean(weighted_pressure) - mean(pressure)
                ),
                "false_negative_pressure_high_low_gap_mean": format_float(mean(pressure_gap)),
                "reliability_pressure_corr_mean": format_float(mean(corr)),
                "reliability_pressure_corr_std": format_float(pstdev(corr)),
                "same_class_candidate_fraction_mean": format_float(mean(values(rows, "same_class_candidate_fraction"))),
                "notes": "label-based diagnostic only; higher pressure means more same-label negatives in denominator mass",
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    comparison_rows = [row for row in read_rows(args.comparison) if row.get("status") == "computed"]
    if not comparison_rows:
        print(f"No computed comparison rows found at {args.comparison}")
        return 0

    run_rows = []
    bucket_rows = []
    for row in comparison_rows:
        run_row, buckets = summarize_run(
            row=row,
            results_dir=args.results_dir,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
        )
        run_rows.append(run_row)
        bucket_rows.extend(buckets)

    run_fieldnames = [
        "dataset",
        "seed",
        "run_id",
        "rw_normal_minus_grace",
        "rw_normal_minus_shuffled",
        "false_negative_pressure_mean",
        "false_negative_pressure_weighted_mean",
        "false_negative_pressure_high_low_gap",
        "reliability_pressure_corr",
        "same_class_candidate_fraction",
        "temperature",
        "status",
        "notes",
    ]
    bucket_fieldnames = [
        "dataset",
        "seed",
        "run_id",
        "bucket",
        "count",
        "rw_normal_minus_grace",
        "false_negative_pressure_mean",
        "reliability_mean",
        "same_class_candidate_fraction",
        "status",
        "notes",
    ]
    summary_fieldnames = [
        "dataset",
        "runs",
        "rw_normal_minus_grace_mean",
        "rw_normal_minus_grace_positive_count",
        "rw_normal_minus_grace_zero_count",
        "rw_normal_minus_grace_negative_count",
        "false_negative_pressure_mean",
        "false_negative_pressure_weighted_mean",
        "false_negative_pressure_weighted_minus_unweighted",
        "false_negative_pressure_high_low_gap_mean",
        "reliability_pressure_corr_mean",
        "reliability_pressure_corr_std",
        "same_class_candidate_fraction_mean",
        "notes",
    ]
    write_csv(args.run_out, run_rows, run_fieldnames)
    write_csv(args.bucket_out, bucket_rows, bucket_fieldnames)
    write_csv(args.summary_out, summary_rows(run_rows), summary_fieldnames)
    print(f"wrote {args.run_out}")
    print(f"wrote {args.bucket_out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
