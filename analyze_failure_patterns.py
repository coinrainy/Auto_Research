#!/usr/bin/env python3
"""Analyze when reliability weighting helps or hurts across datasets."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.data import edge_label_homophily, get_dataset_spec, load_pyg_dataset
from rwgcl.logging_utils import write_csv


BUCKET_LABELS = ["low", "mid", "high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reliability buckets with graph structure and method deltas."
    )
    parser.add_argument(
        "--comparison",
        default="results/diagnostics/rw_gcl_vs_grace_texas_wisconsin_cornell_actor_s0-9.csv",
        help="Per-seed method comparison CSV from summarize_method_comparison.py.",
    )
    parser.add_argument("--results-dir", default="results", help="Experiment results directory.")
    parser.add_argument(
        "--bucket-out",
        default="results/diagnostics/failure_analysis_buckets.csv",
        help="Output CSV with one row per dataset/seed/reliability bucket.",
    )
    parser.add_argument(
        "--run-out",
        default="results/diagnostics/failure_analysis_runs.csv",
        help="Output CSV with one row per normal RW-GCL run.",
    )
    parser.add_argument(
        "--summary-out",
        default="results/diagnostics/failure_analysis_summary.csv",
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


def tensor_mean(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.float().mean().item())


def tensor_std(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return None
    return float(finite.float().std(unbiased=False).item())


def class_entropy(labels: torch.Tensor, num_classes: int) -> tuple[float | None, float | None, int]:
    if labels.numel() == 0 or num_classes <= 0:
        return None, None, 0
    counts = torch.bincount(labels.long().clamp_min(0), minlength=num_classes).float()
    total = float(counts.sum().item())
    if total == 0.0:
        return None, None, 0
    probs = counts[counts > 0] / total
    entropy = float((-(probs * torch.log(probs))).sum().item())
    norm = entropy / math.log(num_classes) if num_classes > 1 else 0.0
    majority_fraction = float(counts.max().item() / total)
    coverage = int((counts > 0).sum().item())
    return norm, majority_fraction, coverage


def graph_profile(dataset_name: str) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    dataset = load_pyg_dataset(spec)
    data = dataset[0].cpu()
    labels = data.y.long().cpu()
    edge_index = data.edge_index.long().cpu()
    src, dst = edge_index
    num_nodes = int(data.num_nodes)
    num_edges = int(edge_index.size(1))
    degree = torch.bincount(src, minlength=num_nodes).float()

    valid = (labels[src] >= 0) & (labels[dst] >= 0)
    valid_src = src[valid]
    same = (labels[src][valid] == labels[dst][valid]).float()
    valid_degree = torch.bincount(valid_src, minlength=num_nodes).float()
    same_count = torch.zeros(num_nodes, dtype=torch.float)
    if valid_src.numel() > 0:
        same_count.index_add_(0, valid_src, same)
    local_homophily = torch.full((num_nodes,), float("nan"))
    has_neighbors = valid_degree > 0
    local_homophily[has_neighbors] = same_count[has_neighbors] / valid_degree[has_neighbors]

    return {
        "dataset": dataset_name,
        "data": data,
        "labels": labels,
        "degree": degree,
        "local_homophily": local_homophily,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "edge_label_homophily": edge_label_homophily(data),
        "avg_degree": float(degree.mean().item()),
        "degree_std": float(degree.std(unbiased=False).item()),
        "local_homophily_mean": tensor_mean(local_homophily),
        "local_homophily_std": tensor_std(local_homophily),
    }


def run_payload(results_dir: str | Path, run_id: str) -> dict:
    path = resolve_path(results_dir) / "raw" / run_id / "embeddings.pt"
    if not path.exists():
        return {}
    return torch.load(path, map_location="cpu", weights_only=True)


def mask_fraction(mask: torch.Tensor, idx: torch.Tensor) -> float | None:
    if idx.numel() == 0:
        return None
    return float(mask[idx].bool().float().mean().item())


def bucket_indices(reliability: torch.Tensor) -> list[torch.Tensor]:
    order = torch.argsort(reliability.float())
    return list(torch.chunk(order, len(BUCKET_LABELS)))


def bucket_rows_for_run(row: dict, profile: dict[str, Any], results_dir: str | Path) -> list[dict]:
    run_id = row.get("rw_normal_run_id", "")
    payload = run_payload(results_dir, run_id)
    required = {"positive_reliability", "embedding_stability", "prediction_consistency", "labels"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        return [
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "bucket": "not_available",
                "status": "missing_artifact",
                "notes": f"missing={','.join(missing)}",
            }
        ]

    reliability = payload["positive_reliability"].float()
    stability = payload["embedding_stability"].float()
    consistency = payload["prediction_consistency"].float()
    labels = payload["labels"].long()
    train_mask = payload.get("train_mask", torch.zeros_like(labels, dtype=torch.bool)).bool()
    val_mask = payload.get("val_mask", torch.zeros_like(labels, dtype=torch.bool)).bool()
    test_mask = payload.get("test_mask", torch.zeros_like(labels, dtype=torch.bool)).bool()
    degree = profile["degree"].float()
    local_homophily = profile["local_homophily"].float()
    num_classes = int(profile["num_classes"])

    rows = []
    for bucket, idx in zip(BUCKET_LABELS, bucket_indices(reliability)):
        bucket_labels = labels[idx]
        entropy, majority_fraction, class_coverage = class_entropy(bucket_labels, num_classes)
        bucket_local_homophily = local_homophily[idx]
        bucket_degree = degree[idx]
        rows.append(
            {
                "dataset": row.get("dataset", ""),
                "seed": row.get("seed", ""),
                "run_id": run_id,
                "bucket": bucket,
                "count": str(int(idx.numel())),
                "rw_normal_minus_grace": row.get("rw_normal_minus_grace", ""),
                "rw_normal_minus_shuffled": row.get("rw_normal_minus_shuffled", ""),
                "reliability_mean": format_float(tensor_mean(reliability[idx])),
                "embedding_stability_mean": format_float(tensor_mean(stability[idx])),
                "prediction_consistency_mean": format_float(tensor_mean(consistency[idx])),
                "degree_mean": format_float(tensor_mean(bucket_degree)),
                "degree_std": format_float(tensor_std(bucket_degree)),
                "local_homophily_mean": format_float(tensor_mean(bucket_local_homophily)),
                "local_homophily_std": format_float(tensor_std(bucket_local_homophily)),
                "class_entropy_norm": format_float(entropy),
                "majority_class_fraction": format_float(majority_fraction),
                "class_coverage": str(class_coverage),
                "train_fraction": format_float(mask_fraction(train_mask, idx)),
                "val_fraction": format_float(mask_fraction(val_mask, idx)),
                "test_fraction": format_float(mask_fraction(test_mask, idx)),
                "graph_edge_homophily": format_float(profile["edge_label_homophily"]),
                "graph_avg_degree": format_float(profile["avg_degree"]),
                "graph_local_homophily_mean": format_float(profile["local_homophily_mean"]),
                "status": "computed",
                "notes": "normal RW-GCL run bucketed by positive_reliability",
            }
        )
    return rows


def group_by_dataset(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("dataset", ""), []).append(row)
    return grouped


def group_by_run(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("seed", ""), row.get("run_id", ""))
        grouped.setdefault(key, []).append(row)
    return grouped


def first_float(rows: list[dict], key: str) -> float | None:
    for row in rows:
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def values(rows: list[dict], key: str, bucket: str | None = None) -> list[float]:
    out = []
    for row in rows:
        if bucket is not None and row.get("bucket") != bucket:
            continue
        value = as_float(row.get(key))
        if value is not None:
            out.append(value)
    return out


def high_low_gap(rows: list[dict], key: str) -> float | None:
    high = mean(values(rows, key, bucket="high"))
    low = mean(values(rows, key, bucket="low"))
    if high is None or low is None:
        return None
    return high - low


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


def sign_counts(vals: list[float]) -> tuple[int, int, int]:
    return (
        sum(1 for value in vals if value > 0),
        sum(1 for value in vals if value == 0),
        sum(1 for value in vals if value < 0),
    )


def run_summary_rows(bucket_rows: list[dict]) -> list[dict]:
    rows = []
    for (dataset, seed, run_id), group in sorted(group_by_run([row for row in bucket_rows if row.get("status") == "computed"]).items()):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "run_id": run_id,
                "rw_normal_minus_grace": group[0].get("rw_normal_minus_grace", ""),
                "rw_normal_minus_shuffled": group[0].get("rw_normal_minus_shuffled", ""),
                "graph_edge_homophily": group[0].get("graph_edge_homophily", ""),
                "graph_avg_degree": group[0].get("graph_avg_degree", ""),
                "graph_local_homophily_mean": group[0].get("graph_local_homophily_mean", ""),
                "high_low_reliability_gap": format_float(high_low_gap(group, "reliability_mean")),
                "high_low_embedding_stability_gap": format_float(high_low_gap(group, "embedding_stability_mean")),
                "high_low_prediction_consistency_gap": format_float(high_low_gap(group, "prediction_consistency_mean")),
                "high_low_degree_gap": format_float(high_low_gap(group, "degree_mean")),
                "high_low_local_homophily_gap": format_float(high_low_gap(group, "local_homophily_mean")),
                "high_low_class_entropy_gap": format_float(high_low_gap(group, "class_entropy_norm")),
                "high_bucket_local_homophily_mean": format_float(mean(values(group, "local_homophily_mean", "high"))),
                "low_bucket_local_homophily_mean": format_float(mean(values(group, "local_homophily_mean", "low"))),
                "high_bucket_majority_class_fraction": format_float(
                    mean(values(group, "majority_class_fraction", "high"))
                ),
                "low_bucket_majority_class_fraction": format_float(
                    mean(values(group, "majority_class_fraction", "low"))
                ),
                "status": "computed",
                "notes": "per-run high-low gaps use high reliability bucket minus low reliability bucket",
            }
        )
    return rows


def summary_rows(run_rows: list[dict]) -> list[dict]:
    summary = []
    for dataset, rows in sorted(group_by_dataset([row for row in run_rows if row.get("status") == "computed"]).items()):
        normal_minus_grace = values(rows, "rw_normal_minus_grace")
        normal_minus_shuffled = values(rows, "rw_normal_minus_shuffled")
        pos, zero, neg = sign_counts(normal_minus_grace)
        summary.append(
            {
                "dataset": dataset,
                "runs": str(len({row.get("seed", "") for row in rows})),
                "rw_normal_minus_grace_mean": format_float(mean(normal_minus_grace)),
                "rw_normal_minus_grace_std": format_float(pstdev(normal_minus_grace)),
                "rw_normal_minus_grace_positive_count": str(pos),
                "rw_normal_minus_grace_zero_count": str(zero),
                "rw_normal_minus_grace_negative_count": str(neg),
                "rw_normal_minus_shuffled_mean": format_float(mean(normal_minus_shuffled)),
                "rw_normal_minus_shuffled_std": format_float(pstdev(normal_minus_shuffled)),
                "graph_edge_homophily": format_float(first_float(rows, "graph_edge_homophily")),
                "graph_avg_degree": format_float(first_float(rows, "graph_avg_degree")),
                "graph_local_homophily_mean": format_float(first_float(rows, "graph_local_homophily_mean")),
                "high_low_reliability_gap_mean": format_float(mean(values(rows, "high_low_reliability_gap"))),
                "high_low_embedding_stability_gap_mean": format_float(
                    mean(values(rows, "high_low_embedding_stability_gap"))
                ),
                "high_low_prediction_consistency_gap_mean": format_float(
                    mean(values(rows, "high_low_prediction_consistency_gap"))
                ),
                "high_low_degree_gap_mean": format_float(mean(values(rows, "high_low_degree_gap"))),
                "high_low_local_homophily_gap_mean": format_float(
                    mean(values(rows, "high_low_local_homophily_gap"))
                ),
                "high_low_class_entropy_gap_mean": format_float(mean(values(rows, "high_low_class_entropy_gap"))),
                "corr_delta_vs_degree_gap": format_float(
                    pearson(normal_minus_grace, values(rows, "high_low_degree_gap"))
                ),
                "corr_delta_vs_local_homophily_gap": format_float(
                    pearson(normal_minus_grace, values(rows, "high_low_local_homophily_gap"))
                ),
                "corr_delta_vs_class_entropy_gap": format_float(
                    pearson(normal_minus_grace, values(rows, "high_low_class_entropy_gap"))
                ),
                "high_bucket_local_homophily_mean": format_float(mean(values(rows, "high_bucket_local_homophily_mean"))),
                "low_bucket_local_homophily_mean": format_float(mean(values(rows, "low_bucket_local_homophily_mean"))),
                "high_bucket_majority_class_fraction_mean": format_float(
                    mean(values(rows, "high_bucket_majority_class_fraction"))
                ),
                "low_bucket_majority_class_fraction_mean": format_float(
                    mean(values(rows, "low_bucket_majority_class_fraction"))
                ),
                "notes": "correlations use per-run gaps within each dataset; n is small for early diagnosis",
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    comparison_rows = [row for row in read_rows(args.comparison) if row.get("status") == "computed"]
    if not comparison_rows:
        print(f"No computed comparison rows found at {args.comparison}")
        return 0

    profiles = {dataset: graph_profile(dataset) for dataset in sorted({row["dataset"] for row in comparison_rows})}
    bucket_rows = []
    for row in comparison_rows:
        bucket_rows.extend(bucket_rows_for_run(row, profiles[row["dataset"]], args.results_dir))

    bucket_fieldnames = [
        "dataset",
        "seed",
        "run_id",
        "bucket",
        "count",
        "rw_normal_minus_grace",
        "rw_normal_minus_shuffled",
        "reliability_mean",
        "embedding_stability_mean",
        "prediction_consistency_mean",
        "degree_mean",
        "degree_std",
        "local_homophily_mean",
        "local_homophily_std",
        "class_entropy_norm",
        "majority_class_fraction",
        "class_coverage",
        "train_fraction",
        "val_fraction",
        "test_fraction",
        "graph_edge_homophily",
        "graph_avg_degree",
        "graph_local_homophily_mean",
        "status",
        "notes",
    ]
    summary_fieldnames = [
        "dataset",
        "runs",
        "rw_normal_minus_grace_mean",
        "rw_normal_minus_grace_std",
        "rw_normal_minus_grace_positive_count",
        "rw_normal_minus_grace_zero_count",
        "rw_normal_minus_grace_negative_count",
        "rw_normal_minus_shuffled_mean",
        "rw_normal_minus_shuffled_std",
        "graph_edge_homophily",
        "graph_avg_degree",
        "graph_local_homophily_mean",
        "high_low_reliability_gap_mean",
        "high_low_embedding_stability_gap_mean",
        "high_low_prediction_consistency_gap_mean",
        "high_low_degree_gap_mean",
        "high_low_local_homophily_gap_mean",
        "high_low_class_entropy_gap_mean",
        "corr_delta_vs_degree_gap",
        "corr_delta_vs_local_homophily_gap",
        "corr_delta_vs_class_entropy_gap",
        "high_bucket_local_homophily_mean",
        "low_bucket_local_homophily_mean",
        "high_bucket_majority_class_fraction_mean",
        "low_bucket_majority_class_fraction_mean",
        "notes",
    ]
    run_fieldnames = [
        "dataset",
        "seed",
        "run_id",
        "rw_normal_minus_grace",
        "rw_normal_minus_shuffled",
        "graph_edge_homophily",
        "graph_avg_degree",
        "graph_local_homophily_mean",
        "high_low_reliability_gap",
        "high_low_embedding_stability_gap",
        "high_low_prediction_consistency_gap",
        "high_low_degree_gap",
        "high_low_local_homophily_gap",
        "high_low_class_entropy_gap",
        "high_bucket_local_homophily_mean",
        "low_bucket_local_homophily_mean",
        "high_bucket_majority_class_fraction",
        "low_bucket_majority_class_fraction",
        "status",
        "notes",
    ]
    run_rows = run_summary_rows(bucket_rows)
    write_csv(args.bucket_out, bucket_rows, bucket_fieldnames)
    write_csv(args.run_out, run_rows, run_fieldnames)
    write_csv(args.summary_out, summary_rows(run_rows), summary_fieldnames)
    print(f"wrote {args.bucket_out}")
    print(f"wrote {args.run_out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
