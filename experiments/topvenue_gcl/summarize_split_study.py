import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/split_study")
    parser.add_argument("--baseline-method", default="grace")
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    return parser.parse_args()


def read_run_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    diagnostics = payload.get("diagnostics", {})
    config = payload.get("config", {})
    return {
        "run_dir": str(path.parent),
        "dataset": payload.get("dataset"),
        "method": payload.get("method"),
        "seed": payload.get("seed"),
        "split_index": payload.get("split_index"),
        "cache_control": diagnostics.get("cache_control", "none"),
        "epochs": config.get("epochs"),
        "F1Mi": metrics.get("F1Mi", metrics.get("accuracy")),
        "F1Ma": metrics.get("F1Ma"),
        "accuracy": metrics.get("accuracy"),
        "best_c": metrics.get("best_c"),
        "energy_ratio_mean": diagnostics.get("energy_ratio_mean"),
        "cache_low_sim_mean": diagnostics.get("cache_low_sim_mean"),
        "sspnv_semantic_sim_mean": diagnostics.get("sspnv_semantic_sim_mean"),
        "sspnv_spatial_self_fraction": diagnostics.get("sspnv_spatial_self_fraction"),
        "sspnv_random_semantic": diagnostics.get("sspnv_random_semantic"),
        "sspnv_random_spatial": diagnostics.get("sspnv_random_spatial"),
        "afpnv_semantic_weight_mean": diagnostics.get("afpnv_semantic_weight_mean"),
        "afpnv_spatial_weight_mean": diagnostics.get("afpnv_spatial_weight_mean"),
        "afpnv_semantic_conf_mean": diagnostics.get("afpnv_semantic_conf_mean"),
        "afpnv_spatial_conf_mean": diagnostics.get("afpnv_spatial_conf_mean"),
        "bspnv_semantic_prob_mean": diagnostics.get("bspnv_semantic_prob_mean"),
        "bspnv_spatial_prob_mean": diagnostics.get("bspnv_spatial_prob_mean"),
        "bspnv_bootstrap_prob_mean": diagnostics.get("bspnv_bootstrap_prob_mean"),
        "bspnv_semantic_win_fraction": diagnostics.get("bspnv_semantic_win_fraction"),
        "bspnv_spatial_win_fraction": diagnostics.get("bspnv_spatial_win_fraction"),
        "bspnv_bootstrap_win_fraction": diagnostics.get("bspnv_bootstrap_win_fraction"),
        "aompnv_semantic_prob_mean": diagnostics.get("aompnv_semantic_prob_mean"),
        "aompnv_spatial_prob_mean": diagnostics.get("aompnv_spatial_prob_mean"),
        "aompnv_bootstrap_prob_mean": diagnostics.get("aompnv_bootstrap_prob_mean"),
        "aompnv_semantic_win_fraction": diagnostics.get("aompnv_semantic_win_fraction"),
        "aompnv_spatial_win_fraction": diagnostics.get("aompnv_spatial_win_fraction"),
        "aompnv_bootstrap_win_fraction": diagnostics.get("aompnv_bootstrap_win_fraction"),
        "aompnv_semantic_conf_mean": diagnostics.get("aompnv_semantic_conf_mean"),
        "aompnv_spatial_conf_mean": diagnostics.get("aompnv_spatial_conf_mean"),
        "aompnv_semantic_loss_mean": diagnostics.get("aompnv_semantic_loss_mean"),
        "aompnv_spatial_loss_mean": diagnostics.get("aompnv_spatial_loss_mean"),
        "aompnv_bootstrap_loss_mean": diagnostics.get("aompnv_bootstrap_loss_mean"),
        "aompnv_semantic_pos_mean": diagnostics.get("aompnv_semantic_pos_mean"),
        "aompnv_spatial_pos_mean": diagnostics.get("aompnv_spatial_pos_mean"),
        "aompnv_shuffle_positives": diagnostics.get("aompnv_shuffle_positives"),
        "mpnv_semantic_pos_mean": diagnostics.get("mpnv_semantic_pos_mean"),
        "mpnv_spatial_pos_mean": diagnostics.get("mpnv_spatial_pos_mean"),
        "mpnv_semantic_density": diagnostics.get("mpnv_semantic_density"),
        "mpnv_spatial_density": diagnostics.get("mpnv_spatial_density"),
        "mpnv_shuffle_positives": diagnostics.get("mpnv_shuffle_positives"),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(v) for v in values if v is not None and v != ""]
    if not values:
        return ""
    return sum(values) / len(values)


def aggregate(rows, baseline_method):
    key_to_baseline = {}
    for row in rows:
        key = (row["dataset"], row["split_index"], row["seed"])
        if row["method"] == baseline_method:
            key_to_baseline[key] = row

    enriched = []
    for row in rows:
        item = dict(row)
        key = (row["dataset"], row["split_index"], row["seed"])
        baseline = key_to_baseline.get(key)
        if baseline and row["method"] != baseline_method:
            for metric in ["F1Mi", "F1Ma", "accuracy"]:
                base_value = baseline.get(metric)
                value = row.get(metric)
                item[f"delta_vs_{baseline_method}_{metric}"] = (
                    float(value) - float(base_value)
                    if value is not None and base_value is not None
                    else ""
                )
        else:
            for metric in ["F1Mi", "F1Ma", "accuracy"]:
                item[f"delta_vs_{baseline_method}_{metric}"] = ""
        enriched.append(item)

    groups = defaultdict(list)
    for row in enriched:
        groups[(row["dataset"], row["method"], row["cache_control"])].append(row)

    aggregate_rows = []
    for (dataset, method, cache_control), items in sorted(groups.items()):
        agg = {
            "dataset": dataset,
            "method": method,
            "cache_control": cache_control,
            "num_runs": len(items),
            "F1Mi_mean": mean([item.get("F1Mi") for item in items]),
            "F1Ma_mean": mean([item.get("F1Ma") for item in items]),
            "accuracy_mean": mean([item.get("accuracy") for item in items]),
            f"delta_vs_{baseline_method}_F1Mi_mean": mean([
                item.get(f"delta_vs_{baseline_method}_F1Mi") for item in items
            ]),
            f"delta_vs_{baseline_method}_F1Ma_mean": mean([
                item.get(f"delta_vs_{baseline_method}_F1Ma") for item in items
            ]),
            f"delta_vs_{baseline_method}_accuracy_mean": mean([
                item.get(f"delta_vs_{baseline_method}_accuracy") for item in items
            ]),
            "positive_delta_F1Mi_count": sum(
                1 for item in items
                if item.get(f"delta_vs_{baseline_method}_F1Mi") not in ["", None]
                and float(item[f"delta_vs_{baseline_method}_F1Mi"]) > 0
            ),
            "negative_delta_F1Mi_count": sum(
                1 for item in items
                if item.get(f"delta_vs_{baseline_method}_F1Mi") not in ["", None]
                and float(item[f"delta_vs_{baseline_method}_F1Mi"]) < 0
            ),
        }
        aggregate_rows.append(agg)
    return enriched, aggregate_rows


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = [
        read_run_json(path)
        for path in sorted(runs_dir.glob("*/run.json"))
    ]
    rows, aggregate_rows = aggregate(rows, args.baseline_method)
    out = args.out or str(runs_dir / "split_study_runs.csv")
    aggregate_out = args.aggregate_out or str(runs_dir / "split_study_aggregate.csv")
    write_csv(out, rows)
    write_csv(aggregate_out, aggregate_rows)
    print(f"Wrote {len(rows)} run rows to {out}")
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {aggregate_out}")


if __name__ == "__main__":
    main()
