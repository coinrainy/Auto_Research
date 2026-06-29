import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/homophily_118")
    parser.add_argument("--baseline-method", default="grace")
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    return parser.parse_args()


def read_run(path):
    with open(path, "r", encoding="utf-8") as handle:
        p = json.load(handle)
    m = p["metrics"]
    s = p["split_stats"]
    d = p.get("diagnostics", {})
    return {
        "run_dir": str(path.parent),
        "dataset": p["dataset"],
        "method": p["method"],
        "seed": p["seed"],
        "split_index": p["split_index"],
        "accuracy": m["accuracy"],
        "F1Mi": m["F1Mi"],
        "F1Ma": m["F1Ma"],
        "val_accuracy": m["val_accuracy"],
        "best_c": m["best_c"],
        "train_ratio": s["train_ratio"],
        "val_ratio": s["val_ratio"],
        "test_ratio": s["test_ratio"],
        "shuffle_positives": d.get("shuffle_positives"),
        "neg_suppression": d.get("neg_suppression"),
        "semantic_weight": d.get("semantic_weight"),
        "num_negatives": d.get("num_negatives"),
        "raw_preserved": d.get("raw_preserved"),
        "raw_weight": d.get("raw_weight"),
        "gcl_weight": d.get("gcl_weight"),
        "rpgcl_auto_choice": d.get("rpgcl_auto_choice"),
        "semantic_top1_key_sim": d.get("semantic_top1_key_sim"),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(v) for v in values if v not in ("", None)]
    return "" if not values else sum(values) / len(values)


def main():
    args = parse_args()
    rows = [read_run(path) for path in sorted(Path(args.runs_dir).glob("*/run.json"))]
    baseline = {}
    for row in rows:
        if row["method"] == args.baseline_method:
            baseline[(row["dataset"], row["split_index"], row["seed"])] = row
    enriched = []
    for row in rows:
        item = dict(row)
        base = baseline.get((row["dataset"], row["split_index"], row["seed"]))
        for metric in ["accuracy", "F1Mi", "F1Ma"]:
            item[f"delta_vs_{args.baseline_method}_{metric}"] = (
                float(row[metric]) - float(base[metric])
                if base is not None and row["method"] != args.baseline_method
                else ""
            )
        enriched.append(item)
    grouped = defaultdict(list)
    for row in enriched:
        grouped[(
            row["dataset"],
            row["method"],
            row["shuffle_positives"],
            row["neg_suppression"],
            row["semantic_weight"],
            row["raw_preserved"],
        )].append(row)
    agg = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        dataset, method, shuffle, neg_supp, semantic_weight, raw_preserved = key
        agg.append({
            "dataset": dataset,
            "method": method,
            "shuffle_positives": shuffle,
            "neg_suppression": neg_supp,
            "semantic_weight": semantic_weight,
            "raw_preserved": raw_preserved,
            "num_runs": len(items),
            "accuracy_mean": mean([r["accuracy"] for r in items]),
            f"delta_vs_{args.baseline_method}_accuracy_mean": mean([
                r[f"delta_vs_{args.baseline_method}_accuracy"] for r in items
            ]),
            "positive_accuracy": sum(
                1 for r in items
                if r[f"delta_vs_{args.baseline_method}_accuracy"] not in ("", None)
                and float(r[f"delta_vs_{args.baseline_method}_accuracy"]) > 0
            ),
            "negative_accuracy": sum(
                1 for r in items
                if r[f"delta_vs_{args.baseline_method}_accuracy"] not in ("", None)
                and float(r[f"delta_vs_{args.baseline_method}_accuracy"]) < 0
            ),
            "F1Mi_mean": mean([r["F1Mi"] for r in items]),
            "F1Ma_mean": mean([r["F1Ma"] for r in items]),
            f"delta_vs_{args.baseline_method}_F1Mi_mean": mean([
                r[f"delta_vs_{args.baseline_method}_F1Mi"] for r in items
            ]),
            f"delta_vs_{args.baseline_method}_F1Ma_mean": mean([
                r[f"delta_vs_{args.baseline_method}_F1Ma"] for r in items
            ]),
            "positive_F1Mi": sum(
                1 for r in items
                if r[f"delta_vs_{args.baseline_method}_F1Mi"] not in ("", None)
                and float(r[f"delta_vs_{args.baseline_method}_F1Mi"]) > 0
            ),
            "negative_F1Mi": sum(
                1 for r in items
                if r[f"delta_vs_{args.baseline_method}_F1Mi"] not in ("", None)
                and float(r[f"delta_vs_{args.baseline_method}_F1Mi"]) < 0
            ),
        })
    write_csv(args.out or Path(args.runs_dir) / "runs_vs_baseline.csv", enriched)
    write_csv(args.aggregate_out or Path(args.runs_dir) / "aggregate_vs_baseline.csv", agg)
    print(json.dumps({"runs": len(enriched), "groups": len(agg)}, indent=2))


if __name__ == "__main__":
    main()
