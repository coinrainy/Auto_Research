import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

CANDIDATES = ["hpfs", "raw_preserved_hpfs", "raw_features"]


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
    row = {
        "run_dir": str(path.parent),
        "dataset": p["dataset"],
        "method": p["method"],
        "seed": p["seed"],
        "model_seed": p.get("model_seed", p["seed"]),
        "split_seed": p.get("split_seed", int(p["config"].get("split_base_seed", 2026)) + int(p["split_index"])),
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
    for name in CANDIDATES:
        row[f"rpgcl_auto_{name}_val_accuracy"] = d.get(f"rpgcl_auto_{name}_val_accuracy")
        row[f"rpgcl_auto_{name}_test_accuracy"] = d.get(
            f"rpgcl_auto_{name}_test_accuracy",
            d.get(f"rpgcl_auto_{name}_test_F1Mi"),
        )
    available = [
        (
            name,
            row[f"rpgcl_auto_{name}_test_accuracy"],
        )
        for name in CANDIDATES
        if row[f"rpgcl_auto_{name}_test_accuracy"] not in ("", None)
    ]
    if available:
        oracle_name, oracle_acc = max(available, key=lambda item: float(item[1]))
        row["rpgcl_auto_oracle_choice"] = oracle_name
        row["rpgcl_auto_oracle_test_accuracy"] = float(oracle_acc)
    else:
        row["rpgcl_auto_oracle_choice"] = ""
        row["rpgcl_auto_oracle_test_accuracy"] = ""
    return row


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


def count_choice(items, choice):
    return sum(1 for row in items if row.get("rpgcl_auto_choice") == choice)


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
        )].append(row)
    agg = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        dataset, method, shuffle, neg_supp, semantic_weight = key
        deltas = [
            r[f"delta_vs_{args.baseline_method}_accuracy"]
            for r in items
            if r[f"delta_vs_{args.baseline_method}_accuracy"] not in ("", None)
        ]
        agg.append({
            "dataset": dataset,
            "method": method,
            "shuffle_positives": shuffle,
            "neg_suppression": neg_supp,
            "semantic_weight": semantic_weight,
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
            "zero_accuracy": sum(1 for delta in deltas if float(delta) == 0.0),
            "rpgcl_auto_choice_hpfs": count_choice(items, "hpfs"),
            "rpgcl_auto_choice_raw_preserved_hpfs": count_choice(items, "raw_preserved_hpfs"),
            "rpgcl_auto_choice_raw_features": count_choice(items, "raw_features"),
            "rpgcl_auto_oracle_test_accuracy_mean": mean([
                r["rpgcl_auto_oracle_test_accuracy"] for r in items
            ]),
            "rpgcl_auto_oracle_gap_mean": mean([
                float(r["rpgcl_auto_oracle_test_accuracy"]) - float(r["accuracy"])
                for r in items
                if r["rpgcl_auto_oracle_test_accuracy"] not in ("", None)
            ]),
        })
    write_csv(args.out or Path(args.runs_dir) / "runs_vs_baseline.csv", enriched)
    write_csv(args.aggregate_out or Path(args.runs_dir) / "aggregate_vs_baseline.csv", agg)
    print(json.dumps({"runs": len(enriched), "groups": len(agg)}, indent=2))


if __name__ == "__main__":
    main()
