import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHODS = ["grace", "hpfs_gcl", "rpgcl_hpfs", "cg_hpfs"]
FIXED_BRANCHES = ["hpfs_gcl", "rpgcl_hpfs"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-runs-dir", default="runs/rpgcl_auto_selector_controls_splits0-9_e50")
    parser.add_argument("--gate-runs-dir", default="runs/complement_gate_splits0-9_e50")
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    return parser.parse_args()


def read_run(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    diagnostics = payload.get("diagnostics", {})
    return {
        "run_dir": str(path.parent),
        "dataset": payload["dataset"],
        "method": payload["method"],
        "seed": int(payload.get("model_seed", payload["seed"])),
        "split_index": int(payload["split_index"]),
        "split_seed": int(payload.get("split_seed", int(payload["config"].get("split_base_seed", 2026)) + int(payload["split_index"]))),
        "accuracy": float(payload["metrics"]["accuracy"]),
        "val_accuracy": float(payload["metrics"]["val_accuracy"]),
        "gate_alpha": diagnostics.get("gate_alpha", ""),
        "gate_soft_alpha": diagnostics.get("gate_soft_alpha", ""),
        "gate_signal_value": diagnostics.get("gate_signal_value", ""),
        "gate_threshold": diagnostics.get("gate_threshold", ""),
        "raw_preserved": diagnostics.get("raw_preserved", ""),
    }


def load_runs(*dirs):
    rows = []
    for runs_dir in dirs:
        for path in sorted(Path(runs_dir).glob("*/run.json")):
            row = read_run(path)
            if row["method"] in METHODS:
                rows.append(row)
    return rows


def mean(values):
    values = [float(value) for value in values if value not in ("", None)]
    return "" if not values else sum(values) / len(values)


def count_delta(rows, field, sign):
    values = [float(row[field]) for row in rows if row[field] not in ("", None)]
    if sign == "positive":
        return sum(1 for value in values if value > 0.0)
    if sign == "negative":
        return sum(1 for value in values if value < 0.0)
    return sum(1 for value in values if value == 0.0)


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


def main():
    args = parse_args()
    rows = load_runs(args.control_runs_dir, args.gate_runs_dir)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["split_index"], row["seed"])][row["method"]] = row

    pairs = []
    for dataset, split_index, seed in sorted(grouped):
        method_rows = grouped[(dataset, split_index, seed)]
        if "cg_hpfs" not in method_rows:
            continue
        item = {
            "dataset": dataset,
            "split_index": split_index,
            "seed": seed,
            "split_seed": method_rows["cg_hpfs"]["split_seed"],
        }
        for method in METHODS:
            row = method_rows.get(method)
            item[f"{method}_accuracy"] = "" if row is None else row["accuracy"]
        fixed = [
            (method, float(item[f"{method}_accuracy"]))
            for method in FIXED_BRANCHES
            if item[f"{method}_accuracy"] not in ("", None)
        ]
        best_method, best_acc = max(fixed, key=lambda pair: pair[1])
        cg_acc = float(item["cg_hpfs_accuracy"])
        item["best_fixed_method"] = best_method
        item["best_fixed_accuracy"] = best_acc
        item["cg_minus_grace_light"] = "" if item["grace_accuracy"] == "" else cg_acc - float(item["grace_accuracy"])
        item["cg_minus_hpfs_gcl"] = cg_acc - float(item["hpfs_gcl_accuracy"])
        item["cg_minus_rpgcl_hpfs"] = cg_acc - float(item["rpgcl_hpfs_accuracy"])
        item["cg_minus_best_fixed"] = cg_acc - best_acc
        cg = method_rows["cg_hpfs"]
        item["gate_alpha"] = cg["gate_alpha"]
        item["gate_soft_alpha"] = cg["gate_soft_alpha"]
        item["gate_signal_value"] = cg["gate_signal_value"]
        item["gate_threshold"] = cg["gate_threshold"]
        item["gate_raw_preserved"] = cg["raw_preserved"]
        pairs.append(item)

    aggregate = []
    by_dataset = defaultdict(list)
    for row in pairs:
        by_dataset[row["dataset"]].append(row)
    for dataset, items in sorted(by_dataset.items()):
        aggregate.append({
            "dataset": dataset,
            "num_splits": len(items),
            "grace_light_accuracy_mean": mean(row["grace_accuracy"] for row in items),
            "hpfs_gcl_accuracy_mean": mean(row["hpfs_gcl_accuracy"] for row in items),
            "raw_preserved_hpfs_accuracy_mean": mean(row["rpgcl_hpfs_accuracy"] for row in items),
            "cg_hpfs_accuracy_mean": mean(row["cg_hpfs_accuracy"] for row in items),
            "best_fixed_accuracy_mean": mean(row["best_fixed_accuracy"] for row in items),
            "cg_minus_grace_light_mean": mean(row["cg_minus_grace_light"] for row in items),
            "cg_minus_hpfs_gcl_mean": mean(row["cg_minus_hpfs_gcl"] for row in items),
            "cg_minus_raw_preserved_hpfs_mean": mean(row["cg_minus_rpgcl_hpfs"] for row in items),
            "cg_minus_best_fixed_mean": mean(row["cg_minus_best_fixed"] for row in items),
            "cg_positive_vs_best_fixed": count_delta(items, "cg_minus_best_fixed", "positive"),
            "cg_negative_vs_best_fixed": count_delta(items, "cg_minus_best_fixed", "negative"),
            "cg_zero_vs_best_fixed": count_delta(items, "cg_minus_best_fixed", "zero"),
            "gate_alpha_mean": mean(row["gate_alpha"] for row in items),
            "gate_signal_value_mean": mean(row["gate_signal_value"] for row in items),
            "gate_raw_preserved_count": sum(str(row["gate_raw_preserved"]) == "True" for row in items),
        })

    out = args.out or Path(args.gate_runs_dir) / "complement_gate_pairs.csv"
    aggregate_out = args.aggregate_out or Path(args.gate_runs_dir) / "complement_gate_aggregate.csv"
    write_csv(out, pairs)
    write_csv(aggregate_out, aggregate)
    print(json.dumps({"pairs": len(pairs), "datasets": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
