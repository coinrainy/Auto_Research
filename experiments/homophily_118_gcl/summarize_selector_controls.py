import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METHODS = ["raw_features", "grace", "hpfs_gcl", "rpgcl_hpfs", "rpgcl_auto"]
FIXED_CONTROLS = ["raw_features", "hpfs_gcl", "rpgcl_hpfs"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-vs", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    return parser.parse_args()


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def count_delta(rows, field, sign):
    values = [float(row[field]) for row in rows if row[field] not in ("", None)]
    if sign == "positive":
        return sum(1 for value in values if value > 0.0)
    if sign == "negative":
        return sum(1 for value in values if value < 0.0)
    return sum(1 for value in values if value == 0.0)


def method_acc(method_rows, method):
    row = method_rows.get(method)
    return "" if row is None else float(row["accuracy"])


def main():
    args = parse_args()
    rows = read_rows(args.runs_vs)
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], int(row["split_index"]), int(row["seed"]))
        grouped[key][row["method"]] = row

    pair_rows = []
    for dataset, split_index, seed in sorted(grouped):
        method_rows = grouped[(dataset, split_index, seed)]
        if "rpgcl_auto" not in method_rows:
            continue
        item = {
            "dataset": dataset,
            "split_index": split_index,
            "seed": seed,
        }
        for method in METHODS:
            item[f"{method}_accuracy"] = method_acc(method_rows, method)
        fixed = [
            (method, method_acc(method_rows, method))
            for method in FIXED_CONTROLS
            if method_acc(method_rows, method) != ""
        ]
        if fixed:
            best_method, best_acc = max(fixed, key=lambda pair: pair[1])
            item["best_fixed_method"] = best_method
            item["best_fixed_accuracy"] = best_acc
        else:
            item["best_fixed_method"] = ""
            item["best_fixed_accuracy"] = ""

        auto = item["rpgcl_auto_accuracy"]
        for method in ["grace", *FIXED_CONTROLS]:
            base = item[f"{method}_accuracy"]
            item[f"auto_minus_{method}"] = "" if base == "" else float(auto) - float(base)
        item["auto_minus_best_fixed"] = (
            "" if item["best_fixed_accuracy"] == "" else float(auto) - float(item["best_fixed_accuracy"])
        )
        auto_row = method_rows["rpgcl_auto"]
        item["rpgcl_auto_choice"] = auto_row.get("rpgcl_auto_choice", "")
        item["rpgcl_auto_oracle_test_accuracy"] = auto_row.get("rpgcl_auto_oracle_test_accuracy", "")
        item["rpgcl_auto_oracle_gap"] = (
            ""
            if item["rpgcl_auto_oracle_test_accuracy"] in ("", None)
            else float(item["rpgcl_auto_oracle_test_accuracy"]) - float(auto)
        )
        pair_rows.append(item)

    aggregate = []
    by_dataset = defaultdict(list)
    for row in pair_rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, items in sorted(by_dataset.items()):
        aggregate.append({
            "dataset": dataset,
            "num_pairs": len(items),
            "raw_features_accuracy_mean": mean([row["raw_features_accuracy"] for row in items]),
            "grace_light_accuracy_mean": mean([row["grace_accuracy"] for row in items]),
            "hpfs_gcl_accuracy_mean": mean([row["hpfs_gcl_accuracy"] for row in items]),
            "rpgcl_hpfs_accuracy_mean": mean([row["rpgcl_hpfs_accuracy"] for row in items]),
            "rpgcl_auto_accuracy_mean": mean([row["rpgcl_auto_accuracy"] for row in items]),
            "best_fixed_accuracy_mean": mean([row["best_fixed_accuracy"] for row in items]),
            "auto_minus_grace_light_mean": mean([row["auto_minus_grace"] for row in items]),
            "auto_minus_raw_features_mean": mean([row["auto_minus_raw_features"] for row in items]),
            "auto_minus_hpfs_gcl_mean": mean([row["auto_minus_hpfs_gcl"] for row in items]),
            "auto_minus_rpgcl_hpfs_mean": mean([row["auto_minus_rpgcl_hpfs"] for row in items]),
            "auto_minus_best_fixed_mean": mean([row["auto_minus_best_fixed"] for row in items]),
            "auto_positive_vs_best_fixed": count_delta(items, "auto_minus_best_fixed", "positive"),
            "auto_negative_vs_best_fixed": count_delta(items, "auto_minus_best_fixed", "negative"),
            "auto_zero_vs_best_fixed": count_delta(items, "auto_minus_best_fixed", "zero"),
            "choice_hpfs": sum(1 for row in items if row["rpgcl_auto_choice"] == "hpfs"),
            "choice_raw_preserved_hpfs": sum(
                1 for row in items if row["rpgcl_auto_choice"] == "raw_preserved_hpfs"
            ),
            "choice_raw_features": sum(1 for row in items if row["rpgcl_auto_choice"] == "raw_features"),
        })

    base = Path(args.runs_vs).parent
    write_csv(args.out or base / "selector_control_pairs.csv", pair_rows)
    write_csv(args.aggregate_out or base / "selector_control_aggregate.csv", aggregate)
    print(json.dumps({
        "pairs": len(pair_rows),
        "datasets": len(aggregate),
    }, indent=2))


if __name__ == "__main__":
    main()
