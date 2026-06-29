from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison between two methods.")
    parser.add_argument("--input-dirs", default="results/specprop_multisplit")
    parser.add_argument("--baseline", default="autopropcat")
    parser.add_argument("--candidate", default="specprop")
    parser.add_argument("--output-csv", default="results/paired_compare.csv")
    return parser.parse_args()


def load_rows(input_dirs: str) -> pd.DataFrame:
    rows = []
    for item in input_dirs.split(","):
        root = Path(item.strip())
        if not root:
            continue
        for path in sorted(root.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            protocol = payload["protocol"]
            metrics = payload["metrics"]
            rows.append(
                {
                    "dataset": payload["dataset"],
                    "method": payload["method"],
                    "split_protocol": protocol.get("split_protocol", ""),
                    "split_seed": protocol.get("split_seed", protocol.get("split_index", 0)),
                    "model_seed": protocol.get("model_seed", 0),
                    "eval_seed": protocol.get("eval_seed", 0),
                    "test_acc": metrics["test_acc"],
                    "val_acc": metrics["val_acc"],
                    "selected_prop_steps": metrics.get("selected_prop_steps", ""),
                    "selected_pca_rank": metrics.get("selected_pca_rank", ""),
                    "file": str(path),
                }
            )
    if not rows:
        raise SystemExit(f"No JSON files found under {input_dirs}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = load_rows(args.input_dirs)
    keys = ["dataset", "split_seed", "model_seed", "eval_seed"]
    left = df[df["method"] == args.baseline].copy()
    right = df[df["method"] == args.candidate].copy()
    if left.empty or right.empty:
        raise SystemExit(f"Need both methods: baseline={args.baseline}, candidate={args.candidate}")
    paired = left.merge(right, on=keys, suffixes=("_baseline", "_candidate"), how="inner")
    paired["delta_test_acc"] = paired["test_acc_candidate"] - paired["test_acc_baseline"]
    paired["delta_val_acc"] = paired["val_acc_candidate"] - paired["val_acc_baseline"]
    paired["candidate_wins"] = paired["delta_test_acc"] > 0
    paired["candidate_losses"] = paired["delta_test_acc"] < 0

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_csv, index=False)
    summary = paired.groupby("dataset").agg(
        pairs=("delta_test_acc", "count"),
        baseline_mean=("test_acc_baseline", "mean"),
        candidate_mean=("test_acc_candidate", "mean"),
        delta_mean=("delta_test_acc", "mean"),
        delta_std=("delta_test_acc", "std"),
        wins=("candidate_wins", "sum"),
        losses=("candidate_losses", "sum"),
    )
    print("Paired rows:")
    print(
        paired[
            [
                "dataset",
                "split_seed",
                "test_acc_baseline",
                "test_acc_candidate",
                "delta_test_acc",
                "selected_pca_rank_candidate",
            ]
        ].to_string(index=False)
    )
    print("\nPaired summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote paired table: {args.output_csv}")


if __name__ == "__main__":
    main()
