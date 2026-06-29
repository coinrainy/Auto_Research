from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-gated candidate selector.")
    parser.add_argument("--input-dir", default="results/smoke")
    parser.add_argument("--output-csv", default="results/selected_summary.csv")
    parser.add_argument(
        "--candidates",
        default="autopropcat,propcat,gracecat,ccacat,grace,prop",
        help="Comma-separated method names eligible for validation selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = {item.strip() for item in args.candidates.split(",") if item.strip()}
    rows = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if item["method"] not in allowed:
            continue
        hp = item["hparams"]
        rows.append(
            {
                "dataset": item["dataset"],
                "method": item["method"],
                "split_index": item["protocol"]["split_index"],
                "model_seed": item["protocol"]["model_seed"],
                "eval_seed": item["protocol"]["eval_seed"],
                "probe": hp.get("probe", ""),
                "prop_steps": hp.get("prop_steps", ""),
                "edge_drop": hp.get("edge_drop", ""),
                "feat_drop": hp.get("feat_drop", ""),
                "epochs": hp.get("epochs", ""),
                "val_acc": item["metrics"]["val_acc"],
                "test_acc": item["metrics"]["test_acc"],
                "file": str(path),
            }
        )
    if not rows:
        raise SystemExit(f"No eligible candidate JSON files found under {args.input_dir}")

    df = pd.DataFrame(rows)
    selected = (
        df.sort_values(
            ["dataset", "split_index", "model_seed", "eval_seed", "val_acc", "method"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(["dataset", "split_index", "model_seed", "eval_seed"], as_index=False)
        .head(1)
        .sort_values(["dataset", "model_seed", "eval_seed"])
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    summary = selected.groupby("dataset")["test_acc"].agg(["count", "mean", "std", "min", "max"])
    print("Selected candidates:")
    print(
        selected[
            [
                "dataset",
                "model_seed",
                "method",
                "prop_steps",
                "edge_drop",
                "feat_drop",
                "val_acc",
                "test_acc",
            ]
        ].to_string(index=False)
    )
    print("\nValidation-gated summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote selected table: {args.output_csv}")


if __name__ == "__main__":
    main()
