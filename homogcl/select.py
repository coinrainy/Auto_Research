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
        default="tierccacat,tierspecprop,corespecprop,specprop,autopropcat,propcat,gracecat,ccacat,grace,prop",
        help="Comma-separated method names eligible for validation selection.",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "config"],
        default="config",
        help="run selects per seed; config selects one validation-best configuration per dataset.",
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
                "split_seed": item["protocol"].get("split_seed", item["protocol"]["split_index"]),
                "split_protocol": item["protocol"].get("split_protocol", ""),
                "model_seed": item["protocol"]["model_seed"],
                "eval_seed": item["protocol"]["eval_seed"],
                "probe": hp.get("probe", ""),
                "prop_steps": hp.get("prop_steps", ""),
                "selected_prop_steps": item["metrics"].get("selected_prop_steps", ""),
                "selected_pca_rank": item["metrics"].get("selected_pca_rank", ""),
                "max_prop_steps": hp.get("max_prop_steps", ""),
                "autoprop_plateau_ratio": hp.get("autoprop_plateau_ratio", ""),
                "specprop_high_concentration": hp.get("specprop_high_concentration", ""),
                "specprop_mid_concentration": hp.get("specprop_mid_concentration", ""),
                "corespecprop_min_rank": hp.get("corespecprop_min_rank", ""),
                "corespecprop_max_rank": hp.get("corespecprop_max_rank", ""),
                "corespecprop_participation_divisor": hp.get("corespecprop_participation_divisor", ""),
                "tierspecprop_wide_concentration": hp.get("tierspecprop_wide_concentration", ""),
                "tierspecprop_narrow_rank": hp.get("tierspecprop_narrow_rank", ""),
                "tierspecprop_wide_rank": hp.get("tierspecprop_wide_rank", ""),
                "fusion_applied": item["metrics"].get("fusion_applied", ""),
                "fusion_core_dim": item["metrics"].get("fusion_core_dim", ""),
                "fusion_residual_dim": item["metrics"].get("fusion_residual_dim", ""),
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
    config_cols = [
        "dataset",
        "method",
        "probe",
        "prop_steps",
        "max_prop_steps",
        "autoprop_plateau_ratio",
        "specprop_high_concentration",
        "specprop_mid_concentration",
        "corespecprop_min_rank",
        "corespecprop_max_rank",
        "corespecprop_participation_divisor",
        "tierspecprop_wide_concentration",
        "tierspecprop_narrow_rank",
        "tierspecprop_wide_rank",
        "fusion_applied",
        "fusion_core_dim",
        "fusion_residual_dim",
        "edge_drop",
        "feat_drop",
        "epochs",
    ]
    if args.mode == "run":
        selected = (
            df.sort_values(
                ["dataset", "split_index", "model_seed", "eval_seed", "val_acc", "method"],
                ascending=[True, True, True, True, False, True],
            )
            .groupby(["dataset", "split_protocol", "split_seed", "model_seed", "eval_seed"], as_index=False)
            .head(1)
            .sort_values(["dataset", "split_seed", "model_seed", "eval_seed"])
        )
    else:
        ranked = (
            df.groupby(config_cols, dropna=False)
            .agg(val_mean=("val_acc", "mean"), val_count=("val_acc", "count"))
            .reset_index()
            .sort_values(["dataset", "val_mean", "val_count", "method"], ascending=[True, False, False, True])
            .groupby("dataset", as_index=False)
            .head(1)
        )
        selected = df.merge(ranked[config_cols], on=config_cols, how="inner")
        selected = selected.sort_values(["dataset", "split_seed", "model_seed", "eval_seed"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    summary = selected.groupby("dataset")["test_acc"].agg(["count", "mean", "std", "min", "max"])
    print(f"Selected candidates (mode={args.mode}):")
    print(
        selected[
            [
                "dataset",
                "model_seed",
                "split_seed",
                "method",
                "prop_steps",
                "selected_prop_steps",
                "selected_pca_rank",
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
