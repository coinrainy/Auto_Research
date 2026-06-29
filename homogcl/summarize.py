from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HomoGCL JSON result files.")
    parser.add_argument("--input-dir", default="results/smoke")
    parser.add_argument("--output-csv", default="results/smoke_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        rows.append(
            {
                "dataset": item["dataset"],
                "method": item["method"],
                "split_index": item["protocol"]["split_index"],
                "split_seed": item["protocol"].get("split_seed", item["protocol"]["split_index"]),
                "split_protocol": item["protocol"].get("split_protocol", ""),
                "model_seed": item["protocol"]["model_seed"],
                "eval_seed": item["protocol"]["eval_seed"],
                "probe": item["hparams"].get("probe", "unknown"),
                "prop_steps": item["hparams"].get("prop_steps", ""),
                "selected_prop_steps": item["metrics"].get("selected_prop_steps", ""),
                "selected_pca_rank": item["metrics"].get("selected_pca_rank", ""),
                "max_prop_steps": item["hparams"].get("max_prop_steps", ""),
                "autoprop_plateau_ratio": item["hparams"].get("autoprop_plateau_ratio", ""),
                "specprop_high_concentration": item["hparams"].get("specprop_high_concentration", ""),
                "specprop_mid_concentration": item["hparams"].get("specprop_mid_concentration", ""),
                "corespecprop_min_rank": item["hparams"].get("corespecprop_min_rank", ""),
                "corespecprop_max_rank": item["hparams"].get("corespecprop_max_rank", ""),
                "corespecprop_participation_divisor": item["hparams"].get(
                    "corespecprop_participation_divisor", ""
                ),
                "tierspecprop_wide_concentration": item["hparams"].get(
                    "tierspecprop_wide_concentration", ""
                ),
                "tierspecprop_narrow_rank": item["hparams"].get("tierspecprop_narrow_rank", ""),
                "tierspecprop_wide_rank": item["hparams"].get("tierspecprop_wide_rank", ""),
                "fusion_applied": item["metrics"].get("fusion_applied", ""),
                "fusion_core_dim": item["metrics"].get("fusion_core_dim", ""),
                "fusion_residual_dim": item["metrics"].get("fusion_residual_dim", ""),
                "edge_drop": item["hparams"].get("edge_drop", ""),
                "feat_drop": item["hparams"].get("feat_drop", ""),
                "epochs": item["hparams"].get("epochs", ""),
                "cca_lambd": item["hparams"].get("cca_lambd", ""),
                "bank_drop": item["hparams"].get("bank_drop", ""),
                "test_acc": item["metrics"]["test_acc"],
                "val_acc": item["metrics"]["val_acc"],
                "train_acc": item["metrics"]["train_acc"],
                "best_epoch": item["metrics"]["best_epoch"],
                "ridge_alpha": item["metrics"].get("ridge_alpha", ""),
                "logreg_C": item["metrics"].get("logreg_C", ""),
                "ssl_loss": item["metrics"].get("ssl_loss", 0.0),
                "file": str(path),
            }
        )
    if not rows:
        raise SystemExit(f"No JSON files found under {args.input_dir}")
    df = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    group_cols = [
        "dataset",
        "method",
        "probe",
        "split_protocol",
        "prop_steps",
        "selected_prop_steps",
        "selected_pca_rank",
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
        "cca_lambd",
        "bank_drop",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)["test_acc"]
        .agg(["count", "mean", "std", "min", "max"])
        .sort_index()
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote raw table: {args.output_csv}")


if __name__ == "__main__":
    main()
