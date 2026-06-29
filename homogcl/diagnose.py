from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose SpecProp spectral gates and paired deltas.")
    parser.add_argument("--input-dirs", required=True)
    parser.add_argument("--baseline", default="autopropcat")
    parser.add_argument("--candidate", default="specprop")
    parser.add_argument("--output-csv", default="results/specprop_diagnostics.csv")
    return parser.parse_args()


def load_rows(input_dirs: str) -> pd.DataFrame:
    rows = []
    for raw in input_dirs.split(","):
        root = Path(raw.strip())
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
                    "split_seed": protocol.get("split_seed", protocol.get("split_index", 0)),
                    "model_seed": protocol.get("model_seed", 0),
                    "eval_seed": protocol.get("eval_seed", 0),
                    "test_acc": metrics["test_acc"],
                    "val_acc": metrics["val_acc"],
                    "selected_prop_steps": metrics.get("selected_prop_steps", ""),
                    "selected_pca_rank": metrics.get("selected_pca_rank", ""),
                    "spectral_top10_energy": metrics.get("spectral_top10_energy", ""),
                    "spectral_participation_rank": metrics.get("spectral_participation_rank", ""),
                    "spectral_energy_80_rank": metrics.get("spectral_energy_80_rank", ""),
                    "spectral_energy_95_rank": metrics.get("spectral_energy_95_rank", ""),
                    "edge_homophily": protocol.get("edge_homophily_diagnostic_uses_labels", ""),
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
    baseline = df[df["method"] == args.baseline].copy()
    candidate = df[df["method"] == args.candidate].copy()
    paired = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), how="inner")
    if paired.empty:
        raise SystemExit("No paired baseline/candidate rows found.")

    paired["delta_test_acc"] = paired["test_acc_candidate"] - paired["test_acc_baseline"]
    paired["gate_action"] = paired["selected_pca_rank_candidate"].apply(
        lambda value: "compress" if float(value or 0) > 0 else "fallback"
    )
    out_cols = [
        "dataset",
        "split_seed",
        "test_acc_baseline",
        "test_acc_candidate",
        "delta_test_acc",
        "gate_action",
        "selected_prop_steps_candidate",
        "selected_pca_rank_candidate",
        "spectral_top10_energy_candidate",
        "spectral_participation_rank_candidate",
        "spectral_energy_80_rank_candidate",
        "spectral_energy_95_rank_candidate",
        "edge_homophily_candidate",
    ]
    diagnostics = paired[out_cols].sort_values(["dataset", "split_seed"])
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(args.output_csv, index=False)

    summary = diagnostics.groupby(["dataset", "gate_action"]).agg(
        count=("delta_test_acc", "count"),
        delta_mean=("delta_test_acc", "mean"),
        top10_mean=("spectral_top10_energy_candidate", "mean"),
        participation_mean=("spectral_participation_rank_candidate", "mean"),
    )
    print("SpecProp diagnostics:")
    print(diagnostics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nGate summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote diagnostics: {args.output_csv}")


if __name__ == "__main__":
    main()
