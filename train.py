#!/usr/bin/env python3
"""Training entry point for RW-GCL experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rwgcl.config import load_method_config
from rwgcl.data import get_dataset_spec
from rwgcl.seed import set_seed
from rwgcl.trainers import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RW-GCL training scaffold.")
    parser.add_argument("--config", required=True, help="Path to method config YAML.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cora or Texas.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--results-dir", default="results", help="Output results directory.")
    parser.add_argument(
        "--mode",
        choices=["scaffold", "execute"],
        default="scaffold",
        help="scaffold validates the experiment layout; execute is reserved for real training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    method_config = load_method_config(args.config)
    dataset_spec = get_dataset_spec(args.dataset)
    trainer_name = str(method_config.get("method", {}).get("trainer", "base"))
    trainer_cls = build_trainer(trainer_name)
    trainer = trainer_cls(
        method_config=method_config,
        dataset_spec=dataset_spec,
        seed=args.seed,
        results_dir=args.results_dir,
    )
    result = trainer.run(mode="scaffold" if args.mode == "scaffold" else "execute")
    print(f"status={result.status}")
    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"metrics_path={result.metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
