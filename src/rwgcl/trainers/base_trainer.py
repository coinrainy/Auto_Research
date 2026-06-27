"""Base scaffold trainer.

This file intentionally does not train a model yet. It validates configuration,
creates the result layout, and writes stable placeholder artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rwgcl.data import DatasetSpec, dataset_summary
from rwgcl.encoders import encoder_from_config
from rwgcl.logging_utils import append_csv, ensure_dir, make_run_id, now_utc, write_csv, write_json, write_yaml
from rwgcl.losses import loss_summary


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    metrics_path: Path
    status: str


class BaseTrainer:
    trainer_name = "base"

    def __init__(
        self,
        method_config: dict[str, Any],
        dataset_spec: DatasetSpec,
        seed: int,
        results_dir: str | Path,
        device: str = "auto",
        split_index: int = 0,
    ) -> None:
        self.method_config = method_config
        self.dataset_spec = dataset_spec
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.device_name = device
        self.split_index = split_index
        self.method_name = str(method_config.get("method", {}).get("name", self.trainer_name))
        self.run_id = make_run_id(self.method_name, dataset_spec.name, seed, split_index=split_index)
        self.run_dir = self.results_dir / "raw" / self.run_id

    def run(self, mode: str = "scaffold") -> RunResult:
        if mode != "scaffold":
            raise NotImplementedError(
                f"{self.trainer_name} does not implement execute mode yet."
            )
        return self.run_scaffold()

    def run_scaffold(self) -> RunResult:
        ensure_dir(self.run_dir)
        config_snapshot = {
            "created_at": now_utc(),
            "run_id": self.run_id,
            "status": "scaffold_only",
            "method": self.method_config,
            "dataset": dataset_summary(self.dataset_spec),
            "seed": self.seed,
            "model_seed": self.seed,
            "split_index": self.split_index,
            "trainer": self.trainer_name,
            "encoder": encoder_from_config(self.method_config.get("encoder", {})).__dict__,
            "loss": loss_summary(self.method_config),
        }
        write_yaml(self.run_dir / "config.yaml", config_snapshot)
        write_json(self.run_dir / "run_metadata.json", config_snapshot)
        write_csv(
            self.run_dir / "train_log.csv",
            [
                {
                    "epoch": 0,
                    "split": "scaffold",
                    "loss": "",
                    "metric": "",
                    "status": "scaffold_only",
                    "notes": "No model training performed yet.",
                }
            ],
            ["epoch", "split", "loss", "metric", "status", "notes"],
        )
        metrics_path = self.results_dir / "metrics" / "main_results.csv"
        append_csv(
            metrics_path,
            {
                "run_id": self.run_id,
                "dataset": self.dataset_spec.name,
                "dataset_group": self.dataset_spec.group,
                "method": self.method_name,
                "trainer": self.trainer_name,
                "seed": self.seed,
                "model_seed": self.seed,
                "split_index": self.split_index,
                "metric": self.dataset_spec.metric,
                "value": "",
                "status": "scaffold_only",
                "notes": "Entry-point scaffold validated; real training not implemented.",
            },
            [
                "run_id",
                "dataset",
                "dataset_group",
                "method",
                "trainer",
                "seed",
                "model_seed",
                "split_index",
                "metric",
                "value",
                "status",
                "notes",
            ],
        )
        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            metrics_path=metrics_path,
            status="scaffold_only",
        )
