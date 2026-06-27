"""Two-stage reliability-weighted GCL scaffold trainer."""

from __future__ import annotations

from rwgcl.logging_utils import write_json
from rwgcl.reliability import scaffold_reliability_summary

from .base_trainer import BaseTrainer, RunResult


class RWGCLTrainer(BaseTrainer):
    trainer_name = "rw_gcl"

    def run_scaffold(self) -> RunResult:
        result = super().run_scaffold()
        write_json(
            self.run_dir / "reliability_plan.json",
            {
                "run_id": self.run_id,
                "status": "scaffold_only",
                "reliability": scaffold_reliability_summary(self.method_config),
                "next_step": "Implement warm-up embeddings, EMA teacher, and reliability-weighted InfoNCE.",
            },
        )
        return result
