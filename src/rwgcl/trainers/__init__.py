"""Trainer factory."""

from __future__ import annotations

from .base_trainer import BaseTrainer
from .bgrl_trainer import BGRLTrainer
from .grace_trainer import GRACETrainer
from .rw_gcl_trainer import RWGCLTrainer


def build_trainer(trainer_name: str) -> type[BaseTrainer]:
    trainers: dict[str, type[BaseTrainer]] = {
        "base": BaseTrainer,
        "grace": GRACETrainer,
        "bgrl": BGRLTrainer,
        "rw_gcl": RWGCLTrainer,
    }
    try:
        return trainers[trainer_name]
    except KeyError as exc:
        known = ", ".join(sorted(trainers))
        raise ValueError(f"Unknown trainer '{trainer_name}'. Known trainers: {known}") from exc
