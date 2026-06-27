"""Graph augmentation placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentationPlan:
    edge_drop: float
    feature_mask: float
    stage: str


def conservative_plan(edge_drop: float = 0.1, feature_mask: float = 0.1) -> AugmentationPlan:
    return AugmentationPlan(edge_drop=edge_drop, feature_mask=feature_mask, stage="warmup")


def standard_plan(edge_drop: float = 0.2, feature_mask: float = 0.2) -> AugmentationPlan:
    return AugmentationPlan(edge_drop=edge_drop, feature_mask=feature_mask, stage="stage2")
