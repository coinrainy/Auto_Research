"""Reliability score utilities for the scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilitySpec:
    embedding_stability_weight: float
    prediction_consistency_weight: float
    negative_enabled: bool
    samples_per_anchor: int


def reliability_from_config(config: dict) -> ReliabilitySpec:
    reliability = config.get("reliability", {})
    positive = reliability.get("positive", {})
    negative = reliability.get("negative", {})
    return ReliabilitySpec(
        embedding_stability_weight=float(positive.get("embedding_stability_weight", 0.5)),
        prediction_consistency_weight=float(positive.get("prediction_consistency_weight", 0.5)),
        negative_enabled=bool(negative.get("enabled", False)),
        samples_per_anchor=int(negative.get("samples_per_anchor", 256)),
    )


def scaffold_reliability_summary(config: dict) -> dict:
    spec = reliability_from_config(config)
    return {
        "positive_signals": ["embedding_stability", "prediction_consistency"],
        "embedding_stability_weight": spec.embedding_stability_weight,
        "prediction_consistency_weight": spec.prediction_consistency_weight,
        "negative_weighting_enabled": spec.negative_enabled,
        "samples_per_anchor": spec.samples_per_anchor,
        "implemented": True,
        "implementation_scope": "positive_pair_weighting_only",
    }


def teacher_student_embedding_stability(
    student_h1: torch.Tensor,
    student_h2: torch.Tensor,
    teacher_h1: torch.Tensor,
    teacher_h2: torch.Tensor,
) -> torch.Tensor:
    sim1 = F.cosine_similarity(student_h1, teacher_h1, dim=1)
    sim2 = F.cosine_similarity(student_h2, teacher_h2, dim=1)
    return ((sim1 + sim2) * 0.25 + 0.5).clamp(0.0, 1.0)


def cross_view_prediction_consistency(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    p1 = F.softmax(z1, dim=1)
    p2 = F.softmax(z2, dim=1)
    l1_distance = (p1 - p2).abs().sum(dim=1)
    return (1.0 - 0.5 * l1_distance).clamp(0.0, 1.0)


def positive_pair_reliability(
    student_h1: torch.Tensor,
    student_h2: torch.Tensor,
    teacher_h1: torch.Tensor,
    teacher_h2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    spec: ReliabilitySpec,
    shuffled: bool = False,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    stability = teacher_student_embedding_stability(
        student_h1.detach(),
        student_h2.detach(),
        teacher_h1.detach(),
        teacher_h2.detach(),
    )
    consistency = cross_view_prediction_consistency(z1.detach(), z2.detach())
    score = (
        spec.embedding_stability_weight * stability
        + spec.prediction_consistency_weight * consistency
    )
    denom = spec.embedding_stability_weight + spec.prediction_consistency_weight
    score = (score / max(denom, 1e-6)).clamp(0.0, 1.0)
    if shuffled:
        score = score[torch.randperm(score.numel(), device=score.device)]
    summary = {
        "reliability_mean": float(score.mean().item()),
        "reliability_min": float(score.min().item()),
        "reliability_max": float(score.max().item()),
        "embedding_stability_mean": float(stability.mean().item()),
        "prediction_consistency_mean": float(consistency.mean().item()),
    }
    components = {
        "embedding_stability": stability.detach(),
        "prediction_consistency": consistency.detach(),
    }
    return score, summary, components
