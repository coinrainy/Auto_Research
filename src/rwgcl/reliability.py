"""Reliability score utilities for the scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilitySpec:
    embedding_stability_weight: float
    projection_distribution_consistency_weight: float
    prediction_temperature: float
    prediction_confidence_power: float
    negative_enabled: bool
    samples_per_anchor: int


def reliability_from_config(config: dict) -> ReliabilitySpec:
    reliability = config.get("reliability", {})
    positive = reliability.get("positive", {})
    negative = reliability.get("negative", {})
    projection_weight = positive.get(
        "projection_distribution_consistency_weight",
        positive.get("prediction_consistency_weight", 0.5),
    )
    return ReliabilitySpec(
        embedding_stability_weight=float(positive.get("embedding_stability_weight", 0.5)),
        projection_distribution_consistency_weight=float(projection_weight),
        prediction_temperature=float(positive.get("prediction_temperature", 0.2)),
        prediction_confidence_power=float(positive.get("prediction_confidence_power", 0.25)),
        negative_enabled=bool(negative.get("enabled", False)),
        samples_per_anchor=int(negative.get("samples_per_anchor", 256)),
    )


def scaffold_reliability_summary(config: dict) -> dict:
    spec = reliability_from_config(config)
    return {
        "positive_signals": ["embedding_stability", "projection_distribution_consistency"],
        "embedding_stability_weight": spec.embedding_stability_weight,
        "projection_distribution_consistency_weight": spec.projection_distribution_consistency_weight,
        "legacy_prediction_consistency_name": "projection_distribution_consistency",
        "prediction_temperature": spec.prediction_temperature,
        "prediction_confidence_power": spec.prediction_confidence_power,
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


def distribution_confidence(prob: torch.Tensor) -> torch.Tensor:
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=1)
    max_entropy = torch.log(torch.tensor(prob.size(1), device=prob.device, dtype=prob.dtype))
    return (1.0 - entropy / max_entropy.clamp_min(1e-12)).clamp(0.0, 1.0)


def cross_view_projection_distribution_consistency(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float,
    confidence_power: float,
) -> torch.Tensor:
    p1 = F.softmax(z1 / max(temperature, 1e-6), dim=1)
    p2 = F.softmax(z2 / max(temperature, 1e-6), dim=1)
    l1_distance = (p1 - p2).abs().sum(dim=1)
    agreement = (1.0 - 0.5 * l1_distance).clamp(0.0, 1.0)
    confidence = 0.5 * (distribution_confidence(p1) + distribution_confidence(p2))
    return (agreement * confidence.clamp_min(1e-6).pow(confidence_power)).clamp(0.0, 1.0)


cross_view_prediction_consistency = cross_view_projection_distribution_consistency


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
    consistency = cross_view_projection_distribution_consistency(
        z1.detach(),
        z2.detach(),
        temperature=spec.prediction_temperature,
        confidence_power=spec.prediction_confidence_power,
    )
    score = (
        spec.embedding_stability_weight * stability
        + spec.projection_distribution_consistency_weight * consistency
    )
    denom = spec.embedding_stability_weight + spec.projection_distribution_consistency_weight
    score = (score / max(denom, 1e-6)).clamp(0.0, 1.0)
    if shuffled:
        score = score[torch.randperm(score.numel(), device=score.device)]
    summary = {
        "reliability_mean": float(score.mean().item()),
        "reliability_std": float(score.std(unbiased=False).item()),
        "reliability_min": float(score.min().item()),
        "reliability_max": float(score.max().item()),
        "embedding_stability_mean": float(stability.mean().item()),
        "embedding_stability_std": float(stability.std(unbiased=False).item()),
        "embedding_stability_min": float(stability.min().item()),
        "embedding_stability_max": float(stability.max().item()),
        "projection_distribution_consistency_mean": float(consistency.mean().item()),
        "projection_distribution_consistency_std": float(consistency.std(unbiased=False).item()),
        "projection_distribution_consistency_min": float(consistency.min().item()),
        "projection_distribution_consistency_max": float(consistency.max().item()),
        "prediction_consistency_mean": float(consistency.mean().item()),
        "prediction_consistency_std": float(consistency.std(unbiased=False).item()),
        "prediction_consistency_min": float(consistency.min().item()),
        "prediction_consistency_max": float(consistency.max().item()),
    }
    components = {
        "embedding_stability": stability.detach(),
        "projection_distribution_consistency": consistency.detach(),
        "prediction_consistency": consistency.detach(),
    }
    return score, summary, components
