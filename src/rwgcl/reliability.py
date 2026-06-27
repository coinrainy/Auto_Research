"""Reliability score utilities for the scaffold."""

from __future__ import annotations

from dataclasses import dataclass


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
        "implemented": False,
    }
