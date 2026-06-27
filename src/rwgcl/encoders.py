"""Encoder configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EncoderSpec:
    type: str
    hidden_dim: int
    out_dim: int
    num_layers: int


def encoder_from_config(config: dict[str, Any]) -> EncoderSpec:
    return EncoderSpec(
        type=str(config.get("type", "gcn")),
        hidden_dim=int(config.get("hidden_dim", 256)),
        out_dim=int(config.get("out_dim", 256)),
        num_layers=int(config.get("num_layers", 2)),
    )
