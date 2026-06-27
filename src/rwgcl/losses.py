"""Loss placeholders for the scaffold."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def loss_summary(method_config: dict[str, Any]) -> dict[str, Any]:
    method_name = method_config.get("method", {}).get("name", "unknown")
    if method_name == "rw_gcl_two_stage":
        return {
            "name": "reliability_weighted_infonce",
            "implemented": False,
            "status": "scaffold_only",
        }
    if method_name == "bgrl":
        return {"name": "bootstrap_prediction_loss", "implemented": False, "status": "scaffold_only"}
    return {"name": "infonce_or_method_default", "implemented": False, "status": "scaffold_only"}


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits_12 = z1 @ z2.t() / temperature
    logits_21 = z2 @ z1.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits_12, labels) + F.cross_entropy(logits_21, labels))
