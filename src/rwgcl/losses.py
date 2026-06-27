"""Loss placeholders for the scaffold."""

from __future__ import annotations

from typing import Any


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
