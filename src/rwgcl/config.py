"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def dataset_config_path(dataset: str) -> Path:
    return PROJECT_ROOT / "configs" / "datasets" / f"{dataset.lower()}.yaml"


def load_dataset_config(dataset: str) -> dict[str, Any]:
    return load_yaml(dataset_config_path(dataset))


def load_method_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)
