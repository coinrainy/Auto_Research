"""Dataset specification helpers.

The real PyG dataset objects will be wired in the next implementation step.
For now, this module validates metadata and keeps CLI runs reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import load_dataset_config


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    group: str
    loader: str
    metric: str
    root: str
    split: str
    notes: str = ""


def get_dataset_spec(dataset: str) -> DatasetSpec:
    cfg = load_dataset_config(dataset)
    return DatasetSpec(
        name=str(cfg["name"]),
        group=str(cfg["group"]),
        loader=str(cfg["loader"]),
        metric=str(cfg["metric"]),
        root=str(cfg["root"]),
        split=str(cfg.get("split", "default")),
        notes=str(cfg.get("notes", "")),
    )


def dataset_summary(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "group": spec.group,
        "loader": spec.loader,
        "metric": spec.metric,
        "root": spec.root,
        "split": spec.split,
        "notes": spec.notes,
    }


def load_pyg_dataset(_: DatasetSpec) -> Any:
    raise NotImplementedError(
        "PyG dataset loading is not implemented in the scaffold. "
        "Next step: wire Planetoid/WebKB/Actor/WikipediaNetwork loaders."
    )
