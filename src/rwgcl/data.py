"""PyG dataset loading and graph statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch
from torch import Tensor
from torch_geometric import transforms as T
from torch_geometric.datasets import Actor, HeterophilousGraphDataset, Planetoid, WebKB, WikipediaNetwork
from torch_geometric.utils import is_undirected

from .config import PROJECT_ROOT, load_dataset_config


PLANETOID_RAW_NAMES = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
PLANETOID_RAW_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


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


def load_pyg_dataset(spec: DatasetSpec) -> Any:
    root = PROJECT_ROOT / spec.root
    transform = T.NormalizeFeatures()
    loader = spec.loader.lower()

    if loader == "planetoid":
        prefetch_planetoid_raw(root=root, name=spec.name)
        return Planetoid(root=str(root), name=spec.name, split=spec.split, transform=transform)
    if loader == "webkb":
        return WebKB(root=str(root), name=spec.name, transform=transform)
    if loader == "actor":
        return Actor(root=str(root), transform=transform)
    if loader == "wikipedianetwork":
        return WikipediaNetwork(
            root=str(root),
            name=spec.name.lower(),
            geom_gcn_preprocess=True,
            transform=transform,
        )
    if loader == "heterophilousgraphdataset":
        return HeterophilousGraphDataset(root=str(root), name=spec.name, transform=transform)

    raise ValueError(f"Unsupported PyG dataset loader: {spec.loader}")


def download_file(url: str, target: Path, timeout: int = 120) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Auto-Research-RWGCL/0.1"})
    with urlopen(request, timeout=timeout) as response:
        target.write_bytes(response.read())


def prefetch_planetoid_raw(root: Path, name: str) -> None:
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = name.lower()
    missing = [
        raw_name
        for raw_name in PLANETOID_RAW_NAMES
        if not (raw_dir / f"ind.{prefix}.{raw_name}").exists()
    ]
    for raw_name in missing:
        filename = f"ind.{prefix}.{raw_name}"
        download_file(f"{PLANETOID_RAW_URL}/{filename}", raw_dir / filename)


def _mask_count(mask: Tensor | None, split_index: int = 0) -> int:
    if mask is None:
        return 0
    selected = select_split_mask(mask, split_index)
    return int(selected.sum().item())


def select_split_mask(mask: Tensor, split_index: int = 0) -> Tensor:
    if mask.dim() == 1:
        return mask.bool()
    if mask.dim() == 2:
        if split_index < 0 or split_index >= mask.size(1):
            raise IndexError(f"split_index={split_index} is out of range for mask with {mask.size(1)} splits")
        return mask[:, split_index].bool()
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def get_split_masks(data: Any, split_index: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    train_mask = getattr(data, "train_mask", None)
    val_mask = getattr(data, "val_mask", None)
    test_mask = getattr(data, "test_mask", None)
    if train_mask is None or val_mask is None or test_mask is None:
        raise ValueError("Dataset does not provide train/val/test masks; add split generation before training.")
    return (
        select_split_mask(train_mask, split_index),
        select_split_mask(val_mask, split_index),
        select_split_mask(test_mask, split_index),
    )


def edge_label_homophily(data: Any) -> float | None:
    y = getattr(data, "y", None)
    edge_index = getattr(data, "edge_index", None)
    if y is None or edge_index is None or edge_index.numel() == 0:
        return None
    src, dst = edge_index
    valid = (y[src] >= 0) & (y[dst] >= 0)
    if int(valid.sum().item()) == 0:
        return None
    return float((y[src][valid] == y[dst][valid]).float().mean().item())


def pyg_data_stats(dataset: Any, split_index: int = 0) -> dict[str, Any]:
    data = dataset[0]
    y = getattr(data, "y", None)
    num_classes = int(dataset.num_classes) if hasattr(dataset, "num_classes") else None
    if num_classes is None and y is not None:
        num_classes = int(y.max().item() + 1)
    stats = {
        "dataset": dataset.__class__.__name__,
        "num_graphs": len(dataset),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(dataset.num_features),
        "num_classes": num_classes,
        "is_undirected": bool(is_undirected(data.edge_index)),
        "edge_label_homophily": edge_label_homophily(data),
        "split_index": split_index,
        "train_nodes": _mask_count(getattr(data, "train_mask", None), split_index),
        "val_nodes": _mask_count(getattr(data, "val_mask", None), split_index),
        "test_nodes": _mask_count(getattr(data, "test_mask", None), split_index),
    }
    return stats


def format_data_stats(stats: dict[str, Any]) -> str:
    keys = [
        "dataset",
        "num_graphs",
        "num_nodes",
        "num_edges",
        "num_features",
        "num_classes",
        "is_undirected",
        "edge_label_homophily",
        "split_index",
        "train_nodes",
        "val_nodes",
        "test_nodes",
    ]
    return "\n".join(f"{key}: {stats.get(key)}" for key in keys)
