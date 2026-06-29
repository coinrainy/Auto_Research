from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.data import Data


PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


@dataclass
class LoadedGraph:
    name: str
    data: Data
    num_features: int
    num_classes: int
    split_protocol: str
    split_index: int
    split_seed: int


def _select_mask(mask: torch.Tensor, split_index: int) -> torch.Tensor:
    if mask.dim() == 1:
        if split_index != 0:
            raise ValueError("1D public mask only supports split_index=0")
        return mask.bool()
    if split_index < 0 or split_index >= mask.size(1):
        raise ValueError(f"split_index={split_index} outside mask shape {tuple(mask.shape)}")
    return mask[:, split_index].bool()


def _class_balanced_random_masks(
    y: torch.Tensor,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)

    used = torch.zeros(y.size(0), dtype=torch.bool)
    for cls in range(num_classes):
        idx = (y.cpu() == cls).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        needed = train_per_class + val_per_class + max(test_per_class, 0)
        if idx.numel() < train_per_class + val_per_class:
            raise ValueError(
                f"Class {cls} has {idx.numel()} nodes, fewer than "
                f"train_per_class + val_per_class = {train_per_class + val_per_class}"
            )
        if test_per_class > 0 and idx.numel() < needed:
            raise ValueError(f"Class {cls} has {idx.numel()} nodes, fewer than requested {needed}")
        train_idx = idx[:train_per_class]
        val_idx = idx[train_per_class : train_per_class + val_per_class]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        used[train_idx] = True
        used[val_idx] = True
        if test_per_class > 0:
            test_idx = idx[train_per_class + val_per_class : needed]
            test_mask[test_idx] = True
            used[test_idx] = True

    if test_per_class <= 0:
        test_mask = ~used
    return train_mask, val_mask, test_mask


def load_graph(
    name: str,
    root: str = "data",
    split: str = "public",
    split_index: int = 0,
    split_seed: int | None = None,
    train_per_class: int = 20,
    val_per_class: int = 30,
    test_per_class: int = 0,
) -> LoadedGraph:
    key = name.lower()
    if key in PLANETOID_NAMES:
        dataset_name = PLANETOID_NAMES[key]
        dataset_split = "public" if split in {"class-random", "random"} else split
        dataset = Planetoid(root=str(Path(root) / "Planetoid"), name=dataset_name, split=dataset_split)
        split_protocol = f"Planetoid:{split}"
    elif key in {"computers", "photo"}:
        dataset_name = "Computers" if key == "computers" else "Photo"
        dataset = Amazon(root=str(Path(root) / "Amazon"), name=dataset_name)
        split_protocol = f"Amazon:{split}"
    elif key in {"cs", "physics"}:
        dataset_name = "CS" if key == "cs" else "Physics"
        dataset = Coauthor(root=str(Path(root) / "Coauthor"), name=dataset_name)
        split_protocol = f"Coauthor:{split}"
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    data = dataset[0]
    actual_seed = split_index if split_seed is None else split_seed
    if split in {"class-random", "random"}:
        data.train_mask, data.val_mask, data.test_mask = _class_balanced_random_masks(
            y=data.y,
            num_classes=dataset.num_classes,
            train_per_class=train_per_class,
            val_per_class=val_per_class,
            test_per_class=test_per_class,
            seed=actual_seed,
        )
        split_protocol = (
            f"{split_protocol}:seed={actual_seed}:train_per_class={train_per_class}:"
            f"val_per_class={val_per_class}:test_per_class={test_per_class or 'rest'}"
        )
    elif not hasattr(data, "train_mask") or data.train_mask is None:
        raise ValueError(
            f"{name} has no built-in train/val/test masks in this prototype. "
            "Use --split class-random for datasets without built-in masks."
        )
    else:
        data.train_mask = _select_mask(data.train_mask, split_index)
        data.val_mask = _select_mask(data.val_mask, split_index)
        data.test_mask = _select_mask(data.test_mask, split_index)
    return LoadedGraph(
        name=dataset.name,
        data=data,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        split_protocol=split_protocol,
        split_index=split_index,
        split_seed=actual_seed,
    )


def mask_counts(data: Data) -> dict[str, int]:
    return {
        "train": int(data.train_mask.sum().item()),
        "val": int(data.val_mask.sum().item()),
        "test": int(data.test_mask.sum().item()),
    }


def edge_homophily(data: Data) -> float:
    row, col = data.edge_index
    same = data.y[row] == data.y[col]
    return float(same.float().mean().item())
