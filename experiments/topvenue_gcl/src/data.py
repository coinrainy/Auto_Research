from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    Actor,
    CitationFull,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.utils import is_undirected


PLANETOID = {"Cora", "CiteSeer", "PubMed"}
WEBKB = {"Texas", "Cornell", "Wisconsin"}
WIKINETWORK = {"Chameleon", "Squirrel"}
HETEROPHILOUS = {
    "Roman-empire",
    "Amazon-ratings",
    "Minesweeper",
    "Tolokers",
    "Questions",
}
MASK_EVAL_DATASETS = WEBKB | WIKINETWORK | {"Actor"} | HETEROPHILOUS


def load_dataset(root, name):
    root = Path(root)
    transform = T.NormalizeFeatures()
    if name in PLANETOID:
        dataset = Planetoid(str(root / "Planetoid"), name, transform=transform)
    elif name == "DBLP":
        dataset = CitationFull(str(root / "CitationFull"), "dblp", transform=transform)
    elif name in WEBKB:
        dataset = WebKB(str(root / "WebKB"), name, transform=transform)
    elif name == "Actor":
        dataset = Actor(str(root / "Actor"), transform=transform)
    elif name in WIKINETWORK:
        dataset = WikipediaNetwork(str(root / "WikipediaNetwork"), name.lower(), transform=transform)
    elif name in HETEROPHILOUS:
        dataset = HeterophilousGraphDataset(str(root / "HeterophilousGraphDataset"), name, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return dataset


def select_mask(mask, split_index):
    if mask is None:
        return None
    if mask.dim() == 1:
        return mask.bool()
    if split_index < 0 or split_index >= mask.size(1):
        raise ValueError(
            f"split_index={split_index} out of range for {mask.size(1)} splits"
        )
    return mask[:, split_index].bool()


def split_masks(data, split_index):
    if not all(hasattr(data, key) for key in ["train_mask", "val_mask", "test_mask"]):
        return None, None, None
    return (
        select_mask(data.train_mask, split_index),
        select_mask(data.val_mask, split_index),
        select_mask(data.test_mask, split_index),
    )


def should_use_mask_eval(dataset_name, data, split_index, eval_mode):
    if eval_mode == "random":
        return False
    if eval_mode == "mask":
        return True
    train_mask, val_mask, test_mask = split_masks(data, split_index)
    return (
        dataset_name in MASK_EVAL_DATASETS
        and train_mask is not None
        and val_mask is not None
        and test_mask is not None
    )


@torch.no_grad()
def edge_label_homophily(data):
    y = data.y
    if y.dim() > 1:
        y = y.view(-1)
    src, dst = data.edge_index
    if src.numel() == 0:
        return float("nan")
    return float((y[src] == y[dst]).float().mean().item())


def graph_stats(dataset, data):
    train_mask, val_mask, test_mask = split_masks(data, 0)
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "is_undirected": bool(is_undirected(data.edge_index)),
        "edge_label_homophily": edge_label_homophily(data),
        "has_masks": bool(train_mask is not None and val_mask is not None and test_mask is not None),
        "num_splits": int(data.train_mask.size(1)) if hasattr(data, "train_mask") and data.train_mask.dim() > 1 else 1,
    }
