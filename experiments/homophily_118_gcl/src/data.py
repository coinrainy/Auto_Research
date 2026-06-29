from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree, is_undirected


def load_planetoid(data_root, name):
    return Planetoid(root=str(Path(data_root) / "Planetoid"), name=name)


def graph_stats(dataset, data):
    src, dst = data.edge_index
    same = (data.y[src] == data.y[dst]).float()
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "is_undirected": bool(is_undirected(data.edge_index)),
        "edge_label_homophily": float(same.mean().item()) if same.numel() else 0.0,
        "avg_degree": float(degree(src, data.num_nodes).mean().item()),
    }


def stratified_118_split(labels, split_index, train_ratio=0.1, val_ratio=0.1, base_seed=2026):
    labels = labels.detach().cpu()
    generator = torch.Generator()
    generator.manual_seed(int(base_seed) + int(split_index))
    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    for cls in labels.unique(sorted=True):
        idx = torch.where(labels == cls)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        n_train = max(1, int(round(idx.numel() * float(train_ratio))))
        n_val = max(1, int(round(idx.numel() * float(val_ratio))))
        if n_train + n_val >= idx.numel():
            n_train = max(1, idx.numel() // 3)
            n_val = max(1, idx.numel() // 3)
        train[perm[:n_train]] = True
        val[perm[n_train:n_train + n_val]] = True
        test[perm[n_train + n_val:]] = True
    return train, val, test


def split_stats(train_mask, val_mask, test_mask):
    total = train_mask.numel()
    return {
        "train_count": int(train_mask.sum().item()),
        "val_count": int(val_mask.sum().item()),
        "test_count": int(test_mask.sum().item()),
        "train_ratio": float(train_mask.float().mean().item()),
        "val_ratio": float(val_mask.float().mean().item()),
        "test_ratio": float(test_mask.float().mean().item()),
        "split_protocol": "stratified_1:1:8",
        "total_count": int(total),
    }
