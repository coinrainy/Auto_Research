import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_csv(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            old_header = next(reader, None)
        if old_header and old_header != fieldnames:
            fieldnames = list(dict.fromkeys(old_header + fieldnames))
            rows = []
            with open(path, "r", encoding="utf-8") as handle:
                for item in csv.DictReader(handle):
                    rows.append(item)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for item in rows:
                    writer.writerow({key: item.get(key, "") for key in fieldnames})
            exists = True
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def row_normalized_propagate(x, edge_index, add_self=True):
    source, target = edge_index
    out = torch.zeros_like(x)
    degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    out.index_add_(0, target, x[source])
    degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
    if add_self:
        out = out + x
        degree = degree + 1.0
    return out / degree.clamp_min(1.0).view(-1, 1)


@torch.no_grad()
def propagation_signature(x, edge_index, hops=2):
    blocks = [F.normalize(x.float(), dim=1)]
    current = x.float()
    for _ in range(max(0, hops)):
        propagated = row_normalized_propagate(current, edge_index, add_self=True)
        residual = current - propagated
        blocks.append(F.normalize(propagated, dim=1))
        blocks.append(F.normalize(residual, dim=1))
        current = propagated
    return F.normalize(torch.cat(blocks, dim=1), dim=1)


def feature_drop(x, drop_prob):
    if drop_prob <= 0.0:
        return x
    keep = torch.empty(
        x.size(1),
        dtype=x.dtype,
        device=x.device,
    ).uniform_(0, 1) > drop_prob
    return x * keep.view(1, -1)


def off_diagonal(x):
    rows, cols = x.shape
    assert rows == cols
    return x.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


@torch.no_grad()
def topk_cache_indices(keys, topk, chunk_size=2048, exclude_self=True):
    num_nodes = keys.size(0)
    if topk <= 0:
        return torch.arange(num_nodes, device=keys.device).view(-1, 1)
    topk = min(topk, max(1, num_nodes - int(exclude_self)))
    keys = F.normalize(keys, dim=1)
    all_indices = []
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        scores = keys[start:end] @ keys.t()
        if exclude_self:
            row = torch.arange(end - start, device=keys.device)
            col = torch.arange(start, end, device=keys.device)
            scores[row, col] = -float("inf")
        indices = scores.topk(k=topk, dim=1).indices
        all_indices.append(indices)
    return torch.cat(all_indices, dim=0)


def masked_mean_std(values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item())
