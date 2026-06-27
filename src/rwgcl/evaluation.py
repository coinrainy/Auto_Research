"""Evaluation summary helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn


def read_metric_rows(path: str | Path) -> list[dict]:
    metric_path = Path(path)
    if not metric_path.exists():
        return []
    with metric_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_metric_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    summary = []
    for key_values, group_rows in grouped.items():
        out = {key: value for key, value in zip(keys, key_values)}
        out["runs"] = str(len(group_rows))
        statuses = sorted({row.get("status", "") for row in group_rows})
        out["statuses"] = "|".join(statuses)
        summary.append(out)
    return summary


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def linear_probe(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float = 0.01,
    weight_decay: float = 0.0,
) -> dict[str, float]:
    device = embeddings.device
    classifier = nn.Linear(embeddings.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    best_test = 0.0
    for _ in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(embeddings)
        loss = nn.functional.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            logits = classifier(embeddings)
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
    return {"val_accuracy": best_val, "test_accuracy": best_test}
