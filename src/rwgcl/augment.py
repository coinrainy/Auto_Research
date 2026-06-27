"""Graph augmentation placeholders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.utils import dropout_edge


@dataclass(frozen=True)
class AugmentationPlan:
    edge_drop: float
    feature_mask: float
    stage: str


def conservative_plan(edge_drop: float = 0.1, feature_mask: float = 0.1) -> AugmentationPlan:
    return AugmentationPlan(edge_drop=edge_drop, feature_mask=feature_mask, stage="warmup")


def standard_plan(edge_drop: float = 0.2, feature_mask: float = 0.2) -> AugmentationPlan:
    return AugmentationPlan(edge_drop=edge_drop, feature_mask=feature_mask, stage="stage2")


def drop_features(x: Tensor, mask_prob: float) -> Tensor:
    if mask_prob <= 0:
        return x
    keep_mask = torch.rand(x.size(1), device=x.device) >= mask_prob
    return x * keep_mask.to(dtype=x.dtype).view(1, -1)


def augment_graph(x: Tensor, edge_index: Tensor, edge_drop: float, feature_mask: float) -> tuple[Tensor, Tensor]:
    aug_edge_index, _ = dropout_edge(edge_index, p=edge_drop, force_undirected=False, training=True)
    aug_x = drop_features(x, feature_mask)
    return aug_x, aug_edge_index
