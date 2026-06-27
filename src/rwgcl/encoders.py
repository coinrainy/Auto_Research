"""Encoder configuration placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import GCNConv


@dataclass(frozen=True)
class EncoderSpec:
    type: str
    hidden_dim: int
    out_dim: int
    num_layers: int


def encoder_from_config(config: dict[str, Any]) -> EncoderSpec:
    return EncoderSpec(
        type=str(config.get("type", "gcn")),
        hidden_dim=int(config.get("hidden_dim", 256)),
        out_dim=int(config.get("out_dim", 256)),
        num_layers=int(config.get("num_layers", 2)),
    )


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim]
        if num_layers == 1:
            dims.append(out_dim)
        else:
            dims.extend([hidden_dim] * (num_layers - 1))
            dims.append(out_dim)
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if idx < len(self.convs) - 1:
                h = torch.relu(h)
        return h


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRACEModel(nn.Module):
    def __init__(self, in_dim: int, spec: EncoderSpec) -> None:
        super().__init__()
        self.encoder = GCNEncoder(
            in_dim=in_dim,
            hidden_dim=spec.hidden_dim,
            out_dim=spec.out_dim,
            num_layers=spec.num_layers,
        )
        self.projector = ProjectionHead(spec.out_dim, spec.out_dim, spec.out_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)
