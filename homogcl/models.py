from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=False)
        self.conv2 = GCNConv(hidden_dim, out_dim, cached=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GCLModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        proj_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, out_dim, dropout)
        self.projector = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.PReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return self.projector(z)

    @torch.no_grad()
    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x, edge_index)
