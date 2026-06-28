import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .utils import row_normalized_propagate


def activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "prelu":
        return nn.PReLU()
    raise ValueError(f"Unknown activation: {name}")


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.5, act="prelu"):
        super().__init__()
        layers = []
        for idx in range(num_layers):
            input_dim = in_dim if idx == 0 else hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation(act))
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.5, act="prelu"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.dropout = dropout
        for idx in range(num_layers):
            input_dim = in_dim if idx == 0 else hidden_dim
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            self.acts.append(activation(act))

    def forward(self, x, edge_index):
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            x = conv(x, edge_index)
            x = norm(x)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ELU(),
            nn.Linear(proj_dim, in_dim),
        )

    def forward(self, x):
        return self.net(x)


class GraceModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, num_layers, dropout):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.projector = ProjectionHead(hidden_dim, proj_dim)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def project(self, z):
        return self.projector(z)


class EnergyRoutedCacheGCL(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, num_layers, dropout):
        super().__init__()
        self.ego_encoder = MLPEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.graph_encoder = GCNEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.ego_predictor = ProjectionHead(hidden_dim, proj_dim)
        self.high_predictor = ProjectionHead(hidden_dim, proj_dim)
        self.final_norm = nn.LayerNorm(hidden_dim * 2)

    def encode_parts(self, x, edge_index):
        ego = self.ego_encoder(x)
        graph = self.graph_encoder(x, edge_index)
        low = row_normalized_propagate(graph, edge_index, add_self=True)
        high = graph - low
        final = self.compose_final(ego, graph, low, high, mode="ego_high")
        return {
            "ego": ego,
            "graph": graph,
            "low": low,
            "high": high,
            "final": final,
        }

    def compose_final(self, ego, graph, low, high, mode):
        del low
        if mode == "ego":
            return ego
        if mode == "graph":
            return graph
        if mode == "high":
            return high
        if mode == "ego_high":
            return self.final_norm(torch.cat([
                F.normalize(ego, dim=1),
                F.normalize(high, dim=1),
            ], dim=1))
        if mode == "ego_graph":
            return self.final_norm(torch.cat([
                F.normalize(ego, dim=1),
                F.normalize(graph, dim=1),
            ], dim=1))
        raise ValueError(f"Unknown final representation mode: {mode}")

    def forward(self, x, edge_index, final_mode="ego_high"):
        parts = self.encode_parts(x, edge_index)
        parts["final"] = self.compose_final(
            parts["ego"],
            parts["graph"],
            parts["low"],
            parts["high"],
            final_mode,
        )
        return parts

    def pred_ego(self, z):
        return self.ego_predictor(z)

    def pred_high(self, z):
        return self.high_predictor(z)

