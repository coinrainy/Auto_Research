import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(int(num_layers)):
            src_dim = in_dim if layer == 0 else hidden_dim
            self.convs.append(GCNConv(src_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.prelu(x, weight=x.new_tensor(0.25))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ELU(),
            nn.Linear(proj_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class GCLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, proj_dim, num_layers, dropout):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.projector = ProjectionHead(hidden_dim, proj_dim)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def project(self, z):
        return self.projector(z)
