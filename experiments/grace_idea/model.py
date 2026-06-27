import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class EgoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 k: int = 2):
        super(EgoEncoder, self).__init__()
        assert k >= 2
        self.layers = [nn.Linear(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.layers.append(nn.Linear(2 * out_channels, 2 * out_channels))
        self.layers.append(nn.Linear(2 * out_channels, out_channels))
        self.layers = nn.ModuleList(self.layers)
        self.activation = activation
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        del edge_index
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.norm(x)


class ResidualEgoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, gate_init: float = 0.5):
        super(ResidualEgoEncoder, self).__init__()
        assert k >= 2
        self.gcn_encoder = Encoder(
            in_channels,
            out_channels,
            activation,
            base_model=base_model,
            k=k,
        )
        self.ego_layers = [nn.Linear(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.ego_layers.append(nn.Linear(2 * out_channels, 2 * out_channels))
        self.ego_layers.append(nn.Linear(2 * out_channels, out_channels))
        self.ego_layers = nn.ModuleList(self.ego_layers)
        self.activation = activation
        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(
            torch.logit(torch.tensor(float(gate_init)))
        )
        self.norm = nn.LayerNorm(out_channels)

    @property
    def ego_gate(self):
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        gcn_out = self.gcn_encoder(x, edge_index)
        ego_out = x
        for layer in self.ego_layers:
            ego_out = self.activation(layer(ego_out))
        gate = self.ego_gate
        return self.norm(gate * gcn_out + (1.0 - gate) * ego_out)


class GatedEgoGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2, gate_temperature: float = 0.5,
                 gate_threshold: float = 0.0, gate_min: float = 0.0,
                 gate_max: float = 1.0):
        super(GatedEgoGraphEncoder, self).__init__()
        assert k >= 2
        self.gcn_encoder = Encoder(
            in_channels,
            out_channels,
            activation,
            base_model=base_model,
            k=k,
        )
        self.ego_encoder = EgoEncoder(
            in_channels,
            out_channels,
            activation,
            k=k,
        )
        self.gate_temperature = max(float(gate_temperature), 1e-12)
        self.gate_threshold = float(gate_threshold)
        self.gate_min = min(max(float(gate_min), 0.0), 1.0)
        self.gate_max = min(max(float(gate_max), self.gate_min), 1.0)
        self.norm = nn.LayerNorm(out_channels)
        self.last_graph_gate = None

    @torch.no_grad()
    def graph_usage_gate(self, x: torch.Tensor, edge_index: torch.Tensor):
        source, target = edge_index
        features = F.normalize(x.detach().float(), dim=1)
        aggregate = torch.zeros_like(features)
        degree = torch.zeros(features.size(0), device=features.device,
                             dtype=features.dtype)
        aggregate.index_add_(0, target, features[source])
        degree.index_add_(0, target, torch.ones_like(target, dtype=features.dtype))
        neighbor_mean = aggregate / degree.clamp_min(1.0).view(-1, 1)
        neighbor_mean = F.normalize(neighbor_mean, dim=1)
        agreement = (features * neighbor_mean).sum(1)
        agreement = torch.where(degree > 0, agreement, torch.zeros_like(agreement))
        score = (
            (agreement - agreement.mean())
            / agreement.std(unbiased=False).clamp_min(1e-12)
        )
        gate = torch.sigmoid((score - self.gate_threshold) / self.gate_temperature)
        if self.gate_min > 0.0 or self.gate_max < 1.0:
            gate = gate * (self.gate_max - self.gate_min) + self.gate_min
        return gate.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        gcn_out = self.gcn_encoder(x, edge_index)
        ego_out = self.ego_encoder(x, edge_index)
        gate = self.graph_usage_gate(x, edge_index).to(
            gcn_out.device,
            dtype=gcn_out.dtype,
        ).view(-1, 1)
        self.last_graph_gate = gate.detach()
        return self.norm(gate * gcn_out + (1.0 - gate) * ego_out)


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                  denominator_weights: torch.Tensor = None,
                  pair_denominator_weights: torch.Tensor = None):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        positive = between_sim.diag()

        if pair_denominator_weights is not None:
            weights = pair_denominator_weights.to(
                z1.device,
                dtype=z1.dtype,
            ).clamp_min(0.0)
            diag_weights = weights.diag()
            denominator = (
                (refl_sim * weights).sum(1)
                - refl_sim.diag() * diag_weights
                + (between_sim * weights).sum(1)
                - positive * diag_weights
                + positive
            )
        elif denominator_weights is None:
            denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        else:
            weights = denominator_weights.to(z1.device, dtype=z1.dtype).clamp_min(0.0)
            candidate_weights = weights.view(1, -1)
            denominator = (
                (refl_sim * candidate_weights).sum(1)
                - refl_sim.diag() * weights
                + (between_sim * candidate_weights).sum(1)
                - positive * weights
                + positive
            )

        return -torch.log(positive / denominator.clamp_min(1e-12))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int,
                          denominator_weights: torch.Tensor = None,
                          pair_denominator_weights: torch.Tensor = None):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        weights = None
        if denominator_weights is not None:
            weights = denominator_weights.to(device, dtype=z1.dtype).clamp_min(0.0)
        pair_weights = None
        if pair_denominator_weights is not None:
            pair_weights = pair_denominator_weights.to(
                device,
                dtype=z1.dtype,
            ).clamp_min(0.0)
        losses = []

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            mask = indices[start:end]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]
            positive = between_sim[:, start:end].diag()

            if pair_weights is not None:
                pair_rows = pair_weights[mask]
                batch_diag_weights = pair_rows[:, start:end].diag()
                denominator = (
                    (refl_sim * pair_rows).sum(1)
                    - refl_sim[:, start:end].diag() * batch_diag_weights
                    + (between_sim * pair_rows).sum(1)
                    - positive * batch_diag_weights
                    + positive
                )
            elif weights is None:
                denominator = (
                    refl_sim.sum(1)
                    + between_sim.sum(1)
                    - refl_sim[:, start:end].diag()
                )
            else:
                candidate_weights = weights.view(1, -1)
                batch_weights = weights[mask]
                denominator = (
                    (refl_sim * candidate_weights).sum(1)
                    - refl_sim[:, start:end].diag() * batch_weights
                    + (between_sim * candidate_weights).sum(1)
                    - positive * batch_weights
                    + positive
                )

            losses.append(-torch.log(positive / denominator.clamp_min(1e-12)))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0,
             pair_weights: torch.Tensor = None,
             denominator_weights: torch.Tensor = None,
             pair_denominator_weights: torch.Tensor = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(
                h1,
                h2,
                denominator_weights,
                pair_denominator_weights,
            )
            l2 = self.semi_loss(
                h2,
                h1,
                denominator_weights,
                pair_denominator_weights,
            )
        else:
            l1 = self.batched_semi_loss(
                h1,
                h2,
                batch_size,
                denominator_weights,
                pair_denominator_weights,
            )
            l2 = self.batched_semi_loss(
                h2,
                h1,
                batch_size,
                denominator_weights,
                pair_denominator_weights,
            )

        ret = (l1 + l2) * 0.5
        if pair_weights is not None:
            weights = pair_weights.to(ret.device, dtype=ret.dtype).clamp_min(0.0)
            ret = (ret * weights).sum() if not mean else (
                (ret * weights).sum() / weights.sum().clamp_min(1e-12)
            )
        else:
            ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
