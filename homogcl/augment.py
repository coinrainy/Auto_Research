from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree


def propagate_features(x: torch.Tensor, edge_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Symmetric GCN-style propagation without trainable weights."""
    if steps <= 0:
        return x
    num_nodes = x.size(0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=x.dtype).clamp_min(1.0)
    norm = deg[row].pow(-0.5) * deg[col].pow(-0.5)
    out = x
    for _ in range(steps):
        nxt = torch.zeros_like(out)
        nxt.index_add_(0, col, out[row] * norm.unsqueeze(-1))
        out = nxt
    return out


@torch.no_grad()
def propagation_stack(x: torch.Tensor, edge_index: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Return [X, SX, ..., S^KX] under symmetric GCN-style propagation."""
    stack = [x]
    current = x
    for _ in range(max(steps, 0)):
        current = propagate_features(current, edge_index, 1)
        stack.append(current)
    return stack


@torch.no_grad()
def propagation_residual_weights(stack: list[torch.Tensor], temperature: float = 0.2) -> torch.Tensor:
    """Node-wise reliability over propagation scales from low residual drift."""
    if len(stack) == 1:
        return torch.ones(stack[0].size(0), 1, device=stack[0].device, dtype=stack[0].dtype)
    residuals = [torch.zeros(stack[0].size(0), device=stack[0].device, dtype=stack[0].dtype)]
    residuals.extend((stack[i] - stack[i - 1]).pow(2).mean(dim=1) for i in range(1, len(stack)))
    score = -torch.stack(residuals, dim=1)
    scale = score.std(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.softmax(score / (temperature * scale), dim=1)


@torch.no_grad()
def horp_embedding(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    temperature: float = 0.2,
    include_residuals: bool = True,
) -> torch.Tensor:
    """Homophily-aware relative propagation teacher representation.

    HoRP keeps a node-specific smooth propagation mixture and exposes the full
    propagation trace to the linear probe. Residual channels let downstream
    diagnostics detect oversmoothing instead of assuming a single global depth.
    """
    stack = propagation_stack(x, edge_index, steps)
    weights = propagation_residual_weights(stack, temperature=temperature)
    mix = sum(weights[:, i : i + 1] * stack[i] for i in range(len(stack)))
    blocks = [F.normalize(mix, p=2, dim=1)]
    blocks.extend(F.normalize(item, p=2, dim=1) for item in stack)
    if include_residuals:
        blocks.extend(F.normalize(stack[i] - stack[i - 1], p=2, dim=1) for i in range(1, len(stack)))
    return torch.cat(blocks, dim=1)


@torch.no_grad()
def feature_dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    diff = x[row] - x[col]
    numerator = diff.pow(2).mean(dim=0)
    denominator = x.pow(2).mean(dim=0).clamp_min(1e-12)
    energy = numerator / denominator
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def feature_drop_homophily(
    x: torch.Tensor,
    drop_rate: float,
    energy: torch.Tensor,
) -> torch.Tensor:
    if drop_rate <= 0:
        return x
    scaled = energy / energy.mean().clamp_min(1e-12)
    drop_prob = (drop_rate * scaled).clamp(0.0, 0.95)
    keep = torch.rand(x.size(1), device=x.device) > drop_prob.to(x.device)
    return x * keep.to(x.dtype).unsqueeze(0)


@torch.no_grad()
def feature_drop_random(x: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return x
    keep = torch.rand(x.size(1), device=x.device) > drop_rate
    return x * keep.to(x.dtype).unsqueeze(0)


@torch.no_grad()
def edge_drop_random(edge_index: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep]


@torch.no_grad()
def edge_reliability_from_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    x_norm = F.normalize(x, p=2, dim=1)
    sim = (x_norm[row] * x_norm[col]).sum(dim=-1)
    return sim.clamp(min=0.0, max=1.0)


@torch.no_grad()
def edge_drop_homophily(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    drop_rate: float,
    min_keep_prob: float = 0.05,
) -> torch.Tensor:
    if drop_rate <= 0:
        return edge_index
    reliability = edge_reliability_from_features(x, edge_index)
    raw_drop = (1.0 - reliability).clamp_min(0.0)
    drop_prob = drop_rate * raw_drop / raw_drop.mean().clamp_min(1e-12)

    row, col = edge_index
    deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
    low_degree = torch.minimum(deg[row], deg[col]) <= 2
    drop_prob = drop_prob.where(~low_degree, drop_prob * 0.25)
    drop_prob = drop_prob.clamp(0.0, 1.0 - min_keep_prob)
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_prob
    return edge_index[:, keep]


@torch.no_grad()
def positive_index(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    prop_steps: int,
    pos_k: int,
    max_dense_nodes: int = 6000,
) -> torch.Tensor:
    """Return [num_nodes, 1 + pos_k] positive ids; column 0 is always self."""
    num_nodes = x.size(0)
    pos = torch.arange(num_nodes, device=x.device).view(-1, 1).repeat(1, pos_k + 1)
    if pos_k <= 0:
        return pos[:, :1]

    prop = F.normalize(propagate_features(x, edge_index, prop_steps), p=2, dim=1)
    if num_nodes <= max_dense_nodes:
        sim = prop @ prop.t()
        sim.fill_diagonal_(-1.0)
        nn = sim.topk(k=pos_k, dim=1).indices
        pos[:, 1:] = nn
        return pos

    reliability = edge_reliability_from_features(prop, edge_index).detach().cpu()
    row, col = edge_index.detach().cpu()
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for src, dst, score in zip(row.tolist(), col.tolist(), reliability.tolist()):
        if src != dst:
            buckets[src].append((score, dst))
    for node in range(num_nodes):
        if not buckets[node]:
            continue
        best = sorted(buckets[node], reverse=True)[:pos_k]
        for j, (_, dst) in enumerate(best, start=1):
            pos[node, j] = dst
    return pos


@torch.no_grad()
def relative_teacher_indices(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    prop_steps: int,
    pos_k: int,
    neg_k: int,
    max_dense_nodes: int = 6000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher positives and negatives for relative-order constraints."""
    pos = positive_index(
        x=x,
        edge_index=edge_index,
        prop_steps=prop_steps,
        pos_k=pos_k,
        max_dense_nodes=max_dense_nodes,
    )
    num_nodes = x.size(0)
    neg = torch.arange(num_nodes, device=x.device).view(-1, 1).repeat(1, max(neg_k, 1))
    if neg_k <= 0:
        return pos, neg[:, :0]

    teacher = F.normalize(horp_embedding(x, edge_index, prop_steps), p=2, dim=1)
    if num_nodes <= max_dense_nodes:
        sim = teacher @ teacher.t()
        sim.fill_diagonal_(1.0)
        neg = sim.topk(k=neg_k, largest=False, dim=1).indices
        return pos, neg

    # Sparse fallback: use low-similarity random candidates to avoid dense N^2 memory.
    candidates = torch.randint(0, num_nodes, (num_nodes, neg_k * 8), device=x.device)
    candidates = candidates.where(candidates != torch.arange(num_nodes, device=x.device).view(-1, 1), (candidates + 1) % num_nodes)
    cand_sim = (teacher.unsqueeze(1) * teacher[candidates]).sum(dim=-1)
    pick = cand_sim.topk(k=neg_k, largest=False, dim=1).indices
    return pos, candidates.gather(1, pick)
