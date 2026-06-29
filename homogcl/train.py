from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .augment import (
    edge_drop_homophily,
    edge_drop_random,
    feature_dirichlet_energy,
    feature_drop_homophily,
    feature_drop_random,
    horp_embedding,
    positive_index,
    propagate_features,
    propagation_stack,
    relative_teacher_indices,
)
from .data import edge_homophily, load_graph, mask_counts
from .evaluate import linear_probe
from .models import GCLModel
from .utils import dependency_state, ensure_dir, git_state, set_seed, write_json


def multi_positive_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pos: torch.Tensor,
    tau: float,
    batch_size: int,
) -> torch.Tensor:
    def one_way(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        losses = []
        for start in range(0, a.size(0), batch_size):
            end = min(start + batch_size, a.size(0))
            logits = (a[start:end] @ b.t()) / tau
            pos_logits = logits.gather(1, pos[start:end])
            losses.append((torch.logsumexp(logits, dim=1) - torch.logsumexp(pos_logits, dim=1)).mean())
        return torch.stack(losses).mean()

    return 0.5 * (one_way(z1, z2) + one_way(z2, z1))


def variance_covariance_loss(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.t() @ z) / max(z.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / z.size(1)
    return var_loss, cov_loss


def relative_margin_loss(
    z: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    margin: float,
    batch_size: int,
) -> torch.Tensor:
    if neg.numel() == 0 or pos.size(1) <= 1:
        return z.new_tensor(0.0)
    z = F.normalize(z, p=2, dim=1)
    teacher_pos = pos[:, 1:]
    losses = []
    for start in range(0, z.size(0), batch_size):
        end = min(start + batch_size, z.size(0))
        anchor = z[start:end].unsqueeze(1)
        pos_sim = (anchor * z[teacher_pos[start:end]]).sum(dim=-1).mean(dim=1, keepdim=True)
        neg_sim = (anchor * z[neg[start:end]]).sum(dim=-1)
        losses.append(F.relu(margin - pos_sim + neg_sim).mean())
    return torch.stack(losses).mean()


def train_ssl(args: argparse.Namespace, graph) -> tuple[torch.Tensor, dict[str, float]]:
    data = graph.data
    x = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)
    model = GCLModel(
        in_dim=graph.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    self_pos = torch.arange(x.size(0), device=args.device).view(-1, 1)
    if args.method in {"homogcl", "horpgcl"}:
        energy = feature_dirichlet_energy(x, edge_index)
        if args.method == "horpgcl":
            pos, neg = relative_teacher_indices(
                x=x,
                edge_index=edge_index,
                prop_steps=args.prop_steps,
                pos_k=args.pos_k,
                neg_k=args.neg_k,
                max_dense_nodes=args.max_dense_nodes,
            )
            pos = pos.to(args.device)
            neg = neg.to(args.device)
        else:
            pos = positive_index(
                x=x,
                edge_index=edge_index,
                prop_steps=args.prop_steps,
                pos_k=args.pos_k,
                max_dense_nodes=args.max_dense_nodes,
            ).to(args.device)
            neg = torch.empty(x.size(0), 0, dtype=torch.long, device=args.device)
    else:
        energy = None
        pos = self_pos
        neg = torch.empty(x.size(0), 0, dtype=torch.long, device=args.device)

    last = {"ssl_loss": 0.0, "nce_loss": 0.0, "rank_loss": 0.0, "var_loss": 0.0, "cov_loss": 0.0}
    for epoch in range(args.epochs):
        model.train()
        if args.method in {"homogcl", "horpgcl"}:
            edge1 = edge_drop_homophily(x, edge_index, args.edge_drop)
            edge2 = edge_drop_homophily(x, edge_index, args.edge_drop)
            x1 = feature_drop_homophily(x, args.feat_drop, energy)
            x2 = feature_drop_homophily(x, args.feat_drop, energy)
        elif args.method in {"grace", "gracecat"}:
            edge1 = edge_drop_random(edge_index, args.edge_drop)
            edge2 = edge_drop_random(edge_index, args.edge_drop)
            x1 = feature_drop_random(x, args.feat_drop)
            x2 = feature_drop_random(x, args.feat_drop)
        else:
            raise ValueError(f"Unsupported SSL method: {args.method}")

        z1 = model(x1, edge1)
        z2 = model(x2, edge2)
        nce_loss = multi_positive_nce(z1, z2, pos, args.tau, args.contrast_batch_size)
        rank_loss = relative_margin_loss(
            0.5 * (z1 + z2),
            pos=pos,
            neg=neg,
            margin=args.rank_margin,
            batch_size=args.contrast_batch_size,
        )
        var1, cov1 = variance_covariance_loss(z1)
        var2, cov2 = variance_covariance_loss(z2)
        var_loss = 0.5 * (var1 + var2)
        cov_loss = 0.5 * (cov1 + cov2)
        loss = nce_loss
        if args.method == "homogcl":
            loss = loss + args.var_weight * var_loss + args.cov_weight * cov_loss
        elif args.method == "horpgcl":
            loss = (
                loss
                + args.rank_weight * rank_loss
                + args.var_weight * var_loss
                + args.cov_weight * cov_loss
            )

        opt.zero_grad()
        loss.backward()
        opt.step()
        last = {
            "ssl_loss": float(loss.item()),
            "nce_loss": float(nce_loss.item()),
            "rank_loss": float(rank_loss.item()),
            "var_loss": float(var_loss.item()),
            "cov_loss": float(cov_loss.item()),
        }
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(
                f"epoch={epoch + 1:04d} loss={last['ssl_loss']:.4f} "
                f"nce={last['nce_loss']:.4f} rank={last['rank_loss']:.4f} "
                f"var={last['var_loss']:.4f} cov={last['cov_loss']:.4f}",
                flush=True,
            )

    emb = model.embed(x, edge_index)
    if args.method == "gracecat":
        emb = torch.cat([propagation_bank(x, edge_index, args.prop_steps), emb], dim=1)
    return emb, last


def propagation_bank(x: torch.Tensor, edge_index: torch.Tensor, steps: int) -> torch.Tensor:
    blocks = [F.normalize(current, p=2, dim=1) for current in propagation_stack(x, edge_index, steps)]
    return torch.cat(blocks, dim=1)


def select_autoprop_steps(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_steps: int,
    plateau_ratio: float,
) -> int:
    stack = propagation_stack(x, edge_index, max_steps)
    if len(stack) <= 2:
        return max_steps
    residuals = [
        (stack[i] - stack[i - 1]).pow(2).mean().sqrt().item()
        for i in range(1, len(stack))
    ]
    for idx in range(1, len(residuals)):
        if residuals[idx - 1] > 0 and residuals[idx] / residuals[idx - 1] >= plateau_ratio:
            return idx + 1
    return max_steps


def spectral_profile(x: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = x.float()
    x_centered = x - x.mean(dim=0, keepdim=True)
    q = min(max_rank, min(x_centered.shape) - 1)
    q = max(q, 1)
    u, s, _ = torch.pca_lowrank(x_centered, q=q, center=False, niter=4)
    energy = s.square()
    energy_share = energy / energy.sum().clamp_min(1e-12)
    cumulative = energy_share.cumsum(dim=0)
    participation = energy.sum().square() / energy.square().sum().clamp_min(1e-12)
    top10 = energy_share[: min(10, energy_share.numel())].sum()
    return u, s, {
        "spectral_rank_limit": float(q),
        "spectral_participation_rank": float(participation.item()),
        "spectral_top10_energy": float(top10.item()),
        "spectral_energy_80_rank": float((cumulative >= 0.80).nonzero()[0].item() + 1),
        "spectral_energy_90_rank": float((cumulative >= 0.90).nonzero()[0].item() + 1),
        "spectral_energy_95_rank": float((cumulative >= 0.95).nonzero()[0].item() + 1),
    }


def select_specprop_rank(
    profile: dict[str, float],
    high_concentration: float,
    mid_concentration: float,
    low_rank: int,
) -> int:
    top10 = profile["spectral_top10_energy"]
    if top10 >= high_concentration:
        return low_rank
    if top10 >= mid_concentration:
        return int(profile["spectral_energy_95_rank"])
    return 0


def select_corespecprop_rank(
    profile: dict[str, float],
    high_concentration: float,
    min_rank: int,
    max_rank: int,
    participation_divisor: float,
) -> int:
    if profile["spectral_top10_energy"] < high_concentration:
        return 0
    participation = profile["spectral_participation_rank"]
    core_rank = round(participation / max(participation_divisor, 1e-6))
    return max(min_rank, min(max_rank, int(core_rank)))


def select_tierspecprop_rank(
    profile: dict[str, float],
    high_concentration: float,
    wide_concentration: float,
    narrow_rank: int,
    wide_rank: int,
) -> int:
    top10 = profile["spectral_top10_energy"]
    if top10 < high_concentration:
        return 0
    if top10 >= wide_concentration:
        return wide_rank
    return narrow_rank


def specprop_embedding(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_steps: int,
    plateau_ratio: float,
    max_rank: int,
    low_rank: int,
    high_concentration: float,
    mid_concentration: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected_steps = select_autoprop_steps(
        x,
        edge_index,
        max_steps=max_steps,
        plateau_ratio=plateau_ratio,
    )
    bank = propagation_bank(x, edge_index, selected_steps)
    u, s, profile = spectral_profile(bank, max_rank=max_rank)
    selected_rank = select_specprop_rank(
        profile,
        high_concentration=high_concentration,
        mid_concentration=mid_concentration,
        low_rank=low_rank,
    )
    metrics = {
        "ssl_loss": 0.0,
        "selected_prop_steps": float(selected_steps),
        "selected_pca_rank": float(selected_rank),
        **profile,
    }
    if selected_rank <= 0:
        return bank, metrics
    rank = min(selected_rank, s.numel())
    return u[:, :rank] * s[:rank], metrics


def tierspecprop_embedding(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_steps: int,
    plateau_ratio: float,
    max_rank: int,
    high_concentration: float,
    wide_concentration: float,
    narrow_rank: int,
    wide_rank: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected_steps = select_autoprop_steps(
        x,
        edge_index,
        max_steps=max_steps,
        plateau_ratio=plateau_ratio,
    )
    bank = propagation_bank(x, edge_index, selected_steps)
    u, s, profile = spectral_profile(bank, max_rank=max_rank)
    selected_rank = select_tierspecprop_rank(
        profile,
        high_concentration=high_concentration,
        wide_concentration=wide_concentration,
        narrow_rank=narrow_rank,
        wide_rank=wide_rank,
    )
    metrics = {
        "ssl_loss": 0.0,
        "selected_prop_steps": float(selected_steps),
        "selected_pca_rank": float(selected_rank),
        "tierspecprop_wide_concentration": float(wide_concentration),
        **profile,
    }
    if selected_rank <= 0:
        return bank, metrics
    rank = min(selected_rank, s.numel())
    return u[:, :rank] * s[:rank], metrics


def corespecprop_embedding(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    max_steps: int,
    plateau_ratio: float,
    max_rank: int,
    high_concentration: float,
    min_rank: int,
    max_core_rank: int,
    participation_divisor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected_steps = select_autoprop_steps(
        x,
        edge_index,
        max_steps=max_steps,
        plateau_ratio=plateau_ratio,
    )
    bank = propagation_bank(x, edge_index, selected_steps)
    u, s, profile = spectral_profile(bank, max_rank=max_rank)
    selected_rank = select_corespecprop_rank(
        profile,
        high_concentration=high_concentration,
        min_rank=min_rank,
        max_rank=max_core_rank,
        participation_divisor=participation_divisor,
    )
    metrics = {
        "ssl_loss": 0.0,
        "selected_prop_steps": float(selected_steps),
        "selected_pca_rank": float(selected_rank),
        "corespecprop_participation_divisor": float(participation_divisor),
        **profile,
    }
    if selected_rank <= 0:
        return bank, metrics
    rank = min(selected_rank, s.numel())
    return u[:, :rank] * s[:rank], metrics


def random_bank_drop(bank: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return bank
    keep = torch.rand(bank.size(1), device=bank.device) > drop_rate
    return bank * keep.to(bank.dtype).unsqueeze(0)


def cca_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z1 = (z1 - z1.mean(0)) / z1.std(0).clamp_min(1e-6)
    z2 = (z2 - z2.mean(0)) / z2.std(0).clamp_min(1e-6)
    n = z1.size(0)
    c = (z1.t() @ z2) / n
    c1 = (z1.t() @ z1) / n
    c2 = (z2.t() @ z2) / n
    eye = torch.eye(c.size(0), device=z1.device, dtype=z1.dtype)
    inv = -torch.diagonal(c).sum()
    dec = (eye - c1).pow(2).sum() + (eye - c2).pow(2).sum()
    return inv + lambd * dec, inv, dec


def train_propcca(args: argparse.Namespace, graph) -> tuple[torch.Tensor, dict[str, float]]:
    data = graph.data
    x = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)
    bank = propagation_bank(x, edge_index, args.prop_steps).detach()
    encoder = nn.Linear(bank.size(1), args.out_dim).to(args.device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    last = {"ssl_loss": 0.0, "cca_inv": 0.0, "cca_dec": 0.0}
    for epoch in range(args.epochs):
        encoder.train()
        view1 = random_bank_drop(bank, args.bank_drop)
        view2 = random_bank_drop(bank, args.bank_drop)
        z1 = encoder(view1)
        z2 = encoder(view2)
        loss, inv, dec = cca_loss(z1, z2, args.cca_lambd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = {
            "ssl_loss": float(loss.item()),
            "cca_inv": float(inv.item()),
            "cca_dec": float(dec.item()),
        }
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(
                f"epoch={epoch + 1:04d} loss={last['ssl_loss']:.4f} "
                f"inv={last['cca_inv']:.4f} dec={last['cca_dec']:.4f}",
                flush=True,
            )

    encoder.eval()
    with torch.no_grad():
        learned = encoder(bank)
        emb = torch.cat([bank, learned], dim=1) if args.method == "propccat" else learned
    return emb, last


def train_cca_gcn(args: argparse.Namespace, graph) -> tuple[torch.Tensor, dict[str, float]]:
    data = graph.data
    x = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)
    model = GCLModel(
        in_dim=graph.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    last = {"ssl_loss": 0.0, "cca_inv": 0.0, "cca_dec": 0.0}
    for epoch in range(args.epochs):
        model.encoder.train()
        edge1 = edge_drop_random(edge_index, args.edge_drop)
        edge2 = edge_drop_random(edge_index, args.edge_drop)
        x1 = feature_drop_random(x, args.feat_drop)
        x2 = feature_drop_random(x, args.feat_drop)
        z1 = model.encoder(x1, edge1)
        z2 = model.encoder(x2, edge2)
        loss, inv, dec = cca_loss(z1, z2, args.cca_lambd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = {
            "ssl_loss": float(loss.item()),
            "cca_inv": float(inv.item()),
            "cca_dec": float(dec.item()),
        }
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(
                f"epoch={epoch + 1:04d} loss={last['ssl_loss']:.4f} "
                f"inv={last['cca_inv']:.4f} dec={last['cca_dec']:.4f}",
                flush=True,
            )

    emb = model.embed(x, edge_index)
    if args.method == "ccacat":
        emb = torch.cat([propagation_bank(x, edge_index, args.prop_steps), emb], dim=1)
    return emb, last


def build_embedding(args: argparse.Namespace, graph) -> tuple[torch.Tensor, dict[str, float]]:
    data = graph.data
    x = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)
    if args.method == "raw":
        return x, {"ssl_loss": 0.0}
    if args.method == "prop":
        return propagate_features(x, edge_index, args.prop_steps), {"ssl_loss": 0.0}
    if args.method == "propcat":
        return propagation_bank(x, edge_index, args.prop_steps), {"ssl_loss": 0.0}
    if args.method == "autopropcat":
        selected_steps = select_autoprop_steps(
            x,
            edge_index,
            max_steps=args.max_prop_steps,
            plateau_ratio=args.autoprop_plateau_ratio,
        )
        return propagation_bank(x, edge_index, selected_steps), {
            "ssl_loss": 0.0,
            "selected_prop_steps": float(selected_steps),
        }
    if args.method == "specprop":
        return specprop_embedding(
            x,
            edge_index,
            max_steps=args.max_prop_steps,
            plateau_ratio=args.autoprop_plateau_ratio,
            max_rank=args.specprop_max_rank,
            low_rank=args.specprop_low_rank,
            high_concentration=args.specprop_high_concentration,
            mid_concentration=args.specprop_mid_concentration,
        )
    if args.method == "corespecprop":
        return corespecprop_embedding(
            x,
            edge_index,
            max_steps=args.max_prop_steps,
            plateau_ratio=args.autoprop_plateau_ratio,
            max_rank=args.specprop_max_rank,
            high_concentration=args.specprop_high_concentration,
            min_rank=args.corespecprop_min_rank,
            max_core_rank=args.corespecprop_max_rank,
            participation_divisor=args.corespecprop_participation_divisor,
        )
    if args.method == "tierspecprop":
        return tierspecprop_embedding(
            x,
            edge_index,
            max_steps=args.max_prop_steps,
            plateau_ratio=args.autoprop_plateau_ratio,
            max_rank=args.specprop_max_rank,
            high_concentration=args.specprop_high_concentration,
            wide_concentration=args.tierspecprop_wide_concentration,
            narrow_rank=args.tierspecprop_narrow_rank,
            wide_rank=args.tierspecprop_wide_rank,
        )
    if args.method == "horp":
        return horp_embedding(
            x,
            edge_index,
            steps=args.prop_steps,
            temperature=args.horp_temperature,
            include_residuals=not args.no_horp_residuals,
        ), {"ssl_loss": 0.0}
    if args.method in {"propcca", "propccat"}:
        return train_propcca(args, graph)
    if args.method in {"ccassg", "ccacat"}:
        return train_cca_gcn(args, graph)
    return train_ssl(args, graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomoGCL smoke/research runner")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="public")
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=30)
    parser.add_argument("--test-per-class", type=int, default=0, help="0 means all remaining nodes.")
    parser.add_argument(
        "--method",
        choices=[
            "raw",
            "prop",
            "propcat",
            "autopropcat",
            "specprop",
            "corespecprop",
            "tierspecprop",
            "propcca",
            "propccat",
            "ccassg",
            "ccacat",
            "grace",
            "gracecat",
            "homogcl",
            "horp",
            "horpgcl",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0, help="SSL/model seed.")
    parser.add_argument("--eval-seed", type=int, default=0, help="Linear probe seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--edge-drop", type=float, default=0.25)
    parser.add_argument("--feat-drop", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument("--prop-steps", type=int, default=2)
    parser.add_argument("--max-prop-steps", type=int, default=10)
    parser.add_argument("--autoprop-plateau-ratio", type=float, default=0.75)
    parser.add_argument("--specprop-max-rank", type=int, default=1024)
    parser.add_argument("--specprop-low-rank", type=int, default=32)
    parser.add_argument("--specprop-high-concentration", type=float, default=0.34)
    parser.add_argument("--specprop-mid-concentration", type=float, default=0.34)
    parser.add_argument("--corespecprop-min-rank", type=int, default=16)
    parser.add_argument("--corespecprop-max-rank", type=int, default=32)
    parser.add_argument("--corespecprop-participation-divisor", type=float, default=3.0)
    parser.add_argument("--tierspecprop-wide-concentration", type=float, default=0.36)
    parser.add_argument("--tierspecprop-narrow-rank", type=int, default=16)
    parser.add_argument("--tierspecprop-wide-rank", type=int, default=32)
    parser.add_argument("--pos-k", type=int, default=4)
    parser.add_argument("--neg-k", type=int, default=16)
    parser.add_argument("--max-dense-nodes", type=int, default=6000)
    parser.add_argument("--var-weight", type=float, default=0.05)
    parser.add_argument("--cov-weight", type=float, default=0.005)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--horp-temperature", type=float, default=0.2)
    parser.add_argument("--no-horp-residuals", action="store_true")
    parser.add_argument("--cca-lambd", type=float, default=1e-3)
    parser.add_argument("--bank-drop", type=float, default=0.2)
    parser.add_argument("--contrast-batch-size", type=int, default=1024)
    parser.add_argument("--probe", choices=["sklogreg", "torchlogreg", "logreg", "ridge"], default="sklogreg")
    parser.add_argument("--ridge-alphas", default="0.001,0.01,0.1,1,10,100")
    parser.add_argument("--logreg-c-grid", default=",".join(str(2.0**i) for i in range(-10, 11)))
    parser.add_argument("--linear-lr", type=float, default=0.01)
    parser.add_argument("--linear-weight-decay", type=float, default=0.0)
    parser.add_argument("--linear-epochs", type=int, default=1000)
    parser.add_argument("--linear-patience", type=int, default=100)
    parser.add_argument("--output-dir", default="results/smoke")
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def result_signature(args: argparse.Namespace) -> str:
    parts = [f"k{args.prop_steps}"]
    if args.method == "autopropcat":
        parts = [f"maxk{args.max_prop_steps}", f"plateau{args.autoprop_plateau_ratio:g}"]
    if args.method == "specprop":
        parts = [
            f"maxk{args.max_prop_steps}",
            f"plateau{args.autoprop_plateau_ratio:g}",
            f"sr{args.specprop_max_rank}",
            f"lr{args.specprop_low_rank}",
            f"hc{args.specprop_high_concentration:g}",
            f"mc{args.specprop_mid_concentration:g}",
        ]
    if args.method == "corespecprop":
        parts = [
            f"maxk{args.max_prop_steps}",
            f"plateau{args.autoprop_plateau_ratio:g}",
            f"sr{args.specprop_max_rank}",
            f"hc{args.specprop_high_concentration:g}",
            f"cr{args.corespecprop_min_rank}-{args.corespecprop_max_rank}",
            f"pd{args.corespecprop_participation_divisor:g}",
        ]
    if args.method == "tierspecprop":
        parts = [
            f"maxk{args.max_prop_steps}",
            f"plateau{args.autoprop_plateau_ratio:g}",
            f"sr{args.specprop_max_rank}",
            f"hc{args.specprop_high_concentration:g}",
            f"wc{args.tierspecprop_wide_concentration:g}",
            f"tr{args.tierspecprop_narrow_rank}-{args.tierspecprop_wide_rank}",
        ]
    if args.method in {"homogcl", "horpgcl"}:
        parts.extend([f"pk{args.pos_k}", f"ed{args.edge_drop:g}", f"fd{args.feat_drop:g}"])
    if args.method == "horpgcl":
        parts.extend([f"nk{args.neg_k}", f"rw{args.rank_weight:g}", f"rm{args.rank_margin:g}"])
    if args.method == "horp":
        parts.append(f"ht{args.horp_temperature:g}")
        if args.no_horp_residuals:
            parts.append("nores")
    if args.method in {"propcca", "propccat"}:
        parts.extend([f"bd{args.bank_drop:g}", f"cl{args.cca_lambd:g}"])
    if args.method in {"ccassg", "ccacat"}:
        parts.extend([f"ed{args.edge_drop:g}", f"fd{args.feat_drop:g}", f"cl{args.cca_lambd:g}"])
    return "_".join(parts)


def split_signature(args: argparse.Namespace, graph) -> str:
    safe_split = args.split.replace("_", "-")
    return f"{safe_split}_split{graph.split_index}_sseed{graph.split_seed}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    graph = load_graph(
        args.dataset,
        root=args.data_root,
        split=args.split,
        split_index=args.split_index,
        split_seed=None if args.split_seed < 0 else args.split_seed,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
    )
    graph.data = graph.data.to(args.device)
    emb, train_metrics = build_embedding(args, graph)

    data = graph.data
    probe = linear_probe(
        emb=emb,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_classes=graph.num_classes,
        seed=args.eval_seed,
        probe=args.probe,
        alphas=[float(x) for x in args.ridge_alphas.split(",") if x.strip()],
        c_grid=[float(x) for x in args.logreg_c_grid.split(",") if x.strip()],
        lr=args.linear_lr,
        weight_decay=args.linear_weight_decay,
        epochs=args.linear_epochs,
        patience=args.linear_patience,
    )
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    payload = {
        "timestamp_utc": now,
        "dataset": graph.name,
        "method": args.method,
        "metrics": {**train_metrics, **probe},
        "protocol": {
            "task": "node_classification",
            "metric": "accuracy",
            "encoder_eval": f"frozen_encoder_{args.probe}_linear_probe",
            "split_protocol": graph.split_protocol,
            "split_index": graph.split_index,
            "split_seed": graph.split_seed,
            "mask_counts": mask_counts(data),
            "edge_homophily_diagnostic_uses_labels": edge_homophily(data),
            "model_seed": args.seed,
            "eval_seed": args.eval_seed,
            "test_labels_used_in_ssl": False,
        },
        "hparams": vars(args),
        "dependencies": dependency_state(),
        "git": git_state(Path.cwd()),
        "command": " ".join(os.sys.argv),
    }
    out_dir = ensure_dir(args.output_dir)
    out_file = out_dir / (
        f"{graph.name}_{args.method}_{result_signature(args)}_"
        f"{split_signature(args, graph)}_seed{args.seed}_eval{args.eval_seed}.json"
    )
    write_json(out_file, payload)
    print(
        f"RESULT dataset={graph.name} method={args.method} "
        f"test_acc={probe['test_acc']:.4f} val_acc={probe['val_acc']:.4f} file={out_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
