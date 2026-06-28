import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from torch_geometric.utils import dropout_edge
except ImportError:  # pragma: no cover
    from torch_geometric.utils import dropout_adj as dropout_edge

from src.data import graph_stats, load_dataset, should_use_mask_eval, split_masks
from src.eval import linear_probe_random, linear_probe_with_masks
from src.losses import (
    info_nce_loss,
    multi_positive_info_nce,
    negative_cosine,
    sampled_info_nce,
    vicreg_regularizer,
    weighted_negative_cosine,
)
from src.models import EnergyRoutedCacheGCL, GraceModel
from src.utils import (
    append_csv,
    ensure_dir,
    feature_drop,
    load_yaml,
    propagation_signature,
    row_normalized_propagate,
    set_seed,
    topk_cache_indices,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", default="energy_spgcl",
                        choices=[
                            "grace",
                            "danv_gcl",
                            "danv_degree_gcl",
                            "fdnv_gcl",
                            "sspnv_gcl",
                            "afpnv_gcl",
                            "bspnv_gcl",
                            "mpnv_gcl",
                            "energy_spgcl",
                            "gcn_mlp_gcl",
                            "er_residual_gcl",
                            "er_cache_gcl",
                        ])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--eval-mode", default=None,
                        choices=[None, "auto", "mask", "random"])
    parser.add_argument("--final-repr", default=None,
                        choices=[None, "ego", "graph", "high", "ego_high", "ego_graph"])
    parser.add_argument("--cache-topk", type=int, default=None)
    parser.add_argument("--cache-key-mode", default=None,
                        choices=[None, "raw_low", "raw_signature", "learned_low"])
    parser.add_argument("--cache-update-interval", type=int, default=None)
    parser.add_argument("--danv-alignment-weight", type=float, default=None)
    parser.add_argument("--danv-disagreement-weight", type=float, default=None)
    parser.add_argument("--danv-gate-temperature", type=float, default=None)
    parser.add_argument("--danv-min-align-weight", type=float, default=None)
    parser.add_argument("--danv-degree-threshold", type=float, default=None)
    parser.add_argument("--danv-degree-temperature", type=float, default=None)
    parser.add_argument("--fdnv-route-weight", type=float, default=None)
    parser.add_argument("--fdnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--fdnv-filter-temperature", type=float, default=None)
    parser.add_argument("--fdnv-min-filter-weight", type=float, default=None)
    parser.add_argument("--sspnv-semantic-weight", type=float, default=None)
    parser.add_argument("--sspnv-spatial-weight", type=float, default=None)
    parser.add_argument("--sspnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--sspnv-semantic-topk", type=int, default=None)
    parser.add_argument("--sspnv-random-semantic", action="store_true")
    parser.add_argument("--sspnv-random-spatial", action="store_true")
    parser.add_argument("--afpnv-semantic-conf-threshold", type=float, default=None)
    parser.add_argument("--afpnv-spatial-conf-threshold", type=float, default=None)
    parser.add_argument("--afpnv-conf-temperature", type=float, default=None)
    parser.add_argument("--afpnv-min-branch-weight", type=float, default=None)
    parser.add_argument("--bspnv-branch-temperature", type=float, default=None)
    parser.add_argument("--bspnv-bootstrap-bias", type=float, default=None)
    parser.add_argument("--mpnv-semantic-weight", type=float, default=None)
    parser.add_argument("--mpnv-spatial-weight", type=float, default=None)
    parser.add_argument("--mpnv-bootstrap-weight", type=float, default=None)
    parser.add_argument("--mpnv-shuffle-positives", action="store_true")
    parser.add_argument("--shuffle-cache", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_device(gpu_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def override_config(config, args):
    merged = dict(config)
    if args.seed is not None:
        merged["seed"] = args.seed
    if args.epochs is not None:
        merged["epochs"] = args.epochs
    if args.eval_mode is not None:
        merged["eval_mode"] = args.eval_mode
    if args.final_repr is not None:
        merged["final_repr"] = args.final_repr
    if args.cache_topk is not None:
        merged["cache_topk"] = args.cache_topk
    if args.cache_key_mode is not None:
        merged["cache_key_mode"] = args.cache_key_mode
    if args.cache_update_interval is not None:
        merged["cache_update_interval"] = args.cache_update_interval
    if args.danv_alignment_weight is not None:
        merged["danv_alignment_weight"] = args.danv_alignment_weight
    if args.danv_disagreement_weight is not None:
        merged["danv_disagreement_weight"] = args.danv_disagreement_weight
    if args.danv_gate_temperature is not None:
        merged["danv_gate_temperature"] = args.danv_gate_temperature
    if args.danv_min_align_weight is not None:
        merged["danv_min_align_weight"] = args.danv_min_align_weight
    if args.danv_degree_threshold is not None:
        merged["danv_degree_threshold"] = args.danv_degree_threshold
    if args.danv_degree_temperature is not None:
        merged["danv_degree_temperature"] = args.danv_degree_temperature
    if args.fdnv_route_weight is not None:
        merged["fdnv_route_weight"] = args.fdnv_route_weight
    if args.fdnv_bootstrap_weight is not None:
        merged["fdnv_bootstrap_weight"] = args.fdnv_bootstrap_weight
    if args.fdnv_filter_temperature is not None:
        merged["fdnv_filter_temperature"] = args.fdnv_filter_temperature
    if args.fdnv_min_filter_weight is not None:
        merged["fdnv_min_filter_weight"] = args.fdnv_min_filter_weight
    if args.sspnv_semantic_weight is not None:
        merged["sspnv_semantic_weight"] = args.sspnv_semantic_weight
    if args.sspnv_spatial_weight is not None:
        merged["sspnv_spatial_weight"] = args.sspnv_spatial_weight
    if args.sspnv_bootstrap_weight is not None:
        merged["sspnv_bootstrap_weight"] = args.sspnv_bootstrap_weight
    if args.sspnv_semantic_topk is not None:
        merged["sspnv_semantic_topk"] = args.sspnv_semantic_topk
    if args.sspnv_random_semantic:
        merged["sspnv_random_semantic"] = True
    if args.sspnv_random_spatial:
        merged["sspnv_random_spatial"] = True
    if args.afpnv_semantic_conf_threshold is not None:
        merged["afpnv_semantic_conf_threshold"] = args.afpnv_semantic_conf_threshold
    if args.afpnv_spatial_conf_threshold is not None:
        merged["afpnv_spatial_conf_threshold"] = args.afpnv_spatial_conf_threshold
    if args.afpnv_conf_temperature is not None:
        merged["afpnv_conf_temperature"] = args.afpnv_conf_temperature
    if args.afpnv_min_branch_weight is not None:
        merged["afpnv_min_branch_weight"] = args.afpnv_min_branch_weight
    if args.bspnv_branch_temperature is not None:
        merged["bspnv_branch_temperature"] = args.bspnv_branch_temperature
    if args.bspnv_bootstrap_bias is not None:
        merged["bspnv_bootstrap_bias"] = args.bspnv_bootstrap_bias
    if args.mpnv_semantic_weight is not None:
        merged["mpnv_semantic_weight"] = args.mpnv_semantic_weight
    if args.mpnv_spatial_weight is not None:
        merged["mpnv_spatial_weight"] = args.mpnv_spatial_weight
    if args.mpnv_bootstrap_weight is not None:
        merged["mpnv_bootstrap_weight"] = args.mpnv_bootstrap_weight
    if args.mpnv_shuffle_positives:
        merged["mpnv_shuffle_positives"] = True
    return merged


def make_run_dir(args, config):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = args.run_name or (
        f"{timestamp}_{args.method}_{args.dataset}_"
        f"seed{config['seed']}_split{args.split_index}"
    )
    run_dir = Path(args.runs_dir) / stem
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory already exists and is non-empty: {run_dir}. "
                "Use --overwrite or choose a new --run-name."
            )
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    return run_dir


def train_grace(model, data, config, args):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        edge_1 = dropout_edge(
            data.edge_index,
            p=float(config["drop_edge_rate_1"]),
            force_undirected=False,
            training=True,
        )[0]
        edge_2 = dropout_edge(
            data.edge_index,
            p=float(config["drop_edge_rate_2"]),
            force_undirected=False,
            training=True,
        )[0]
        x_1 = feature_drop(data.x, float(config["drop_feature_rate_1"]))
        x_2 = feature_drop(data.x, float(config["drop_feature_rate_2"]))
        z_1 = model.project(model(x_1, edge_1))
        z_2 = model.project(model(x_2, edge_2))
        loss = info_nce_loss(z_1, z_2, float(config["tau"]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        row = {"epoch": epoch, "loss": float(loss.item())}
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0:
            print(f"epoch={epoch:03d} loss={row['loss']:.6f}")
    model.eval()
    with torch.no_grad():
        final = model(data.x, data.edge_index)
    return final.detach(), history, {}


def _cache_positive_mean(z, indices):
    if indices.dim() == 1:
        indices = indices.view(-1, 1)
    positives = z[indices.reshape(-1)].view(indices.size(0), indices.size(1), -1)
    return positives.mean(dim=1)


@torch.no_grad()
def _cache_confidence(cache_keys, cache_idx, config):
    if cache_keys is None:
        return None
    keys = F.normalize(cache_keys, dim=1)
    positive = keys[cache_idx[:, 0]]
    sim = (keys * positive).sum(dim=1)
    threshold = float(config["cache_confidence_threshold"])
    temperature = max(float(config["cache_confidence_temperature"]), 1e-12)
    min_weight = float(config["cache_confidence_min_weight"])
    weight = torch.sigmoid((sim - threshold) / temperature)
    weight = weight * (1.0 - min_weight) + min_weight
    return sim, weight


def _sample_positive_indices(cache_idx):
    if cache_idx.size(1) == 1:
        return cache_idx[:, 0]
    choice = torch.randint(
        low=0,
        high=cache_idx.size(1),
        size=(cache_idx.size(0),),
        device=cache_idx.device,
    )
    return cache_idx[torch.arange(cache_idx.size(0), device=cache_idx.device), choice]


def _sample_negative_indices(num_nodes, num_negatives, device):
    num_negatives = min(max(1, int(num_negatives)), max(1, num_nodes - 1))
    negatives = torch.randint(
        low=0,
        high=num_nodes,
        size=(num_nodes, num_negatives),
        device=device,
    )
    row = torch.arange(num_nodes, device=device).view(-1, 1)
    negatives = torch.where(negatives == row, (negatives + 1) % num_nodes, negatives)
    return negatives


def _standardize(vector):
    return (vector - vector.mean()) / vector.std(unbiased=False).clamp_min(1e-12)


@torch.no_grad()
def _danv_alignment_gate(data, parts, config):
    raw_low_raw = row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    raw = F.normalize(data.x.detach().float(), dim=1)
    raw_low = F.normalize(raw_low_raw.float(), dim=1)
    raw_agreement = (raw * raw_low).sum(dim=1)
    raw_residual = (data.x.detach().float() - raw_low_raw.float()).norm(dim=1)
    raw_scale = data.x.detach().float().norm(dim=1).clamp_min(1e-12)
    raw_residual_energy = raw_residual / raw_scale
    view_cosine = (
        F.normalize(parts["ego"].detach(), dim=1)
        * F.normalize(parts["graph"].detach(), dim=1)
    ).sum(dim=1)
    score = (
        _standardize(raw_agreement)
        + _standardize(view_cosine)
        - _standardize(raw_residual_energy)
    )
    gate = torch.sigmoid(score / max(float(config["danv_gate_temperature"]), 1e-12))
    min_weight = float(config["danv_min_align_weight"])
    return gate * (1.0 - min_weight) + min_weight


def _weighted_cosine_abs(z1, z2, weight):
    cosine = (
        F.normalize(z1, dim=1)
        * F.normalize(z2, dim=1)
    ).sum(dim=1).abs()
    weight = weight.detach().to(cosine.device, dtype=cosine.dtype)
    weight = weight / weight.mean().clamp_min(1e-12)
    return (cosine * weight).mean()


@torch.no_grad()
def _degree_disagreement_gate(data, config):
    num_nodes = data.num_nodes
    degree = torch.zeros(num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)
    threshold = float(config["danv_degree_threshold"])
    temperature = max(float(config["danv_degree_temperature"]), 1e-12)
    return torch.sigmoid((log_degree - threshold) / temperature)


@torch.no_grad()
def _fdnv_filter_gate(data, config):
    raw_low_raw = row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    raw = F.normalize(data.x.detach().float(), dim=1)
    raw_low = F.normalize(raw_low_raw.float(), dim=1)
    raw_agreement = (raw * raw_low).sum(dim=1)
    raw_residual = (data.x.detach().float() - raw_low_raw.float()).norm(dim=1)
    raw_scale = data.x.detach().float().norm(dim=1).clamp_min(1e-12)
    raw_residual_energy = raw_residual / raw_scale

    degree = torch.zeros(data.num_nodes, device=data.edge_index.device, dtype=torch.float32)
    ones = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    degree.scatter_add_(0, data.edge_index[0], ones)
    degree.scatter_add_(0, data.edge_index[1], ones)
    log_degree = torch.log1p(degree)

    score = (
        _standardize(raw_residual_energy)
        - _standardize(raw_agreement)
        + 0.5 * _standardize(log_degree)
    )
    temperature = max(float(config["fdnv_filter_temperature"]), 1e-12)
    high_gate = torch.sigmoid(score / temperature)
    min_weight = float(config["fdnv_min_filter_weight"])
    high_gate = high_gate * (1.0 - min_weight) + min_weight
    low_gate = (1.0 - high_gate) * (1.0 - min_weight) + min_weight
    return high_gate, low_gate


@torch.no_grad()
def _semantic_positive_indices(data, config):
    keys = propagation_signature(data.x.detach(), data.edge_index, hops=1)
    return topk_cache_indices(
        keys,
        topk=int(config["sspnv_semantic_topk"]),
        chunk_size=int(config["cache_chunk_size"]),
        exclude_self=True,
    )


@torch.no_grad()
def _random_positive_indices(num_nodes, topk, device):
    topk = max(1, int(topk))
    if num_nodes <= 1:
        return torch.zeros((num_nodes, topk), device=device, dtype=torch.long)
    row = torch.arange(num_nodes, device=device).view(-1, 1)
    positive = torch.randint(
        low=0,
        high=num_nodes - 1,
        size=(num_nodes, topk),
        device=device,
    )
    return positive + (positive >= row).long()


@torch.no_grad()
def _random_single_positive_indices(num_nodes, device):
    if num_nodes <= 1:
        return torch.zeros(num_nodes, device=device, dtype=torch.long)
    row = torch.arange(num_nodes, device=device)
    positive = torch.randint(
        low=0,
        high=num_nodes - 1,
        size=(num_nodes,),
        device=device,
    )
    return positive + (positive >= row).long()


@torch.no_grad()
def _spatial_positive_indices(data):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    source = torch.cat([edge_index[0], edge_index[1]], dim=0)
    target = torch.cat([edge_index[1], edge_index[0]], dim=0)
    positive = torch.arange(num_nodes, device=edge_index.device)
    if source.numel() == 0:
        return positive
    source_sorted, order = torch.sort(source)
    target_sorted = target[order]
    unique, counts = torch.unique_consecutive(source_sorted, return_counts=True)
    starts = torch.cat([
        torch.zeros(1, device=edge_index.device, dtype=torch.long),
        counts.cumsum(dim=0)[:-1],
    ])
    positive[unique] = target_sorted[starts]
    return positive


@torch.no_grad()
def _multi_positive_masks(data, semantic_idx, config):
    num_nodes = data.num_nodes
    device = data.x.device
    semantic_mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    row = torch.arange(num_nodes, device=device).view(-1, 1).expand_as(semantic_idx)
    semantic_mask[row.reshape(-1), semantic_idx.reshape(-1)] = True

    spatial_mask = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.bool)
    source, target = data.edge_index
    spatial_mask[source, target] = True
    spatial_mask[target, source] = True
    if bool(config.get("mpnv_include_self", True)):
        diag = torch.arange(num_nodes, device=device)
        semantic_mask[diag, diag] = True
        spatial_mask[diag, diag] = True
    if bool(config.get("mpnv_shuffle_positives", False)):
        perm = torch.randperm(num_nodes, device=device)
        semantic_mask = semantic_mask[:, perm]
        spatial_mask = spatial_mask[:, perm]
    return semantic_mask, spatial_mask


@torch.no_grad()
def _positive_confidence(cache_keys, semantic_idx, spatial_idx):
    keys = F.normalize(cache_keys, dim=1)
    semantic_positive = keys[semantic_idx.reshape(-1)].view(
        semantic_idx.size(0),
        semantic_idx.size(1),
        -1,
    )
    semantic_sim = (keys.view(keys.size(0), 1, -1) * semantic_positive).sum(dim=2).mean(dim=1)
    spatial_sim = (keys * keys[spatial_idx]).sum(dim=1)
    return semantic_sim, spatial_sim


@torch.no_grad()
def _afpnv_branch_weights(cache_keys, semantic_idx, spatial_idx, config):
    semantic_sim, spatial_sim = _positive_confidence(cache_keys, semantic_idx, spatial_idx)
    temperature = max(float(config["afpnv_conf_temperature"]), 1e-12)
    min_weight = float(config["afpnv_min_branch_weight"])
    semantic_weight = torch.sigmoid(
        (semantic_sim - float(config["afpnv_semantic_conf_threshold"])) / temperature
    )
    spatial_weight = torch.sigmoid(
        (spatial_sim - float(config["afpnv_spatial_conf_threshold"])) / temperature
    )
    semantic_weight = semantic_weight * (1.0 - min_weight) + min_weight
    spatial_weight = spatial_weight * (1.0 - min_weight) + min_weight
    return semantic_weight, spatial_weight, semantic_sim, spatial_sim


@torch.no_grad()
def _bspnv_branch_weights(cache_keys, semantic_idx, spatial_idx, config):
    semantic_sim, spatial_sim = _positive_confidence(cache_keys, semantic_idx, spatial_idx)
    bootstrap_logit = torch.full_like(semantic_sim, float(config["bspnv_bootstrap_bias"]))
    logits = torch.stack([semantic_sim, spatial_sim, bootstrap_logit], dim=1)
    temperature = max(float(config["bspnv_branch_temperature"]), 1e-12)
    probs = torch.softmax(logits / temperature, dim=1)
    return probs[:, 0], probs[:, 1], probs[:, 2], semantic_sim, spatial_sim


def _sspnv_control_name(config):
    tags = []
    if bool(config.get("sspnv_random_semantic", False)):
        tags.append("random_semantic")
    if bool(config.get("sspnv_random_spatial", False)):
        tags.append("random_spatial")
    semantic_active = float(config["sspnv_semantic_weight"]) > 0.0
    spatial_active = float(config["sspnv_spatial_weight"]) > 0.0
    if semantic_active and not spatial_active:
        tags.append("semantic_only")
    elif spatial_active and not semantic_active:
        tags.append("spatial_only")
    elif not semantic_active and not spatial_active:
        tags.append("bootstrap_only")
    return "sspnv" if not tags else "sspnv_" + "_".join(tags)


@torch.no_grad()
def _static_cache_keys(data, config):
    mode = config.get("cache_key_mode", "raw_signature")
    if mode == "raw_low":
        return row_normalized_propagate(data.x.detach(), data.edge_index, add_self=True)
    if mode == "raw_signature":
        return propagation_signature(data.x.detach(), data.edge_index, hops=2)
    if mode == "learned_low":
        return None
    raise ValueError(f"Unknown cache_key_mode: {mode}")


def _cache_diagnostics(parts, cache_idx, cache_keys, cache_weight=None):
    high_norm = parts["high"].norm(dim=1)
    graph_norm = parts["graph"].norm(dim=1).clamp_min(1e-12)
    energy_ratio = high_norm / graph_norm
    if cache_idx.size(1) > 0:
        keys = F.normalize(
            parts["low"] if cache_keys is None else cache_keys,
            dim=1,
        )
        anchor = keys
        positive = keys[cache_idx[:, 0]]
        cache_sim = (anchor * positive).sum(dim=1)
    else:
        cache_sim = torch.zeros_like(energy_ratio)
    return {
        "energy_ratio_mean": float(energy_ratio.mean().item()),
        "energy_ratio_std": float(energy_ratio.std(unbiased=False).item()),
        "cache_low_sim_mean": float(cache_sim.mean().item()),
        "cache_low_sim_std": float(cache_sim.std(unbiased=False).item()),
        "cache_weight_mean": (
            float(cache_weight.mean().item()) if cache_weight is not None else 1.0
        ),
        "cache_weight_std": (
            float(cache_weight.std(unbiased=False).item()) if cache_weight is not None else 0.0
        ),
    }


def train_er_cache_gcl(model, data, config, args):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    energy_spgcl = args.method == "energy_spgcl"
    danv_gcl = args.method in {"danv_gcl", "danv_degree_gcl"}
    danv_degree_gcl = args.method == "danv_degree_gcl"
    fdnv_gcl = args.method == "fdnv_gcl"
    sspnv_gcl = args.method in {"sspnv_gcl", "afpnv_gcl", "bspnv_gcl"}
    afpnv_gcl = args.method == "afpnv_gcl"
    bspnv_gcl = args.method == "bspnv_gcl"
    mpnv_gcl = args.method == "mpnv_gcl"
    residual_only = args.method in {
        "er_residual_gcl",
        "gcn_mlp_gcl",
        "danv_gcl",
        "danv_degree_gcl",
        "fdnv_gcl",
        "sspnv_gcl",
        "afpnv_gcl",
        "bspnv_gcl",
        "mpnv_gcl",
    }
    graph_target = args.method in {
        "gcn_mlp_gcl",
        "danv_gcl",
        "danv_degree_gcl",
        "fdnv_gcl",
        "sspnv_gcl",
        "afpnv_gcl",
        "bspnv_gcl",
        "mpnv_gcl",
    }
    topk = 0 if (args.disable_cache or residual_only) else int(config["cache_topk"])
    cache_update = max(1, int(config["cache_update_interval"]))
    cache_idx = None
    cache_keys = _static_cache_keys(data, config)
    semantic_idx = _semantic_positive_indices(data, config) if (sspnv_gcl or mpnv_gcl) else None
    spatial_idx = _spatial_positive_indices(data) if sspnv_gcl else None
    if sspnv_gcl and bool(config.get("sspnv_random_semantic", False)):
        semantic_idx = _random_positive_indices(
            data.num_nodes,
            int(config["sspnv_semantic_topk"]),
            data.x.device,
        )
    if sspnv_gcl and bool(config.get("sspnv_random_spatial", False)):
        spatial_idx = _random_single_positive_indices(data.num_nodes, data.x.device)
    afpnv_branch_stats = None
    if afpnv_gcl:
        afpnv_branch_stats = _afpnv_branch_weights(
            cache_keys,
            semantic_idx,
            spatial_idx,
            config,
        )
    bspnv_branch_stats = None
    if bspnv_gcl:
        bspnv_branch_stats = _bspnv_branch_weights(
            cache_keys,
            semantic_idx,
            spatial_idx,
            config,
        )
    mpnv_masks = None
    if mpnv_gcl:
        mpnv_masks = _multi_positive_masks(data, semantic_idx, config)
    history = []
    diagnostics = {}
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        parts = model(data.x, data.edge_index, final_mode=config["final_repr"])
        target = parts["graph"] if graph_target else parts["high"]
        with torch.no_grad():
            if cache_idx is None or epoch == 1 or epoch % cache_update == 0:
                if args.disable_cache or residual_only:
                    cache_idx = torch.arange(
                        data.num_nodes,
                        device=data.x.device,
                    ).view(-1, 1)
                else:
                    keys = parts["low"].detach() if cache_keys is None else cache_keys
                    cache_idx = topk_cache_indices(
                        keys,
                        topk=topk,
                        chunk_size=int(config["cache_chunk_size"]),
                        exclude_self=True,
                    )
                if args.shuffle_cache:
                    perm = torch.randperm(cache_idx.size(0), device=cache_idx.device)
                    cache_idx = cache_idx[perm]

        pred_ego = model.pred_ego(parts["ego"])
        pred_high = model.pred_high(target)
        if danv_gcl:
            align_gate = _danv_alignment_gate(data, parts, config)
            disagreement_gate = 1.0 - align_gate
            if danv_degree_gcl:
                disagreement_gate = disagreement_gate * _degree_disagreement_gate(data, config)
            loss_align = 0.5 * (
                weighted_negative_cosine(pred_ego, parts["graph"], align_gate)
                + weighted_negative_cosine(pred_high, parts["ego"], align_gate)
            )
            loss_disagreement = _weighted_cosine_abs(
                parts["ego"],
                parts["graph"],
                disagreement_gate,
            )
            loss_self = (
                float(config["danv_alignment_weight"]) * loss_align
                + float(config["danv_disagreement_weight"]) * loss_disagreement
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif fdnv_gcl:
            high_gate, low_gate = _fdnv_filter_gate(data, config)
            loss_route = 0.5 * (
                weighted_negative_cosine(pred_ego, parts["high"], high_gate)
                + weighted_negative_cosine(pred_ego, parts["low"], low_gate)
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_self = (
                float(config["fdnv_route_weight"]) * loss_route
                + float(config["fdnv_bootstrap_weight"]) * loss_bootstrap
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif sspnv_gcl:
            sem_pos_idx = _sample_positive_indices(semantic_idx)
            neg_idx = _sample_negative_indices(
                data.num_nodes,
                int(config["num_negative_samples"]),
                data.x.device,
            )
            loss_semantic = sampled_info_nce(
                pred_ego,
                parts["high"][sem_pos_idx],
                parts["high"][neg_idx],
                float(config["tau"]),
                (
                    afpnv_branch_stats[0] if afpnv_gcl else
                    bspnv_branch_stats[0] if bspnv_gcl else
                    None
                ),
            )
            loss_spatial = sampled_info_nce(
                pred_ego,
                parts["low"][spatial_idx],
                parts["low"][neg_idx],
                float(config["tau"]),
                (
                    afpnv_branch_stats[1] if afpnv_gcl else
                    bspnv_branch_stats[1] if bspnv_gcl else
                    None
                ),
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            semantic_scale = (
                float(bspnv_branch_stats[0].mean().item()) if bspnv_gcl else 1.0
            )
            spatial_scale = (
                float(bspnv_branch_stats[1].mean().item()) if bspnv_gcl else 1.0
            )
            loss_self = (
                float(config["sspnv_bootstrap_weight"]) * loss_bootstrap
                + semantic_scale * float(config["sspnv_semantic_weight"]) * loss_semantic
                + spatial_scale * float(config["sspnv_spatial_weight"]) * loss_spatial
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif mpnv_gcl:
            semantic_mask, spatial_mask = mpnv_masks
            loss_semantic = multi_positive_info_nce(
                pred_ego,
                parts["high"],
                semantic_mask,
                float(config["tau"]),
            )
            loss_spatial = multi_positive_info_nce(
                pred_ego,
                parts["low"],
                spatial_mask,
                float(config["tau"]),
            )
            loss_bootstrap = 0.5 * (
                negative_cosine(pred_ego, parts["graph"])
                + negative_cosine(pred_high, parts["ego"])
            )
            loss_self = (
                float(config["mpnv_bootstrap_weight"]) * loss_bootstrap
                + float(config["mpnv_semantic_weight"]) * loss_semantic
                + float(config["mpnv_spatial_weight"]) * loss_spatial
            )
            loss_cache = parts["final"].new_tensor(0.0)
        elif energy_spgcl:
            pos_idx = _sample_positive_indices(cache_idx)
            neg_idx = _sample_negative_indices(
                data.num_nodes,
                int(config["num_negative_samples"]),
                data.x.device,
            )
            high_proj = model.pred_high(parts["high"])
            loss_self = sampled_info_nce(
                high_proj,
                high_proj[pos_idx],
                high_proj[neg_idx],
                float(config["tau"]),
            )
            loss_cache = parts["final"].new_tensor(0.0)
        else:
            loss_self = 0.5 * (
                negative_cosine(pred_ego, target)
                + negative_cosine(pred_high, parts["ego"])
            )
        if residual_only or energy_spgcl:
            loss_cache = parts["final"].new_tensor(0.0)
        else:
            pos_high = _cache_positive_mean(parts["high"].detach(), cache_idx)
            pos_ego = _cache_positive_mean(parts["ego"].detach(), cache_idx)
            confidence_result = _cache_confidence(cache_keys, cache_idx, config)
            cache_weight = None if confidence_result is None else confidence_result[1]
            if cache_weight is None:
                loss_cache = 0.5 * (
                    negative_cosine(pred_ego, pos_high)
                    + negative_cosine(pred_high, pos_ego)
                )
            else:
                loss_cache = 0.5 * (
                    weighted_negative_cosine(pred_ego, pos_high, cache_weight)
                    + weighted_negative_cosine(pred_high, pos_ego, cache_weight)
                )
        var_loss, cov_loss = vicreg_regularizer(parts["final"])
        loss = (
            float(config["self_loss_weight"]) * loss_self
            + (0.0 if residual_only else float(config["cache_loss_weight"])) * loss_cache
            + float(config["variance_loss_weight"]) * var_loss
            + float(config["covariance_loss_weight"]) * cov_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "self_loss": float(loss_self.item()),
            "cache_loss": float(loss_cache.item()),
            "variance_loss": float(var_loss.item()),
            "covariance_loss": float(cov_loss.item()),
        }
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} loss={row['loss']:.6f} "
                f"self={row['self_loss']:.6f} cache={row['cache_loss']:.6f}"
            )
    model.eval()
    with torch.no_grad():
        parts = model(data.x, data.edge_index, final_mode=config["final_repr"])
        final = parts["final"].detach()
        confidence_result = _cache_confidence(cache_keys, cache_idx, config)
        cache_weight = None if confidence_result is None else confidence_result[1]
        diagnostics = _cache_diagnostics(parts, cache_idx, cache_keys, cache_weight)
        if danv_gcl:
            gate = _danv_alignment_gate(data, parts, config)
            diagnostics["danv_gate_mean"] = float(gate.mean().item())
            diagnostics["danv_gate_std"] = float(gate.std(unbiased=False).item())
        if danv_degree_gcl:
            degree_gate = _degree_disagreement_gate(data, config)
            diagnostics["danv_degree_gate_mean"] = float(degree_gate.mean().item())
            diagnostics["danv_degree_gate_std"] = float(degree_gate.std(unbiased=False).item())
        if fdnv_gcl:
            high_gate, low_gate = _fdnv_filter_gate(data, config)
            diagnostics["fdnv_high_gate_mean"] = float(high_gate.mean().item())
            diagnostics["fdnv_high_gate_std"] = float(high_gate.std(unbiased=False).item())
            diagnostics["fdnv_low_gate_mean"] = float(low_gate.mean().item())
            diagnostics["fdnv_low_gate_std"] = float(low_gate.std(unbiased=False).item())
        if sspnv_gcl:
            sem_first = semantic_idx[:, 0]
            semantic_sim = (
                F.normalize(cache_keys, dim=1)
                * F.normalize(cache_keys[sem_first], dim=1)
            ).sum(dim=1)
            spatial_is_self = spatial_idx == torch.arange(data.num_nodes, device=data.x.device)
            diagnostics["sspnv_semantic_sim_mean"] = float(semantic_sim.mean().item())
            diagnostics["sspnv_semantic_sim_std"] = float(semantic_sim.std(unbiased=False).item())
            diagnostics["sspnv_spatial_self_fraction"] = float(spatial_is_self.float().mean().item())
            diagnostics["sspnv_semantic_topk"] = int(config["sspnv_semantic_topk"])
            diagnostics["sspnv_random_semantic"] = bool(config.get("sspnv_random_semantic", False))
            diagnostics["sspnv_random_spatial"] = bool(config.get("sspnv_random_spatial", False))
        if afpnv_gcl:
            semantic_weight, spatial_weight, semantic_conf, spatial_conf = afpnv_branch_stats
            diagnostics["afpnv_semantic_weight_mean"] = float(semantic_weight.mean().item())
            diagnostics["afpnv_semantic_weight_std"] = float(
                semantic_weight.std(unbiased=False).item()
            )
            diagnostics["afpnv_spatial_weight_mean"] = float(spatial_weight.mean().item())
            diagnostics["afpnv_spatial_weight_std"] = float(
                spatial_weight.std(unbiased=False).item()
            )
            diagnostics["afpnv_semantic_conf_mean"] = float(semantic_conf.mean().item())
            diagnostics["afpnv_spatial_conf_mean"] = float(spatial_conf.mean().item())
        if bspnv_gcl:
            semantic_prob, spatial_prob, bootstrap_prob, semantic_conf, spatial_conf = bspnv_branch_stats
            winners = torch.stack([semantic_prob, spatial_prob, bootstrap_prob], dim=1).argmax(dim=1)
            diagnostics["bspnv_semantic_prob_mean"] = float(semantic_prob.mean().item())
            diagnostics["bspnv_spatial_prob_mean"] = float(spatial_prob.mean().item())
            diagnostics["bspnv_bootstrap_prob_mean"] = float(bootstrap_prob.mean().item())
            diagnostics["bspnv_semantic_win_fraction"] = float((winners == 0).float().mean().item())
            diagnostics["bspnv_spatial_win_fraction"] = float((winners == 1).float().mean().item())
            diagnostics["bspnv_bootstrap_win_fraction"] = float((winners == 2).float().mean().item())
            diagnostics["bspnv_semantic_conf_mean"] = float(semantic_conf.mean().item())
            diagnostics["bspnv_spatial_conf_mean"] = float(spatial_conf.mean().item())
        if mpnv_gcl:
            semantic_mask, spatial_mask = mpnv_masks
            diagnostics["mpnv_semantic_pos_mean"] = float(semantic_mask.float().sum(dim=1).mean().item())
            diagnostics["mpnv_spatial_pos_mean"] = float(spatial_mask.float().sum(dim=1).mean().item())
            diagnostics["mpnv_semantic_density"] = float(semantic_mask.float().mean().item())
            diagnostics["mpnv_spatial_density"] = float(spatial_mask.float().mean().item())
            diagnostics["mpnv_shuffle_positives"] = bool(config.get("mpnv_shuffle_positives", False))
    diagnostics["cache_control"] = (
        ("danv_degree" if danv_degree_gcl else "danv") if danv_gcl else
        "fdnv" if fdnv_gcl else
        ("mpnv_shuffled" if bool(config.get("mpnv_shuffle_positives", False)) else "mpnv") if mpnv_gcl else
        "bspnv" if bspnv_gcl else
        "afpnv" if afpnv_gcl else
        _sspnv_control_name(config) if sspnv_gcl else
        "energy_spgcl" if energy_spgcl else
        "gcn_mlp_only" if graph_target else
        "residual_only" if residual_only else
        "disabled_self_only" if args.disable_cache else
        "shuffled" if args.shuffle_cache else
        "normal"
    )
    diagnostics["cache_topk"] = int(topk)
    diagnostics["cache_key_mode"] = config.get("cache_key_mode", "raw_signature")
    return final, history, diagnostics


def evaluate_embeddings(embeddings, data, dataset_name, split_index, config, args):
    if args.skip_eval:
        return {}
    if should_use_mask_eval(dataset_name, data, split_index, config["eval_mode"]):
        train_mask, val_mask, test_mask = split_masks(data, split_index)
        return linear_probe_with_masks(
            embeddings,
            data.y,
            train_mask,
            val_mask,
            test_mask,
        )
    return linear_probe_random(
        embeddings,
        data.y,
        ratio=float(args.eval_ratio),
        seed=int(config["seed"]),
    )


def main():
    args = parse_args()
    config = override_config(load_yaml(args.config), args)
    if args.method == "gcn_mlp_gcl" and args.final_repr is None:
        config["final_repr"] = "ego_graph"
    if args.method in {"danv_gcl", "danv_degree_gcl", "fdnv_gcl", "sspnv_gcl", "afpnv_gcl", "bspnv_gcl", "mpnv_gcl"} and args.final_repr is None:
        config["final_repr"] = "ego_graph"
    if args.method == "energy_spgcl" and args.final_repr is None:
        config["final_repr"] = "ego_high"
    set_seed(int(config["seed"]))
    device = get_device(args.gpu_id)
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root or str(project_root / "data")
    dataset = load_dataset(data_root, args.dataset)
    data = dataset[0].to(device)
    run_dir = make_run_dir(args, config)

    stats = graph_stats(dataset, data)
    print(json.dumps({"dataset": args.dataset, **stats}, indent=2, sort_keys=True))
    if args.method == "grace":
        model = GraceModel(
            dataset.num_features,
            int(config["hidden_dim"]),
            int(config["proj_dim"]),
            int(config["num_layers"]),
            float(config["dropout"]),
        ).to(device)
        embeddings, history, diagnostics = train_grace(model, data, config, args)
    else:
        model = EnergyRoutedCacheGCL(
            dataset.num_features,
            int(config["hidden_dim"]),
            int(config["proj_dim"]),
            int(config["num_layers"]),
            float(config["dropout"]),
        ).to(device)
        embeddings, history, diagnostics = train_er_cache_gcl(model, data, config, args)

    metrics = evaluate_embeddings(
        embeddings,
        data,
        args.dataset,
        args.split_index,
        config,
        args,
    )
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(config["seed"]),
        "split_index": int(args.split_index),
        "config": config,
        "graph_stats": stats,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "run.json", payload)
    torch.save(
        {
            "embeddings": embeddings.cpu(),
            "labels": data.y.detach().cpu(),
            "payload": payload,
        },
        run_dir / "artifacts.pt",
    )
    for row in history:
        append_csv(run_dir / "train_log.csv", row)
    summary_row = {
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(config["seed"]),
        "split_index": int(args.split_index),
        "cache_control": diagnostics.get("cache_control", "none"),
        **{f"metric_{key}": value for key, value in metrics.items()},
        **{f"diag_{key}": value for key, value in diagnostics.items()},
    }
    append_csv(Path(args.runs_dir) / "summary.csv", summary_row)
    print(json.dumps(summary_row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
