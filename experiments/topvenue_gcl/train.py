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
    danv_gcl = args.method == "danv_gcl"
    residual_only = args.method in {"er_residual_gcl", "gcn_mlp_gcl", "danv_gcl"}
    graph_target = args.method in {"gcn_mlp_gcl", "danv_gcl"}
    topk = 0 if (args.disable_cache or residual_only) else int(config["cache_topk"])
    cache_update = max(1, int(config["cache_update_interval"]))
    cache_idx = None
    cache_keys = _static_cache_keys(data, config)
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
    diagnostics["cache_control"] = (
        "danv" if danv_gcl else
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
    if args.method == "danv_gcl" and args.final_repr is None:
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
