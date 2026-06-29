import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from src.data import graph_stats, load_planetoid, split_stats, stratified_118_split
from src.eval import linear_eval
from src.losses import sampled_contrastive, sample_negative_indices
from src.models import GCLModel
from src.signature import propagation_signature, topk_similar
from src.utils import append_csv, ensure_dir, feature_drop, load_yaml, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument(
        "--method",
        default="hpfs_gcl",
        choices=[
            "raw_features",
            "grace",
            "hpfs_gcl",
            "rpgcl_grace",
            "rpgcl_hpfs",
            "rpgcl_auto",
        ],
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--semantic-weight", type=float, default=None)
    parser.add_argument("--semantic-topk", type=int, default=None)
    parser.add_argument("--num-negatives", type=int, default=None)
    parser.add_argument("--neg-threshold", type=float, default=None)
    parser.add_argument("--neg-min-weight", type=float, default=None)
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--gcl-weight", type=float, default=1.0)
    parser.add_argument("--shuffle-positives", action="store_true")
    parser.add_argument("--disable-neg-suppression", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def device_from_args(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        return torch.device(f"cuda:{args.gpu_id}")
    return torch.device("cpu")


def merged_config(args):
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.semantic_weight is not None:
        cfg["semantic_weight"] = args.semantic_weight
    if args.semantic_topk is not None:
        cfg["semantic_topk"] = args.semantic_topk
    if args.num_negatives is not None:
        cfg["num_negatives"] = args.num_negatives
    if args.neg_threshold is not None:
        cfg["neg_threshold"] = args.neg_threshold
    if args.neg_min_weight is not None:
        cfg["neg_min_weight"] = args.neg_min_weight
    if args.disable_neg_suppression:
        cfg["neg_suppression"] = False
    return cfg


def make_run_dir(args, cfg):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"{stamp}_{args.dataset}_{args.method}_seed{cfg['seed']}_split{args.split_index}"
    path = Path(args.runs_dir) / name
    if path.exists() and any(path.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{path} already exists; use --overwrite")
        shutil.rmtree(path)
    ensure_dir(path)
    return path


@torch.no_grad()
def compact_signature(keys, dim, seed):
    dim = int(dim)
    if dim <= 0 or keys.size(1) <= dim:
        return F.normalize(keys.float(), dim=1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    proj = torch.randn(keys.size(1), dim, generator=gen, dtype=torch.float32)
    proj = proj.to(keys.device) / (float(dim) ** 0.5)
    return F.normalize(keys.float() @ proj, dim=1)


@torch.no_grad()
def negative_weights(keys, neg_idx, cfg):
    if not bool(cfg.get("neg_suppression", True)):
        return None, None
    anchor = keys.unsqueeze(1)
    neg = keys[neg_idx.reshape(-1)].view(neg_idx.size(0), neg_idx.size(1), -1)
    sim = (anchor * neg).sum(dim=2)
    threshold = float(cfg["neg_threshold"])
    temperature = max(float(cfg["neg_temperature"]), 1e-8)
    min_weight = float(cfg["neg_min_weight"])
    weight = torch.sigmoid((threshold - sim) / temperature)
    weight = min_weight + (1.0 - min_weight) * weight
    return weight.detach(), sim.detach()


def train_gcl(model, data, cfg, args, keys, semantic_pos):
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    history = []
    eye_pos = torch.arange(data.num_nodes, device=data.x.device)
    if args.shuffle_positives:
        semantic_pos = semantic_pos[torch.randperm(semantic_pos.size(0), device=semantic_pos.device)]
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        edge1 = dropout_edge(data.edge_index, p=float(cfg["drop_edge_rate_1"]), training=True)[0]
        edge2 = dropout_edge(data.edge_index, p=float(cfg["drop_edge_rate_2"]), training=True)[0]
        x1 = feature_drop(data.x, float(cfg["drop_feature_rate_1"]))
        x2 = feature_drop(data.x, float(cfg["drop_feature_rate_2"]))
        z1 = model.project(model.encode(x1, edge1))
        z2 = model.project(model.encode(x2, edge2))
        neg_idx = sample_negative_indices(data.num_nodes, int(cfg["num_negatives"]), data.x.device)
        if args.method in {"hpfs_gcl", "rpgcl_hpfs", "rpgcl_auto"}:
            neg_weight, neg_sim = negative_weights(keys, neg_idx, cfg)
        else:
            neg_weight, neg_sim = None, None
        loss_self = sampled_contrastive(z1, z2, eye_pos, neg_idx, float(cfg["tau"]), neg_weight)
        if args.method in {"hpfs_gcl", "rpgcl_hpfs", "rpgcl_auto"}:
            loss_sem = sampled_contrastive(z1, z2, semantic_pos, neg_idx, float(cfg["tau"]), neg_weight)
            loss = float(cfg["self_weight"]) * loss_self + float(cfg["semantic_weight"]) * loss_sem
        else:
            loss_sem = z1.new_tensor(0.0)
            loss = loss_self
        opt.zero_grad()
        loss.backward()
        opt.step()
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "self_loss": float(loss_self.item()),
            "semantic_loss": float(loss_sem.item()),
            "neg_weight_mean": float(neg_weight.mean().item()) if neg_weight is not None else 1.0,
            "neg_key_sim_mean": float(neg_sim.mean().item()) if neg_sim is not None else 0.0,
        }
        history.append(row)
        if epoch == 1 or epoch % int(args.log_every) == 0:
            print(json.dumps(row, sort_keys=True))
    model.eval()
    with torch.no_grad():
        emb = model.encode(data.x, data.edge_index).detach()
    return emb, history


def main():
    args = parse_args()
    cfg = merged_config(args)
    set_seed(int(cfg["seed"]))
    device = device_from_args(args)
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root or str(project_root / "data")
    dataset = load_planetoid(data_root, args.dataset)
    data = dataset[0].to(device)
    train_mask, val_mask, test_mask = stratified_118_split(
        data.y,
        args.split_index,
        cfg["split_train_ratio"],
        cfg["split_val_ratio"],
        cfg["split_base_seed"],
    )
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    run_dir = make_run_dir(args, cfg)
    stats = graph_stats(dataset, data)
    splits = split_stats(train_mask, val_mask, test_mask)
    print(json.dumps({"dataset": args.dataset, **stats, **splits}, indent=2, sort_keys=True))

    keys = propagation_signature(data.x, data.edge_index, int(cfg["signature_hops"]))
    keys = compact_signature(keys, int(cfg["signature_dim"]), int(cfg["split_base_seed"]))
    semantic_pos = topk_similar(keys, int(cfg["semantic_topk"]), exclude_self=True)

    if args.method == "raw_features":
        emb = data.x.detach().float()
        history = []
    else:
        model = GCLModel(
            dataset.num_features,
            int(cfg["hidden_dim"]),
            int(cfg["proj_dim"]),
            int(cfg["num_layers"]),
            float(cfg["dropout"]),
        ).to(device)
        emb, history = train_gcl(model, data, cfg, args, keys, semantic_pos)
        if args.method in {"rpgcl_grace", "rpgcl_hpfs"}:
            emb = torch.cat([
                float(args.raw_weight) * F.normalize(data.x.detach().float(), dim=1),
                float(args.gcl_weight) * F.normalize(emb.detach().float(), dim=1),
            ], dim=1)
    diagnostics = {
        "shuffle_positives": bool(args.shuffle_positives),
        "neg_suppression": bool(args.method in {"hpfs_gcl", "rpgcl_hpfs", "rpgcl_auto"} and cfg.get("neg_suppression", True)),
        "semantic_topk": int(cfg["semantic_topk"]),
        "semantic_weight": float(cfg["semantic_weight"]),
        "num_negatives": int(cfg["num_negatives"]),
        "signature_hops": int(cfg["signature_hops"]),
        "signature_dim": int(keys.size(1)),
        "raw_preserved": bool(args.method in {"rpgcl_grace", "rpgcl_hpfs"}),
        "raw_weight": float(args.raw_weight),
        "gcl_weight": float(args.gcl_weight),
    }
    if args.method in {"hpfs_gcl", "rpgcl_hpfs", "rpgcl_auto"}:
        first_pos = semantic_pos[:, 0]
        diagnostics["semantic_top1_key_sim"] = float((keys * keys[first_pos]).sum(dim=1).mean().item())
    if args.method == "rpgcl_auto":
        candidates = {
            "hpfs": emb,
            "raw_preserved_hpfs": torch.cat([
                float(args.raw_weight) * F.normalize(data.x.detach().float(), dim=1),
                float(args.gcl_weight) * F.normalize(emb.detach().float(), dim=1),
            ], dim=1),
            "raw_features": data.x.detach().float(),
        }
        candidate_metrics = {
            name: linear_eval(
                cand,
                data.y,
                train_mask,
                val_mask,
                test_mask,
                max_iter=int(cfg["eval_max_iter"]),
            )
            for name, cand in candidates.items()
        }
        choice = max(candidate_metrics, key=lambda name: candidate_metrics[name]["val_accuracy"])
        emb = candidates[choice]
        metrics = candidate_metrics[choice]
        diagnostics["raw_preserved"] = choice == "raw_preserved_hpfs"
        diagnostics["rpgcl_auto_choice"] = choice
        for name, values in candidate_metrics.items():
            diagnostics[f"rpgcl_auto_{name}_val_accuracy"] = float(values["val_accuracy"])
            diagnostics[f"rpgcl_auto_{name}_test_F1Mi"] = float(values["F1Mi"])
            diagnostics[f"rpgcl_auto_{name}_test_F1Ma"] = float(values["F1Ma"])
    else:
        metrics = linear_eval(
            emb,
            data.y,
            train_mask,
            val_mask,
            test_mask,
            max_iter=int(cfg["eval_max_iter"]),
        )
    payload = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(cfg["seed"]),
        "split_index": int(args.split_index),
        "config": cfg,
        "graph_stats": stats,
        "split_stats": splits,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    write_json(run_dir / "run.json", payload)
    torch.save({"embeddings": emb.cpu(), "labels": data.y.cpu(), "payload": payload}, run_dir / "artifacts.pt")
    for row in history:
        append_csv(run_dir / "train_log.csv", row)
    summary = {
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "method": args.method,
        "seed": int(cfg["seed"]),
        "split_index": int(args.split_index),
        **{f"metric_{k}": v for k, v in metrics.items()},
        **{f"split_{k}": v for k, v in splits.items()},
        **{f"diag_{k}": v for k, v in diagnostics.items()},
    }
    append_csv(Path(args.runs_dir) / "summary.csv", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
