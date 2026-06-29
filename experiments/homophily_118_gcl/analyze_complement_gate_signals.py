import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from src.data import load_planetoid
from src.signature import propagation_signature
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs/rpgcl_auto_selector_controls_splits0-9_e50")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--aggregate-out", default=None)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["run_dir"] = str(path.parent)
    return payload


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(value) for value in values if value not in ("", None)]
    return "" if not values else sum(values) / len(values)


def safe_std(values):
    values = torch.as_tensor([float(value) for value in values if value not in ("", None)])
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


@torch.no_grad()
def sample_nodes(num_nodes, sample_size, seed):
    if sample_size <= 0 or sample_size >= num_nodes:
        return torch.arange(num_nodes)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(num_nodes, generator=generator)[:sample_size]


@torch.no_grad()
def cosine_edge_stats(x, edge_index):
    x = F.normalize(x.float(), dim=1)
    src, dst = edge_index
    edge_cos = (x[src] * x[dst]).sum(dim=1)
    generator = torch.Generator(device=edge_index.device)
    generator.manual_seed(2026)
    rand_src = torch.randint(0, x.size(0), (src.numel(),), generator=generator, device=edge_index.device)
    rand_dst = torch.randint(0, x.size(0), (src.numel(),), generator=generator, device=edge_index.device)
    rand_cos = (x[rand_src] * x[rand_dst]).sum(dim=1)
    return {
        "edge_feature_cos_mean": float(edge_cos.mean().item()),
        "edge_feature_cos_std": float(edge_cos.std(unbiased=False).item()),
        "random_feature_cos_mean": float(rand_cos.mean().item()),
        "edge_feature_cos_lift": float((edge_cos.mean() - rand_cos.mean()).item()),
    }


@torch.no_grad()
def pairwise_similarity_alignment(raw, hpfs, sample_size, seed):
    idx = sample_nodes(raw.size(0), sample_size, seed)
    raw_s = F.normalize(raw[idx].float(), dim=1)
    hpfs_s = F.normalize(hpfs[idx].float(), dim=1)
    raw_sim = raw_s @ raw_s.t()
    hpfs_sim = hpfs_s @ hpfs_s.t()
    mask = ~torch.eye(raw_sim.size(0), dtype=torch.bool, device=raw_sim.device)
    raw_vec = raw_sim[mask]
    hpfs_vec = hpfs_sim[mask]
    raw_centered = raw_vec - raw_vec.mean()
    hpfs_centered = hpfs_vec - hpfs_vec.mean()
    denom = raw_centered.norm() * hpfs_centered.norm()
    corr = torch.tensor(0.0, device=raw.device) if denom.item() == 0.0 else (raw_centered * hpfs_centered).sum() / denom
    gap = (raw_sim - hpfs_sim).abs()[mask]
    return {
        "raw_hpfs_pair_sim_corr": float(corr.item()),
        "raw_hpfs_pair_sim_abs_gap": float(gap.mean().item()),
    }


@torch.no_grad()
def propagation_change_stats(x, edge_index, hops):
    raw = F.normalize(x.float(), dim=1)
    sig = F.normalize(propagation_signature(x, edge_index, int(hops)).float(), dim=1)
    cosine = (raw * sig[:, : raw.size(1)]).sum(dim=1) if sig.size(1) >= raw.size(1) else torch.zeros(raw.size(0))
    return {
        "raw_propagation_cos_mean": float(cosine.mean().item()),
        "raw_propagation_cos_std": float(cosine.std(unbiased=False).item()),
    }


def load_artifact(run_dir):
    path = Path(run_dir) / "artifacts.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def best_fixed_label(group_rows):
    candidates = ["hpfs_gcl", "rpgcl_hpfs"]
    acc = {}
    for row in group_rows:
        if row["method"] in candidates:
            acc[row["method"]] = float(row["metrics"]["accuracy"])
    if not acc:
        return "", ""
    method, value = max(acc.items(), key=lambda item: item[1])
    return method, value


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root or str(project_root / "data")
    runs_dir = Path(args.runs_dir)
    runs = [read_json(path) for path in sorted(runs_dir.glob("*/run.json"))]
    by_key = defaultdict(list)
    for run in runs:
        by_key[(run["dataset"], int(run["split_index"]), int(run.get("model_seed", run["seed"])))].append(run)

    rows = []
    for key, group in sorted(by_key.items()):
        dataset_name, split_index, model_seed = key
        hpfs_run = next((run for run in group if run["method"] == "hpfs_gcl"), None)
        rawhpfs_run = next((run for run in group if run["method"] == "rpgcl_hpfs"), None)
        grace_run = next((run for run in group if run["method"] == "grace"), None)
        if hpfs_run is None or rawhpfs_run is None:
            continue

        dataset = load_planetoid(data_root, dataset_name)
        data = dataset[0]
        x = data.x.cpu().float()
        hpfs_artifact = load_artifact(hpfs_run["run_dir"])
        if hpfs_artifact is None:
            continue
        hpfs_emb = hpfs_artifact["embeddings"].float()
        edge_stats = cosine_edge_stats(x, data.edge_index)
        align_stats = pairwise_similarity_alignment(x, hpfs_emb, args.sample_size, args.seed + split_index)
        prop_stats = propagation_change_stats(x, data.edge_index, hpfs_run["config"].get("signature_hops", 2))
        deg = degree(data.edge_index[0], data.num_nodes)
        nnz = (x != 0).float().sum(dim=1)
        best_method, best_acc = best_fixed_label(group)
        hpfs_acc = float(hpfs_run["metrics"]["accuracy"])
        rawhpfs_acc = float(rawhpfs_run["metrics"]["accuracy"])
        grace_acc = "" if grace_run is None else float(grace_run["metrics"]["accuracy"])
        raw_branch_gain = rawhpfs_acc - hpfs_acc
        row = {
            "dataset": dataset_name,
            "split_index": split_index,
            "model_seed": model_seed,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_features": int(dataset.num_features),
            "feature_density": float((x != 0).float().mean().item()),
            "feature_nnz_mean": float(nnz.mean().item()),
            "feature_nnz_std": float(nnz.std(unbiased=False).item()),
            "avg_degree": float(deg.mean().item()),
            "hpfs_accuracy": hpfs_acc,
            "raw_preserved_hpfs_accuracy": rawhpfs_acc,
            "grace_light_accuracy": grace_acc,
            "raw_branch_gain": raw_branch_gain,
            "best_fixed_method": best_method,
            "best_fixed_accuracy": best_acc,
            "gate_target_raw_preserve": int(raw_branch_gain > 0.0),
            **edge_stats,
            **align_stats,
            **prop_stats,
        }
        rows.append(row)

    aggregate = []
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset_name, items in sorted(by_dataset.items()):
        aggregate.append({
            "dataset": dataset_name,
            "num_runs": len(items),
            "hpfs_accuracy_mean": mean(row["hpfs_accuracy"] for row in items),
            "raw_preserved_hpfs_accuracy_mean": mean(row["raw_preserved_hpfs_accuracy"] for row in items),
            "raw_branch_gain_mean": mean(row["raw_branch_gain"] for row in items),
            "raw_branch_gain_std": safe_std(row["raw_branch_gain"] for row in items),
            "raw_branch_positive": sum(1 for row in items if float(row["raw_branch_gain"]) > 0.0),
            "raw_branch_negative": sum(1 for row in items if float(row["raw_branch_gain"]) < 0.0),
            "feature_density": mean(row["feature_density"] for row in items),
            "feature_nnz_mean": mean(row["feature_nnz_mean"] for row in items),
            "avg_degree": mean(row["avg_degree"] for row in items),
            "edge_feature_cos_mean": mean(row["edge_feature_cos_mean"] for row in items),
            "random_feature_cos_mean": mean(row["random_feature_cos_mean"] for row in items),
            "edge_feature_cos_lift": mean(row["edge_feature_cos_lift"] for row in items),
            "raw_hpfs_pair_sim_corr": mean(row["raw_hpfs_pair_sim_corr"] for row in items),
            "raw_hpfs_pair_sim_abs_gap": mean(row["raw_hpfs_pair_sim_abs_gap"] for row in items),
            "raw_propagation_cos_mean": mean(row["raw_propagation_cos_mean"] for row in items),
            "raw_propagation_cos_std": mean(row["raw_propagation_cos_std"] for row in items),
        })

    write_csv(args.out or runs_dir / "complement_gate_signals.csv", rows)
    write_csv(args.aggregate_out or runs_dir / "complement_gate_signal_aggregate.csv", aggregate)
    print(json.dumps({"rows": len(rows), "datasets": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
