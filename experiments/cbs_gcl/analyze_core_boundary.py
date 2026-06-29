import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree


CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="runs/core_boundary_diagnostics_splits0-4")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-base-seed", type=int, default=2026)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def append_csv(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_dataset(data_root, name):
    return Planetoid(root=str(Path(data_root) / "Planetoid"), name=name)


def stratified_split(labels, split_index, train_ratio, val_ratio, base_seed):
    labels = labels.detach().cpu()
    generator = torch.Generator()
    generator.manual_seed(int(base_seed) + int(split_index))
    train = torch.zeros(labels.numel(), dtype=torch.bool)
    val = torch.zeros(labels.numel(), dtype=torch.bool)
    test = torch.zeros(labels.numel(), dtype=torch.bool)
    for cls in labels.unique(sorted=True):
        idx = torch.where(labels == cls)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        n_train = max(1, int(round(idx.numel() * float(train_ratio))))
        n_val = max(1, int(round(idx.numel() * float(val_ratio))))
        if n_train + n_val >= idx.numel():
            n_train = max(1, idx.numel() // 3)
            n_val = max(1, idx.numel() // 3)
        train[perm[:n_train]] = True
        val[perm[n_train:n_train + n_val]] = True
        test[perm[n_train + n_val:]] = True
    return train, val, test


def row_propagate(x, edge_index):
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    src, dst = edge_index
    deg = degree(dst, x.size(0), dtype=x.dtype).clamp_min(1.0)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src] / deg[dst].view(-1, 1))
    return out


def percentile_rank(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks / max(1, x.numel() - 1)


def pearson(x, y):
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float(((x * y).sum() / denom).item())


@torch.no_grad()
def node_mean_neighbor_cos(h, edge_index):
    h = F.normalize(h.float(), dim=1)
    src, dst = edge_index
    cos = (h[src] * h[dst]).sum(dim=1)
    out = torch.zeros(h.size(0), dtype=torch.float32)
    cnt = torch.zeros(h.size(0), dtype=torch.float32)
    out.index_add_(0, dst.cpu(), cos.cpu())
    cnt.index_add_(0, dst.cpu(), torch.ones_like(cos.cpu()))
    return out / cnt.clamp_min(1.0)


@torch.no_grad()
def compute_views_and_core(data):
    x = data.x.float().cpu()
    edge_index = data.edge_index.cpu()
    h0 = F.normalize(x, dim=1)
    h1 = F.normalize(row_propagate(x, edge_index), dim=1)
    h2 = F.normalize(row_propagate(row_propagate(x, edge_index), edge_index), dim=1)
    residual = F.normalize(x - row_propagate(x, edge_index), dim=1)

    depth_agreement = 0.5 * ((h0 * h1).sum(dim=1) + (h1 * h2).sum(dim=1))
    neighbor_consistency = node_mean_neighbor_cos(h1, edge_index)
    residual_norm = (x - row_propagate(x, edge_index)).norm(dim=1)
    raw_core = (
        percentile_rank(depth_agreement)
        + percentile_rank(neighbor_consistency)
        + (1.0 - percentile_rank(residual_norm))
    ) / 3.0
    deg = degree(edge_index[0], x.size(0)).float()
    deg_log = torch.log1p(deg)
    # Remove the linear degree component so the score is not merely a high-degree detector.
    centered_deg = deg_log - deg_log.mean()
    denom = (centered_deg * centered_deg).sum().clamp_min(1e-8)
    beta = ((raw_core - raw_core.mean()) * centered_deg).sum() / denom
    core_score = raw_core - beta * centered_deg
    core_score = percentile_rank(core_score)
    return {
        "raw": h0,
        "prop1": h1,
        "prop2": h2,
        "residual": residual,
        "prop_residual_concat": F.normalize(torch.cat([h2, residual], dim=1), dim=1),
    }, {
        "core_score": core_score,
        "raw_core_score": raw_core,
        "depth_agreement": depth_agreement,
        "neighbor_consistency": neighbor_consistency,
        "residual_norm": residual_norm,
        "degree": deg,
    }


def bucketize(core_score):
    q1 = torch.quantile(core_score, 1.0 / 3.0)
    q2 = torch.quantile(core_score, 2.0 / 3.0)
    buckets = torch.full_like(core_score, 1, dtype=torch.long)
    buckets[core_score <= q1] = 0
    buckets[core_score >= q2] = 2
    return buckets


def fit_eval_probe(x, y, train_mask, val_mask, test_mask):
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    train = train_mask.numpy()
    val = val_mask.numpy()
    test = test_mask.numpy()
    best = None
    for c in CS:
        clf = LogisticRegression(
            C=c,
            max_iter=3000,
            solver="lbfgs",
            random_state=0,
        )
        clf.fit(x_np[train], y_np[train])
        val_acc = accuracy_score(y_np[val], clf.predict(x_np[val]))
        if best is None or val_acc > best["val_accuracy"]:
            best = {"clf": clf, "C": c, "val_accuracy": val_acc}
    pred = best["clf"].predict(x_np)
    return {
        "best_c": best["C"],
        "val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(accuracy_score(y_np[test], pred[test])),
        "pred": torch.as_tensor(pred),
    }


@torch.no_grad()
def topk_positive_label_agreement(signature, labels, topk, chunk_size):
    signature = F.normalize(signature.float(), dim=1)
    labels = labels.cpu()
    n = signature.size(0)
    topk = min(int(topk), n - 1)
    agreement = torch.zeros(n, dtype=torch.float32)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        sim = signature[start:end] @ signature.t()
        rows = torch.arange(end - start)
        cols = torch.arange(start, end)
        sim[rows, cols] = -2.0
        idx = torch.topk(sim, k=topk, dim=1).indices.cpu()
        agreement[start:end] = (labels[idx] == labels[start:end].view(-1, 1)).float().mean(dim=1)
    return agreement


@torch.no_grad()
def edge_label_agreement_by_node(edge_index, labels):
    src, dst = edge_index.cpu()
    labels = labels.cpu()
    same = (labels[src] == labels[dst]).float()
    out = torch.zeros(labels.numel(), dtype=torch.float32)
    cnt = torch.zeros(labels.numel(), dtype=torch.float32)
    out.index_add_(0, dst, same)
    cnt.index_add_(0, dst, torch.ones_like(same))
    return out / cnt.clamp_min(1.0)


def bucket_mean(values, mask):
    if int(mask.sum().item()) == 0:
        return ""
    return float(values[mask].float().mean().item())


def analyze_one(dataset_name, split_index, args):
    dataset = load_dataset(args.data_root, dataset_name)
    data = dataset[0].cpu()
    labels = data.y.cpu()
    train_mask, val_mask, test_mask = stratified_split(
        labels,
        split_index,
        args.train_ratio,
        args.val_ratio,
        args.split_base_seed,
    )
    views, signals = compute_views_and_core(data)
    buckets = bucketize(signals["core_score"])
    topk_agree = topk_positive_label_agreement(views["prop2"], labels, args.topk, args.chunk_size)
    edge_agree = edge_label_agreement_by_node(data.edge_index, labels)

    probe_results = {
        name: fit_eval_probe(view, labels, train_mask, val_mask, test_mask)
        for name, view in views.items()
    }

    run_rows = []
    for view_name, result in probe_results.items():
        pred = result["pred"]
        error = (pred != labels).float()
        row = {
            "dataset": dataset_name,
            "split_index": split_index,
            "split_seed": int(args.split_base_seed) + int(split_index),
            "view": view_name,
            "best_c": result["best_c"],
            "val_accuracy": result["val_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "test_error": 1.0 - result["test_accuracy"],
        }
        for bucket_id, bucket_name in enumerate(["low_core", "mid_core", "high_core"]):
            mask = (buckets == bucket_id) & test_mask
            row[f"{bucket_name}_test_accuracy"] = (
                "" if int(mask.sum().item()) == 0 else float((pred[mask] == labels[mask]).float().mean().item())
            )
            row[f"{bucket_name}_test_error"] = (
                "" if int(mask.sum().item()) == 0 else float(error[mask].mean().item())
            )
        run_rows.append(row)

    bucket_rows = []
    raw_error = (probe_results["raw"]["pred"] != labels).float()
    prop2_error = (probe_results["prop2"]["pred"] != labels).float()
    concat_error = (probe_results["prop_residual_concat"]["pred"] != labels).float()
    for bucket_id, bucket_name in enumerate(["low_core", "mid_core", "high_core"]):
        all_mask = buckets == bucket_id
        test_bucket = all_mask & test_mask
        bucket_rows.append({
            "dataset": dataset_name,
            "split_index": split_index,
            "bucket": bucket_name,
            "num_nodes": int(all_mask.sum().item()),
            "num_test": int(test_bucket.sum().item()),
            "core_score_mean": bucket_mean(signals["core_score"], all_mask),
            "degree_mean": bucket_mean(signals["degree"], all_mask),
            "depth_agreement_mean": bucket_mean(signals["depth_agreement"], all_mask),
            "neighbor_consistency_mean": bucket_mean(signals["neighbor_consistency"], all_mask),
            "residual_norm_mean": bucket_mean(signals["residual_norm"], all_mask),
            "topk_positive_label_agreement": bucket_mean(topk_agree, all_mask),
            "edge_label_agreement": bucket_mean(edge_agree, all_mask),
            "raw_test_error": bucket_mean(raw_error, test_bucket),
            "prop2_test_error": bucket_mean(prop2_error, test_bucket),
            "concat_test_error": bucket_mean(concat_error, test_bucket),
            "prop2_minus_raw_error": (
                "" if int(test_bucket.sum().item()) == 0 else bucket_mean(prop2_error - raw_error, test_bucket)
            ),
            "concat_minus_raw_error": (
                "" if int(test_bucket.sum().item()) == 0 else bucket_mean(concat_error - raw_error, test_bucket)
            ),
        })

    summary = {
        "dataset": dataset_name,
        "split_index": split_index,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(dataset.num_features),
        "num_classes": int(dataset.num_classes),
        "train_ratio": float(train_mask.float().mean().item()),
        "val_ratio": float(val_mask.float().mean().item()),
        "test_ratio": float(test_mask.float().mean().item()),
        "core_degree_corr": pearson(signals["core_score"], torch.log1p(signals["degree"])),
        "core_topk_label_agreement_corr": pearson(signals["core_score"], topk_agree),
        "core_edge_label_agreement_corr": pearson(signals["core_score"], edge_agree),
        "raw_accuracy": probe_results["raw"]["test_accuracy"],
        "prop1_accuracy": probe_results["prop1"]["test_accuracy"],
        "prop2_accuracy": probe_results["prop2"]["test_accuracy"],
        "residual_accuracy": probe_results["residual"]["test_accuracy"],
        "prop_residual_concat_accuracy": probe_results["prop_residual_concat"]["test_accuracy"],
    }
    for bucket_row in bucket_rows:
        name = bucket_row["bucket"]
        summary[f"{name}_topk_positive_label_agreement"] = bucket_row["topk_positive_label_agreement"]
        summary[f"{name}_edge_label_agreement"] = bucket_row["edge_label_agreement"]
        summary[f"{name}_raw_test_error"] = bucket_row["raw_test_error"]
        summary[f"{name}_prop2_test_error"] = bucket_row["prop2_test_error"]
    summary["high_minus_low_topk_agreement"] = (
        float(summary["high_core_topk_positive_label_agreement"])
        - float(summary["low_core_topk_positive_label_agreement"])
    )
    summary["high_minus_low_raw_error"] = (
        float(summary["high_core_raw_test_error"]) - float(summary["low_core_raw_test_error"])
    )
    summary["high_minus_low_prop2_error"] = (
        float(summary["high_core_prop2_test_error"]) - float(summary["low_core_prop2_test_error"])
    )
    return run_rows, bucket_rows, summary


def aggregate_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    out = []
    fields = [
        "core_degree_corr",
        "core_topk_label_agreement_corr",
        "core_edge_label_agreement_corr",
        "raw_accuracy",
        "prop1_accuracy",
        "prop2_accuracy",
        "residual_accuracy",
        "prop_residual_concat_accuracy",
        "high_minus_low_topk_agreement",
        "high_minus_low_raw_error",
        "high_minus_low_prop2_error",
    ]
    for dataset, items in sorted(grouped.items()):
        row = {"dataset": dataset, "num_splits": len(items)}
        for field in fields:
            vals = [float(item[field]) for item in items]
            row[f"{field}_mean"] = sum(vals) / len(vals)
        out.append(row)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    run_rows = []
    bucket_rows = []
    summary_rows = []
    for dataset in args.datasets:
        for split_index in args.splits:
            print(json.dumps({"dataset": dataset, "split_index": split_index, "status": "start"}))
            runs, buckets, summary = analyze_one(dataset, split_index, args)
            run_rows.extend(runs)
            bucket_rows.extend(buckets)
            summary_rows.append(summary)
            append_csv(out_dir / "core_boundary_summary_incremental.csv", summary)
            print(json.dumps({"dataset": dataset, "split_index": split_index, "status": "done"}))

    write_csv(out_dir / "core_boundary_view_results.csv", run_rows)
    write_csv(out_dir / "core_boundary_buckets.csv", bucket_rows)
    write_csv(out_dir / "core_boundary_summary.csv", summary_rows)
    aggregate = aggregate_summary(summary_rows)
    write_csv(out_dir / "core_boundary_aggregate.csv", aggregate)
    print(json.dumps({"runs": len(summary_rows), "datasets": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
