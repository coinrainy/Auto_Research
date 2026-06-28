import argparse
import csv
import os.path as osp
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from evaluate_propagation_calibration import (
    block_l2,
    build_variants,
    find_artifacts,
    fit_logreg,
    load_metadata,
)
from train import get_dataset, get_split_masks, should_use_mask_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[])
    parser.add_argument('--runs-dir', action='append', default=[])
    parser.add_argument('--out',
                        default='runs/summaries/sparc_mode_proxy.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/sparc_mode_proxy_aggregate.csv')
    parser.add_argument('--eval-mode', choices=['auto', 'mask'], default='auto')
    parser.add_argument('--include-methods', nargs='*', default=None)
    parser.add_argument('--include-datasets', nargs='*', default=None)
    parser.add_argument('--split-indices', nargs='*', type=int, default=None)
    parser.add_argument('--max-hop', type=int, default=2)
    parser.add_argument('--modes', nargs='*', default=[
        'ssl',
        'ssl_prop1',
        'ssl_prop2',
        'ssl_resid1',
        'ssl_resid2',
    ])
    parser.add_argument('--c-values', nargs='*', type=float,
                        default=[4.0, 16.0, 64.0])
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--proxy-min-effective-rank', type=float, default=5.0)
    parser.add_argument('--proxy-rank-weight', type=float, default=0.25)
    parser.add_argument('--proxy-anisotropy-weight', type=float, default=0.25)
    parser.add_argument('--proxy-ssl-corr-weight', type=float, default=0.15)
    parser.add_argument('--adaptive-feature-contrast-threshold', type=float,
                        default=0.0)
    parser.add_argument('--max-metric-edges', type=int, default=60000)
    parser.add_argument('--max-metric-pairs', type=int, default=60000)
    return parser.parse_args()


def l2_tensor(features):
    return F.normalize(features.detach().cpu().float(), dim=1)


def sample_indices(num_items, max_items, seed):
    if num_items <= max_items:
        return torch.arange(num_items)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_items, generator=generator)[:max_items]


def sampled_pairs(num_nodes, max_pairs, seed):
    generator = torch.Generator().manual_seed(seed)
    pair_count = min(max_pairs, max(1, num_nodes * 10))
    src = torch.randint(num_nodes, (pair_count,), generator=generator)
    dst = torch.randint(num_nodes, (pair_count,), generator=generator)
    return src, dst


def effective_rank(features):
    z = l2_tensor(features)
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.t().matmul(z) / max(1, z.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum().clamp_min(1e-12)
    probs = eigvals / total
    nonzero = probs > 0
    entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
    participation = total.pow(2) / eigvals.pow(2).sum().clamp_min(1e-12)
    anisotropy = eigvals.max() / total
    return {
        'effective_rank': float(torch.exp(entropy).item()),
        'participation_ratio': float(participation.item()),
        'anisotropy': float(anisotropy.item()),
    }


def graph_contrast(features, edge_index, max_edges, max_pairs):
    z = l2_tensor(features)
    src, dst = edge_index.detach().cpu()
    edge_idx = sample_indices(src.numel(), max_edges, seed=17)
    src = src[edge_idx]
    dst = dst[edge_idx]
    edge_cos = (z[src] * z[dst]).sum(dim=1).mean()
    pair_src, pair_dst = sampled_pairs(z.size(0), max_pairs, seed=29)
    random_cos = (z[pair_src] * z[pair_dst]).sum(dim=1).mean()
    return {
        'edge_cosine_mean': float(edge_cos.item()),
        'random_cosine_mean': float(random_cos.item()),
        'edge_random_contrast': float((edge_cos - random_cos).item()),
    }


def pair_similarity_correlation(features, reference, max_pairs):
    z = l2_tensor(features)
    ref = l2_tensor(reference)
    pair_src, pair_dst = sampled_pairs(z.size(0), max_pairs, seed=43)
    z_sim = (z[pair_src] * z[pair_dst]).sum(dim=1).numpy()
    ref_sim = (ref[pair_src] * ref[pair_dst]).sum(dim=1).numpy()
    if z_sim.std() < 1e-12 or ref_sim.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(z_sim, ref_sim)[0, 1])


def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def mode_priority(mode):
    priorities = {
        'ssl': 0,
        'ssl_prop1': 1,
        'ssl_resid1': 2,
        'ssl_prop2': 3,
        'ssl_resid2': 4,
    }
    return priorities.get(mode, -1)


def candidate_proxy_metrics(data, variants, modes, args, feature_graph):
    metrics = {}
    reference = variants['ssl']
    for mode in modes:
        features = variants[mode]
        record = {}
        record.update(effective_rank(features))
        record.update(graph_contrast(
            features,
            data.edge_index,
            args.max_metric_edges,
            args.max_metric_pairs,
        ))
        record['ssl_similarity_correlation'] = pair_similarity_correlation(
            features,
            reference,
            args.max_metric_pairs,
        )
        record['feature_edge_cosine_mean'] = feature_graph['edge_cosine_mean']
        record['feature_random_cosine_mean'] = feature_graph['random_cosine_mean']
        record['feature_edge_random_contrast'] = feature_graph[
            'edge_random_contrast'
        ]
        record['proxy_eligible'] = (
            record['effective_rank'] >= args.proxy_min_effective_rank
        )
        metrics[mode] = record

    edge_scores = zscore([
        metrics[mode]['edge_random_contrast'] for mode in modes
    ])
    rank_scores = zscore([
        np.log(max(metrics[mode]['effective_rank'], 1e-12)) for mode in modes
    ])
    anisotropy_scores = zscore([
        metrics[mode]['anisotropy'] for mode in modes
    ])
    ssl_corr_scores = zscore([
        metrics[mode]['ssl_similarity_correlation'] for mode in modes
    ])
    for idx, mode in enumerate(modes):
        metrics[mode]['proxy_score'] = float(
            edge_scores[idx]
            + args.proxy_rank_weight * rank_scores[idx]
            - args.proxy_anisotropy_weight * anisotropy_scores[idx]
            + args.proxy_ssl_corr_weight * ssl_corr_scores[idx]
        )
        metrics[mode]['edge_contrast_z'] = float(edge_scores[idx])
        metrics[mode]['effective_rank_z'] = float(rank_scores[idx])
        metrics[mode]['anisotropy_z'] = float(anisotropy_scores[idx])
        metrics[mode]['ssl_similarity_correlation_z'] = float(ssl_corr_scores[idx])
    return metrics


def select_by_metric(metrics, modes, metric, reverse=False):
    eligible = [
        mode for mode in modes
        if metrics[mode].get('proxy_eligible', True)
    ]
    choices = eligible if eligible else list(modes)
    sign = -1.0 if reverse else 1.0
    return max(
        choices,
        key=lambda mode: (
            sign * float(metrics[mode][metric]),
            mode_priority(mode),
        ),
    )


def evaluate_features(features, labels, train_mask, val_mask, test_mask,
                      c_values, max_iter):
    y = labels.detach().cpu().numpy()
    train = train_mask.detach().cpu().numpy().astype(bool)
    val = val_mask.detach().cpu().numpy().astype(bool)
    test = test_mask.detach().cpu().numpy().astype(bool)
    best_c = c_values[0]
    best_score = -1.0
    for c_value in c_values:
        clf = fit_logreg(features[train], y[train], c_value, max_iter)
        pred = clf.predict(features[val])
        score = f1_score(y[val], pred, average='micro', zero_division=0)
        if score > best_score:
            best_score = score
            best_c = c_value
    clf = fit_logreg(features[train], y[train], best_c, max_iter)
    pred = clf.predict(features[test])
    return {
        'best_c': float(best_c),
        'val_micro': float(best_score),
        'test_micro': f1_score(y[test], pred, average='micro', zero_division=0),
        'test_macro': f1_score(y[test], pred, average='macro', zero_division=0),
    }


def prefix_row(artifact_path, dataset_name, method, seed, split_index):
    return {
        'artifact_path': str(artifact_path),
        'run_dir': str(artifact_path.parent),
        'dataset': dataset_name,
        'method': method,
        'seed': seed,
        'split_index': split_index,
    }


def selected_row(prefix, status, mode, scored, metrics):
    score = scored[mode]
    metric = metrics[mode]
    return {
        **prefix,
        'status': status,
        'mode': mode,
        'selected_mode': mode,
        **score,
        **metric,
    }


def candidate_row(prefix, mode, scored, metrics):
    return {
        **prefix,
        'status': 'candidate',
        'mode': mode,
        'selected_mode': '',
        **scored[mode],
        **metrics[mode],
    }


def rows_from_artifact(artifact_path, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        artifact = torch.load(artifact_path, map_location='cpu')
    metadata, artifact_args = load_metadata(artifact_path, artifact)
    dataset_name = artifact_args.get('dataset') or metadata.get('dataset')
    method = artifact_args.get('method') or metadata.get('method')
    seed = int(artifact_args.get('resolved_seed', artifact_args.get('seed', 0)) or 0)
    artifact_split_index = int(
        artifact_args.get('split_index', metadata.get('split_index', 0))
    )
    if dataset_name is None or method is None:
        raise ValueError(f'Missing dataset/method metadata in {artifact_path}')
    if args.include_methods and method not in args.include_methods:
        return []
    if args.include_datasets and dataset_name not in args.include_datasets:
        return []

    dataset = get_dataset(osp.join(osp.expanduser('~'), 'datasets', dataset_name),
                          dataset_name)
    data = dataset[0]
    eval_args = SimpleNamespace(dataset=dataset_name, eval_mode=args.eval_mode)
    if not should_use_mask_eval(eval_args, data):
        raise ValueError('SPARC mode proxy currently expects mask eval.')
    split_indices = args.split_indices if args.split_indices is not None else [
        artifact_split_index
    ]
    embeddings = artifact['embeddings'].detach().cpu().float()
    variants = build_variants(embeddings, data.edge_index, args.max_hop)
    modes = [mode for mode in args.modes if mode in variants]
    if 'ssl' not in modes:
        modes = ['ssl'] + modes
    missing = [mode for mode in args.modes if mode not in variants]
    if missing:
        raise ValueError(f'Missing requested modes for {artifact_path}: {missing}')
    feature_graph = graph_contrast(
        data.x,
        data.edge_index,
        args.max_metric_edges,
        args.max_metric_pairs,
    )
    metrics = candidate_proxy_metrics(data, variants, modes, args, feature_graph)
    selected_proxy = select_by_metric(metrics, modes, 'proxy_score')
    selected_contrast = select_by_metric(metrics, modes, 'edge_random_contrast')
    selected_rank = select_by_metric(metrics, modes, 'effective_rank')
    selected_low_anisotropy = select_by_metric(
        metrics,
        modes,
        'anisotropy',
        reverse=True,
    )
    if (
        feature_graph['edge_random_contrast']
        < args.adaptive_feature_contrast_threshold
    ):
        selected_adaptive = selected_rank
    else:
        selected_adaptive = selected_contrast

    features = {mode: block_l2(variants[mode]) for mode in modes}
    rows = []
    for split_index in split_indices:
        train_mask, val_mask, test_mask = get_split_masks(data, split_index)
        prefix = prefix_row(artifact_path, dataset_name, method, seed, split_index)
        scored = {}
        for mode in modes:
            scored[mode] = evaluate_features(
                features[mode],
                data.y.detach().cpu(),
                train_mask,
                val_mask,
                test_mask,
                args.c_values,
                args.max_iter,
            )
            rows.append(candidate_row(prefix, mode, scored, metrics))
        selected_validation = max(
            modes,
            key=lambda mode: (scored[mode]['val_micro'], mode_priority(mode)),
        )
        selected_oracle = max(
            modes,
            key=lambda mode: (scored[mode]['test_micro'], mode_priority(mode)),
        )
        selections = {
            'selected_proxy': selected_proxy,
            'selected_feature_adaptive': selected_adaptive,
            'selected_edge_contrast': selected_contrast,
            'selected_effective_rank': selected_rank,
            'selected_low_anisotropy': selected_low_anisotropy,
            'selected_validation': selected_validation,
            'selected_oracle': selected_oracle,
            'selected_ssl': 'ssl',
        }
        for status, mode in selections.items():
            if mode in scored:
                rows.append(selected_row(prefix, status, mode, scored, metrics))
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    selected = [row for row in rows if row.get('status', '').startswith('selected_')]
    groups = defaultdict(list)
    for row in selected:
        groups[(row['dataset'], row['method'], row['status'])].append(row)
    out = []
    for (dataset, method, status), group in sorted(groups.items()):
        rec = {
            'dataset': dataset,
            'method': method,
            'status': status,
            'num_runs': len(group),
        }
        for metric in [
            'test_micro',
            'test_macro',
            'val_micro',
            'proxy_score',
            'effective_rank',
            'participation_ratio',
            'anisotropy',
            'edge_cosine_mean',
            'random_cosine_mean',
            'edge_random_contrast',
            'ssl_similarity_correlation',
            'feature_edge_cosine_mean',
            'feature_random_cosine_mean',
            'feature_edge_random_contrast',
        ]:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            rec[f'{metric}_mean'] = float(values.mean())
            rec[f'{metric}_std'] = float(values.std())
        counts = defaultdict(int)
        for row in group:
            counts[row['selected_mode']] += 1
        rec['selected_counts'] = ';'.join(
            f'{key}:{counts[key]}' for key in sorted(counts)
        )
        out.append(rec)
    return out


def main():
    args = parse_args()
    rows = []
    for artifact_path in find_artifacts(args):
        rows.extend(rows_from_artifact(artifact_path, args))
    if not rows:
        raise SystemExit('No rows produced.')
    fieldnames = [
        'artifact_path',
        'run_dir',
        'dataset',
        'method',
        'seed',
        'split_index',
        'status',
        'mode',
        'selected_mode',
        'best_c',
        'val_micro',
        'test_micro',
        'test_macro',
        'proxy_score',
        'proxy_eligible',
        'edge_contrast_z',
        'effective_rank_z',
        'anisotropy_z',
        'ssl_similarity_correlation_z',
        'feature_edge_cosine_mean',
        'feature_random_cosine_mean',
        'feature_edge_random_contrast',
        'effective_rank',
        'participation_ratio',
        'anisotropy',
        'edge_cosine_mean',
        'random_cosine_mean',
        'edge_random_contrast',
        'ssl_similarity_correlation',
    ]
    write_csv(args.out, rows, fieldnames)
    summary = aggregate(rows)
    write_csv(
        args.aggregate_out,
        summary,
        [
            'dataset',
            'method',
            'status',
            'num_runs',
            'test_micro_mean',
            'test_micro_std',
            'test_macro_mean',
            'test_macro_std',
            'val_micro_mean',
            'val_micro_std',
            'proxy_score_mean',
            'proxy_score_std',
            'effective_rank_mean',
            'effective_rank_std',
            'participation_ratio_mean',
            'participation_ratio_std',
            'anisotropy_mean',
            'anisotropy_std',
            'edge_cosine_mean_mean',
            'edge_cosine_mean_std',
            'random_cosine_mean_mean',
            'random_cosine_mean_std',
            'edge_random_contrast_mean',
            'edge_random_contrast_std',
            'ssl_similarity_correlation_mean',
            'ssl_similarity_correlation_std',
            'feature_edge_cosine_mean_mean',
            'feature_edge_cosine_mean_std',
            'feature_random_cosine_mean_mean',
            'feature_random_cosine_mean_std',
            'feature_edge_random_contrast_mean',
            'feature_edge_random_contrast_std',
            'selected_counts',
        ],
    )
    selected_proxy = sum(1 for row in rows if row['status'] == 'selected_proxy')
    print(
        f'(I) | rows={len(rows)}, selected_proxy={selected_proxy}, '
        f'out={args.out}, aggregate={args.aggregate_out}'
    )


if __name__ == '__main__':
    main()
