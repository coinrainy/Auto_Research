import argparse
import csv
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch
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
    parser.add_argument('--out-prefix',
                        default='runs/summaries/sparc_diagnostics')
    parser.add_argument('--eval-mode', choices=['auto', 'mask'], default='auto')
    parser.add_argument('--include-methods', nargs='*', default=None)
    parser.add_argument('--include-datasets', nargs='*', default=None)
    parser.add_argument('--split-indices', nargs='*', type=int, default=None)
    parser.add_argument('--modes', nargs='*', default=['ssl_prop2', 'ssl_resid1'])
    parser.add_argument('--base-mode', default='ssl')
    parser.add_argument('--max-hop', type=int, default=2)
    parser.add_argument('--c-values', nargs='*', type=float, default=[4.0, 16.0, 64.0])
    parser.add_argument('--max-iter', type=int, default=500)
    return parser.parse_args()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fit_predict(features, labels, train_mask, val_mask, test_mask, c_values,
                max_iter):
    y = labels.detach().cpu().numpy()
    train_mask = train_mask.detach().cpu().numpy().astype(bool)
    val_mask = val_mask.detach().cpu().numpy().astype(bool)
    test_mask = test_mask.detach().cpu().numpy().astype(bool)
    best_c = c_values[0]
    best_score = -1.0
    for c_value in c_values:
        clf = fit_logreg(features[train_mask], y[train_mask], c_value, max_iter)
        val_pred = clf.predict(features[val_mask])
        score = f1_score(y[val_mask], val_pred, average='micro', zero_division=0)
        if score > best_score:
            best_score = score
            best_c = c_value
    clf = fit_logreg(features[train_mask], y[train_mask], best_c, max_iter)
    pred_all = clf.predict(features)
    return pred_all, float(best_c), float(best_score)


def node_degree(edge_index, num_nodes):
    source, target = edge_index.detach().cpu()
    degree = torch.zeros(num_nodes, dtype=torch.float)
    ones = torch.ones(source.numel(), dtype=torch.float)
    degree.index_add_(0, source, ones)
    degree.index_add_(0, target, ones)
    return degree.numpy()


def local_label_homophily(edge_index, labels, num_nodes):
    source, target = edge_index.detach().cpu()
    labels = labels.detach().cpu()
    same = (labels[source] == labels[target]).float()
    total = torch.zeros(num_nodes, dtype=torch.float)
    match = torch.zeros(num_nodes, dtype=torch.float)
    total.index_add_(0, source, torch.ones_like(same))
    total.index_add_(0, target, torch.ones_like(same))
    match.index_add_(0, source, same)
    match.index_add_(0, target, same)
    values = match / total.clamp_min(1.0)
    values[total == 0] = np.nan
    return values.numpy()


def rank_buckets(values):
    values = np.asarray(values, dtype=float)
    buckets = np.empty(values.shape[0], dtype=object)
    finite = np.isfinite(values)
    buckets[~finite] = 'missing'
    if finite.sum() == 0:
        return buckets
    order = np.argsort(values[finite], kind='mergesort')
    finite_indices = np.where(finite)[0][order]
    n = len(finite_indices)
    first = n // 3
    second = (2 * n) // 3
    buckets[finite_indices[:first]] = 'low'
    buckets[finite_indices[first:second]] = 'mid'
    buckets[finite_indices[second:]] = 'high'
    return buckets


def summarize_subset(row_prefix, indices, labels, base_correct, mode_correct):
    if len(indices) == 0:
        return None
    base = base_correct[indices]
    mode = mode_correct[indices]
    gains = np.logical_and(~base, mode)
    losses = np.logical_and(base, ~mode)
    out = dict(row_prefix)
    out.update({
        'num_test': int(len(indices)),
        'ssl_correct_rate': float(base.mean()),
        'mode_correct_rate': float(mode.mean()),
        'delta_correct_rate': float(mode.mean() - base.mean()),
        'gain_count': int(gains.sum()),
        'loss_count': int(losses.sum()),
        'net_gain_count': int(gains.sum() - losses.sum()),
        'label_entropy': label_entropy(labels[indices]),
        'majority_class_fraction': majority_fraction(labels[indices]),
    })
    return out


def label_entropy(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def majority_fraction(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def aggregate(rows, group_fields):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(group_fields, key)}
        numeric_fields = [
            'num_test',
            'ssl_correct_rate',
            'mode_correct_rate',
            'delta_correct_rate',
            'gain_count',
            'loss_count',
            'net_gain_count',
            'label_entropy',
            'majority_class_fraction',
        ]
        for field in numeric_fields:
            values = np.array([float(item[field]) for item in group], dtype=float)
            if field in ['num_test', 'gain_count', 'loss_count', 'net_gain_count']:
                row[f'{field}_sum'] = float(values.sum())
                row[f'{field}_mean'] = float(values.mean())
            else:
                row[f'{field}_mean'] = float(values.mean())
                row[f'{field}_std'] = float(values.std())
        row['num_splits'] = len(group)
        row['positive_delta_count'] = int(
            sum(float(item['delta_correct_rate']) > 0 for item in group)
        )
        row['negative_delta_count'] = int(
            sum(float(item['delta_correct_rate']) < 0 for item in group)
        )
        out.append(row)
    return out


def edge_homophily(edge_index, labels):
    source, target = edge_index.detach().cpu()
    labels = labels.detach().cpu()
    if source.numel() == 0:
        return 0.0
    return float((labels[source] == labels[target]).float().mean().item())


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
        return [], [], []
    if args.include_datasets and dataset_name not in args.include_datasets:
        return [], [], []

    dataset = get_dataset(osp.join(osp.expanduser('~'), 'datasets', dataset_name), dataset_name)
    data = dataset[0]
    eval_args = SimpleNamespace(dataset=dataset_name, eval_mode=args.eval_mode)
    if not should_use_mask_eval(eval_args, data):
        raise ValueError('SPARC diagnostics currently expects mask eval.')
    split_indices = args.split_indices if args.split_indices is not None else [
        artifact_split_index
    ]
    embeddings = artifact['embeddings'].detach().cpu().float()
    variants = build_variants(embeddings, data.edge_index, args.max_hop)
    required_modes = [args.base_mode] + [mode for mode in args.modes if mode != args.base_mode]
    missing = [mode for mode in required_modes if mode not in variants]
    if missing:
        raise ValueError(f'Missing modes for {artifact_path}: {missing}')

    labels = data.y.detach().cpu().numpy()
    degree = node_degree(data.edge_index, data.num_nodes)
    local_h = local_label_homophily(data.edge_index, data.y, data.num_nodes)
    bucket_values = {
        'degree': rank_buckets(degree),
        'local_homophily': rank_buckets(local_h),
    }
    graph_stats = {
        'edge_homophily': edge_homophily(data.edge_index, data.y),
        'degree_mean': float(np.mean(degree)),
        'degree_std': float(np.std(degree)),
        'local_homophily_mean': float(np.nanmean(local_h)),
        'local_homophily_std': float(np.nanstd(local_h)),
    }

    split_rows = []
    bucket_rows = []
    class_rows = []
    features = {mode: block_l2(variants[mode]) for mode in required_modes}
    for split_index in split_indices:
        train_mask, val_mask, test_mask = get_split_masks(data, split_index)
        test_mask_np = test_mask.detach().cpu().numpy().astype(bool)
        test_indices = np.where(test_mask_np)[0]
        predictions = {}
        probe_stats = {}
        for mode in required_modes:
            pred, best_c, val_micro = fit_predict(
                features[mode],
                data.y.detach().cpu(),
                train_mask,
                val_mask,
                test_mask,
                args.c_values,
                args.max_iter,
            )
            predictions[mode] = pred
            probe_stats[mode] = {'best_c': best_c, 'val_micro': val_micro}
        base_correct = predictions[args.base_mode] == labels
        for mode in args.modes:
            mode_correct = predictions[mode] == labels
            split_prefix = {
                'artifact_path': str(artifact_path),
                'dataset': dataset_name,
                'method': method,
                'seed': seed,
                'split_index': split_index,
                'mode': mode,
                'base_mode': args.base_mode,
                'best_c': probe_stats[mode]['best_c'],
                'base_best_c': probe_stats[args.base_mode]['best_c'],
                'val_micro': probe_stats[mode]['val_micro'],
                'base_val_micro': probe_stats[args.base_mode]['val_micro'],
                **graph_stats,
            }
            split_row = summarize_subset(
                split_prefix,
                test_indices,
                labels,
                base_correct,
                mode_correct,
            )
            split_rows.append(split_row)

            for bucket_type, buckets in bucket_values.items():
                for bucket in ['low', 'mid', 'high', 'missing']:
                    indices = test_indices[buckets[test_indices] == bucket]
                    row = summarize_subset(
                        {
                            **split_prefix,
                            'bucket_type': bucket_type,
                            'bucket': bucket,
                            'degree_mean_bucket': (
                                float(np.mean(degree[indices])) if len(indices) else ''
                            ),
                            'local_homophily_mean_bucket': (
                                float(np.nanmean(local_h[indices])) if len(indices) else ''
                            ),
                        },
                        indices,
                        labels,
                        base_correct,
                        mode_correct,
                    )
                    if row is not None:
                        bucket_rows.append(row)

            for class_id in sorted(np.unique(labels[test_indices]).tolist()):
                indices = test_indices[labels[test_indices] == class_id]
                row = summarize_subset(
                    {**split_prefix, 'class_id': int(class_id)},
                    indices,
                    labels,
                    base_correct,
                    mode_correct,
                )
                if row is not None:
                    class_rows.append(row)
    return split_rows, bucket_rows, class_rows


def main():
    args = parse_args()
    split_rows = []
    bucket_rows = []
    class_rows = []
    artifacts = find_artifacts(args)
    for artifact_path in artifacts:
        s_rows, b_rows, c_rows = rows_from_artifact(artifact_path, args)
        split_rows.extend(s_rows)
        bucket_rows.extend(b_rows)
        class_rows.extend(c_rows)

    out_prefix = Path(args.out_prefix)
    split_fields = [
        'artifact_path',
        'dataset',
        'method',
        'seed',
        'split_index',
        'mode',
        'base_mode',
        'best_c',
        'base_best_c',
        'val_micro',
        'base_val_micro',
        'edge_homophily',
        'degree_mean',
        'degree_std',
        'local_homophily_mean',
        'local_homophily_std',
        'num_test',
        'ssl_correct_rate',
        'mode_correct_rate',
        'delta_correct_rate',
        'gain_count',
        'loss_count',
        'net_gain_count',
        'label_entropy',
        'majority_class_fraction',
    ]
    bucket_fields = split_fields + [
        'bucket_type',
        'bucket',
        'degree_mean_bucket',
        'local_homophily_mean_bucket',
    ]
    class_fields = split_fields + ['class_id']
    write_csv(f'{out_prefix}_splits.csv', split_rows, split_fields)
    write_csv(f'{out_prefix}_buckets.csv', bucket_rows, bucket_fields)
    write_csv(f'{out_prefix}_classes.csv', class_rows, class_fields)

    split_agg = aggregate(split_rows, ['dataset', 'method', 'mode'])
    bucket_agg = aggregate(
        bucket_rows,
        ['dataset', 'method', 'mode', 'bucket_type', 'bucket'],
    )
    class_agg = aggregate(class_rows, ['dataset', 'method', 'mode', 'class_id'])
    write_csv(
        f'{out_prefix}_splits_aggregate.csv',
        split_agg,
        list(split_agg[0].keys()) if split_agg else ['dataset', 'method', 'mode'],
    )
    write_csv(
        f'{out_prefix}_buckets_aggregate.csv',
        bucket_agg,
        list(bucket_agg[0].keys()) if bucket_agg else [
            'dataset', 'method', 'mode', 'bucket_type', 'bucket'
        ],
    )
    write_csv(
        f'{out_prefix}_classes_aggregate.csv',
        class_agg,
        list(class_agg[0].keys()) if class_agg else [
            'dataset', 'method', 'mode', 'class_id'
        ],
    )
    print(
        f'(I) | artifacts={len(artifacts)}, split_rows={len(split_rows)}, '
        f'bucket_rows={len(bucket_rows)}, class_rows={len(class_rows)}, '
        f'out_prefix={out_prefix}'
    )


if __name__ == '__main__':
    main()
