import argparse
import csv
import json
import os.path as osp
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from train import get_dataset, get_split_masks, should_use_mask_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[])
    parser.add_argument('--runs-dir', action='append', default=[])
    parser.add_argument('--out',
                        default='runs/summaries/propagation_calibration.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/propagation_calibration_aggregate.csv')
    parser.add_argument('--eval-mode', choices=['auto', 'mask'], default='auto')
    parser.add_argument('--include-methods', nargs='*', default=None)
    parser.add_argument('--include-datasets', nargs='*', default=None)
    parser.add_argument('--split-indices', nargs='*', type=int, default=None)
    parser.add_argument('--max-hop', type=int, default=2)
    parser.add_argument('--modes', nargs='*', default=[
        'ssl',
        'prop1',
        'ssl_prop1',
        'ssl_resid1',
        'prop2',
        'ssl_prop2',
        'ssl_resid2',
    ])
    parser.add_argument('--c-values', nargs='*', type=float,
                        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    parser.add_argument('--max-iter', type=int, default=1000)
    return parser.parse_args()


def find_artifacts(args):
    paths = []
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        artifact = run_path / 'artifacts.pt'
        if artifact.exists():
            paths.append(artifact)
        elif run_path.name == 'artifacts.pt' and run_path.exists():
            paths.append(run_path)
        else:
            raise FileNotFoundError(f'No artifacts.pt found at {run_dir}')
    for runs_dir in args.runs_dir:
        paths.extend(sorted(Path(runs_dir).rglob('artifacts.pt')))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_metadata(artifact_path, artifact):
    metadata_path = artifact_path.parent / 'metadata.json'
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    artifact_args = artifact.get('args') or {}
    metadata_args = metadata.get('args') or {}
    merged_args = dict(metadata_args)
    merged_args.update(artifact_args)
    return metadata, merged_args


def row_propagate(x, edge_index):
    source, target = edge_index
    aggregate = torch.zeros_like(x)
    degree = torch.zeros(x.size(0), dtype=x.dtype)
    aggregate.index_add_(0, target, x[source])
    degree.index_add_(0, target.cpu(), torch.ones(target.numel(), dtype=x.dtype))
    aggregate = aggregate + x
    degree = degree + 1.0
    return aggregate / degree.clamp_min(1.0).view(-1, 1)


def block_l2(tensor):
    return normalize(tensor.detach().cpu().numpy(), norm='l2')


def build_variants(embeddings, edge_index, max_hop):
    variants = {'ssl': embeddings}
    current = embeddings
    for hop in range(1, max_hop + 1):
        propagated = row_propagate(current, edge_index)
        residual = current - propagated
        variants[f'prop{hop}'] = propagated
        variants[f'resid{hop}'] = residual
        variants[f'ssl_prop{hop}'] = torch.cat([
            F.normalize(embeddings, dim=1),
            F.normalize(propagated, dim=1),
        ], dim=1)
        variants[f'ssl_resid{hop}'] = torch.cat([
            F.normalize(embeddings, dim=1),
            F.normalize(residual, dim=1),
        ], dim=1)
        current = propagated
    return variants


def fit_logreg(x_train, y_train, c_value, max_iter):
    clf = LogisticRegression(
        solver='lbfgs',
        C=float(c_value),
        max_iter=max_iter,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', FutureWarning)
        clf.fit(x_train, y_train)
    return clf


def evaluate_with_masks(features, labels, train_mask, val_mask, test_mask, args):
    y = labels.detach().cpu().numpy()
    train_mask = train_mask.detach().cpu().numpy().astype(bool)
    val_mask = val_mask.detach().cpu().numpy().astype(bool)
    test_mask = test_mask.detach().cpu().numpy().astype(bool)
    best_c = args.c_values[0]
    best_score = -1.0
    for c_value in args.c_values:
        clf = fit_logreg(features[train_mask], y[train_mask], c_value, args.max_iter)
        val_pred = clf.predict(features[val_mask])
        score = f1_score(y[val_mask], val_pred, average='micro', zero_division=0)
        if score > best_score:
            best_score = score
            best_c = c_value
    clf = fit_logreg(features[train_mask], y[train_mask], best_c, args.max_iter)
    pred = clf.predict(features[test_mask])
    return {
        'F1Mi': f1_score(y[test_mask], pred, average='micro', zero_division=0),
        'F1Ma': f1_score(y[test_mask], pred, average='macro', zero_division=0),
        'best_c': float(best_c),
        'val_micro': float(best_score),
    }


def rows_from_artifact(artifact_path, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
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

    dataset = get_dataset(osp.join(osp.expanduser('~'), 'datasets', dataset_name), dataset_name)
    data = dataset[0]
    eval_args = SimpleNamespace(dataset=dataset_name, eval_mode=args.eval_mode)
    if not should_use_mask_eval(eval_args, data):
        raise ValueError('evaluate_propagation_calibration currently expects mask eval.')
    split_indices = args.split_indices if args.split_indices is not None else [
        artifact_split_index
    ]
    embeddings = artifact['embeddings'].detach().cpu().float()
    variants = build_variants(embeddings, data.edge_index, args.max_hop)

    rows = []
    for split_index in split_indices:
        train_mask, val_mask, test_mask = get_split_masks(data, split_index)
        for mode in args.modes:
            if mode not in variants:
                continue
            features = block_l2(variants[mode])
            metrics = evaluate_with_masks(
                features,
                data.y.detach().cpu(),
                train_mask,
                val_mask,
                test_mask,
                args,
            )
            rows.append({
                'artifact_path': str(artifact_path),
                'run_dir': str(artifact_path.parent),
                'dataset': dataset_name,
                'method': method,
                'seed': seed,
                'split_index': split_index,
                'mode': mode,
                'num_nodes': int(embeddings.size(0)),
                'embedding_dim': int(embeddings.size(1)),
                'feature_dim': int(features.shape[1]),
                'F1Mi': metrics['F1Mi'],
                'F1Ma': metrics['F1Ma'],
                'best_c': metrics['best_c'],
                'val_micro': metrics['val_micro'],
            })
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['method'], row['mode']), []).append(row)
    out = []
    ssl_lookup = {
        (row['dataset'], row['method'], row['seed'], row['split_index']): row
        for row in rows
        if row['mode'] == 'ssl'
    }
    for (dataset, method, mode), group in sorted(groups.items()):
        f1mi = np.array([float(row['F1Mi']) for row in group])
        f1ma = np.array([float(row['F1Ma']) for row in group])
        deltas_mi = []
        deltas_ma = []
        for row in group:
            key = (row['dataset'], row['method'], row['seed'], row['split_index'])
            base = ssl_lookup.get(key)
            if base is not None:
                deltas_mi.append(float(row['F1Mi']) - float(base['F1Mi']))
                deltas_ma.append(float(row['F1Ma']) - float(base['F1Ma']))
        out.append({
            'dataset': dataset,
            'method': method,
            'mode': mode,
            'num_runs': len(group),
            'F1Mi_mean': float(f1mi.mean()),
            'F1Mi_std': float(f1mi.std()),
            'F1Ma_mean': float(f1ma.mean()),
            'F1Ma_std': float(f1ma.std()),
            'delta_ssl_F1Mi_mean': float(np.mean(deltas_mi)) if deltas_mi else '',
            'delta_ssl_F1Ma_mean': float(np.mean(deltas_ma)) if deltas_ma else '',
            'delta_ssl_F1Mi_positive': int(sum(delta > 0 for delta in deltas_mi)),
            'delta_ssl_F1Mi_negative': int(sum(delta < 0 for delta in deltas_mi)),
        })
    return out


def main():
    args = parse_args()
    rows = []
    for artifact_path in find_artifacts(args):
        rows.extend(rows_from_artifact(artifact_path, args))
    fieldnames = [
        'artifact_path',
        'run_dir',
        'dataset',
        'method',
        'seed',
        'split_index',
        'mode',
        'num_nodes',
        'embedding_dim',
        'feature_dim',
        'F1Mi',
        'F1Ma',
        'best_c',
        'val_micro',
    ]
    write_csv(args.out, rows, fieldnames)
    aggregate_rows = aggregate(rows)
    aggregate_fieldnames = [
        'dataset',
        'method',
        'mode',
        'num_runs',
        'F1Mi_mean',
        'F1Mi_std',
        'F1Ma_mean',
        'F1Ma_std',
        'delta_ssl_F1Mi_mean',
        'delta_ssl_F1Ma_mean',
        'delta_ssl_F1Mi_positive',
        'delta_ssl_F1Mi_negative',
    ]
    write_csv(args.aggregate_out, aggregate_rows, aggregate_fieldnames)
    print(
        f'(I) | artifacts={len(find_artifacts(args))}, rows={len(rows)}, '
        f'out={args.out}, aggregate={args.aggregate_out}'
    )


if __name__ == '__main__':
    main()
