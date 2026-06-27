import argparse
import csv
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize

from train import get_dataset, get_split_masks, should_use_mask_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[],
                        help='Run directory containing artifacts.pt. Can be repeated.')
    parser.add_argument('--runs-dir', action='append', default=[],
                        help='Directory to scan recursively for artifacts.pt. Can be repeated.')
    parser.add_argument('--out', default='runs/summaries/feature_fusion_eval_runs.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/feature_fusion_eval_aggregate.csv')
    parser.add_argument('--eval-mode', choices=['auto', 'mask', 'random'], default='auto')
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    parser.add_argument('--include-methods', nargs='*', default=None)
    parser.add_argument('--include-datasets', nargs='*', default=None)
    parser.add_argument('--modes', nargs='*', default=['raw', 'ssl', 'concat'],
                        choices=['raw', 'ssl', 'concat'])
    parser.add_argument('--solver', default='liblinear',
                        choices=['liblinear', 'lbfgs', 'saga'])
    parser.add_argument('--max-iter', type=int, default=3000)
    parser.add_argument('--c-min-power', type=int, default=-8)
    parser.add_argument('--c-max-power', type=int, default=8)
    parser.add_argument('--final-l2-normalize', action='store_true',
                        help='Apply an additional L2 normalization after block concat.')
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


def block_l2(x):
    return normalize(x.detach().cpu().numpy(), norm='l2')


def build_features(mode, raw_features, ssl_embeddings, final_l2_normalize):
    if mode == 'raw':
        return block_l2(raw_features)
    if mode == 'ssl':
        return block_l2(ssl_embeddings)
    if mode == 'concat':
        features = np.concatenate([block_l2(raw_features), block_l2(ssl_embeddings)], axis=1)
        if final_l2_normalize:
            features = normalize(features, norm='l2')
        return features
    raise ValueError(f'Unsupported mode: {mode}')


def fit_logreg(x_train, y_train, c_value, solver, max_iter):
    estimator = LogisticRegression(
        solver=solver,
        C=float(c_value),
        max_iter=max_iter,
        n_jobs=None if solver == 'liblinear' else 4,
        random_state=0,
    )
    clf = OneVsRestClassifier(estimator) if solver == 'liblinear' else estimator
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
    c_values = 2.0 ** np.arange(args.c_min_power, args.c_max_power)

    best_c = c_values[0]
    best_score = -1.0
    for c_value in c_values:
        clf = fit_logreg(
            features[train_mask],
            y[train_mask],
            c_value,
            args.solver,
            args.max_iter,
        )
        y_val_pred = clf.predict(features[val_mask])
        score = f1_score(y[val_mask], y_val_pred, average='micro', zero_division=0)
        if score > best_score:
            best_score = score
            best_c = c_value

    clf = fit_logreg(
        features[train_mask],
        y[train_mask],
        best_c,
        args.solver,
        args.max_iter,
    )
    y_pred = clf.predict(features[test_mask])
    return {
        'F1Mi': f1_score(y[test_mask], y_pred, average='micro', zero_division=0),
        'F1Ma': f1_score(y[test_mask], y_pred, average='macro', zero_division=0),
        'best_c': float(best_c),
        'val_micro': float(best_score),
    }


def evaluate_random_split(features, labels, args, seed):
    y = labels.detach().cpu().numpy()
    scores = []
    for repeat_idx in range(3):
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            y,
            test_size=1 - args.eval_ratio,
            random_state=seed + repeat_idx,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        clf = fit_logreg(x_train, y_train, 1.0, args.solver, args.max_iter)
        y_pred = clf.predict(x_test)
        scores.append({
            'F1Mi': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'F1Ma': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'best_c': 1.0,
            'val_micro': np.nan,
        })
    return {
        key: float(np.mean([score[key] for score in scores]))
        for key in scores[0]
    }


def row_from_artifact(artifact_path, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        artifact = torch.load(artifact_path, map_location='cpu')
    metadata, artifact_args = load_metadata(artifact_path, artifact)
    dataset_name = artifact_args.get('dataset') or metadata.get('dataset')
    method = artifact_args.get('method') or metadata.get('method')
    split_index = int(artifact_args.get('split_index', metadata.get('split_index', 0)))
    seed = int(artifact_args.get('resolved_seed', artifact_args.get('seed', 0)) or 0)
    if dataset_name is None or method is None:
        raise ValueError(f'Missing dataset/method metadata in {artifact_path}')
    if args.include_methods and method not in args.include_methods:
        return []
    if args.include_datasets and dataset_name not in args.include_datasets:
        return []

    path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    dataset = get_dataset(path, dataset_name)
    data = dataset[0]
    train_mask, val_mask, test_mask = get_split_masks(data, split_index)
    eval_args = SimpleNamespace(
        dataset=dataset_name,
        eval_mode=args.eval_mode,
    )
    use_mask = should_use_mask_eval(eval_args, data)
    labels = data.y.detach().cpu()
    raw_features = data.x.detach().cpu()
    ssl_embeddings = artifact['embeddings'].detach().cpu()
    if raw_features.size(0) != ssl_embeddings.size(0):
        raise ValueError(
            f'Node count mismatch for {artifact_path}: '
            f'raw={raw_features.size(0)}, ssl={ssl_embeddings.size(0)}'
        )

    rows = []
    for mode in args.modes:
        features = build_features(
            mode,
            raw_features,
            ssl_embeddings,
            args.final_l2_normalize,
        )
        if use_mask:
            metrics = evaluate_with_masks(
                features,
                labels,
                train_mask,
                val_mask,
                test_mask,
                args,
            )
        else:
            metrics = evaluate_random_split(features, labels, args, seed)
        rows.append({
            'artifact_path': str(artifact_path),
            'run_dir': str(artifact_path.parent),
            'dataset': dataset_name,
            'method': method,
            'seed': seed,
            'split_index': split_index,
            'mode': mode,
            'eval_mode': 'mask' if use_mask else 'random',
            'num_nodes': int(raw_features.size(0)),
            'raw_dim': int(raw_features.size(1)),
            'ssl_dim': int(ssl_embeddings.size(1)),
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


def aggregate_rows(rows):
    grouped = defaultdict(list)
    by_key_mode = defaultdict(dict)
    for row in rows:
        key = (row['dataset'], row['method'], row['seed'], row['split_index'])
        by_key_mode[key][row['mode']] = row
    delta_rows = []
    for key, mode_rows in by_key_mode.items():
        dataset, method, seed, split_index = key
        if 'concat' not in mode_rows:
            continue
        concat = mode_rows['concat']
        raw = mode_rows.get('raw')
        ssl = mode_rows.get('ssl')
        delta_rows.append({
            'dataset': dataset,
            'method': method,
            'seed': seed,
            'split_index': split_index,
            'concat_F1Mi': concat['F1Mi'],
            'concat_F1Ma': concat['F1Ma'],
            'raw_F1Mi': np.nan if raw is None else raw['F1Mi'],
            'raw_F1Ma': np.nan if raw is None else raw['F1Ma'],
            'ssl_F1Mi': np.nan if ssl is None else ssl['F1Mi'],
            'ssl_F1Ma': np.nan if ssl is None else ssl['F1Ma'],
            'concat_minus_raw_F1Mi': (
                np.nan if raw is None else concat['F1Mi'] - raw['F1Mi']
            ),
            'concat_minus_raw_F1Ma': (
                np.nan if raw is None else concat['F1Ma'] - raw['F1Ma']
            ),
            'concat_minus_ssl_F1Mi': (
                np.nan if ssl is None else concat['F1Mi'] - ssl['F1Mi']
            ),
            'concat_minus_ssl_F1Ma': (
                np.nan if ssl is None else concat['F1Ma'] - ssl['F1Ma']
            ),
        })

    for row in delta_rows:
        grouped[(row['dataset'], row['method'])].append(row)

    aggregates = []
    for (dataset, method), group in sorted(grouped.items()):
        aggregate = {
            'dataset': dataset,
            'method': method,
            'num_runs': len(group),
        }
        metrics = [
            'concat_F1Mi',
            'concat_F1Ma',
            'raw_F1Mi',
            'raw_F1Ma',
            'ssl_F1Mi',
            'ssl_F1Ma',
            'concat_minus_raw_F1Mi',
            'concat_minus_raw_F1Ma',
            'concat_minus_ssl_F1Mi',
            'concat_minus_ssl_F1Ma',
        ]
        for metric in metrics:
            values = np.array([row[metric] for row in group], dtype=float)
            aggregate[f'{metric}_mean'] = float(np.nanmean(values))
            aggregate[f'{metric}_std'] = float(np.nanstd(values))
            if metric.startswith('concat_minus'):
                aggregate[f'{metric}_positive'] = int(np.nansum(values > 1e-12))
                aggregate[f'{metric}_zero'] = int(np.nansum(np.abs(values) <= 1e-12))
                aggregate[f'{metric}_negative'] = int(np.nansum(values < -1e-12))
        aggregates.append(aggregate)
    return delta_rows, aggregates


def main():
    args = parse_args()
    artifact_paths = find_artifacts(args)
    rows = []
    for artifact_path in artifact_paths:
        try:
            rows.extend(row_from_artifact(artifact_path, args))
        except Exception as exc:
            rows.append({
                'artifact_path': str(artifact_path),
                'run_dir': str(artifact_path.parent),
                'dataset': '',
                'method': '',
                'seed': '',
                'split_index': '',
                'mode': 'error',
                'eval_mode': '',
                'num_nodes': '',
                'raw_dim': '',
                'ssl_dim': '',
                'feature_dim': '',
                'F1Mi': np.nan,
                'F1Ma': np.nan,
                'best_c': np.nan,
                'val_micro': np.nan,
                'error': str(exc),
            })

    run_fieldnames = [
        'artifact_path',
        'run_dir',
        'dataset',
        'method',
        'seed',
        'split_index',
        'mode',
        'eval_mode',
        'num_nodes',
        'raw_dim',
        'ssl_dim',
        'feature_dim',
        'F1Mi',
        'F1Ma',
        'best_c',
        'val_micro',
        'error',
    ]
    for row in rows:
        row.setdefault('error', '')
    write_csv(args.out, rows, run_fieldnames)

    valid_rows = [row for row in rows if row['mode'] != 'error']
    delta_rows, aggregates = aggregate_rows(valid_rows)
    delta_path = Path(args.aggregate_out).with_name(
        Path(args.aggregate_out).stem.replace('_aggregate', '_paired')
        + Path(args.aggregate_out).suffix
    )
    if delta_rows:
        write_csv(delta_path, delta_rows, list(delta_rows[0].keys()))
    if aggregates:
        write_csv(args.aggregate_out, aggregates, list(aggregates[0].keys()))
    print(
        f'(I) | artifacts={len(artifact_paths)}, rows={len(rows)}, '
        f'valid_rows={len(valid_rows)}, out={args.out}, aggregate={args.aggregate_out}'
    )


if __name__ == '__main__':
    main()
