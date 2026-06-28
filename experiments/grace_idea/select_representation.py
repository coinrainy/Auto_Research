import argparse
import csv
import json
import os.path as osp
from collections import defaultdict
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from train import get_dataset, get_split_masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[])
    parser.add_argument('--runs-dir', action='append', default=[])
    parser.add_argument('--out', default='runs/summaries/representation_selection.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/representation_selection_aggregate.csv')
    parser.add_argument('--include-datasets', nargs='*', default=None)
    parser.add_argument('--include-methods', nargs='*', default=None)
    parser.add_argument('--candidate-names', nargs='*', default=None)
    parser.add_argument('--c-min-power', type=int, default=-8)
    parser.add_argument('--c-max-power', type=int, default=8)
    parser.add_argument('--max-iter', type=int, default=3000)
    parser.add_argument('--solver', default='liblinear',
                        choices=['liblinear', 'lbfgs'])
    parser.add_argument('--selection-eval-mode', default='auto',
                        choices=['auto', 'mask', 'random'])
    parser.add_argument('--train-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--random-repeats', type=int, default=3)
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


def fit_logreg(x_train, y_train, c_value, max_iter, solver):
    clf = LogisticRegression(
        solver=solver,
        C=float(c_value),
        max_iter=max_iter,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', FutureWarning)
        clf.fit(x_train, y_train)
    return clf


def to_numpy(features):
    return normalize(features.detach().cpu().numpy(), norm='l2')


def candidate_features(data, artifact):
    candidates = {
        'raw': data.x.detach().cpu(),
        'saved': artifact['embeddings'].detach().cpu(),
    }
    complement = artifact.get('final_complement')
    graph_context = artifact.get('final_graph_context')
    raw_anchor = artifact.get('final_raw_anchor')
    if complement is not None:
        complement = complement.detach().cpu()
        candidates['anchor'] = torch.cat([
            F.normalize(data.x.detach().cpu(), dim=1),
            F.normalize(complement, dim=1),
        ], dim=1)
        candidates['complement'] = complement
    if graph_context is not None:
        candidates['graph'] = graph_context.detach().cpu()
    if raw_anchor is not None and complement is not None:
        candidates['hidden'] = torch.cat([
            raw_anchor.detach().cpu(),
            complement.detach().cpu(),
        ], dim=1)
    return candidates


def filter_candidates(candidates, names):
    if not names:
        return candidates
    selected = {}
    for name in names:
        if name in candidates:
            selected[name] = candidates[name]
    if not selected:
        available = ','.join(sorted(candidates))
        requested = ','.join(names)
        raise ValueError(
            f'No requested candidates are available. requested={requested}; '
            f'available={available}'
        )
    return selected


def evaluate_candidate(features, labels, train_mask, val_mask, test_mask, args):
    x = to_numpy(features)
    y = labels.detach().cpu().numpy()
    train_mask = train_mask.detach().cpu().numpy().astype(bool)
    val_mask = val_mask.detach().cpu().numpy().astype(bool)
    test_mask = test_mask.detach().cpu().numpy().astype(bool)
    c_values = 2.0 ** np.arange(args.c_min_power, args.c_max_power)
    best = None
    for c_value in c_values:
        clf = fit_logreg(
            x[train_mask],
            y[train_mask],
            c_value,
            args.max_iter,
            args.solver,
        )
        y_val = clf.predict(x[val_mask])
        val_micro = f1_score(y[val_mask], y_val, average='micro', zero_division=0)
        if best is None or val_micro > best['val_micro']:
            best = {
                'c': float(c_value),
                'val_micro': float(val_micro),
            }
    clf = fit_logreg(
        x[train_mask],
        y[train_mask],
        best['c'],
        args.max_iter,
        args.solver,
    )
    y_test = clf.predict(x[test_mask])
    return {
        'best_c': best['c'],
        'val_micro': best['val_micro'],
        'test_micro': f1_score(y[test_mask], y_test, average='micro', zero_division=0),
        'test_macro': f1_score(y[test_mask], y_test, average='macro', zero_division=0),
    }


def random_split_masks(labels, train_ratio, val_ratio, seed):
    y = labels.detach().cpu().numpy()
    num_nodes = len(y)
    indices = np.arange(num_nodes)
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError(
            '--train-ratio and --val-ratio must be positive and sum to less than 1.'
        )
    try:
        train_idx, rest_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
            stratify=y,
        )
        rest_y = y[rest_idx]
        val_fraction_of_rest = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx,
            train_size=val_fraction_of_rest,
            random_state=seed + 7919,
            stratify=rest_y,
        )
    except ValueError:
        train_idx, rest_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=seed,
            stratify=None,
        )
        val_fraction_of_rest = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx,
            train_size=val_fraction_of_rest,
            random_state=seed + 7919,
            stratify=None,
        )

    masks = []
    for idx in [train_idx, val_idx, test_idx]:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.as_tensor(idx, dtype=torch.long)] = True
        masks.append(mask)
    return tuple(masks)


def should_use_random_selection(args, dataset_name, train_mask, val_mask, test_mask):
    if args.selection_eval_mode == 'random':
        return True
    if args.selection_eval_mode == 'mask':
        return False
    if train_mask is None or val_mask is None or test_mask is None:
        return True
    return dataset_name in ['Cora', 'CiteSeer', 'PubMed']


def row_prefix(artifact_path, dataset_name, method, seed, split_index,
               repeat, eval_mode):
    return {
        'artifact_path': str(artifact_path),
        'run_dir': str(artifact_path.parent),
        'dataset': dataset_name,
        'method': method,
        'seed': seed,
        'split_index': split_index,
        'repeat': repeat,
        'eval_mode': eval_mode,
    }


def candidate_priority(name):
    priority = {
        'saved': 5,
        'anchor': 4,
        'graph': 3,
        'raw': 2,
        'complement': 1,
        'hidden': 0,
    }
    return priority.get(name, -1)


def evaluate_artifact(artifact_path, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        artifact = torch.load(artifact_path, map_location='cpu')
    _, artifact_args = load_metadata(artifact_path, artifact)
    dataset_name = artifact_args.get('dataset')
    method = artifact_args.get('method')
    split_index = int(artifact_args.get('split_index', 0))
    seed = int(artifact_args.get('resolved_seed', artifact_args.get('seed', 0)) or 0)
    if args.include_datasets and dataset_name not in args.include_datasets:
        return []
    if args.include_methods and method not in args.include_methods:
        return []

    dataset = get_dataset(osp.join(osp.expanduser('~'), 'datasets', dataset_name),
                          dataset_name)
    data = dataset[0]
    train_mask, val_mask, test_mask = get_split_masks(data, split_index)
    use_random = should_use_random_selection(
        args,
        dataset_name,
        train_mask,
        val_mask,
        test_mask,
    )
    if (
        not use_random
        and (train_mask is None or val_mask is None or test_mask is None)
    ):
        return [{
            **row_prefix(
                artifact_path,
                dataset_name,
                method,
                seed,
                split_index,
                '',
                'mask',
            ),
            'status': 'skipped_no_masks',
        }]

    rows = []
    candidates = filter_candidates(candidate_features(data, artifact),
                                   args.candidate_names)
    repeats = range(args.random_repeats) if use_random else range(1)
    eval_mode = 'random' if use_random else 'mask'
    for repeat in repeats:
        current_train, current_val, current_test = train_mask, val_mask, test_mask
        if use_random:
            current_train, current_val, current_test = random_split_masks(
                data.y,
                args.train_ratio,
                args.val_ratio,
                seed + repeat * 104729,
            )
        scored = {}
        for name, features in candidates.items():
            metrics = evaluate_candidate(
                features,
                data.y,
                current_train,
                current_val,
                current_test,
                args,
            )
            scored[name] = metrics
            rows.append({
                **row_prefix(
                    artifact_path,
                    dataset_name,
                    method,
                    seed,
                    split_index,
                    repeat,
                    eval_mode,
                ),
                'status': 'candidate',
                'candidate': name,
                'selected_candidate': '',
                'best_c': metrics['best_c'],
                'val_micro': metrics['val_micro'],
                'test_micro': metrics['test_micro'],
                'test_macro': metrics['test_macro'],
            })
        selected_name = max(
            scored,
            key=lambda name: (scored[name]['val_micro'], candidate_priority(name)),
        )
        selected = scored[selected_name]
        rows.append({
            **row_prefix(
                artifact_path,
                dataset_name,
                method,
                seed,
                split_index,
                repeat,
                eval_mode,
            ),
            'status': 'selected',
            'candidate': 'selected',
            'selected_candidate': selected_name,
            'best_c': selected['best_c'],
            'val_micro': selected['val_micro'],
            'test_micro': selected['test_micro'],
            'test_macro': selected['test_macro'],
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
    selected = [row for row in rows if row.get('status') == 'selected']
    groups = defaultdict(list)
    for row in selected:
        groups[(row['dataset'], row['method'])].append(row)
    out = []
    for (dataset, method), group in sorted(groups.items()):
        rec = {
            'dataset': dataset,
            'method': method,
            'num_runs': len(group),
        }
        for metric in ['test_micro', 'test_macro', 'val_micro']:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            rec[f'{metric}_mean'] = float(values.mean())
            rec[f'{metric}_std'] = float(values.std())
        counts = defaultdict(int)
        for row in group:
            counts[row['selected_candidate']] += 1
        rec['selected_counts'] = ';'.join(
            f'{key}:{counts[key]}' for key in sorted(counts)
        )
        out.append(rec)
    return out


def main():
    args = parse_args()
    rows = []
    for artifact_path in find_artifacts(args):
        try:
            rows.extend(evaluate_artifact(artifact_path, args))
        except Exception as exc:
            rows.append({
                'artifact_path': str(artifact_path),
                'run_dir': str(artifact_path.parent),
                'dataset': '',
                'method': '',
                'seed': '',
                'split_index': '',
                'repeat': '',
                'eval_mode': '',
                'status': 'error',
                'candidate': '',
                'selected_candidate': '',
                'best_c': '',
                'val_micro': '',
                'test_micro': '',
                'test_macro': '',
                'error': str(exc),
            })
    fieldnames = [
        'artifact_path',
        'run_dir',
        'dataset',
        'method',
        'seed',
        'split_index',
        'repeat',
        'eval_mode',
        'status',
        'candidate',
        'selected_candidate',
        'best_c',
        'val_micro',
        'test_micro',
        'test_macro',
        'error',
    ]
    for row in rows:
        row.setdefault('error', '')
    write_csv(args.out, rows, fieldnames)
    agg = aggregate(rows)
    if agg:
        write_csv(args.aggregate_out, agg, list(agg[0].keys()))
    print(f'(I) | rows={len(rows)}, selected={len([r for r in rows if r.get("status") == "selected"])}')


if __name__ == '__main__':
    main()
