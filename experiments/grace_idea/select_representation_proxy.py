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

from select_representation import (
    candidate_features,
    candidate_priority,
    evaluate_candidate,
    filter_candidates,
    find_artifacts,
    load_metadata,
    random_split_masks,
    should_use_random_selection,
)
from train import get_dataset, get_split_masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[])
    parser.add_argument('--runs-dir', action='append', default=[])
    parser.add_argument('--out', default='runs/summaries/representation_proxy.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/representation_proxy_aggregate.csv')
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
    parser.add_argument('--random-selection-repeats', type=int, default=0)
    parser.add_argument('--proxy-min-effective-rank', type=float, default=5.0)
    parser.add_argument('--proxy-raw-alignment-weight', type=float, default=0.08)
    parser.add_argument('--proxy-raw-candidate-penalty', type=float, default=0.10)
    parser.add_argument('--proxy-small-graph-threshold', type=int, default=500)
    parser.add_argument('--proxy-small-graph-raw-penalty', type=float, default=0.0)
    parser.add_argument('--max-metric-edges', type=int, default=60000)
    parser.add_argument('--max-metric-pairs', type=int, default=60000)
    return parser.parse_args()


def l2_normalize(features):
    return F.normalize(features.detach().cpu().float(), dim=1)


def sample_indices(num_items, max_items, seed):
    if num_items <= max_items:
        return torch.arange(num_items)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_items, generator=generator)[:max_items]


def sampled_pairs(num_nodes, max_pairs, seed):
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(num_nodes, (max_pairs,), generator=generator)
    dst = torch.randint(num_nodes, (max_pairs,), generator=generator)
    return src, dst


def effective_rank(features):
    z = l2_normalize(features)
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
    z = l2_normalize(features)
    src, dst = edge_index.detach().cpu()
    edge_idx = sample_indices(src.numel(), max_edges, seed=17)
    src = src[edge_idx]
    dst = dst[edge_idx]
    edge_cos = (z[src] * z[dst]).sum(dim=1).mean()
    pair_src, pair_dst = sampled_pairs(
        z.size(0),
        min(max_pairs, max(1, z.size(0) * 10)),
        seed=29,
    )
    random_cos = (z[pair_src] * z[pair_dst]).sum(dim=1).mean()
    return {
        'edge_cosine_mean': float(edge_cos.item()),
        'random_cosine_mean': float(random_cos.item()),
        'edge_random_contrast': float((edge_cos - random_cos).item()),
    }


def raw_similarity_correlation(features, raw_features, max_pairs):
    z = l2_normalize(features)
    raw = l2_normalize(raw_features)
    pair_src, pair_dst = sampled_pairs(
        z.size(0),
        min(max_pairs, max(1, z.size(0) * 10)),
        seed=41,
    )
    z_sim = (z[pair_src] * z[pair_dst]).sum(dim=1).numpy()
    raw_sim = (raw[pair_src] * raw[pair_dst]).sum(dim=1).numpy()
    if z_sim.std() < 1e-12 or raw_sim.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(z_sim, raw_sim)[0, 1])


def candidate_proxy_metrics(data, candidates, args):
    metrics = {}
    for name, features in candidates.items():
        record = {}
        record.update(effective_rank(features))
        record.update(graph_contrast(
            features,
            data.edge_index,
            args.max_metric_edges,
            args.max_metric_pairs,
        ))
        record['raw_similarity_correlation'] = raw_similarity_correlation(
            features,
            data.x,
            args.max_metric_pairs,
        )
        raw_penalty = 0.0
        if name == 'raw':
            if data.num_nodes <= args.proxy_small_graph_threshold:
                raw_penalty = args.proxy_small_graph_raw_penalty
            else:
                raw_penalty = args.proxy_raw_candidate_penalty
        record['proxy_score'] = (
            record['edge_random_contrast']
            + args.proxy_raw_alignment_weight * record['raw_similarity_correlation']
            - raw_penalty
        )
        record['proxy_eligible'] = (
            record['effective_rank'] >= args.proxy_min_effective_rank
        )
        metrics[name] = record
    return metrics


def select_proxy(metrics):
    eligible = [
        name for name, record in metrics.items()
        if record['proxy_eligible']
    ]
    names = eligible if eligible else list(metrics)
    return max(
        names,
        key=lambda name: (metrics[name]['proxy_score'], candidate_priority(name)),
    )


def prefix_row(artifact_path, dataset_name, method, seed, split_index,
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


def selected_row(prefix, status, selected_name, scored, metrics):
    selected = scored[selected_name]
    metric = metrics[selected_name]
    return {
        **prefix,
        'status': status,
        'candidate': status,
        'selected_candidate': selected_name,
        'best_c': selected['best_c'],
        'val_micro': selected['val_micro'],
        'test_micro': selected['test_micro'],
        'test_macro': selected['test_macro'],
        **metric,
    }


def candidate_row(prefix, name, scored, metrics):
    metric = metrics[name]
    score = scored[name]
    return {
        **prefix,
        'status': 'candidate',
        'candidate': name,
        'selected_candidate': '',
        'best_c': score['best_c'],
        'val_micro': score['val_micro'],
        'test_micro': score['test_micro'],
        'test_macro': score['test_macro'],
        **metric,
    }


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
            **prefix_row(
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

    candidates = filter_candidates(candidate_features(data, artifact),
                                   args.candidate_names)
    metrics = candidate_proxy_metrics(data, candidates, args)
    proxy_name = select_proxy(metrics)
    rows = []
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
        prefix = prefix_row(
            artifact_path,
            dataset_name,
            method,
            seed,
            split_index,
            repeat,
            eval_mode,
        )
        scored = {}
        for name, features in candidates.items():
            scored[name] = evaluate_candidate(
                features,
                data.y,
                current_train,
                current_val,
                current_test,
                args,
            )
            rows.append(candidate_row(prefix, name, scored, metrics))
        validation_name = max(
            scored,
            key=lambda name: (scored[name]['val_micro'], candidate_priority(name)),
        )
        rows.append(selected_row(
            prefix,
            'selected_validation',
            validation_name,
            scored,
            metrics,
        ))
        rows.append(selected_row(
            prefix,
            'selected_proxy',
            proxy_name,
            scored,
            metrics,
        ))
        candidate_names = sorted(scored)
        for control_idx in range(args.random_selection_repeats):
            rng = np.random.default_rng(
                seed * 1000003 + split_index * 9176 + repeat * 101
                + control_idx * 37
            )
            random_name = candidate_names[int(rng.integers(len(candidate_names)))]
            rows.append(selected_row(
                prefix,
                'selected_random',
                random_name,
                scored,
                metrics,
            ))
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    selected = [
        row for row in rows
        if row.get('status') in [
            'selected_validation',
            'selected_proxy',
            'selected_random',
        ]
    ]
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
            'raw_similarity_correlation',
        ]:
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
        rows.extend(evaluate_artifact(artifact_path, args))
    if not rows:
        raise SystemExit('No rows produced.')
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
        'proxy_score',
        'proxy_eligible',
        'effective_rank',
        'participation_ratio',
        'anisotropy',
        'edge_cosine_mean',
        'random_cosine_mean',
        'edge_random_contrast',
        'raw_similarity_correlation',
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
            'raw_similarity_correlation_mean',
            'raw_similarity_correlation_std',
            'selected_counts',
        ],
    )
    selected_proxy = sum(1 for row in rows if row['status'] == 'selected_proxy')
    print(f'(I) | rows={len(rows)}, selected_proxy={selected_proxy}')


if __name__ == '__main__':
    main()
