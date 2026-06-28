import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--patched-aggregate',
        action='append',
        required=True,
        help='Aggregate CSV from patched residual-branch evaluation.',
    )
    parser.add_argument(
        '--seed-label',
        action='append',
        required=True,
        help='Seed label matching each --patched-aggregate, e.g. seed42.',
    )
    parser.add_argument(
        '--posthoc-summary',
        required=True,
        help='Selector summary CSV containing the post-hoc resid1 baseline.',
    )
    parser.add_argument(
        '--posthoc-selector',
        default='resid1',
        help='Selector name used as the post-hoc comparator.',
    )
    parser.add_argument(
        '--out',
        default='runs/summaries/spgcl_sparc_residual_vs_posthoc.csv',
    )
    parser.add_argument(
        '--aggregate-out',
        default='runs/summaries/spgcl_sparc_residual_vs_posthoc_aggregate.csv',
    )
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_patched_rows(paths, labels):
    if len(paths) != len(labels):
        raise SystemExit('--seed-label count must match --patched-aggregate count.')
    rows = []
    for path, label in zip(paths, labels):
        for row in read_csv(path):
            rows.append(
                {
                    'source_label': label,
                    'dataset': row['dataset'],
                    'patched_method': row['method'],
                    'patched_F1Mi_mean': float(row['F1Mi_mean']),
                    'patched_F1Mi_std': float(row['F1Mi_std']),
                    'patched_F1Ma_mean': float(row['F1Ma_mean']),
                    'patched_F1Ma_std': float(row['F1Ma_std']),
                    'patched_num_runs': int(row['num_runs']),
                    'patched_source': path,
                }
            )
    return rows


def load_posthoc_rows(path, selector):
    rows = {}
    for row in read_csv(path):
        if row['selector'] != selector:
            continue
        key = (row['source_label'], row['dataset'])
        rows[key] = row
    return rows


def build_comparison_rows(patched_rows, posthoc_rows, posthoc_selector):
    out = []
    for row in patched_rows:
        key = (row['source_label'], row['dataset'])
        posthoc = posthoc_rows.get(key)
        if posthoc is None:
            continue
        patched_f1mi = row['patched_F1Mi_mean']
        patched_f1ma = row['patched_F1Ma_mean']
        posthoc_f1mi = float(posthoc['F1Mi_mean'])
        posthoc_f1ma = float(posthoc['F1Ma_mean'])
        out.append(
            {
                'source_label': row['source_label'],
                'dataset': row['dataset'],
                'patched_method': row['patched_method'],
                'posthoc_selector': posthoc_selector,
                'patched_num_runs': row['patched_num_runs'],
                'posthoc_num_runs': posthoc['num_runs'],
                'patched_F1Mi_mean': patched_f1mi,
                'posthoc_F1Mi_mean': posthoc_f1mi,
                'delta_vs_posthoc_F1Mi': patched_f1mi - posthoc_f1mi,
                'patched_F1Ma_mean': patched_f1ma,
                'posthoc_F1Ma_mean': posthoc_f1ma,
                'delta_vs_posthoc_F1Ma': patched_f1ma - posthoc_f1ma,
                'patched_F1Mi_std': row['patched_F1Mi_std'],
                'posthoc_F1Mi_std': posthoc['F1Mi_std'],
                'patched_F1Ma_std': row['patched_F1Ma_std'],
                'posthoc_F1Ma_std': posthoc['F1Ma_std'],
                'patched_source': row['patched_source'],
            }
        )
    return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset']].append(row)
        groups['ALL'].append(row)

    out = []
    for dataset in sorted(groups, key=lambda name: (name == 'ALL', name)):
        group = groups[dataset]
        rec = {'dataset': dataset, 'num_rows': len(group)}
        for metric in [
            'patched_F1Mi_mean',
            'posthoc_F1Mi_mean',
            'delta_vs_posthoc_F1Mi',
            'patched_F1Ma_mean',
            'posthoc_F1Ma_mean',
            'delta_vs_posthoc_F1Ma',
        ]:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            rec[f'{metric}_mean'] = float(values.mean())
            rec[f'{metric}_std'] = float(values.std())
        rec['positive_F1Mi_count'] = sum(
            1 for row in group if float(row['delta_vs_posthoc_F1Mi']) > 0
        )
        rec['negative_F1Mi_count'] = sum(
            1 for row in group if float(row['delta_vs_posthoc_F1Mi']) < 0
        )
        rec['positive_F1Ma_count'] = sum(
            1 for row in group if float(row['delta_vs_posthoc_F1Ma']) > 0
        )
        rec['negative_F1Ma_count'] = sum(
            1 for row in group if float(row['delta_vs_posthoc_F1Ma']) < 0
        )
        out.append(rec)
    return out


def main():
    args = parse_args()
    patched_rows = load_patched_rows(args.patched_aggregate, args.seed_label)
    posthoc_rows = load_posthoc_rows(args.posthoc_summary, args.posthoc_selector)
    rows = build_comparison_rows(patched_rows, posthoc_rows, args.posthoc_selector)
    fields = [
        'source_label',
        'dataset',
        'patched_method',
        'posthoc_selector',
        'patched_num_runs',
        'posthoc_num_runs',
        'patched_F1Mi_mean',
        'posthoc_F1Mi_mean',
        'delta_vs_posthoc_F1Mi',
        'patched_F1Ma_mean',
        'posthoc_F1Ma_mean',
        'delta_vs_posthoc_F1Ma',
        'patched_F1Mi_std',
        'posthoc_F1Mi_std',
        'patched_F1Ma_std',
        'posthoc_F1Ma_std',
        'patched_source',
    ]
    write_csv(args.out, rows, fields)
    aggregate_rows = aggregate(rows)
    write_csv(
        args.aggregate_out,
        aggregate_rows,
        [
            'dataset',
            'num_rows',
            'patched_F1Mi_mean_mean',
            'patched_F1Mi_mean_std',
            'posthoc_F1Mi_mean_mean',
            'posthoc_F1Mi_mean_std',
            'delta_vs_posthoc_F1Mi_mean',
            'delta_vs_posthoc_F1Mi_std',
            'patched_F1Ma_mean_mean',
            'patched_F1Ma_mean_std',
            'posthoc_F1Ma_mean_mean',
            'posthoc_F1Ma_mean_std',
            'delta_vs_posthoc_F1Ma_mean',
            'delta_vs_posthoc_F1Ma_std',
            'positive_F1Mi_count',
            'negative_F1Mi_count',
            'positive_F1Ma_count',
            'negative_F1Ma_count',
        ],
    )
    print(
        f'(I) | patched={len(patched_rows)}, compared={len(rows)}, '
        f'out={args.out}, aggregate={args.aggregate_out}'
    )


if __name__ == '__main__':
    main()
