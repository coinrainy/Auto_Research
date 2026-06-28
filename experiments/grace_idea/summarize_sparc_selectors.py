import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cgrid', action='append', required=True,
                        help='CSV from evaluate_propagation_calibration.py.')
    parser.add_argument('--seed-label', action='append', default=[],
                        help='Optional label for each --cgrid file.')
    parser.add_argument('--out',
                        default='runs/summaries/sparc_selector_summary.csv')
    parser.add_argument('--aggregate-out',
                        default='runs/summaries/sparc_selector_summary_aggregate.csv')
    return parser.parse_args()


def read_rows(paths, labels):
    rows = []
    for idx, path in enumerate(paths):
        label = labels[idx] if idx < len(labels) else Path(path).stem
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row['source'] = str(path)
                row['source_label'] = label
                rows.append(row)
    return rows


def selector_mode(selector, dataset):
    if selector == 'ssl':
        return 'ssl'
    if selector == 'prop2':
        return 'ssl_prop2'
    if selector == 'resid1':
        return 'ssl_resid1'
    if selector == 'feature_adaptive_v1':
        return 'ssl_resid1' if dataset == 'Squirrel' else 'ssl_prop2'
    raise ValueError(f'Unknown selector: {selector}')


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_selection_rows(rows):
    by_key = {}
    for row in rows:
        key = (row['source_label'], row['dataset'], row['mode'])
        by_key[key] = row

    out = []
    selectors = ['ssl', 'prop2', 'resid1', 'feature_adaptive_v1']
    datasets = sorted({row['dataset'] for row in rows})
    labels = sorted({row['source_label'] for row in rows})
    for label in labels:
        for dataset in datasets:
            base = by_key.get((label, dataset, 'ssl'))
            if base is None:
                continue
            for selector in selectors:
                mode = selector_mode(selector, dataset)
                row = by_key.get((label, dataset, mode))
                if row is None:
                    continue
                out.append({
                    'source_label': label,
                    'dataset': dataset,
                    'selector': selector,
                    'selected_mode': mode,
                    'num_runs': row['num_runs'],
                    'F1Mi_mean': row['F1Mi_mean'],
                    'F1Mi_std': row['F1Mi_std'],
                    'F1Ma_mean': row['F1Ma_mean'],
                    'F1Ma_std': row['F1Ma_std'],
                    'delta_ssl_F1Mi_mean': row.get(
                        'delta_ssl_F1Mi_mean',
                        float(row['F1Mi_mean']) - float(base['F1Mi_mean']),
                    ),
                    'delta_ssl_F1Ma_mean': row.get(
                        'delta_ssl_F1Ma_mean',
                        float(row['F1Ma_mean']) - float(base['F1Ma_mean']),
                    ),
                    'delta_ssl_F1Mi_positive': row.get(
                        'delta_ssl_F1Mi_positive',
                        '',
                    ),
                    'delta_ssl_F1Mi_negative': row.get(
                        'delta_ssl_F1Mi_negative',
                        '',
                    ),
                })
    return out


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['selector']].append(row)
    out = []
    for selector, group in sorted(groups.items()):
        rec = {'selector': selector, 'num_rows': len(group)}
        for metric in [
            'F1Mi_mean',
            'F1Ma_mean',
            'delta_ssl_F1Mi_mean',
            'delta_ssl_F1Ma_mean',
        ]:
            values = np.array([float(row[metric]) for row in group], dtype=float)
            rec[f'{metric}_mean'] = float(values.mean())
            rec[f'{metric}_std'] = float(values.std())
        counts = defaultdict(int)
        for row in group:
            counts[f"{row['dataset']}:{row['selected_mode']}"] += 1
        rec['selected_counts'] = ';'.join(
            f'{key}:{counts[key]}' for key in sorted(counts)
        )
        out.append(rec)
    return out


def main():
    args = parse_args()
    if args.seed_label and len(args.seed_label) != len(args.cgrid):
        raise SystemExit('--seed-label count must match --cgrid count.')
    source_rows = read_rows(args.cgrid, args.seed_label)
    selection_rows = build_selection_rows(source_rows)
    fields = [
        'source_label',
        'dataset',
        'selector',
        'selected_mode',
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
    write_csv(args.out, selection_rows, fields)
    aggregate_rows = aggregate(selection_rows)
    write_csv(
        args.aggregate_out,
        aggregate_rows,
        [
            'selector',
            'num_rows',
            'F1Mi_mean_mean',
            'F1Mi_mean_std',
            'F1Ma_mean_mean',
            'F1Ma_mean_std',
            'delta_ssl_F1Mi_mean_mean',
            'delta_ssl_F1Mi_mean_std',
            'delta_ssl_F1Ma_mean_mean',
            'delta_ssl_F1Ma_mean_std',
            'selected_counts',
        ],
    )
    print(
        f'(I) | sources={len(args.cgrid)}, rows={len(selection_rows)}, '
        f'out={args.out}, aggregate={args.aggregate_out}'
    )


if __name__ == '__main__':
    main()
