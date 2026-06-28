import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--splits', nargs='+', type=int, required=True)
    parser.add_argument('--raw-dir', default='runs/raw_feature_smoke')
    parser.add_argument('--grace-dir', required=True)
    parser.add_argument('--raw-complement-dir', required=True)
    parser.add_argument('--paired-out', required=True)
    parser.add_argument('--aggregate-out', required=True)
    return parser.parse_args()


def read_eval(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    return {
        'F1Mi': float(row['F1Mi_mean']),
        'F1Ma': float(row['F1Ma_mean']),
    }


def raw_path(raw_dir, dataset, split):
    return Path(raw_dir) / f'{dataset}_split{split}' / 'eval_summary.csv'


def run_path(root, dataset, method, split):
    return Path(root) / f'{dataset}_{method}_seed0_split{split}' / 'eval_summary.csv'


def make_paired(args):
    rows = []
    for dataset in args.datasets:
        for split in args.splits:
            raw = read_eval(raw_path(args.raw_dir, dataset, split))
            grace = read_eval(run_path(args.grace_dir, dataset, 'grace', split))
            rc = read_eval(run_path(
                args.raw_complement_dir,
                dataset,
                'raw_complement_gcl',
                split,
            ))
            row = {
                'dataset': dataset,
                'split_index': split,
                'status': 'computed' if raw and grace and rc else 'missing',
            }
            for prefix, record in [
                ('raw', raw),
                ('grace', grace),
                ('raw_complement', rc),
            ]:
                for metric in ['F1Mi', 'F1Ma']:
                    row[f'{prefix}_{metric}'] = (
                        record[metric] if record is not None else ''
                    )
            if raw and grace and rc:
                for metric in ['F1Mi', 'F1Ma']:
                    row[f'delta_rc_minus_raw_{metric}'] = (
                        rc[metric] - raw[metric]
                    )
                    row[f'delta_rc_minus_grace_{metric}'] = (
                        rc[metric] - grace[metric]
                    )
                    row[f'delta_grace_minus_raw_{metric}'] = (
                        grace[metric] - raw[metric]
                    )
            rows.append(row)
    return rows


def summarize(values):
    return {
        'mean': mean(values),
        'std': pstdev(values) if len(values) > 1 else 0.0,
        'positive': sum(value > 0 for value in values),
        'zero': sum(value == 0 for value in values),
        'negative': sum(value < 0 for value in values),
    }


def make_aggregate(rows):
    out = []
    for dataset in sorted({row['dataset'] for row in rows}):
        group = [
            row for row in rows
            if row['dataset'] == dataset and row['status'] == 'computed'
        ]
        record = {
            'dataset': dataset,
            'num_splits': len(group),
        }
        for key in [
            'delta_rc_minus_raw_F1Mi',
            'delta_rc_minus_raw_F1Ma',
            'delta_rc_minus_grace_F1Mi',
            'delta_rc_minus_grace_F1Ma',
            'delta_grace_minus_raw_F1Mi',
            'delta_grace_minus_raw_F1Ma',
        ]:
            values = [float(row[key]) for row in group if row.get(key) != '']
            if not values:
                continue
            stats = summarize(values)
            for stat_name, value in stats.items():
                record[f'{key}_{stat_name}'] = value
        out.append(record)
    return out


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paired = make_paired(args)
    aggregate = make_aggregate(paired)
    write_csv(args.paired_out, paired)
    write_csv(args.aggregate_out, aggregate)
    print(f'(I) paired={len(paired)} aggregate={len(aggregate)}')


if __name__ == '__main__':
    main()
