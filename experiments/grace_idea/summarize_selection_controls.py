import argparse
import csv
from pathlib import Path


METRIC_KEYS = [
    'test_micro_mean',
    'test_macro_mean',
    'val_micro_mean',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aggregate-file',
        action='append',
        default=[],
        help='Aggregate CSV produced by select_representation.py. Can repeat.',
    )
    parser.add_argument(
        '--aggregate-glob',
        action='append',
        default=[],
        help='Glob pattern for aggregate CSV files. Can repeat.',
    )
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def collect_paths(args):
    paths = [Path(path) for path in args.aggregate_file]
    for pattern in args.aggregate_glob:
        paths.extend(sorted(Path().glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['_source_file'] = str(path)
    return rows


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = (row['dataset'], row['method'])
        indexed.setdefault(key, {})[row['status']] = row
    return indexed


def as_float(row, key):
    value = row.get(key, '')
    return float(value) if value != '' else ''


def make_summary(paths):
    all_rows = []
    for path in paths:
        all_rows.extend(read_rows(path))
    indexed = index_rows(all_rows)
    summary = []
    for (dataset, method), statuses in sorted(indexed.items()):
        selected = statuses.get('selected')
        random = statuses.get('selected_random')
        if selected is None or random is None:
            continue
        row = {
            'dataset': dataset,
            'method': method,
            'selected_num_runs': selected['num_runs'],
            'random_num_runs': random['num_runs'],
            'selected_counts': selected.get('selected_counts', ''),
            'random_counts': random.get('selected_counts', ''),
            'source_file': selected['_source_file'],
        }
        for key in METRIC_KEYS:
            selected_value = as_float(selected, key)
            random_value = as_float(random, key)
            row[f'selected_{key}'] = selected_value
            row[f'random_{key}'] = random_value
            row[f'delta_{key}'] = selected_value - random_value
        for key in ['test_micro_std', 'test_macro_std', 'val_micro_std']:
            row[f'selected_{key}'] = as_float(selected, key)
            row[f'random_{key}'] = as_float(random, key)
        summary.append(row)
    return summary


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    paths = collect_paths(args)
    if not paths:
        raise SystemExit('No aggregate files matched.')
    rows = make_summary(paths)
    write_csv(args.out, rows)
    print(f'(I) wrote {len(rows)} rows to {args.out}')


if __name__ == '__main__':
    main()
