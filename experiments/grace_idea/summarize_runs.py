import argparse
import csv
import re
from pathlib import Path
from statistics import mean, pstdev


RUN_PATTERN = re.compile(
    r'^(?P<dataset>.+)_(?P<method>grace|es_weighted)_seed(?P<seed>\d+)'
    r'(?:_split(?P<split>\d+))?$'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--paired-out', required=True)
    parser.add_argument('--aggregate-out', required=True)
    return parser.parse_args()


def read_eval_summary(path):
    with path.open() as handle:
        return next(csv.DictReader(handle))


def read_last_train_row(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1]


def load_runs(runs_dir):
    runs = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.match(run_dir.name)
        if match is None:
            continue
        eval_path = run_dir / 'eval_summary.csv'
        train_path = run_dir / 'train_log.csv'
        if not eval_path.exists() or not train_path.exists():
            continue
        eval_row = read_eval_summary(eval_path)
        train_row = read_last_train_row(train_path)
        row = {
            'dataset': match.group('dataset'),
            'method': match.group('method'),
            'seed': int(match.group('seed')),
            'split': int(match.group('split') or 0),
            'F1Mi': float(eval_row['F1Mi_mean']),
            'F1Ma': float(eval_row['F1Ma_mean']),
            'final_loss': float(train_row['loss']),
            'weight_mean': float(train_row['weight_mean']) if train_row['weight_mean'] else '',
            'weight_std': float(train_row['weight_std']) if train_row['weight_std'] else '',
        }
        for key, value in eval_row.items():
            if key.startswith('F1Class') and key.endswith('_mean'):
                row[key.removesuffix('_mean')] = float(value)
        runs.append(row)
    return runs


def make_paired_rows(runs):
    keys = sorted({(r['dataset'], r['seed'], r['split']) for r in runs})
    paired = []
    for dataset, seed, split in keys:
        grace = next(
            (r for r in runs if r['dataset'] == dataset and r['seed'] == seed
             and r['split'] == split and r['method'] == 'grace'),
            None,
        )
        es_weighted = next(
            (r for r in runs if r['dataset'] == dataset and r['seed'] == seed
             and r['split'] == split and r['method'] == 'es_weighted'),
            None,
        )
        if grace is None or es_weighted is None:
            continue
        row = {
            'dataset': dataset,
            'seed': seed,
            'split': split,
            'grace_F1Mi': grace['F1Mi'],
            'es_weighted_F1Mi': es_weighted['F1Mi'],
            'delta_F1Mi': es_weighted['F1Mi'] - grace['F1Mi'],
            'grace_F1Ma': grace['F1Ma'],
            'es_weighted_F1Ma': es_weighted['F1Ma'],
            'delta_F1Ma': es_weighted['F1Ma'] - grace['F1Ma'],
            'grace_final_loss': grace['final_loss'],
            'es_weighted_final_loss': es_weighted['final_loss'],
            'es_weighted_weight_mean': es_weighted['weight_mean'],
            'es_weighted_weight_std': es_weighted['weight_std'],
        }
        class_keys = sorted(k for k in grace if k.startswith('F1Class'))
        for class_key in class_keys:
            row[f'grace_{class_key}'] = grace[class_key]
            row[f'es_weighted_{class_key}'] = es_weighted[class_key]
            row[f'delta_{class_key}'] = es_weighted[class_key] - grace[class_key]
        paired.append(row)
    return paired


def summarize(values):
    return {
        'mean': mean(values),
        'pop_std': pstdev(values) if len(values) > 1 else 0.0,
        'positive': sum(v > 0 for v in values),
        'zero': sum(v == 0 for v in values),
        'negative': sum(v < 0 for v in values),
    }


def make_aggregate_rows(paired):
    aggregate = []
    for dataset in sorted({row['dataset'] for row in paired}):
        rows = [row for row in paired if row['dataset'] == dataset]
        f1mi = summarize([row['delta_F1Mi'] for row in rows])
        f1ma = summarize([row['delta_F1Ma'] for row in rows])
        aggregate.append({
            'dataset': dataset,
            'num_pairs': len(rows),
            'delta_F1Mi_mean': f1mi['mean'],
            'delta_F1Mi_pop_std': f1mi['pop_std'],
            'delta_F1Mi_positive': f1mi['positive'],
            'delta_F1Mi_zero': f1mi['zero'],
            'delta_F1Mi_negative': f1mi['negative'],
            'delta_F1Ma_mean': f1ma['mean'],
            'delta_F1Ma_pop_std': f1ma['pop_std'],
            'delta_F1Ma_positive': f1ma['positive'],
            'delta_F1Ma_zero': f1ma['zero'],
            'delta_F1Ma_negative': f1ma['negative'],
        })
    return aggregate


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    runs = load_runs(args.runs_dir)
    paired = make_paired_rows(runs)
    aggregate = make_aggregate_rows(paired)
    write_csv(args.paired_out, paired)
    write_csv(args.aggregate_out, aggregate)
    print(f'loaded_runs={len(runs)} paired_rows={len(paired)} aggregate_rows={len(aggregate)}')


if __name__ == '__main__':
    main()
