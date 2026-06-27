import argparse
import csv
import json
import os.path as osp
from pathlib import Path

import torch

from eval import label_classification, label_classification_with_masks
from train import get_dataset, get_split_masks, should_use_mask_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split-index', type=int, default=0)
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    parser.add_argument('--eval-mode', type=str, default='auto',
                        choices=['auto', 'random', 'mask'])
    parser.add_argument('--out-dir', type=str, default=None)
    return parser.parse_args()


def save_eval_summary(out_dir, eval_stats):
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {}
    for metric, values in eval_stats.items():
        if metric.startswith('_'):
            continue
        row[f'{metric}_mean'] = values['mean']
        row[f'{metric}_std'] = values['std']
    with (out_dir / 'eval_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    details = eval_stats.get('_details')
    if details is not None:
        with (out_dir / 'eval_details.json').open('w') as handle:
            json.dump(details, handle, indent=2)


def main():
    args = parse_args()
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    train_mask, val_mask, test_mask = get_split_masks(data, args.split_index)
    use_mask = should_use_mask_eval(args, data)
    features = data.x.detach()

    if use_mask:
        eval_stats = label_classification_with_masks(
            features,
            data.y,
            train_mask,
            val_mask,
            test_mask,
        )
    else:
        eval_stats = label_classification(
            features,
            data.y,
            args.eval_ratio,
            random_state=0,
        )
    save_eval_summary(args.out_dir, eval_stats)
    if args.out_dir is not None:
        torch.save({
            'features': features.cpu(),
            'labels': data.y.detach().cpu(),
            'dataset': args.dataset,
            'split_index': args.split_index,
            'eval_mode': args.eval_mode,
        }, Path(args.out_dir) / 'artifacts.pt')


if __name__ == '__main__':
    main()
