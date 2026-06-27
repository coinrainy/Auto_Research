import argparse
import csv
import json
import os.path as osp
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter as t

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import yaml
from torch_geometric.datasets import Actor, CitationFull, Planetoid, WebKB
from torch_geometric.nn import GCNConv
from yaml import SafeLoader

from eval import label_classification, label_classification_with_masks
from model import Encoder, Model, drop_feature

try:
    from torch_geometric.utils import dropout_edge
except ImportError:
    from torch_geometric.utils import dropout_adj as dropout_edge


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='grace',
                        choices=['grace', 'es_weighted'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--ema-decay', type=float, default=0.99)
    parser.add_argument('--weight-power', type=float, default=1.0)
    parser.add_argument('--min-weight', type=float, default=0.05)
    parser.add_argument('--negative-weighting', action='store_true')
    parser.add_argument('--no-anchor-weighting', action='store_true')
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--split-index', type=int, default=0)
    parser.add_argument('--eval-mode', type=str, default='auto',
                        choices=['auto', 'random', 'mask'])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if torch.cuda.is_available():
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f'gpu_id={gpu_id} is invalid for this machine.')
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def get_dataset(path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']:
        pyg_name = 'dblp' if name == 'DBLP' else name
        dataset_cls = CitationFull if pyg_name == 'dblp' else Planetoid
        return dataset_cls(path, pyg_name, transform=T.NormalizeFeatures())
    if name in ['Texas', 'Cornell', 'Wisconsin']:
        return WebKB(path, name, transform=T.NormalizeFeatures())
    if name == 'Actor':
        return Actor(path, transform=T.NormalizeFeatures())
    raise ValueError(f'Unsupported dataset: {name}')


def select_mask(mask, split_index):
    if mask is None:
        return None
    if mask.dim() == 1:
        return mask.bool()
    if split_index < 0 or split_index >= mask.size(1):
        raise ValueError(
            f'split_index={split_index} is out of range for mask with '
            f'{mask.size(1)} splits.'
        )
    return mask[:, split_index].bool()


def get_split_masks(data, split_index):
    if not all(hasattr(data, attr) for attr in ['train_mask', 'val_mask', 'test_mask']):
        return None, None, None
    return (
        select_mask(data.train_mask, split_index),
        select_mask(data.val_mask, split_index),
        select_mask(data.test_mask, split_index),
    )


def should_use_mask_eval(args, data):
    if args.eval_mode == 'random':
        return False
    has_masks = all(hasattr(data, attr) for attr in ['train_mask', 'val_mask', 'test_mask'])
    heterophily_dataset = args.dataset in ['Texas', 'Cornell', 'Wisconsin', 'Actor']
    return args.eval_mode == 'mask' or (args.eval_mode == 'auto' and has_masks and heterophily_dataset)


def build_model(config, dataset, device):
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    encoder = Encoder(
        dataset.num_features,
        config['num_hidden'],
        activation,
        base_model=base_model,
        k=config['num_layers'],
    ).to(device)
    return Model(
        encoder,
        config['num_hidden'],
        config['num_proj_hidden'],
        config['tau'],
    ).to(device)


def make_views(data, config):
    edge_index_1 = dropout_edge(data.edge_index, p=config['drop_edge_rate_1'])[0]
    edge_index_2 = dropout_edge(data.edge_index, p=config['drop_edge_rate_2'])[0]
    x_1 = drop_feature(data.x, config['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, config['drop_feature_rate_2'])
    return x_1, edge_index_1, x_2, edge_index_2


def build_teacher(model):
    teacher = deepcopy(model)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher(teacher, student, decay):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)


@torch.no_grad()
def embedding_stability_weights(teacher, z1, z2, data, args):
    teacher.eval()
    target = teacher(data.x, data.edge_index)
    sim1 = F.cosine_similarity(z1.detach(), target, dim=1)
    sim2 = F.cosine_similarity(z2.detach(), target, dim=1)
    weights = ((sim1 + sim2) * 0.5 + 1.0) * 0.5
    weights = weights.clamp(0.0, 1.0)
    if args.weight_power != 1.0:
        weights = weights.pow(args.weight_power)
    if args.min_weight > 0.0:
        weights = weights * (1.0 - args.min_weight) + args.min_weight
    return weights.detach()


def train_epoch(model, teacher, data, optimizer, config, args, epoch):
    model.train()
    optimizer.zero_grad()

    x_1, edge_index_1, x_2, edge_index_2 = make_views(data, config)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    weights = None
    use_weighting = args.method == 'es_weighted' and epoch > args.warmup_epochs
    if use_weighting:
        weights = embedding_stability_weights(teacher, z1, z2, data, args)

    pair_weights = None if args.no_anchor_weighting else weights
    denominator_weights = weights if args.negative_weighting else None
    loss = model.loss(
        z1,
        z2,
        batch_size=args.batch_size,
        pair_weights=pair_weights,
        denominator_weights=denominator_weights,
    )

    loss.backward()
    optimizer.step()
    if teacher is not None:
        update_teacher(teacher, model, args.ema_decay)

    log = {
        'loss': loss.item(),
        'stage': 'weighted' if use_weighting else 'warmup_or_grace',
    }
    if weights is not None:
        log.update({
            'weight_mean': weights.mean().item(),
            'weight_std': weights.std(unbiased=False).item(),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
        })
    return log, weights


@torch.no_grad()
def encode(model, data):
    model.eval()
    return model(data.x, data.edge_index)


def prepare_save_dir(args, seed):
    if args.save_dir is None:
        return None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    split_dataset = args.dataset in ['Texas', 'Cornell', 'Wisconsin', 'Actor']
    split_suffix = f'_split{args.split_index}' if args.eval_mode == 'mask' or split_dataset else ''
    run_name = f'{args.dataset}_{args.method}_seed{seed}{split_suffix}'
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_train_log(run_dir, row):
    if run_dir is None:
        return
    path = run_dir / 'train_log.csv'
    fieldnames = [
        'epoch',
        'loss',
        'stage',
        'weight_mean',
        'weight_std',
        'weight_min',
        'weight_max',
        'epoch_time',
        'total_time',
    ]
    exists = path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})


def save_eval_summary(run_dir, eval_stats):
    if run_dir is None or eval_stats is None:
        return
    path = run_dir / 'eval_summary.csv'
    row = {}
    for metric, values in eval_stats.items():
        row[f'{metric}_mean'] = values['mean']
        row[f'{metric}_std'] = values['std']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_metadata(run_dir, args, config, seed, device, eval_stats):
    if run_dir is None:
        return
    metadata = {
        'dataset': args.dataset,
        'method': args.method,
        'seed': seed,
        'device': str(device),
        'args': vars(args),
        'config': config,
        'eval_stats': eval_stats,
    }
    with (run_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2)


def save_artifacts(run_dir, model, data, embeddings, final_weights, args, config):
    if run_dir is None:
        return
    torch.save({
        'embeddings': embeddings.detach().cpu(),
        'labels': data.y.detach().cpu(),
        'train_mask': (
            data.train_mask.detach().cpu() if hasattr(data, 'train_mask') else None
        ),
        'val_mask': data.val_mask.detach().cpu() if hasattr(data, 'val_mask') else None,
        'test_mask': data.test_mask.detach().cpu() if hasattr(data, 'test_mask') else None,
        'final_weights': None if final_weights is None else final_weights.detach().cpu(),
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'config': config,
    }, run_dir / 'artifacts.pt')


def main():
    args = parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    seed = config['seed'] if args.seed is None else args.seed
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    set_seed(seed)

    device = get_device(args.gpu_id)
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0].to(device)
    train_mask, val_mask, test_mask = get_split_masks(data, args.split_index)

    model = build_model(config, dataset, device)
    teacher = build_teacher(model) if args.method == 'es_weighted' else None
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    run_dir = prepare_save_dir(args, seed)

    print(
        f'(I) | dataset={args.dataset}, method={args.method}, seed={seed}, '
        f'split_index={args.split_index}, device={device}'
    )
    if args.method == 'es_weighted':
        print(
            f'(I) | warmup_epochs={args.warmup_epochs}, ema_decay={args.ema_decay}, '
            f'anchor_weighting={not args.no_anchor_weighting}, '
            f'negative_weighting={args.negative_weighting}'
        )

    start = t()
    prev = start
    final_weights = None
    for epoch in range(1, config['num_epochs'] + 1):
        log, epoch_weights = train_epoch(model, teacher, data, optimizer, config, args, epoch)
        if epoch_weights is not None:
            final_weights = epoch_weights

        now = t()
        log.update({
            'epoch': epoch,
            'epoch_time': now - prev,
            'total_time': now - start,
        })
        weight_text = ''
        if 'weight_mean' in log:
            weight_text = (
                f", weight_mean={log['weight_mean']:.4f}, "
                f"weight_std={log['weight_std']:.4f}"
            )
        should_log = (
            args.log_every > 0
            and (epoch == 1 or epoch == config['num_epochs'] or epoch % args.log_every == 0)
        )
        if should_log:
            print(
                f"(T) | Epoch={epoch:03d}, loss={log['loss']:.4f}, "
                f"stage={log['stage']}{weight_text}, "
                f"this epoch {log['epoch_time']:.4f}, total {log['total_time']:.4f}"
            )
        append_train_log(run_dir, log)
        prev = now

    print('=== Final ===')
    embeddings = encode(model, data)
    eval_stats = None
    if not args.skip_eval:
        if should_use_mask_eval(args, data):
            eval_stats = label_classification_with_masks(
                embeddings,
                data.y,
                train_mask,
                val_mask,
                test_mask,
            )
        else:
            eval_stats = label_classification(
                embeddings,
                data.y,
                ratio=args.eval_ratio,
                random_state=seed,
            )
    save_eval_summary(run_dir, eval_stats)
    save_metadata(run_dir, args, config, seed, device, eval_stats)
    save_artifacts(run_dir, model, data, embeddings, final_weights, args, config)


if __name__ == '__main__':
    main()
