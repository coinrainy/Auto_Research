import argparse
import csv
import json
import os.path as osp
import random
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
import yaml
from torch_geometric.datasets import (
    Actor,
    CitationFull,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.nn import GCNConv
from yaml import SafeLoader

from eval import label_classification, label_classification_with_masks
from model import (
    EgoEncoder,
    Encoder,
    GatedEgoGraphEncoder,
    Model,
    RawComplementEncoder,
    ResidualEgoEncoder,
    drop_feature,
)

try:
    from torch_geometric.utils import dropout_edge
except ImportError:
    from torch_geometric.utils import dropout_adj as dropout_edge


HETEROPHILY_DATASETS = [
    'Texas',
    'Cornell',
    'Wisconsin',
    'Actor',
    'Chameleon',
    'Squirrel',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='grace',
                        choices=[
                            'grace',
                            'ego_grace',
                            'residual_grace',
                            'gated_ego_graph_grace',
                            'es_weighted',
                            'sgfn',
                            'spectral_mix',
                            'pbcl',
                            'pccl',
                            'rr_gcl',
                            'hybrid_rr_gcl',
                            'cbr_gcl',
                            'gated_cbr_gcl',
                            'stable_cluster_cbr_gcl',
                            'raw_complement_gcl',
                        ])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--ego-gate-init', type=float, default=0.5)
    parser.add_argument('--graph-gate-temperature', type=float, default=0.5)
    parser.add_argument('--graph-gate-threshold', type=float, default=0.0)
    parser.add_argument('--graph-gate-min', type=float, default=0.0)
    parser.add_argument('--graph-gate-max', type=float, default=1.0)
    parser.add_argument('--ema-decay', type=float, default=0.99)
    parser.add_argument('--weight-power', type=float, default=1.0)
    parser.add_argument('--min-weight', type=float, default=0.05)
    parser.add_argument('--fn-risk-margin', type=float, default=1.0)
    parser.add_argument('--fn-risk-temperature', type=float, default=0.5)
    parser.add_argument('--fn-attenuation-power', type=float, default=1.0)
    parser.add_argument('--fn-attraction-weight', type=float, default=0.0)
    parser.add_argument('--fn-consensus', type=str, default='none',
                        choices=['none', 'feature'])
    parser.add_argument('--fn-context-gate', type=str, default='none',
                        choices=[
                            'none',
                            'local_feature',
                            'degree_inverse',
                            'local_feature_degree',
                        ])
    parser.add_argument('--fn-context-temperature', type=float, default=0.5)
    parser.add_argument('--fn-context-threshold', type=float, default=0.0)
    parser.add_argument('--fn-degree-threshold', type=float, default=0.0)
    parser.add_argument('--fn-context-pair-mode', type=str, default='product',
                        choices=['product', 'min', 'anchor'])
    parser.add_argument('--spectral-mix-mode', type=str, default='adaptive',
                        choices=['adaptive', 'low', 'high', 'random'])
    parser.add_argument('--spectral-mix-temperature', type=float, default=0.5)
    parser.add_argument('--spectral-mix-jitter', type=float, default=0.1)
    parser.add_argument('--spectral-high-scale', type=float, default=1.0)
    parser.add_argument('--spectral-residual-alpha', type=float, default=1.0)
    parser.add_argument('--pbcl-num-prototypes', type=int, default=0)
    parser.add_argument('--pbcl-kmeans-iters', type=int, default=10)
    parser.add_argument('--pbcl-weight-power', type=float, default=1.0)
    parser.add_argument('--pbcl-min-weight', type=float, default=0.25)
    parser.add_argument('--pbcl-max-weight', type=float, default=4.0)
    parser.add_argument('--pccl-num-prototypes', type=int, default=0)
    parser.add_argument('--pccl-kmeans-iters', type=int, default=10)
    parser.add_argument('--pccl-prototype-temperature', type=float, default=0.2)
    parser.add_argument('--pccl-target-temperature', type=float, default=0.1)
    parser.add_argument('--pccl-consistency-weight', type=float, default=0.05)
    parser.add_argument('--pccl-balance-weight', type=float, default=0.01)
    parser.add_argument('--rr-offdiag-weight', type=float, default=0.005)
    parser.add_argument('--rr-loss-scale', type=float, default=1.0)
    parser.add_argument('--hybrid-rr-weight', type=float, default=0.01)
    parser.add_argument('--cbr-rr-weight', type=float, default=0.001)
    parser.add_argument('--cbr-num-clusters', type=int, default=0)
    parser.add_argument('--cbr-kmeans-iters', type=int, default=10)
    parser.add_argument('--cbr-min-weight', type=float, default=0.25)
    parser.add_argument('--cbr-max-weight', type=float, default=4.0)
    parser.add_argument('--cbr-gate-min-diag', type=float, default=0.82)
    parser.add_argument('--cbr-gate-temperature', type=float, default=0.03)
    parser.add_argument('--cbr-gate-min-scale', type=float, default=0.0)
    parser.add_argument('--cbr-stability-min-margin', type=float, default=0.05)
    parser.add_argument('--cbr-stability-temperature', type=float, default=0.03)
    parser.add_argument('--cbr-stability-min-scale', type=float, default=0.25)
    parser.add_argument('--raw-complement-weight', type=float, default=0.05)
    parser.add_argument('--raw-complement-detach-anchor',
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--raw-complement-eval-mode', type=str, default='anchor',
                        choices=['anchor', 'hidden', 'graph', 'anchor_graph'])
    parser.add_argument('--pair-shuffle-mode', type=str, default='column',
                        choices=['column', 'row'])
    parser.add_argument('--pair-normalization', type=str, default='none',
                        choices=['none', 'row_mean', 'blend_row_mean'])
    parser.add_argument('--pair-reallocation-alpha', type=float, default=0.5)
    parser.add_argument('--negative-weighting', action='store_true')
    parser.add_argument('--no-anchor-weighting', action='store_true')
    parser.add_argument('--shuffle-weights', action='store_true')
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--control-seed', type=int, default=None)
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--split-index', type=int, default=0)
    parser.add_argument('--eval-mode', type=str, default='auto',
                        choices=['auto', 'random', 'mask'])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args):
    if args.shuffle_weights and args.random_weights:
        raise ValueError('Use at most one of --shuffle-weights and --random-weights.')
    if (args.shuffle_weights or args.random_weights) and args.method not in [
            'es_weighted', 'sgfn', 'pbcl', 'pccl', 'rr_gcl', 'hybrid_rr_gcl',
            'cbr_gcl', 'gated_cbr_gcl', 'stable_cluster_cbr_gcl']:
        raise ValueError('Weight controls are only valid with weighted methods.')


def weight_control_name(args):
    if args.method not in [
            'es_weighted', 'sgfn', 'pbcl', 'pccl', 'rr_gcl', 'hybrid_rr_gcl',
            'cbr_gcl', 'gated_cbr_gcl', 'stable_cluster_cbr_gcl']:
        return 'none'
    if args.shuffle_weights:
        return 'shuffled'
    if args.random_weights:
        return 'uniform_random'
    return 'normal'


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
    if name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(
            path,
            name.lower(),
            transform=T.NormalizeFeatures(),
        )
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
    heterophily_dataset = args.dataset in HETEROPHILY_DATASETS
    return args.eval_mode == 'mask' or (args.eval_mode == 'auto' and has_masks and heterophily_dataset)


def build_model(config, dataset, device, args):
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    if args.method == 'ego_grace':
        encoder = EgoEncoder(
            dataset.num_features,
            config['num_hidden'],
            activation,
            k=config['num_layers'],
        ).to(device)
    elif args.method == 'gated_ego_graph_grace':
        encoder = GatedEgoGraphEncoder(
            dataset.num_features,
            config['num_hidden'],
            activation,
            base_model=base_model,
            k=config['num_layers'],
            gate_temperature=args.graph_gate_temperature,
            gate_threshold=args.graph_gate_threshold,
            gate_min=args.graph_gate_min,
            gate_max=args.graph_gate_max,
        ).to(device)
    elif args.method == 'residual_grace':
        encoder = ResidualEgoEncoder(
            dataset.num_features,
            config['num_hidden'],
            activation,
            base_model=base_model,
            k=config['num_layers'],
            gate_init=args.ego_gate_init,
        ).to(device)
    elif args.method == 'raw_complement_gcl':
        encoder = RawComplementEncoder(
            dataset.num_features,
            config['num_hidden'],
            activation,
            base_model=base_model,
            k=config['num_layers'],
            detach_raw_anchor=args.raw_complement_detach_anchor,
        ).to(device)
    else:
        encoder = Encoder(
            dataset.num_features,
            config['num_hidden'],
            activation,
            base_model=base_model,
            k=config['num_layers'],
        ).to(device)
    model_hidden = (
        config['num_hidden'] * 2
        if args.method == 'raw_complement_gcl'
        else config['num_hidden']
    )
    return Model(
        encoder,
        model_hidden,
        config['num_proj_hidden'],
        config['tau'],
    ).to(device)


@torch.no_grad()
def neighbor_mean_features(x, edge_index):
    source, target = edge_index
    aggregate = torch.zeros_like(x)
    degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    aggregate.index_add_(0, target, x[source])
    degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
    neighbor_mean = aggregate / degree.clamp_min(1.0).view(-1, 1)
    return torch.where(degree.view(-1, 1) > 0, neighbor_mean, x)


@torch.no_grad()
def spectral_mix_gate(data, args):
    if args.spectral_mix_mode == 'low':
        return torch.ones(data.x.size(0), device=data.x.device, dtype=data.x.dtype)
    if args.spectral_mix_mode == 'high':
        return torch.zeros(data.x.size(0), device=data.x.device, dtype=data.x.dtype)
    if args.spectral_mix_mode == 'random':
        return torch.rand(data.x.size(0), device=data.x.device, dtype=data.x.dtype)

    agreement = local_feature_agreement(data)
    score = standardized_vector(agreement)
    temperature = max(args.spectral_mix_temperature, 1e-12)
    return torch.sigmoid(score / temperature).to(data.x.dtype)


@torch.no_grad()
def spectral_mix_features(data, args, view_index):
    low = neighbor_mean_features(data.x.detach(), data.edge_index)
    high = data.x.detach() - low
    gate = spectral_mix_gate(data, args)
    if args.spectral_mix_jitter > 0.0:
        noise = (
            torch.rand_like(gate)
            * (2.0 * args.spectral_mix_jitter)
            - args.spectral_mix_jitter
        )
        if view_index % 2 == 0:
            gate = gate + noise
        else:
            gate = gate - noise
    gate = gate.clamp(0.0, 1.0).view(-1, 1)
    spectral = gate * low + (1.0 - gate) * args.spectral_high_scale * high
    alpha = max(0.0, min(args.spectral_residual_alpha, 1.0))
    mixed = (1.0 - alpha) * data.x.detach() + alpha * spectral
    return mixed.to(data.x.dtype)


def make_views(data, config, args):
    edge_index_1 = dropout_edge(data.edge_index, p=config['drop_edge_rate_1'])[0]
    edge_index_2 = dropout_edge(data.edge_index, p=config['drop_edge_rate_2'])[0]
    if args.method == 'spectral_mix':
        x_base_1 = spectral_mix_features(data, args, 1)
        x_base_2 = spectral_mix_features(data, args, 2)
    else:
        x_base_1 = data.x
        x_base_2 = data.x
    x_1 = drop_feature(x_base_1, config['drop_feature_rate_1'])
    x_2 = drop_feature(x_base_2, config['drop_feature_rate_2'])
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
def teacher_clean_embeddings(teacher, data):
    teacher.eval()
    return teacher(data.x, data.edge_index)


@torch.no_grad()
def embedding_stability_weights(target, z1, z2, args):
    sim1 = F.cosine_similarity(z1.detach(), target, dim=1)
    sim2 = F.cosine_similarity(z2.detach(), target, dim=1)
    weights = ((sim1 + sim2) * 0.5 + 1.0) * 0.5
    weights = weights.clamp(0.0, 1.0)
    if args.weight_power != 1.0:
        weights = weights.pow(args.weight_power)
    if args.min_weight > 0.0:
        weights = weights * (1.0 - args.min_weight) + args.min_weight
    return weights.detach()


@torch.no_grad()
def kmeans_assignments_and_centers(embeddings, num_clusters, num_iters, args, epoch):
    embeddings = F.normalize(embeddings.detach().float(), dim=1)
    num_nodes = embeddings.size(0)
    num_clusters = max(1, min(num_clusters, num_nodes))
    generator = make_control_generator(args, epoch)
    init = torch.randperm(num_nodes, generator=generator)[:num_clusters]
    centers = embeddings[init.to(embeddings.device)].clone()

    assignments = torch.zeros(num_nodes, device=embeddings.device, dtype=torch.long)
    for _ in range(max(1, num_iters)):
        assignments = torch.mm(embeddings, centers.t()).argmax(dim=1)
        updated = torch.zeros_like(centers)
        counts = torch.bincount(assignments, minlength=num_clusters).to(
            embeddings.device,
            dtype=embeddings.dtype,
        )
        updated.index_add_(0, assignments, embeddings)
        non_empty = counts > 0
        centers = torch.where(
            non_empty.view(-1, 1),
            updated / counts.clamp_min(1.0).view(-1, 1),
            centers,
        )
        centers = F.normalize(centers, dim=1)
    return assignments, centers


@torch.no_grad()
def kmeans_assignments(embeddings, num_clusters, num_iters, args, epoch):
    assignments, _ = kmeans_assignments_and_centers(
        embeddings,
        num_clusters,
        num_iters,
        args,
        epoch,
    )
    return assignments


@torch.no_grad()
def prototype_balance_weights(z1, z2, args, epoch):
    consensus = (z1.detach() + z2.detach()) * 0.5
    assignments = kmeans_assignments(
        consensus,
        args.resolved_pbcl_num_prototypes,
        args.pbcl_kmeans_iters,
        args,
        epoch,
    )
    counts = torch.bincount(
        assignments,
        minlength=args.resolved_pbcl_num_prototypes,
    ).to(consensus.device, dtype=consensus.dtype)
    used_counts = counts[counts > 0]
    if used_counts.numel() == 0:
        return torch.ones(consensus.size(0), device=consensus.device)
    mean_count = used_counts.mean()
    weights = mean_count / counts[assignments].clamp_min(1.0)
    if args.pbcl_weight_power != 1.0:
        weights = weights.pow(args.pbcl_weight_power)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.clamp(
        min=max(args.pbcl_min_weight, 0.0),
        max=max(args.pbcl_max_weight, args.pbcl_min_weight),
    )
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.detach()


def prototype_consistency_objective(z1, z2, args, epoch):
    consensus = F.normalize(((z1.detach() + z2.detach()) * 0.5).float(), dim=1)
    _, centers = kmeans_assignments_and_centers(
        consensus,
        args.resolved_pccl_num_prototypes,
        args.pccl_kmeans_iters,
        args,
        epoch,
    )
    centers = centers.to(z1.device, dtype=z1.dtype).detach()
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    logits1 = torch.mm(z1_norm, centers.t()) / max(args.pccl_prototype_temperature, 1e-12)
    logits2 = torch.mm(z2_norm, centers.t()) / max(args.pccl_prototype_temperature, 1e-12)

    with torch.no_grad():
        target_logits = (
            torch.mm(consensus.to(z1.device, dtype=z1.dtype), centers.t())
            / max(args.pccl_target_temperature, 1e-12)
        )
        targets = F.softmax(target_logits, dim=1)
        if args.shuffle_weights:
            generator = make_control_generator(args, epoch)
            permutation = torch.randperm(targets.size(0), generator=generator)
            targets = targets[permutation.to(targets.device)]
        elif args.random_weights:
            generator = make_control_generator(args, epoch)
            targets = torch.rand(targets.shape, generator=generator).to(
                targets.device,
                dtype=targets.dtype,
            )
            targets = targets / targets.sum(1, keepdim=True).clamp_min(1e-12)

    consistency = (
        -(targets * F.log_softmax(logits1, dim=1)).sum(1).mean()
        + -(targets * F.log_softmax(logits2, dim=1)).sum(1).mean()
    ) * 0.5
    usage = (
        F.softmax(logits1, dim=1).mean(0)
        + F.softmax(logits2, dim=1).mean(0)
    ) * 0.5
    uniform = usage.new_full(usage.shape, 1.0 / usage.numel())
    balance = (usage * (usage.clamp_min(1e-12) / uniform).log()).sum()
    entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
    diagnostics = {
        'prototype_usage_entropy': entropy.item(),
        'prototype_usage_max': usage.max().item(),
        'prototype_usage_min': usage.min().item(),
    }
    return consistency, balance, diagnostics


@torch.no_grad()
def row_standardized_risk(sim, args):
    num_nodes = sim.size(0)
    if num_nodes <= 1:
        return torch.ones_like(sim)

    diag = sim.diag()
    row_mean = (sim.sum(1) - diag) / (num_nodes - 1)
    centered = sim - row_mean.view(-1, 1)
    row_var = (
        centered.pow(2).sum(1)
        - (diag - row_mean).pow(2)
    ) / (num_nodes - 1)
    row_std = row_var.clamp_min(1e-12).sqrt()
    row_score = centered / row_std.view(-1, 1)

    return torch.sigmoid(
        (row_score - args.fn_risk_margin) / args.fn_risk_temperature
    )


@torch.no_grad()
def false_negative_pair_weights(target, data, args):
    target = F.normalize(target.detach(), dim=1)
    sim = torch.mm(target, target.t())
    risk = row_standardized_risk(sim, args)

    if args.fn_consensus == 'feature':
        features = F.normalize(data.x.detach(), dim=1)
        feature_sim = torch.mm(features, features.t())
        risk = risk * row_standardized_risk(feature_sim, args)

    context_gate = false_negative_context_gate(data, args)
    if context_gate is not None:
        risk = risk * pair_context_gate(context_gate, args)

    keep = (1.0 - risk).clamp(0.0, 1.0)
    if args.fn_attenuation_power != 1.0:
        keep = keep.pow(args.fn_attenuation_power)
    if args.min_weight > 0.0:
        keep = keep * (1.0 - args.min_weight) + args.min_weight
    keep.fill_diagonal_(1.0)
    return keep.detach(), None if context_gate is None else context_gate.detach()


@torch.no_grad()
def standardized_vector(values):
    values = values.detach().float()
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-12)


@torch.no_grad()
def local_feature_agreement(data):
    x = F.normalize(data.x.detach().float(), dim=1)
    source, target = data.edge_index
    aggregate = torch.zeros_like(x)
    degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    aggregate.index_add_(0, target, x[source])
    degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
    neighbor_mean = aggregate / degree.clamp_min(1.0).view(-1, 1)
    agreement = F.cosine_similarity(x, neighbor_mean, dim=1)
    agreement = torch.where(degree > 0, agreement, torch.zeros_like(agreement))
    return agreement


@torch.no_grad()
def inverse_degree_confidence(data, args):
    _, target = data.edge_index
    degree = torch.zeros(data.x.size(0), device=data.x.device, dtype=torch.float32)
    degree.index_add_(0, target, torch.ones_like(target, dtype=degree.dtype))
    degree_score = standardized_vector(torch.log1p(degree))
    return torch.sigmoid(
        (args.fn_degree_threshold - degree_score)
        / max(args.fn_context_temperature, 1e-12)
    )


@torch.no_grad()
def false_negative_context_gate(data, args):
    if args.fn_context_gate == 'none':
        return None
    temperature = max(args.fn_context_temperature, 1e-12)
    gates = []
    if args.fn_context_gate in ['local_feature', 'local_feature_degree']:
        local_score = standardized_vector(local_feature_agreement(data))
        gates.append(torch.sigmoid((local_score - args.fn_context_threshold) / temperature))
    if args.fn_context_gate in ['degree_inverse', 'local_feature_degree']:
        gates.append(inverse_degree_confidence(data, args))
    if not gates:
        return None
    gate = gates[0]
    for other in gates[1:]:
        gate = gate * other
    return gate.clamp(0.0, 1.0)


@torch.no_grad()
def pair_context_gate(node_gate, args):
    if args.fn_context_pair_mode == 'anchor':
        return node_gate.view(-1, 1)
    if args.fn_context_pair_mode == 'min':
        return torch.minimum(node_gate.view(-1, 1), node_gate.view(1, -1))
    return node_gate.view(-1, 1) * node_gate.view(1, -1)


def make_control_generator(args, epoch):
    base_seed = args.control_seed
    if base_seed is None:
        base_seed = args.resolved_seed + 1000003
    generator = torch.Generator(device='cpu')
    generator.manual_seed(base_seed + epoch * 9176 + args.split_index * 131)
    return generator


def apply_weight_control(weights, args, epoch):
    if weights is None:
        return None
    if args.shuffle_weights:
        generator = make_control_generator(args, epoch)
        permutation = torch.randperm(weights.numel(), generator=generator)
        return weights[permutation.to(weights.device)].detach()
    if args.random_weights:
        generator = make_control_generator(args, epoch)
        random_weights = torch.rand(weights.numel(), generator=generator)
        random_weights = random_weights.to(weights.device, dtype=weights.dtype)
        if args.min_weight > 0.0:
            random_weights = random_weights * (1.0 - args.min_weight) + args.min_weight
        return random_weights.detach()
    return weights


def apply_pair_weight_control(weights, args, epoch):
    if weights is None:
        return None
    if args.shuffle_weights:
        generator = make_control_generator(args, epoch)
        if args.pair_shuffle_mode == 'row':
            noise = torch.rand(weights.shape, generator=generator)
            permutation = noise.argsort(dim=1).to(weights.device)
            return torch.gather(weights, 1, permutation).detach()
        permutation = torch.randperm(weights.size(1), generator=generator)
        return weights[:, permutation.to(weights.device)].detach()
    if args.random_weights:
        generator = make_control_generator(args, epoch)
        random_weights = torch.rand(weights.shape, generator=generator)
        random_weights = random_weights.to(weights.device, dtype=weights.dtype)
        if args.min_weight > 0.0:
            random_weights = random_weights * (1.0 - args.min_weight) + args.min_weight
        random_weights.fill_diagonal_(1.0)
        return random_weights.detach()
    return weights


def normalize_pair_weights(weights, args):
    if weights is None or args.pair_normalization == 'none':
        return weights
    if args.pair_normalization not in ['row_mean', 'blend_row_mean']:
        raise ValueError(f'Unsupported pair normalization: {args.pair_normalization}')

    weights = weights.clamp_min(0.0).clone()
    original = weights.clone()
    num_nodes = weights.size(0)
    if num_nodes <= 1:
        return weights
    offdiag = ~torch.eye(num_nodes, dtype=torch.bool, device=weights.device)
    row_sum = weights.masked_fill(~offdiag, 0.0).sum(1, keepdim=True)
    row_mean = row_sum / max(num_nodes - 1, 1)
    weights = weights / row_mean.clamp_min(1e-12)
    weights.fill_diagonal_(1.0)
    if args.pair_normalization == 'blend_row_mean':
        alpha = min(max(args.pair_reallocation_alpha, 0.0), 1.0)
        weights = original * (1.0 - alpha) + weights * alpha
        weights.fill_diagonal_(1.0)
    return weights.detach()


def weight_diagnostics(weights):
    if weights is None:
        return {}
    weights = weights.detach().float().clamp_min(0.0)
    ess = weights.sum().pow(2) / weights.pow(2).sum().clamp_min(1e-12)
    return {
        'weight_ess': ess.item(),
        'weight_ess_ratio': (ess / weights.numel()).item(),
    }


def false_negative_attraction_loss(z1, z2, pair_keep_weights):
    if pair_keep_weights is None:
        return z1.new_tensor(0.0)
    num_nodes = z1.size(0)
    if num_nodes <= 1:
        return z1.new_tensor(0.0)

    risk = (1.0 - pair_keep_weights.detach()).clamp_min(0.0).to(
        z1.device,
        dtype=z1.dtype,
    )
    risk.fill_diagonal_(0.0)
    risk_sum = risk.sum().clamp_min(1e-12)
    consensus = F.normalize((z1 + z2) * 0.5, dim=1)
    distance = 1.0 - torch.mm(consensus, consensus.t())
    return (risk * distance).sum() / risk_sum


def off_diagonal(matrix):
    num_rows, num_cols = matrix.shape
    if num_rows != num_cols:
        raise ValueError('off_diagonal expects a square matrix.')
    return matrix.flatten()[:-1].view(num_rows - 1, num_rows + 1)[:, 1:].flatten()


def redundancy_reduction_objective(model, z1, z2, args, epoch):
    h1 = model.projection(z1)
    h2 = model.projection(z2)
    if args.shuffle_weights:
        generator = make_control_generator(args, epoch)
        permutation = torch.randperm(h2.size(0), generator=generator).to(h2.device)
        h2 = h2[permutation]
    elif args.random_weights:
        generator = make_control_generator(args, epoch)
        h2 = torch.randn(h2.shape, generator=generator).to(h2.device, dtype=h2.dtype)

    h1 = (h1 - h1.mean(0)) / h1.std(0, unbiased=False).clamp_min(1e-4)
    h2 = (h2 - h2.mean(0)) / h2.std(0, unbiased=False).clamp_min(1e-4)
    cross_correlation = torch.mm(h1.t(), h2) / h1.size(0)
    on_diag = (torch.diagonal(cross_correlation) - 1.0).pow(2).sum()
    off_diag = off_diagonal(cross_correlation).pow(2).sum()
    loss = args.rr_loss_scale * (on_diag + args.rr_offdiag_weight * off_diag)
    diagnostics = {
        'rr_on_diag_loss': on_diag.item(),
        'rr_off_diag_loss': off_diag.item(),
        'rr_cross_corr_diag_mean': torch.diagonal(cross_correlation).mean().item(),
        'rr_cross_corr_offdiag_mean_abs': (
            off_diagonal(cross_correlation).abs().mean().item()
        ),
    }
    return loss, diagnostics


@torch.no_grad()
def cluster_balance_weights(z1, z2, args, epoch, stability_weighted=False):
    consensus = F.normalize(((z1.detach() + z2.detach()) * 0.5).float(), dim=1)
    assignments, centers = kmeans_assignments_and_centers(
        consensus,
        args.resolved_cbr_num_clusters,
        args.cbr_kmeans_iters,
        args,
        epoch,
    )
    counts = torch.bincount(
        assignments,
        minlength=args.resolved_cbr_num_clusters,
    ).to(consensus.device, dtype=consensus.dtype)
    used_counts = counts[counts > 0]
    if used_counts.numel() == 0:
        weights = torch.ones(consensus.size(0), device=consensus.device)
    else:
        weights = used_counts.mean() / counts[assignments].clamp_min(1.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = weights.clamp(
            min=max(args.cbr_min_weight, 0.0),
            max=max(args.cbr_max_weight, args.cbr_min_weight),
        )
        weights = weights / weights.mean().clamp_min(1e-12)
    similarities = torch.mm(consensus, centers.t())
    assigned_sim = similarities.gather(1, assignments.view(-1, 1)).squeeze(1)
    if similarities.size(1) > 1:
        top2 = similarities.topk(2, dim=1).values
        cluster_margin = top2[:, 0] - top2[:, 1]
    else:
        cluster_margin = assigned_sim
    stability_scale = torch.ones_like(weights)
    if stability_weighted:
        temperature = max(args.cbr_stability_temperature, 1e-12)
        stability_scale = torch.sigmoid(
            (cluster_margin - args.cbr_stability_min_margin) / temperature
        )
        min_scale = min(max(args.cbr_stability_min_scale, 0.0), 1.0)
        if min_scale > 0.0:
            stability_scale = stability_scale * (1.0 - min_scale) + min_scale
        weights = weights * stability_scale.to(weights.device, dtype=weights.dtype)
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = weights.clamp(
            min=max(args.cbr_min_weight, 0.0),
            max=max(args.cbr_max_weight, args.cbr_min_weight),
        )
        weights = weights / weights.mean().clamp_min(1e-12)
    usage = counts / counts.sum().clamp_min(1.0)
    active = (counts > 0).sum()
    entropy = -(usage[usage > 0] * usage[usage > 0].log()).sum()
    diagnostics = {
        'cbr_num_active_clusters': active.item(),
        'cbr_cluster_entropy': entropy.item(),
        'cbr_cluster_margin_mean': cluster_margin.mean().item(),
        'cbr_cluster_margin_std': cluster_margin.std(unbiased=False).item(),
        'cbr_cluster_margin_min': cluster_margin.min().item(),
        'cbr_cluster_margin_max': cluster_margin.max().item(),
        'cbr_cluster_assigned_sim_mean': assigned_sim.mean().item(),
        'cbr_stability_scale_mean': stability_scale.mean().item(),
        'cbr_stability_scale_std': stability_scale.std(unbiased=False).item(),
        'cbr_stability_scale_min': stability_scale.min().item(),
        'cbr_stability_scale_max': stability_scale.max().item(),
        'cbr_weight_mean': weights.mean().item(),
        'cbr_weight_std': weights.std(unbiased=False).item(),
        'cbr_weight_min': weights.min().item(),
        'cbr_weight_max': weights.max().item(),
    }
    diagnostics.update({
        f'cbr_{key}': value for key, value in weight_diagnostics(weights).items()
    })
    return weights.detach(), diagnostics


def weighted_standardize(values, weights):
    weights = weights.to(values.device, dtype=values.dtype).view(-1, 1)
    denom = weights.sum().clamp_min(1e-12)
    mean = (weights * values).sum(0, keepdim=True) / denom
    centered = values - mean
    var = (weights * centered.pow(2)).sum(0, keepdim=True) / denom
    return centered / var.sqrt().clamp_min(1e-4)


def cluster_balanced_redundancy_reduction_objective(
        model, z1, z2, args, epoch, gated=False, stability_weighted=False):
    weights, diagnostics = cluster_balance_weights(
        z1,
        z2,
        args,
        epoch,
        stability_weighted=stability_weighted,
    )
    h1 = model.projection(z1)
    h2 = model.projection(z2)
    if args.shuffle_weights:
        generator = make_control_generator(args, epoch)
        permutation = torch.randperm(h2.size(0), generator=generator).to(h2.device)
        h2 = h2[permutation]
    elif args.random_weights:
        generator = make_control_generator(args, epoch)
        h2 = torch.randn(h2.shape, generator=generator).to(h2.device, dtype=h2.dtype)

    weights = weights.to(h1.device, dtype=h1.dtype)
    h1 = weighted_standardize(h1, weights)
    h2 = weighted_standardize(h2, weights)
    weighted_h2 = h2 * weights.view(-1, 1)
    cross_correlation = torch.mm(h1.t(), weighted_h2) / weights.sum().clamp_min(1e-12)
    diag_mean = torch.diagonal(cross_correlation).mean()
    on_diag = (torch.diagonal(cross_correlation) - 1.0).pow(2).sum()
    off_diag = off_diagonal(cross_correlation).pow(2).sum()
    raw_loss = args.rr_loss_scale * (on_diag + args.rr_offdiag_weight * off_diag)
    gate_scale = raw_loss.new_tensor(1.0)
    if gated:
        temperature = max(args.cbr_gate_temperature, 1e-12)
        gate_scale = torch.sigmoid(
            (diag_mean.detach() - args.cbr_gate_min_diag) / temperature
        )
        min_scale = min(max(args.cbr_gate_min_scale, 0.0), 1.0)
        if min_scale > 0.0:
            gate_scale = gate_scale * (1.0 - min_scale) + min_scale
    loss = raw_loss * gate_scale
    diagnostics.update({
        'cbr_raw_rr_loss': raw_loss.item(),
        'cbr_gate_scale': gate_scale.item(),
        'rr_on_diag_loss': on_diag.item(),
        'rr_off_diag_loss': off_diag.item(),
        'rr_cross_corr_diag_mean': diag_mean.item(),
        'rr_cross_corr_offdiag_mean_abs': (
            off_diagonal(cross_correlation).abs().mean().item()
        ),
    })
    return loss, diagnostics


def raw_complement_objective(raw1, comp1, raw2, comp2):
    raw = torch.cat([raw1, raw2], dim=0)
    comp = torch.cat([comp1, comp2], dim=0)
    raw = (raw - raw.mean(0)) / raw.std(0, unbiased=False).clamp_min(1e-4)
    comp = (comp - comp.mean(0)) / comp.std(0, unbiased=False).clamp_min(1e-4)
    cross_correlation = torch.mm(raw.t(), comp) / raw.size(0)
    loss = cross_correlation.pow(2).mean()
    diagnostics = {
        'raw_complement_corr_mean_abs': cross_correlation.abs().mean().item(),
        'raw_complement_corr_max_abs': cross_correlation.abs().max().item(),
        'raw_complement_raw_norm': raw.norm(dim=1).mean().item(),
        'raw_complement_comp_norm': comp.norm(dim=1).mean().item(),
    }
    return loss, diagnostics


def train_epoch(model, teacher, data, optimizer, config, args, epoch):
    model.train()
    optimizer.zero_grad()

    x_1, edge_index_1, x_2, edge_index_2 = make_views(data, config, args)
    z1 = model(x_1, edge_index_1)
    raw_anchor_1 = getattr(model.encoder, 'last_raw_anchor', None)
    complement_1 = getattr(model.encoder, 'last_complement', None)
    z2 = model(x_2, edge_index_2)
    raw_anchor_2 = getattr(model.encoder, 'last_raw_anchor', None)
    complement_2 = getattr(model.encoder, 'last_complement', None)

    weights = None
    raw_weights = None
    context_gate = None
    use_weighting = (
        args.method in ['es_weighted', 'sgfn', 'pbcl']
        and epoch > args.warmup_epochs
    )
    if use_weighting:
        if args.method == 'es_weighted':
            target = teacher_clean_embeddings(teacher, data)
            raw_weights = embedding_stability_weights(target, z1, z2, args)
            weights = apply_weight_control(raw_weights, args, epoch)
        elif args.method == 'sgfn':
            target = teacher_clean_embeddings(teacher, data)
            raw_weights, context_gate = false_negative_pair_weights(target, data, args)
            weights = apply_pair_weight_control(raw_weights, args, epoch)
            weights = normalize_pair_weights(weights, args)
        else:
            raw_weights = prototype_balance_weights(z1, z2, args, epoch)
            weights = apply_weight_control(raw_weights, args, epoch)

    pair_weights = (
        None if args.method == 'sgfn' or args.no_anchor_weighting else weights
    )
    denominator_weights = (
        weights if args.method == 'es_weighted' and args.negative_weighting else None
    )
    pair_denominator_weights = weights if args.method == 'sgfn' else None
    rr_loss = z1.new_tensor(0.0)
    cbr_rr_loss = z1.new_tensor(0.0)
    rr_diagnostics = {}
    if args.method == 'rr_gcl':
        contrastive_loss, rr_diagnostics = redundancy_reduction_objective(
            model,
            z1,
            z2,
            args,
            epoch,
        )
    else:
        contrastive_loss = model.loss(
            z1,
            z2,
            batch_size=args.batch_size,
            pair_weights=pair_weights,
            denominator_weights=denominator_weights,
            pair_denominator_weights=pair_denominator_weights,
        )
        if args.method == 'hybrid_rr_gcl':
            rr_loss, rr_diagnostics = redundancy_reduction_objective(
                model,
                z1,
                z2,
                args,
                epoch,
            )
        elif args.method in [
                'cbr_gcl',
                'gated_cbr_gcl',
                'stable_cluster_cbr_gcl',
        ] and epoch > args.warmup_epochs:
            cbr_rr_loss, rr_diagnostics = (
                cluster_balanced_redundancy_reduction_objective(
                    model,
                    z1,
                    z2,
                    args,
                    epoch,
                    gated=args.method == 'gated_cbr_gcl',
                    stability_weighted=args.method == 'stable_cluster_cbr_gcl',
                )
            )
        else:
            rr_loss = z1.new_tensor(0.0)
    attraction_loss = z1.new_tensor(0.0)
    prototype_consistency_loss = z1.new_tensor(0.0)
    prototype_balance_loss = z1.new_tensor(0.0)
    raw_complement_loss = z1.new_tensor(0.0)
    prototype_diagnostics = {}
    raw_complement_diagnostics = {}
    if (
        args.method == 'sgfn'
        and use_weighting
        and args.fn_attraction_weight > 0.0
    ):
        attraction_loss = false_negative_attraction_loss(z1, z2, weights)
    if args.method == 'pccl' and epoch > args.warmup_epochs:
        (
            prototype_consistency_loss,
            prototype_balance_loss,
            prototype_diagnostics,
        ) = prototype_consistency_objective(z1, z2, args, epoch)
    if args.method == 'raw_complement_gcl':
        raw_complement_loss, raw_complement_diagnostics = raw_complement_objective(
            raw_anchor_1,
            complement_1,
            raw_anchor_2,
            complement_2,
        )
    loss = (
        contrastive_loss
        + args.hybrid_rr_weight * rr_loss
        + args.cbr_rr_weight * cbr_rr_loss
        + args.fn_attraction_weight * attraction_loss
        + args.pccl_consistency_weight * prototype_consistency_loss
        + args.pccl_balance_weight * prototype_balance_loss
        + args.raw_complement_weight * raw_complement_loss
    )

    loss.backward()
    optimizer.step()
    if teacher is not None:
        update_teacher(teacher, model, args.ema_decay)

    stage = 'weighted' if use_weighting else 'warmup_or_grace'
    if args.method == 'pccl' and epoch > args.warmup_epochs:
        stage = 'prototype'
    if args.method == 'rr_gcl':
        stage = 'redundancy_reduction'
    if args.method == 'hybrid_rr_gcl':
        stage = 'contrastive_plus_rr'
    if args.method in [
            'cbr_gcl',
            'gated_cbr_gcl',
            'stable_cluster_cbr_gcl',
    ] and epoch > args.warmup_epochs:
        stage = (
            'gated_cluster_balanced_rr'
            if args.method == 'gated_cbr_gcl'
            else 'stable_cluster_balanced_rr'
            if args.method == 'stable_cluster_cbr_gcl'
            else 'cluster_balanced_rr'
        )
    log = {
        'loss': loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'rr_loss': rr_loss.item(),
        'cbr_rr_loss': cbr_rr_loss.item(),
        'fn_attraction_loss': attraction_loss.item(),
        'prototype_consistency_loss': prototype_consistency_loss.item(),
        'prototype_balance_loss': prototype_balance_loss.item(),
        'raw_complement_loss': raw_complement_loss.item(),
        'stage': stage,
        'weight_control': weight_control_name(args),
    }
    if args.method == 'raw_complement_gcl':
        log['stage'] = 'raw_complement'
    if hasattr(model.encoder, 'ego_gate'):
        log['ego_gate'] = model.encoder.ego_gate.item()
    if getattr(model.encoder, 'last_graph_gate', None) is not None:
        graph_gate = model.encoder.last_graph_gate.float()
        log.update({
            'graph_gate_mean': graph_gate.mean().item(),
            'graph_gate_std': graph_gate.std(unbiased=False).item(),
            'graph_gate_min': graph_gate.min().item(),
            'graph_gate_max': graph_gate.max().item(),
        })
    log.update(prototype_diagnostics)
    log.update(raw_complement_diagnostics)
    log.update(rr_diagnostics)
    if weights is not None:
        log.update({
            'weight_mean': weights.mean().item(),
            'weight_std': weights.std(unbiased=False).item(),
            'weight_min': weights.min().item(),
            'weight_max': weights.max().item(),
        })
        log.update(weight_diagnostics(weights))
    if raw_weights is not None:
        raw_diag = weight_diagnostics(raw_weights)
        log.update({
            'raw_weight_mean': raw_weights.mean().item(),
            'raw_weight_std': raw_weights.std(unbiased=False).item(),
            'raw_weight_min': raw_weights.min().item(),
            'raw_weight_max': raw_weights.max().item(),
            'raw_weight_ess': raw_diag.get('weight_ess'),
            'raw_weight_ess_ratio': raw_diag.get('weight_ess_ratio'),
        })
    if context_gate is not None:
        log.update({
            'context_gate_mean': context_gate.mean().item(),
            'context_gate_std': context_gate.std(unbiased=False).item(),
            'context_gate_min': context_gate.min().item(),
            'context_gate_max': context_gate.max().item(),
        })
    return log, weights, raw_weights, context_gate


@torch.no_grad()
def encode(model, data, args=None):
    model.eval()
    embeddings = model(data.x, data.edge_index)
    complement = getattr(model.encoder, 'last_complement', None)
    if complement is not None:
        mode = 'anchor' if args is None else args.raw_complement_eval_mode
        if mode == 'hidden':
            return embeddings
        if mode == 'graph':
            graph_context = getattr(model.encoder, 'last_graph_context', None)
            if graph_context is not None:
                return graph_context
            return embeddings
        raw_anchor = F.normalize(data.x.detach(), dim=1)
        complement = F.normalize(complement.detach(), dim=1)
        if mode == 'anchor_graph':
            graph_context = getattr(model.encoder, 'last_graph_context', None)
            if graph_context is not None:
                graph_context = F.normalize(graph_context.detach(), dim=1)
                return torch.cat([raw_anchor, complement, graph_context], dim=1)
        return torch.cat([raw_anchor, complement], dim=1)
    return embeddings


def prepare_save_dir(args, seed):
    if args.save_dir is None:
        return None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    split_dataset = args.dataset in HETEROPHILY_DATASETS
    split_suffix = f'_split{args.split_index}' if args.eval_mode == 'mask' or split_dataset else ''
    control = weight_control_name(args)
    control_suffix = (
        f'_{control}'
        if args.method in [
            'es_weighted',
            'sgfn',
            'pbcl',
            'pccl',
            'rr_gcl',
            'hybrid_rr_gcl',
            'cbr_gcl',
            'gated_cbr_gcl',
            'stable_cluster_cbr_gcl',
        ] and control != 'normal'
        else ''
    )
    run_name = f'{args.dataset}_{args.method}{control_suffix}_seed{seed}{split_suffix}'
    run_dir = save_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f'Run directory already exists and is not empty: {run_dir}. '
                'Use --overwrite or choose a different --save-dir.'
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_train_log(run_dir, row):
    if run_dir is None:
        return
    path = run_dir / 'train_log.csv'
    fieldnames = [
        'epoch',
        'loss',
        'contrastive_loss',
        'rr_loss',
        'cbr_rr_loss',
        'cbr_raw_rr_loss',
        'cbr_gate_scale',
        'fn_attraction_loss',
        'prototype_consistency_loss',
        'prototype_balance_loss',
        'raw_complement_loss',
        'ego_gate',
        'graph_gate_mean',
        'graph_gate_std',
        'graph_gate_min',
        'graph_gate_max',
        'raw_complement_corr_mean_abs',
        'raw_complement_corr_max_abs',
        'raw_complement_raw_norm',
        'raw_complement_comp_norm',
        'prototype_usage_entropy',
        'prototype_usage_max',
        'prototype_usage_min',
        'rr_on_diag_loss',
        'rr_off_diag_loss',
        'rr_cross_corr_diag_mean',
        'rr_cross_corr_offdiag_mean_abs',
        'cbr_num_active_clusters',
        'cbr_cluster_entropy',
        'cbr_cluster_margin_mean',
        'cbr_cluster_margin_std',
        'cbr_cluster_margin_min',
        'cbr_cluster_margin_max',
        'cbr_cluster_assigned_sim_mean',
        'cbr_stability_scale_mean',
        'cbr_stability_scale_std',
        'cbr_stability_scale_min',
        'cbr_stability_scale_max',
        'cbr_weight_mean',
        'cbr_weight_std',
        'cbr_weight_min',
        'cbr_weight_max',
        'cbr_weight_ess',
        'cbr_weight_ess_ratio',
        'stage',
        'weight_mean',
        'weight_std',
        'weight_min',
        'weight_max',
        'weight_ess',
        'weight_ess_ratio',
        'raw_weight_mean',
        'raw_weight_std',
        'raw_weight_min',
        'raw_weight_max',
        'raw_weight_ess',
        'raw_weight_ess_ratio',
        'context_gate_mean',
        'context_gate_std',
        'context_gate_min',
        'context_gate_max',
        'weight_control',
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
        if metric.startswith('_'):
            continue
        row[f'{metric}_mean'] = values['mean']
        row[f'{metric}_std'] = values['std']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_eval_details(run_dir, eval_stats):
    if run_dir is None or eval_stats is None:
        return
    details = eval_stats.get('_details')
    if details is None:
        return
    with (run_dir / 'eval_details.json').open('w') as handle:
        json.dump(details, handle, indent=2)


def run_command(command, cwd=None):
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def repo_root():
    return Path(__file__).resolve().parents[2]


def collect_runtime_metadata(device):
    root = repo_root()
    return {
        'argv': sys.argv,
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'torch_geometric_version': torch_geometric.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device': str(device),
        'cuda_device_name': (
            torch.cuda.get_device_name(device) if device.type == 'cuda' else None
        ),
        'git_commit': run_command(['git', 'rev-parse', 'HEAD'], cwd=root),
        'git_status_short': run_command(['git', 'status', '--short'], cwd=root),
        'grace_submodule_status': run_command(
            ['git', 'submodule', 'status', 'baselines/GRACE'],
            cwd=root,
        ),
    }


def save_metadata(run_dir, args, config, seed, device, eval_stats):
    if run_dir is None:
        return
    metadata = {
        'dataset': args.dataset,
        'method': args.method,
        'seed': seed,
        'model_seed': seed,
        'split_index': args.split_index,
        'weight_control': weight_control_name(args),
        'ego_gate_init': args.ego_gate_init,
        'graph_gate_temperature': args.graph_gate_temperature,
        'graph_gate_threshold': args.graph_gate_threshold,
        'graph_gate_min': args.graph_gate_min,
        'graph_gate_max': args.graph_gate_max,
        'fn_risk_margin': args.fn_risk_margin,
        'fn_risk_temperature': args.fn_risk_temperature,
        'fn_attenuation_power': args.fn_attenuation_power,
        'fn_attraction_weight': args.fn_attraction_weight,
        'fn_consensus': args.fn_consensus,
        'fn_context_gate': args.fn_context_gate,
        'fn_context_temperature': args.fn_context_temperature,
        'fn_context_threshold': args.fn_context_threshold,
        'fn_degree_threshold': args.fn_degree_threshold,
        'fn_context_pair_mode': args.fn_context_pair_mode,
        'spectral_mix_mode': args.spectral_mix_mode,
        'spectral_mix_temperature': args.spectral_mix_temperature,
        'spectral_mix_jitter': args.spectral_mix_jitter,
        'spectral_high_scale': args.spectral_high_scale,
        'spectral_residual_alpha': args.spectral_residual_alpha,
        'pbcl_num_prototypes': args.pbcl_num_prototypes,
        'resolved_pbcl_num_prototypes': args.resolved_pbcl_num_prototypes,
        'pbcl_kmeans_iters': args.pbcl_kmeans_iters,
        'pbcl_weight_power': args.pbcl_weight_power,
        'pbcl_min_weight': args.pbcl_min_weight,
        'pbcl_max_weight': args.pbcl_max_weight,
        'pccl_num_prototypes': args.pccl_num_prototypes,
        'resolved_pccl_num_prototypes': args.resolved_pccl_num_prototypes,
        'pccl_kmeans_iters': args.pccl_kmeans_iters,
        'pccl_prototype_temperature': args.pccl_prototype_temperature,
        'pccl_target_temperature': args.pccl_target_temperature,
        'pccl_consistency_weight': args.pccl_consistency_weight,
        'pccl_balance_weight': args.pccl_balance_weight,
        'rr_offdiag_weight': args.rr_offdiag_weight,
        'rr_loss_scale': args.rr_loss_scale,
        'hybrid_rr_weight': args.hybrid_rr_weight,
        'cbr_rr_weight': args.cbr_rr_weight,
        'cbr_num_clusters': args.cbr_num_clusters,
        'resolved_cbr_num_clusters': args.resolved_cbr_num_clusters,
        'cbr_kmeans_iters': args.cbr_kmeans_iters,
        'cbr_min_weight': args.cbr_min_weight,
        'cbr_max_weight': args.cbr_max_weight,
        'cbr_gate_min_diag': args.cbr_gate_min_diag,
        'cbr_gate_temperature': args.cbr_gate_temperature,
        'cbr_gate_min_scale': args.cbr_gate_min_scale,
        'cbr_stability_min_margin': args.cbr_stability_min_margin,
        'cbr_stability_temperature': args.cbr_stability_temperature,
        'cbr_stability_min_scale': args.cbr_stability_min_scale,
        'raw_complement_weight': args.raw_complement_weight,
        'raw_complement_detach_anchor': args.raw_complement_detach_anchor,
        'raw_complement_eval_mode': args.raw_complement_eval_mode,
        'pair_shuffle_mode': args.pair_shuffle_mode,
        'pair_normalization': args.pair_normalization,
        'pair_reallocation_alpha': args.pair_reallocation_alpha,
        'device': str(device),
        'args': vars(args),
        'config': config,
        'eval_stats': eval_stats,
        'runtime': collect_runtime_metadata(device),
    }
    with (run_dir / 'metadata.json').open('w') as handle:
        json.dump(metadata, handle, indent=2, default=str)


def save_artifacts(run_dir, model, data, embeddings, final_weights, final_raw_weights,
                   final_context_gate, args, config):
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
        'final_raw_weights': (
            None if final_raw_weights is None else final_raw_weights.detach().cpu()
        ),
        'final_context_gate': (
            None if final_context_gate is None else final_context_gate.detach().cpu()
        ),
        'final_raw_anchor': (
            None if getattr(model.encoder, 'last_raw_anchor', None) is None
            else model.encoder.last_raw_anchor.detach().cpu()
        ),
        'final_complement': (
            None if getattr(model.encoder, 'last_complement', None) is None
            else model.encoder.last_complement.detach().cpu()
        ),
        'final_graph_context': (
            None if getattr(model.encoder, 'last_graph_context', None) is None
            else model.encoder.last_graph_context.detach().cpu()
        ),
        'weight_control': weight_control_name(args),
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'config': config,
    }, run_dir / 'artifacts.pt')


def main():
    args = parse_args()
    validate_args(args)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    seed = config['seed'] if args.seed is None else args.seed
    args.resolved_seed = seed
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    set_seed(seed)

    device = get_device(args.gpu_id)
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0].to(device)
    train_mask, val_mask, test_mask = get_split_masks(data, args.split_index)

    model = build_model(config, dataset, device, args)
    args.resolved_pbcl_num_prototypes = (
        args.pbcl_num_prototypes if args.pbcl_num_prototypes > 0
        else dataset.num_classes
    )
    args.resolved_pccl_num_prototypes = (
        args.pccl_num_prototypes if args.pccl_num_prototypes > 0
        else dataset.num_classes
    )
    args.resolved_cbr_num_clusters = (
        args.cbr_num_clusters if args.cbr_num_clusters > 0
        else dataset.num_classes
    )
    teacher = build_teacher(model) if args.method in ['es_weighted', 'sgfn'] else None
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
    if args.method in ['es_weighted', 'sgfn', 'pbcl', 'pccl']:
        print(
            f'(I) | warmup_epochs={args.warmup_epochs}, ema_decay={args.ema_decay}, '
            f'anchor_weighting={not args.no_anchor_weighting}, '
            f'negative_weighting={args.negative_weighting}, '
            f'weight_control={weight_control_name(args)}'
        )
        if args.method == 'sgfn':
            print(
                f'(I) | fn_risk_margin={args.fn_risk_margin}, '
                f'fn_risk_temperature={args.fn_risk_temperature}, '
                f'fn_attenuation_power={args.fn_attenuation_power}, '
                f'fn_attraction_weight={args.fn_attraction_weight}, '
                f'fn_consensus={args.fn_consensus}, '
                f'fn_context_gate={args.fn_context_gate}, '
                f'fn_context_pair_mode={args.fn_context_pair_mode}, '
                f'pair_shuffle_mode={args.pair_shuffle_mode}, '
                f'pair_normalization={args.pair_normalization}, '
                f'pair_reallocation_alpha={args.pair_reallocation_alpha}'
            )
        if args.method == 'pbcl':
            print(
                f'(I) | pbcl_num_prototypes={args.resolved_pbcl_num_prototypes}, '
                f'pbcl_kmeans_iters={args.pbcl_kmeans_iters}, '
                f'pbcl_weight_power={args.pbcl_weight_power}, '
                f'pbcl_min_weight={args.pbcl_min_weight}, '
                f'pbcl_max_weight={args.pbcl_max_weight}'
            )
        if args.method == 'pccl':
            print(
                f'(I) | pccl_num_prototypes={args.resolved_pccl_num_prototypes}, '
                f'pccl_kmeans_iters={args.pccl_kmeans_iters}, '
                f'pccl_prototype_temperature={args.pccl_prototype_temperature}, '
                f'pccl_target_temperature={args.pccl_target_temperature}, '
                f'pccl_consistency_weight={args.pccl_consistency_weight}, '
                f'pccl_balance_weight={args.pccl_balance_weight}'
            )
    if args.method in ['rr_gcl', 'hybrid_rr_gcl']:
        print(
            f'(I) | rr_offdiag_weight={args.rr_offdiag_weight}, '
            f'rr_loss_scale={args.rr_loss_scale}, '
            f'hybrid_rr_weight={args.hybrid_rr_weight}, '
            f'positive_control={weight_control_name(args)}'
        )
    if args.method in ['cbr_gcl', 'gated_cbr_gcl', 'stable_cluster_cbr_gcl']:
        print(
            f'(I) | warmup_epochs={args.warmup_epochs}, '
            f'cbr_num_clusters={args.resolved_cbr_num_clusters}, '
            f'cbr_kmeans_iters={args.cbr_kmeans_iters}, '
            f'cbr_rr_weight={args.cbr_rr_weight}, '
            f'rr_offdiag_weight={args.rr_offdiag_weight}, '
            f'cbr_gate_min_diag={args.cbr_gate_min_diag}, '
            f'cbr_gate_temperature={args.cbr_gate_temperature}, '
            f'cbr_stability_min_margin={args.cbr_stability_min_margin}, '
            f'cbr_stability_temperature={args.cbr_stability_temperature}, '
            f'cbr_stability_min_scale={args.cbr_stability_min_scale}, '
            f'positive_control={weight_control_name(args)}'
        )
    if args.method == 'spectral_mix':
        print(
            f'(I) | spectral_mix_mode={args.spectral_mix_mode}, '
            f'spectral_mix_temperature={args.spectral_mix_temperature}, '
            f'spectral_mix_jitter={args.spectral_mix_jitter}, '
            f'spectral_high_scale={args.spectral_high_scale}, '
            f'spectral_residual_alpha={args.spectral_residual_alpha}'
        )
    if args.method == 'residual_grace':
        print(f'(I) | ego_gate_init={args.ego_gate_init}')
    if args.method == 'ego_grace':
        print('(I) | ego_encoder=mlp_only')
    if args.method == 'gated_ego_graph_grace':
        print(
            f'(I) | graph_gate_temperature={args.graph_gate_temperature}, '
            f'graph_gate_threshold={args.graph_gate_threshold}, '
            f'graph_gate_min={args.graph_gate_min}, '
            f'graph_gate_max={args.graph_gate_max}'
        )
    if args.method == 'raw_complement_gcl':
        print(
            f'(I) | raw_complement_weight={args.raw_complement_weight}, '
            f'raw_complement_detach_anchor={args.raw_complement_detach_anchor}, '
            f'raw_complement_eval_mode={args.raw_complement_eval_mode}'
        )

    start = t()
    prev = start
    final_weights = None
    final_raw_weights = None
    final_context_gate = None
    for epoch in range(1, config['num_epochs'] + 1):
        log, epoch_weights, epoch_raw_weights, epoch_context_gate = train_epoch(
            model,
            teacher,
            data,
            optimizer,
            config,
            args,
            epoch,
        )
        if epoch_weights is not None:
            final_weights = epoch_weights
        if epoch_raw_weights is not None:
            final_raw_weights = epoch_raw_weights
        if epoch_context_gate is not None:
            final_context_gate = epoch_context_gate

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
    embeddings = encode(model, data, args)
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
    save_eval_details(run_dir, eval_stats)
    save_metadata(run_dir, args, config, seed, device, eval_stats)
    save_artifacts(
        run_dir,
        model,
        data,
        embeddings,
        final_weights,
        final_raw_weights,
        final_context_gate,
        args,
        config,
    )


if __name__ == '__main__':
    main()
