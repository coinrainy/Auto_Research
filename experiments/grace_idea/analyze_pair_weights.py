import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--aggregate-out', default=None)
    parser.add_argument('--control-paired-out', default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--pattern', default='*')
    return parser.parse_args()


def load_run(run_dir):
    artifacts_path = run_dir / 'artifacts.pt'
    metadata_path = run_dir / 'metadata.json'
    if not artifacts_path.exists() or not metadata_path.exists():
        return None, None
    try:
        artifacts = torch.load(
            artifacts_path,
            map_location='cpu',
            weights_only=True,
        )
    except TypeError:
        artifacts = torch.load(artifacts_path, map_location='cpu')
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    return artifacts, metadata


def select_mask(artifacts, split_index):
    test_mask = artifacts.get('test_mask')
    if test_mask is None:
        return None
    if test_mask.dim() == 1:
        return test_mask.bool()
    return test_mask[:, split_index].bool()


def offdiag_mask(num_nodes):
    return ~torch.eye(num_nodes, dtype=torch.bool)


def mean_or_empty(values):
    if values.numel() == 0:
        return ''
    return values.float().mean().item()


def pressure_stats(embeddings, labels, weights, tau):
    num_nodes = labels.numel()
    valid = offdiag_mask(num_nodes)
    same_label = labels.view(-1, 1).eq(labels.view(1, -1)) & valid
    diff_label = labels.view(-1, 1).ne(labels.view(1, -1)) & valid

    normalized = F.normalize(embeddings.float(), dim=1)
    sim_pressure = torch.exp(torch.mm(normalized, normalized.t()) / tau)
    sim_pressure = sim_pressure.masked_fill(~valid, 0.0)

    weighted_pressure = sim_pressure * weights.float().clamp_min(0.0)
    weighted_total = weighted_pressure.sum(1).clamp_min(1e-12)
    unweighted_total = sim_pressure.sum(1).clamp_min(1e-12)
    weighted_fn_share = (
        weighted_pressure.masked_fill(~same_label, 0.0).sum(1) / weighted_total
    )
    unweighted_fn_share = (
        sim_pressure.masked_fill(~same_label, 0.0).sum(1) / unweighted_total
    )

    return {
        'pair_keep_mean': weights[valid].float().mean().item(),
        'pair_keep_std': weights[valid].float().std(unbiased=False).item(),
        'false_negative_keep_mean': mean_or_empty(weights[same_label]),
        'true_negative_keep_mean': mean_or_empty(weights[diff_label]),
        'fn_keep_minus_tn_keep': (
            mean_or_empty(weights[same_label]) - mean_or_empty(weights[diff_label])
            if weights[same_label].numel() > 0 and weights[diff_label].numel() > 0
            else ''
        ),
        'weighted_fn_pressure_share_mean': weighted_fn_share.mean().item(),
        'unweighted_fn_pressure_share_mean': unweighted_fn_share.mean().item(),
        'weighted_minus_unweighted_fn_pressure': (
            weighted_fn_share.mean() - unweighted_fn_share.mean()
        ).item(),
    }


def run_stats(run_dir, artifacts, metadata, tau_override):
    final_weights = artifacts.get('final_weights')
    if final_weights is None or final_weights.dim() != 2:
        return None

    labels = artifacts['labels'].view(-1)
    embeddings = artifacts['embeddings']
    split_index = int(metadata.get('split_index', 0))
    tau = tau_override
    if tau is None:
        tau = float(metadata.get('config', {}).get('tau', 0.5))

    row = {
        'run_dir': str(run_dir),
        'dataset': metadata.get('dataset', ''),
        'method': metadata.get('method', ''),
        'weight_control': metadata.get('weight_control', ''),
        'model_seed': metadata.get('model_seed', metadata.get('seed', '')),
        'split_index': split_index,
        'tau': tau,
        'fn_risk_margin': metadata.get('fn_risk_margin', ''),
        'fn_risk_temperature': metadata.get('fn_risk_temperature', ''),
        'fn_attenuation_power': metadata.get('fn_attenuation_power', ''),
        'fn_attraction_weight': metadata.get('fn_attraction_weight', ''),
        'fn_consensus': metadata.get('fn_consensus', ''),
        'pair_shuffle_mode': metadata.get('pair_shuffle_mode', ''),
        'pair_normalization': metadata.get('pair_normalization', ''),
        'pair_reallocation_alpha': metadata.get('pair_reallocation_alpha', ''),
    }

    row.update(pressure_stats(embeddings, labels, final_weights, tau))

    test_mask = select_mask(artifacts, split_index)
    if test_mask is not None and test_mask.any():
        test_indices = test_mask.nonzero(as_tuple=False).view(-1)
        sub_weights = final_weights[test_indices]
        sub_embeddings = embeddings[test_indices]
        sub_labels = labels[test_indices]
        # This test-only subgraph diagnostic is intentionally reported separately:
        # it asks whether held-out anchors receive lower same-label pressure.
        row.update({
            f'test_{key}': value
            for key, value in pressure_stats(
                sub_embeddings,
                sub_labels,
                sub_weights[:, test_indices],
                tau,
            ).items()
        })
    return row


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_value(row, key):
    value = row.get(key, '')
    if value == '' or value is None:
        return None
    return float(value)


def aggregate_rows(rows):
    metric_keys = [
        'fn_keep_minus_tn_keep',
        'weighted_minus_unweighted_fn_pressure',
        'test_fn_keep_minus_tn_keep',
        'test_weighted_minus_unweighted_fn_pressure',
        'pair_keep_mean',
        'pair_keep_std',
    ]
    groups = {}
    for row in rows:
        key = (
            row.get('dataset', ''),
            row.get('method', ''),
            row.get('weight_control', ''),
            row.get('fn_consensus', ''),
            row.get('pair_normalization', ''),
            row.get('pair_reallocation_alpha', ''),
        )
        groups.setdefault(key, []).append(row)

    output = []
    for key, group in sorted(groups.items()):
        (
            dataset,
            method,
            weight_control,
            fn_consensus,
            pair_normalization,
            pair_reallocation_alpha,
        ) = key
        out = {
            'dataset': dataset,
            'method': method,
            'weight_control': weight_control,
            'fn_consensus': fn_consensus,
            'pair_normalization': pair_normalization,
            'pair_reallocation_alpha': pair_reallocation_alpha,
            'num_runs': len(group),
        }
        for metric in metric_keys:
            values = [numeric_value(row, metric) for row in group]
            values = [value for value in values if value is not None]
            if not values:
                continue
            out[f'{metric}_mean'] = mean(values)
            out[f'{metric}_pop_std'] = pstdev(values) if len(values) > 1 else 0.0
            out[f'{metric}_negative'] = sum(value < 0 for value in values)
            out[f'{metric}_zero'] = sum(value == 0 for value in values)
            out[f'{metric}_positive'] = sum(value > 0 for value in values)
        output.append(out)
    return output


def control_paired_rows(rows):
    normal_by_key = {}
    controls = {}
    for row in rows:
        key = (
            row.get('dataset', ''),
            row.get('method', ''),
            row.get('model_seed', ''),
            row.get('split_index', ''),
            row.get('fn_consensus', ''),
            row.get('pair_normalization', ''),
            row.get('pair_reallocation_alpha', ''),
        )
        if row.get('weight_control') == 'normal':
            normal_by_key[key] = row
        else:
            controls.setdefault(key, []).append(row)

    metrics = [
        'fn_keep_minus_tn_keep',
        'weighted_minus_unweighted_fn_pressure',
        'test_fn_keep_minus_tn_keep',
        'test_weighted_minus_unweighted_fn_pressure',
    ]
    output = []
    for key, normal in sorted(normal_by_key.items()):
        for control in controls.get(key, []):
            row = {
                'dataset': key[0],
                'method': key[1],
                'model_seed': key[2],
                'split_index': key[3],
                'fn_consensus': key[4],
                'pair_normalization': key[5],
                'pair_reallocation_alpha': key[6],
                'control': control.get('weight_control', ''),
            }
            for metric in metrics:
                normal_value = numeric_value(normal, metric)
                control_value = numeric_value(control, metric)
                if normal_value is None or control_value is None:
                    continue
                row[f'normal_{metric}'] = normal_value
                row[f'control_{metric}'] = control_value
                row[f'normal_minus_control_{metric}'] = normal_value - control_value
            output.append(row)
    return output


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = []
    for run_dir in sorted(path for path in runs_dir.glob(args.pattern) if path.is_dir()):
        artifacts, metadata = load_run(run_dir)
        if artifacts is None:
            continue
        row = run_stats(run_dir, artifacts, metadata, args.tau)
        if row is not None:
            rows.append(row)
    write_rows(args.out, rows)
    if args.aggregate_out is not None:
        write_rows(args.aggregate_out, aggregate_rows(rows))
    if args.control_paired_out is not None:
        write_rows(args.control_paired_out, control_paired_rows(rows))
    print(f'pair_weight_rows={len(rows)} out={args.out}')


if __name__ == '__main__':
    main()
