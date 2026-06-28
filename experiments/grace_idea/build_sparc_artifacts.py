import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch

from evaluate_propagation_calibration import (
    build_variants,
    find_artifacts,
    load_metadata,
)
from train import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', default=[])
    parser.add_argument('--runs-dir', action='append', default=[])
    parser.add_argument('--out-dir', default='runs/sparc_artifacts')
    parser.add_argument('--mode', default='ssl_resid1',
                        choices=['ssl_prop1', 'ssl_prop2', 'ssl_resid1', 'ssl_resid2'])
    parser.add_argument('--method-suffix', default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def metadata_value(metadata, artifact_args, key, default=None):
    return artifact_args.get(key, metadata.get(key, default))


def artifact_output_dir(out_dir, dataset, method, seed, split_index):
    name = f'{dataset}_{method}_seed{seed}_split{split_index}'
    return Path(out_dir) / name


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(payload, handle, indent=2, default=str)


def build_one(artifact_path, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        artifact = torch.load(artifact_path, map_location='cpu')
    metadata, artifact_args = load_metadata(artifact_path, artifact)
    dataset_name = metadata_value(metadata, artifact_args, 'dataset')
    source_method = metadata_value(metadata, artifact_args, 'method')
    seed = int(metadata_value(metadata, artifact_args, 'resolved_seed',
                              metadata_value(metadata, artifact_args, 'seed', 0)) or 0)
    split_index = int(metadata_value(metadata, artifact_args, 'split_index', 0) or 0)
    if dataset_name is None or source_method is None:
        raise ValueError(f'Missing dataset/method metadata in {artifact_path}')

    dataset = get_dataset(str(Path('~/datasets').expanduser() / dataset_name), dataset_name)
    data = dataset[0]
    embeddings = artifact['embeddings'].detach().cpu().float()
    max_hop = 2 if args.mode.endswith('2') else 1
    variants = build_variants(embeddings, data.edge_index, max_hop=max_hop)
    calibrated = variants[args.mode].detach().cpu()

    suffix = args.method_suffix
    if suffix is None:
        suffix = f'sparc_{args.mode.replace("ssl_", "")}'
    method = f'{source_method}_{suffix}'
    out_dir = artifact_output_dir(args.out_dir, dataset_name, method, seed, split_index)
    artifact_out = out_dir / 'artifacts.pt'
    if artifact_out.exists() and not args.overwrite:
        raise FileExistsError(f'{artifact_out} exists; use --overwrite to replace it.')
    out_dir.mkdir(parents=True, exist_ok=True)

    new_args = dict(artifact_args)
    new_args.update({
        'dataset': dataset_name,
        'method': method,
        'source_method': source_method,
        'source_artifact_path': str(artifact_path),
        'sparc_mode': args.mode,
        'resolved_seed': seed,
        'seed': seed,
        'split_index': split_index,
    })
    new_artifact = dict(artifact)
    new_artifact.update({
        'embeddings': calibrated,
        'source_embeddings_dim': int(embeddings.size(1)),
        'sparc_mode': args.mode,
        'source_method': source_method,
        'source_artifact_path': str(artifact_path),
        'args': new_args,
    })
    torch.save(new_artifact, artifact_out)
    new_metadata = dict(metadata)
    new_metadata.update({
        'dataset': dataset_name,
        'method': method,
        'source_method': source_method,
        'source_artifact_path': str(artifact_path),
        'sparc_mode': args.mode,
        'seed': seed,
        'resolved_seed': seed,
        'split_index': split_index,
        'embedding_dim': int(calibrated.size(1)),
        'source_embedding_dim': int(embeddings.size(1)),
        'args': new_args,
    })
    write_json(out_dir / 'metadata.json', new_metadata)
    return {
        'source': str(artifact_path),
        'artifact': str(artifact_out),
        'dataset': dataset_name,
        'method': method,
        'seed': seed,
        'split_index': split_index,
        'mode': args.mode,
        'source_dim': int(embeddings.size(1)),
        'embedding_dim': int(calibrated.size(1)),
    }


def main():
    args = parse_args()
    artifact_args = SimpleNamespace(run_dir=args.run_dir, runs_dir=args.runs_dir)
    rows = [build_one(path, args) for path in find_artifacts(artifact_args)]
    for row in rows:
        print(
            f"(artifact) dataset={row['dataset']} seed={row['seed']} "
            f"mode={row['mode']} dim={row['source_dim']}->{row['embedding_dim']} "
            f"path={row['artifact']}"
        )
    print(f'(I) | built={len(rows)}, out_dir={args.out_dir}')


if __name__ == '__main__':
    main()
