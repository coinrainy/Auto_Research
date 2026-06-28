import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WikipediaNetwork


def mask_to_index(mask):
    return torch.where(mask.bool())[0].cpu().numpy()


def export_dataset(dataset_name, pyg_root, out_root):
    dataset = WikipediaNetwork(
        root=str(pyg_root),
        name=dataset_name.lower(),
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0]
    lower_name = dataset_name.lower()
    data_root = out_root / 'dataset' / 'non_homophilous_benchmark_data'
    split_root = data_root / 'splits'
    data_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    scipy.io.savemat(
        data_root / f'{lower_name}.mat',
        {
            'edge_index': data.edge_index.cpu().numpy().astype(np.int64),
            'node_feat': data.x.cpu().numpy().astype(np.float32),
            'label': data.y.cpu().numpy().astype(np.int64),
        },
    )

    splits = []
    for split_idx in range(data.train_mask.size(1)):
        splits.append({
            'train': mask_to_index(data.train_mask[:, split_idx]),
            'valid': mask_to_index(data.val_mask[:, split_idx]),
            'test': mask_to_index(data.test_mask[:, split_idx]),
        })
    np.save(
        split_root / f'{lower_name}-splits.npy',
        np.array(splits, dtype=object),
        allow_pickle=True,
    )
    print(
        f'(I) exported {dataset_name}: nodes={data.num_nodes}, '
        f'features={data.num_features}, splits={len(splits)}, '
        f'out={data_root}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['Chameleon', 'Squirrel'])
    parser.add_argument(
        '--pyg-root',
        default='../../data',
        help='PyG WikipediaNetwork cache root, relative to this script working directory.',
    )
    parser.add_argument(
        '--out-root',
        default='../../third_party_baselines/SPGCL',
        help='SPGCL repository root, relative to this script working directory.',
    )
    args = parser.parse_args()
    pyg_root = Path(args.pyg_root).resolve()
    out_root = Path(args.out_root).resolve()
    for dataset_name in args.datasets:
        export_dataset(dataset_name, pyg_root, out_root)


if __name__ == '__main__':
    main()
