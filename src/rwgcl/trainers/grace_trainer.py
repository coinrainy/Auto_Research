"""Minimal GRACE trainer for smoke experiments."""

from __future__ import annotations

import time

import torch

from rwgcl.augment import augment_graph
from rwgcl.data import get_split_masks, load_pyg_dataset, pyg_data_stats
from rwgcl.encoders import GRACEModel, encoder_from_config
from rwgcl.evaluation import linear_probe
from rwgcl.logging_utils import append_csv, ensure_dir, now_utc, write_csv, write_json, write_yaml
from rwgcl.losses import info_nce_loss

from .base_trainer import BaseTrainer, RunResult


class GRACETrainer(BaseTrainer):
    trainer_name = "grace"

    def run(self, mode: str = "scaffold") -> RunResult:
        if mode == "scaffold":
            return self.run_scaffold()
        if mode == "execute":
            return self.run_execute()
        raise ValueError(f"Unknown mode: {mode}")

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    def run_execute(self) -> RunResult:
        ensure_dir(self.run_dir)
        dataset = load_pyg_dataset(self.dataset_spec)
        data = dataset[0]
        stats = pyg_data_stats(dataset, split_index=self.split_index)
        device = self._device()
        data = data.to(device)
        train_mask, val_mask, test_mask = get_split_masks(data, split_index=self.split_index)

        encoder_spec = encoder_from_config(self.method_config.get("encoder", {}))
        model = GRACEModel(in_dim=int(dataset.num_features), spec=encoder_spec).to(device)
        training_cfg = self.method_config.get("training", {})
        augmentation_cfg = self.method_config.get("augmentation", {})
        eval_cfg = self.method_config.get("evaluation", {})
        epochs = int(training_cfg.get("epochs", 200))
        lr = float(training_cfg.get("lr", 0.001))
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        temperature = float(training_cfg.get("temperature", 0.5))
        edge_drop = float(augmentation_cfg.get("edge_drop", 0.2))
        feature_mask = float(augmentation_cfg.get("feature_mask", 0.2))
        eval_epochs = int(eval_cfg.get("eval_epochs", 100))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_rows = []
        start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            x1, edge_index1 = augment_graph(data.x, data.edge_index, edge_drop=edge_drop, feature_mask=feature_mask)
            x2, edge_index2 = augment_graph(data.x, data.edge_index, edge_drop=edge_drop, feature_mask=feature_mask)
            h1 = model.encode(x1, edge_index1)
            h2 = model.encode(x2, edge_index2)
            z1 = model.project(h1)
            z2 = model.project(h2)
            loss = info_nce_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            train_rows.append(
                {
                    "epoch": epoch,
                    "split": "train",
                    "loss": f"{float(loss.item()):.6f}",
                    "metric": "",
                    "status": "ok",
                    "notes": "",
                }
            )

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.x, data.edge_index).detach()
        probe = linear_probe(
            embeddings=embeddings,
            labels=data.y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_classes=int(dataset.num_classes),
            epochs=eval_epochs,
        )
        elapsed = time.time() - start

        config_snapshot = {
            "created_at": now_utc(),
            "run_id": self.run_id,
            "status": "completed",
            "method": self.method_config,
            "dataset": stats,
            "seed": self.seed,
            "model_seed": self.seed,
            "split_index": self.split_index,
            "trainer": self.trainer_name,
            "device": str(device),
            "elapsed_seconds": elapsed,
        }
        write_yaml(self.run_dir / "config.yaml", config_snapshot)
        write_json(self.run_dir / "run_metadata.json", config_snapshot)
        write_json(self.run_dir / "dataset_stats.json", stats)
        write_csv(
            self.run_dir / "train_log.csv",
            train_rows,
            ["epoch", "split", "loss", "metric", "status", "notes"],
        )
        torch.save(
            {
                "embeddings": embeddings.cpu(),
                "labels": data.y.detach().cpu(),
                "train_mask": train_mask.detach().cpu(),
                "val_mask": val_mask.detach().cpu(),
                "test_mask": test_mask.detach().cpu(),
            },
            self.run_dir / "embeddings.pt",
        )

        metrics_path = self.results_dir / "metrics" / "main_results.csv"
        append_csv(
            metrics_path,
            {
                "run_id": self.run_id,
                "dataset": self.dataset_spec.name,
                "dataset_group": self.dataset_spec.group,
                "method": self.method_name,
                "trainer": self.trainer_name,
                "seed": self.seed,
                "model_seed": self.seed,
                "split_index": self.split_index,
                "metric": self.dataset_spec.metric,
                "value": f"{probe['test_accuracy']:.6f}",
                "status": "completed",
                "notes": (
                    f"val_accuracy={probe['val_accuracy']:.6f}; "
                    f"elapsed_seconds={elapsed:.2f}; "
                    f"split_index={self.split_index}; model_seed={self.seed}"
                ),
            },
            [
                "run_id",
                "dataset",
                "dataset_group",
                "method",
                "trainer",
                "seed",
                "model_seed",
                "split_index",
                "metric",
                "value",
                "status",
                "notes",
            ],
        )
        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            metrics_path=metrics_path,
            status="completed",
        )
