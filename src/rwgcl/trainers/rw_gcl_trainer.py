"""Two-stage reliability-weighted GCL trainer."""

from __future__ import annotations

import copy
import time

import torch

from rwgcl.augment import augment_graph
from rwgcl.data import get_split_masks, load_pyg_dataset, pyg_data_stats
from rwgcl.encoders import GRACEModel, encoder_from_config
from rwgcl.evaluation import linear_probe
from rwgcl.logging_utils import append_csv, ensure_dir, now_utc, write_csv, write_json, write_yaml
from rwgcl.losses import info_nce_loss, weighted_info_nce_loss
from rwgcl.reliability import positive_pair_reliability, reliability_from_config, scaffold_reliability_summary

from .base_trainer import BaseTrainer, RunResult


class RWGCLTrainer(BaseTrainer):
    trainer_name = "rw_gcl"

    def run(self, mode: str = "scaffold") -> RunResult:
        if mode == "scaffold":
            return self.run_scaffold()
        if mode == "execute":
            return self.run_execute()
        raise ValueError(f"Unknown mode: {mode}")

    def run_scaffold(self) -> RunResult:
        result = super().run_scaffold()
        write_json(
            self.run_dir / "reliability_plan.json",
            {
                "run_id": self.run_id,
                "status": "scaffold_only",
                "reliability": scaffold_reliability_summary(self.method_config),
                "next_step": "Implement warm-up embeddings, EMA teacher, and reliability-weighted InfoNCE.",
            },
        )
        return result

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    @staticmethod
    @torch.no_grad()
    def _update_ema(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float) -> None:
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    @staticmethod
    def _freeze(module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    def run_execute(self) -> RunResult:
        ensure_dir(self.run_dir)
        dataset = load_pyg_dataset(self.dataset_spec)
        data = dataset[0]
        stats = pyg_data_stats(dataset, split_index=self.split_index)
        device = self._device()
        data = data.to(device)
        train_mask, val_mask, test_mask = get_split_masks(data, split_index=self.split_index)

        encoder_spec = encoder_from_config(self.method_config.get("encoder", {}))
        student = GRACEModel(in_dim=int(dataset.num_features), spec=encoder_spec).to(device)
        teacher = copy.deepcopy(student).to(device)
        self._freeze(teacher)

        training_cfg = self.method_config.get("training", {})
        augmentation_cfg = self.method_config.get("augmentation", {})
        eval_cfg = self.method_config.get("evaluation", {})
        reliability_cfg = self.method_config.get("reliability", {})
        reliability_spec = reliability_from_config(self.method_config)

        warmup_epochs = int(training_cfg.get("warmup_epochs", 50))
        stage2_epochs = int(training_cfg.get("stage2_epochs", 150))
        lr = float(training_cfg.get("lr", 0.001))
        weight_decay = float(training_cfg.get("weight_decay", 0.0))
        temperature = float(training_cfg.get("temperature", 0.5))
        ema_momentum = float(training_cfg.get("ema_momentum", 0.99))
        warmup_edge_drop = float(augmentation_cfg.get("warmup_edge_drop", 0.1))
        warmup_feature_mask = float(augmentation_cfg.get("warmup_feature_mask", 0.1))
        stage2_edge_drop = float(augmentation_cfg.get("stage2_edge_drop", 0.2))
        stage2_feature_mask = float(augmentation_cfg.get("stage2_feature_mask", 0.2))
        eval_epochs = int(eval_cfg.get("eval_epochs", 100))
        shuffled_control = bool(reliability_cfg.get("shuffled_control", False))

        optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
        train_rows = []
        last_reliability = None
        last_summary = {}
        start = time.time()

        for epoch in range(1, warmup_epochs + 1):
            student.train()
            optimizer.zero_grad()
            x1, edge_index1 = augment_graph(
                data.x,
                data.edge_index,
                edge_drop=warmup_edge_drop,
                feature_mask=warmup_feature_mask,
            )
            x2, edge_index2 = augment_graph(
                data.x,
                data.edge_index,
                edge_drop=warmup_edge_drop,
                feature_mask=warmup_feature_mask,
            )
            h1 = student.encode(x1, edge_index1)
            h2 = student.encode(x2, edge_index2)
            z1 = student.project(h1)
            z2 = student.project(h2)
            loss = info_nce_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()
            self._update_ema(student, teacher, momentum=ema_momentum)
            train_rows.append(
                {
                    "epoch": epoch,
                    "split": "warmup",
                    "loss": f"{float(loss.item()):.6f}",
                    "metric": "",
                    "status": "ok",
                    "notes": "standard_infonce",
                }
            )

        for stage_epoch in range(1, stage2_epochs + 1):
            epoch = warmup_epochs + stage_epoch
            student.train()
            optimizer.zero_grad()
            x1, edge_index1 = augment_graph(
                data.x,
                data.edge_index,
                edge_drop=stage2_edge_drop,
                feature_mask=stage2_feature_mask,
            )
            x2, edge_index2 = augment_graph(
                data.x,
                data.edge_index,
                edge_drop=stage2_edge_drop,
                feature_mask=stage2_feature_mask,
            )
            h1 = student.encode(x1, edge_index1)
            h2 = student.encode(x2, edge_index2)
            z1 = student.project(h1)
            z2 = student.project(h2)
            with torch.no_grad():
                teacher_h1 = teacher.encode(x1, edge_index1)
                teacher_h2 = teacher.encode(x2, edge_index2)
            reliability, summary = positive_pair_reliability(
                student_h1=h1,
                student_h2=h2,
                teacher_h1=teacher_h1,
                teacher_h2=teacher_h2,
                z1=z1,
                z2=z2,
                spec=reliability_spec,
                shuffled=shuffled_control,
            )
            loss = weighted_info_nce_loss(
                z1,
                z2,
                temperature=temperature,
                positive_weights=reliability,
            )
            loss.backward()
            optimizer.step()
            self._update_ema(student, teacher, momentum=ema_momentum)
            last_reliability = reliability.detach().cpu()
            last_summary = summary
            train_rows.append(
                {
                    "epoch": epoch,
                    "split": "stage2",
                    "loss": f"{float(loss.item()):.6f}",
                    "metric": f"{summary['reliability_mean']:.6f}",
                    "status": "ok",
                    "notes": "weighted_infonce_positive_only",
                }
            )

        student.eval()
        with torch.no_grad():
            embeddings = student.encode(data.x, data.edge_index).detach()
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
            "trainer": self.trainer_name,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "shuffled_reliability": shuffled_control,
            "last_reliability_summary": last_summary,
        }
        write_yaml(self.run_dir / "config.yaml", config_snapshot)
        write_json(self.run_dir / "run_metadata.json", config_snapshot)
        write_json(self.run_dir / "dataset_stats.json", stats)
        write_json(
            self.run_dir / "reliability_plan.json",
            {
                "run_id": self.run_id,
                "status": "completed",
                "mode": "positive_weighting_only",
                "shuffled_reliability": shuffled_control,
                "reliability": scaffold_reliability_summary(self.method_config),
                "last_reliability_summary": last_summary,
            },
        )
        write_csv(
            self.run_dir / "train_log.csv",
            train_rows,
            ["epoch", "split", "loss", "metric", "status", "notes"],
        )
        save_payload = {
            "embeddings": embeddings.cpu(),
            "labels": data.y.detach().cpu(),
            "train_mask": train_mask.detach().cpu(),
            "val_mask": val_mask.detach().cpu(),
            "test_mask": test_mask.detach().cpu(),
        }
        if last_reliability is not None:
            save_payload["positive_reliability"] = last_reliability
        torch.save(save_payload, self.run_dir / "embeddings.pt")

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
                "metric": self.dataset_spec.metric,
                "value": f"{probe['test_accuracy']:.6f}",
                "status": "completed",
                "notes": (
                    f"val_accuracy={probe['val_accuracy']:.6f}; "
                    f"elapsed_seconds={elapsed:.2f}; "
                    f"reliability_mean={last_summary.get('reliability_mean', '')}; "
                    f"shuffled_reliability={shuffled_control}"
                ),
            },
            [
                "run_id",
                "dataset",
                "dataset_group",
                "method",
                "trainer",
                "seed",
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
