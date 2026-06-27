# GRACE Idea 实验日志

## 2026-06-27 Cora 单数据集 sanity

当前代码位置：`experiments/grace_idea/`

本轮目的：

- 验证从 GRACE 副本改出的 `es_weighted` 路径能完整训练和评估。
- 检查 embedding-stability reliability 在 homophily Cora 上是否至少不明显退化。
- 确认结果可以落盘到每个 run 的 `eval_summary.csv`、`metadata.json`、`train_log.csv`。

### 单 seed 对照

命令：

```bash
python train.py --dataset Cora --method grace --epochs 200 --save-dir runs/cora_compare_saved
python train.py --dataset Cora --method es_weighted --epochs 200 --warmup-epochs 20 --negative-weighting --save-dir runs/cora_compare_saved
```

结果：

| Method | Seed | F1Mi mean | F1Mi std | F1Ma mean | F1Ma std | Note |
|---|---:|---:|---:|---:|---:|---|
| grace | 39788 | 0.833060 | 0.004115 | 0.822161 | 0.004653 | config default seed |
| es_weighted | 39788 | 0.832786 | 0.003882 | 0.821930 | 0.004539 | final weight mean 0.985422 |

paired delta (`es_weighted - grace`):

- F1Mi：-0.000273
- F1Ma：-0.000231

### Seeds 0-2 mini sweep

命令模式：

```bash
python train.py --dataset Cora --method grace --seed <seed> --epochs 200 --save-dir runs/cora_seed_sweep_s0-2 --log-every 50
python train.py --dataset Cora --method es_weighted --seed <seed> --epochs 200 --warmup-epochs 20 --negative-weighting --save-dir runs/cora_seed_sweep_s0-2 --log-every 50
```

结果：

| Seed | GRACE F1Mi | ES-weighted F1Mi | Delta F1Mi | GRACE F1Ma | ES-weighted F1Ma | Delta F1Ma |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.828275 | 0.829232 | +0.000957 | 0.808149 | 0.808464 | +0.000315 |
| 1 | 0.814055 | 0.814055 | +0.000000 | 0.802638 | 0.802798 | +0.000160 |
| 2 | 0.832513 | 0.831829 | -0.000684 | 0.819223 | 0.818997 | -0.000226 |

Aggregate over seeds 0-2:

| Method | F1Mi mean | F1Mi pop std | F1Ma mean | F1Ma pop std |
|---|---:|---:|---:|---:|
| grace | 0.824948 | 0.007894 | 0.810003 | 0.006897 |
| es_weighted | 0.825039 | 0.007839 | 0.810086 | 0.006712 |

Mean paired delta (`es_weighted - grace`):

- F1Mi：+0.000091
- F1Ma：+0.000083

### 当前解释

- Cora 上 `es_weighted` 基本与 GRACE 持平，暂时支持 homophily non-degradation，但没有性能收益证据。
- `es_weighted` 的最后一轮 reliability 权重很饱和，seeds 0-2 的 final weight mean 约为 0.9837-0.9853，说明 Cora 这类 homophily 图上 embedding stability 信号区分度较弱。
- 当前结果不应作为论文级结论；它只是代码路径和 homophily sanity check。

### 下一步建议

- 优先补 `experiments/grace_idea/train.py` 的 heterophily dataset 与 split 支持。
- 再跑 Texas/Cornell/Wisconsin/Actor 的 matched GRACE vs `es_weighted` 对照。
- 不建议此时继续堆新 reliability 信号；先看 embedding-stability-only 在 heterophily split 上是否有可重复信号。

