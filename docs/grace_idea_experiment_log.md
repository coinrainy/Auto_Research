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

## 2026-06-27 Heterophily split 0 sanity

本轮代码修正：

- `train.py` 支持 `Texas/Cornell/Wisconsin/Actor`。
- 新增 `--split-index` 与 `--eval-mode auto|random|mask`。
- 对 `Texas/Cornell/Wisconsin/Actor`，`auto` 默认使用 PyG 提供的固定 `train/val/test` mask。
- `eval.py` 新增 `label_classification_with_masks`，使用固定 split 的 train mask 训练 logistic regression，用 val mask 选 `C`，在 test mask 上报告 F1。
- Cora/CiteSeer/PubMed/DBLP 默认仍保持原 GRACE 风格随机 linear probe；如需固定 mask，可显式传 `--eval-mode mask`。

数据集检查：

| Dataset | Nodes | Edges | Features | Classes | Split 0 train/val/test |
|---|---:|---:|---:|---:|---|
| Texas | 183 | 325 | 1703 | 5 | 87 / 59 / 37 |
| Cornell | 183 | 298 | 1703 | 5 | 87 / 59 / 37 |
| Wisconsin | 251 | 515 | 1703 | 5 | 120 / 80 / 51 |
| Actor | 7600 | 30019 | 932 | 5 | 3648 / 2432 / 1520 |

命令模式：

```bash
python train.py --dataset <dataset> --method grace --seed 0 --split-index 0 --epochs 100 --save-dir runs/hetero_split0_seed0_e100 --log-every 50
python train.py --dataset <dataset> --method es_weighted --seed 0 --split-index 0 --epochs 100 --warmup-epochs 20 --negative-weighting --save-dir runs/hetero_split0_seed0_e100 --log-every 50
```

Actor 额外使用 `--batch-size 1024` 控制 InfoNCE 显存。

结果：

| Dataset | GRACE F1Mi | ES-weighted F1Mi | Delta F1Mi | GRACE F1Ma | ES-weighted F1Ma | Delta F1Ma | Final weight mean | Final weight std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Texas | 0.648649 | 0.648649 | +0.000000 | 0.321429 | 0.426667 | +0.105238 | 0.967653 | 0.014592 |
| Cornell | 0.486486 | 0.486486 | +0.000000 | 0.321053 | 0.328571 | +0.007519 | 0.968657 | 0.013489 |
| Wisconsin | 0.509804 | 0.529412 | +0.019608 | 0.200909 | 0.208021 | +0.007113 | 0.977015 | 0.017866 |
| Actor | 0.295395 | 0.299342 | +0.003947 | 0.243058 | 0.248689 | +0.005631 | 0.946209 | 0.020900 |

Mean paired delta over the four datasets:

- F1Mi：+0.005889
- F1Ma：+0.031375

当前解释：

- 这是单 split、单 seed、100 epoch 的 sanity，不能作为稳定结论。
- 四个数据集没有出现负向，说明当前 `es_weighted` 至少值得扩展到更多 splits。
- Texas 的 macro-F1 提升较大，但 micro-F1 持平，可能与类别分布或少数类预测变化有关，需要后续 confusion matrix / per-class F1 诊断。
- Final weight mean 仍偏高，说明 embedding stability 权重没有形成强过滤，但 Actor 的 std 相对更高，可能更适合观察 reliability 分层。

下一步建议：

- 先扩展到 `split_index=0,1,2`、`seed=0`，仍保持 100 epoch，验证 split 稳定性。
- 同时补 per-class F1 / confusion matrix 诊断，尤其检查 Texas macro-F1 提升来自哪些类别。
- 若 split 0-2 仍不负，再考虑 seeds 0-2 或 200 epoch。
