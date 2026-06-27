# CBR-GCL 候选研究日志

日期：2026-06-28

## 候选问题

Hybrid RR-GCL 说明全局固定 RR 正则会带来 macro/少数类覆盖线索，但 Texas 与 normal-vs-shuffled control 不稳。CBR-GCL 将 RR 从全图均匀统计改为 cluster-balanced 统计：

> 在 GRACE InfoNCE 主损失之外，warm-up 后用两个 view 的 consensus embedding 做无监督聚类；按 cluster inverse-frequency 对 RR cross-correlation 加权，使小 cluster 不被大 cluster 淹没；再用 shuffled positive correspondence 作为机制 control。

该方向不是“首次使用 redundancy reduction”或“首次使用 clustering”。Barlow Twins、Graph Barlow Twins、CCA-SSG、prototype/cluster GCL 都已覆盖相关思想。当前 CBR-GCL 的研究价值在于：验证 cluster-balanced RR 是否能把此前 RR 线索中的 macro/少数类收益变得更稳定，并为后续 anti-degradation gate 提供证据。

参考边界：

- Barlow Twins: https://arxiv.org/abs/2103.03230
- Graph Barlow Twins: https://openreview.net/forum?id=MRGFutr0p5e
- CCA-SSG: https://proceedings.neurips.cc/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf
- BalanceGCL / hard-negative balancing 相关新近方向：AAAI 2026 paper snippet from search result, `Graph Contrastive Learning with Balanced Hard Negatives and Fine-Grained Semantic-Aware Positive Graphs`

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Texas --method cbr_gcl \
  --warmup-epochs 20 \
  --cbr-rr-weight 0.001 \
  --cbr-kmeans-iters 10
```

核心机制：

- warm-up 前完全等同 GRACE；
- warm-up 后计算 GRACE InfoNCE 主损失；
- 用 `(z1 + z2) / 2` 的 stop-gradient consensus embedding 做 KMeans；
- cluster 数默认等于数据集类别数，也可用 `--cbr-num-clusters` 指定；
- 对每个节点赋予 inverse cluster-size 权重，并裁剪到 `[cbr_min_weight, cbr_max_weight]`；
- 用加权均值/方差标准化 projection features；
- 用 cluster-balanced weighted cross-correlation 计算 RR loss；
- 总损失为 `InfoNCE + cbr_rr_weight * CBR_RR`；
- `--shuffle-weights` 只打乱 RR 的 positive node correspondence，InfoNCE 与 cluster weights 不变。

新增参数：

- `--cbr-rr-weight`，默认 `0.001`；
- `--cbr-num-clusters`，默认 0，表示使用 dataset.num_classes；
- `--cbr-kmeans-iters`，默认 10；
- `--cbr-min-weight` / `--cbr-max-weight`。

新增日志：

- `cbr_rr_loss`；
- `cbr_num_active_clusters`、`cbr_cluster_entropy`；
- `cbr_weight_mean/std/min/max/ess/ess_ratio`；
- 继续记录 RR cross-correlation diagnostics。

## 初筛结果

### split0-2 sanity

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs，比较 GRACE、CBR-GCL normal、CBR-GCL shuffled。

| Dataset | F1Mi delta vs GRACE | F1Ma delta vs GRACE | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: |
| Actor | -0.002851 | -0.006628 | -0.001535 | -0.000296 |
| Cornell | +0.036036 | +0.047113 | ~0.000000 | +0.011886 |
| Texas | +0.036036 | +0.016573 | +0.027027 | +0.030177 |
| Wisconsin | +0.013072 | +0.004736 | +0.039216 | -0.011451 |

判断：比固定 Hybrid RR 更好，尤其 Texas 的 normal-vs-shuffled 由负转正；Actor 小幅负向。

### split0-9 复核

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-9 × seed0 × 100 epochs，共 120 个 run，全部 completed。

输出：

- `experiments/grace_idea/runs/cbr_gcl_splits0-9_seed0_e100`
- `experiments/grace_idea/runs/summaries/cbr_gcl_splits0-9_seed0_e100_paired.csv`
- `experiments/grace_idea/runs/summaries/cbr_gcl_splits0-9_seed0_e100_aggregate.csv`

| Dataset | F1Mi delta vs GRACE | pos/zero/neg | F1Ma delta vs GRACE | pos/zero/neg | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | -0.000789 | 6/1/3 | -0.002619 | 5/0/5 | -0.000329 | -0.001300 |
| Cornell | +0.000000 | 4/2/4 | +0.008028 | 5/0/5 | -0.002703 | +0.013165 |
| Texas | +0.005405 | 2/6/2 | +0.010329 | 2/6/2 | +0.005405 | +0.008799 |
| Wisconsin | +0.003922 | 4/3/3 | +0.029514 | 6/3/1 | +0.009804 | +0.032024 |

## 当前判断

CBR-GCL 是目前 RR 方向中最值得保留的条件性候选，但仍不是 SOTA-ready 主方法：

- 相比固定 `hybrid_rr_gcl`，CBR 修复了 Texas 上 normal 被 shuffled 系统性压过的问题；
- Wisconsin macro 与 normal-vs-shuffled macro 有较稳定正向，支持“cluster-balanced RR 可改善少数/弱类覆盖”的机制线索；
- Texas 有轻微正向，但多数 split 持平，效果不够强；
- Cornell 相对 GRACE 的 macro 小正，但 normal-vs-shuffled micro 为负，说明 cluster balancing 本身与 positive correspondence 的贡献还没有完全分离；
- Actor 基本近零略负，提示该正则不应无条件启用。

## 决策

保留 CBR-GCL 作为 active conditional candidate，不把它包装成最终方法。下一步不是继续扩更多 seed 或堆更大 baseline，而是实现 anti-degradation gate：

1. 先诊断哪些 split/数据集上 CBR 带来收益：看 cluster weight ESS、cluster entropy、RR diag/offdiag、class-level delta、local homophily/degree；
2. 设计 gate 只在 CBR 预期有益的区域或训练阶段启用；
3. 新 gate 必须通过 `normal > shuffled` 和 `normal >= GRACE` 的双重约束，尤其不能继续伤 Actor。

## 下一步建议

下一轮优先实现 `gated_cbr_gcl`，而不是继续调 `cbr_rr_weight`：

- 候选 gate 1：若 cluster-balanced weights 过于接近均匀或 RR 对角相关提升不足，则跳过 CBR；
- 候选 gate 2：按 cluster confidence / cluster compactness 调节 RR 权重；
- 候选 gate 3：使用 EMA cluster assignments，降低 KMeans 抖动；
- 必须保留 shuffled positive correspondence control。

建议下一轮命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2" SEEDS="0" \
METHODS="grace gated_cbr_gcl" ES_CONTROLS="normal shuffled" \
EPOCHS=100 WARMUP_EPOCHS=20 SAVE_DIR="runs/gated_cbr_gcl_splits0-2_seed0_e100" \
MANIFEST_PATH="runs/gated_cbr_gcl_splits0-2_seed0_e100/run_manifest.csv" \
OVERWRITE=1 LOG_EVERY=100 scripts/run_split_study.sh
```
