# Ego-Preserving / Graph-Usage Calibrated GCL 候选研究日志

日期：2026-06-28

## 候选问题

前面的 reliability weighting、false-negative attenuation、spectral mix、prototype objective、RR/CBR 与多种 gate 都只带来小幅或条件性收益。新的证据显示，很多异配图失败并非主要来自 contrastive loss 的样本权重，而是来自 GRACE 默认 GCN encoder 的低通邻域聚合：节点自身特征中的类别语义被异类邻居混合冲淡。

本轮候选假设：

> 在异配图节点分类的 GCL 中，encoder 必须显式保留 ego-feature channel，并校准 graph propagation 的使用强度；否则标准 GCN-based GRACE 会在少数类/弱类上产生系统性类别覆盖失败。

## 当前实现

实现位置：

- `experiments/grace_idea/model.py`
- `experiments/grace_idea/train.py`

新增方法：

- `--method ego_grace`：纯 ego-feature MLP encoder，忽略 edge_index，只保留 GRACE 的 feature-drop 双视图和 InfoNCE。
- `--method residual_grace`：GCN branch + ego MLP branch，使用可学习全局 scalar gate 融合。
- `--method gated_ego_graph_grace`：GCN branch + ego MLP branch，使用节点级 local feature-neighborhood agreement gate 融合。

`gated_ego_graph_grace` 的当前 gate：

- 计算每个节点与邻居均值特征的 cosine agreement；
- 对 agreement 做图内标准化；
- `graph_gate = sigmoid((score - threshold) / temperature)`；
- graph gate 越高，越偏向 GCN branch；越低，越偏向 ego branch；
- 当前默认：`--graph-gate-temperature 0.5 --graph-gate-threshold 0.0`。

## 关键实验结果

### 1. `residual_grace` 异配 10 split 复核

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-9 × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/residual_grace_splits0-9_seed0_e100`
- `experiments/grace_idea/runs/summaries/residual_grace_splits0-9_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta | F1Mi positive/zero/negative | F1Ma positive/zero/negative |
| --- | ---: | ---: | --- | --- |
| Actor | +0.064868 | +0.090895 | 10/0/0 | 10/0/0 |
| Cornell | +0.172973 | +0.183641 | 10/0/0 | 10/0/0 |
| Texas | +0.102703 | +0.188203 | 9/1/0 | 10/0/0 |
| Wisconsin | +0.180392 | +0.199546 | 10/0/0 | 10/0/0 |

诊断：这是目前所有候选中最强、最稳定的异配信号。尤其 Actor 也稳定正向，解决了 CBR/RR 系列一直存在的 Actor 退化问题。

最终 `ego_gate` 基本停在 0.5 附近：

| Dataset | ego_gate mean |
| --- | ---: |
| Texas | 0.498319 |
| Cornell | 0.497999 |
| Wisconsin | 0.498898 |
| Actor | 0.492705 |

说明收益主要来自稳定的 ego + graph 双通道表示容量，而不是 scalar gate 学到了强数据集特化。

### 2. `residual_grace` 同配 quick sanity

实验设置：Cora/CiteSeer/PubMed × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/summaries/residual_grace_homophily_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.008477 | -0.009624 |
| CiteSeer | +0.006678 | +0.012899 |
| PubMed | +0.012003 | +0.014253 |

诊断：`residual_grace` 没有系统性同配退化，但 Cora 有约 0.8-1.0 个点下降，后续需要 10 seeds/splits 复核，并加入 homophily-safe gate 或 residual schedule。

### 3. `ego_grace` MLP-only ablation

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/summaries/ego_grace_splits0-2_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | +0.067763 | +0.084266 |
| Cornell | +0.252252 | +0.261006 |
| Texas | +0.135135 | +0.221388 |
| Wisconsin | +0.228758 | +0.345109 |

与 `residual_grace` 的 split0-2 对比：

| Dataset | ego - residual F1Mi | ego - residual F1Ma |
| --- | ---: | ---: |
| Actor | +0.006140 | +0.007301 |
| Cornell | +0.036036 | +0.000088 |
| Texas | +0.045045 | +0.071625 |
| Wisconsin | +0.039216 | +0.099954 |

诊断：当前异配收益的核心不是“多加一点 RR/weighting”，而是 ego-feature preservation。纯 MLP-only 在小异配图上非常强，这既是机会也是风险：论文不能简单声称 graph branch 本身带来全部收益，必须证明 graph usage 的校准何时有益。

### 4. `gated_ego_graph_grace` 初版

异配 split0-2：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | +0.066009 | +0.084107 |
| Cornell | +0.261261 | +0.280904 |
| Texas | +0.180180 | +0.298539 |
| Wisconsin | +0.209150 | +0.200034 |

和 ego/residual 对比：

| Dataset | Best current note |
| --- | --- |
| Texas | gated 最高 |
| Cornell | gated 略高于 ego/residual |
| Wisconsin | ego macro 明显最高，gated micro 仍强 |
| Actor | gated 与 ego 接近 |

graph gate 诊断：

| Dataset | graph_gate mean | graph_gate std |
| --- | ---: | ---: |
| Texas | 0.501034 | 0.342162 |
| Cornell | 0.503500 | 0.349249 |
| Wisconsin | 0.507854 | 0.342581 |
| Actor | 0.427728 | 0.159707 |

同配 quick sanity：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.201805 | -0.241269 |
| CiteSeer | -0.056984 | -0.053311 |
| PubMed | -0.037173 | -0.031199 |

诊断：当前 `gated_ego_graph_grace` v1 异配很强，但同配安全性失败，不能直接作为最终主方法。它说明 local feature-neighborhood agreement gate 有异配收益，但 homophily-safe graph usage calibration 还没解决。

## 当前判断

当前最有潜力的方向不是继续改 contrastive pair weighting，而是转向：

> Ego-Preserving Graph Contrastive Learning / Graph-Usage Calibrated GCL。

保守主线：

- `residual_grace` 是当前最稳的 active candidate：异配 10 split 全面正向，同配 quick sanity 基本安全。
- `ego_grace` 是必须保留的强 ablation：它证明 ego-feature preservation 是关键机制，但也会被 reviewer 质疑“这还是 graph learning 吗”。
- `gated_ego_graph_grace` 是下一版方法雏形：异配强，但同配失败，必须加入 homophily-safe fallback 或 dataset/region-level graph usage calibration。

不应声称：

- 不能说当前 calibrated gate 已经 SOTA-ready；
- 不能说 graph branch 当前一定优于 ego-only；
- 不能忽略 MLP-only baseline，否则论文很容易被 reviewer 击中。

## 下一步

优先级从高到低：

1. 跑 `ego_grace` 的 10 split 异配复核，与 `residual_grace` 的 10 split 对齐；
2. 为 `gated_ego_graph_grace` 设计 homophily-safe gate，例如 graph gate lower bound / homophily detector / validation-free global graph-use prior；
3. 实现 `mlp_only supervised/linear` 或 feature-only SSL baseline，确认不是简单特征分类器已经足够；
4. 对 `residual_grace` 做 Cora/CiteSeer/PubMed 多 seed 复核；
5. 加入 Chameleon/Squirrel，测试这个 encoder-level idea 是否能扩展到更大异配图；
6. 机制诊断：按 class、degree、local feature agreement 分桶，证明 ego-preserving channel 恢复了低图一致性节点或少数类覆盖。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" METHODS="grace ego_grace" EPOCHS=100 SAVE_DIR="runs/ego_grace_splits0-9_seed0_e100" MANIFEST_PATH="runs/ego_grace_splits0-9_seed0_e100/run_manifest.csv" OVERWRITE=1 LOG_EVERY=100 scripts/run_split_study.sh
python summarize_runs.py --runs-dir runs/ego_grace_splits0-9_seed0_e100 --target-method ego_grace --paired-out runs/summaries/ego_grace_splits0-9_seed0_e100_paired.csv --aggregate-out runs/summaries/ego_grace_splits0-9_seed0_e100_aggregate.csv
```
