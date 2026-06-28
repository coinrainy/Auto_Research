# Natural-View GCL Foundation Memo

日期：2026-06-28

## 当前裁决

`gcn_mlp_gcl` 从普通对照升级为当前最有价值的 architecture foundation。

它不是最终论文方法，因为“GCN branch + MLP branch bootstrap”本身创新不足；但它在四个 heterophily 数据集的轻量 split sanity 中全部超过当前 GRACE scaffold，说明后续新 idea 不应继续围绕 GRACE 随机双增强小修小补，而应以 **natural views** 为底座：

- ego/raw-feature view：MLP encoder；
- graph/message-passing view：GCN encoder；
- loss：same-node cross-view bootstrap；
- representation：`[ego, graph]`。

## 证据

实验设置：

- split: 0/1/2
- seed: 0
- epochs: 50
- baseline: same scaffold GRACE
- evaluator: PyG fixed masks + validation-selected logistic C

Aggregate:

| Dataset | F1Mi mean | F1Ma mean | ΔF1Mi vs GRACE | ΔF1Ma vs GRACE | Positive/Negative F1Mi |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.639640 | 0.322785 | +0.036036 | +0.048717 | 3/0 |
| Actor | 0.353728 | 0.311257 | +0.076535 | +0.076413 | 3/0 |
| Chameleon | 0.416667 | 0.403677 | +0.021930 | +0.018146 | 3/0 |
| Squirrel | 0.309638 | 0.300817 | +0.025296 | +0.022114 | 3/0 |

## 解释

这说明在当前 scaffold 下，异配图上的关键增益可能不是：

- 更复杂的 random augmentation；
- low-pass positive cache；
- high-energy residual bootstrap；
- projection stability/reliability weighting。

更合理的工作假设是：

> 异配图 GCL 中，原始特征视图和图传播视图本身就是天然互补 views；随机双增强 GRACE 反而可能在局部异配区域制造语义破坏。一个强方法应学习何时对齐、何时保留分歧，而不是无条件压平 GCN 与 MLP 的差异。

## 下一代 idea

暂命名：**Disagreement-Aware Natural-View GCL (DANV-GCL)**。

核心不是重新发明 positive cache，而是在 GCN-MLP foundation 上引入：

1. **agreement-preserving alignment**：对 raw-feature 与 graph-view 一致的节点增强对齐；
2. **disagreement-preserving decorrelation**：对两 view 明显冲突的节点，不强行对齐，而保留互补维度；
3. **label-free gate**：使用 raw-neighbor agreement、view cosine、feature propagation residual energy 构造 stop-gradient gate；
4. **hard control**：必须超过 `gcn_mlp_gcl`，否则视为无效模块。

## 当前实现

已新增 `--method danv_gcl`：

- 默认输出 `ego_graph`；
- 使用 raw-neighbor agreement、view cosine、raw propagation residual energy 计算 stop-gradient alignment gate；
- 高 gate 节点执行加权 GCN-MLP alignment；
- 低 gate 节点增加 disagreement-preserving cosine penalty，降低强制相似；
- 记录 `diag_danv_gate_mean/std`。

## DANV split0 early gate

命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="grace gcn_mlp_gcl danv_gcl" \
SPLITS="0" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_danv_s0_split0_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh

python summarize_split_study.py \
  --runs-dir runs/split_study_danv_s0_split0_e50 \
  --baseline-method gcn_mlp_gcl \
  --out runs/split_study_danv_s0_split0_e50/split_study_runs_vs_gcn_mlp.csv \
  --aggregate-out runs/split_study_danv_s0_split0_e50/split_study_aggregate_vs_gcn_mlp.csv
```

结果：

| Dataset | DANV F1Mi | DANV F1Ma | ΔF1Mi vs GCN-MLP | ΔF1Ma vs GCN-MLP | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.648649 | 0.276786 | -0.054054 | -0.123677 | 失败 |
| Actor | 0.360526 | 0.327103 | +0.005263 | +0.014987 | 小幅正向 |
| Chameleon | 0.451754 | 0.447842 | +0.006579 | +0.009536 | 小幅正向 |
| Squirrel | 0.325648 | 0.314833 | +0.003842 | +0.002403 | 小幅正向 |

裁决：

- DANV 不是成功方法，只是 active-but-risky candidate；
- 3/4 数据集在 split0 上超过 GCN-MLP，说明 disagreement gate 可能有价值；
- Texas 明显伤害，说明当前 gate 或 disagreement penalty 会破坏 WebKB 小图上的强 alignment；
- 下一步必须跑 splits 0/1/2，并做 gate ablation：`danv_disagreement_weight=0`、更小 penalty、或 Texas-safe fallback。

## 保留标准

下一轮实现 DANV-GCL 时，最低 early gate：

- Texas/Actor/Chameleon/Squirrel × splits 0/1/2 × seed0 × 50 epoch；
- 对 GRACE 和 GCN-MLP 同时为正；
- 至少 3/4 数据集 micro-F1 mean 超过 GCN-MLP；
- 不能明显牺牲 macro-F1；
- 必须补 homophily safety：Cora/CiteSeer/PubMed。
