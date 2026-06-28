# Filter-Decoupled Natural-View Candidate

日期：2026-06-28

## 动机

DANV 家族已经放弃：固定全局 disagreement penalty 和简单 degree gate 都不够稳定。

但 `gcn_mlp_gcl` 的 Natural-View foundation 仍然是当前最稳工程信号。参考 S3GCL / PolyGCL 后，下一代候选不再做“对齐 vs 去相关”的全局权重，而改为显式拆分 graph view 中的 low-pass / high-pass target。

## 候选：FDNV-GCL

名称：Filter-Decoupled Natural-View GCL  
入口：`--method fdnv_gcl`

核心想法：

- ego/raw-feature view：MLP encoder；
- graph/message-passing view：GCN encoder；
- low-pass target：`P graph`；
- high-pass target：`graph - P graph`；
- label-free filter gate：由 raw feature propagation residual、raw-neighbor agreement 与 log-degree 估计节点更应对齐 high-pass 还是 low-pass；
- loss：MLP ego view 同时学习 routed high/low filtered targets，并保留 GCN-MLP natural-view bootstrap。

默认实现：

- `fdnv_route_weight=0.5`
- `fdnv_bootstrap_weight=1.0`
- `fdnv_filter_temperature=1.0`
- `fdnv_min_filter_weight=0.05`
- 默认 final representation：`ego_graph`

## Early Gate

必须先过：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl fdnv_gcl" \
SPLITS="0" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_fdnv_s0_split0_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

若 split0 不能至少 3/4 数据集 micro 不低于 GCN-MLP，立即放弃或大改，不进入 splits 0/1/2。

## Early Gate 结果

### 默认 `fdnv_route_weight=0.5`

命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl fdnv_gcl" \
SPLITS="0" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_fdnv_s0_split0_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | ---: | ---: | --- |
| Texas | 0.000000 | 0.000000 | 持平 |
| Actor | +0.008553 | +0.010262 | 正向 |
| Chameleon | -0.004386 | -0.012705 | 失败 |
| Squirrel | 0.000000 | -0.000494 | 基本持平 |

裁决：默认 route 太强，Chameleon 明确受伤，不扩大。

### 保守 `fdnv_route_weight=0.1`

命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl fdnv_gcl" \
SPLITS="0" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_fdnv_r01_s0_split0_e50" \
EXTRA_ARGS="--fdnv-route-weight 0.1" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | ---: | ---: | --- |
| Texas | 0.000000 | +0.045083 | macro 正向 |
| Actor | +0.001316 | +0.010971 | 小幅正向 |
| Chameleon | -0.008772 | -0.011839 | 失败 |
| Squirrel | +0.003842 | +0.006560 | 小幅正向 |

裁决：保守 route 有局部信号，但仍被 Chameleon 卡住，不进入 splits 0/1/2。

## 当前裁决

FDNV-GCL 第一版不作为 active main idea。

保留价值：

- filter-decoupled route 比 DANV penalty 更安全；
- Texas macro、Actor、Squirrel 有局部正向；
- 说明 low/high target routing 可能有价值。

失败点：

- Chameleon 对 routed filter target 敏感，两个 route weight 都负向；
- 当前 high/low gate 的均值没有清楚区分 WikipediaNetwork 内部差异；
- 只用 raw residual/agreement/degree 估计 filter routing 不够。

下一步不再调 `fdnv_route_weight`，应换成更清楚的 filter objective，例如：

- 对 high/low branch 做互补性约束，而非让 ego 同时追两个 target；
- 或引入 semantic positive / structural positive 的双 mask，参考 S3GCL 的 semantic/spatial split。
