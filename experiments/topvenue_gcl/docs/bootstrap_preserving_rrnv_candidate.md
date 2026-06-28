# Bootstrap-Preserving Selective RRNV 候选记录

## 方法动机

此前 RRNV 系列的失败边界很清楚：

- `rrnv_gcl` 和 `dsrrnv_gcl` 在 Texas/Chameleon 上有强信号，但 Squirrel safety 与 shuffled control 不干净；
- `rwirrnv_gcl` 证明 invariance attenuation 有价值，但 per-node reliability 排序没有通过 shuffled/constant control；
- `eairrnv_gcl` 的 graph-level energy attenuation 能放大 Texas/Chameleon，但 Squirrel 仍负；
- `darrnv_gcl` 的 density auxiliary gate 能保护 Squirrel，却损失 Texas 主信号。

BPRRNV 的核心判断是：RRNV 不应该替代 `gcn_mlp_gcl` 的 Natural-View bootstrap，而应该作为一个被密度和能量冲突选择性调节的轻量辅助正则。

## 当前实现

入口：

```bash
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0
```

训练目标：

```text
loss = bootstrap_loss + bprrnv_rr_weight * aux_gate * rr_core
```

其中：

- `bootstrap_loss` 沿用 `gcn_mlp_gcl` 的 ego/graph 双向 negative cosine；
- `rr_core = invariance + 0.1 * variance + 0.01 * covariance`；
- `density_factor = sigmoid((log1p(degree_threshold) - log1p(avg_degree)) / degree_temperature)`；
- `energy_factor = 1 - energy_strength * (energy_ratio / (energy_ratio + threshold))^power`；
- `aux_gate = density_factor * energy_factor`。

必要 controls：

```bash
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0 --rrnv-shuffle-pairs
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0 --bprrnv-uniform-gate
```

## 初筛结果

实验设置：Texas/Chameleon/Squirrel/Actor × splits 0/1/2 × model seed 0 × 50 epoch。

`bprrnv_rr_weight=0.25` 失败，normal 相对 `gcn_mlp_gcl` 的 overall mean micro delta 为 -0.005938。

`bprrnv_rr_weight=0.1` 初筛：

| dataset | normal - GCN-MLP | uniform - GCN-MLP | shuffled - GCN-MLP | normal - shuffled | aux gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Texas | +0.027027 | +0.000000 | -0.027027 | +0.054054 | 0.883684 |
| Chameleon | +0.003655 | +0.006579 | -0.010234 | +0.013889 | 0.526757 |
| Squirrel | +0.000640 | -0.005123 | -0.007365 | +0.008005 | 0.194363 |
| Actor | +0.005044 | +0.002632 | -0.001754 | +0.006798 | 0.686431 |
| Overall | +0.009092 | +0.001022 | -0.011595 | +0.020687 | 0.572809 |

输出文件：

- `runs/bprrnv_w01_candidate_comparison_s0_splits0-2_e50.csv`
- `runs/bprrnv_w01_candidate_aggregate_s0_splits0-2_e50.csv`

## 10 split 复核

已执行 Texas/Chameleon/Squirrel/Actor × splits 0-9 × model seed 0 × 50 epoch。

normal vs `gcn_mlp_gcl`：

| dataset | normal - GCN-MLP | macro delta | 正/平/负 split | aux gate |
| --- | ---: | ---: | --- | ---: |
| Texas | +0.000000 | -0.004765 | 2/4/4 | 0.883961 |
| Chameleon | +0.005044 | +0.006057 | 7/1/2 | 0.526719 |
| Squirrel | -0.000768 | -0.003631 | 6/0/4 | 0.194958 |
| Actor | +0.001316 | -0.001021 | 6/0/4 | 0.686506 |
| Overall | +0.001398 | -0.000840 | 21/5/14 | 0.573036 |

输出文件：

- `runs/bprrnv_w01_s0_splits0-9_e50/runs_vs_gcn_mlp.csv`
- `runs/bprrnv_w01_s0_splits0-9_e50/aggregate_vs_gcn_mlp.csv`
- `runs/bprrnv_w01_s0_splits0-9_e50/bprrnv_vs_gcn_mlp_decision_summary.csv`

## Chameleon targeted controls

由于 Chameleon 是唯一有相对清楚小正均值的数据集，补跑 Chameleon splits0-9 的 shuffled pair 与 uniform gate controls。

| variant | mean acc | delta vs GCN-MLP | 正/负 vs GCN-MLP | delta vs normal |
| --- | ---: | ---: | --- | ---: |
| normal | 0.419737 | +0.005044 | 7/2 | 0.000000 |
| shuffled | 0.418421 | +0.003728 | 6/3 | -0.001316 |
| uniform | 0.416667 | +0.001974 | 4/3 | -0.003070 |

normal-vs-shuffled mean 只有 +0.001316，且 normal 只在 3/10 split 高于 shuffled、5/10 低于 shuffled。这个结果不足以支撑 pair correspondence 机制。

输出文件：

- `runs/bprrnv_w01_chameleon_controls_s0_splits0-9_e50/chameleon_control_comparison.csv`
- `runs/bprrnv_w01_chameleon_controls_s0_splits0-9_e50/chameleon_control_aggregate.csv`

## 当前裁决

BPRRNV 降级为失败/弱正则资产，不再作为 active candidate。

原因：

- 10 split overall micro 只有 +0.001398，macro 为负；
- Texas 均值基本 0，Squirrel 为负，Actor 只有噪声级小正；
- Chameleon 虽有 +0.005044，但 shuffled control 过近，normal-vs-shuffled split 级别不干净；
- density/energy selector 没有显示出足够机制优势，uniform gate 与 shuffled pair 都能接近部分收益；
- 继续补多 seed、homophily safety 或 no-density/no-energy controls 的收益不高。

后续不再继续调 `bprrnv_rr_weight`、density threshold、energy strength 或 no-density/no-energy gates。若继续 Natural-View foundation，应转向直接处理 false-negative / negative suppression / downstream separability 的训练目标。
