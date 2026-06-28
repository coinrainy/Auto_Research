# Redundancy-Reduced Natural-View GCL 候选记录

日期：2026-06-29

## 背景

在 PCNV、LCOS、LCM 和 DSP-GCL 均未通过 early gate 后，本轮将搜索方向从“节点/样本权重”和“graph/high-pass mix”转向训练目标本身：保留 `gcn_mlp_gcl` 的 MLP ego view 与 GCN graph view 双分支，但用 redundancy reduction 目标替代 BYOL 式 negative-cosine bootstrap。

## DSP-GCL 裁决

`dsp_gcl` 尝试用 label-free downstream separability proxy 给 per-node bootstrap loss 加权。proxy 由 ego+graph representation 的 kNN density margin 与 ego/graph view consistency 组成，并提供 `--dsp-shuffle-weight` 控制。

split0 seed0 50 epoch 结果：

| Dataset | ΔF1Mi vs GCN-MLP | normal - shuffled | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.000000 | -0.027027 | shuffled 更强 |
| Actor | +0.005921 | +0.014474 | 小正 |
| Chameleon | -0.010965 | +0.002193 | baseline 失败 |
| Squirrel | -0.013449 | +0.011527 | baseline 失败 |

裁决：DSP-GCL 降级为失败/诊断资产。separability proxy 有权重方差，但没有稳定转化为分类收益，且 Texas 上 shuffled control 反超。

## RRNV-GCL 方法

`rrnv_gcl` 暂名 Redundancy-Reduced Natural-View GCL。

核心目标：

- 使用 `gcn_mlp_gcl` 的 MLP ego view 与 GCN graph view；
- 用 predictor 后的 ego/graph 表示执行 VICReg/CCA 风格 redundancy reduction；
- final representation 仍为 `[ego, graph]`；
- `--rrnv-shuffle-pairs` 打乱 ego/graph 节点配对，用作机制 control。

默认损失：

```text
L = 25 * invariance(z_ego, z_graph)
  + 25 * variance(z_ego, z_graph)
  + 1  * covariance(z_ego, z_graph)
```

这一路线的主要假设是：对异配图而言，MLP ego view 与 GCN graph view 不是普通增强视图，负样本/positive mining 的小改动不稳定；更强的 redundancy reduction 可以改善双视图表示的下游线性可分性。

## split0 gate

输出目录：`runs/rrnv_split0_s0_e50/`

| Dataset | ΔF1Mi vs GCN-MLP | normal - shuffled | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.081081 | +0.081081 | 强正 |
| Actor | +0.019079 | +0.003289 | 小正，control 弱 |
| Chameleon | +0.026316 | +0.024123 | 正向且 control 较干净 |
| Squirrel | -0.021134 | -0.003842 | 失败 |

split0 裁决：RRNV 比 DSP/LCM/PCNV 更健康，允许进入 splits 0-2 复核；但 Squirrel 退化是 major risk。

## splits 0-2 复核

输出目录：`runs/rrnv_s0_splits0-2_e50/`

| Dataset | mean ΔF1Mi vs GCN-MLP | ΔF1Mi by split | mean normal - shuffled | 裁决 |
| --- | ---: | --- | ---: | --- |
| Texas | +0.099099 | +0.081081,+0.135135,+0.081081 | +0.054054 | 当前最强证据 |
| Actor | +0.002412 | +0.006579,+0.001316,-0.000658 | +0.006360 | 仅弱正 |
| Chameleon | +0.008041 | +0.017544,+0.010965,-0.004386 | +0.007310 | 小正，split2 风险 |
| Squirrel | -0.008325 | -0.025937,+0.006724,-0.005764 | +0.002241 | 均值失败 |

当前裁决：RRNV 升级为 active-but-risky candidate。它不是最终成功方法，也不能声称通用 heterophily SOTA；但 Texas 强正、Chameleon 小正且 normal-vs-shuffled 基本支持真实视图配对机制，值得继续做 safety / graph-type adaptation。

## DARRNV safety 变体裁决

`darrnv_gcl` 尝试用图级平均度 gate 调节 RRNV 目标，在高密度图上退回 Natural-View BYOL 底座：

```text
L = L_BYOL + density_gate * 0.05 * L_RRNV
```

split0 运行到 Texas/Actor 后触发停止条件并中止：

| Dataset | ΔF1Mi vs GCN-MLP | 裁决 |
| --- | ---: | --- |
| Texas | -0.054054 | 丢失 RRNV 主信号 |
| Actor | -0.005921 | 低于 baseline |

裁决：DARRNV 当前实现失败，不继续跑 Chameleon/Squirrel。简单把 RRNV 作为小辅助项太保守，无法保留 Texas/Actor 上的主信号。

## DS-RRNV safety 变体

`dsrrnv_gcl` 暂名 Density-Safe RRNV。它保留纯 RRNV 训练目标，但在 final representation 中根据图平均度使用 graph/high residual mix：

```text
final = [ego, (1 - gate) graph + gate high]
gate = sigmoid((log(1 + avg_degree) - log(1 + threshold)) / temperature)
```

默认 `threshold=30`、`temperature=0.25`，使 Texas/Actor gate 近似 0，Chameleon gate 约 0.076，Squirrel gate 约 0.744。

split0 seed0 结果：

| Dataset | ΔF1Mi vs GCN-MLP | normal - shuffled | high gate | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | +0.054054 | +0.081081 | 0.000061 | 保住 RRNV 信号 |
| Actor | +0.006579 | +0.003947 | 0.000617 | 小正 |
| Chameleon | +0.043860 | +0.028509 | 0.076343 | 强正 |
| Squirrel | -0.011527 | -0.013449 | 0.744020 | 仍失败，shuffled 更强 |

splits 0-2 seed0 复核：

| Dataset | mean ΔF1Mi vs GCN-MLP | mean normal - shuffled | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.090090 | +0.072072 | 强正且 control 干净 |
| Actor | +0.001974 | +0.005482 | 弱正 |
| Chameleon | +0.013158 | +0.014620 | 小正且 control 较干净 |
| Squirrel | +0.006724 | -0.012168 | 均值转正但 shuffled 更强 |

裁决：DS-RRNV 是当前比 RRNV 更好的 active-but-risky candidate。它在 Texas/Chameleon 上保留或增强信号，并把 Squirrel 从 RRNV 的均值负向拉到均值小正；但 Squirrel normal-vs-shuffled 为负，说明高密度图上的机制仍未被证明。

## DIRRNV 尝试与放弃

`dirrnv_gcl` 暂名 Density-adaptive Invariance RRNV。它在 DS-RRNV 基础上对高密度图衰减 true-pair invariance：

```text
invariance_scale = (1 - high_gate)^2
```

split0 seed0 初筛结果：

| Dataset | ΔF1Mi vs GCN-MLP | normal - shuffled | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.000000 | +0.027027 | 弱于 DS-RRNV |
| Actor | +0.007237 | +0.003947 | 小正 |
| Chameleon | +0.039474 | +0.030702 | 正向 |
| Squirrel | -0.000961 | +0.000000 | 未救回 Squirrel |

裁决：DIRRNV 不扩大到 splits 0-2。降低 invariance 没有解决 Squirrel，也削弱了 Texas 主信号。

## DPRRNV 高密度扰动配对尝试

`dprrnv_gcl` 暂名 Density-Perturbed RRNV。它不是替代 DS-RRNV 的主线，而是针对 DS-RRNV 的 Squirrel 反证做一次机制验证：如果高密度图上的真实 ego/graph 节点配对不可靠，则在 RRNV invariance target 中按密度混入随机配对目标。

```text
shuffle_prob = high_gate
target = (1 - shuffle_prob) graph + shuffle_prob shuffled(graph)
final = DS-RRNV 的 density-mixed final
```

默认下 Texas/Actor 的 `shuffle_prob` 近似 0，Chameleon 约 0.073，Squirrel 约 0.707。`--rrnv-shuffle-pairs` 仍作为 full-shuffled control，此时 `shuffle_prob=1.0`。

split0 seed0 结果：

| Dataset | ΔF1Mi vs GCN-MLP | normal - full-shuffled | shuffle prob | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | +0.027027 | +0.027027 | 0.000058 | 正向但弱于 DS-RRNV |
| Actor | +0.003289 | -0.005263 | 0.000586 | full-shuffled 更强 |
| Chameleon | +0.002193 | -0.015351 | 0.072526 | full-shuffled 更强 |
| Squirrel | +0.026897 | +0.019212 | 0.706819 | 修复 Squirrel |

裁决：DPRRNV 不升级为主方法。它证明“高密度图上弱化真实配对、引入扰动配对”能修复 Squirrel split0，但同时显著削弱 Texas/Chameleon，且 Actor/Chameleon 的 full-shuffled control 更强。当前只保留为高密度图机制线索或 DS-RRNV 的可选诊断模块，不进入 splits 0-2 扩展。

## NPRRNV 节点级配对可靠性尝试与放弃

`nprrnv_gcl` 暂名 Node-level Pair-Reliable RRNV。它沿用 DS-RRNV final representation，但把 DPRRNV 的图级扰动概率改成节点级：

```text
node_gate = sigmoid(degree + raw_residual - raw_agreement - ego_graph_view_cosine)
shuffle_prob_i = graph_high_gate * (min_local_scale + (1 - min_local_scale) node_gate_i)
target_i = (1 - shuffle_prob_i) graph_i + shuffle_prob_i shuffled(graph)_i
```

默认 `min_local_scale=0.5`，另测 strict 版本 `min_local_scale=0.0`。

默认 split0 seed0 结果：

| Dataset | ΔF1Mi vs GCN-MLP | shuffle prob mean | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.000000 | 0.000043 | 没有保住 DS-RRNV 主信号 |
| Actor | -0.011184 | 0.000438 | 低于 baseline |
| Chameleon | -0.039474 | 0.053132 | 明显失败 |
| Squirrel | +0.014409 | 0.518364 | 正向但弱于 DPRRNV |

strict 版本只跑 Chameleon/Squirrel：

| Dataset | ΔF1Mi vs GCN-MLP | shuffle prob mean | 裁决 |
| --- | ---: | ---: | --- |
| Chameleon | -0.013158 | 0.033761 | 仍低于 baseline |
| Squirrel | +0.028818 | 0.329999 | 正向但不能抵消 Chameleon 失败 |

裁决：NPRRNV 不升级，不进入 splits 0-2，也不继续调 `min_local_scale`。它说明“节点级扰动目标”仍会伤害中密度图 Chameleon；后续若继承该线索，应从扰动 target 改为 reliability-weighted invariance 或 unreliable-pair filtering，而不是继续混入 shuffled graph target。

## RWIRRNV reliability-weighted invariance 候选

`rwirrnv_gcl` 暂名 Reliability-Weighted Invariance RRNV。它继承 NPRRNV 的节点级不可靠性估计，但不再扰动 target，而是只降低不可靠节点的 RRNV invariance MSE 权重；variance/covariance 仍在全体节点上计算。

```text
unreliable_i = graph_high_gate * local_unreliable_gate_i
reliability_i = 1 - unreliable_i
L_inv = mean_i reliability_i * ||norm(ego_i) - norm(graph_i)||^2
L = 25 L_inv + 25 L_var + 1 L_cov
```

默认 `rwirrnv_min_reliability=0.1`、`rwirrnv_weight_power=1.0`，并提供 `--rwirrnv-shuffle-weight` 作为 reliability 排序 control，`--rwirrnv-constant-weight` 作为同均值常数权重 control。

split0 seed0 结果：

| Dataset | ΔF1Mi vs GCN-MLP | normal - shuffled-weight | reliability mean | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | +0.081081 | +0.108108 | 0.999961 | 强正且 control 支持 |
| Actor | +0.009211 | -0.003289 | 0.999606 | 小正但 control 不干净 |
| Chameleon | +0.043860 | +0.028509 | 0.952180 | 强正且 control 支持 |
| Squirrel | -0.008646 | 未跑 | 0.533684 | 失败 |

splits 0-2、seed0、50 epoch 复核结果：

| Dataset | normal ΔF1Mi vs GCN-MLP | shuffled-weight ΔF1Mi vs GCN-MLP | normal - shuffled-weight | normal 正/负 split | 裁决 |
| --- | ---: | ---: | ---: | --- | --- |
| Texas | +0.117117 | +0.054054 | +0.063063 | 3/0 | 强正，control 清楚 |
| Chameleon | +0.013889 | +0.005117 | +0.008772 | 2/0 | 小正，control 基本支持 |
| Squirrel | +0.014089 | +0.012488 | +0.001601 | 2/1 | 性能小正，但 control 弱 |
| Actor | +0.002851 | +0.000658 | +0.002193 | 1/1 | 边缘正，证据很弱 |

splits 0-9、seed0、50 epoch 硬门控结果：

| Dataset | normal ΔF1Mi | shuffled-weight ΔF1Mi | constant-weight ΔF1Mi | normal - shuffled | normal - constant | 裁决 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Texas | +0.075676 | +0.032432 | +0.072973 | +0.043243 | +0.002703 | 性能强，但权重几乎全 1，排序机制不关键 |
| Chameleon | +0.013158 | +0.014254 | +0.015132 | -0.001096 | -0.001974 | 三种权重均小正，常数最好，排序 claim 失败 |
| Squirrel | +0.022574 | +0.026705 | +0.018636 | -0.004131 | +0.003939 | 降权有用，但 shuffled 最强，排序 claim 失败 |
| Actor | -0.001513 | -0.004934 | -0.001184 | +0.003421 | -0.000329 | 低于 baseline，边界/失败数据集 |

裁决：RWIRRNV 的原始“节点 reliability 排序”主张降级，不再作为可包装的主贡献。10 split 结果说明 invariance attenuation 本身有性能信号：Texas 强正，Chameleon/Squirrel 正向；但 Chameleon 的常数权重最好、Squirrel 的 shuffled 权重最好，证明当前 reliability score 的节点对应关系没有通过机制门控。RWIRRNV 保留为下一代 `invariance attenuation / graph-level reliability calibration` 的实验线索，不应继续以 per-node reliability score 为主线扩展。

## 下一步

保留 `rwirrnv_gcl` 为机制线索，但放弃把当前 per-node reliability 排序写成主贡献。后续必须解决三个问题：

- 排序机制失败：Chameleon 常数权重、Squirrel shuffled 权重均不弱于 normal；后续若继续，应把 hypothesis 改成 graph/dataset-level invariance attenuation，而不是节点 reliability ranking；
- Actor boundary：三种 RWIRRNV 变体在 Actor 上均低于 `gcn_mlp_gcl`，说明该目标不适合所有异配图；
- 高密度扰动配对：DPRRNV 在 Squirrel 有修复信号，但 full-shuffled control 不够干净；NPRRNV 进一步说明节点级 target perturbation 仍会伤害 Chameleon；若后续继承该线索，必须转向 reliability-weighted invariance / filtering；
- 强基线对齐：RRNV 仍只与内部 `gcn_mlp_gcl` 对齐，尚未和 PolyGCL / S3GCL / GraphECL 等强基线同协议比较。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
cat runs/dsrrnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
cat runs/dprrnv_split0_s0_e50/aggregate_vs_gcn_mlp.csv
cat runs/nprrnv_split0_s0_e50/aggregate_vs_gcn_mlp.csv
cat runs/nprrnv_strict_split0_s0_e50/aggregate_vs_gcn_mlp.csv
cat runs/rwirrnv_split0_s0_e50/aggregate_vs_gcn_mlp.csv
cat runs/rwirrnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

若继续方法实验，不应继续调当前 reliability score；优先设计下一代 graph-level / schedule-level invariance attenuation，并使用 normal、shuffled、constant 三重 control 作为硬门槛。停止 `darrnv_gcl`、`dirrnv_gcl` 和 `nprrnv_gcl` 主线，DPRRNV/NPRRNV 仅作为 Squirrel 机制线索。
