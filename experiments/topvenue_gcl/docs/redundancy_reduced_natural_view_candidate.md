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

split0 seed0 结果：

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

## 下一步

保留 `dsrrnv_gcl` 为当前最有价值候选，但后续必须解决两个问题：

- Squirrel mechanism：当前 Squirrel 均值已由负转正，但 shuffled 更强；后续需要解释或修复高密度图上 true-pair invariance 不可靠的问题；
- 高密度扰动配对：DPRRNV 在 Squirrel 有修复信号，但 full-shuffled control 不够干净，不能作为主方法；若后续继承该线索，必须设计节点级而非图级的配对可靠性 gate；
- 强基线对齐：RRNV 仍只与内部 `gcn_mlp_gcl` 对齐，尚未和 PolyGCL / S3GCL / GraphECL 等强基线同协议比较。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
cat runs/dsrrnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
cat runs/dprrnv_split0_s0_e50/aggregate_vs_gcn_mlp.csv
```

若继续方法实验，优先围绕 DS-RRNV 做高密度图机制诊断；停止 `darrnv_gcl` 和 `dirrnv_gcl`，DPRRNV 仅作为 Squirrel 机制线索，不作为下一轮默认扩展对象。
