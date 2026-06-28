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

## 下一步

保留 `rrnv_gcl` 为当前最有价值候选，但后续必须解决两个问题：

- Squirrel safety：不能用牺牲 Texas/Chameleon 的弱辅助项来解决，应考虑 representation-level fallback 或 graph-view reliability，而不是 loss-level 小权重；
- 强基线对齐：RRNV 仍只与内部 `gcn_mlp_gcl` 对齐，尚未和 PolyGCL / S3GCL / GraphECL 等强基线同协议比较。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
cat runs/rrnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

若继续方法实验，优先实现 RRNV 的 safety 版本，但停止 `darrnv_gcl` 这条 density-gated auxiliary-loss 路线。
