# Gated CBR-GCL 候选研究日志

日期：2026-06-28

## 候选问题

CBR-GCL 在 Texas/Wisconsin 上有小幅正向，并在 Wisconsin macro 上显示少数/弱类覆盖线索，但 Actor 近零略负，说明 cluster-balanced RR 不应无条件启用。Gated CBR-GCL 尝试用训练时可观测信号动态调节 CBR 强度：

> 如果两个增强视图在 cluster-balanced RR cross-correlation 的 diagonal mean 上不够高，就降低 CBR_RR 权重，避免在 positive correspondence 不可靠时强行施加 RR。

该 gate 不使用标签、不使用验证集、不按数据集名特殊处理。

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Texas --method gated_cbr_gcl \
  --warmup-epochs 20 \
  --cbr-gate-min-diag 0.82 \
  --cbr-gate-temperature 0.03
```

核心机制：

- 继承 `cbr_gcl` 的 cluster-balanced RR；
- 在每个 epoch 计算 weighted cross-correlation diagonal mean；
- gate scale 为 `sigmoid((diag_mean - cbr_gate_min_diag) / cbr_gate_temperature)`；
- 实际 CBR loss 为 `raw_CBR_RR * gate_scale`；
- `--shuffle-weights` 仍只打乱 RR positive correspondence；
- 新增日志 `cbr_raw_rr_loss` 与 `cbr_gate_scale`。

## 实验结果

### 默认 gate：`cbr_gate_min_diag=0.82`

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs。

输出：

- `runs/gated_cbr_gcl_splits0-2_seed0_e100`
- `runs/summaries/gated_cbr_gcl_splits0-2_seed0_e100_aggregate.csv`

| Dataset | F1Mi delta vs GRACE | F1Ma delta vs GRACE | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: |
| Actor | +0.000219 | -0.001782 | +0.001535 | +0.001241 |
| Cornell | +0.018018 | +0.052043 | -0.009009 | -0.014896 |
| Texas | +0.009009 | -0.010165 | 0.000000 | 0.000000 |
| Wisconsin | -0.032680 | +0.010650 | -0.032680 | +0.010650 |

诊断：该 gate 保护了 Actor，但明显削弱 Texas 收益，并使 Wisconsin micro 明显退化。

最终 gate scale 均值：

- Texas：0.878973；
- Cornell：0.844730；
- Wisconsin：0.850532；
- Actor：0.055022。

说明 gate 确实区分了 Actor，但过度依赖 diagonal mean 会把 CBR 的有益部分也压掉或改变训练轨迹。

### 宽松 gate：`cbr_gate_min_diag=0.78`

实验设置同上。

输出：

- `runs/gated_cbr_gcl_diag078_splits0-2_seed0_e100`
- `runs/summaries/gated_cbr_gcl_diag078_splits0-2_seed0_e100_aggregate.csv`

| Dataset | F1Mi delta vs GRACE | F1Ma delta vs GRACE | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: |
| Actor | -0.000439 | -0.002001 | +0.001754 | +0.000017 |
| Cornell | -0.036036 | -0.080680 | -0.072072 | -0.112982 |
| Texas | +0.027027 | +0.024448 | +0.027027 | +0.012444 |
| Wisconsin | ~0.000000 | -0.000417 | ~0.000000 | -0.000417 |

诊断：降低阈值恢复了 Texas 的收益，但 Cornell 崩溃，且 Wisconsin 不再保留 CBR 的 macro 线索。

## 当前判断

单一 RR diagonal confidence gate 失败，不能作为 active candidate：

- `0.82`：太保守，保护 Actor 但牺牲 Texas/Wisconsin；
- `0.78`：保留 Texas，但 Cornell normal 明显低于 GRACE 与 shuffled；
- 两档都没有同时满足 `normal >= GRACE` 与 `normal > shuffled`；
- 说明“view positive correspondence 强度”不是决定 CBR 是否有益的充分信号。

## 保留资产

- `--method gated_cbr_gcl`；
- `--cbr-gate-min-diag`、`--cbr-gate-temperature`、`--cbr-gate-min-scale`；
- `cbr_raw_rr_loss` 与 `cbr_gate_scale` 日志；
- 作为后续 gate 消融的负结果基线。

## 下一步

停止继续调 diagonal threshold。下一轮应转向更结构化的 gate：

1. cluster compactness / separation gate：只在无监督 clusters 有足够紧致度和可分离性时启用 CBR；
2. EMA cluster assignment stability gate：只在连续 epoch 的 cluster assignment 稳定时启用 CBR；
3. region-level gate：按节点/cluster 局部同配度、degree 或 feature-neighborhood agreement 调节 CBR，而非全图单一 scale；
4. 所有 gate 必须保留 shuffled positive correspondence control。

建议下一步实现 `stable_cluster_cbr_gcl`，先记录 cluster assignment stability 与 compactness，再决定是否将它用于 loss。
