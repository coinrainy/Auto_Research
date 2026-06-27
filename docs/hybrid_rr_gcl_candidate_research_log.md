# Hybrid RR-GCL 候选研究日志

日期：2026-06-28

## 候选问题

裸 `rr_gcl` 在 Cornell 上有明显 class-level / macro 信号，但在 Texas、Wisconsin、Actor 上不稳。Hybrid RR-GCL 尝试更保守的目标：

> 保留 GRACE 的 InfoNCE 主损失，只把 redundancy reduction 作为小权重正则项，检验是否能保留 Cornell 的类别收益，同时减少裸 RR objective 对其他数据集的退化。

这不是独立新理论，而是对 RR-GCL 线索的风险控制实验。

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Cornell --method hybrid_rr_gcl \
  --hybrid-rr-weight 0.001 \
  --rr-offdiag-weight 0.005
```

核心机制：

- 先按 GRACE 计算两个增强视图的 InfoNCE；
- 同时对两个 view 的 projection features 计算 RR/Barlow 风格 cross-correlation loss；
- 总损失为 `InfoNCE + hybrid_rr_weight * RR`;
- `--shuffle-weights` 只打乱 RR 正样本对应关系，InfoNCE 仍保持正常，用于机制 control；
- 训练日志新增 `rr_loss`，并继续记录 `rr_on_diag_loss`、`rr_off_diag_loss`、`rr_cross_corr_diag_mean`、`rr_cross_corr_offdiag_mean_abs`。

## 初筛结果

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs，比较 GRACE、Hybrid RR normal、Hybrid RR shuffled。

### `hybrid_rr_weight=0.01`

输出目录：`experiments/grace_idea/runs/hybrid_rr_gcl_splits0-2_seed0_e100`

| Dataset | F1Mi delta vs GRACE | F1Ma delta vs GRACE | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: |
| Actor | -0.001535 | -0.003084 | +0.006798 | +0.018421 |
| Cornell | -0.018018 | +0.053575 | -0.027027 | +0.022745 |
| Texas | ~0.000000 | +0.014508 | -0.036036 | -0.071037 |
| Wisconsin | ~0.000000 | -0.005397 | +0.013072 | -0.024415 |

判断：macro 局部改善，但 Texas normal 明显低于 shuffled；Cornell micro 也低于 GRACE。不能作为主候选。

### `hybrid_rr_weight=0.001`

输出目录：`experiments/grace_idea/runs/hybrid_rr_gcl_w001_splits0-2_seed0_e100`

| Dataset | F1Mi delta vs GRACE | F1Ma delta vs GRACE | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: |
| Actor | ~0.000000 | -0.004574 | +0.004825 | -0.008117 |
| Cornell | +0.009009 | +0.060894 | -0.009009 | -0.005948 |
| Texas | -0.018018 | -0.015623 | -0.036036 | -0.042888 |
| Wisconsin | -0.019608 | +0.114149 | +0.013072 | +0.145617 |

判断：弱 RR 正则强化了 Cornell/Wisconsin macro 线索，但 Texas 明确失败，且 Cornell normal 仍未稳定优于 shuffled。固定全局 RR 权重仍不能作为 active SOTA candidate。

## 当前判断

Hybrid RR-GCL 降级为条件性诊断资产，而非主方法：

- RR 辅助目标确实能改变类别覆盖，尤其 Cornell/Wisconsin macro 与少数类 F1 有信号；
- 但 micro accuracy 与 normal-vs-shuffled 对照不稳；
- 固定全局 RR 正则无法解释“何时该启用 RR、对哪些节点/类别/维度启用 RR”；
- 如果继续该方向，必须转为 adaptive / class-sensitive / region-sensitive RR，而不是继续调全局 `hybrid_rr_weight`。

## 保留资产

- `--method hybrid_rr_gcl`；
- `--hybrid-rr-weight`；
- InfoNCE + RR 正则训练路径；
- shuffled RR positive correspondence control；
- RR loss 与 cross-correlation 诊断日志；
- Cornell/Wisconsin macro/少数类覆盖线索。

## 下一步

不再继续固定全局 hybrid RR 权重搜索。下一步应进入一个更明确的新候选：

1. Adaptive RR gate：只在 representation redundancy 高、类别/局部区域可能受益的位置启用 RR；
2. Class/cluster-balanced RR：把 RR 从全局 batch correlation 改为 prototype/cluster 条件下的 decorrelation，避免牺牲主流类别 micro accuracy；
3. 设计机制 control：adaptive gate normal 必须优于 shuffled gate，并且不能只靠 macro 偶然提升。

建议下一轮先做最小实现：`GRACE InfoNCE + cluster-balanced RR regularizer`，用 EMA/consensus embedding 聚类后在每个 cluster 内或 cluster-balanced sample 上计算 RR，目标是保留 Cornell/Wisconsin macro 收益，同时避免 Texas/Actor micro 退化。
