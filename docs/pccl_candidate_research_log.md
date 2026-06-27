# Prototype Consistency Contrastive Learning 候选研究日志

日期：2026-06-28

## 候选问题

PBCL 证明了“只做 prototype-density anchor reweighting”不足以支撑新方法。PCCL 的候选假设进一步推进为：

> 如果 node-level InfoNCE 缺少簇级语义约束，那么可以在 GRACE 上增加 prototype-level cross-view consistency，使两个增强视图不仅节点实例对齐，也在无标签原型空间中保持一致，同时用 prototype usage balance 抑制塌缩。

这个候选真正改变 objective，而不是只改变增强或 anchor 权重。

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Texas --method pccl \
  --warmup-epochs 20 \
  --pccl-num-prototypes 0 \
  --pccl-kmeans-iters 10
```

核心机制：

- warmup 前使用标准 GRACE；
- warmup 后用两个 view 的 encoder embedding 均值做 consensus embedding；
- 对 consensus embedding 做 KMeans，得到 prototype centers；
- 用 consensus-to-prototype soft target 监督两个 view 的 prototype assignment；
- 增加 prototype usage balance loss，避免所有节点集中到少数 prototype；
- `--shuffle-weights` 在 PCCL 中表示打乱节点-prototype soft target 对应关系，用作机制 control。

关键参数：

- `--pccl-num-prototypes`
- `--pccl-kmeans-iters`
- `--pccl-prototype-temperature`
- `--pccl-target-temperature`
- `--pccl-consistency-weight`
- `--pccl-balance-weight`

默认权重较保守：`pccl_consistency_weight=0.05`，`pccl_balance_weight=0.01`。较大的 prototype loss 在 smoke 中会明显拉高总 loss，容易压过 GRACE。

## 初筛结果

命令目录：`experiments/grace_idea/runs/pccl_splits0-2_seed0_e100`

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs，比较 GRACE、PCCL-normal、PCCL-shuffled。

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | -0.000877 | 1/0/2 | -0.001739 | 1/0/2 | -0.000658 | +0.002131 |
| Cornell | +0.009009 | 1/2/0 | +0.034401 | 2/0/1 | -0.009009 | -0.015757 |
| Texas | -0.009009 | 0/2/1 | -0.001840 | 0/2/1 | -0.027027 | -0.023827 |
| Wisconsin | -0.026144 | 0/1/2 | +0.019288 | 1/1/1 | -0.019608 | +0.002649 |

class-level 线索：

- Cornell `F1Class0` 平均 +0.077082，`F1Class2` 平均 +0.106061；
- Wisconsin `F1Class4` 平均 +0.133333，但 `F1Class2` 平均 -0.030225；
- Texas 基本没有类别收益，`F1Class3` 轻微下降；
- Actor 接近零但略负。

## 当前判断

PCCL 未通过最小标准，不能作为 active candidate。主要原因：

- Texas 和 Wisconsin 的 micro-F1 为负；
- Actor 近零略负；
- Cornell macro 有局部正向，但 normal 明显不优于 shuffled；
- shuffled target control 多次等于或超过 normal，说明当前 prototype target 与节点语义的对应关系不可靠；
- 该 objective 可能只是轻微正则化，而不是可解释的 prototype-level 语义约束。

## 保留资产

- `--method pccl` 提供了 prototype-level consistency / balance objective 的可运行框架；
- normal/shuffled prototype-target control 已打通；
- 训练日志记录 `prototype_consistency_loss`、`prototype_balance_loss`、`prototype_usage_entropy`、`prototype_usage_min/max`；
- 结果显示 Cornell 某些类别可能受益于 prototype objective，但当前 KMeans soft target 不够可靠。

## 下一步

当前不建议继续扩 PCCL 到 10 splits。prototype 路线如果要继续，必须解决原型目标可靠性问题，例如：

1. 使用跨 epoch EMA prototype，而不是每个 epoch 重新 KMeans；
2. 只对 high-confidence prototype assignments 加 consistency；
3. 引入 graph-structure-aware prototype，而不是纯 embedding KMeans；
4. 将 prototype objective 改成 prototype decorrelation / redundancy reduction，而不是节点级 soft target imitation。

在没有这些修正前，PCCL 与 PBCL 都应作为失败边界，而不是论文主线。
