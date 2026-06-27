# Redundancy-Reduced GCL 候选研究日志

日期：2026-06-28

## 候选问题

前几条路线的共同失败点是：false-negative attenuation、spectral view mix、prototype reweighting / consistency 都容易依赖不可靠的节点或原型对应关系。RR-GCL 转向 negative-free redundancy reduction：

> 不再使用 InfoNCE 的负样本，也不使用即时原型伪标签，而是让两个增强视图的投影特征交叉相关矩阵逼近单位阵：对角项保证同一节点跨视图一致，非对角项降低维度冗余并防止塌缩。

这条路线不是原创空白。CCA-SSG 与 Graph Barlow Twins 已经证明 redundancy reduction 在图 SSL 中有效。因此当前实现的目标不是宣称首次提出，而是判断：在当前 GRACE/WebKB/Actor 协议下，negative-free objective 是否能提供比前几条失败路线更稳的异配图线索。

参考来源：

- CCA-SSG, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf
- CCA-SSG official code: https://github.com/hengruizhang98/CCA-SSG
- Graph Barlow Twins: https://arxiv.org/abs/2106.02466
- Barlow Twins: https://arxiv.org/abs/2103.03230

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Cornell --method rr_gcl \
  --rr-offdiag-weight 0.005 \
  --rr-loss-scale 1.0
```

核心机制：

- 使用 GRACE 原增强视图；
- 通过原 GRACE projection head 得到两个 view 的 projected features；
- 对每个 feature dimension 做 batch 标准化；
- 计算两个 view 的 cross-correlation matrix；
- 对角项逼近 1，非对角项逼近 0；
- `--shuffle-weights` 在 RR-GCL 中表示打乱 positive node correspondence，用作机制 control。

当前 RR-GCL 不使用 InfoNCE，不使用负样本，不使用原型，也不使用 teacher。

## 初筛结果

命令目录：`experiments/grace_idea/runs/rr_gcl_splits0-2_seed0_e100`

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs，比较 GRACE、RR-GCL-normal、RR-GCL-shuffled-positive。

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | -0.003509 | 2/0/1 | -0.023657 | 1/0/2 | +0.015351 | +0.050111 |
| Cornell | +0.036036 | 2/0/1 | +0.082584 | 3/0/0 | +0.009009 | +0.016352 |
| Texas | +0.000000 | 1/1/1 | -0.051877 | 1/0/2 | +0.027027 | -0.007387 |
| Wisconsin | -0.006536 | 1/1/1 | -0.016696 | 0/0/3 | -0.019608 | -0.037842 |

class-level 线索：

- Cornell 五个类别平均 delta 全部非负，`F1Class2` +0.132536，`F1Class4` +0.111111；
- Actor 的 `F1Class4` 为 +0.012880，但 `F1Class0` 为 -0.067279；
- Texas `F1Class3` 为 +0.011953，但 `F1Class0` 为 -0.205942；
- Wisconsin `F1Class1` 为 -0.089213。

## 当前判断

RR-GCL 是目前几条失败路线中最有研究线索的一个，但仍不能作为 active SOTA candidate：

- Cornell 上 normal 同时优于 GRACE 和 shuffled，且 macro / class-level 信号清楚；
- Actor 上 normal 明显优于 shuffled，但仍低于 GRACE macro；
- Texas/Wisconsin 不稳，尤其 macro 退化明显；
- 说明 negative-free redundancy reduction 可以避免一部分 false-negative / prototype-target 问题，但裸 objective 不适合所有异配图。

## 保留资产

- `--method rr_gcl` 提供 negative-free redundancy-reduction objective；
- shuffled positive correspondence control 已打通；
- 训练日志记录 `rr_on_diag_loss`、`rr_off_diag_loss`、`rr_cross_corr_diag_mean`、`rr_cross_corr_offdiag_mean_abs`；
- Cornell 的 class-level 结果提示 redundancy reduction 可能改善少数类/弱类。

## 下一步

不建议直接把 RR-GCL 扩成论文主方法。更合理的下一步是：

1. 设计 adaptive RR-GCL：先判断数据集/节点区域是否适合 redundancy reduction，再决定和 InfoNCE 的混合比例；
2. 研究为什么 Cornell 受益而 Texas/Wisconsin/Actor 不稳：重点看 class imbalance、local homophily、feature dimension redundancy、projection collapse；
3. 尝试 hybrid objective：`GRACE InfoNCE + small RR regularizer`，而不是完全替换 InfoNCE；
4. 用 shuffled positive control 保持机制可证伪：hybrid normal 必须稳定优于 shuffled。

当前建议的下一步命令不是扩 10 splits，而是先跑 hybrid 小实验。
