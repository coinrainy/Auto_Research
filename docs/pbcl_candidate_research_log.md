# Prototype-Balanced Contrastive Learning 候选研究日志

日期：2026-06-28

## 候选问题

前两条路线暴露出一个共同问题：单纯依赖样本对可靠性或增强视图语义，容易在少数类、低密度区域或 split 变化时失效。PBCL 的候选假设是：

> 在无标签节点分类的 GCL 中，InfoNCE 的 anchor 平均化会让大簇/高密度区域主导训练；如果用无标签原型簇大小估计节点所处区域的密度，并对低密度原型中的节点提高 anchor 权重，可能改善 macro-F1 或少数类表现。

这个候选不是 positive mining，也不是 false-negative attenuation；它只改变 anchor loss aggregation。

## 文献边界

快速边界刷新显示，以下方向已经非常拥挤：

- adaptive augmentation / augmentation-aware GCL；
- spectral augmentation / heterophily GCL；
- false negative、hard negative、positive mining；
- degree-bias adaptive reweighting；
- prototype-based graph clustering。

因此 PBCL 不能宣称“首次做原型”或“首次做 reweighting”。更保守的潜在边界是：用 shuffled prototype-density control 明确检验“低密度原型 anchor reweighting 是否真的依赖节点-原型对应关系”。

参考来源包括：

- Self-Reinforced Graph Contrastive Learning, arXiv 2025: https://arxiv.org/abs/2505.13650
- Does GCL Need a Large Number of Negative Samples?, arXiv 2025: https://arxiv.org/abs/2503.17908
- Mitigating Degree Bias Adaptively with Hard-to-Learn Nodes in GCL, arXiv 2025: https://arxiv.org/abs/2506.05214
- Positive Mining Graph Contrastive Learning, 2025: https://link.springer.com/article/10.1007/s13042-025-02924-2

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Texas --method pbcl \
  --warmup-epochs 20 \
  --pbcl-num-prototypes 0 \
  --pbcl-kmeans-iters 10
```

核心机制：

- warmup 前使用标准 GRACE；
- warmup 后用两个 view 的 encoder embedding 均值作为 consensus embedding；
- 对 consensus embedding 做轻量 torch KMeans；
- 默认 `K = dataset.num_classes`，也可用 `--pbcl-num-prototypes` 显式指定；
- 对小原型簇中的节点赋予更高 anchor weight；
- 使用 `--shuffle-weights` 打乱节点-权重对应关系，作为主要证伪 control。

关键参数：

- `--pbcl-num-prototypes`
- `--pbcl-kmeans-iters`
- `--pbcl-weight-power`
- `--pbcl-min-weight`
- `--pbcl-max-weight`

## 初筛结果

命令目录：`experiments/grace_idea/runs/pbcl_splits0-2_seed0_e100`

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs，比较 GRACE、PBCL-normal、PBCL-shuffled。

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg | normal - shuffled F1Mi | normal - shuffled F1Ma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor | -0.008114 | 0/0/3 | -0.011311 | 0/0/3 | -0.008114 | -0.006809 |
| Cornell | +0.018018 | 2/1/0 | +0.019889 | 2/1/0 | -0.009009 | -0.002785 |
| Texas | +0.009009 | 1/2/0 | -0.002744 | 1/1/1 | +0.018018 | -0.013491 |
| Wisconsin | +0.000000 | 1/1/1 | -0.013541 | 0/1/2 | +0.013072 | -0.033674 |

class-level 线索：

- Actor 五个类别平均 delta 全部为负；
- Cornell `F1Class0` 平均 +0.088304，但 `F1Class2` 平均 -0.010101；
- Texas 主要改善 `F1Class3`，但 `F1Class0` 轻微下降；
- Wisconsin `F1Class1` 平均 -0.079312。

## 当前判断

PBCL 原型未通过最小标准，不能作为 active candidate。主要原因：

- Actor 三个 split 全部负向；
- Wisconsin macro 负向；
- Cornell/Texas 虽有局部正向，但 normal 并不稳定优于 shuffled；
- shuffled control 经常接近或超过 normal，说明收益可能来自一般 reweighting 正则，而不是可靠的节点-原型密度对应关系。

## 保留资产

- `--method pbcl` 提供了一个可复用的 prototype-density anchor weighting 框架；
- normal/shuffled control 链路已打通；
- 该原型证明“简单逆簇频率 reweighting”不足以形成顶会级主线；
- 后续若继续 density/prototype 方向，必须引入更强语义约束，例如稳定原型、跨视图原型一致性、类别偏置诊断或 prototype-level contrastive objective，而不是只做 anchor reweighting。

## 下一步

当前不建议继续扩 PBCL 到 10 splits 或更多数据集。下一轮应重新构思，优先考虑：

1. 改变 contrastive objective，而不是只改 loss 权重；
2. 显式建模 view selection decision，而不是固定增强；
3. 引入可证伪机制诊断，且 normal 必须稳定优于 shuffled/random control；
4. 若继续 prototype 方向，应从 node anchor reweighting 转为 prototype-level alignment / decorrelation / anti-collapse。

建议下一步不是直接跑更多 PBCL，而是先做新 idea 设计或实现 prototype-level objective。
