# CoreSpecProp 研究简报（2026-06-29）

## 阶段判定

`CoreSpecProp` 是当前保留的条件性候选。它不是继续微调已经失败的 `homogcl` / `horpgcl`，而是沿着 2025-2026 图对比学习中的传播充分性发现继续推进：先承认训练免费传播是强基线，再只在无标签谱诊断显示存在稳定低秩核心时做压缩去噪。

当前结论：可以继续投入论文级扩展，但不能宣称已经达到 2026 SOTA。下一阶段必须纳入 PROPGCL、HomoGCL、BGRL、CCA-SSG、GCA/GRACE、SGRL/RELGCL/IRGCL 等强 baseline，并扩展更多同配图。

## 方法定义

1. 用 AutoProp 规则从传播残差平台期自动选择传播深度 `K`。
2. 构造传播银行 `[X, PX, ..., P^KX]`，所有块做 L2 normalize。
3. 对传播银行做 PCA 谱诊断，记录 top-10 能量占比、参与秩和能量累计秩。
4. 若 top-10 能量占比 `< 0.34`，直接回退为 AutoProp 表征，避免中等谱集中图被错误压缩。
5. 若 top-10 能量占比 `>= 0.34`，按 `round(participation_rank / 3)` 选择核心秩，并裁剪到 `[16, 32]`；当前正例中选择 rank=16。

这个规则不使用测试标签。标签只用于 linear probe、验证集选择 probe 超参、最终测试集评估和 homophily 诊断字段。

## 与现有工作的差异

- HomoGCL 强调同配邻居正样本扩展；本方法不扩展正样本集合，而是判断传播银行是否存在可去噪的谱核心。
- PROPGCL 指出传播本身对 GCL 很强，且变换层可能与下游目标错配；本方法进一步问：在传播已经很强时，哪些图需要低秩去噪，哪些图应该完全不动。
- 固定 rank 的 `specprop` 在部分图上有效，但规则不够自适应；`CoreSpecProp` 用参与秩决定核心压缩秩，并保留安全回退。

## 当前证据

### 5 数据集、3 split 早筛

使用 class-balanced random split seeds 0/1/2，训练/验证每类 20/30，剩余节点测试。对比 `autopropcat`：

| Dataset | AutoProp Mean | CoreSpecProp Mean | Mean Delta | Wins/Losses | Gate |
|---|---:|---:|---:|---:|---|
| Cora | 0.8212 | 0.8212 | +0.0000 | 0/0 | fallback |
| CiteSeer | 0.7107 | 0.7107 | +0.0000 | 0/0 | fallback |
| PubMed | 0.7530 | 0.7804 | +0.0274 | 3/0 | compress |
| Computers | 0.7965 | 0.7965 | +0.0000 | 0/0 | fallback |
| Photo | 0.8745 | 0.9051 | +0.0306 | 3/0 | compress |

### 正例图、10 split 压力测试

使用 PubMed 和 Photo 的 split seeds 0-9。对比 `autopropcat`：

| Dataset | AutoProp Mean | CoreSpecProp Mean | Mean Delta | Std Delta | Min Delta | Wins/Losses | Wilcoxon greater p |
|---|---:|---:|---:|---:|---:|---:|---:|
| PubMed | 0.7541 | 0.7739 | +0.0198 | 0.0108 | +0.0043 | 10/0 | 0.000977 |
| Photo | 0.8817 | 0.9002 | +0.0185 | 0.0134 | +0.0014 | 10/0 | 0.000977 |

诊断结果一致：PubMed top-10 谱能量约 0.3555、参与秩约 44.73；Photo top-10 谱能量约 0.3691、参与秩约 46.46。二者均触发压缩，选择 rank=16。

## 失败边界

- Cora、CiteSeer、Computers 当前只证明“不伤害”，不是性能贡献。
- 10-split 压力测试只覆盖两个正例图，仍不足以支撑顶会主表。
- 当前实现是训练免费传播谱去噪，不应被包装成传统 augment-contrast GCL；更合理的论文叙事是“传播充分性之后的谱核心去噪”，再讨论它对 GCL 表征学习的启发。
- 现有评估仍缺少官方强 baseline；若 PROPGCL 或 BGRL/CCA-SSG 在相同协议下显著更强，必须承认本方法只是高效证伪器或辅助模块。

## 下一步门槛

1. 扩展更多同配数据集：WikiCS、Coauthor CS/Physics、ogbn-arxiv 或其他可承受规模数据。
2. 将 `CoreSpecProp` 与 PROPGCL/PROP、BGRL、CCA-SSG、GRACE/GCA、HomoGCL、SGRL/RELGCL/IRGCL 做同 split、公平 probe 对比。
3. 加入消融：固定 rank=16/32、无 gate、只 AutoProp、只 PCA 原特征、不同高谱阈值。
4. 若学习式 GCL 继续推进，应把 `CoreSpecProp` 作为 teacher/filter，而不是重新引入容易过拟合的随机增强 encoder。
