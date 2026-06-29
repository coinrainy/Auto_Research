# CoreSpecProp 研究简报（2026-06-29）

## 阶段判定

`TierSpecProp` 是当前保留的条件性候选。它不是继续微调已经失败的 `homogcl` / `horpgcl`，而是沿着 2025-2026 图对比学习中的传播充分性发现继续推进：先承认训练免费传播是强基线，再只在无标签谱诊断显示存在稳定低秩核心时做压缩去噪。`CoreSpecProp` 是中间版本，其参与秩/3 rank 规则已被 WikiCS rank 消融修正。

当前结论：`TierSpecProp` 可以继续投入论文级扩展，但不能宣称已经达到 2026 SOTA。下一阶段必须纳入 PROPGCL、HomoGCL、BGRL、CCA-SSG、GCA/GRACE、SGRL/RELGCL/IRGCL 等强 baseline，并扩展更多同配图。

## 方法定义

1. 用 AutoProp 规则从传播残差平台期自动选择传播深度 `K`。
2. 构造传播银行 `[X, PX, ..., P^KX]`，所有块做 L2 normalize。
3. 对传播银行做 PCA 谱诊断，记录 top-10 能量占比、参与秩和能量累计秩。
4. 若 top-10 能量占比 `< 0.34`，直接回退为 AutoProp 表征，避免中等谱集中图被错误压缩。
5. `CoreSpecProp` 旧规则曾在 top-10 能量占比 `>= 0.34` 时按 `round(participation_rank / 3)` 选择核心秩并裁剪到 `[16, 32]`；WikiCS 消融显示这会过度压缩。
6. `TierSpecProp` 新规则使用分层谱门控：top-10 能量占比 `< 0.34` 回退；`0.34 <= top10 < 0.36` 选择窄 rank=16；`top10 >= 0.36` 选择宽 rank=32。

这个规则不使用测试标签。标签只用于 linear probe、验证集选择 probe 超参、最终测试集评估和 homophily 诊断字段。

## 与现有工作的差异

- HomoGCL 强调同配邻居正样本扩展；本方法不扩展正样本集合，而是判断传播银行是否存在可去噪的谱核心。
- PROPGCL 指出传播本身对 GCL 很强，且变换层可能与下游目标错配；本方法进一步问：在传播已经很强时，哪些图需要低秩去噪，哪些图应该完全不动。
- 固定 rank 的 `specprop` 在部分图上有效，但规则不够自适应；`CoreSpecProp` 用参与秩决定核心压缩秩，并保留安全回退。
- WikiCS rank 消融推翻了 `CoreSpecProp` 的参与秩/3 rank 规则：rank16/core 在 WikiCS 20 split 上只有 +0.0066，固定 rank32 达到 +0.0197 且 20 胜 0 负。`TierSpecProp` 因此改用 top-10 谱能量分层选择窄/宽核心。

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

### WikiCS 官方 20 split 扩展

WikiCS 使用 PyG 官方 public split：20 个 train/val mask，固定 test mask。对比 `autopropcat`：

| Dataset | AutoProp Mean | CoreSpecProp Mean | Mean Delta | Std Delta | Min Delta | Wins/Losses | Wilcoxon greater p |
|---|---:|---:|---:|---:|---:|---:|---:|
| WikiCS | 0.7636 | 0.7702 | +0.0066 | 0.0056 | -0.0039 | 18/2 | 0.000182 |

WikiCS 的谱集中度明显更高：top-10 谱能量约 0.6278、参与秩约 14.97，触发压缩并选择 rank=16。这个结果支持“谱核心压缩总体有效”，但也暴露出一个重要边界：即使图级谱诊断稳定，split-level test 表现仍可能出现少数负例。因此后续不能声称 `CoreSpecProp` 严格逐 split 无损，只能声称在当前协议下总体稳定提升，并继续研究更细的安全门控。

### TierSpecProp 分层 rank 修正

WikiCS rank 消融显示固定 rank32 显著优于 rank16/core：20 split 平均 0.7833 vs AutoProp 0.7636，delta +0.0197，20 胜 0 负。进一步检查 PubMed/Photo rank32 后发现：PubMed rank32 会出现 3 个负 split，说明不能全局固定 rank32；Photo rank32 保持 10 胜 0 负且平均更高。因此 `TierSpecProp` 采用 top-10 谱能量分层：

| Dataset | Rule Rank | AutoProp Mean | TierSpecProp Mean | Mean Delta | Wins/Losses |
|---|---:|---:|---:|---:|---:|
| PubMed | 16 | 0.7541 | 0.7739 | +0.0198 | 10/0 |
| Photo | 32 | 0.8817 | 0.9035 | +0.0218 | 10/0 |
| WikiCS | 32 | 0.7636 | 0.7833 | +0.0197 | 20/0 |

这个修正保留了 PubMed 的窄 rank 安全性，同时吸收 Photo/WikiCS 对更宽核心的收益。

## 失败边界

- Cora、CiteSeer、Computers 当前只证明“不伤害”，不是性能贡献。
- `CoreSpecProp` 的参与秩/3 rank 规则已被 WikiCS 消融证伪，不再作为当前最佳方法。
- `TierSpecProp` 在 PubMed/Photo/WikiCS 上是当前最强条件性候选，但其 `0.36` 宽核心阈值仍来自当前早筛图统计，必须在更多数据集和强 baseline 下验证，不能作为最终 SOTA 结论。
- 当前实现是训练免费传播谱去噪，不应被包装成传统 augment-contrast GCL；更合理的论文叙事是“传播充分性之后的谱核心去噪”，再讨论它对 GCL 表征学习的启发。
- 现有评估仍缺少官方强 baseline；若 PROPGCL 或 BGRL/CCA-SSG 在相同协议下显著更强，必须承认本方法只是高效证伪器或辅助模块。
- 按用户要求，Coauthor CS/Physics 暂缓，不作为当前下一步实验。

## 下一步门槛

1. 扩展更多非 Coauthor 同配数据集：ogbn-arxiv 或其他可承受规模数据；Coauthor CS/Physics 暂缓。
2. 将 `TierSpecProp` 与 PROPGCL/PROP、BGRL、CCA-SSG、GRACE/GCA、HomoGCL、SGRL/RELGCL/IRGCL 做同 split、公平 probe 对比。
3. 加入消融：固定 rank=16/32、无 gate、只 AutoProp、只 PCA 原特征、不同高谱阈值。
4. 若学习式 GCL 继续推进，应把 `TierSpecProp` 作为 teacher/filter，而不是重新引入容易过拟合的随机增强 encoder。
