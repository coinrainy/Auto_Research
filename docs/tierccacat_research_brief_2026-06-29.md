# TierCCACat 研究简报（2026-06-29）

## 阶段判定

`TierCCACat` 是当前保留的主候选。它不是回到已失败的随机增强 GCL 主线，而是在 `TierSpecProp` 已经证明强于 AutoProp 的基础上，吸收 `ccacat` 在部分 PubMed split 上的互补优势，同时在高谱集中图上安全退回纯 `TierSpecProp`。

关键判断：纯 `TierSpecProp` 不能继续包装成最终主方法。PubMed 本地基线面板显示，它对 `propccat` 和 `gracecat` 仍然较稳，但对 `ccacat` 只有很小均值优势且 split 胜负不稳。因此下一步应验证“安全谱核心 + 去相关残差”的条件融合，而不是继续微调纯谱压缩。

## 方法定义

1. 先运行 `TierSpecProp`：
   - AutoProp 传播残差平台期选择传播深度。
   - 构造传播银行。
   - 根据 top-10 PCA 能量选择回退、窄核心 rank=16 或宽核心 rank=32。
2. 只在窄谱过渡区启用残差融合：
   - 若 `selected_pca_rank == 16`，训练 CCA-GCN 分支，并拼接 L2-normalized 谱核心与 L2-normalized CCA 表征。
   - 若回退或选择宽核心 rank=32，直接返回 `TierSpecProp`，不训练残差分支。
3. 这个门控不使用测试标签，也不使用验证标签选择方法；验证标签仍只用于 frozen linear probe 的常规模型选择。

## 当前证据

### PubMed 本地基线面板

使用 class-balanced random split seeds 0-9，训练/验证每类 20/30，剩余节点测试，probe 为 sklearn logistic regression，C-grid 为 0.25/1/4/16。

| 对比 | Baseline Mean | TierCCACat Mean | Delta Mean | Wins/Ties/Losses | Wilcoxon greater p |
|---|---:|---:|---:|---:|---:|
| TierSpecProp | 0.7739 | 0.7828 | +0.0089 | 8/0/2 | 0.0098 |
| ccacat | 0.7714 | 0.7828 | +0.0115 | 7/1/2 | 0.0820 |
| propccat | 0.7603 | 0.7828 | +0.0225 | 10/0/0 | 0.0010 |
| gracecat | 0.7588 | 0.7828 | +0.0241 | 9/0/1 | 0.0020 |

这个结果支持继续验证 `TierCCACat`，但还不能宣称已达到 SOTA。尤其相对 `ccacat` 的 p 值仍不足，需要更多数据集和 split。

### Photo 与 WikiCS 宽谱安全性

`TierCCACat` 在 Photo 和 WikiCS 上均选择宽核心 rank=32，因此 `fusion_applied=0`，逐 split 等价于 `TierSpecProp`。这验证了条件门控修复了无条件融合在 Photo seed 0 上的损伤。

| Dataset | Splits | Fusion Applied | Mean Acc | vs TierSpecProp | vs AutoProp |
|---|---:|---:|---:|---:|---:|
| Photo | 10 | 0/10 | 0.9035 | +0.0000, 0 胜 10 平 0 负 | +0.0218, 10 胜 0 负, p=0.0010 |
| WikiCS | 20 | 0/20 | 0.7833 | +0.0000, 0 胜 20 平 0 负 | +0.0197, 20 胜 0 负, p=4.42e-05 |

### 条件融合必要性

无条件拼接在 Photo split 0 上会损伤强谱核心：`TierSpecProp` 为 0.8985，而无条件融合为 0.8898。因此当前实现改为条件融合：

| Dataset/Split | Rank | Fusion Applied | TierCCACat Test Acc | 解释 |
|---|---:|---:|---:|---|
| PubMed seed 0 | 16 | 1 | 0.7916 | 窄谱过渡区，残差分支有收益 |
| Photo seed 0 | 32 | 0 | 0.8985 | 宽谱核心已强，跳过残差避免损伤 |

## 失败边界

- `TierCCACat` 目前在 PubMed/Photo/WikiCS 上完成候选级验证，但仍缺少更多非 Coauthor 数据集和官方强 baseline。
- 相对 `ccacat` 的 PubMed 优势尚未达到很强统计显著性，不能作为论文结论。
- 条件规则依赖当前谱能量阈值 0.34/0.36，必须在 WikiCS、Photo 多 split、更多非 Coauthor 同配图上继续验证。
- 现有 `ccacat`、`gracecat`、`propccat` 是仓库内轻量实现，不是官方强 baseline。论文级比较仍需 PROPGCL/PROP、HomoGCL、BGRL、CCA-SSG、GRACE/GCA、SGRL/RELGCL/IRGCL 等官方或复现级 baseline。

## 下一步门槛

1. 扩展更多非 Coauthor 同配数据集，优先考虑 ogbn-arxiv 或可承受规模的公开同配图。
2. 跑 PubMed/Photo/WikiCS 的 `TierCCACat` vs `ccacat`、`propccat`、`gracecat` 更完整本地面板，尤其补齐 Photo/WikiCS 上训练型 baseline。
3. 若 `TierCCACat` 相对 `ccacat` 的优势不能扩大，应放弃 `TierCCACat` 作为主方法，仅把谱门控保留为诊断工具。
4. 若更多数据集保持稳定，再进入 PROPGCL/PROP、HomoGCL、BGRL、CCA-SSG、GRACE/GCA、SGRL/RELGCL/IRGCL 等官方强 baseline 阶段。
