# 同配图 GCL 研究简报（2026-06-29）

## 2026-06-29 更新：主线调整

### 前沿压力与命名冲突

- KDD 2023 已有 **HomoGCL: Rethinking Homophily in Graph Contrastive Learning**，其核心是利用同配性扩展正样本集合，因此本项目不能继续把 `HomoGCL` 作为论文主方法名或主要创新叙事。
- ICLR 2025 投稿 **Propagation Alone is Enough for Graph Contrastive Learning** 指出训练免费传播 `PROP` 已能在节点分类上与多种 GCL 方法竞争，并提出 PROPGCL。这要求任何新 GCL 必须显式超过强传播基线。
- IJCAI 2025 **RELGCL** 从绝对相似转向相对相似保持；ICLR 2026 投稿 **IRGCL** 同时处理特征噪声、结构可靠性和度不平衡。这压缩了“同配可靠增强 + 多正样本”的新意空间。

### 当前判定

- 原 `homogcl` 候选失败：在 `results/smoke_summary.csv` 中，CiteSeer 比 `propcat` 低 0.0685，Cora 低 0.0165，PubMed 仅高 0.004，不足以继续作为主线。
- `horpgcl` 候选失败：Cora 快速测试 test accuracy 为 0.796，低于 `autopropcat` 的 0.831。
- 当前保留方向：把 `autopropcat` 作为同配图 GCL 的强证伪器，并推进 `specprop`：基于无标签谱集中度判断传播银行是否需要低秩去噪。

### AutoProp 早筛结果

| Dataset | Method | Selected K | Test Acc | Val Acc |
|---|---|---:|---:|---:|
| Cora | autopropcat | 6 | 0.831 | 0.796 |
| CiteSeer | autopropcat | 6 | 0.726 | 0.726 |
| PubMed | autopropcat | 7 | 0.789 | 0.810 |

这些结果是单 seed public split + 短 C 网格，不能声明 SOTA；它们只说明当前学习式候选没有越过传播充分性边界。

### SpecProp 候选结果

`SpecProp` 的无标签规则：先用 AutoProp 选择传播深度；计算传播银行 PCA 谱。若 top-10 能量占比 >= 0.30，则压缩到 rank=32；若 top-10 能量占比 >= 0.20，则压缩到 95% 能量 rank；否则不压缩，回退到 AutoProp。

| Dataset | AutoProp Full Grid | SpecProp Full Grid | Selected K | Selected Rank | Delta |
|---|---:|---:|---:|---:|---:|
| Cora | 0.824 | 0.834 | 6 | 647 | +0.010 |
| CiteSeer | 0.723 | 0.723 | 6 | 0 | +0.000 |
| PubMed | 0.793 | 0.798 | 7 | 32 | +0.005 |

当前证据说明 `SpecProp` 是本仓库第一条没有损伤 Planetoid 三图、且越过 AutoProp full-grid 边界的候选。但它仍是单 seed public split 证据，不能宣称 SOTA。

### SpecProp 多随机划分压力测试

使用 safe gate：只有 top-10 PCA 能量占比 >= 0.34 时才压缩到 rank=32，否则回退到 AutoProp。class-balanced random split seeds 0/1/2，训练/验证每类分别为 20/30，剩余节点为测试集。相对 AutoProp 的 paired delta：

| Dataset | AutoProp Mean | SpecProp Mean | Mean Delta | Wins/Losses | Interpretation |
|---|---:|---:|---:|---:|---|
| Cora | 0.8212 | 0.8212 | +0.0000 | 0/0 | strict rule 回退，避免中等谱集中区域损伤。 |
| CiteSeer | 0.7107 | 0.7107 | +0.0000 | 0/0 | 谱分散触发回退，按设计持平。 |
| PubMed | 0.7530 | 0.7710 | +0.0180 | 3/0 | 低秩去噪稳定有效，是当前最强信号。 |

Amazon class-random seeds 0/1/2 显示同一 safe gate 在 Photo 上平均 0.9071 vs AutoProp 0.8745，delta +0.0326，3 胜 0 负，rank=32；在 Computers 上平均 0.7965 vs AutoProp 0.7965，回退持平，修复了 0.30 阈值版本的压缩损伤。结论：`SpecProp` 不是已完成的通用 SOTA idea，而是一个有明确条件边界的候选：在传播银行谱高度集中时低秩瓶颈有价值；在谱分散或中等集中时应回退。

## Research Question Brief

### Topic Area

图对比学习；优先面向同配图的无标签节点表征学习。

### Primary Research Question

在同配属性图的节点分类任务中，无标签传播银行的谱集中度是否能预测何时需要低秩去噪，并解释 `SpecProp` 何时能够稳定超过自动选择传播深度的训练免费强基线？

### FINER Assessment

| Criterion | Score | Justification |
|---|---:|---|
| Feasible | 5/5 | Planetoid 同配图、PyG 数据加载、GRACE/PROP/HomoGCL 可在当前仓库内自实现并快速验证。 |
| Interesting | 4/5 | 2024-2026 文献共同指向“随机增强、传播强基线、结构可靠性”三类矛盾，适合形成清晰贡献。 |
| Novel | 4/5 | 单纯同配正样本扩展已有 HomoGCL；`SpecProp` 的新意在于把传播充分性边界、无标签谱集中度和低秩瓶颈连接成可证伪规则。 |
| Ethical | 5/5 | 使用公开图数据和标准节点分类协议，无人类受试者或敏感数据收集。 |
| Relevant | 4/5 | 若成立，可为 GCL 在同配图上摆脱随机增强和复杂 encoder 提供更简单有效的设计原则。 |
| **Average** | **4.2/5** | 达到继续做原型与 smoke test 的阈值。 |

### Scope Boundaries

**In Scope:** 同配图节点分类；Cora、CiteSeer、PubMed；frozen encoder + 线性 logistic regression probe；随机增强 GCL、训练免费传播、原始特征基线；后续强 baseline 对齐 IRGCL、PROPGCL、GOUDA、SGRL、BGRL、CCA-SSG；无标签自监督训练。

**Out of Scope:** 异配图主表、LLM 文本增强、监督式对比、端到端 fine-tuning、大规模 OGB 主实验、图分类任务。

**Key Assumptions:** 特征相似度与低 Dirichlet energy 能作为同配可靠性的无标签代理；Planetoid public split 可用于早筛，但不能独立支撑顶会主张。

### Sub-questions

1. 为什么 `specprop` 在 PubMed/Photo 这类高谱集中图上稳定超过 `autopropcat`？
2. 谱集中度阈值和 rank 规则是否真正解释收益，而不是 Planetoid public split 偶然现象？
3. 哪些图统计量可以预测“传播已足够”“需要低秩去噪”或“应回退到 AutoProp”？

### Candidate Questions Considered

| # | Candidate | FINER Avg | Why not selected |
|---|---|---:|---|
| 1 | 当前 RQ：同配可靠性驱动的 GCL 是否超过随机增强和传播强基线？ | 4.2 | Selected。 |
| 2 | 能否设计同时适配同配/异配图的统一 GCL？ | 3.4 | 范围过宽，早期容易被异配机制拖散。 |
| 3 | 能否用 LLM 生成文本增强提升 TAG 上的 GCL？ | 3.2 | 依赖外部模型和文本数据，不符合当前快速可复现目标。 |

## Methodology Blueprint

### Research Paradigm

**Selected:** Positivist / computational empirical study。

**Justification:** RQ 关注一个可度量的算法效果差异，主证据应来自可复现实验。

### Method

**Type:** Quantitative。

**Specific Method:** 控制变量的算法基准实验与消融实验。

**Justification:** 同一数据划分、同一 linear probe、同一 seed 预算下比较方法差异，能直接回答“是否优于 baseline”。

### Data Strategy

**Data Type:** Secondary public benchmark data。

**Sources:** PyG Planetoid：Cora、CiteSeer、PubMed。

**Sampling:** 第一阶段使用 public split `split_index=0` 与 2 个 model/eval seed 做 smoke test；论文级实验需扩展到多 split、多 seed 和更大同配数据。

**Time Frame:** 2026-06-29 启动早筛。

### Analytical Framework

**Technique:** frozen encoder + one-vs-rest logistic regression probe accuracy；mean/std；paired delta；消融比较；后续与 2026 强 baseline 表格对齐。默认 `sklogreg` 参考 GRACE/BGRL，备选 `torchlogreg` 参考 GCA/CCA-SSG。

**Steps:** 先跑 `raw`/`prop`/`grace`/`homogcl` smoke；若 HomoGCL 未超过 `prop` 或 GRACE，记录失败并改路线；若超过，再扩展调参、消融、更大数据集，并补齐 IRGCL/PROPGCL/GOUDA/SGRL 等 2026 SOTA 级 baseline。

**Tools:** Python 3.10、PyTorch、PyTorch Geometric。

### Validity Criteria

| Criterion | Strategy to Ensure |
|---|---|
| Internal validity | 统一 split、linear probe、metric、seed 记录；自监督训练禁用测试标签。 |
| Construct validity | 同时比较随机增强 GCL 与传播基线，避免只打弱 baseline。 |
| External validity | smoke 只覆盖 Planetoid；论文级需要 Amazon/Coauthor/OGB 等同配图扩展。 |
| Reproducibility | 每次 run 保存 JSON metadata、依赖版本、git 状态和完整命令。 |

### Limitations (By Design)

- Planetoid public split 数量有限，早筛结果不能作为最终 SOTA 证据。
- 当前原型未复现所有强 GCL 官方实现，第一阶段只判断 idea 是否值得继续。
- 特征相似度代理可能在文本稀疏或噪声强的图上失效。

### Ethical Considerations

使用公开基准数据；不收集个人数据；报告中必须明确 smoke 与主表差异，避免夸大。

### Reporting Standard

建议按机器学习可复现性清单报告，不适用 PRISMA/CONSORT/STROBE。

### Preregistration

不做正式预注册；仓库文档记录先验失败标准与后续扩展规则。

## Devil's Advocate Checkpoint 1

### Verdict: PASS WITH MAJOR CAUTION

### Critical Issues

No critical issues identified.

### Major Issues

1. **“足以发 2026 顶会顶刊”的目标不能由 smoke test 或弱 baseline 证明**
   - **Type:** Scope / Evidence
   - **Problem:** Cora/CiteSeer/PubMed 上的小幅提升无法支撑 SOTA 论文主张。
   - **Recommendation:** 把第一阶段定位为证伪；只有超过传播强基线后才扩到论文级证据，最终必须纳入 2026 顶会顶刊强 baseline。

2. **与 IRGCL/PROPGCL/GOUDA 等最新方法的边界需要后续严格比较**
   - **Type:** Baseline
   - **Problem:** 当前原型只包含自实现基线，不等同于完整 SOTA。
   - **Recommendation:** 若早筛通过，下一阶段纳入官方/复现代码或严格重实现。

### Strongest Counter-Argument

HomoGCL 可能只是把传播特征和特征相似度重新包装，真正性能来自 `prop`，而不是对比学习本身。

### Stress Test Results

| Test | Result |
|---|---|
| Remove strongest source — does argument hold? | 尚未验证 |
| Flip the research question — is opposing view credible? | Yes |
| Apply to different context — does finding generalize? | 尚未验证 |
| "So what?" — is the significance justified? | 仅在超过传播强基线后成立 |
