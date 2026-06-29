# CBS-GCL 实验协作记录

## 固定规则

- 本目录是 2026-06-29 文献重扫后的新实验方向，不引用旧 `topvenue_gcl`、`homophily_118_gcl` 或其他历史实验目录代码。
- 回答、实验记录和文档尽量使用中文。
- 主任务仍为图对比学习 / 图自监督节点分类，目标是寻找可冲击 2026 顶会/顶刊的候选 idea。
- 评估主指标为 accuracy；split seed 与 model seed 必须分开记录。

## 当前候选

**CBS-GCL: Core-Boundary Self-Guided Graph Contrastive Learning**

核心假设：同配图或中高同配图中，节点可分为较稳定的同配核心节点与疑似类别边界节点。核心节点适合邻域/扩散正样本 compactness；边界节点不应被强制邻居对齐，而应保留 raw/residual/position 区分性。

## 当前阶段

- 阶段 0：只实现无标签 core/boundary 诊断，不训练新模型。
- 通过条件：`core_score` 分桶应与 test error、positive label agreement、raw-vs-propagation 表现差异存在稳定关系。
- 失败条件：`core_score` 只是 degree proxy，或与下游错误/正样本质量无稳定关系，则放弃 CBS-GCL。
