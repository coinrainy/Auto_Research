# 2026-06-29 同配图 GCL 文献重扫与新 idea 设计备忘录

## 本轮重置决定

用户要求放弃此前已做路线，重新进行文献搜索与 idea 设计。因此以下路线从本轮起不再作为默认主线继续推进：

- reliability-weighted GCL / projection consistency / embedding stability 系列；
- RAGC、RRNV、TNS、BPRRNV 及其 validation selector / fixed concat / complement gate；
- RPGCL-Auto、HPFS-GCL、`cg_hpfs` 及其 seed sanity 扩展。

这些实验只保留为负结果和经验记录：验证集 selector 容易过拟合，raw+learned 固定拼接不稳定，重复训练会掩盖 representation gate 的真实贡献。新方向必须从文献空隙重新定义，不继承旧模块作为默认答案。

## 搜索策略

- 搜索日期：2026-06-29。
- 主要来源：OpenReview、arXiv、ACM/会议论文页、论文 PDF。
- 时间范围：重点 2024-2026，保留少量基础 baseline 文献。
- 查询主题：
  - graph contrastive learning homophily / positive samples / propagation；
  - graph self-supervised learning node classification 2025/2026；
  - propagation-only / coarsening / MLP-GNN cross-model GCL；
  - masked graph autoencoder / positional autoencoder。
- 纳入标准：节点分类相关、图自监督/图对比学习相关、能启发同配图实验设计。
- 排除标准：纯推荐、纯图分类、只做异质图或动态图且无法迁移到当前 Planetoid 同配图早筛。

## 关键文献图谱

| 方向 | 代表工作 | 对新 idea 的约束 |
| --- | --- | --- |
| 传播本身很强 | PROPGCL / Propagation Alone is Enough 指出训练后的 transformation 可能不如传播本身，contrastive 目标和下游目标错位时会伤害分类。 | 新方法不能只堆更复杂 encoder；必须和传播 teacher / training-free baseline 对齐。 |
| 正样本学习被 message passing 预对齐削弱 | SPGCL 提出 pre-alignment effect，并用 Dirichlet energy 做 feature-wise separated propagation 与 positive sampling。 | 不能再把“能量分解 + positive sampling”包装成全新主贡献；若用能量，必须有节点级或机制级新切入。 |
| 随机增强/一对一正样本正在被替代 | CL-GCL 用 coarsening 与 manifold relation 处理随机增强语义扭曲和一对一采样问题；GraphECL 用 MLP/GNN cross-model positives 提升推理效率。 | 新 idea 应避免随机增强主线，且要说明和 coarsening / MLP-GNN cross-model 的区别。 |
| 同配图邻居可作为 noisy positives | BLNN 指出 BGRL 忽略 homophily，扩展 node-neighbor positive pairs，并学习 neighbor supportiveness。ICML 2023 “Contrastive Learning Meets Homophily” 也将 homophily 和 positive sampling 结合。 | 不能简单说“邻居当正样本”；需要解决 inter-class boundary edges 和核心/边界差异。 |
| 图频谱/低频信息是 GCL 的基础解释 | SpCo 从频谱解释 GCL 主要编码低频信息；GraphPAE 指出传统 masking 主要学低频，提出特征/位置双路径以学习多频信息。 | 同配图方法需要保护低频类别平滑，同时避免边界节点被过度平滑。 |
| 生成式 SSL 是强 baseline | GraphMAE/GraphMAE2/GraphPAE 等 masked autoencoding 已经成为强自监督对照。 | 新方法若只做 contrastive，论文表必须包含 GraphMAE/GraphPAE 或至少 GraphMAE 系列强对照。 |
| 节点可区分性比粗粒度 homophily 更精细 | “When Do GNNs Help with Node Classification?” 强调 intra/inter-class node distinguishability，而不是只看全图 homophily。 | 新方向应从节点级可区分性或核心-边界结构出发，而不是全图 homophily gate。 |

## 文献空隙判断

当前拥挤区：

- 学习/选择 augmentation：PiNGDA、GCA、GraphCL 类已经很多；
- feature energy / propagation separation：SPGCL 已经非常接近；
- coarsening / manifold / community positives：CL-GCL 已经占位；
- MLP-GNN cross-model distillation：GraphECL、SimMLP、Less-is-More 已经占位；
- homophily neighbor positives：BLNN、homophily-meets-CL 已经占位。

仍有机会的窄缝：

1. **同配图不是所有节点都应被同样平滑/对齐**。核心节点适合邻域 positive compactness，边界节点强行邻居对齐会伤害可分性。
2. 现有方法多从 feature dimension、graph coarsening 或模型视角处理可靠性，较少将 **node-level core/boundary distinguishability** 作为自监督目标本身。
3. 旧实验也反复暴露：全图固定拼接、全图 validation selector、全图 gate 都不稳定；需要节点级差异化，而不是 graph-level 二选一。

## 候选 idea 初筛

| 候选 | 核心 | 新颖性 | 可实现性 | 风险 | 裁决 |
| --- | --- | ---: | ---: | --- | --- |
| A. Propagation-teacher MLP distillation | 用传播结果训练快速 MLP/GNN | 低 | 高 | 太接近 GraphECL/PROPGCL/SimMLP | 放弃 |
| B. Coarsened multi-positive GCL | 用社区/粗图替代一对一正样本 | 中低 | 中 | 太接近 CL-GCL，2026 已有强工作 | 放弃 |
| C. Energy-separated positive sampling | 用 Dirichlet energy 选特征和正样本 | 中低 | 高 | SPGCL 已经直接命中 | 放弃 |
| D. Core-Boundary Self-Guided GCL | 无标签识别同配核心与类别边界，核心做邻域 compactness，边界做 residual/position preservation | 中高 | 中 | core/boundary proxy 可能不对应真实错误；需强诊断 | **保留为新主候选** |

## 新主候选：Core-Boundary Self-Guided GCL（暂名 CBS-GCL）

### 一句话 idea

在同配图节点分类中，GCL 不应对所有节点使用同一种正样本策略：同配核心节点应被邻域/扩散正样本拉近，而疑似边界节点应优先保持 raw/residual/position 区分性，避免被邻居 compactness 过度平滑。

### 无标签核心信号

对每个节点计算多深度传播视图：

- `H0 = normalize(X)`
- `H1 = normalize(SX)`
- `H2 = normalize(S^2X)`
- 可选 `R1 = normalize(X - SX)` 作为 residual/high-frequency view。

核心置信度 `core_score(i)` 使用不含标签的组合信号：

- propagation-depth agreement：`cos(H0_i, H1_i)`、`cos(H1_i, H2_i)` 高；
- neighborhood consistency：节点与邻居的低频 signature 相似；
- residual risk：`||X_i - SX_i||` 低；
- degree-normalized uncertainty：避免高 degree 节点天然更稳定。

直觉：

- 高 `core_score`：节点位于同配核心，邻居大概率同类，适合 multi-positive compactness；
- 低 `core_score`：节点可能在类别边界或结构噪声区，强行拉近邻居可能制造错误正样本。

### 训练目标

1. **Core compactness loss**
   - 对高 `core_score` 节点，从一跳邻居、二跳扩散相似节点、低频 signature top-k 中采样 positives；
   - 使用 multi-positive InfoNCE 或 negative-free CCA/Barlow 风格 alignment；
   - positive 权重由 `core_score(anchor) * pair_similarity` 决定。

2. **Boundary residual preservation loss**
   - 对低 `core_score` 节点，不扩展邻居 positives；
   - 预测或保持 residual view `R = X - SX` / positional encoding；
   - 目标是让边界节点保留区分性，而不是被低频平滑吞掉。

3. **Global decorrelation / anti-collapse**
   - 使用 variance-covariance regularization 或 CCA-style decorrelation；
   - 避免不用 negatives 时表示坍塌。

### 和已有工作的区别

- 不同于 BLNN：不是所有邻居都作为 noisy positives，而是先区分核心/边界。
- 不同于 SPGCL：SPGCL 主要 feature-wise energy separation；CBS-GCL 是 node-wise core-boundary curriculum，并显式保护边界 residual。
- 不同于 CL-GCL：不以 coarsening/community 为主，而是节点级核心-边界风险控制。
- 不同于 GraphPAE：不是纯位置/特征重构，而是把 residual preservation 作为边界节点的 anti-oversmoothing SSL 目标。
- 不同于旧 `cg_hpfs`：不做全图 graph-level raw gate，不依赖重复训练造成的偶然差异。

## 最小实验计划

### 阶段 0：诊断先行，不写大模型

目标：验证 `core_score` 是否真的对应下游分类难度和 raw-vs-propagation 的收益差异。

数据：

- 同配优先：Cora、CiteSeer、PubMed；
- 协议：分层 `train:val:test = 1:1:8`；
- 指标：accuracy；
- split/model seed 分开记录。

诊断表：

| 检验 | 通过标准 | 失败则 |
| --- | --- | --- |
| core_score 分桶后 test error 是否单调下降 | high-core bucket error 明显低于 low-core | 放弃 core-boundary proxy |
| low-core 节点 raw/residual 是否比 propagated view 更保留可分性 | low-core 上 raw/residual linear probe 更优或不更差 | 改 proxy 或放弃 boundary residual |
| high-core 邻域 positives 的同标签率是否更高（仅诊断用标签） | high-core positive label agreement 高于 low-core | 放弃 neighbor positive 扩展 |
| core_score 是否只是 degree proxy | 与 degree 相关但不完全等价；degree-controlled 后仍有诊断力 | 加 degree residualization 或放弃 |

### 阶段 1：最小原型

只实现：

- deterministic propagation views；
- core_score；
- core weighted multi-positive loss；
- boundary residual preservation loss；
- CCA-style anti-collapse。

不实现：

- learned augmentation generator；
- validation selector；
- graph-level gate；
- 大规模 OGB；
- 复杂理论包装。

### 阶段 2：早筛表

| Dataset | Raw | PROP | GRACE-light | CCA-SSG/BGRL | GraphMAE/GraphPAE | CBS-GCL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| CiteSeer | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |
| PubMed | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 |

## 停止标准

立即放弃 CBS-GCL，如果满足任一项：

- `core_score` 分桶与下游 test error、positive label agreement 没有稳定关系；
- CBS-GCL 不超过 PROP / raw propagation teacher，说明训练没有必要；
- 提升只来自 boundary residual reconstruction，而 core compactness 无贡献；
- 在 Cora/CiteSeer/PubMed 中至少两个数据集低于 GRACE-light 或 GraphMAE/CCA-SSG 强 baseline；
- 结果依赖 validation selector 或 graph-level threshold 调参。

## 下一步命令建议

阶段 0 诊断已完成，不急于直接包装方法：

| Dataset | Raw Acc | Prop1 Acc | Prop2 Acc | Prop+Residual Acc | core-degree corr | high-low topk agreement | high-low raw error | high-low prop2 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.647505 | 0.820055 | 0.842144 | 0.817560 | -0.021566 | +0.069906 | -0.136846 | -0.039711 |
| CiteSeer | 0.657873 | 0.715145 | 0.724915 | 0.713717 | -0.018767 | +0.003381 | -0.025397 | -0.003324 |
| PubMed | 0.844278 | 0.850238 | 0.855031 | 0.861244 | -0.006070 | +0.024817 | -0.006702 | -0.011509 |

诊断裁决：

- CBS-GCL 未被阶段 0 淘汰：high-core 节点在三图上均有更低 test error，且 `core_score` 与 degree 相关性接近 0。
- Cora/PubMed 的 top-k positive label agreement 支持 core positive expansion；CiteSeer 很弱，是当前最大风险。
- Prop2/PROP 本身非常强，后续训练式方法必须超过 Prop2，否则不应继续。

下一步应实现最小训练原型：

```bash
cd /root/autodl-tmp/Auto_Research
cd experiments/cbs_gcl
python train.py --dataset Cora --split-index 0 --method cbs_gcl --epochs 50
```

已实现的阶段 0 诊断文件：

```bash
experiments/cbs_gcl/analyze_core_boundary.py
```

它应输出：

- `core_boundary_buckets.csv`
- `core_boundary_summary.csv`
- `raw_vs_propagation_by_bucket.csv`

只有诊断通过后，再实现 `train.py --method cbs_gcl`。

## 主要参考来源

- SPGCL / positive-sample pre-alignment：<https://arxiv.org/html/2606.10284v1>
- PROPGCL / propagation alone：<https://openreview.net/forum?id=i4qdY4vQU9>
- CL-GCL / coarsening and manifold positives：<https://openreview.net/forum?id=bl2EkyUDEn>
- GraphECL / MLP-GNN cross-model contrast：<https://openreview.net/forum?id=3yyGlNHnlj>
- BGRL：<https://openreview.net/forum?id=0UXT6PpRpW>
- CCA-SSG：<https://proceedings.neurips.cc/paper_files/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf>
- BLNN / homophily neighbor positives：<https://arxiv.org/html/2408.05087v1>
- PiNGDA / beneficial augmentation noise：<https://openreview.net/forum?id=iilSR3cKTx>
- GraphPAE：<https://arxiv.org/abs/2505.23345>
- GraphMAE：<https://arxiv.org/abs/2205.10803>
- Contrastive Learning Meets Homophily：<https://proceedings.mlr.press/v202/he23c/he23c.pdf>
- POT / node compactness imbalance：<https://openreview.net/forum?id=Xasl21tSOf>
- SpCo / graph spectrum view of GCL：<https://openreview.net/forum?id=L0U7TUWRt_X>
- Node distinguishability and homophily：<https://cs.stanford.edu/people/jure/pubs/when-neurips23.pdf>
