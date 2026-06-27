# 图对比学习实验协议确认清单

日期：2026-06-27

用途：在实现新的 Graph Contrastive Learning / Graph Self-Supervised Learning 方法前，先确认实验协议，避免因代码基底、数据划分、评估流程或调参策略不一致造成不可比结果。

## 1. 代码基底与复现策略

- 优先确认是否基于官方作者代码、PyGCL 等公开框架，还是自建最小实验框架。
- 若基于官方代码，需要记录 commit、依赖版本、原始命令、原始配置文件和本项目改动点。
- 若基于 PyGCL，需要记录使用了哪些模块：augmentation、contrasting mode、contrastive objective、negative mining。
- 若自实现 baseline，必须先做 sanity reproduction：至少复现 1-2 个公开表格或官方 README 中的代表性结果。
- 新方法核心模块可以自实现，但 baseline、数据加载、split 和 evaluation 应尽量复用公开实现或统一框架。

参考依据：
- PyGCL 明确将 GCL 拆成 augmentation、contrasting mode、objective、negative mining，并提供 standardized evaluation / experiment management：https://github.com/PyGCL/PyGCL
- GRACE 官方代码给出原始依赖和训练入口，说明复现时版本差异可能影响结果：https://github.com/CRIPAC-DIG/GRACE
- ProGCL 官方代码记录了其 PyTorch / PyG 版本和训练命令，应作为 false-negative 相关 baseline 的优先参考：https://github.com/junxia97/ProGCL

## 2. 数据集与 split 协议

- Planetoid 数据集必须明确使用 `public`、`full`、`geom-gcn` 还是 `random` split。
- 如果使用 `random` split，需要固定生成规则：每类训练样本数、验证集数量、测试集数量、split seed。
- 对 heterophily 数据集，优先使用公开固定 split；WebKB / WikipediaNetwork / Actor 等应记录 mask 形状和 split index。
- 同一论文主表中不要混用不可比 split；如确实混用，需要分表或明确说明。
- 对小数据集必须报告 split-level 波动，不只报告单次 split。

参考依据：
- PyG Planetoid 文档列出 `public`、`full`、`geom-gcn`、`random` 多种 split，且默认行为不同：https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.Planetoid.html
- PyG WebKB 文档说明 Cornell/Texas/Wisconsin 来自 Geom-GCN 数据集：https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.WebKB.html
- PyG WikipediaNetwork 文档说明 `geom_gcn_preprocess=True` 时会提供多 split mask：https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.datasets.WikipediaNetwork.html
- Shchur et al. 指出单一固定 split 和不一致训练流程会导致 GNN 比较失真：https://arxiv.org/pdf/1811.05868

## 3. Evaluation protocol

- 先确认任务：node classification、node clustering、link prediction，还是 transfer / robustness。
- 对 node classification，必须明确是 frozen encoder + logistic regression / linear classifier，还是 fine-tuning。
- 如果是 linear evaluation，需要记录：
  - encoder 是否完全冻结；
  - classifier 类型；
  - 是否使用 L2 regularization；
  - `C` / weight decay / learning rate 的选择方式；
  - classifier 训练 epoch、early stopping、validation criterion。
- 对 transductive SSL，需要明确无监督预训练是否使用全图结构和全图特征；下游标签只在 probe 阶段使用。
- 如果论文主张“通用表征”，不能只看同一数据集上的 node classification；至少在附录说明任务覆盖边界。

参考依据：
- GRACE 页面说明采用 unsupervised training 后用 L2-regularized logistic regression 做 linear evaluation：https://sxkdz.github.io/research/GraphCL/
- BGRL 论文说明先无监督训练 encoder，再对 frozen embedding 训练带 L2 regularization 的 logistic regression，且梯度不回传 encoder：https://misovalko.github.io/publications/thakoor2022bootstrapped.pdf
- BGRL 官方代码 README 提到 evaluation 周期、logistic regression probe，以及随机 split 会导致结果差异：https://github.com/nerdslab/bgrl
- “Overcoming Pitfalls in GCL Evaluation” 指出现有 GCL 往往只在同一数据集 node classification 上评估，且 pretraining 超参可能借助 downstream validation set，被认为有 protocol 风险：https://arxiv.org/html/2402.15680v1

## 4. Baseline 与调参公平性

- 每个 baseline 需要确认来源：官方代码、PyGCL 实现、第三方复现、还是本项目自实现。
- baseline 和新方法应使用同等调参预算，至少要记录哪些超参调过、候选集合是什么。
- 如果新方法只在 baseline 默认配置上插入模块，需要确认 baseline 默认配置是否适合当前数据集。
- augmentation rate、temperature、hidden dimension、encoder depth、dropout、learning rate、weight decay、warmup、batch size / full-batch 都需要固定或公平搜索。
- 如果 GPU 限制导致 baseline 无法按原设置运行，需要记录替代设置和潜在偏差。

参考依据：
- Shchur et al. 指出不同 early stopping、训练流程和调参方式会让复杂模型与简单模型比较不公平：https://arxiv.org/pdf/1811.05868
- OpenGSL 论文指出 graph learning 方法的进展会被不一致的数据处理、split 策略和实验协议干扰：https://arxiv.org/abs/2306.10280

## 5. GCL 组件级控制

- augmentation：记录 edge drop、feature masking、node drop、diffusion 等具体策略、概率和是否 adaptive。
- positive construction：记录正样本定义，是同一节点跨视图、邻域级、cluster/prototype，还是 teacher/pseudo-label。
- negative set：记录 denominator 来源，是全图、mini-batch、memory bank / queue、intra-view、inter-view，还是 hard-mined negatives。
- negative mining：如果研究 false negative contamination，需要至少包含：
  - 普通 InfoNCE / GRACE-style baseline；
  - debiased or false-negative-aware baseline；
  - hard-negative baseline；
  - random/shuffled control 或污染度诊断。
- loss 复杂度：记录是否 O(N²)、是否使用采样近似、是否受显存限制。

参考依据：
- PyGCL 的组件划分覆盖 augmentation、contrasting mode、objective、negative mining，可作为实验变量控制框架：https://github.com/PyGCL/PyGCL
- ProGCL 指出 GCL 中按相似度选出的 hard negatives 很多可能是 false negatives，并将 false negative 问题和 message passing 联系起来：https://proceedings.mlr.press/v162/xia22b/xia22b.pdf
- CCA-SSG 对比了 DGI/MVGRL/GRACE/GCA/BGRL 等方法是否需要 negatives、projector、asymmetric structure，以及空间复杂度：https://proceedings.neurips.cc/paper_files/paper/2021/file/00ac8ed3b4327bdd4ebbebcb2ba10a00-Paper.pdf

## 6. 随机性、统计汇总与显著性

- 至少区分两类随机性：model seed 和 split seed。
- 小型 heterophily 数据集建议报告 split × seed，而不是只报告 seed。
- 主表报告 mean ± std；同时保留 paired delta，避免只比较独立均值。
- 对核心 claim，建议报告 win/tie/loss counts、paired t-test 或 bootstrap confidence interval。
- 所有 run 必须保存 metadata：git commit、dirty status、依赖版本、数据集统计、split index、seed、完整命令。

## 7. False negative contamination 专项诊断

- 在有标签评估阶段，可以用标签离线统计 false-negative mass，但不能让训练使用测试标签。
- 至少诊断以下对象：
  - high-similarity negatives 中同类节点比例；
  - denominator 中同类节点比例；
  - hard-negative mining 前后 false negative 占比；
  - 不同 homophily / heterophily 数据集上的污染差异；
  - 训练早期、中期、后期污染度是否变化。
- 如果方法声称降低 false negative contamination，需要设置随机化或置换对照，证明收益不是一般 reweighting / regularization。
- 如果只提升 accuracy 而污染度诊断不支持机制，则不能把机制 claim 写强。

参考依据：
- ProGCL 把 hard negatives 中的 false negatives 作为核心问题，并展示了图领域与视觉领域 negative similarity 分布差异：https://proceedings.mlr.press/v162/xia22b/xia22b.pdf

## 8. 实现前必须向用户确认的问题

1. 本轮实验基于哪个代码基底：官方代码、PyGCL、还是自建？
2. baseline 清单是什么？每个 baseline 的代码来源是什么？
3. 使用哪些数据集？每个数据集使用哪个 split 协议？
4. 使用 linear evaluation 还是 fine-tuning？probe 的训练配置是什么？
5. 使用多少 split、多少 model seed？主表如何汇总？
6. 是否调参？调参预算如何保证 baseline 与新方法公平？
7. false negative contamination 的诊断指标是什么？是否只离线使用标签？
8. 哪些结果是 smoke test，哪些结果可以进入论文主表？
9. 显存/时间不足时，哪些实验可以降级，哪些不能降级？

## 当前建议

后续新 false-negative GCL 方法不应先写代码。推荐先做三件事：

1. 选定代码基底：优先评估 PyGCL + 官方 ProGCL/GRACE 复现的可行性。
2. 选定 split 协议：Planetoid 使用 `public` 或 `geom-gcn` 需明确；heterophily 数据集优先固定 10 splits。
3. 先实现诊断：在不改模型的情况下，统计 GRACE/GCA/ProGCL-style negative set 的 false-negative contamination，确认痛点在目标数据集上真实存在。
