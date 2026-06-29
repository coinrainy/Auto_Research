# Homophily 1:1:8 GCL 工作区

本目录是一次新的实验重置，不沿用 `experiments/topvenue_gcl` 的代码框架。

## 目标

- 优先验证同配图节点分类：Cora、CiteSeer、PubMed。
- 默认协议为分层 `train:val:test = 1:1:8`，即每个类别约 10%/10%/80%。
- 先跑轻量 baseline，再验证新候选 HPFS-GCL。

## 当前候选

当前 active candidate：**RPGCL-Auto**，即 Raw-Preserved Graph Contrastive Learning with validation-gated representation selection。

此前候选 **HPFS-GCL**：Homophily-Preserving positive expansion + False-negative Suppression，已保留为 RPGCL-Auto 的训练分支，但不是当前单独主方法。

核心假设：在同配图中，raw feature 与低阶传播特征的相似节点更可能共享类别。训练时：

- 将 propagation signature 的 top-k 相似节点作为额外正样本；
- 对 sampled negatives 中与 anchor signature 高相似的节点降低 denominator 权重；
- 保留 self positive 的 GRACE/InfoNCE 主目标，避免完全依赖启发式正样本。

当前主指标统一为 **accuracy**。F1Mi/F1Ma 只作为附属记录。

RPGCL-Auto 在 Cora/CiteSeer/PubMed × splits 0/1/2 × seed0 × 50 epoch 的 1:1:8 协议下，相对纯 `grace` 的 mean Δaccuracy 分别为：

- Cora：+0.002311，2/3 split 为正；
- CiteSeer：+0.013905，3/3 split 为正；
- PubMed：+0.016653，3/3 split 为正；
- Overall：+0.010956，8/9 split 为正。

下一步必须通过的早筛：

- splits 0-9 上相对 `grace` 的 accuracy 大部分为正；
- selector control 证明 validation-gated selection 优于固定 HPFS 或固定 raw-preserved；
- 不能只靠 raw feature 或单数据集偶然提升。

## 快速运行

```bash
cd /root/autodl-tmp/Auto_Research/experiments/homophily_118_gcl
bash scripts/run_smoke.sh
```

正式一点的早筛：

```bash
DATASETS="Cora CiteSeer PubMed" METHODS="raw_features grace hpfs_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=100 bash scripts/run_homophily_118_study.sh
```

shuffle control：

```bash
DATASETS="Cora CiteSeer PubMed" METHODS="hpfs_gcl" EXTRA_ARGS="--shuffle-positives" RUN_TAG=shuffle_pos bash scripts/run_homophily_118_study.sh
```
