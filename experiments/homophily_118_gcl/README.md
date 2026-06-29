# Homophily 1:1:8 GCL 工作区

本目录是一次新的实验重置，不沿用 `experiments/topvenue_gcl` 的代码框架。

## 目标

- 优先验证同配图节点分类：Cora、CiteSeer、PubMed。
- 默认协议为分层 `train:val:test = 1:1:8`，即每个类别约 10%/10%/80%。
- 先跑轻量 baseline，再验证新候选 RPGCL-Auto。

## 当前候选

当前状态：**RPGCL-Auto selector 已降级**。它在 10 split 上优于 GRACE-light，但 selector-control 表明它没有超过 best fixed representation control，因此不再作为主方法继续扩大。

下一步 active subdirection：**Complement-gated raw-preserved GCL**。核心问题从“用 validation accuracy 选全图表示”收缩为“无标签判断何时应保留 raw feature separability，何时应引入 graph contrastive complement”。

此前候选 **HPFS-GCL**：Homophily-Preserving positive expansion + False-negative Suppression，已保留为 RPGCL-Auto 的训练分支，但不是当前单独主方法。

核心假设：在同配图中，raw feature 与低阶传播特征的相似节点更可能共享类别。训练时：

- 将 propagation signature 的 top-k 相似节点作为额外正样本；
- 对 sampled negatives 中与 anchor signature 高相似的节点降低 denominator 权重；
- 保留 self positive 的 GRACE/InfoNCE 主目标，避免完全依赖启发式正样本。

当前主指标统一为 **accuracy**。F1Mi/F1Ma 只作为附属记录。代码中的 `grace` 是本工作区的 `GRACE-light`：轻量 sampled GRACE-style baseline，用于同协议公平早筛，不等同于论文官方调参 GRACE。

RPGCL-Auto 在 Cora/CiteSeer/PubMed × splits 0-9 × model seed0 × 50 epoch 的 1:1:8 协议下，相对 `GRACE-light` 的 accuracy 结果：

| Dataset | GRACE-light Acc | RPGCL-Auto Acc | mean ΔAcc | positive/negative/zero |
| --- | ---: | ---: | ---: | ---: |
| Cora | 0.792699 | 0.799168 | +0.006470 | 7 / 3 / 0 |
| CiteSeer | 0.699624 | 0.715333 | +0.015708 | 10 / 0 / 0 |
| PubMed | 0.833412 | 0.851176 | +0.017765 | 10 / 0 / 0 |

但 selector-control gate 显示 Auto 不超过 best fixed control：

| Dataset | HPFS Acc | Raw+HPFS Acc | RPGCL-Auto Acc | Best fixed Acc | Auto - Best fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.799445 | 0.779344 | 0.800139 | 0.800185 | -0.000046 |
| CiteSeer | 0.711650 | 0.718828 | 0.715295 | 0.720669 | -0.005374 |
| PubMed | 0.834065 | 0.851176 | 0.851113 | 0.851176 | -0.000063 |

当前解释：

- Cora 主要适合 HPFS；Raw+HPFS 会伤害 accuracy。
- CiteSeer/PubMed 主要适合 Raw+HPFS；固定融合已经解释了 Auto 的收益。
- 因此后续不继续调 validation selector，而是转向 complement gate。

下一步必须通过的早筛：

- complement gate 在 Cora 上接近 HPFS，同时在 CiteSeer/PubMed 上接近 Raw+HPFS；
- 不能只靠 validation label selection；
- 对齐官方/强调参 GRACE 与更多强 baseline，避免把 GRACE-light 的低数值误当作论文级结论。

## 快速运行

```bash
cd /root/autodl-tmp/Auto_Research/experiments/homophily_118_gcl
bash scripts/run_smoke.sh
```

正式一点的早筛：

```bash
DATASETS="Cora CiteSeer PubMed" METHODS="raw_features grace rpgcl_auto" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/rpgcl_auto_homophily_splits0-9_e50" OVERWRITE=1 bash scripts/run_homophily_118_study.sh
```

shuffle control：

```bash
DATASETS="Cora CiteSeer PubMed" METHODS="hpfs_gcl" EXTRA_ARGS="--shuffle-positives" RUN_TAG=shuffle_pos bash scripts/run_homophily_118_study.sh
```
