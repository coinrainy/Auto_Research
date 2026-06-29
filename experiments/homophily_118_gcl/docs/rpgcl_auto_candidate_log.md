# RPGCL-Auto 候选方法记录

日期：2026-06-29

## 重置边界

用户要求放弃已有代码框架，重新进行实验，并确保同配图协议大部分采用 `1:1:8`。因此本候选在 `experiments/homophily_118_gcl/` 中从零实现，不沿用 `experiments/topvenue_gcl` 的训练框架。

后续主指标统一为 **accuracy**。F1Mi/F1Ma 只作为附属记录，不作为主要裁决依据。

## 协议

- 数据集：Cora、CiteSeer、PubMed。
- split：按类别分层构造 `train:val:test ~= 0.1:0.1:0.8`。
- 当前早筛：splits 0-9、seed 0、50 epoch。
- baseline：`raw_features` 与 `GRACE-light`。代码中方法名仍为 `grace`，但该实现是本工作区的轻量 sampled GRACE-style baseline，不等同于论文官方调参 GRACE。
- 输出目录：`runs/rpgcl_auto_homophily_splits0-9_e50/`。

实际 split 比例示例：

- Cora：272 / 272 / 2164，比例约 0.1004 / 0.1004 / 0.7991。
- CiteSeer：333 / 333 / 2661，比例约 0.1001 / 0.1001 / 0.7998。
- PubMed：1972 / 1972 / 15773，比例约 0.1000 / 0.1000 / 0.8000。

## 方法

**RPGCL-Auto**：Raw-Preserved Graph Contrastive Learning with Validation-Gated Representation Selection。

训练阶段：

- 用 GRACE-style sampled self-positive contrastive learning 作为基础；
- HPFS 分支加入 propagation-signature semantic positives；
- 对 signature 相似的 sampled negatives 做 soft denominator suppression。

评估阶段：

- 在同一个 encoder 输出上构造三种候选表示：
  1. HPFS/GCL embedding；
  2. raw features；
  3. L2-normalized raw features + L2-normalized GCL embedding；
- 用 validation accuracy 选择最终表示；
- test accuracy 只在选择后报告。

当前理解：同配图上的关键不只是训练 loss，而是避免 SSL embedding 覆盖 raw feature 的可分性；raw-preserved 表示在 PubMed/CiteSeer 上很强，但在 Cora 上会退化，因此需要 validation gate。

## 早筛结果

`rpgcl_auto` vs `GRACE-light`，主指标 accuracy：

| Dataset | splits | GRACE-light Acc | RPGCL-Auto Acc | mean ΔAcc | positive/negative/zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 10 | 0.792699 | 0.799168 | +0.006470 | 7 / 3 / 0 |
| CiteSeer | 10 | 0.699624 | 0.715333 | +0.015708 | 10 / 0 / 0 |
| PubMed | 10 | 0.833412 | 0.851176 | +0.017765 | 10 / 0 / 0 |

Selector 选择：

| Dataset | HPFS | Raw+HPFS | Raw | Oracle gap |
| --- | ---: | ---: | ---: | ---: |
| Cora | 9 | 1 | 0 | 0.000462 |
| CiteSeer | 4 | 6 | 0 | 0.005487 |
| PubMed | 0 | 10 | 0 | 0.000000 |

注意：`GRACE-light` 的数值低于 GRACE 论文报告值，主要因为这里采用用户指定的 `1:1:8` 协议、50 epoch、sampled contrastive loss 与统一默认超参数；它只能作为本地公平早筛 baseline，不能直接代表官方 GRACE。

## 当前裁决

RPGCL-Auto selector 降级为 **failed selector branch**，不再作为主方法继续扩大。

理由：

- 在 1:1:8 同配协议下，三个同配图 10 split 均值均高于 GRACE-light；
- PubMed 与 CiteSeer 的 accuracy 增益较清楚，且 10/10 split 均为正；
- Cora 仍是小正，存在 3/10 split 轻微负，说明方法在 Cora 上还不够强；
- HPFS 训练目标本身不是稳定主贡献，raw-preserved validation-gated representation selection 才是当前最强信号；
- PubMed 上 10/10 split 均选择 Raw+HPFS，支持 raw feature separability 与 learned graph context 的互补假设。

## Selector-control gate

新增 selector-control 实验：Cora/CiteSeer/PubMed × splits 0-9 × model seed 0 × 50 epoch，比较 `raw_features`、`GRACE-light`、固定 `HPFS`、固定 `Raw+HPFS` 与 `RPGCL-Auto`。

| Dataset | HPFS Acc | Raw+HPFS Acc | RPGCL-Auto Acc | Best fixed Acc | Auto - Best fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.799445 | 0.779344 | 0.800139 | 0.800185 | -0.000046 |
| CiteSeer | 0.711650 | 0.718828 | 0.715295 | 0.720669 | -0.005374 |
| PubMed | 0.834065 | 0.851176 | 0.851113 | 0.851176 | -0.000063 |

| Dataset | Auto positive vs best fixed | Auto negative vs best fixed | Auto zero vs best fixed | Auto choices |
| --- | ---: | ---: | ---: | --- |
| Cora | 3 | 4 | 3 | HPFS 9 / Raw+HPFS 1 / Raw 0 |
| CiteSeer | 0 | 4 | 6 | HPFS 4 / Raw+HPFS 6 / Raw 0 |
| PubMed | 4 | 6 | 0 | HPFS 0 / Raw+HPFS 10 / Raw 0 |

结论：

- Auto selector 没有超过 best fixed control，尤其 CiteSeer 明显低于固定 Raw+HPFS；
- Cora 的收益主要来自 HPFS，Raw+HPFS 明显伤害；
- PubMed 的收益几乎完全来自固定 Raw+HPFS，HPFS-only 接近 GRACE-light；
- 因此主问题应从 “validation-gated representation selection” 收缩为 “何时保留 raw feature separability，何时引入 graph contrastive complement”。

后续 active subdirection：**Complement-gated raw-preserved GCL**。核心不再是用 validation accuracy 选全图表示，而是设计无标签、协议一致的 gate，避免 Cora 式 Raw+HPFS 伤害，同时保留 CiteSeer/PubMed 式 Raw+HPFS 互补增益。

## Complement-gated HPFS 初筛

新增方法：`cg_hpfs`。

定义：

- 训练仍使用 HPFS-GCL；
- 表示阶段使用无标签 `edge_feature_cos_lift` 决定是否拼接 raw feature branch；
- `edge_feature_cos_lift = edge feature cosine mean - deterministic random-pair feature cosine mean`；
- 默认 hard gate：`edge_feature_cos_lift >= 0.13` 时使用 Raw+HPFS，否则使用 HPFS-only；
- gate 不读取标签、validation accuracy、test accuracy 或 split mask。

无标签信号与 raw branch 事后增益：

| Dataset | Raw+HPFS - HPFS | edge feature cosine lift | gate alpha |
| --- | ---: | ---: | ---: |
| Cora | -0.020102 | 0.111855 | 0 |
| CiteSeer | +0.007178 | 0.145687 | 1 |
| PubMed | +0.017112 | 0.200030 | 1 |

`cg_hpfs` 早筛：Cora/CiteSeer/PubMed × splits 0-9 × model seed 0 × 50 epoch。

| Dataset | GRACE-light Acc | HPFS Acc | Raw+HPFS Acc | CG-HPFS Acc | Best fixed Acc | CG - Best fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.792560 | 0.799445 | 0.779344 | 0.801710 | 0.800185 | +0.001525 |
| CiteSeer | 0.699324 | 0.711650 | 0.718828 | 0.717813 | 0.720669 | -0.002856 |
| PubMed | 0.833095 | 0.834065 | 0.851176 | 0.851138 | 0.851176 | -0.000038 |

当前裁决：

- `cg_hpfs` 比 `rpgcl_auto` selector 更符合无标签协议，暂时保留；
- 它能避开 Cora 的 Raw+HPFS 伤害，并保留 PubMed 的 Raw+HPFS 收益；
- CiteSeer 上低于 best fixed 约 0.29 个百分点，因此尚不能升级为主方法；
- 阈值只在三张 Planetoid 同配图上早筛，论文级主张必须补多 model seed、阈值敏感性、更多同配/中同配图和强 baseline。

## 下一步停止/推进标准

推进：

- 放弃当前 validation selector 作为主方法，不继续调 `selector-margin`；
- 设计无标签 complement gate，用结构/特征/embedding complementarity 判断 raw 与 graph branch 的融合强度；
- 对齐官方/强调参 GRACE，当前 `GRACE-light` 不能用于论文级 claim；
- 对比 CCA-SSG/BGRL/GraphECL 或至少补 CCA-style baseline。

停止：

- 若 complement gate 不能在 Cora 上接近 HPFS，同时在 CiteSeer/PubMed 上接近 Raw+HPFS，则放弃该方法路线；
- 若补强 baseline 后 PubMed/CiteSeer 的 Raw+HPFS 增益消失，则降级为 representation-selection 诊断而不是完整方法。
