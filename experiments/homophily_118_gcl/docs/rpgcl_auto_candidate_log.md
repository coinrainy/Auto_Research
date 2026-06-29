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

RPGCL-Auto 继续作为 **active candidate**，但还不能称为 SOTA idea。

理由：

- 在 1:1:8 同配协议下，三个同配图 10 split 均值均高于 GRACE-light；
- PubMed 与 CiteSeer 的 accuracy 增益较清楚，且 10/10 split 均为正；
- Cora 仍是小正，存在 3/10 split 轻微负，说明方法在 Cora 上还不够强；
- HPFS 训练目标本身不是稳定主贡献，raw-preserved validation-gated representation selection 才是当前最强信号；
- PubMed 上 10/10 split 均选择 Raw+HPFS，支持 raw feature separability 与 learned graph context 的互补假设。

## 下一步停止/推进标准

推进：

- 加 `rpgcl_auto` 的 selector control：always-HPFS、always-raw-preserved、raw-only、oracle upper bound；
- 加 `--selector-margin`，避免 validation 差距太小时过度切换；
- 对齐官方/强调参 GRACE，当前 `GRACE-light` 不能用于论文级 claim；
- 对比 CCA-SSG/BGRL/GraphECL 或至少补 CCA-style baseline。

停止：

- 若 validation selector 不优于 fixed raw-preserved 或 fixed HPFS；
- 若 raw-only 与 raw-preserved 的差距解释不了，说明贡献只是线性探针选择技巧。
- 若补强 baseline 后 PubMed/CiteSeer 增益消失，或 selector control 证明固定 Raw+HPFS 足够，则需要降级为 representation-selection 诊断而不是完整方法。
