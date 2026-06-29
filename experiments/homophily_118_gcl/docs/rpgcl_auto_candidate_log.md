# RPGCL-Auto 候选方法记录

日期：2026-06-29

## 重置边界

用户要求放弃已有代码框架，重新进行实验，并确保同配图协议大部分采用 `1:1:8`。因此本候选在 `experiments/homophily_118_gcl/` 中从零实现，不沿用 `experiments/topvenue_gcl` 的训练框架。

后续主指标统一为 **accuracy**。F1Mi/F1Ma 只作为附属记录，不作为主要裁决依据。

## 协议

- 数据集：Cora、CiteSeer、PubMed。
- split：按类别分层构造 `train:val:test ~= 0.1:0.1:0.8`。
- 当前早筛：splits 0/1/2、seed 0、50 epoch。
- baseline：`raw_features` 与纯 `grace`。
- 输出目录：`runs/rpgcl_auto_homophily_splits0-2_e50/`。

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

`rpgcl_auto` vs `grace`，主指标 accuracy：

| Dataset | splits | mean ΔAcc | positive/negative |
| --- | ---: | ---: | ---: |
| Cora | 3 | +0.002311 | 2 / 1 |
| CiteSeer | 3 | +0.013905 | 3 / 0 |
| PubMed | 3 | +0.016653 | 3 / 0 |
| Overall | 9 | +0.010956 | 8 / 1 |

## 当前裁决

RPGCL-Auto 升级为 **active candidate**，但还不能称为 SOTA idea。

理由：

- 在 1:1:8 同配协议下，三个同配图均值均高于纯 GRACE；
- PubMed 与 CiteSeer 的 accuracy 增益较清楚；
- Cora 只有小正且有 1/3 split 轻微负，说明 selector 仍需更稳；
- HPFS 训练目标本身不是稳定主贡献，raw-preserved validation-gated representation selection 才是当前最强信号。

## 下一步停止/推进标准

推进：

- 扩展到 splits 0-9；
- 加 `rpgcl_auto` 的 selector control：always-HPFS、always-raw-preserved、raw-only、oracle upper bound；
- 加 `--selector-margin`，避免 validation 差距太小时过度切换；
- 对比 CCA-SSG/BGRL/GraphECL 或至少补 CCA-style baseline。

停止：

- 若 splits 0-9 上 Cora 平均 accuracy 负，或 overall 平均 ΔAcc 低于 +0.005；
- 若 validation selector 不优于 fixed raw-preserved 或 fixed HPFS；
- 若 raw-only 与 raw-preserved 的差距解释不了，说明贡献只是线性探针选择技巧。
