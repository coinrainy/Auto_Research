# 同配图 GCL / 传播证伪原型

本工作区用于快速证伪和孵化“优先面向同配图”的图对比学习研究想法。当前结论很明确：最初的 `homogcl` 候选没有通过 smoke test，不应继续包装为主方法；`tierspecprop` 已验证为强谱核心分支，但在 PubMed 本地基线面板中无法逐 split 压制 `ccacat`。当前新候选是 `tierccacat`，即在窄谱过渡区把 TierSpecProp 的无标签谱核心与 CCA-GCN 去相关残差拼接，在宽谱核心或回退区保持纯 TierSpecProp。

## 当前方法入口

- `raw`：原始节点特征 + frozen linear probe。
- `prop`：GCN 式训练免费特征传播。
- `propcat`：多阶传播银行 `[X, SX, ..., S^KX]`。
- `autopropcat`：用无标签传播残差平台期自动选择 `K` 的传播银行；当前作为所有 GCL 候选必须击败的强证伪器。
- `specprop`：AutoProp + 安全谱集中度门控。只有传播银行 top-10 PCA 能量占比达到 0.34 时才压缩到 rank=32，否则回退到 AutoProp。
- `corespecprop`：AutoProp + 安全谱集中度门控 + 参与秩自适应核心压缩。top-10 PCA 能量占比低于 0.34 时回退到 AutoProp；触发压缩时按参与秩选择核心 rank，并裁剪到 16-32。
- `tierspecprop`：强谱核心分支。top-10 PCA 能量低于 0.34 回退；0.34-0.36 选择 rank=16；不低于 0.36 选择 rank=32。
- `tierccacat`：当前待验证融合候选。只在 TierSpecProp 选择窄 rank=16 时拼接 L2-normalized 谱核心与 CCA-GCN 去相关残差；回退或宽 rank=32 时不融合，用来避免 Photo/WikiCS 这类强谱核心图被残差分支损伤。
- `grace` / `gracecat`：随机增强 InfoNCE 及其传播拼接诊断。
- `homogcl`：失败候选；同配保真增强 + 多正样本 InfoNCE。
- `horp`：HoRP 教师表示；节点级传播残差门控 + 传播轨迹/残差拼接。
- `horpgcl`：失败候选；HoRP 教师排序 + 多正样本对比 + 相对排序 margin loss。
- `propcca` / `propccat` / `ccassg` / `ccacat`：CCA/去相关类诊断方法；当前未超过传播证伪器。

## 已验证的早筛结论

- `homogcl` 在 Cora/CiteSeer/PubMed smoke test 中未超过 `propcat`，尤其 CiteSeer 明显落后，判定为失败主线。
- `horpgcl` 在 Cora 快速测试中 test accuracy 为 0.796，未超过传播基线，暂不作为主线。
- `propccat` 在 Cora `K=10` 快速测试中 test accuracy 为 0.821，未超过 `propcat/autopropcat`。
- `autopropcat` 使用无标签残差平台期选择传播深度，在单 seed public split 上得到：
  - Cora：0.831，selected `K=6`
  - CiteSeer：0.726，selected `K=6`
  - PubMed：0.789，selected `K=7`
- `specprop` 在 full C-grid 单 seed public split 上相对 `autopropcat` full-grid 取得：
  - Cora：0.834 vs 0.824，selected `K=6`、PCA rank=647
  - CiteSeer：0.723 vs 0.723，selected `K=6`、不压缩
  - PubMed：0.798 vs 0.793，selected `K=7`、PCA rank=32
- `specprop` safe gate 在 class-balanced random split seeds 0/1/2 上相对 `autopropcat` 的 paired delta：
  - Cora：平均 +0.000，回退持平
  - CiteSeer：平均 +0.000，完全回退持平
  - PubMed：平均 +0.018，3 胜 0 负，谱压缩信号最强
- `specprop` safe gate 在 Amazon class-random seeds 0/1/2 上取得：
  - Photo：平均 0.9071 vs AutoProp 0.8745，delta +0.0326，3 胜 0 负，rank=32。
  - Computers：平均 0.7965 vs AutoProp 0.7965，完全回退持平，修复了低阈值版本的压缩损伤。
- `corespecprop` 当前优于固定 rank 的 `specprop`，主要证据为：
  - PubMed class-random seeds 0-9：0.7739 vs AutoProp 0.7541，delta +0.0198，10 胜 0 负。
  - Photo class-random seeds 0-9：0.9002 vs AutoProp 0.8817，delta +0.0185，10 胜 0 负。
  - WikiCS 官方 20 split：0.7702 vs AutoProp 0.7636，delta +0.0066，18 胜 2 负。
- `tierspecprop` 修正了 `corespecprop` 在 WikiCS 上过度压缩的问题，当前关键结果：
  - PubMed class-random seeds 0-9：0.7739 vs AutoProp 0.7541，delta +0.0198，10 胜 0 负，rank=16。
  - Photo class-random seeds 0-9：0.9035 vs AutoProp 0.8817，delta +0.0218，10 胜 0 负，rank=32。
  - WikiCS 官方 20 split：0.7833 vs AutoProp 0.7636，delta +0.0197，20 胜 0 负，rank=32。
- 这些结果只能作为早筛，不足以支撑 SOTA 或顶会投稿结论。
- `scripts/run_local_baseline_key_multisplit.sh` 和 `scripts/run_local_baseline_wikics_multisplit.sh` 提供非 Coauthor 的本地基线面板，用于把 `tierccacat` / `tierspecprop` 与仓库内已有的 `propccat`、`ccacat`、`gracecat` 诊断实现做同 split 对比；这些不是官方强 baseline 的替代品，只用于下一轮筛查。
- PubMed 本地基线面板发现：TierSpecProp 对 `propccat` / `gracecat` 仍为 9 胜 1 负，但对 `ccacat` 只有均值 +0.0025 且 4 胜 6 负。因此不能继续把纯 TierSpecProp 当最终主方法，当前应优先验证 `tierccacat`。`tierccacat` 在 PubMed seeds 0-9 上相对 TierSpecProp 为 +0.0089、8 胜 2 负，Wilcoxon greater p=0.0098；相对 `ccacat` 为 +0.0115、7 胜 1 平 2 负，p=0.0820。
- 当前代码复跑后，`tierccacat` 在 PubMed seeds 0-9 相对 TierSpecProp 为 +0.0090、8 胜 2 负；Photo seeds 0-9 和 WikiCS 官方 20 split 均因 rank=32 跳过融合，与 TierSpecProp 逐 split 持平。

## 快速运行

```bash
python -m homogcl.train --dataset Cora --method specprop --max-prop-steps 10 --probe sklogreg
python -m homogcl.summarize --input-dir results/specprop_fullgrid --output-csv results/specprop_fullgrid_summary.csv
```

完整 smoke test：

```bash
bash scripts/run_homogcl_smoke.sh
bash scripts/run_autoprop_smoke.sh
bash scripts/run_specprop_smoke.sh
bash scripts/run_specprop_multisplit.sh
bash scripts/run_specprop_amazon_smoke.sh
bash scripts/run_specprop_amazon_multisplit.sh
bash scripts/run_corespecprop_smoke.sh
bash scripts/run_corespecprop_multisplit.sh
bash scripts/run_corespecprop_key_multisplit.sh
bash scripts/run_corespecprop_wikics_multisplit.sh
bash scripts/run_tierspecprop_multisplit.sh
bash scripts/run_tierspecprop_key_multisplit.sh
bash scripts/run_tierspecprop_wikics_multisplit.sh
bash scripts/run_local_baseline_key_multisplit.sh
bash scripts/run_local_baseline_wikics_multisplit.sh
bash scripts/run_tierccacat_multisplit.sh
```

## 当前协议

- 任务：节点分类。
- 主指标：test accuracy。
- 评估：自监督/无训练表征冻结后训练线性 logistic regression probe。默认 `--probe sklogreg` 参考 GRACE/BGRL 的 sklearn `LogisticRegression` + One-vs-Rest + C 网格；`--probe torchlogreg` 参考 GCA/CCA-SSG 的单层 `nn.Linear` + 交叉熵；`--probe ridge` 仅作为普通线性回归诊断，不作为主表默认。
- 数据划分：Planetoid public split，`split_index=0`。
- 标签使用：自监督训练不用标签；标签仅用于 linear probe train/val/test 和 edge homophily 诊断元数据。
- 结果文件名必须包含关键超参签名，避免传播阶数、排序权重等实验被覆盖。
