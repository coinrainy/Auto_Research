# 项目协作记录

## 固定协作规则

- 回答与项目文档尽量使用中文。
- 在当前工作目录内工作，不参考其他目录的项目代码。
- 新项目需要维护 `AGENTS.md`，每次任务结束同步更新当前状态。
- 如当前目录是 GitHub 仓库，任务结束后按用户要求同步上传；当前目录已初始化为 Git 仓库，远程目标为 GitHub 公开仓库 `coinrainy/Auto_Research`。
- 每轮任务结束时给出建议的后续命令。

## 当前状态

- 日期：2026-06-29（UTC）。
- 当前研究方向：图对比学习，优先同配图节点分类。
- 已建立 `homogcl/` 原型工作区、`scripts/` 运行脚本和 `results/` 早筛结果。
- 原 `homogcl` 候选（同配增强 + 多正样本 InfoNCE）未通过 smoke test，已标记为失败候选，不应继续包装为主方法。
- 新增 `horp` / `horpgcl` 诊断路线；`horpgcl` 在 Cora 快速测试中未超过传播证伪器，暂判失败。
- `autopropcat`：基于无标签传播残差平台期自动选择传播深度，作为后续 GCL 候选必须击败的强证伪器。
- 新增当前条件性候选 `specprop`：AutoProp + 安全谱集中度门控 + 低秩去噪；仅当 top-10 PCA 能量占比 >= 0.34 时压缩到 rank=32，否则回退到 AutoProp。
- 当前 full C-grid public split 快速结果：SpecProp 相对 AutoProp 为 Cora +0.010、CiteSeer +0.000、PubMed +0.005。
- 当前 strict SpecProp class-balanced random split seeds 0/1/2 paired 结果：Cora +0.000（回退持平）、CiteSeer +0.000（回退持平）、PubMed +0.018（3 胜 0 负）。
- Amazon class-random seeds 0/1/2 paired 结果：Photo 平均 0.9071 vs AutoProp 0.8745，delta +0.0326，3 胜 0 负；Computers 平均 0.7965 vs AutoProp 0.7965，回退持平。
- 新增条件性候选 `corespecprop`：AutoProp 传播银行 + 安全谱集中度门控 + 参与秩自适应核心压缩；top-10 PCA 能量占比 < 0.34 时回退到 AutoProp，触发时将 rank 裁剪到 16-32。该版本后续被 WikiCS rank 消融修正，不再是当前最佳主线。
- `corespecprop` 5 数据集 seeds 0/1/2 paired 结果：Cora/CiteSeer/Computers 回退持平；PubMed 平均 0.7804 vs AutoProp 0.7530，delta +0.0274，3 胜 0 负；Photo 平均 0.9051 vs AutoProp 0.8745，delta +0.0306，3 胜 0 负。
- `corespecprop` 正例图 seeds 0-9 压力测试：PubMed 平均 0.7739 vs AutoProp 0.7541，delta +0.0198，10 胜 0 负，Wilcoxon greater p=0.000977；Photo 平均 0.9002 vs AutoProp 0.8817，delta +0.0185，10 胜 0 负，Wilcoxon greater p=0.000977。
- 新增 WikiCS 官方 20 split 支持与结果：CoreSpecProp 平均 0.7702 vs AutoProp 0.7636，delta +0.0066，18 胜 2 负，Wilcoxon greater p=0.000182；这说明高谱集中图上总体有效，但不再是严格逐 split 无损。
- `tierspecprop`：top-10 PCA 能量 < 0.34 回退；0.34-0.36 选择窄 rank=16；>=0.36 选择宽 rank=32。WikiCS rank 消融显示固定 rank=32 平均 0.7833 vs AutoProp 0.7636，delta +0.0197，20 胜 0 负，显著优于 core rank16 的 0.7702；PubMed rank32 出现 3 个负 split，因此采用分层 rank。当前已降级为强谱核心分支，而不是最终主方法。
- `tierspecprop` 当前关键结果：PubMed seeds 0-9 平均 0.7739 vs 0.7541，delta +0.0198，10 胜 0 负；Photo seeds 0-9 平均 0.9035 vs 0.8817，delta +0.0218，10 胜 0 负；WikiCS 官方 20 split 平均 0.7833 vs 0.7636，delta +0.0197，20 胜 0 负，Wilcoxon greater p=4.42e-05。
- 用户明确要求 Coauthor CS/Physics 先不做；后续建议暂缓 Coauthor 扩展。
- 新增非 Coauthor 本地基线面板脚本：`scripts/run_local_baseline_key_multisplit.sh` 覆盖 PubMed/Photo，`scripts/run_local_baseline_wikics_multisplit.sh` 覆盖 WikiCS；默认比较 `tierccacat`、`tierspecprop`、`autopropcat`、`propccat`、`ccacat`、`gracecat`，用于下一轮强基线前的仓库内快速压力测试。
- PubMed 本地基线面板已完成 seeds 0-9：TierSpecProp vs `propccat` 为均值 +0.0136、9 胜 1 负；vs `gracecat` 为均值 +0.0152、9 胜 1 负；但 vs `ccacat` 仅均值 +0.0025、4 胜 6 负。结论：纯 TierSpecProp 不再作为最终主方法包装，应降级为谱核心分支。
- 新增当前保留候选 `tierccacat`：仅在窄谱核心 rank=16 时拼接 L2-normalized TierSpecProp 谱核心与 CCA-GCN 去相关残差；回退或宽谱核心 rank=32 时保持纯 TierSpecProp。当前代码复跑 PubMed seeds 0-9 相对 TierSpecProp 为 +0.0090、8 胜 2 负，Wilcoxon greater p=0.0068；相对 `ccacat` 为 +0.0115、7 胜 3 负，p=0.0967；相对 `propccat` 为 +0.0226、10 胜 0 负。Photo seeds 0-9 与 WikiCS 官方 20 split 均为 rank=32、fusion_applied=0，逐 split 等价于 TierSpecProp；相对 AutoProp 分别为 Photo +0.0218（10 胜 0 负）和 WikiCS +0.0197（20 胜 0 负）。
- 新增 `scripts/run_tierccacat_multisplit.sh`，用于复跑 PubMed/Photo/WikiCS 上 `autopropcat`、`tierspecprop`、`tierccacat` 的非 Coauthor 多 split 面板。
- 协议细节见：
  - `docs/gcl_experiment_protocol_checklist.md`
  - `docs/context_reset_protocol_only_2026-06-29.md`
  - `docs/homogcl_research_brief_2026-06-29.md`
  - `docs/homogcl_experiment_plan_2026-06-29.md`
  - `docs/corespecprop_research_brief_2026-06-29.md`
  - `docs/tierccacat_research_brief_2026-06-29.md`

## 后续原则

- 下一轮研究主线应围绕 `tierccacat` 验证“安全谱核心 + 去相关残差”是否能在更多非 Coauthor 同配图上稳定压过 TierSpecProp 与 `ccacat`；如果不能，应放弃该融合候选，不要继续微调已失败的 `homogcl` / `horpgcl`。
- 若继续做学习式 GCL，必须纳入 HomoGCL(KDD 2023)、PROPGCL、IRGCL、RELGCL、SGRL、BGRL、CCA-SSG 等强 baseline。
- 论文级证据必须扩展到多 split、多 seed、更大同配图，并报告测试集不可见的超参选择规则。
- 如果需要写新方法，应保持在当前仓库内清晰隔离，不能参考其他目录代码。

## 建议后续命令

```bash
git status --short
bash scripts/run_autoprop_smoke.sh
bash scripts/run_specprop_smoke.sh
bash scripts/run_specprop_multisplit.sh
bash scripts/run_specprop_amazon_smoke.sh
bash scripts/run_specprop_amazon_multisplit.sh
bash scripts/run_corespecprop_smoke.sh
bash scripts/run_corespecprop_multisplit.sh
bash scripts/run_corespecprop_key_multisplit.sh
bash scripts/run_corespecprop_wikics_multisplit.sh
bash scripts/run_corespecprop_wikics_rank_ablation.sh
bash scripts/run_tierspecprop_multisplit.sh
bash scripts/run_tierspecprop_key_multisplit.sh
bash scripts/run_tierspecprop_wikics_multisplit.sh
EPOCHS=1 SPLIT_SEEDS="0" METHODS="autopropcat tierspecprop tierccacat" OUT_DIR=results/local_baseline_key_syntax bash scripts/run_local_baseline_key_multisplit.sh
EPOCHS=1 SPLIT_INDICES="0" METHODS="autopropcat tierspecprop tierccacat" OUT_DIR=results/local_baseline_wikics_syntax bash scripts/run_local_baseline_wikics_multisplit.sh
bash scripts/run_local_baseline_key_multisplit.sh
bash scripts/run_local_baseline_wikics_multisplit.sh
bash scripts/run_tierccacat_multisplit.sh
python -m homogcl.train --dataset PubMed --split class-random --split-seed 0 --method tierccacat --epochs 200 --prop-steps 10 --max-prop-steps 10 --autoprop-plateau-ratio 0.75 --specprop-high-concentration 0.34 --tierspecprop-wide-concentration 0.36 --tierspecprop-narrow-rank 16 --tierspecprop-wide-rank 32 --probe sklogreg --logreg-c-grid 0.25,1,4,16 --output-dir results/tierccacat_pubmed_probe
python -m homogcl.compare --input-dirs results/local_baseline_key_multisplit,results/tierccacat_pubmed_probe --baseline tierspecprop --candidate tierccacat --output-csv results/tierccacat_pubmed_vs_tierspecprop.csv
python -m homogcl.compare --input-dirs results/local_baseline_key_multisplit,results/tierccacat_pubmed_probe --baseline ccacat --candidate tierccacat --output-csv results/tierccacat_pubmed_vs_ccacat.csv
python -m homogcl.compare --input-dirs results/specprop_safe_multisplit --baseline autopropcat --candidate specprop --output-csv results/specprop_safe_multisplit_paired.csv
```
