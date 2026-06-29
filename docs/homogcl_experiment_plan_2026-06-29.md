# 同配图 GCL / SpecProp 实验计划（2026-06-29）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-29
- Verification Status: PARTIALLY_VERIFIED
- Version Label: code_plan_v3_specprop_candidate

## 2026-06-29 更新

- 原 `homogcl` 候选已按失败标准放弃：未超过 `propcat`，尤其 CiteSeer 明显落后。
- 新增 `horp` / `horpgcl`：用于验证 HoRP 教师排序是否能改进 GCL；Cora 快速结果显示 `horpgcl=0.796`，暂判失败。
- 新增 `autopropcat`：无标签传播残差平台期自动选择传播深度，作为后续所有学习式 GCL 候选必须击败的强证伪器。
- 新增 safe-gated `specprop`：AutoProp + 无标签谱集中度门控 + 低秩去噪。仅在 top-10 PCA 能量占比 >= 0.34 时压缩到 rank=32，否则回退到 AutoProp。class-balanced random split 下 PubMed 稳定提升，Cora/CiteSeer 回退持平；Amazon Photo smoke 显示显著提升，Computers 回退持平。

## Experiment Overview

- **Title**: SpecProp 同配图谱充分性与低秩去噪 smoke test
- **Objective**: 判断无标签谱集中度能否预测传播银行是否需要压缩，并识别 SpecProp 何时应回退到 AutoProp。
- **Hypothesis**: 如果传播银行的前 10 个主成分能量占比足够高，则低秩瓶颈可降低小标签 linear probe 过拟合；如果谱分散或中等集中，则回退到 AutoProp 可避免损伤。
- **Type**: training / representation evaluation

## Setup

- **Language/Framework**: Python 3.10、PyTorch 2.5.1、PyG 2.8.0。
- **Entry Command**: `bash scripts/run_homogcl_smoke.sh`、`bash scripts/run_autoprop_smoke.sh`、`bash scripts/run_specprop_smoke.sh`、`bash scripts/run_specprop_multisplit.sh`、`bash scripts/run_specprop_photo_smoke.sh`
- **Working Directory**: `/root/autodl-tmp/Auto_Research`
- **Dependencies**: 使用当前环境已安装的 torch / torch_geometric / pandas。

## Inputs

| Input | Path | Description |
|---|---|---|
| Planetoid cache | `data/Planetoid` | Cora、CiteSeer、PubMed public split 数据缓存；缺失时由 PyG 下载。 |
| Source code | `homogcl/` | 方法、基线、评估与汇总脚本。 |

## Expected Outputs

| Output | Path | Format | Success Criterion |
|---|---|---|---|
| Run metadata | `results/*/*.json` | JSON | 每个 dataset/method/seed 有一个文件，含 split、seed、git、metric 和关键超参签名。 |
| Summary table | `results/smoke_summary.csv` / `results/autoprop_summary.csv` / `results/specprop_fullgrid_summary.csv` / `results/specprop_multisplit_paired.csv` | CSV | 包含 dataset、method、seed、test_acc、selected K、selected rank 和 paired delta。 |
| Console summary | stdout | text | 按 dataset/method 输出 count、mean、std、min、max。 |

## Monitoring Configuration

- **Timeout**: 第一轮建议 30-60 分钟。
- **Monitor files**: `results/smoke/*.json`、`results/autoprop/*.json`、`results/specprop_fullgrid/*.json`、对应 summary CSV。
- **Experiment type override**: training。
- **Metric file**: JSON 中 `metrics.test_acc`；主 probe 固定为 `sklogreg` 线性 logistic regression。
- **Metric key**: `test_acc`。

## Analysis Plan

- **Primary metric**: public split frozen encoder + sklearn one-vs-rest logistic regression probe test accuracy。
- **Success threshold**:
  - 保留 `specprop`：在至少 2/3 数据集超过或持平 `autopropcat`，平均 test accuracy 高于 AutoProp，且 rank 选择只依赖无标签谱统计。
  - 放弃学习式 idea：候选方法明显低于 `autopropcat`，或其收益可由 `propcat`/`propccat` 解释。
- **Comparison**: 第一阶段 `raw`、`prop`、`propcat`、`autopropcat`、`specprop`、`grace`、`homogcl`、`horpgcl`；第二阶段必须纳入 HomoGCL(KDD 2023)、PROPGCL、IRGCL、RELGCL、SGRL、BGRL、CCA-SSG 等强 baseline。
- **Next ablation if retained**: 谱集中度阈值、低秩 rank 规则、AutoProp K 阈值、PCA vs randomized SVD、是否拼接原 propagation bank。

## 已完成快速结果

| Dataset | Method | Test Acc | Notes |
|---|---|---:|---|
| Cora | autopropcat full-grid | 0.824 | selected `K=6` |
| CiteSeer | autopropcat full-grid | 0.723 | selected `K=6` |
| PubMed | autopropcat full-grid | 0.793 | selected `K=7` |
| Cora | specprop full-grid | 0.834 | selected `K=6`, rank=647 |
| CiteSeer | specprop full-grid | 0.723 | selected `K=6`, no compression |
| PubMed | specprop full-grid | 0.798 | selected `K=7`, rank=32 |
| Cora | specprop strict random split mean | 0.8212 | delta +0.0000，回退持平 |
| CiteSeer | specprop random split mean | 0.7107 | delta +0.0000，回退持平 |
| PubMed | specprop random split mean | 0.7710 | delta +0.0180，3 胜 0 负 |
| Photo | specprop safe random split seed 0 | 0.8985 | AutoProp 0.8644，delta +0.0341 |
| Computers | specprop safe random split seed 0 | 0.7984 | AutoProp 0.7984，回退持平 |
| Cora | horpgcl | 0.796 | 失败候选 |
| Cora | propccat | 0.821 | 未超过 AutoProp |

## 当前结论

Safe-gated `SpecProp` 是当前最值得继续的条件性候选：它在 class-balanced random split seeds 0/1/2 上对 Cora/CiteSeer 无损回退，对 PubMed 稳定提升；Amazon Photo smoke 也出现大幅提升，而 Computers 反例被 0.34 阈值安全回退。下一步必须扩展到更多 Amazon/Coauthor split，验证“高谱集中 -> 低秩去噪有效”的规律是否稳健。
