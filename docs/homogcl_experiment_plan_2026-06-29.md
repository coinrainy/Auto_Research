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
- 新增 `specprop`：AutoProp + 无标签谱集中度门控 + 低秩去噪。full C-grid 单 seed public split 下，Cora/PubMed 超过 AutoProp，CiteSeer 持平。

## Experiment Overview

- **Title**: SpecProp 同配图谱充分性与低秩去噪 smoke test
- **Objective**: 判断无标签谱集中度能否预测传播银行是否需要压缩，从而稳定超过 AutoProp。
- **Hypothesis**: 如果传播银行的前 10 个主成分能量占比足够高，则低秩瓶颈可降低小标签 linear probe 过拟合；如果谱分散，则回退到 AutoProp 可避免损伤。
- **Type**: training / representation evaluation

## Setup

- **Language/Framework**: Python 3.10、PyTorch 2.5.1、PyG 2.8.0。
- **Entry Command**: `bash scripts/run_homogcl_smoke.sh`、`bash scripts/run_autoprop_smoke.sh`、`bash scripts/run_specprop_smoke.sh`
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
| Summary table | `results/smoke_summary.csv` / `results/autoprop_summary.csv` / `results/specprop_fullgrid_summary.csv` | CSV | 包含 dataset、method、seed、test_acc、selected K 和 selected rank。 |
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
| Cora | horpgcl | 0.796 | 失败候选 |
| Cora | propccat | 0.821 | 未超过 AutoProp |

## 当前结论

`SpecProp` 是当前最值得继续的候选：它在 full-grid 单 seed public split 上对 AutoProp 的增益为 Cora +0.010、CiteSeer +0.000、PubMed +0.005。该证据仍不足以声明 SOTA；下一步必须做多 seed、多 split、阈值消融和更大同配图验证。
