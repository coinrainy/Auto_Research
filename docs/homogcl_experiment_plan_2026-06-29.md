# 同配图 GCL / AutoProp 实验计划（2026-06-29）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-29
- Verification Status: PARTIALLY_VERIFIED
- Version Label: code_plan_v2_autoprop_boundary

## 2026-06-29 更新

- 原 `homogcl` 候选已按失败标准放弃：未超过 `propcat`，尤其 CiteSeer 明显落后。
- 新增 `horp` / `horpgcl`：用于验证 HoRP 教师排序是否能改进 GCL；Cora 快速结果显示 `horpgcl=0.796`，暂判失败。
- 新增 `autopropcat`：无标签传播残差平台期自动选择传播深度，作为后续所有学习式 GCL 候选必须击败的强证伪器。

## Experiment Overview

- **Title**: 同配图 GCL 传播充分性 smoke test
- **Objective**: 快速判断学习式 GCL 是否能超过自动多阶传播证伪器；若不能，及时放弃当前 idea。
- **Hypothesis**: 在 Cora、CiteSeer、PubMed public split 上，任何主方法候选都必须超过 `autopropcat` 或至少在多 split/多 seed 下显著缩小其优势；否则不具备论文主线资格。
- **Type**: training / representation evaluation

## Setup

- **Language/Framework**: Python 3.10、PyTorch 2.5.1、PyG 2.8.0。
- **Entry Command**: `bash scripts/run_homogcl_smoke.sh` 与 `bash scripts/run_autoprop_smoke.sh`
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
| Summary table | `results/smoke_summary.csv` / `results/autoprop_summary.csv` | CSV | 包含 dataset、method、seed、test_acc。 |
| Console summary | stdout | text | 按 dataset/method 输出 count、mean、std、min、max。 |

## Monitoring Configuration

- **Timeout**: 第一轮建议 30-60 分钟。
- **Monitor files**: `results/smoke/*.json`、`results/autoprop/*.json`、对应 summary CSV。
- **Experiment type override**: training。
- **Metric file**: JSON 中 `metrics.test_acc`；主 probe 固定为 `sklogreg` 线性 logistic regression。
- **Metric key**: `test_acc`。

## Analysis Plan

- **Primary metric**: public split frozen encoder + sklearn one-vs-rest logistic regression probe test accuracy。
- **Success threshold**:
  - 保留学习式 idea：候选方法在至少 2/3 数据集超过 `autopropcat`，且不能只靠测试集挑超参；需多 seed 结果稳定。
  - 放弃学习式 idea：候选方法明显低于 `autopropcat`，或其收益可由 `propcat`/`propccat` 解释。
- **Comparison**: 第一阶段 `raw`、`prop`、`propcat`、`autopropcat`、`grace`、`homogcl`、`horpgcl`；第二阶段必须纳入 HomoGCL(KDD 2023)、PROPGCL、IRGCL、RELGCL、SGRL、BGRL、CCA-SSG 等强 baseline。
- **Next ablation if retained**: 自动 K 选择阈值、传播 residual channel、相对排序 margin、degree-aware weighting、feature reliability filtering。

## 已完成快速结果

| Dataset | Method | Test Acc | Notes |
|---|---|---:|---|
| Cora | autopropcat | 0.831 | selected `K=6` |
| CiteSeer | autopropcat | 0.726 | selected `K=6` |
| PubMed | autopropcat | 0.789 | selected `K=7` |
| Cora | horpgcl | 0.796 | 失败候选 |
| Cora | propccat | 0.821 | 未超过 AutoProp |

## 当前结论

本轮没有找到已通过 smoke test 的学习式 SOTA idea。最有价值的产出是一个更强的同配图传播证伪器和更严格的失败标准。下一轮应围绕“如何超过 AutoProp”设计，而不是继续微调已失败的 `homogcl` / `horpgcl`。
