# Two-stage Reliability-weighted GCL 最小实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-27
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title**: Two-stage Reliability-weighted Graph Contrastive Learning for Node Classification
- **Objective**: 验证无标签 pair reliability 是否可以削弱 GCL 中不可靠 positive pair 与疑似 false negative 的错误对比信号，并在 heterophily graphs 上提升稳健性，同时不显著损害 homophily graphs。
- **Primary Hypothesis**: 由跨视图 embedding stability 与 prediction consistency 得到的 pair reliability，用于 weighted InfoNCE 后，应优于标准 GCL 与 shuffled reliability control，并降低 weighted false negative mass。
- **Type**: training + analysis
- **Compute Boundary**: 单卡 RTX 3060 12GB，优先小中型全图训练；PubMed 与 Squirrel/Chameleon 视显存情况采用 sampled negatives 或 chunked similarity。

## Recommended Code Structure

```text
Auto_Research/
  AGENTS.md
  docs/
    two_stage_reliability_weighted_gcl_experiment_plan.md
  configs/
    datasets/
      cora.yaml
      citeseer.yaml
      pubmed.yaml
      texas.yaml
      wisconsin.yaml
      cornell.yaml
      actor.yaml
      chameleon.yaml
      squirrel.yaml
    methods/
      grace.yaml
      bgrl.yaml
      cca_ssg.yaml
      rw_gcl_two_stage.yaml
    sweeps/
      smoke.yaml
      main_heterophily.yaml
      diagnostics.yaml
  src/
    rwgcl/
      __init__.py
      data.py
      augment.py
      encoders.py
      losses.py
      reliability.py
      trainers/
        base_trainer.py
        grace_trainer.py
        bgrl_trainer.py
        rw_gcl_trainer.py
      evaluation.py
      diagnostics.py
      logging_utils.py
      seed.py
  scripts/
    run_smoke.sh
    run_baselines.sh
    run_rw_gcl.sh
    run_diagnostics.sh
  results/
    raw/
    metrics/
    diagnostics/
    figures/
  requirements.txt
  train.py
  eval.py
  diagnose.py
```

第一周可以先不实现全部目录，但建议从一开始固定 `configs/`、`results/metrics/` 和 `results/diagnostics/` 的格式，避免后面实验表格返工。

## Setup

- **Language/Framework**: Python 3.10 或 3.11，PyTorch，PyTorch Geometric，NumPy，pandas，PyYAML，scikit-learn。
- **Entry Commands**:
  - Baseline smoke test: `python train.py --config configs/methods/grace.yaml --dataset Cora --seed 0`
  - Method smoke test: `python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0`
  - Diagnostics: `python diagnose.py --run_id <run_id> --diagnostics shuffled_reliability false_negative_mass view_consistency`
- **Working Directory**: `/root/autodl-tmp/Auto_Research`
- **Environment**: Linux + CUDA，单卡 RTX 3060 12GB。

## Datasets

### Phase 0: Smoke Tests

| Group | Dataset | Purpose | Expected Cost |
|---|---|---|---|
| Homophily | Cora | 快速验证数据加载、训练、linear eval、日志 | 低 |
| Heterophily | Texas | 快速验证异配小图、diagnostics 可跑通 | 低 |

### Phase 1: Minimal Main Results

| Group | Dataset | Loader | Metric |
|---|---|---|---|
| Homophily | Cora | PyG Planetoid | Accuracy |
| Homophily | CiteSeer | PyG Planetoid | Accuracy |
| Homophily | PubMed | PyG Planetoid | Accuracy |
| Heterophily | Texas | PyG WebKB | Accuracy |
| Heterophily | Wisconsin | PyG WebKB | Accuracy |
| Heterophily | Cornell | PyG WebKB | Accuracy |
| Heterophily | Actor | PyG Actor | Accuracy |
| Heterophily | Chameleon | PyG WikipediaNetwork | Accuracy |
| Heterophily | Squirrel | PyG WikipediaNetwork | Accuracy |

### Phase 2: Optional Robustness Extension

| Dataset | Loader | Note |
|---|---|---|
| Roman-empire | PyG HeterophilousGraphDataset | 中等规模，适合扩展 |
| Amazon-ratings | PyG HeterophilousGraphDataset | 中等规模，注意显存 |
| Minesweeper | PyG HeterophilousGraphDataset | 可能用 ROC-AUC |
| Tolokers | PyG HeterophilousGraphDataset | 可能用 ROC-AUC |
| Questions | PyG HeterophilousGraphDataset | 可作为最后扩展 |

最小投稿实验先保证 Phase 1。Phase 2 只在原型稳定且时间允许时加入。

## Baselines

### 必须优先复现

| Baseline | Role | Priority |
|---|---|---|
| GRACE | 标准 graph contrastive learning，对比增强敏感 | P0 |
| BGRL | 无负样本/bootstrapping 参照，帮助说明保留负样本的价值 | P0 |
| DGI 或 CCA-SSG | 轻量 GSSL 参照，补足经典 baseline | P1 |

### 第二阶段补充

| Baseline | Role | Priority |
|---|---|---|
| GCA | adaptive augmentation 参照 | P1 |
| ProGCL | hard negative / false negative 相关强对照 | P1 |
| MVGRL | 经典多视图 GCL，工程量略高 | P2 |
| GBT | redundancy reduction 类参照 | P2 |
| GraphACL / HeterGCL / HLCL | heterophily-aware GCL 参照，有结果后再决定是否复现或引用官方结果 | P2 |

### Sanity Baselines

MLP、GCN、GAT 作为监督或半监督 sanity baseline，用于确认数据 split 与评价流程正常。它们不是主对比对象。

## Method Definition

### Stage 1: Conservative Warm-up

- 使用保守 graph augmentation，例如较低 edge drop rate 与 feature mask rate。
- 训练一个 GRACE-like encoder，得到 online/student encoder。
- 同步维护 EMA teacher encoder，用 stop-gradient 避免 reliability score 直接追逐当前噪声。
- warm-up 结束后保存每个节点的 teacher embedding 与 predictor output。

### Stage 2: Reliability-weighted Contrastive Training

Positive pair reliability:

```text
s_i = normalized cosine stability between student view and EMA teacher view
c_i = normalized cross-view prediction consistency
r_i_pos = lambda * s_i + (1 - lambda) * c_i
```

其中 `prediction consistency` 不使用真实标签。建议先实现为 unlabeled predictor/prototype head 的输出一致性，或者 projection/prediction vector 的 softmax consistency。若使用类别数作为 prototype 数量，需要在论文中说明只使用数据集元信息，不使用标签分配。

Negative pair reliability:

```text
q_ij = semantic similarity or prediction agreement between anchor i and negative j
w_ij_neg = 1 - q_ij
```

`w_ij_neg` 越低，说明该 negative 越像疑似 false negative，应减少被强推远的力度。为了显存安全，先采用 sampled negatives，每个 anchor 采样 `K=256` 或 `512` 个 negatives；PubMed/Squirrel 可用 chunked similarity。

Weighted InfoNCE:

```text
L_i = - r_i_pos * log exp(sim(z_i^1, z_i^2) / tau)
      / [exp(sim(z_i^1, z_i^2) / tau) + sum_j w_ij_neg * exp(sim(z_i^1, z_j^2) / tau)]
```

实现时可以先做两个版本：

1. positive weighting only：只加 `r_i_pos`，最稳、最简单。
2. positive + negative weighting：同时降低疑似 false negative 的 denominator 权重。

若版本 2 不稳定，主方法先使用版本 1，negative reliability 进入诊断或附录。

## Core Metrics

| Metric | Type | Purpose |
|---|---|---|
| Node classification accuracy / ROC-AUC | Performance | 主任务指标 |
| Mean ± std over seeds | Robustness | 至少 5 seeds，最终建议 10 seeds |
| Homophily non-degradation | Safety | 同配图不应明显低于 GRACE/BGRL/CCA-SSG |
| Runtime per epoch / total runtime | Practicality | 证明 RTX 3060 可复现 |
| Peak GPU memory | Practicality | 避免方法被认为过重 |

建议成功阈值：

- heterophily 上至少 3 个数据集优于 GRACE 或 BGRL，且 shuffled reliability 明显更弱；
- homophily 上相对主 baseline 下降不超过 1-2 个百分点，或落在标准差范围内；
- diagnostics 支持机制，即使 accuracy 不是全面 SOTA，也可作为机制性贡献。

## Diagnostic Experiments

### D1. Shuffled Reliability Control

目的：回应“只是任意加权或更多超参数”的质疑。

设计：

- 保留完整训练流程；
- 保留 reliability score 的数值分布；
- 保留 loss weighting 形式；
- 打乱 reliability score 与 node/pair 的对应关系；
- 比较 full reliability vs shuffled reliability。

关键判据：

- full reliability 在 heterophily 数据集上明显优于 shuffled；
- 若 shuffled 接近 full，则 reliability 机制不成立，需要回退方法叙事。

### D2. Weighted False Negative Mass

目的：验证方法是否真的降低同类节点被强推远的训练质量。

训练不使用标签，诊断阶段使用标签。

示例定义：

```text
WFNM = sum_i sum_j w_ij_neg * I[y_i = y_j] / sum_i sum_j I[y_i = y_j]
```

也可以报告 false negative 在 denominator weighted mass 中的占比。

关键判据：

- RW-GCL 的 WFNM 低于 GRACE/InfoNCE；
- full reliability 低于 shuffled reliability；
- 降低 WFNM 的数据集最好与性能提升数据集部分重合。

### D3. View Consistency by Reliability Bucket

目的：验证高 reliability pair 确实更稳定。

设计：

- 将 positive pairs 按 reliability 分为 high / mid / low 三档；
- 分别统计 embedding stability、prediction consistency、post-hoc label agreement；
- 可加跨 seed 稳定性。

关键判据：

- high-reliability bucket 的 view consistency 高于 low-reliability bucket；
- 如果 high/low 分组无差异，reliability score 解释力不足。

### D4. Core Ablation

| Variant | Purpose |
|---|---|
| no reliability weighting | 验证 weighted loss 必要性 |
| no embedding stability | 验证 stability 信号贡献 |
| no prediction consistency | 验证 consistency 信号贡献 |
| shuffled reliability | 最关键反证 |
| positive weighting only | 检查更简单版本是否足够 |
| positive + negative weighting | 检查完整版本收益与稳定性 |

### D5. Optional 2x2 Closed-loop Ablation

仅当 two-stage 最小方法成立后再做。

| Reliability Update | Augmentation Control | Variant |
|---|---|---|
| fixed after warm-up | fixed augmentation | two-stage baseline |
| EMA slow update | fixed augmentation | dynamic reliability only |
| fixed after warm-up | reliability-guided augmentation | augmentation feedback only |
| EMA slow update | reliability-guided augmentation | full closed-loop |

如果 closed-loop 不优于 two-stage，主文不要强推 closed-loop。

## Expected Outputs

| Output | Path | Format | Success Criterion |
|---|---|---|---|
| run config snapshot | `results/raw/<run_id>/config.yaml` | YAML | 完整记录 dataset/method/seed |
| training log | `results/raw/<run_id>/train_log.csv` | CSV | 至少包含 epoch/loss/time |
| final metrics | `results/metrics/main_results.csv` | CSV | 包含 dataset/method/seed/metric |
| ablation metrics | `results/metrics/ablations.csv` | CSV | 每个 variant 至少 5 seeds |
| shuffled reliability report | `results/diagnostics/shuffled_reliability.csv` | CSV | full vs shuffled 可比较 |
| false negative mass | `results/diagnostics/false_negative_mass.csv` | CSV | 包含 method/dataset/seed/WFNM |
| view consistency | `results/diagnostics/view_consistency.csv` | CSV | 包含 reliability bucket |
| runtime report | `results/metrics/runtime.csv` | CSV | time/memory 可复现 |

## Monitoring Configuration

- **Smoke timeout**: 每个 dataset/method/seed 30 分钟。
- **Main run timeout**: 小图 1 小时，PubMed/Squirrel/Chameleon 2-3 小时。
- **Monitor files**:
  - `results/raw/<run_id>/train_log.csv`
  - `results/raw/<run_id>/stderr.log`
  - `results/raw/<run_id>/stdout.log`
- **Failure signals**:
  - loss NaN；
  - GPU OOM；
  - accuracy 明显低于 MLP/GCN sanity baseline；
  - shuffled reliability 与 full reliability 无差异；
  - homophily 数据集明显退化。

## First-week Task Table

| Day | Goal | Concrete Tasks | Exit Criterion |
|---|---|---|---|
| 1 | 仓库骨架与环境 | 建立目录结构、`requirements.txt`、seed 工具、config loader、日志格式 | `python train.py --help` 可运行 |
| 2 | 数据集加载 | 实现 Planetoid、WebKB、Actor、WikipediaNetwork loader；统一 split 与 metric | Cora 与 Texas 可打印数据统计 |
| 3 | baseline smoke | 实现或接入 GRACE；可选 DGI/BGRL 骨架 | GRACE 在 Cora/Texas 单 seed 跑通 |
| 4 | evaluation pipeline | linear eval、mean/std 聚合、CSV 结果表 | `results/metrics/main_results.csv` 自动生成 |
| 5 | two-stage warm-up | 实现 conservative augmentation、EMA teacher、warm-up checkpoint | Cora/Texas 生成 warm-up embedding |
| 6 | reliability-weighted loss | 实现 embedding stability、prediction consistency、positive weighting only | RW-GCL 在 Cora/Texas 跑通 |
| 7 | 三个诊断雏形 | shuffled reliability、false negative mass、view consistency bucket | 能输出 3 个 diagnostics CSV |

第一周不追求 SOTA，只追求可运行、可诊断、可复现。

## Go / No-go Criteria After Week 1

### Green

- GRACE 与 RW-GCL 都能在 Cora/Texas 跑通；
- shuffled reliability 比 full reliability 弱；
- false negative mass 或 view consistency 至少一个诊断支持机制。

### Yellow

- accuracy 没有提升，但 diagnostics 有信号；
- 先继续完善 method 与更多 seeds。

### Red

- shuffled reliability 与 full reliability 完全无差异；
- reliability bucket 无法区分 view consistency；
- homophily 明显退化且无法通过简化 weighting 解决。

Red 时应回退为“错误对比信号诊断论文”或重新设计 reliability score。

## What Not To Do Yet

- 暂不写完整论文大纲。
- 暂不复现所有 baseline。
- 暂不加入 high/low-pass gate 作为主方法。
- 暂不实现 full closed-loop augmentation。
- 暂不跑大规模 OGB。
- 暂不为了追 accuracy 增加过多 tricks。

## Immediate Next Commands

```bash
mkdir -p configs/datasets configs/methods configs/sweeps src/rwgcl/trainers scripts results/raw results/metrics results/diagnostics results/figures
touch requirements.txt train.py eval.py diagnose.py
```

随后优先实现：

```bash
python train.py --config configs/methods/grace.yaml --dataset Cora --seed 0
python train.py --config configs/methods/rw_gcl_two_stage.yaml --dataset Texas --seed 0
```
