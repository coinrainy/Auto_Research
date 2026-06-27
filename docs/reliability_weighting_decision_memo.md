# Reliability-weighted GCL 实验决策备忘录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / plan handoff
- Origin Date: 2026-06-27
- Verification Status: ANALYZED
- Version Label: decision_memo_v1
- Evidence Scope: 本备忘录基于当前仓库已完成的 6 个 heterophily 数据集与 3 个 homophily 数据集、10 seeds、GRACE baseline 与 RW-GCL normal/shuffled 对照结果。

## 当前判断

当前 two-stage positive reliability-weighted GCL 不能表述为通用 heterophily 提升方法。更稳妥的表述是：

> embedding stability + prediction consistency 得到的 positive reliability ranking 在 6 个 heterophily 数据集上稳定对应更高的跨视图一致性，但 accuracy 收益具有明显数据集条件性，主要出现在 Texas 与 Chameleon。

这意味着下一步必须做路线选择：

- 路线 A：继续作为方法论文推进，加入最小 degree/local-graph-aware gate，检验当前 failure analysis 发现的偏置是否可修正。
- 路线 B：收缩为机制诊断论文，暂不堆模块，补强 false negative / negative weighting / homophily non-degradation 证据。

## 证据边界

### 已支持

| Evidence | Current result | Interpretation |
|---|---:|---|
| Reliability ranking -> view consistency | 6/6 数据集 high-low view consistency gap 均为正 | 机制排序信号稳定存在 |
| Texas performance | normal - GRACE = +0.024324；normal - shuffled = +0.029730 | 最强正向数据集，且 shuffled control 支持 reliability 非随机有效 |
| Chameleon performance | normal - GRACE = +0.008772；normal - shuffled = +0.003509 | 小幅正向，值得作为第二个正例 |
| Failure pattern | Cornell/Wisconsin/Squirrel 为负或不稳定 | 方法收益有边界，不能泛化宣传 |

### 尚未支持

| Claim | Status | Missing evidence |
|---|---|---|
| 通用 heterophily SOTA | 不支持 | 6 数据集中只有 Texas/Chameleon 明显正向 |
| reliability weighting 能减少 false negatives | 未验证 | 当前实现只有 positive weighting，false negative mass 仍是 not applicable |
| closed-loop augmentation 必要 | 未验证 | 当前仍是 two-stage fixed augmentation |
| high/low-pass gate 是核心贡献 | 不支持 | 当前主方法没有 gate，且未做 gate ablation |
| homophily non-degradation | 初步支持但有风险 | Cora/CiteSeer 基本不退化，PubMed 约 -0.0112 |

## 六数据集结果

| Dataset | normal - GRACE | normal - shuffled | view consistency gap | Current role |
|---|---:|---:|---:|---|
| Texas | +0.024324 | +0.029730 | 0.101067 | Strong positive case |
| Chameleon | +0.008772 | +0.003509 | 0.170735 | Weak positive case |
| Actor | +0.002500 | -0.000790 | 0.092376 | Near-zero boundary case |
| Squirrel | -0.003554 | -0.003074 | 0.243694 | Failure / high-degree stress case |
| Cornell | -0.016216 | -0.013514 | 0.112930 | Failure case |
| Wisconsin | -0.027451 | -0.003922 | 0.116375 | Failure case |

## Homophily Non-degradation

| Dataset | normal - GRACE | normal - shuffled | view consistency gap | Current role |
|---|---:|---:|---:|---|
| Cora | +0.003300 | +0.001600 | 0.048800 | Non-degradation supported |
| CiteSeer | +0.000100 | -0.002700 | 0.037227 | Essentially neutral |
| PubMed | -0.011200 | +0.002800 | 0.071131 | Mild degradation risk |

当前解释：

- Cora 与 CiteSeer 支持“不显著伤害 homophily baseline”的安全性叙事。
- PubMed 有约 1.12 个百分点下降，仍在早期设定的 1-2 个百分点观察带内，但不能写成完全无退化。
- homophily 结果不支持继续把方法包装为全局提升器；更适合写成条件性方法，并在限制部分诚实说明 PubMed 风险。

## Label-based False-negative Pressure

该诊断只用于离线分析，不进入训练。定义为：对每个 anchor，计算同标签非自身节点在 embedding softmax denominator 中占的质量。数值越高，表示 InfoNCE 中潜在同类负样本压力越大。

| Dataset | FN pressure | weighted - unweighted | high-low FN pressure gap | reliability-pressure corr | Interpretation |
|---|---:|---:|---:|---:|---|
| Texas | 0.404443 | -0.003087 | -0.124443 | -0.207178 | 最支持当前机制：高 reliability 桶同类负样本压力明显更低，positive weighting 略微降低整体压力 |
| Wisconsin | 0.360142 | -0.000733 | +0.004351 | -0.065191 | 加权影响很弱，不能解释负向性能 |
| Cornell | 0.300963 | -0.000449 | -0.012915 | -0.045121 | 机制信号弱，性能负向 |
| Actor | 0.214085 | -0.000187 | -0.011010 | -0.085250 | 机制信号弱，性能近零 |
| Chameleon | 0.221022 | +0.000194 | +0.011854 | +0.065086 | 小幅正向性能不是由降低 FN pressure 解释 |
| Squirrel | 0.203202 | +0.000022 | +0.003296 | +0.010146 | view consistency 强但 FN pressure 不降，支持高 degree 稳定性偏置解释 |
| Cora | 0.263036 | +0.000081 | +0.004606 | +0.025784 | homophily 上加权几乎不改变 FN pressure |
| CiteSeer | 0.252294 | +0.000100 | +0.006916 | +0.069722 | homophily 上加权几乎不改变 FN pressure |
| PubMed | 0.414388 | +0.000007 | +0.003068 | +0.005991 | PubMed 退化不由 FN pressure 增加直接解释 |

当前最强机制证据来自 Texas，而不是所有数据集。更准确的表述应是：

> reliability weighting 在 Texas 上同时满足性能提升、shuffled control 支持、以及 label-based false-negative pressure 下降；但在 Chameleon/Squirrel/Actor 等数据集上，view consistency gap 与 FN pressure 改善并不一致。

## Failure Analysis 线索

| Dataset | high-low degree gap | high-low local homophily gap | high-low class entropy gap | Interpretation |
|---|---:|---:|---:|---|
| Texas | -1.059017 | +0.005332 | +0.241309 | high reliability 不偏高 degree，且更类别分散；可能缓解低可靠桶类别集中 |
| Chameleon | +5.603426 | -0.039109 | -0.020375 | high reliability 偏高 degree；小幅收益可能来自 hubs/高连接区域 |
| Actor | +1.163126 | -0.017520 | +0.037775 | 近零收益，degree 偏置存在但不明显转化为提升 |
| Squirrel | +2.691801 | -0.020759 | +0.006813 | view consistency 最强但 accuracy 负向，疑似 reliability 过度偏向高 degree 稳定节点 |
| Cornell | -0.467213 | -0.040697 | +0.054649 | 小图高方差，reliability 未带来性能收益 |
| Wisconsin | -0.607458 | +0.014662 | +0.068348 | reliability ranking 存在，但性能显著负向 |

当前最可检验的机制假设：

> 在部分图上，positive reliability 更像“跨视图稳定性/高 degree 稳定性”而不一定等价于“有利于节点分类的语义可靠性”。Texas 的正向可能来自低可靠桶类别集中偏置被降权；Squirrel/Actor/Chameleon 则需要检验 high-degree 稳定节点是否被过度放大。

## 路线 A：最小 Gate 方法实验

目标：验证 failure analysis 发现的 degree / local graph context 是否能修正 positive reliability weighting 的条件性失败。

最小实现原则：

- 不引入 closed-loop augmentation。
- 不改 encoder、augmentation、linear probe 与现有训练主流程。
- 只在 stage2 positive reliability 上加一个可关闭的 post-hoc gate。
- 默认新增方法名建议：`rw_gcl_degree_gate` 或 `rw_gcl_context_gate`。

建议 gate 形式：

```text
degree_score_i = normalized_log_degree_i
local_homophily_i = local label-free proxy 或暂用 structural homophily proxy
gate_i = clip(1 - alpha * degree_score_i, min_gate, 1)
r'_i = normalize_or_clip(r_i * gate_i)
```

第一版不要使用真实标签 local homophily 参与训练；标签信息只能用于诊断。训练 gate 优先只用 degree，因为它无标签、稳定、工程成本低。

### 路线 A 成功标准

进入下一阶段需要至少满足：

- Squirrel 或 Actor 任一数据集 normal - GRACE 提升到非负且 normal - shuffled 不再为负；
- Texas normal - GRACE 不下降超过 0.01；
- Chameleon normal - GRACE 保持非负；
- Cora/CiteSeer/PubMed 的 mean degradation 不得比当前 two-stage 更差，PubMed 下降不超过 0.015；
- view consistency gap 仍为正；
- false-negative pressure weighted - unweighted 不得比当前 two-stage 更差；Squirrel/Actor 至少一个数据集的 high-low FN pressure gap 应下降；
- degree high-low gap 相比原方法有下降，说明 gate 确实改变了 failure analysis 指向的偏置。

### 路线 A 停止标准

任一情况出现时停止堆模块：

- Texas 正向信号被破坏；
- PubMed 或任一 homophily 数据集退化超过 0.02；
- Squirrel/Actor/Cornell/Wisconsin 没有任何改善；
- gate 改善 accuracy 但显著增加 false-negative pressure；
- shuffled reliability 与 normal reliability 结果接近或更好；
- gate 只改善一个数据集但引入 2 个以上数据集明显退化；
- 新增超参数超过 2 个仍无法稳定。

### 路线 A 建议命令

```bash
python train.py --config configs/methods/rw_gcl_degree_gate.yaml --dataset Texas --seed 0 --mode execute --warmup-epochs 20 --stage2-epochs 50 --eval-epochs 50
```

如果 smoke 成功，再跑：

```bash
METHOD_CONFIG=configs/methods/rw_gcl_degree_gate.yaml METHOD_NAME=rw_gcl_degree_gate DATASETS="Texas Chameleon Squirrel Actor" SEEDS="0 1 2" WARMUP_EPOCHS=20 STAGE2_EPOCHS=50 EVAL_EPOCHS=50 PAIRS_PATH=results/diagnostics/reliability_pair_runs_degree_gate_tiny.csv SUMMARY_PATH=results/diagnostics/reliability_pair_summary_degree_gate_tiny.csv bash scripts/run_small_reliability_study.sh
```

`scripts/run_small_reliability_study.sh` 已支持 `METHOD_CONFIG` 与 `METHOD_NAME` 环境变量；默认仍是 `configs/methods/rw_gcl_two_stage.yaml`，因此旧命令不受影响。

## 路线 B：机制诊断论文实验

目标：不继续堆方法模块，而是把贡献收缩为 reliability-weighted GCL 的机制分析与条件性方法。

最小补强内容：

- 实现 negative weighting，或将当前 label-based false-negative pressure 诊断扩展为 sampled-negative / denominator-level 机制证据。
- 深化 homophily non-degradation：解释 PubMed 轻微退化，必要时加入 homophily-specific safety gate 或报告为限制。
- 做 shuffled reliability 与 random reliability 的双 control。
- 增加 reliability bucket 的 label agreement / false positive positive-pair risk 诊断。
- 将 Wisconsin/Cornell/Squirrel 作为 honest negative results，而不是隐藏失败。

### 路线 B 成功标准

- 证明 reliability ranking 稳定对应 view consistency，并至少在核心正例上对应错误对比信号下降；
- 证明 shuffled/random reliability 不能复制机制诊断；
- homophily 数据集不显著退化；
- PubMed 退化能被诊断解释，或通过 conservative setting 缓解；
- performance claim 降级为 conditional improvement，而非 SOTA。

### 路线 B 停止标准

- label-based false negative / false positive 诊断不支持 reliability；
- homophily 数据集明显退化；
- shuffled/random reliability 在机制诊断上也成立；
- 机制诊断只能解释 view consistency，不能解释任何错误对比信号。

## 当前推荐

推荐先走路线 A 的最小 degree gate ablation，但只给它一次短窗口：

1. 先实现 degree-only gate。
2. 只跑 Texas/Chameleon/Squirrel/Actor × seeds 0-2。
3. 若不满足路线 A 成功标准，立即停止方法扩展，切换路线 B。

理由：当前 failure analysis 给出了一个非常具体、低成本、可证伪的假设；如果 degree gate 无效，就能快速证明“继续堆 gate”不值得。
