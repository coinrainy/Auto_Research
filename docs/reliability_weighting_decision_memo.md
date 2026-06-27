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

### 2026-06-27 实现审查后的证据降级

用户对当前实现提出 5 个关键问题后，原先“立即进入路线 A degree gate”的推荐需要暂停，先修正实验定义和评估协议：

1. 当前 `prediction_consistency` 并不是分类预测一致性，而是 projection head 输出维度上的 softmax 分布一致性。它更准确的名字应是 `projection_distribution_consistency` 或 `projection_stability`，不能直接声称捕捉分类语义。
2. 当前 `view_consistency` 诊断存在循环验证：reliability 本身由 embedding stability 与 projection consistency 组成，再按 reliability 分桶统计这两个组成量，高低桶差异不能作为强独立机制证据。
3. 当前 loss 只做 positive anchor weighting，没有调整 negative denominator，因此还没有直接解决 false negative / hard negative imbalance。低 reliability 节点训练信号被削弱，但仍作为其他节点的负样本存在。
4. 当前 reliability 很可能偏向 degree 或 augmentation-stable 节点。Chameleon、Squirrel、Actor 的 high-reliability bucket 倾向高 degree，与 teacher-student stability 的定义一致，但高稳定不必然等于分类语义可靠。
5. 当前 “10 seeds” 主要是同一 split 下的模型初始化与增强随机性变化，不是标准 Geom-GCN 10 splits。论文级实验必须显式循环 `split_index=0..9`，并区分 model seed 与 split index。

因此，当前证据边界进一步收缩为：

> 已有结果只能说明当前 reliability 分数能稳定排序“投影/嵌入跨视图稳定性”，并在 Texas 上与 accuracy 与 false-negative pressure 下降同时出现；它尚不能证明该分数等价于分类语义可靠性，也尚未证明方法解决了 false negative / hard negative imbalance。

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

本备忘录原先推荐先走路线 A 的最小 degree gate ablation，但在实现审查后应调整为：先修正实验协议，再决定 A/B。立即做 degree gate 会把“degree bias 的修补”建立在尚未澄清的 reliability 定义上，风险过高。

### 修正后第一批 no-regret 实验

这些实验不涉及路线 A/B 选择，应优先完成：

- 已实现 split-aware runner：同时记录 `dataset`、`split_index`、`model_seed`，并把 `split_index` 写入 metrics 与汇总表。
- 已复跑小规模标准 split sanity：Texas、Chameleon、Squirrel、Actor × `split_index=0..2` × `model_seed=0`。
- 已增加独立诊断入口 `downstream_error`：在保存的 embeddings 上重新训练线性探针，输出 bucket-wise test accuracy/error 与 reliability-error correlation。
- 已修正命名：新结果使用 `projection_distribution_consistency`，旧 `prediction_consistency` 仅作为兼容 alias；后续论文中不得把它表述为分类预测一致性。
- 已增加 reliability component ablation 配置：`configs/methods/rw_gcl_embedding_only.yaml` 与 `configs/methods/rw_gcl_projection_only.yaml`。
- 仍暂缓 random reliability 与 degree gate；split-aware 复跑已经显示 combined 方法没有跨 split 稳定正向，component ablation 暂时更支持 embedding-stability-only。

建议的 split sanity 命令：

```bash
DATASETS="Texas Chameleon Squirrel Actor" SPLITS="0 1 2" SEEDS="0" WARMUP_EPOCHS=20 STAGE2_EPOCHS=50 EVAL_EPOCHS=50 PAIRS_PATH=results/diagnostics/reliability_pair_runs_split_sanity.csv SUMMARY_PATH=results/diagnostics/reliability_pair_summary_split_sanity.csv bash scripts/run_small_reliability_study.sh
```

对应 GRACE baseline：

```bash
DATASETS="Texas Chameleon Squirrel Actor" SPLITS="0 1 2" SEEDS="0" EPOCHS=70 EVAL_EPOCHS=50 RUNS_PATH=results/diagnostics/grace_runs_split_sanity.csv bash scripts/run_baseline_study.sh
```

对齐汇总：

```bash
python summarize_method_comparison.py --rw-summary results/diagnostics/reliability_pair_summary_split_sanity.csv --baseline-runs results/diagnostics/grace_runs_split_sanity.csv --baseline-method grace --out results/diagnostics/rw_gcl_vs_grace_split_sanity.csv --aggregate-out results/diagnostics/rw_gcl_vs_grace_aggregate_split_sanity.csv
```

### 2026-06-27 Split Sanity 结果

范围：Texas、Chameleon、Squirrel、Actor × `split_index=0,1,2` × `model_seed=0`。每个 split 跑 RW-GCL normal、RW-GCL shuffled 与 GRACE baseline。

| Dataset | combined normal - GRACE | combined normal - shuffled | Projection consistency gap | Interpretation |
|---|---:|---:|---:|---|
| Texas | -0.009009 | 0.000000 | 0.082731 | split 0 正向，但 split 2 负向，跨 split 不稳定 |
| Chameleon | -0.003655 | +0.000731 | 0.153936 | 原 10-seed 弱正向不跨 split 成立 |
| Squirrel | -0.006724 | +0.000961 | 0.205979 | projection gap 强，但 accuracy 负向 |
| Actor | +0.003290 | +0.000658 | 0.085416 | 很小正向，证据弱 |

关键解释：

- combined reliability 不支持继续声称 Texas/Chameleon 有稳定跨 split 正向效果。
- projection consistency gap 仍稳定为正，但它是 reliability 组成项，不是独立语义证据。
- 这次结果进一步支持用户此前指出的问题：projection consistency 能解释投影/视图稳定性，但不能稳定解释分类收益。

### Split-aware 机制诊断

| Dataset | FN pressure weighted - unweighted | High-low FN pressure gap | high-low degree gap | downstream high-low error |
|---|---:|---:|---:|---:|
| Texas | -0.001079 | -0.073571 | -1.081967 | +0.022619 |
| Chameleon | +0.000099 | +0.009936 | +4.668862 | -0.045605 |
| Squirrel | +0.000008 | +0.003066 | +2.294545 | -0.090182 |
| Actor | -0.000210 | -0.012763 | +1.171791 | +0.068593 |

解释边界：

- Texas 仍有 false-negative pressure 下降信号，但 downstream error 与 accuracy 不支持稳定正向。
- Chameleon/Squirrel 的 high reliability bucket downstream error 更低，但 false-negative pressure 不下降，说明下游错误改善与 InfoNCE false-negative 机制没有直接对应。
- Actor 的高 reliability bucket downstream error 更高，说明 reliability 不能被解释为通用“分类可靠性”。
- degree bias 仍存在于 Chameleon/Squirrel/Actor，但在 combined 方法不稳定时不应立即做 degree gate。

### Component Ablation 结果

| Variant | Texas | Chameleon | Squirrel | Actor | Summary |
|---|---:|---:|---:|---:|---|
| combined | -0.009009 | -0.003655 | -0.006724 | +0.003290 | 不稳定，不能作为主方法继续扩大 |
| embedding only | +0.009009 | 0.000000 | +0.002241 | +0.003728 | 当前小窗口里最干净，但效果很小 |
| projection only | -0.009009 | -0.005848 | -0.000320 | -0.001316 | 不支持作为 reliability 主信号 |

当前实验含义：

- `projection_distribution_consistency` 应从主 reliability 定义中移除或降级为辅助诊断，不宜作为主方法信号。
- 若继续方法路线，最小候选应改为 `embedding_stability_only`，而不是 combined reliability 或 degree gate。
- 由于 embedding-only 效果很小，下一步必须扩展到更多 splits / seeds 和 homophily safety 后才能判断是否值得作为方法论文推进。

### 仍需用户确认的分岔

完成上述修正后，分岔应更新为：

- A1：把主方法收缩为 `embedding_stability_only`，先跑 10 splits / 多 seed 与 homophily safety，确认微弱正向是否真实。
- B：放弃当前 positive reliability-weighted 方法主线，转为机制诊断/负结果论文路线，强调 projection stability 与分类语义可靠性错位。

旧 route A 建议保留为候选，但不再作为立即执行项：

1. 先实现 degree-only gate。
2. 只跑 Texas/Chameleon/Squirrel/Actor × seeds 0-2。
3. 若不满足路线 A 成功标准，立即停止方法扩展，切换路线 B。

理由：当前 failure analysis 给出了一个非常具体、低成本、可证伪的假设；如果 degree gate 无效，就能快速证明“继续堆 gate”不值得。

## 2026-06-27 代码路线调整

用户决定停止维护此前自建的 RW-GCL scaffold。旧代码框架已经从仓库中删除，后续方法实现不再基于根目录 `train.py` / `src/rwgcl/` / `configs/` / `scripts/` 体系推进。

新的实现约束：

- `baselines/GRACE` 作为外部 baseline submodule 保持不动；
- `experiments/grace_idea/` 是从 GRACE 复制出的工作副本；
- 后续 reliability、negative weighting、split-aware 评估等 idea 应在 `experiments/grace_idea/` 中以最小侵入方式实现；
- 旧实验结论仅作为研究判断记录保留，不再对应当前可运行代码入口。
