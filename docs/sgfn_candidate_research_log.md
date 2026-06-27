# SGFN 候选方法研究日志

日期：2026-06-28（UTC）

## 当前候选 idea

候选名称：SGFN（Stability-Guided False-Negative attenuation for Graph Contrastive Learning）

核心问题：在 GRACE/InfoNCE 式节点级图对比学习中，所有非自身节点都会进入 negative denominator；其中同类节点会被错误推开，形成 false-negative contamination。已有 hard-negative / false-negative mining 工作很多，因此本项目不应只声称“发现假负样本”，而要强调：

- 使用 EMA teacher 在 clean graph 上形成稳定参照；
- 对每个 anchor 做 row-standardized teacher similarity，得到 pair-level false-negative risk；
- 在 InfoNCE denominator 中对疑似 false negative 做有界 attenuation；
- 用 shuffled-pair control 和 label-only false-negative pressure 诊断证明不是简单弱化负样本或随机重加权。

## 文献边界

当前快速核查到的强相关工作包括：

- ProGCL：指出图上 hard negatives 往往是 false negatives，并估计 negative 为 true negative 的概率。
- FD4GCL：使用 attribute / structure-aware 方式检测 false negative。
- GraphRank：通过 ranking objective 规避 InfoNCE 把同类节点当 negative 的问题。
- GRAPE：从 subspace preserving 角度做 expansive/adaptive hard negative mining。
- HLCL / AS-GCL：从 heterophily 与 spectral view 方向改造 GCL。

因此 SGFN 的潜在新意目前只能保守表述为：轻量 teacher-stability pair risk + pair-denominator attenuation + 可证伪机制诊断；不能表述为“首次解决 false negative”或“通用 heterophily SOTA”。

## 已实现内容

代码位置：`experiments/grace_idea/`

- `train.py`
  - 新增 `--method sgfn`。
  - 新增 pair-level denominator weighting：`pair_denominator_weights`。
  - 新增 false-negative risk 参数：
    - `--fn-risk-margin`
    - `--fn-risk-temperature`
    - `--fn-attenuation-power`
    - `--fn-consensus none|feature`
  - 新增控制组：
    - `--shuffle-weights`：打乱 pair weight 映射。
    - `--random-weights`：uniform random pressure test。
    - `--pair-shuffle-mode column|row`。
  - 新增实验性扩展：
    - `--pair-normalization row_mean|blend_row_mean`：denominator mass reallocation 消融。
    - `--fn-attraction-weight`：疑似 false negative attraction 消融。
- `model.py`
  - InfoNCE 支持 pair-specific denominator weights，含 full-batch 与 batched 路径。
- `summarize_runs.py`
  - 支持 `--target-method sgfn`，可汇总 SGFN vs GRACE 与 normal-vs-control。
- `analyze_pair_weights.py`
  - 新增 label-only 机制诊断，计算同标签 false-negative keep weight、异标签 true-negative keep weight、weighted/unweighted false-negative pressure share，并输出 aggregate 与 normal-vs-control 对照。
- `scripts/run_split_study.sh`
  - 支持 `sgfn` 批跑。

## 小规模结果

所有结果仅为 sanity，不是正式论文结果。设置均为 Texas/Cornell/Wisconsin/Actor × splits 0/1/2 × model seed 0 × 50 epochs。

### SGFN attenuation（当前主候选）

命令核心：`METHODS="grace sgfn" ES_CONTROLS="normal shuffled" EPOCHS=50 WARMUP_EPOCHS=20`

SGFN normal - GRACE（F1Mi mean）：

- Texas：+0.027027
- Cornell：+0.009009
- Wisconsin：-0.013072
- Actor：-0.002412

SGFN normal - shuffled（F1Mi mean）：

- Texas：+0.018018
- Cornell：+0.009009
- Wisconsin：-0.013072
- Actor：+0.001316

机制诊断（weighted - unweighted FN pressure，全图 mean）：

- Texas normal：-0.016206；shuffled：约 -0.000233
- Cornell normal：-0.008706；shuffled：约 -0.000138
- Wisconsin normal：-0.015739；shuffled：约 +0.000146
- Actor normal：约 -0.000385；shuffled：约 -0.000018

判断：SGFN attenuation 是当前唯一同时具备一定性能苗头与机制诊断信号的候选，但还远不到 SOTA。

### row-mean reallocation

命令核心：`TRAIN_EXTRA_ARGS="--pair-normalization row_mean"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.009009
- Cornell：+0.018018
- Wisconsin：0.000000
- Actor：-0.004825

判断：机制更干净，权重均值固定为 1，但整体性能不优；适合作为审稿人质疑“只是弱化 denominator”的消融，不适合作为默认主方法。

### blend reallocation

命令核心：`TRAIN_EXTRA_ARGS="--pair-normalization blend_row_mean --pair-reallocation-alpha 0.5"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.027027
- Cornell：-0.018018
- Wisconsin：0.000000
- Actor：-0.007237

判断：失败消融；不应继续扩大。

### attraction

命令核心：`TRAIN_EXTRA_ARGS="--fn-attraction-weight 0.1"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.027027
- Cornell：-0.009009
- Wisconsin：-0.006536
- Actor：-0.008553

判断：疑似 false negative attraction 不稳定，且 shuffled 在部分 split 更强；不应作为主线。

## 当前决策

2026-06-28 追加 10-split 验证后，当前决策更新为：默认 SGFN attenuation 不能作为最终主方法继续推进。它保留为机制诊断原型和后续方法组件，而不是论文主 idea。

### 10-split 复核：默认 SGFN attenuation

设置：Texas/Cornell/Wisconsin/Actor × splits 0-9 × seed0 × 100 epochs；normal 与 shuffled 成对。

输出文件：

- `runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_aggregate.csv`
- `runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_pair_weights_aggregate.csv`
- `runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_pair_weights_controls.csv`

SGFN normal - GRACE：

- Texas：F1Mi +0.027027，F1Ma +0.020583；8/10 split micro 正向，0/10 负向。
- Cornell：F1Mi +0.005405，F1Ma +0.009356；但 normal - shuffled 的 F1Mi 为 -0.013514。
- Wisconsin：F1Mi -0.011765，F1Ma +0.001649；5/10 split micro 负向。
- Actor：F1Mi +0.001053，F1Ma -0.003048；幅度接近噪声。

机制诊断仍然成立：normal 在 Texas/Cornell/Wisconsin 上稳定降低 label-only false-negative pressure，而 shuffled control 接近 0；Actor 机制信号很弱。

判断：

- 默认 SGFN 在 Texas 上有稳定正向，但跨数据集泛化不足。
- Cornell 出现 normal 不如 shuffled 的问题，削弱了 pair mapping 的因果解释。
- Wisconsin micro 退化，Actor 近零，说明当前 teacher-similarity risk 不足以支撑“兼具创新和 SOTA 能力”的主方法。
- 因此，按“当确定当前 idea 无法成功时应该学会放弃”的规则，默认 SGFN 不再作为最终 idea 扩展多 seed 或大规模 baseline。

### feature-consensus SGFN 筛选

动机：用 `--fn-consensus feature` 要求 teacher embedding risk 与原始特征相似性同时支持，减少 self-confirming teacher similarity 的误判。

设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs。

输出文件：

- `runs/summaries/sgfn_feature_consensus_hetero_splits0-2_seed0_e100_aggregate.csv`
- `runs/summaries/sgfn_feature_consensus_hetero_splits0-2_seed0_e100_pair_weights_aggregate.csv`

SGFN feature-consensus normal - GRACE：

- Texas：F1Mi +0.027027，F1Ma +0.055202。
- Cornell：F1Mi +0.018018，F1Ma -0.005436。
- Wisconsin：F1Mi -0.019608，F1Ma +0.008334。
- Actor：F1Mi -0.002193，F1Ma -0.007769。

normal - shuffled：

- Texas：F1Mi 0.000000，F1Ma +0.031376。
- Cornell：F1Mi +0.027027，F1Ma +0.003040。
- Wisconsin：F1Mi -0.019608，F1Ma +0.016112。
- Actor：F1Mi -0.002851，F1Ma -0.007860。

判断：

- feature-consensus 提升了 Texas/Cornell 的局部表现，并让 attenuation 更保守。
- 但它没有解决 Wisconsin micro 退化，也在 Actor 上低于 GRACE 与 shuffled。
- 因此 feature-consensus 也不能作为主方法，只能作为后续“风险触发条件”或 ablation。

### 当前放弃与保留

放弃：

- 放弃“纯 teacher 相似度即可估计 false-negative risk 并通用提升 GCL”的主张。
- 放弃把默认 SGFN 或 feature-consensus SGFN 直接包装成 2026 顶会/顶刊主方法。
- 暂停继续扩展 SGFN 到更多 seed 或更强 baseline，因为当前条件性失败已经足够明确。

保留：

- 保留 pair-level denominator attenuation 的代码资产。
- 保留 shuffled pair mapping control 与 label-only false-negative pressure 诊断。
- 保留 Texas/Cornell 的正向现象作为下一代方法的诊断靶点。

下一轮应转向：先判定节点/区域是否适合 false-negative attenuation，再局部启用 attenuation。候选名称可暂定为 `context-gated false-negative calibration`，其核心不应是“更强 attenuation”，而应是“何时不做 attenuation”。

### context-gated SGFN 筛选

实现更新：

- `train.py` 新增 `--fn-context-gate none|local_feature|degree_inverse|local_feature_degree`。
- `train.py` 新增 `--fn-context-pair-mode product|min|anchor`。
- `local_feature` 基于节点特征与邻居均值特征的一致性，试图识别局部语义可依赖区域。
- `degree_inverse` 基于反 degree confidence，试图避免高连接区域的过度 false-negative attenuation。
- gate 只作为 stop-gradient 风险调节项，不参与反向传播。

第一版 `local_feature_degree + product` 在 Texas split0 即 early stop：normal F1Mi 0.6216，GRACE/shuffled 为 0.6757，说明该 gate 的 pair 映射本身有害。

第二版 `degree_inverse + anchor` 完成 Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs。

输出文件：

- `runs/summaries/cg_sgfn_degree_anchor_splits0-2_seed0_e100_aggregate.csv`
- `runs/summaries/cg_sgfn_degree_anchor_splits0-2_seed0_e100_pair_weights_aggregate.csv`

CG-SGFN degree-anchor normal - GRACE：

- Texas：F1Mi +0.027027，F1Ma +0.024252；3/3 split micro 正向。
- Cornell：F1Mi +0.018018，F1Ma -0.000860；2/3 split micro 正向，1/3 持平。
- Wisconsin：F1Mi -0.039216，F1Ma +0.012694；3/3 split micro 负向。
- Actor：F1Mi -0.001096，F1Ma -0.000727；接近零且略负。

normal - shuffled：

- Texas：F1Mi +0.045045，F1Ma +0.020655。
- Cornell：F1Mi +0.009009，F1Ma +0.006257。
- Wisconsin：F1Mi -0.039216，F1Ma +0.013112。
- Actor：F1Mi -0.003289，F1Ma -0.005172。

判断：

- degree-anchor gate 相比前两版更干净地强化了 Texas，也让 Cornell 略正。
- 但它没有解决最关键的 Wisconsin 退化，且 Actor normal 仍低于 shuffled。
- 由于目标是 2026 顶会/顶刊级新方法，而不是单数据集/条件性技巧，context-gated SGFN 也应降级为失败筛选。

### 当前最终决策：停止 false-negative attenuation 主线

到目前为止，默认 SGFN、feature-consensus SGFN、local-feature-degree gate、degree-anchor gate 都不能同时满足“跨数据集性能提升 + normal 优于 shuffled + 非退化”的最低标准。

因此当前不再继续围绕 teacher-similarity false-negative attenuation 做小修小补。该主线的可复用资产是：

- pair-denominator weighting 实现；
- shuffled pair mapping control；
- label-only false-negative pressure 诊断；
- context gate 代码；
- 一组负结果，用于约束下一轮 idea 不要重蹈 heuristic pair weighting 的路径。

下一轮研究应重新构思，不再以“估计并削弱 false negative”作为默认核心机制。

## 下一步建议命令

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2" SEEDS="0" METHODS="grace sgfn" ES_CONTROLS="normal shuffled" EPOCHS=100 WARMUP_EPOCHS=20 SAVE_DIR="runs/next_context_gated_sgfn_sanity" MANIFEST_PATH="runs/next_context_gated_sgfn_sanity/run_manifest.csv" OVERWRITE=1 LOG_EVERY=100 TRAIN_EXTRA_ARGS="<next-gate-args>" scripts/run_split_study.sh
```

停止标准：

- 若下一代 context-gated 版本不能同时满足：Texas 不低于默认 SGFN、Wisconsin/Actor 不退化、normal 至少在 3/4 个数据集上不低于 shuffled，则继续放弃该 false-negative attenuation 主线，重新构思新的 GCL idea。

该停止标准已经被 degree-anchor gate 触发。建议下一步改换问题定义，而不是继续调参。
