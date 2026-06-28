# 实现原则

## 必须满足

- 主方法代码必须在 `experiments/topvenue_gcl/src/` 内独立实现；
- 训练入口必须可通过 `scripts/*.sh` 或顶层 Python CLI 重现；
- 每个实验必须记录 dataset、split、seed、config、metric 与输出路径；
- linear evaluator 和训练逻辑必须使用同一套数据 split；
- 所有 early gate 都必须包含 random/shuffled 或 no-structure control。

## 禁止作为主线

- 只导出第三方模型 embedding 再做 post-hoc feature trick；
- 只在一个数据集或一个 split 上报结论；
- 只靠 dataset-specific 参数让结果成立；
- 只和弱 GRACE scaffold 比较；
- 继续在 `experiments/grace_idea/` 内堆叠新方法。

## 参考代码范式

- PolyGCL：spectral polynomial views + heterophily benchmark scripts；
- S3GCL：MLP inference + spectral biased views + semantic/spatial positives；
- GraphECL：fast inference + structure encoder / MLP encoder + heterophily scripts。

## 当前主线状态

当前 active foundation 是 `gcn_mlp_gcl`。它是必须击败的 strong control，但不是论文主贡献。

当前没有通过最终复核的 active main idea。当前 active-but-risky candidate 是 `rrnv_gcl`：它用 redundancy reduction 目标替换 Natural-View BYOL，在 Texas 强正、Chameleon 小正，但 Squirrel 均值负向。后续只能围绕 RRNV 的 safety / graph-type adaptation 与强基线同协议对齐推进，不能把它包装成已成功方法。

当前已经停止的主线：

- `er_cache_gcl`；
- `er_residual_gcl`；
- `energy_spgcl`；
- `danv_gcl` / `danv_degree_gcl`；
- `fdnv_gcl`；
- `sspnv_gcl` / `afpnv_gcl` / `bspnv_gcl` 作为最终主方法。
- `mpnv_gcl` 作为最终主方法。
- `aompnv_gcl` 作为最终主方法。
- `srgnv_gcl` 作为最终主方法。
- `dsp_gcl` 作为最终主方法。
- `darrnv_gcl` 作为最终主方法。

后续所有新候选必须同时报告：

- vs GRACE；
- vs `gcn_mlp_gcl`；
- vs shuffled/random/no-structure control；
- 失败数据集和停止条件。

## 历史候选：Energy-SPGCL 与 Natural-View foundation

工作假设：

用 raw low-pass / propagation signature 构造 positive sampler，但在 high-energy propagation residual 表示上执行 sampled InfoNCE；GCN-MLP Natural View GCL 是必须击败的强对照。

low-pass positive cache 在 Texas split0 seed0 early gate 中 normal 低于 shuffled，当前已降级为失败消融，不作为主贡献。

high-energy residual bootstrap 在 Texas/Chameleon 失败，只在 Actor 清楚正向，当前也降级为失败/条件性消融。

GCN-MLP Natural View GCL 已在 Texas/Actor/Chameleon/Squirrel × splits 0/1/2 × seed0 × 50 epoch 中相对 GRACE 全正向，因此升级为下一阶段 strong control / architecture foundation。它本身仍不够创新；后续方法必须稳定超过它。

第一版不要追求复杂模型，先实现：

- MLP online encoder；
- lightweight graph teacher；
- high-energy residual sampled InfoNCE；
- raw low-pass / propagation-signature positive sampler；
- GCN-MLP bootstrap strong control；
- high-energy residual 与 low-pass positive cache 仅作消融；
- 10 split node classification evaluator。

下一步强制要求：

- 新模块必须同时报告 vs GRACE 与 vs GCN-MLP；
- 若只超过 GRACE 而不超过 GCN-MLP，不保留为 active candidate；
- 若只在 WebKB/Actor 正向，需要明确边界，不能包装成通用 heterophily SOTA。

## RRNV 当前裁决

- `rrnv_gcl` 入口为 `--method rrnv_gcl`；
- 它在 MLP ego view 与 GCN graph view 的 predictor 输出上执行 VICReg/CCA 风格 redundancy reduction；
- `--rrnv-shuffle-pairs` 是必须保留的机制 control；
- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 mean ΔF1Mi 分别为 +0.099099、+0.002412、+0.008041、-0.008325；
- normal-vs-shuffled mean ΔF1Mi 分别为 Texas +0.054054、Actor +0.006360、Chameleon +0.007310、Squirrel +0.002241；
- 当前解释：Texas 是最强证据，Chameleon 有小正与较干净 control，Actor 是弱辅助，Squirrel 是明确风险；
- RRNV 升级为 active-but-risky candidate，但不是最终成功方法；
- 后续必须优先解决 Squirrel safety，并补强基线同协议对齐。

## DSP / DARRNV 停止记录

- `dsp_gcl` 用 downstream separability proxy 进行节点级 loss weighting；split0 seed0 中 Texas 被 shuffled 反超，Chameleon/Squirrel 低于 baseline，已降级；
- `darrnv_gcl` 用图平均度 gate 将 RRNV 作为 BYOL 的小辅助项；Texas 丢失 RRNV 主信号、Actor 低于 baseline，已中止；
- 后续不再继续调 DSP weight temperature、DARRNV degree threshold 或 auxiliary weight。

当前下一代方法假设：

- 名称：Disagreement-Aware Natural-View GCL (DANV-GCL)；
- 底座：GCN-MLP natural views；
- 核心：对一致节点做 alignment，对冲突节点保留/去相关互补信息；
- gate：raw-neighbor agreement、view cosine、feature propagation residual energy；
- 控制：必须超过 `gcn_mlp_gcl`，否则放弃。

DANV splits 0/1/2 early gate：

- Texas/Chameleon/Squirrel 的 mean micro 超过 GCN-MLP，Actor mean micro 仅小幅超过但 3 个 split 中 2 个为负；
- Chameleon/Squirrel 共 6 个 split micro 全正，macro 也为正，是当前最干净的 DANV 信号；
- Texas mean micro 正向但 macro 下降，split0 macro 退化尤其明显；
- 当前状态仍是 active-but-risky，不是成功方法；
- 下一步只允许做 gate/penalty 消融与 safety fallback：`danv_disagreement_weight=0`、`0.02`、gate temperature、min-align；
- 若消融不能缓解 Texas macro 与 Actor split instability，应放弃 DANV 主线或收缩为 WikipediaNetwork 条件性机制。

DANV 消融裁决：

- `danv_disagreement_weight=0.0`、`0.02` 和 `danv_degree_gcl` 均未形成稳定主方法；
- 固定全局 disagreement penalty 没有稳定有效区间；
- degree-aware disagreement gate 在 split0 上未超过 early gate；
- DANV 家族降级为失败/条件性消融资产，不再作为当前 active idea；
- 下一代方法仍可继承 `gcn_mlp_gcl` natural-view foundation，但必须换机制。

FDNV 第一版裁决：

- `fdnv_gcl` 显式学习 low/high filtered targets，比 DANV penalty 更贴近 S3GCL/PolyGCL 的 filter 方向；
- `fdnv_route_weight=0.5` 与 `0.1` 都在 Chameleon split0 上失败；
- FDNV 第一版不进入 splits 0/1/2，不作为 active main idea；
- 后续若继续 filter 方向，必须改 objective，而不是继续调 route weight；
- 下一步优先考虑 semantic/spatial positive split 或 high/low branch complementarity。

SSPNV / AFPNV / BSPNV 当前裁决：

- `sspnv_gcl` 的固定完整版本保留为机制原型，但不再作为最终主方法包装，入口为 `--method sspnv_gcl`；
- 方法使用 semantic positives 监督 high-pass target，spatial positives 监督 low-pass target，并保留 GCN-MLP Natural-View bootstrap；
- 10 split / seed0 / 50 epoch 下相对 `gcn_mlp_gcl` 在 Texas、Actor、Chameleon、Squirrel 的 mean micro/macro 均为正；
- Chameleon 10/10 split micro 正向，Squirrel 9/10 split micro 正向，是当前最强实验信号；
- Actor 仅弱正且 4/10 split 为负，必须作为边界而非主成功证据；
- component ablation 与 random-positive control 已完成 Chameleon/Squirrel 10 split；
- Chameleon 上 semantic-only、spatial-only 与 random semantic 均接近或超过完整 SSPNV，说明固定 semantic-spatial 双分支同权相加不是必要机制；
- Squirrel 上 random semantic / random spatial 失败，说明结构化 positives 有条件性价值，但 full SSPNV 增益仍偏小；
- `afpnv_gcl` 已实现为置信度加权版本，入口为 `--method afpnv_gcl`；它在 Chameleon/Squirrel 上均未超过对应最强 SSPNV 变体，当前只保留为 ablation；
- `bspnv_gcl` 已实现为 semantic / spatial / bootstrap branch selection，入口为 `--method bspnv_gcl`；它强于 AFPNV，但仍未同时超过 Chameleon semantic-only 与 Squirrel full SSPNV；
- SSPNV 家族已触发停止条件，全部降级为失败/条件性消融资产；
- 不再继续调 SSPNV/AFPNV/BSPNV 的 threshold、temperature、branch bias；
- 下一代 active idea 必须换训练目标或参考范式，继续保留 `gcn_mlp_gcl` 作为 strong foundation，但不要再把 filter-specific positives 的小变体包装成主线。

MPNV 当前裁决：

- `mpnv_gcl` 入口为 `--method mpnv_gcl`；
- 它不再做单一 semantic/spatial positive 采样，而是构造 dense semantic mask 与 dense spatial mask；
- semantic mask 来源于 raw propagation signature KNN，监督 high-pass target；
- spatial mask 来源于原图一跳邻居，监督 low-pass target；
- 保留 `gcn_mlp_gcl` 的 Natural-View bootstrap；
- `--mpnv-shuffle-positives` 是必须保留的机制 control，用于打乱 positive mask 与节点对应关系；
- Chameleon/Squirrel × splits 0-9 × seed0 × 50 epoch 中，MPNV 相对 `gcn_mlp_gcl` 曾分别取得 +0.017105/+0.019132 与 +0.015082/+0.014767 的 F1Mi/F1Ma mean delta；
- 但 Texas/Actor/Chameleon/Squirrel × splits 0-9 × seeds 1/2 × 50 epoch 复核失败：ΔF1Mi 分别为 +0.002703、-0.001776、+0.000219、-0.000288；
- Squirrel seed1/seed2 不再保持 normal 10/10 split micro 正向，seed0 核心机制证据失效；
- 当前裁决是降级为失败/条件性消融资产，不再作为 active main idea；
- 后续不再继续跑 MPNV shuffled seed1/2，因为 normal gate 已失败；
- 下一代方法必须加入 label-free objective activation / node-level fallback，或彻底更换训练目标。

AOMPNV 当前裁决：

- `aompnv_gcl` 入口为 `--method aompnv_gcl`；
- 它复用 MPNV 的 dense semantic/spatial mask，但不再固定相加 semantic/spatial/bootstrap loss；
- 每个节点根据 semantic dense InfoNCE、spatial dense InfoNCE、bootstrap negative cosine 的相对自监督 loss，以及 raw-signature positive confidence，路由到 semantic / spatial / bootstrap objective；
- `--aompnv-shuffle-positives` 是必须保留的机制 control；
- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 1/2 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.022523、+0.001864、+0.015351、+0.018892；
- normal-vs-shuffled 的 ΔF1Mi 分别为 Texas -0.000000、Actor +0.009539、Chameleon +0.006579、Squirrel +0.003522；
- 已完成 splits 0-9 × seeds 1/2 的 normal/shuffled 硬门控；
- 相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 Texas +0.010811、Actor -0.002829、Chameleon +0.000658、Squirrel +0.018348；
- normal-vs-shuffled 的 ΔF1Mi 分别为 Texas +0.008108、Actor +0.007763、Chameleon -0.011294、Squirrel +0.013593；
- 当前解释：只有 Squirrel 同时给出较清楚的 baseline 增益与 shuffled control 差距；Texas 正向但 split-level 不稳，Actor 低于 baseline，Chameleon shuffled 明显更强；
- AOMPNV 已降级为 regularization / negative-result ablation，不再作为主线；
- 后续不再继续调 AOMPNV 的 router temperature、branch weight 或 confidence threshold；
- 下一代方法必须换训练机制，同时保留 `gcn_mlp_gcl` strong foundation 与 shuffled/random/no-structure control。

SRGNV 当前裁决：

- `srgnv_gcl` 入口为 `--method srgnv_gcl`；
- 它尝试将 graph view 分解为 ego/feature 可解释部分与 structure residual，并用 raw feature propagation residual score 做节点级 gate；
- `--srgnv-shuffle-residual` 是必须保留的 no-structure control；
- Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.000000、+0.001974、-0.041667、-0.005764；
- shuffled residual 的 ΔF1Mi 分别为 -0.027027、+0.002632、-0.021930、-0.028818；
- 当前解释：SRGNV 能优化 residual cosine，但下游分类不稳定受益；Actor 唯一正向还被 shuffled control 超过，Chameleon/Squirrel 明显失败；
- SRGNV 已降级为 negative-result ablation，不进入 splits 0-2 扩展；
- 后续不再继续调 residual weight、threshold 或 temperature；
- 下一代方法必须直接约束或诊断 downstream separability / neighborhood conflict，而不是只蒸馏 representation residual。

PCNV 当前裁决：

- `pcnv_gcl` 入口为 `--method pcnv_gcl`；
- 它保留 `gcn_mlp_gcl` 的 Natural-View bootstrap，并加入 trainable prototypes；
- ego view 与 graph view 在 prototype assignment 空间做双向 consistency，target stop-gradient；
- `--pcnv-shuffle-assignments` 是必须保留的机制 control；
- default PCNV 在 Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.027027、+0.007895、+0.028509、-0.014409；
- default shuffled 的 ΔF1Mi 分别为 +0.000000、+0.013158、+0.030702、-0.016330，说明 Actor/Chameleon 机制 control 不干净；
- sharpened PCNV 使用 `--pcnv-prototype-weight 0.5 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.1 --pcnv-target-temperature 0.03`；
- sharpened PCNV 在 Texas 达到 F1Mi/F1Ma=0.729730/0.459091，强于 default 与 shuffled，是当前最强 Texas 单点；
- 但 sharpened PCNV 在 Squirrel 明显失败，且 Chameleon/Squirrel 的 prototype usage entropy 过低，提示原型坍塌；
- 已新增 entropy-guarded / confidence-weighted PCNV 与 view-agreement gated PCNV；
- sharp guarded PCNV 在 Texas 达到强正向，但 Actor/Squirrel 失败，Chameleon/Squirrel usage entropy 明显过低；
- soft guarded PCNV 是最健康变体：Texas macro 与 Actor 正向，但 Chameleon shuffled control 反超，Squirrel 仍失败；
- view-agreement gated PCNV 在 Texas/Actor/Chameleon/Squirrel split0 seed0 上全面弱于 soft guarded，并使 Squirrel 明显退化；
- 当前裁决：PCNV 已降级为 conditional / diagnostic asset，不再作为 active main idea；
- 后续不再继续调 PCNV 的 temperature、confidence threshold、entropy guard 或 view-agreement gate；
- 下一代方法必须换机制，优先考虑节点级局部结构条件下的 objective selection 或更直接的 downstream separability 代理。

LCOS 当前裁决：

- `lcos_gcl` 入口为 `--method lcos_gcl`；
- 它使用 raw feature local conflict gate 在完整 graph view alignment 与 high-pass view alignment 之间做节点级 objective selection；
- `--lcos-shuffle-gate` 是必须保留的机制 control；
- 已修复 raw residual 数值问题：使用 `||x - P x|| / (||x|| + ||P x||)`，避免零特征节点导致 residual 爆炸；
- Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 -0.054054、+0.009211、-0.004386、+0.013449；
- normal-vs-shuffled 的 ΔF1Mi 分别为 +0.000000、+0.001974、+0.015351、+0.050913；
- 当前解释：Squirrel 给出最清楚的局部冲突 gate 机制线索，但 Texas micro 和 Chameleon baseline gate 失败；
- LCOS 第一版已降级为失败/条件性诊断资产，不进入 splits 0-2 扩展；
- 后续不再继续调 LCOS route threshold、temperature 或 degree weight；
- 已实现 `lcm_gcl` 作为 LCOS 的 final-only 后续变体：训练目标保持 `gcn_mlp_gcl`，只在 final representation 中使用 local-conflict graph/high mix；
- LCM 在 Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 -0.054054、+0.004605、+0.006579、+0.000961；
- LCM normal-vs-shuffled 的 ΔF1Mi 分别为 -0.135135、+0.015132、+0.024123、+0.001921，Texas shuffled 明显更强；
- LCM 也已降级，不进入 splits 0-2 扩展；
- 若继承 LCOS/LCM 线索，必须改变目标设计：局部冲突 gate 应用于 loss reliability、negative suppression 或 downstream separability proxy，而不是直接对齐 high-pass target，也不是简单 graph/high final mix。
