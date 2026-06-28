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

当前 active-but-risky candidate 是 `mpnv_gcl`。它使用 dense semantic/spatial multi-positive mask 和 Natural-View bootstrap，试图把 SSPNV 的单采样 positive 改成更接近 S3GCL/GraphECL 范式的 multi-positive contrastive objective。

当前已经停止的主线：

- `er_cache_gcl`；
- `er_residual_gcl`；
- `energy_spgcl`；
- `danv_gcl` / `danv_degree_gcl`；
- `fdnv_gcl`；
- `sspnv_gcl` / `afpnv_gcl` / `bspnv_gcl` 作为最终主方法。

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
- Chameleon/Squirrel × splits 0-9 × seed0 × 50 epoch 中，MPNV 相对 `gcn_mlp_gcl` 分别取得 +0.017105/+0.019132 与 +0.015082/+0.014767 的 F1Mi/F1Ma mean delta；
- Squirrel normal 10/10 split micro 正向，shuffled control 仅 +0.000961/+0.000668，是当前最干净的机制信号；
- Chameleon shuffled control 也为正，说明 Chameleon 只能作为性能正信号，不能作为机制证明；
- 当前裁决是 active-but-risky，不是成功方法；
- 下一步必须跑 seed1/seed2、Texas/Actor 扩展、homophily safety 和强基线对齐；
- 若 MPNV 的收益在 seed1/seed2 主要来自 shuffled control，或 Texas/Actor 明确退化且无法 label-free 回退，应停止 MPNV 主线。
