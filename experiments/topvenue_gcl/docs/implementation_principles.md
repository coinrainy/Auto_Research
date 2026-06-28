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

## 当前新候选：Energy-SPGCL

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
