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

GCN-MLP Natural View GCL 只作为强对照；若 Energy-SPGCL 无法稳定超过它和 GRACE，也必须降级。

第一版不要追求复杂模型，先实现：

- MLP online encoder；
- lightweight graph teacher；
- high-energy residual sampled InfoNCE；
- raw low-pass / propagation-signature positive sampler；
- GCN-MLP bootstrap strong control；
- high-energy residual 与 low-pass positive cache 仅作消融；
- 10 split node classification evaluator。
