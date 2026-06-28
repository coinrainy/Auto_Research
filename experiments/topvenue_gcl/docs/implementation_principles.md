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

## 当前新候选：Neighbor-Cache Distillation GCL

工作假设：

训练期使用图结构和 semantic/spatial neighbor cache 构造可靠 positives，将结构 encoder 的信号蒸馏到 MLP encoder；推理期只使用 MLP，从而同时满足 heterophily robustness 与 fast inference。

第一版不要追求复杂模型，先实现：

- MLP online encoder；
- lightweight structure teacher；
- semantic/spatial neighbor cache；
- cache confidence / staleness weighting；
- InfoNCE 或 bootstrap loss；
- 10 split node classification evaluator。
