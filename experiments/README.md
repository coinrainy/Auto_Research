# 实验工作区

本目录用于放置从 baseline 复制出来或从头搭建的可改动实验代码。

- `../baselines/GRACE` 是只读 baseline 来源，不在其中直接修改 idea。
- `grace_idea/` 是从 `../baselines/GRACE` 复制出的历史工作副本，保留已完成的可靠性、raw-complement、SPARC 等候选与负结果资产。
- `topvenue_gcl/` 是新的顶会范式实验工作区，要求从第一版开始具备独立训练入口、数据加载、固定 split evaluator、baseline/control 与实验日志。

当前原则：

- 新主方法不要继续堆在 `grace_idea/` 中。
- 不再把 patch 第三方官方代码作为主论文实现方式。
- 候选 idea 必须先通过 small early gate；若 normal/control 不干净，立即降级或放弃。

