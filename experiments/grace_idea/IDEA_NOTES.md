# GRACE Idea 工作副本说明

本目录由 `baselines/GRACE` 复制而来，用于承载当前研究 idea 的代码改动。

- 来源 baseline：`baselines/GRACE`
- 来源版本：`b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- 协作约束：不要直接修改 `baselines/GRACE`；所有方法改动写在本目录。

当前路线调整：

- 删除此前自建的 RW-GCL scaffold 与批跑脚本。
- 以后以 GRACE 原实现为基础做最小侵入式修改。
- 优先保持 baseline 逻辑可对照，再逐步加入 reliability / weighting / diagnostics。

