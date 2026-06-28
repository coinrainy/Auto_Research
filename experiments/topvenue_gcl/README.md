# Top-Venue GCL Experiment Scaffold

本目录用于重启 Graph Contrastive Learning 研究主线。

核心原则：

- 不再依赖 patch third-party official code 作为主方法实现；
- 新 idea 必须拥有独立训练入口、模型定义、配置、脚本与 evaluator；
- 实验协议优先对齐近年顶会代码范式，如 PolyGCL、S3GCL、GraphECL；
- `runs/` 仅保存本地实验输出，不提交到 git；
- 第三方代码只放在 `../../third_party_baselines/reference_gcl/` 作为参考。

当前推荐候选方向：

1. Inference-efficient heterophily GCL with neighbor-cache distillation；
2. Distribution-aware positive pair construction；
3. Spectrum-conditioned augmentation-free GCL。

第一阶段任务：

1. 固化 dataset loader 与 10 split evaluator；
2. 复现一个轻量 MLP-inference GCL baseline；
3. 实现 neighbor-cache / semantic-positive reliability 原型；
4. 在 heterophily + homophily safety 数据集上做小规模 gate。
