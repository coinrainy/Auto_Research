# 第三方 baseline 工作区

本目录用于本地克隆强 baseline 的官方实现，但默认不把第三方源码提交进本仓库。

当前本地克隆：

- `SPGCL`: <https://github.com/haonan3/SPGCL>
- `PolyGCL`: <https://github.com/ChenJY-Count/PolyGCL>

复现策略：

- SP-GCL 优先接入，因为官方代码可在当前 PyTorch/PyG 环境导入；
- PolyGCL 依赖旧版 DGL/PyG，先记录为后续对照，不作为第一优先级；
- HLCL 暂未找到明确官方 GitHub 实现，先作为文献/表格对照候选。

本地兼容注意：

- SP-GCL 官方代码使用旧 NumPy 写法 `np.int`，在 NumPy 1.24+ 需要替换为 `int`；
- SP-GCL 需要 Geom-GCN 风格 `.mat` 数据和 `splits/*.npy`，可用 `experiments/grace_idea/export_spgcl_geom_data.py` 从当前 PyG 缓存导出。
