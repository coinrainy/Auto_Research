# Topvenue GCL Early Gate Summary

日期：2026-06-28

## 结论

本轮新增 scaffold 跑通，但三个新方法候选均未达到继续扩大的门槛。

- `er_cache_gcl`：放弃。Texas split0 seed0 下 normal cache 低于 shuffled/self-only，positive cache 主张失败。
- `er_residual_gcl`：放弃主线。Actor 正向，但 Texas/Chameleon 明显负向，Squirrel 仅 micro 弱正且 macro 下降。
- `energy_spgcl`：放弃当前实现。Texas/Chameleon 均低于 GRACE。
- `gcn_mlp_gcl`：保留为强对照/baseline。Actor 正向，Texas 追平 GRACE，Chameleon micro 小正但 macro 下降，Squirrel 失败；不构成新主方法。

## 关键 early gate 数值

所有结果均为 split0 / seed0 / 20 epoch，仅用于早筛，不作为论文结论。

| Dataset | Method | F1Mi | F1Ma | 裁决 |
| --- | --- | ---: | ---: | --- |
| Texas | GRACE | 0.675676 | 0.344612 | baseline |
| Texas | ER-Residual-GCL | 0.513514 | 0.291795 | 失败 |
| Texas | GCN-MLP-GCL | 0.675676 | 0.334091 | 只追平 baseline |
| Texas | Energy-SPGCL | 0.594595 | 0.373864 | micro 失败 |
| Chameleon | GRACE | 0.407895 | 0.401071 | baseline |
| Chameleon | ER-Residual-GCL | 0.326754 | 0.322908 | 失败 |
| Chameleon | GCN-MLP-GCL | 0.412281 | 0.362119 | macro 失败 |
| Chameleon | Energy-SPGCL | 0.324561 | 0.319509 | 失败 |
| Squirrel | GRACE | 0.304515 | 0.299359 | baseline |
| Squirrel | ER-Residual-GCL | 0.321806 | 0.277833 | macro 失败 |
| Squirrel | GCN-MLP-GCL | 0.274736 | 0.264493 | 失败 |
| Actor | GRACE | 0.261842 | 0.187220 | baseline |
| Actor | ER-Residual-GCL | 0.301316 | 0.279931 | 条件性正向 |
| Actor | GCN-MLP-GCL | 0.332895 | 0.311951 | 正向但非通用 |

## 下一步裁决

不要继续在当前三个失败 loss 上做小参数调参。下一步应回到已验证强信号：

1. 以 SP-GCL / GraphACL / GraphECL / PolyGCL / S3GCL 为强基线门槛；
2. 将 SPARC residual calibration 从 post-hoc/patch 经验，改造成可复现的标准训练或标准 evaluation module；
3. 若无法从头复现 official SP-GCL 级 embedding quality，则不要声称新 SOTA，只记录为 negative result。

