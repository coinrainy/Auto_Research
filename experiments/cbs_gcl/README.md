# CBS-GCL：Core-Boundary Self-Guided GCL

## 背景

本目录是 2026-06-29 文献重扫后的新候选方向，已放弃旧的 reliability/RAGC/RPGCL/CG-HPFS 路线。当前目标不是继续调旧模块，而是验证一个新的图对比学习假设：

> 同配或中高同配图中，核心节点适合邻域/扩散正样本 compactness；疑似边界节点不应被强行邻居对齐，而应保留 raw/residual/position 区分性。

## 当前文件

- `analyze_core_boundary.py`：阶段 0 诊断脚本，不训练模型，只计算无标签 `core_score` 并用标签做离线诊断。
- `runs/core_boundary_diagnostics_splits0-4/`：Cora/CiteSeer/PubMed × splits 0-4 的诊断结果。

## 阶段 0 诊断结果

协议：Cora/CiteSeer/PubMed，分层 `train:val:test = 1:1:8`，splits 0-4。主评估仍看 accuracy；标签只用于诊断 core 分桶是否对应真实错误/正样本质量。

| Dataset | Raw Acc | Prop1 Acc | Prop2 Acc | Prop+Residual Acc | core-degree corr | high-low topk agreement | high-low raw error | high-low prop2 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.647505 | 0.820055 | 0.842144 | 0.817560 | -0.021566 | +0.069906 | -0.136846 | -0.039711 |
| CiteSeer | 0.657873 | 0.715145 | 0.724915 | 0.713717 | -0.018767 | +0.003381 | -0.025397 | -0.003324 |
| PubMed | 0.844278 | 0.850238 | 0.855031 | 0.861244 | -0.006070 | +0.024817 | -0.006702 | -0.011509 |

解释：

- `core_score` 与 degree 的相关性接近 0，说明当前分数不是简单高 degree proxy。
- high-core 节点在三图上的 raw/prop2 test error 都低于 low-core，支持“核心节点更容易、边界节点更难”的诊断。
- top-k positive label agreement 在 Cora/PubMed 有正向差距，CiteSeer 很弱；CBS-GCL 的 positive expansion 在 CiteSeer 上风险较高。
- Propagation baseline 本身很强，尤其 Cora/PubMed；后续模型必须超过 PROP/Prop2，否则不应声称训练式 GCL 有贡献。

## 当前裁决

CBS-GCL **保留为新 active candidate**，但还不是论文级方法。

继续理由：

- 诊断没有失败：核心-边界 proxy 对错误率和正样本质量有稳定但不强的信号；
- degree 相关性低，避免了“只是高 degree 节点更容易”的最直接质疑；
- 文献空隙明确：现有 SPGCL 偏 feature-wise energy，CL-GCL 偏 coarsening，BLNN 偏邻居 positives，而 CBS-GCL 聚焦 node-wise core/boundary。

主要风险：

- CiteSeer 的 high-low positive agreement 只有 +0.003381，可能不足以支撑 neighbor positive expansion；
- Prop2/PROP 已经很强，训练式 CBS-GCL 若不能超过它，应放弃；
- 当前 core score 是 heuristic，需要后续 ablation：去 residual、去 neighbor consistency、去 degree residualization。

## 下一步

实现最小训练原型：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/cbs_gcl
python train.py --dataset Cora --split-index 0 --method cbs_gcl --epochs 50
```

训练原型只包含：

- deterministic propagation views；
- core weighted multi-positive contrastive loss；
- boundary residual preservation loss；
- CCA/VICReg-style anti-collapse；
- raw/PROP/GRACE-light 强 baseline 对齐。

停止标准：

- 若 CBS-GCL 在 Cora/CiteSeer/PubMed 中两个以上数据集低于 Prop2；
- 若 core loss 去掉后性能不降；
- 若 boundary residual loss 去掉后性能不降；
- 若 positive label agreement 诊断无法解释性能变化。
