# GRACE Idea 工作副本说明

本目录由 `baselines/GRACE` 复制而来，用于承载当前研究 idea 的代码改动。

- 来源 baseline：`baselines/GRACE`
- 来源版本：`b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- 协作约束：不要直接修改 `baselines/GRACE`；所有方法改动写在本目录。

当前路线调整：

- 删除此前自建的 RW-GCL scaffold 与批跑脚本。
- 以后以 GRACE 原实现为基础做最小侵入式修改。
- 优先保持 baseline 逻辑可对照，再逐步加入 reliability / weighting / diagnostics。

## 当前已实现的第一版改进

训练入口仍为 `train.py`，但新增 `--method` 参数：

- `--method grace`：保留原始 GRACE 训练路径。
- `--method es_weighted`：新增 embedding-stability weighted GRACE。

`es_weighted` 的设计边界：

- reliability 只来自 encoder embedding stability，不再使用 projection head softmax distribution consistency。
- 通过 EMA teacher 在原图上产生稳定参照 embedding。
- warm-up 后，对两个增强视图下的 student embedding 与 EMA teacher embedding 做余弦相似度，得到节点级 reliability。
- 默认将 reliability 用作 positive anchor weighting。
- 可通过 `--negative-weighting` 将 reliability 同时用于 InfoNCE denominator candidate weighting，低可靠节点作为负样本时贡献更小。
- reliability 只作为 stop-gradient 权重使用，不把权重估计路径反传回模型。

最小 smoke 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python train.py --dataset Cora --method grace --epochs 2 --skip-eval
python train.py --dataset Cora --method es_weighted --epochs 2 --warmup-epochs 1 --negative-weighting --skip-eval --save-dir runs/smoke
```

正式实验前仍需补齐：

- heterophily 数据集加载与 Geom-GCN split 支持；
- GRACE baseline 与 `es_weighted` 的同 split / 同 seed 对齐脚本；
- reliability 与 downstream error、degree、local homophily 的独立诊断；
- negative weighting 的 false-negative pressure 诊断。
