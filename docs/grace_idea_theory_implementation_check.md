# GRACE Idea 理论-实现对齐检查

日期：2026-06-27

## 当前最小理论预设

当前 `experiments/grace_idea/` 不再实现早期 combined reliability 版本，而是收缩为更可审计的最小假设：

> 在 GRACE 的无监督节点表示学习中，EMA teacher 与增强视图 student encoder embedding 的一致性可以作为节点级 reliability；将该 reliability 用作正样本 anchor 权重，可能降低不稳定节点对训练目标的影响。

这一版只检验 `embedding_stability_only`。它不再把 projection head 维度上的 softmax 一致性作为分类语义预测一致性，也不声称已经解决 false negative / hard negative imbalance。

## 逐项对齐结果

| 理论/协议项 | 当前实现 | 对齐状态 |
|---|---|---|
| 基线来源 | `experiments/grace_idea/` 由 `baselines/GRACE` 复制而来，`baselines/GRACE` 不动 | 对齐 |
| reliability 主信号 | `embedding_stability_weights()` 使用 EMA teacher 原图 encoder embedding 与两个增强视图 student encoder embedding 的 cosine similarity | 对齐 |
| projection prediction consistency | 当前实现未使用 projection head softmax 一致性 | 有意移除 |
| stop-gradient reliability | 权重通过 `detach()` 进入 loss，不反传权重估计路径 | 对齐 |
| positive anchor weighting | 默认把 reliability 乘到每个节点的正样本对损失上，并按权重和归一化 | 对齐 |
| negative denominator weighting | 只有显式 `--negative-weighting` 时才启用 candidate weighting | 默认未启用，不能声称解决 false negative |
| 分布保持随机对照 | `--shuffle-weights` 对同一轮 raw reliability 做随机置换，保留分布、打乱节点对应 | 对齐，是主 control |
| uniform random 对照 | `--random-weights` 生成 `[min_weight, 1]` 均匀随机权重 | 不是主 control，只是宽分布正则化压力测试 |
| split 协议 | `--split-index` 选择 PyG 二维 mask；异配数据集默认 mask eval | 对齐 |
| seed 协议 | `--seed` 控制模型/增强随机性；`--split-index` 与 model seed 分开记录 | 对齐 |
| run 可追踪性 | `metadata.json` 记录 args、git commit/status、submodule status、split、seed、weight control | 对齐 |
| 权重强度诊断 | `train_log.csv` 记录 weight mean/std/min/max 与 effective sample size ratio | 对齐 |

## 对旧结果的解释修正

此前 early stop 中的 `random control` 应重新命名为 `uniform_random control`。它的权重均值约 0.52、标准差约 0.27，而 normal reliability 的权重均值约 0.97、标准差约 0.015，两者分布差异很大。

因此：

- `normal > shuffled` 可以支持“reliability 与节点对应关系可能有信息”，因为 `shuffled` 保留了权重分布。
- `uniform_random > normal` 不能直接反证 reliability-node 对应关系，因为它同时改变了权重强度和分布形状。
- `uniform_random` 更适合解释为“强 anchor reweighting / regularization stress test”，不是论文主消融里的 distribution-matched random reliability。

## 后续实验边界

下一轮验证应先跑小范围 sanity，而不是扩大矩阵：

1. 固定 `Texas split_index=0, model_seed=0`，复跑 GRACE、ES normal、ES shuffled、ES uniform_random。
2. 主判断只比较 `normal - shuffled` 与 `normal - GRACE`。
3. `normal - uniform_random` 只用于判断宽分布随机正则化是否仍能反超，不作为 reliability 机制的主证据；解释时必须同时看 weight std 与 effective sample size ratio。
4. 若需要检验 false negative 机制，必须单独启用 `--negative-weighting` 并增加 label-based false-negative pressure 诊断。
