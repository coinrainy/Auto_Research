# DANV-GCL Ablation Decision

日期：2026-06-28

## 裁决

DANV-GCL 家族不再作为当前主方法推进。

保留结论：

- `gcn_mlp_gcl` 仍是 Natural-View GCL strong foundation；
- 固定全局 disagreement penalty 的 DANV 不稳定；
- `danv_degree_gcl` 作为失败/条件性消融资产保留，但不扩大到 splits 0/1/2；
- 下一阶段应换机制，而不是继续给 DANV 堆 gate。

## 外部边界核对

本轮快速核对了近期 GCL / heterophily 方向。相关趋势包括：

- simple / less-is-more heterophily GCL，而非复杂 augmentation 堆叠：[Less is More: Towards Simple Graph Contrastive Learning](https://openreview.net/forum?id=RvCkgg7pdt)；
- high-pass / low-pass graph filters for heterophily：[HLCL](https://proceedings.mlr.press/v244/yang24a.html)；
- structure-text / semantic alignment 在异配图上需要自适应而非静态目标：[GCL-OT](https://arxiv.org/html/2511.16778v2)；
- 2026 年异配图综述继续强调 graph heterophily 下的自适应表示与结构处理：[heterophily survey](https://arxiv.org/abs/2202.07082)。

这与本轮实验一致：固定全局 penalty 不像顶会主方法，后续应转向更清晰的 structure/semantic decoupling 或 filter-based adaptive objective。

## 实验 1：固定 penalty 消融

默认 DANV：`danv_disagreement_weight=0.1`
消融：`0.0` 与 `0.02`

### `w=0.0`

命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl danv_gcl" \
SPLITS="0 1 2" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_danv_ablation_w0_s0_splits0-2_e50" \
EXTRA_ARGS="--danv-disagreement-weight 0.0" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | --- | --- |
| Texas | +0.009009 | +0.044501 | 1/1 | mean 正向但 split 不稳 |
| Actor | +0.003947 | +0.006653 | 2/1 | 较默认更稳但不强 |
| Chameleon | +0.008041 | +0.009381 | 2/0 | 正向 |
| Squirrel | +0.006084 | +0.013229 | 1/2 | mean 正向但 split 不稳 |

### `w=0.02`

命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl danv_gcl" \
SPLITS="0 1 2" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_danv_ablation_w002_s0_splits0-2_e50" \
EXTRA_ARGS="--danv-disagreement-weight 0.02" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | --- | --- |
| Texas | +0.018018 | +0.019142 | 2/1 | 比默认 macro 安全，但仍不稳 |
| Actor | -0.001754 | +0.003021 | 1/1 | micro 失败 |
| Chameleon | -0.000731 | +0.002283 | 1/2 | micro 失败 |
| Squirrel | +0.006724 | +0.004984 | 3/0 | 正向 |

解释：

- 关掉 penalty 会改善 Actor/Texas 的部分风险，但伤害 Squirrel split stability；
- 小 penalty 可以救 Squirrel，但 Actor/Chameleon 变弱；
- 固定全局 weight 没有稳定区间，不值得继续细调。

## 实验 2：degree-aware disagreement gate

新增方法入口：`--method danv_degree_gcl`

设计：

- 保留 DANV alignment gate；
- disagreement loss 的节点权重乘上 incident-degree gate；
- 默认 `danv_degree_threshold=2.5`，`danv_degree_temperature=1.0`；
- 目标是在稀疏图上抑制 decorrelation，在密集异配邻域上保留更多 disagreement signal。

Smoke 诊断符合预期：

- Texas `danv_degree_gate_mean=0.235104`；
- Squirrel `danv_degree_gate_mean=0.615749`。

split0 early gate 命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl danv_degree_gcl" \
SPLITS="0" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_danv_degree_s0_split0_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | ---: | ---: | --- |
| Texas | 0.000000 | -0.003734 | 安全但无增益 |
| Actor | +0.005921 | +0.013567 | 小幅正向 |
| Chameleon | 0.000000 | -0.000556 | 无增益 |
| Squirrel | -0.002882 | +0.004420 | micro 失败 |

裁决：`danv_degree_gcl` 没有通过 split0 early gate，不扩大。

## 下一步

放弃 DANV 家族主线后，保留两个可用资产：

1. `gcn_mlp_gcl`：Natural-View strong foundation；
2. DANV ablation 证据：说明“何时对齐/保留分歧”是合理问题，但固定 penalty / degree gate 不够。

下一代 idea 应优先满足：

- 不依赖固定全局 penalty；
- 直接围绕 structure/semantic decoupling 或 graph filter objective；
- 从第一版就以 GCN-MLP 为 strong control，而不是只打 GRACE。
