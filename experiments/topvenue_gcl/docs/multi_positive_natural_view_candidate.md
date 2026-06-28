# Multi-Positive Natural-View GCL 候选备忘录

日期：2026-06-29

## 当前裁决

`mpnv_gcl` 已降级为失败/条件性消融资产，不再作为 active main idea。

原因：seed0 在 Squirrel 的 10 split / 50 epoch gate 中曾出现强正信号，但 seed1/seed2 扩展门控没有复现稳定优势。MPNV 当前只能证明 dense multi-positive objective 是一个值得保留的 ablation/diagnostic component，不能支撑 2026 顶会/顶刊主方法。

## 方法定义

MPNV-GCL 的目标是把 S3GCL 的 dense semantic/spatial mask、GraphECL 的 MLP inference / graph teacher 思路，以及当前 `gcn_mlp_gcl` Natural-View foundation 合并成一个轻量可复现的训练目标。

核心组成：

- ego/MLP view 作为 online anchor；
- GCN/graph view 作为 natural graph target；
- raw propagation signature 的 KNN 构造 semantic multi-positive mask；
- 原图一跳邻居构造 spatial multi-positive mask；
- semantic mask 对齐 high-pass target；
- spatial mask 对齐 low-pass target；
- 保留 GCN-MLP bootstrap，避免目标只由 dense InfoNCE 主导。

训练入口：

```bash
python train.py --dataset Chameleon --method mpnv_gcl --epochs 50 --split-index 0 --seed 0
```

随机正样本对照：

```bash
python train.py --dataset Chameleon --method mpnv_gcl --epochs 50 --split-index 0 --seed 0 --mpnv-shuffle-positives
```

## 实现位置

- `train.py`：新增 `--method mpnv_gcl`、MPNV 配置覆盖、semantic/spatial dense positive mask 构造与训练分支；
- `src/losses.py`：新增 `multi_positive_info_nce`；
- `configs/default.yaml`：新增 `mpnv_semantic_weight`、`mpnv_spatial_weight`、`mpnv_bootstrap_weight`、`mpnv_include_self`、`mpnv_shuffle_positives`；
- `summarize_split_study.py`：记录 MPNV positive mask 规模、density 与 shuffle control。

## Seed0 10 Split Gate

执行设置：

- Dataset：Chameleon / Squirrel；
- Splits：0-9；
- Seed：0；
- Epochs：50；
- Baseline：`gcn_mlp_gcl`；
- Control：`mpnv_gcl --mpnv-shuffle-positives`。

输出目录：

```text
experiments/topvenue_gcl/runs/mpnv_gate_wiki_s0_splits0-2_e50/
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | Method | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | --- | ---: | ---: | --- | --- |
| Chameleon | MPNV | +0.017105 | +0.019132 | 7/3 | 正向，但 shuffled control 也强 |
| Chameleon | MPNV shuffled | +0.014254 | +0.012411 | 7/3 | 机制对照不够干净 |
| Squirrel | MPNV | +0.015082 | +0.014767 | 10/0 | 当前最强机制信号 |
| Squirrel | MPNV shuffled | +0.000961 | +0.000668 | 5/4 | 接近无效，支持结构化 mask |

Squirrel split-level delta：

```text
MPNV normal:   +0.015370 +0.021134 +0.018252 +0.019212 +0.009606 +0.004803 +0.000961 +0.019212 +0.027858 +0.014409
MPNV shuffled: -0.013449 -0.008646 +0.005764 +0.002882 +0.015370 -0.008646 +0.021134 +0.007685 -0.012488 +0.000000
```

## Seed1/Seed2 复核

执行设置：

- Dataset：Texas / Actor / Chameleon / Squirrel；
- Splits：0-9；
- Seeds：1 / 2；
- Epochs：50；
- Baseline：`gcn_mlp_gcl`。

执行命令：

```bash
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl mpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50 --baseline-method gcn_mlp_gcl --out runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50/runs_vs_gcn_mlp.csv --aggregate-out runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50/aggregate_vs_gcn_mlp.csv
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | MPNV F1Mi mean | MPNV F1Ma mean | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Texas | 0.639189 | 0.363024 | +0.002703 | -0.000075 | 10/9 | micro 近零，macro 不安全 |
| Actor | 0.350658 | 0.318501 | -0.001776 | -0.001529 | 12/7 | 均值负向，不能作为成功证据 |
| Chameleon | 0.428838 | 0.421045 | +0.000219 | +0.000657 | 9/11 | seed0 正信号未复现 |
| Squirrel | 0.310711 | 0.300447 | -0.000288 | +0.001016 | 10/10 | micro 不优于 baseline，seed0 10/10 正向失效 |

Squirrel seed1/seed2 split-level delta：

```text
-0.009606 +0.012488 -0.003842 +0.023055 +0.004803 +0.010567 +0.016330 -0.009606 -0.024015 -0.010567 +0.004803 +0.015370 -0.008646 -0.021134 +0.007685 -0.001921 -0.025937 +0.010567 -0.001921 +0.005764
```

## 最终解释

保留价值：

- MPNV 使用 dense mask，而不是 SSPNV 的单采样 positive，更接近 top-venue reference pattern；
- dense semantic/spatial mask 的工程路径已跑通，可作为后续 selective objective 或 diagnostic component；
- seed0 的 Squirrel normal-vs-shuffled 现象说明 structured positives 可能在部分训练状态下有价值。

失败点：

- seed1/seed2 下四个数据集 mean delta 均接近 0，且 Actor/Squirrel micro 为负；
- Squirrel seed1/seed2 不再保持 normal 10/10 split 正向；
- Chameleon seed1/seed2 不再稳定，positive/negative 为 9/11；
- MPNV 默认开启会伤害一部分 split，缺少无标签选择/回退机制；
- 尚未与 SP-GCL、S3GCL、PolyGCL、GraphECL 等强基线同协议对齐；
- dense mask 在更大图上的复杂度仍需说明，必要时要改为 sampled block 或 sparse mask。

## 停止裁决

MPNV 触发停止条件：

- seed1/seed2 后 Squirrel normal 优势消失；
- 4 个数据集没有形成至少 3/4 dataset mean micro 正向；
- Chameleon/Squirrel 不再提供稳定机制证据；
- 当前版本不能作为 active SOTA candidate。

后续不再继续跑 MPNV shuffled seed1/seed2，因为 normal gate 已经失败；继续跑 shuffled 只能解释失败原因，不能改变主线裁决。

## 下一步方向

下一代方法不能再是“默认添加 multi-positive loss”。更合理的方向是：

- label-free objective activation：只在可预测收益的节点/图上启用 semantic/spatial multi-positive；
- node-level fallback：低置信区域退回 `gcn_mlp_gcl` bootstrap；
- sparse/block multi-positive：降低 dense mask 复杂度；
- 或直接回到 S3GCL / GraphECL / PolyGCL 级参考范式，重新设计训练目标。

## 2026-06-29 分支诊断与后继

为确认 full MPNV 失败来自整个 dense multi-positive 假设，还是来自 semantic/spatial 固定同权组合，已补跑 Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 1/2 × 50 epoch 的 MPNV semantic-only 与 spatial-only 分支诊断。

Aggregate vs `gcn_mlp_gcl`：

| Dataset | semantic-only ΔF1Mi | semantic-only ΔF1Ma | spatial-only ΔF1Mi | spatial-only ΔF1Ma | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | +0.022523 | +0.019688 | -0.004505 | +0.018393 | semantic 分支更稳，spatial micro 风险大 |
| Actor | +0.000439 | +0.004256 | -0.001206 | -0.002260 | 两者都弱，semantic 略好 |
| Chameleon | +0.002193 | +0.005253 | +0.005848 | +0.009463 | spatial 分支略强，但整体小 |
| Squirrel | +0.012648 | +0.012769 | +0.008005 | +0.006115 | 两个分支均正，semantic 更强 |

诊断结论：

- full MPNV 失败不是因为 semantic/spatial 分支完全无效，而是固定同权组合与缺少 fallback 导致收益不稳定；
- 分支效果具有明显数据集依赖，不能用单一 semantic-only 或 spatial-only 作为最终主方法；
- 这支持后继方法 `aompnv_gcl`：将 dense semantic/spatial objectives 交给节点级 label-free objective activation，并保留 bootstrap fallback。

后继状态：

- `aompnv_gcl` 已实现并通过小门控，当前是 active-but-risky candidate；
- 但 AOMPNV 的 shuffled control 仍偏强，因此它尚未证明结构化 mask 机制；
- 若 AOMPNV 在 splits 0-9 × seeds 1/2 的 normal/shuffled 硬门控失败，则整个 MPNV 家族应彻底降级为 regularization / diagnostic ablation。
