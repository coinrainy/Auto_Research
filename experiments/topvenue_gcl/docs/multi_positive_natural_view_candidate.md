# Multi-Positive Natural-View GCL 候选备忘录

日期：2026-06-29

## 当前裁决

`mpnv_gcl` 是 SSPNV / AFPNV / BSPNV 家族停止后新的 active-but-risky candidate。

它不是 SOTA 结论，但已经比前一轮 SSPNV 小变体更值得继续推进：在 Squirrel 的 10 split / seed0 / 50 epoch gate 中，真实 semantic/spatial multi-positive mask 明显强于 shuffled-positive control，说明结构化多正样本目标比单采样 positive 更有机制信号。

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

## 10 Split Gate

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

## 解释边界

支持信号：

- Squirrel 上 normal MPNV 10/10 split micro 正向；
- Squirrel 上 shuffled-positive control 基本退化，说明收益不是简单来自额外 dense loss 形式；
- MPNV 使用 dense mask，而不是 SSPNV 的单采样 positive，更接近 top-venue reference pattern。

风险：

- Chameleon 上 shuffled control 也明显正向，说明该数据集不能作为强机制证据；
- 当前只跑 seed0，尚未验证 model seed 稳定性；
- 尚未与 SP-GCL、S3GCL、PolyGCL、GraphECL 等强基线同协议对齐；
- dense mask 在更大图上的复杂度需要说明，必要时要改为 sampled block 或 sparse mask。

## 推进标准

继续推进的最低标准：

- Texas / Actor / Chameleon / Squirrel 至少 3/4 dataset 相对 `gcn_mlp_gcl` mean F1Mi 为正；
- Squirrel normal 明显强于 shuffled 的结论在 seed1/seed2 保持；
- 至少一个 homophily dataset 不明显退化；
- 强基线对齐前不能写 SOTA claim。

停止标准：

- seed1/seed2 后 Squirrel normal-vs-shuffled 差异消失；
- Texas/Actor 明显负向，且无法通过无标签 gate 识别回退；
- Chameleon/Squirrel 的收益主要来自 shuffled control；
- 强基线对齐后低于 SP-GCL/S3GCL/PolyGCL/GraphECL 且没有新的机制诊断价值。

## 下一步建议命令

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl mpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50" OVERWRITE=1 bash scripts/run_split_study.sh
```

随机正样本对照：

```bash
DATASETS="Chameleon Squirrel" METHODS="mpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="runs/mpnv_gate_wiki_shuffled_s1-2_splits0-9_e50" RUN_TAG="shuffled" EXTRA_ARGS="--mpnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
```
