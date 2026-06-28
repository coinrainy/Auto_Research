# Adaptive Objective-Activated MPNV 候选备忘录

日期：2026-06-29

## 当前裁决

`aompnv_gcl` 已在 10 split x seeds 1/2 硬门控后降级为 **失败/条件性消融资产**，不再作为 active main idea。

理由：它在 Texas/Actor/Chameleon/Squirrel x splits 0-2 x seeds 1/2 x 50 epoch 的小门控中，相对 `gcn_mlp_gcl` strong foundation 四个数据集 mean F1Mi 均为正，且 Chameleon 与 Squirrel 的信号强于失败的 full MPNV 复核。但更硬的 splits 0-9 x seeds 1/2 复核显示，只有 Squirrel 同时满足 normal > baseline 与 normal > shuffled 的较清楚信号；Texas 正向但 split-level 不稳，Actor 对 baseline 为负，Chameleon 反而 shuffled-positive control 更强。

因此当前应放弃把 AOMPNV 包装为 2026 顶会/顶刊主方法。可保留的线索是：**multi-objective / dense-positive 正则有时有效，但结构化 positive routing 尚未证明是因果机制**。

## 方法定义

入口：

```bash
python train.py --dataset Chameleon --method aompnv_gcl --epochs 50 --split-index 0 --seed 0
```

shuffled-positive control：

```bash
python train.py --dataset Chameleon --method aompnv_gcl --epochs 50 --split-index 0 --seed 0 --aompnv-shuffle-positives
```

核心组成：

- 复用 `gcn_mlp_gcl` Natural-View foundation；
- ego/MLP view 为 online anchor；
- semantic dense mask：raw propagation signature KNN；
- spatial dense mask：原图一跳邻居；
- semantic objective 对齐 high-pass target；
- spatial objective 对齐 low-pass target；
- bootstrap objective 保留 GCN-MLP natural-view 对齐；
- 每个节点根据 semantic/spatial/bootstrap 的相对 self-supervised loss 与 raw-signature confidence 做三分支 objective routing；
- 低可靠节点允许回退到 bootstrap，而不是默认启用 dense positive objectives。

当前实现位置：

- `train.py`：新增 `--method aompnv_gcl`、路由参数、训练分支与诊断；
- `src/losses.py`：新增 per-node `multi_positive_info_nce_per_node` 与 `negative_cosine_per_node`；
- `configs/default.yaml`：新增 `aompnv_*` 默认参数；
- `summarize_split_study.py`：新增 AOMPNV 路由概率、win fraction、loss 与 mask 诊断字段。

## 小门控结果

执行设置：

- Dataset：Texas / Actor / Chameleon / Squirrel；
- Splits：0 / 1 / 2；
- Seeds：1 / 2；
- Epochs：50；
- Baseline：`gcn_mlp_gcl`；
- Control：`aompnv_gcl --aompnv-shuffle-positives`；
- 输出目录：`runs/mpnv_branch_diag_ta_wiki_s1-2_splits0-2_e50/`。

执行命令：

```bash
RUNS_DIR="runs/mpnv_branch_diag_ta_wiki_s1-2_splits0-2_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv_shuffled" EXTRA_ARGS="--aompnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | AOMPNV ΔF1Mi | AOMPNV ΔF1Ma | Positive/Zero/Negative F1Mi | Shuffled ΔF1Mi | Normal - Shuffled ΔF1Mi | 裁决 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Texas | +0.022523 | +0.041834 | 3/2/1 | +0.022523 | -0.000000 | 性能正向，但 shuffled 同样强 |
| Actor | +0.001864 | -0.001147 | 3/1/2 | -0.007675 | +0.009539 | 机制较干净但性能弱 |
| Chameleon | +0.015351 | +0.017719 | 5/0/1 | +0.008772 | +0.006579 | 性能正向，shuffled warning |
| Squirrel | +0.018892 | +0.017179 | 6/0/0 | +0.015370 | +0.003522 | 稳定正向，但 shuffled 也强 |

对比 MPNV 分支诊断：

| Dataset | MPNV semantic-only ΔF1Mi | MPNV spatial-only ΔF1Mi | AOMPNV ΔF1Mi |
| --- | ---: | ---: | ---: |
| Texas | +0.022523 | -0.004505 | +0.022523 |
| Actor | +0.000439 | -0.001206 | +0.001864 |
| Chameleon | +0.002193 | +0.005848 | +0.015351 |
| Squirrel | +0.012648 | +0.008005 | +0.018892 |

AOMPNV 在四个数据集上均不弱于单独 semantic/spatial dense 分支，说明 objective activation 至少在小门控中修复了 full MPNV 的固定加权问题。

## 硬门控结果

执行设置：

- Dataset：Texas / Actor / Chameleon / Squirrel；
- Splits：0-9；
- Seeds：1 / 2；
- Epochs：50；
- Baseline：`gcn_mlp_gcl`；
- Control：`aompnv_gcl --aompnv-shuffle-positives`；
- 输出目录：`runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50/`。

Aggregate vs `gcn_mlp_gcl`：

| Dataset | AOMPNV ΔF1Mi | AOMPNV ΔF1Ma | Positive/Zero/Negative F1Mi | Shuffled ΔF1Mi | Normal - Shuffled ΔF1Mi | 裁决 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Texas | +0.010811 | +0.024572 | 9/5/6 | +0.002703 | +0.008108 | 均值正向但 split-level 不稳 |
| Actor | -0.002829 | -0.006672 | 7/1/12 | -0.010592 | +0.007763 | normal > shuffled 但低于 baseline |
| Chameleon | +0.000658 | +0.000853 | 12/1/7 | +0.011952 | -0.011294 | shuffled 明显更强，机制失败 |
| Squirrel | +0.018348 | +0.017634 | 16/1/3 | +0.004755 | +0.013593 | 唯一较清楚成功数据集 |

关键判断：

- AOMPNV 没有达到主方法升级门槛。虽然 3/4 数据集 normal mean F1Mi 不低于 baseline，但 Chameleon 几乎为零且 shuffled 更强，Actor 对 baseline 为负，Texas split-level 正负混杂。
- normal-vs-shuffled 只有 Squirrel 明确干净；Texas 和 Actor 的正差值不足以抵消 baseline/稳定性问题。
- 结构化 semantic/spatial dense positives 与 objective activation 不能作为主贡献叙事。
- AOMPNV 可作为后续论文的 negative result / regularization ablation：说明“多目标正则可能有用，但无标签 positive routing 并不自动带来可发表级稳定提升”。

## 风险

- 硬门控已覆盖 splits 0-9 与 seeds 1/2，但仍未达到主方法门槛；
- Texas normal 只有均值正向，split-level 正负混杂，不能证明结构化 positive mask 稳定有效；
- Squirrel 是唯一较清楚正例，但单数据集成功不足以支撑顶会/顶刊主方法；
- Chameleon shuffled 明显强于 normal，说明一部分收益来自 dense objective regularization 或训练噪声，而不是正确 positive construction；
- Actor normal 低于 baseline，不能作为成功数据集；
- dense mask 仍有 O(N^2) 内存/计算风险，若继续推进必须实现 sparse/block 版本。

## 下一步停止/升级标准

硬门控已经完成，结论为停止主线推进：

- 不继续调 `aompnv_router_temperature`、branch weight 或 confidence threshold；
- 不再围绕 semantic/spatial positive mask 做小修小补；
- 保留结果用于分析 shuffled/multi-objective regularization 为什么在部分数据集很强；
- 下一代 idea 必须换机制，仍以 `gcn_mlp_gcl` 为 strong foundation，并继续保留 shuffled/random/no-structure control。

复现实验与汇总命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
RUNS_DIR="runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv_shuffled" EXTRA_ARGS="--aompnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```
