# Structure-Residual Gated Natural-View GCL 候选备忘录

日期：2026-06-29

## 当前裁决

`srgnv_gcl` 已在 split0 early gate 后降级为 **失败/条件性消融资产**，不进入 splits 0-2 扩展。

理由：该方法只在 Actor split0 seed0 上相对 `gcn_mlp_gcl` 有微弱正向；Texas micro 持平但 macro 下降，Chameleon 与 Squirrel 明显低于 baseline。更关键的是，`--srgnv-shuffle-residual` 在 Actor 上比 normal 更强，在 Chameleon 上也不支持 normal 优势，说明结构残差 target 的因果机制没有成立。

## 方法定义

入口：

```bash
python train.py --dataset Actor --method srgnv_gcl --epochs 50 --split-index 0 --seed 0
```

shuffled residual control：

```bash
python train.py --dataset Actor --method srgnv_gcl --epochs 50 --split-index 0 --seed 0 --srgnv-shuffle-residual
```

核心思路：

- 保留 `gcn_mlp_gcl` 的 Natural-View bootstrap；
- 将 graph view 分解为 ego/feature 方向可解释的 parallel component 与 orthogonal structure residual；
- 用 raw feature propagation residual score `1 - cos(x, P x)` 做节点级 gate；
- 只在 gate 较高处蒸馏 graph residual 给 ego/MLP 分支；
- 用 shuffled residual target 作为 no-structure control。

当前实现位置：

- `train.py`：新增 `--method srgnv_gcl`、SRGNV CLI 参数、训练分支与诊断；
- `configs/default.yaml`：新增 `srgnv_*` 默认参数；
- `summarize_split_study.py`：新增 SRGNV residual/gate/control 诊断字段。

## Split0 Early Gate

执行设置：

- Dataset：Texas / Actor / Chameleon / Squirrel；
- Splits：0；
- Seeds：0；
- Epochs：50；
- Baseline：`gcn_mlp_gcl`；
- Control：`srgnv_gcl --srgnv-shuffle-residual`；
- 输出目录：`runs/srgnv_split0_s0_e50/`。

执行命令：

```bash
RUNS_DIR="runs/srgnv_split0_s0_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="srgnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="srgnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--srgnv-shuffle-residual" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | SRGNV ΔF1Mi | SRGNV ΔF1Ma | Shuffled ΔF1Mi | Shuffled ΔF1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | +0.000000 | -0.028439 | -0.027027 | -0.049272 | micro 持平，macro 退化 |
| Actor | +0.001974 | +0.016069 | +0.002632 | +0.020925 | shuffled 更强，机制不成立 |
| Chameleon | -0.041667 | -0.076563 | -0.021930 | -0.019895 | 明显失败 |
| Squirrel | -0.005764 | -0.019545 | -0.028818 | -0.022141 | normal 仍低于 baseline |

诊断观察：

- raw residual gate 有合理分布，不是全 1：Texas mean=0.404832，Actor=0.522296，Chameleon=0.479019，Squirrel=0.464632；
- residual cosine 能被优化到较高值，但下游 F1 未受益，说明“预测 graph residual”不等价于分类有用结构信息；
- shuffled residual 在 Actor 上更强，Chameleon 上也未崩溃，说明该训练目标有明显正则化成分，不能作为结构残差机制证据。

## 停止原因

- 未达到 split0 early gate：至少 3/4 数据集应超过 `gcn_mlp_gcl`，且 normal 应明显优于 shuffled；
- Chameleon 大幅退化，直接触发停止条件；
- Actor 的唯一正向也被 shuffled control 超过；
- 不继续调 `srgnv_residual_weight`、threshold 或 temperature。

## 保留价值

SRGNV 是一个有用的 negative result：在 Natural-View GCL 中，简单蒸馏 graph view 中与 ego view 正交的残差，会优化自监督残差对齐指标，但不稳定提升节点分类。这提示下一代 idea 不应只问“结构里有什么 ego 没有的信号”，还必须判断这些结构残差是否与下游可分性一致。

下一步应换机制，优先考虑能直接诊断 downstream separability 或 class-conditional neighborhood conflict 的无标签代理，而不是继续对 representation residual 做蒸馏。
