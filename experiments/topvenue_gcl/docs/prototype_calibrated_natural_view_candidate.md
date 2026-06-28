# Prototype-Calibrated Natural-View GCL 候选记录

## 当前裁决

`pcnv_gcl` 暂时保留为 **active-but-risky candidate**，但不能作为主方法定稿。

核心原因：

- default PCNV 在 Texas/Actor/Chameleon split0 seed0 上相对 `gcn_mlp_gcl` 为正，但 Actor/Chameleon 的 shuffled assignment control 同样强甚至更强；
- sharpened PCNV 在 Texas 上给出当前最强单点信号，并且 normal > shuffled，但在 Chameleon/Squirrel 出现 prototype usage entropy 明显下降，提示原型坍塌；
- 当前证据支持“prototype-level natural-view calibration 有潜力”，但不支持“跨视图原型分配本身已被证明是因果机制”。

## 方法定义

PCNV 保留 `gcn_mlp_gcl` 的 Natural-View bootstrap：

- ego / MLP view 与 graph / GCN view 双向 BYOL-style 对齐；
- final representation 仍使用 `ego_graph`，避免通过评估表征切换制造伪提升。

新增模块：

- 维护一组 trainable prototypes；
- 将 ego view 与 graph view 分别映射到 prototype assignment；
- 用 graph assignment 作为 ego assignment 的 stop-gradient soft target，反向也一样；
- 加入 prototype usage balance，避免所有节点集中到少数原型；
- `--pcnv-shuffle-assignments` 打乱 prototype target 与节点对应关系，作为机制 control。

入口：

```bash
python train.py --dataset Texas --method pcnv_gcl --epochs 50 --split-index 0 --seed 0
python train.py --dataset Texas --method pcnv_gcl --epochs 50 --split-index 0 --seed 0 --pcnv-shuffle-assignments
```

主要配置：

- `pcnv_num_prototypes`
- `pcnv_base_weight`
- `pcnv_prototype_weight`
- `pcnv_balance_weight`
- `pcnv_assignment_temperature`
- `pcnv_target_temperature`
- `pcnv_shuffle_assignments`

## 实验记录

### Default PCNV

执行：

```bash
RUNS_DIR="runs/pcnv_split0_s0_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/split_study_runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/split_study_aggregate_vs_gcn_mlp.csv"
```

结果：

| Dataset | Baseline F1Mi/F1Ma | PCNV F1Mi/F1Ma | PCNV ΔF1Mi/ΔF1Ma | Shuffled F1Mi/F1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | 0.675676 / 0.334091 | 0.702703 / 0.419643 | +0.027027 / +0.085552 | 0.675676 / 0.297619 | normal 明显更好 |
| Actor | 0.344737 / 0.310827 | 0.352632 / 0.318250 | +0.007895 / +0.007423 | 0.357895 / 0.333336 | shuffled 更强 |
| Chameleon | 0.442982 / 0.436501 | 0.471491 / 0.466336 | +0.028509 / +0.029835 | 0.473684 / 0.470067 | shuffled 更强 |
| Squirrel | 0.327570 / 0.318651 | 0.313160 / 0.298983 | -0.014409 / -0.019669 | 0.311239 / 0.304145 | baseline 失败 |

### Sharpened PCNV

执行：

```bash
RUNS_DIR="runs/pcnv_sharp_split0_s0_e50"
EXTRA_ARGS="--pcnv-prototype-weight 0.5 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.1 --pcnv-target-temperature 0.03"
DATASETS="Texas Chameleon" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Chameleon" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="$EXTRA_ARGS --pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
```

结果：

| Dataset | Sharp normal F1Mi/F1Ma | Sharp shuffled F1Mi/F1Ma | 观察 |
| --- | ---: | ---: | --- |
| Texas | 0.729730 / 0.459091 | 0.702703 / 0.414448 | 当前最强 Texas 单点，normal > shuffled |
| Actor | 0.351316 / 0.306585 | 0.348026 / 0.319835 | micro normal 略高，macro shuffled 更高 |
| Chameleon | 0.449561 / 0.442837 | 0.438596 / 0.429895 | normal > shuffled，但弱于 default PCNV |
| Squirrel | 0.295869 / 0.273211 | 0.315082 / 0.298631 | 明显失败，且 shuffled 更强 |

## 失败边界

- default 版本的 assignment entropy 较高，prototype target 偏软，因此 shuffled target 仍可能提供类似的平滑正则；
- sharpened 版本让 Texas 获益，但 Chameleon/Squirrel 的 usage entropy 下降到非常低，说明固定 sharpen temperature 容易造成少数 prototype 吞并；
- Squirrel 对 PCNV 系列不友好，当前不应继续用它包装“通用 heterophily SOTA”；
- 下一步不能继续只调一个全局 temperature，应实现 entropy-guarded / adaptive prototype calibration，否则直接放弃 PCNV。

## 下一步停止条件

下一代 PCNV 若满足任一条件，应放弃 prototype calibration 主线：

- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 0-1 中 normal 平均不优于 `gcn_mlp_gcl` 至少 1 个百分点；
- normal-vs-shuffled 平均差值不为正，或仅 1 个数据集为正；
- prototype usage entropy 长期低于 `0.5 * log(K)` 且性能提升依赖这种坍塌；
- homophily Cora/CiteSeer/PubMed 出现超过 1 个百分点平均退化。

建议下一步实现：

- entropy-guarded prototype loss：当 usage entropy 过低时降低 consistency loss，而不是继续硬拉 assignment；
- confidence-weighted assignment：只让高置信 prototype target 贡献主要 consistency loss；
- dataset-agnostic temperature schedule：先软后锐化，但受 usage entropy 约束。
