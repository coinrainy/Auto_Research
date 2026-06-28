# Bootstrap-Preserving Selective RRNV 候选记录

## 方法动机

此前 RRNV 系列的失败边界很清楚：

- `rrnv_gcl` 和 `dsrrnv_gcl` 在 Texas/Chameleon 上有强信号，但 Squirrel safety 与 shuffled control 不干净；
- `rwirrnv_gcl` 证明 invariance attenuation 有价值，但 per-node reliability 排序没有通过 shuffled/constant control；
- `eairrnv_gcl` 的 graph-level energy attenuation 能放大 Texas/Chameleon，但 Squirrel 仍负；
- `darrnv_gcl` 的 density auxiliary gate 能保护 Squirrel，却损失 Texas 主信号。

BPRRNV 的核心判断是：RRNV 不应该替代 `gcn_mlp_gcl` 的 Natural-View bootstrap，而应该作为一个被密度和能量冲突选择性调节的轻量辅助正则。

## 当前实现

入口：

```bash
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0
```

训练目标：

```text
loss = bootstrap_loss + bprrnv_rr_weight * aux_gate * rr_core
```

其中：

- `bootstrap_loss` 沿用 `gcn_mlp_gcl` 的 ego/graph 双向 negative cosine；
- `rr_core = invariance + 0.1 * variance + 0.01 * covariance`；
- `density_factor = sigmoid((log1p(degree_threshold) - log1p(avg_degree)) / degree_temperature)`；
- `energy_factor = 1 - energy_strength * (energy_ratio / (energy_ratio + threshold))^power`；
- `aux_gate = density_factor * energy_factor`。

必要 controls：

```bash
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0 --rrnv-shuffle-pairs
python train.py --dataset Texas --method bprrnv_gcl --epochs 50 --split-index 0 --seed 0 --bprrnv-uniform-gate
```

## 初筛结果

实验设置：Texas/Chameleon/Squirrel/Actor × splits 0/1/2 × model seed 0 × 50 epoch。

`bprrnv_rr_weight=0.25` 失败，normal 相对 `gcn_mlp_gcl` 的 overall mean micro delta 为 -0.005938。

`bprrnv_rr_weight=0.1` 初筛：

| dataset | normal - GCN-MLP | uniform - GCN-MLP | shuffled - GCN-MLP | normal - shuffled | aux gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Texas | +0.027027 | +0.000000 | -0.027027 | +0.054054 | 0.883684 |
| Chameleon | +0.003655 | +0.006579 | -0.010234 | +0.013889 | 0.526757 |
| Squirrel | +0.000640 | -0.005123 | -0.007365 | +0.008005 | 0.194363 |
| Actor | +0.005044 | +0.002632 | -0.001754 | +0.006798 | 0.686431 |
| Overall | +0.009092 | +0.001022 | -0.011595 | +0.020687 | 0.572809 |

输出文件：

- `runs/bprrnv_w01_candidate_comparison_s0_splits0-2_e50.csv`
- `runs/bprrnv_w01_candidate_aggregate_s0_splits0-2_e50.csv`

## 当前裁决

BPRRNV 暂时升级为 active-but-risky candidate，但不是成功主方法。

支持点：

- normal 整体优于 `gcn_mlp_gcl`、uniform gate 和 shuffled pair；
- Texas 的 normal-vs-shuffled 差距最大，机制最干净；
- Squirrel 上低 aux gate 避免了 EAIRRNV/RRNV 式过度施加，safety 有改善。

风险：

- Chameleon 上 uniform gate 高于 normal，说明 selector 不是全局最优；
- Actor 的 normal/shuffled 差距很小；
- 当前只跑了 splits0-2 seed0，不能外推到论文级结论；
- 仍需证明 homophily non-degradation。

## 下一步硬门槛

扩展前必须保持以下停止规则：

- 若 splits0-9 后 normal 不稳定优于 shuffled pair，放弃 pair-correspondence 机制叙事；
- 若 normal 不优于 uniform gate，降级为普通轻量 RR regularizer；
- 若 homophily 数据集出现超过 1 个百分点的稳定退化，必须加入 safety fallback 或放弃；
- 若 no-density 或 no-energy control 接近完整方法，删除无效模块，不保留复杂叙事。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Texas Chameleon Squirrel Actor" METHODS="gcn_mlp_gcl bprrnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/bprrnv_w01_s0_splits0-9_e50" OVERWRITE=1 bash scripts/run_split_study.sh
```

随后补：

```bash
DATASETS="Texas Chameleon Squirrel Actor" METHODS="bprrnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/bprrnv_w01_shuffled_s0_splits0-9_e50" EXTRA_ARGS="--rrnv-shuffle-pairs" RUN_TAG="shuffled" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Chameleon Squirrel Actor" METHODS="bprrnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/bprrnv_w01_uniform_s0_splits0-9_e50" EXTRA_ARGS="--bprrnv-uniform-gate" RUN_TAG="uniform" OVERWRITE=1 bash scripts/run_split_study.sh
```
