# Energy-Adaptive Invariance RRNV 候选记录

## 动机

RWIRRNV 的 10 split control 表明：RRNV 系列的收益更像来自 invariance attenuation，而不是当前 per-node reliability score 的节点对应关系。EAIRRNV 因此放弃逐节点 reliability 排序，改用 graph/high view 的全图 energy ratio 估计当前图是否存在较强 view conflict。

## 方法

入口：

```bash
python train.py --dataset Texas --method eairrnv_gcl --epochs 50 --split-index 0 --seed 0
```

核心公式：

```text
energy_ratio = mean(||z_high|| / ||z_graph||)
conflict = energy_ratio / (energy_ratio + threshold)
invariance_scale = clamp(1 - strength * conflict^power, min_scale, 1)
```

默认参数：

- `eairrnv_energy_threshold: 0.15`
- `eairrnv_strength: 0.6`
- `eairrnv_power: 1.0`
- `eairrnv_min_invariance_scale: 0.25`

诊断字段：

- `eairrnv_energy_ratio_mean`
- `eairrnv_energy_ratio_std`
- `eairrnv_conflict`
- `eairrnv_invariance_scale`

## 已执行实验

smoke：

```bash
python -m py_compile train.py summarize_split_study.py src/*.py
python train.py --dataset Texas --method eairrnv_gcl --epochs 2 --split-index 0 --seed 0 --runs-dir runs/eairrnv_smoke --run-name Texas_eairrnv_smoke --overwrite
python train.py --dataset Squirrel --method eairrnv_gcl --epochs 2 --split-index 0 --seed 0 --runs-dir runs/eairrnv_smoke --run-name Squirrel_eairrnv_smoke --overwrite
```

主初筛：

```bash
DATASETS="Texas Chameleon Squirrel Actor" METHODS="gcn_mlp_gcl eairrnv_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/eairrnv_s0_splits0-2_e50" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir runs/eairrnv_s0_splits0-2_e50 --baseline-method gcn_mlp_gcl --out runs/eairrnv_s0_splits0-2_e50/runs_vs_gcn_mlp.csv --aggregate-out runs/eairrnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

strength sweep：

```bash
DATASETS="Texas Chameleon Squirrel Actor" METHODS="eairrnv_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/eairrnv_strength03_s0_splits0-2_e50" EXTRA_ARGS="--eairrnv-strength 0.3" RUN_TAG="s03" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Chameleon Squirrel Actor" METHODS="eairrnv_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/eairrnv_strength09_s0_splits0-2_e50" EXTRA_ARGS="--eairrnv-strength 0.9" RUN_TAG="s09" OVERWRITE=1 bash scripts/run_split_study.sh
```

DARRNV 对照：

```bash
DATASETS="Texas Chameleon Squirrel Actor" METHODS="darrnv_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/darrnv_s0_splits0-2_e50" OVERWRITE=1 bash scripts/run_split_study.sh
```

## 结果摘要

EAIRRNV strength=0.6 相对 `gcn_mlp_gcl` 的 mean accuracy delta：

- Texas: +0.126126
- Chameleon: +0.031433
- Actor: +0.001754
- Squirrel: -0.005123

strength=0.3：

- Texas: +0.099099
- Chameleon: +0.021930
- Actor: +0.003509
- Squirrel: -0.008005

strength=0.9：

- Texas: +0.099099
- Chameleon: +0.013889
- Actor: +0.003509
- Squirrel: -0.008005

DARRNV 对照：

- Texas: -0.027027
- Chameleon: +0.023392
- Actor: +0.003728
- Squirrel: -0.000640

## 裁决

EAIRRNV 不作为当前 active main idea。理由：

- graph-level energy attenuation 在 Texas/Chameleon 上有清楚正信号；
- 但 Squirrel 在三个 strength 下均为负，说明单一全图 scale 不能提供可靠 safety；
- strength=0.6 是当前整体最强折中，但不是机制充分的全局最优；
- DARRNV 显示 density-aware auxiliary regularization 可以保护 Squirrel，却会损失 Texas 主信号。

保留价值：

- `eairrnv_gcl` 是 graph-level conflict attenuation 的可复现机制资产；
- 其失败边界指向下一代方法：需要判断何时用替代式 RRNV、何时保留 Natural-View bootstrap、何时只加轻量 regularizer。

## 下一步假设

下一代候选不应继续调 `eairrnv_strength`。更合理的方向是：

1. 保留 `gcn_mlp_gcl` bootstrap loss 作为安全底座；
2. 用 density 和 energy jointly 判断是否添加 RRNV regularization；
3. 在高密高能图上降低或关闭 RRNV 的 variance/covariance 整体正则，而不仅是 true-pair invariance；
4. 提供 no-RR、shuffled-pair、constant-scale 或 density-shuffled control。
