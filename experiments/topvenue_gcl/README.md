# Top-Venue GCL 工作区

本目录用于从头实现新的图对比学习候选方法，避免继续依赖 patch 第三方官方代码或旧 `grace_idea/` 中累积的失败原型。

当前 active foundation：**GCN-MLP Natural View GCL**。它在 Texas/Actor/Chameleon/Squirrel 的轻量 split sanity 中稳定超过 GRACE，但本身不够创新；后续 candidate 必须在它之上给出新机制和增益。`ER-Cache`、`ER-Residual`、`Energy-SPGCL`、`DANV` 与 `FDNV` 均已降级为失败/条件性消融。

SSPNV / AFPNV / BSPNV 已降级为机制与消融资产，不再作为 active main idea。固定完整 SSPNV 的入口为 `--method sspnv_gcl`；AFPNV 的入口为 `--method afpnv_gcl`；BSPNV branch selection 的入口为 `--method bspnv_gcl`。BSPNV 强于 AFPNV，但没有同时超过 Chameleon semantic-only 与 Squirrel full SSPNV，因此触发停止条件。

MPNV-GCL 已降级为失败/条件性消融资产，不再作为 active main idea。它用 dense semantic/spatial multi-positive mask 替代 SSPNV 的单采样 positive，并保留 Natural-View bootstrap；seed0 在 Chameleon/Squirrel 上曾出现正信号，但 seed1/seed2 扩展门控未复现稳定优势。Texas/Actor/Chameleon/Squirrel × splits0-9 × seeds1-2 × 50 epoch 下，MPNV 相对 `gcn_mlp_gcl` 的 mean F1Mi delta 分别为 +0.002703、-0.001776、+0.000219、-0.000288，均不足以支撑主方法。

Adaptive Objective-Activated MPNV (AOMPNV) 已降级为失败/条件性消融资产，入口仍为 `--method aompnv_gcl`。它在小门控中曾通过 objective activation 改善 full MPNV，但 Texas/Actor/Chameleon/Squirrel × splits0-9 × seeds1-2 × 50 epoch 硬门控未达到主方法门槛：相对 `gcn_mlp_gcl` 的 mean F1Mi delta 分别为 +0.010811、-0.002829、+0.000658、+0.018348；normal-vs-shuffled delta 分别为 +0.008108、+0.007763、-0.011294、+0.013593。只有 Squirrel 信号较清楚，Chameleon shuffled 更强，Actor 低于 baseline，因此不再作为 active candidate。

新增 `--method afpnv_gcl` 和 `--method bspnv_gcl`：分别对应置信度加权与 semantic/spatial/bootstrap branch selection。二者都已经跑通 Chameleon/Squirrel 10 split，但都没有形成足够强的主线结果。

核心假设：

- MLP/ego 分支与 GCN/graph 分支是异配图上更自然的双视图；
- 不是所有节点都应该强制对齐 ego view 与 graph view；
- high-energy residual、low-pass positive cache、DANV penalty 与 FDNV routed filter target 均已在 early gate 中降级为失败/条件性消融，不作为当前主线；
- SSPNV 的核心待证机制是 filter-specific positive construction，但 random-positive control 已证明固定双分支叙事不够；
- AFPNV/BSPNV 已尝试解释何时选择 semantic、spatial 或 bootstrap-only objective，但未过升级门槛；
- MPNV 将 positive construction 从单采样改为 dense multi-positive mask，但 seed1/seed2 复核失败，当前只保留为机制/消融资产；
- AOMPNV 将 MPNV 的固定 dense objectives 改成 label-free objective activation / node-level fallback，但 10 split 多 seed normal-vs-shuffled 硬门控未过，当前只保留为 regularization/negative-result ablation；
- 当前仍没有可直接包装为 2026 顶会/顶刊主方法的成功 idea；下一代 candidate 必须换机制，而不是继续围绕 semantic/spatial positive mask 或 router 小调参。

最小 smoke：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
bash scripts/run_smoke.sh
```

小规模 split-study：

```bash
DATASETS="Texas Actor" METHODS="grace gcn_mlp_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 OVERWRITE=1 bash scripts/run_split_study.sh
```

同一方法多变体对照可用 `RUN_TAG` 避免 run 目录重名：

```bash
RUN_TAG=random_semantic METHODS="sspnv_gcl" EXTRA_ARGS="--sspnv-random-semantic" bash scripts/run_split_study.sh
```

输出：

- `runs/<study>/split_study_runs.csv`
- `runs/<study>/split_study_aggregate.csv`

单次运行示例：

```bash
python train.py --dataset Texas --method gcn_mlp_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method danv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method danv_gcl --epochs 5 --split-index 0 --seed 0 --danv-disagreement-weight 0.0
python train.py --dataset Texas --method danv_degree_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method fdnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method sspnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method afpnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method bspnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method mpnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method mpnv_gcl --epochs 5 --split-index 0 --seed 0 --mpnv-shuffle-positives
python train.py --dataset Texas --method aompnv_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method aompnv_gcl --epochs 5 --split-index 0 --seed 0 --aompnv-shuffle-positives
python train.py --dataset Texas --method energy_spgcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_residual_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0 --shuffle-cache
python train.py --dataset Texas --method grace --epochs 5 --split-index 0 --seed 0
```
