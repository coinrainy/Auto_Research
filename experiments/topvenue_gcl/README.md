# Top-Venue GCL 工作区

本目录用于从头实现新的图对比学习候选方法，避免继续依赖 patch 第三方官方代码或旧 `grace_idea/` 中累积的失败原型。

当前 active foundation：**GCN-MLP Natural View GCL**。它在 Texas/Actor/Chameleon/Squirrel 的轻量 split sanity 中稳定超过 GRACE，但本身不够创新；后续 candidate 必须在它之上给出新机制和增益。`ER-Cache`、`ER-Residual`、`Energy-SPGCL` 均已降级为失败/条件性消融。

当前裁决：**Disagreement-Aware Natural-View GCL (DANV-GCL)** 家族已降级为失败/条件性消融。`gcn_mlp_gcl` 仍是 strong foundation；下一代 active idea 需要换机制，而不是继续调 DANV penalty。

当前新候选：**Filter-Decoupled Natural-View GCL (FDNV-GCL)**，入口为 `--method fdnv_gcl`。它在 Natural-View foundation 上显式学习 low-pass / high-pass filtered targets，先以 split0 early gate 判断是否保留。

核心假设：

- MLP/ego 分支与 GCN/graph 分支是异配图上更自然的双视图；
- 不是所有节点都应该强制对齐 ego view 与 graph view；
- DANV 使用 label-free gate 判断何时对齐、何时保留分歧；
- high-energy residual 与 low-pass positive cache 已在 early gate 中降级为失败/条件性消融，不作为当前主线。

最小 smoke：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
bash scripts/run_smoke.sh
```

小规模 split-study：

```bash
DATASETS="Texas Actor" METHODS="grace gcn_mlp_gcl" SPLITS="0 1 2" SEEDS="0" EPOCHS=50 OVERWRITE=1 bash scripts/run_split_study.sh
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
python train.py --dataset Texas --method energy_spgcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_residual_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0 --shuffle-cache
python train.py --dataset Texas --method grace --epochs 5 --split-index 0 --seed 0
```
