# Top-Venue GCL 工作区

本目录用于从头实现新的图对比学习候选方法，避免继续依赖 patch 第三方官方代码或旧 `grace_idea/` 中累积的失败原型。

当前 active foundation：**GCN-MLP Natural View GCL**。它在 Texas/Actor/Chameleon/Squirrel 的轻量 split sanity 中稳定超过 GRACE，但本身不够创新；后续 candidate 必须在它之上给出新机制和增益。`ER-Cache`、`ER-Residual`、`Energy-SPGCL`、`DANV` 与 `FDNV` 均已降级为失败/条件性消融。

当前 active mechanism prototype：**Semantic-Spatial Positive Natural-View GCL (SSPNV-GCL)**，入口为 `--method sspnv_gcl`。它将 semantic positives 路由到 high-pass target，将 one-hop spatial positives 路由到 low-pass target，并保留 GCN-MLP Natural-View bootstrap。固定完整 SSPNV 已不再作为最终主方法包装，因为 control 显示 Chameleon 上 random/单分支变体也很强；下一代需要做 branch/objective selection。

10 split / seed0 / 50 epoch 的 early gate 显示 SSPNV 相对 `gcn_mlp_gcl` 在 Texas、Actor、Chameleon、Squirrel 的 mean micro/macro 均为正；其中 Chameleon 为 10/10 split micro 正向，Squirrel 为 9/10。Actor 仅弱正且不稳定，因此暂时只作为边界数据集。

新增 `--method afpnv_gcl`：在 SSPNV 上加入 raw propagation signature 正样本置信度加权。它已经跑通 Chameleon/Squirrel 10 split，但没有超过 full SSPNV 或 semantic-only，因此当前只作为 ablation，不升级为 active main idea。

核心假设：

- MLP/ego 分支与 GCN/graph 分支是异配图上更自然的双视图；
- 不是所有节点都应该强制对齐 ego view 与 graph view；
- high-energy residual、low-pass positive cache、DANV penalty 与 FDNV routed filter target 均已在 early gate 中降级为失败/条件性消融，不作为当前主线；
- SSPNV 的核心待证机制是 filter-specific positive construction，但 random-positive control 已证明固定双分支叙事不够；
- 下一版必须解释何时选择 semantic、spatial 或 bootstrap-only objective，而不是固定同权相加。

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
python train.py --dataset Texas --method energy_spgcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_residual_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0 --shuffle-cache
python train.py --dataset Texas --method grace --epochs 5 --split-index 0 --seed 0
```
