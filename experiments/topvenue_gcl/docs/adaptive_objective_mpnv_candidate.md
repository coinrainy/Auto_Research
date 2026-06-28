# Adaptive Objective-Activated MPNV 候选备忘录

日期：2026-06-29

## 当前裁决

`aompnv_gcl` 暂时升级为 **active-but-risky candidate**，但不能声称已经找到 SOTA idea。

理由：它在 Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 1/2 × 50 epoch 的小门控中，相对 `gcn_mlp_gcl` strong foundation 四个数据集 mean F1Mi 均为正，且 Chameleon 与 Squirrel 的信号强于失败的 full MPNV 复核。但 shuffled-positive control 在 Texas 与 Squirrel 上也保持较强正向，说明“结构化 semantic/spatial positive mask 本身正确”的机制证据不干净。

因此当前应把贡献假设收缩为：**无标签 objective activation / node-level fallback 可能稳定化 dense multi-positive objectives**，而不是“semantic/spatial dense positives 天然可靠”。

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

## 风险

- 当前只覆盖 splits 0-2 与 seeds 1/2，还不是 10 split 多 seed 结论；
- Texas normal 与 shuffled 的 micro 均值完全持平，不能证明结构化 positive mask 有效；
- Squirrel normal 虽然 6/6 正向，但 shuffled 也是 6/6 正向，机制证据偏弱；
- Chameleon shuffled 接近 normal，说明一部分收益可能来自 dense objective regularization，而不是正确 positive construction；
- Actor 性能增益太小，不能作为主成功数据集；
- dense mask 仍有 O(N^2) 内存/计算风险，若继续推进必须实现 sparse/block 版本。

## 下一步停止/升级标准

继续推进前必须跑一个更硬门控：

- Texas/Actor/Chameleon/Squirrel × splits 0-9 × seeds 1/2；
- 同时包含 `gcn_mlp_gcl`、AOMPNV normal、AOMPNV shuffled；
- 升级条件：至少 3/4 数据集 normal mean F1Mi > `gcn_mlp_gcl`，且至少 2/4 数据集 normal - shuffled 明显为正；
- 停止条件：若 normal 与 shuffled 接近，或收益只来自 Chameleon/Squirrel 小幅正向，则把 AOMPNV 降级为 regularization ablation，不再作为主方法。

建议命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
RUNS_DIR="runs/aompnv_gate_ta_wiki_s1-2_splits0-9_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--aompnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```
