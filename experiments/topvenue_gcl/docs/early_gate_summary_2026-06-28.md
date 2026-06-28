# Topvenue GCL Early Gate Summary

日期：2026-06-28

## 结论

本轮新增 scaffold 跑通，但三个新方法候选均未达到继续扩大的门槛。

- `er_cache_gcl`：放弃。Texas split0 seed0 下 normal cache 低于 shuffled/self-only，positive cache 主张失败。
- `er_residual_gcl`：放弃主线。Actor 正向，但 Texas/Chameleon 明显负向，Squirrel 仅 micro 弱正且 macro 下降。
- `energy_spgcl`：放弃当前实现。Texas/Chameleon 均低于 GRACE。
- `gcn_mlp_gcl`：保留为强对照/baseline。Actor 正向，Texas 追平 GRACE，Chameleon micro 小正但 macro 下降，Squirrel 失败；不构成新主方法。

## 关键 early gate 数值

所有结果均为 split0 / seed0 / 20 epoch，仅用于早筛，不作为论文结论。

| Dataset | Method | F1Mi | F1Ma | 裁决 |
| --- | --- | ---: | ---: | --- |
| Texas | GRACE | 0.675676 | 0.344612 | baseline |
| Texas | ER-Residual-GCL | 0.513514 | 0.291795 | 失败 |
| Texas | GCN-MLP-GCL | 0.675676 | 0.334091 | 只追平 baseline |
| Texas | Energy-SPGCL | 0.594595 | 0.373864 | micro 失败 |
| Chameleon | GRACE | 0.407895 | 0.401071 | baseline |
| Chameleon | ER-Residual-GCL | 0.326754 | 0.322908 | 失败 |
| Chameleon | GCN-MLP-GCL | 0.412281 | 0.362119 | macro 失败 |
| Chameleon | Energy-SPGCL | 0.324561 | 0.319509 | 失败 |
| Squirrel | GRACE | 0.304515 | 0.299359 | baseline |
| Squirrel | ER-Residual-GCL | 0.321806 | 0.277833 | macro 失败 |
| Squirrel | GCN-MLP-GCL | 0.274736 | 0.264493 | 失败 |
| Actor | GRACE | 0.261842 | 0.187220 | baseline |
| Actor | ER-Residual-GCL | 0.301316 | 0.279931 | 条件性正向 |
| Actor | GCN-MLP-GCL | 0.332895 | 0.311951 | 正向但非通用 |

## 下一步裁决

不要继续在当前三个失败 loss 上做小参数调参。下一步应回到已验证强信号：

1. 以 SP-GCL / GraphACL / GraphECL / PolyGCL / S3GCL 为强基线门槛；
2. 将 SPARC residual calibration 从 post-hoc/patch 经验，改造成可复现的标准训练或标准 evaluation module；
3. 若无法从头复现 official SP-GCL 级 embedding quality，则不要声称新 SOTA，只记录为 negative result。

## 2026-06-28 追加：GCN-MLP strong control split sanity

已新增 `scripts/run_split_study.sh` 与 `summarize_split_study.py`，并完成 Texas/Actor × splits 0/1/2 × seed0 × 50 epoch 的轻量复核。

命令：

```bash
DATASETS="Texas Actor" \
METHODS="grace gcn_mlp_gcl" \
SPLITS="0 1 2" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_texas_actor_s0_splits0-2_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate：

| Dataset | Method | Runs | F1Mi mean | F1Ma mean | ΔF1Mi vs GRACE | ΔF1Ma vs GRACE | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Texas | `gcn_mlp_gcl` | 3 | 0.639640 | 0.322785 | +0.036036 | +0.048717 | 3/0 |
| Actor | `gcn_mlp_gcl` | 3 | 0.353728 | 0.311257 | +0.076535 | +0.076413 | 3/0 |

裁决更新：

- `gcn_mlp_gcl` 不再只是普通 baseline，而是下一阶段必须击败的 strong control / architecture foundation。
- 它仍不是足够创新的论文主方法；下一步的创新应建立在 GCN-MLP 天然双视图底座上，并证明新增模块稳定超过该底座，而不是只超过 GRACE。
- 优先补 Chameleon/Squirrel 的同配置 split-study，判断 strong control 的边界；如果 WikipediaNetwork 失败，下一代模块应专门解释为什么 GCN-MLP 对 WebKB/Actor 有效、对 Chameleon/Squirrel 不稳。

## 2026-06-28 追加：WikipediaNetwork split sanity

已继续执行 Chameleon/Squirrel × splits 0/1/2 × seed0 × 50 epoch：

```bash
DATASETS="Chameleon Squirrel" \
METHODS="grace gcn_mlp_gcl" \
SPLITS="0 1 2" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_wiki_s0_splits0-2_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate：

| Dataset | Method | Runs | F1Mi mean | F1Ma mean | ΔF1Mi vs GRACE | ΔF1Ma vs GRACE | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chameleon | `gcn_mlp_gcl` | 3 | 0.416667 | 0.403677 | +0.021930 | +0.018146 | 3/0 |
| Squirrel | `gcn_mlp_gcl` | 3 | 0.309638 | 0.300817 | +0.025296 | +0.022114 | 3/0 |

裁决更新：

- `gcn_mlp_gcl` 在 Texas/Actor/Chameleon/Squirrel 的 splits 0/1/2 上全部 micro 正向，已成为当前最强实验线索。
- 它仍低于 SP-GCL 等强 heterophily-specific baseline 的已知水平，因此不能作为 SOTA 主方法。
- 下一代 active idea 应是 **Disagreement-Aware Natural-View GCL (DANV-GCL)**：在 GCN-MLP natural views 上学习“何时对齐、何时保留分歧”，并必须同时超过 GRACE 与 GCN-MLP。

## 2026-06-28 追加：DANV-GCL split0 gate

已实现 `--method danv_gcl`，并在 Texas/Actor/Chameleon/Squirrel split0 seed0 50 epoch 上做 first gate。

相对 GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | ---: | ---: | --- |
| Texas | -0.054054 | -0.123677 | 明显失败 |
| Actor | +0.005263 | +0.014987 | 小幅正向 |
| Chameleon | +0.006579 | +0.009536 | 小幅正向 |
| Squirrel | +0.003842 | +0.002403 | 小幅正向 |

当前裁决：DANV 是 active-but-risky candidate，不是成功方法。它达到“值得扩到 splits 0-2”的边缘标准，但 Texas 退化是 major warning。下一步必须做 split0-2 复核和 penalty/gate 消融。

## 2026-06-28 追加：DANV-GCL splits 0/1/2 复核

已执行：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl danv_gcl" \
SPLITS="0 1 2" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_danv_s0_splits0-2_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

并以 `gcn_mlp_gcl` 为 baseline 重新汇总：

```bash
python summarize_split_study.py \
  --runs-dir runs/split_study_danv_s0_splits0-2_e50 \
  --baseline-method gcn_mlp_gcl \
  --out runs/split_study_danv_s0_splits0-2_e50/runs_vs_gcn_mlp.csv \
  --aggregate-out runs/split_study_danv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

Aggregate：

| Dataset | DANV F1Mi mean | DANV F1Ma mean | ΔF1Mi vs GCN-MLP | ΔF1Ma vs GCN-MLP | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Texas | 0.621622 | 0.282129 | +0.027027 | -0.030448 | 3/0 | micro 正向，macro 不安全 |
| Actor | 0.352632 | 0.308947 | +0.002412 | -0.002621 | 1/2 | split 不稳定 |
| Chameleon | 0.425439 | 0.416210 | +0.005117 | +0.007722 | 3/0 | 稳定小正向 |
| Squirrel | 0.322446 | 0.310770 | +0.008005 | +0.007299 | 3/0 | 稳定小正向 |

裁决更新：

- DANV 不应被包装为已成功的 SOTA idea；它只是通过了继续做消融的最低门槛。
- Chameleon/Squirrel 的 6/6 split micro 正向是目前最干净的机制信号。
- Texas macro 退化与 Actor split 不稳定是 major risk。
- 下一步不扩大数据集，先做 DANV penalty/gate ablation：`danv_disagreement_weight=0`、`0.02`、温度与 min-align 消融；如果无法同时保住 Texas macro 与 Actor 稳定性，应放弃 DANV 主线或收缩到 WikipediaNetwork 条件性方法。

工程补充：`train.py` 已支持 `--danv-alignment-weight`、`--danv-disagreement-weight`、`--danv-gate-temperature`、`--danv-min-align-weight`，方便下一轮不改 YAML 直接跑消融。

## 2026-06-28 追加：DANV 消融后裁决

已新增决策记录：`docs/danv_ablation_decision_2026-06-28.md`。

补充实验：

- `danv_disagreement_weight=0.0`；
- `danv_disagreement_weight=0.02`；
- 新增 `danv_degree_gcl`，用 incident-degree gate 调节 disagreement penalty。

关键结果：

| Variant | Texas ΔF1Mi/ΔF1Ma | Actor ΔF1Mi/ΔF1Ma | Chameleon ΔF1Mi/ΔF1Ma | Squirrel ΔF1Mi/ΔF1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| DANV `w=0.1` | +0.027027/-0.030448 | +0.002412/-0.002621 | +0.005117/+0.007722 | +0.008005/+0.007299 | mean 正向但 macro/split 风险 |
| DANV `w=0.0` | +0.009009/+0.044501 | +0.003947/+0.006653 | +0.008041/+0.009381 | +0.006084/+0.013229 | mean 正向但 split 不稳 |
| DANV `w=0.02` | +0.018018/+0.019142 | -0.001754/+0.003021 | -0.000731/+0.002283 | +0.006724/+0.004984 | Actor/Chameleon micro 失败 |
| `danv_degree_gcl` split0 | 0.000000/-0.003734 | +0.005921/+0.013567 | 0.000000/-0.000556 | -0.002882/+0.004420 | 未过 split0 early gate |

裁决：

- 固定全局 disagreement penalty 没有稳定有效区间；
- degree-aware gate 没有救回 Squirrel/Chameleon 的主张；
- DANV 家族不再作为当前主方法推进，只保留为失败/条件性消融资产；
- `gcn_mlp_gcl` 仍是 Natural-View strong foundation，下一代 idea 必须换机制。

## 2026-06-28 追加：FDNV-GCL 第一版

已新增 `--method fdnv_gcl` 与备忘录：`docs/filter_decoupled_natural_view_candidate.md`。

方法：在 GCN-MLP Natural-View foundation 上显式学习 low-pass / high-pass filtered targets，用 raw feature residual、raw-neighbor agreement 与 degree 构造 filter gate。

split0 early gate：

| Variant | Texas ΔF1Mi/ΔF1Ma | Actor ΔF1Mi/ΔF1Ma | Chameleon ΔF1Mi/ΔF1Ma | Squirrel ΔF1Mi/ΔF1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| FDNV `route=0.5` | 0.000000/0.000000 | +0.008553/+0.010262 | -0.004386/-0.012705 | 0.000000/-0.000494 | Chameleon 失败 |
| FDNV `route=0.1` | 0.000000/+0.045083 | +0.001316/+0.010971 | -0.008772/-0.011839 | +0.003842/+0.006560 | Chameleon 失败 |

裁决：FDNV 第一版有局部信号，但不作为 active main idea，不进入 splits 0/1/2。下一步应重构 filter objective，而不是继续调 route weight。

## 2026-06-28 追加：SSPNV-GCL 10 split early gate

已实现 `--method sspnv_gcl`，将 semantic positives 与 spatial positives 分别路由到 high-pass / low-pass target，并保留 GCN-MLP Natural-View bootstrap。

执行：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl sspnv_gcl" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_sspnv_s0_splits0-2_e50" \
bash scripts/run_split_study.sh
```

并以 `gcn_mlp_gcl` 为 baseline 汇总：

```bash
python summarize_split_study.py \
  --runs-dir runs/split_study_sspnv_s0_splits0-2_e50 \
  --baseline-method gcn_mlp_gcl \
  --out runs/split_study_sspnv_s0_splits0-2_e50/runs_vs_gcn_mlp.csv \
  --aggregate-out runs/split_study_sspnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

Aggregate：

| Dataset | SSPNV F1Mi mean | SSPNV F1Ma mean | ΔF1Mi vs GCN-MLP | ΔF1Ma vs GCN-MLP | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Texas | 0.678378 | 0.420249 | +0.032432 | +0.069760 | 6/2 | 均值强正，macro 明显；仍有负 split |
| Actor | 0.346250 | 0.313962 | +0.000658 | +0.004324 | 6/4 | 弱正且不稳定 |
| Chameleon | 0.439912 | 0.431672 | +0.031140 | +0.032431 | 10/0 | 当前最强信号 |
| Squirrel | 0.321422 | 0.306920 | +0.011720 | +0.009843 | 9/1 | 稳定正向但 macro 有波动 |

裁决更新：

- SSPNV-GCL 升级为当时 active candidate；
- 这不是最终 SOTA 结论，只说明该 idea 通过了比 DANV/FDNV 更硬的 early gate；
- Actor 只能作为边界数据集，不能作为主要成功叙事；
- 下一步不再扩展复杂模块，优先做 semantic-only、spatial-only、random semantic/spatial positive、homophily safety 与强基线同协议对齐；
- 若 random positive 或单分支消融接近完整方法，应立即收缩或放弃 SSPNV 主张。

## 2026-06-28 追加：SSPNV control 与 AFPNV 裁决

已完成 Chameleon/Squirrel × splits 0-9 × seed0 × 50 epoch 的 SSPNV control：

- full SSPNV；
- semantic-only：`--sspnv-spatial-weight 0.0`；
- spatial-only：`--sspnv-semantic-weight 0.0`；
- random semantic：`--sspnv-random-semantic`；
- random spatial：`--sspnv-random-spatial`；
- AFPNV：新增 `--method afpnv_gcl`，用 raw propagation signature 正样本置信度对 semantic/spatial loss 做节点级加权。

Aggregate vs `gcn_mlp_gcl`：

| Dataset | Variant | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | --- | ---: | ---: | --- |
| Chameleon | full SSPNV | +0.027412 | +0.029084 | 正向但不是最强 |
| Chameleon | semantic-only | +0.037281 | +0.038611 | 当前最强 |
| Chameleon | spatial-only | +0.033553 | +0.036636 | 也强，削弱双分支必要性 |
| Chameleon | random semantic | +0.029825 | +0.029735 | major warning：随机 semantic 也强 |
| Chameleon | random spatial | +0.023465 | +0.025592 | 仍正向 |
| Chameleon | AFPNV | +0.025000 | +0.024495 | 未超过 full/semantic-only |
| Squirrel | full SSPNV | +0.007397 | +0.003500 | 当前最强但稳定性一般 |
| Squirrel | semantic-only | +0.004803 | +0.000484 | 接近 full |
| Squirrel | spatial-only | +0.000480 | -0.000961 | 基本无效 |
| Squirrel | random semantic | -0.004131 | -0.009298 | 明确失败 |
| Squirrel | random spatial | -0.000096 | -0.003287 | 基本无效 |
| Squirrel | AFPNV | +0.004995 | -0.000219 | 未超过 full SSPNV |

裁决更新：

- 固定完整 SSPNV 不再作为最终主方法包装；
- Chameleon 上 random semantic 和单分支 control 太强，说明“结构化 semantic-spatial 正样本拆分是必要机制”的主张站不稳；
- Squirrel 上 random positives 失败，说明结构化 positives 仍有条件性价值；
- AFPNV 已实现但未通过升级门槛，保留为 ablation，不作为 active main idea；
- 下一阶段若继续 SSPNV 家族，必须做 branch/objective selection，而不是固定同权相加或简单置信度加权。

## 2026-06-28 追加：BSPNV branch selection 裁决

已实现 `--method bspnv_gcl`：在 SSPNV positive objective 上加入 semantic / spatial / bootstrap 三分支竞争选择。

默认配置：

- `bspnv_branch_temperature=0.1`；
- `bspnv_bootstrap_bias=0.25`；
- 继续使用 `sspnv_semantic_weight=0.1`、`sspnv_spatial_weight=0.1`、`sspnv_bootstrap_weight=1.0`。

已执行 Chameleon/Squirrel × splits 0-9 × seed0 × 50 epoch，并并入 `runs/sspnv_controls_wiki_s0_splits0-9_e50/aggregate_vs_gcn_mlp.csv`。

Aggregate vs `gcn_mlp_gcl`：

| Dataset | BSPNV ΔF1Mi | BSPNV ΔF1Ma | Positive/Negative F1Mi | 对照裁决 |
| --- | ---: | ---: | --- | --- |
| Chameleon | +0.032456 | +0.032602 | 9/1 | 强于 full SSPNV、AFPNV、random semantic；弱于 semantic-only 与 spatial-only |
| Squirrel | +0.006436 | +0.000592 | 6/4 | 强于 AFPNV/semantic-only/spatial-only/random controls；弱于 full SSPNV |

Branch diagnostics：

| Dataset | semantic prob | spatial prob | bootstrap prob | semantic win | spatial win | bootstrap win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chameleon | 0.488484 | 0.186126 | 0.325391 | 0.483092 | 0.112868 | 0.404040 |
| Squirrel | 0.464119 | 0.212066 | 0.323815 | 0.512209 | 0.138627 | 0.349164 |

裁决：

- BSPNV 比 AFPNV 更合理，但未达到升级门槛；
- 预设停止条件是同时超过 Chameleon semantic-only 与 Squirrel full SSPNV，BSPNV 没做到；
- SSPNV / AFPNV / BSPNV 家族全部降级为 ablation assets；
- 不再继续调该家族的 threshold、temperature、branch bias；
- 下一代方法应回到 S3GCL / GraphECL / PolyGCL 级参考范式，寻找不同训练目标，而不是继续做 SSPNV 小变体。

## 2026-06-29 追加：MPNV-GCL multi-positive gate

已实现 `--method mpnv_gcl`：在 GCN-MLP Natural-View foundation 上，用 dense semantic/spatial multi-positive mask 替代 SSPNV 的单采样 positive。

设计来源：

- S3GCL 的 semantic/spatial positive mask；
- GraphECL 的 MLP inference / graph target 思路；
- 当前 `gcn_mlp_gcl` 的 ego-view 与 graph-view natural-view foundation。

训练目标：

- semantic mask：raw propagation signature KNN，监督 high-pass target；
- spatial mask：原图一跳邻居，监督 low-pass target；
- bootstrap：保留 ego view 与 graph view 的 Natural-View alignment；
- control：`--mpnv-shuffle-positives` 打乱 positive mask 的节点对应关系。

执行设置：

```bash
RUNS_DIR="runs/mpnv_gate_wiki_s0_splits0-2_e50"
DATASETS="Chameleon Squirrel" METHODS="gcn_mlp_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="mpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="mpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--mpnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | Method | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | --- | ---: | ---: | --- | --- |
| Chameleon | MPNV | +0.017105 | +0.019132 | 7/3 | 正向，但 shuffled control 也强 |
| Chameleon | MPNV shuffled | +0.014254 | +0.012411 | 7/3 | 机制对照不够干净 |
| Squirrel | MPNV | +0.015082 | +0.014767 | 10/0 | 当前最强机制信号 |
| Squirrel | MPNV shuffled | +0.000961 | +0.000668 | 5/4 | 接近无效，支持结构化 mask |

裁决：

- MPNV 升级为新的 active-but-risky candidate；
- Squirrel 上 normal 10/10 split micro 正向，且 shuffled-positive control 基本失效，是目前比 SSPNV/BSPNV 更干净的机制证据；
- Chameleon 上 normal 与 shuffled 都正向，说明不能把 Chameleon 当作强机制证明，只能作为性能正信号；
- 当前不能声称 SOTA，下一步必须做 seed1/seed2、Texas/Actor 扩展、homophily safety 与强基线同协议对齐；
- 若 seed1/seed2 后 Squirrel normal-vs-shuffled 差异消失，或收益主要来自 shuffled control，应立即放弃 MPNV 主线。

## 2026-06-29 追加：MPNV seed1/seed2 复核与降级

已执行 Texas/Actor/Chameleon/Squirrel × splits 0-9 × seeds 1/2 × 50 epoch 的 MPNV normal gate，并以 `gcn_mlp_gcl` 为 baseline 汇总。

执行：

```bash
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl mpnv_gcl" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
SEEDS="1 2" \
EPOCHS=50 \
RUNS_DIR="runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

Aggregate vs `gcn_mlp_gcl`：

| Dataset | MPNV F1Mi mean | MPNV F1Ma mean | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Texas | 0.639189 | 0.363024 | +0.002703 | -0.000075 | 10/9 | 近零，不是强信号 |
| Actor | 0.350658 | 0.318501 | -0.001776 | -0.001529 | 12/7 | 均值负向 |
| Chameleon | 0.428838 | 0.421045 | +0.000219 | +0.000657 | 9/11 | seed0 正信号未复现 |
| Squirrel | 0.310711 | 0.300447 | -0.000288 | +0.001016 | 10/10 | seed0 10/10 正向失效 |

裁决：

- MPNV 不再作为 active main idea；
- seed0 的 Squirrel normal-vs-shuffled 现象保留为 diagnostic clue，但不能支撑方法主张；
- Chameleon/Squirrel seed1/2 均未复现稳定优势，因此跳过 shuffled-positive seed1/2 扩展，避免继续消耗算力解释一个已经失败的主线；
- 下一步应设计带无标签选择/回退机制的新候选，或者回到 S3GCL / GraphECL / PolyGCL 级参考范式重做训练目标。

## 2026-06-29 追加：AOMPNV objective activation 小门控

已实现 `--method aompnv_gcl`：将 MPNV 的固定 dense semantic/spatial multi-positive objectives 改为节点级无标签 objective activation。每个节点在 semantic dense InfoNCE、spatial dense InfoNCE、Natural-View bootstrap 三个目标之间路由，路由信号来自相对 self-supervised loss 与 raw-signature confidence。

已验证：

```bash
python -m py_compile train.py summarize_split_study.py src/*.py
python train.py --dataset Texas --method aompnv_gcl --epochs 5 --seed 0 --split-index 0 --runs-dir runs/aompnv_smoke --run-name texas_aompnv_e5 --overwrite
python train.py --dataset Chameleon --method aompnv_gcl --epochs 5 --seed 0 --split-index 0 --runs-dir runs/aompnv_smoke --run-name chameleon_aompnv_e5 --overwrite
```

已执行 Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 1/2 × 50 epoch 小门控，并补 `--aompnv-shuffle-positives` control。输出目录：`runs/mpnv_branch_diag_ta_wiki_s1-2_splits0-2_e50/`。

AOMPNV vs `gcn_mlp_gcl`：

| Dataset | ΔF1Mi | ΔF1Ma | Positive/Zero/Negative F1Mi | 裁决 |
| --- | ---: | ---: | --- | --- |
| Texas | +0.022523 | +0.041834 | 3/2/1 | 性能正向，但 shuffled 同样强 |
| Actor | +0.001864 | -0.001147 | 3/1/2 | 性能弱，机制 control 相对更干净 |
| Chameleon | +0.015351 | +0.017719 | 5/0/1 | 正向，但 shuffled 也正 |
| Squirrel | +0.018892 | +0.017179 | 6/0/0 | 当前最稳性能信号，但 shuffled 也强 |

AOMPNV normal-vs-shuffled：

| Dataset | ΔF1Mi | ΔF1Ma | 裁决 |
| --- | ---: | ---: | --- |
| Texas | -0.000000 | +0.016728 | micro 机制不干净 |
| Actor | +0.009539 | +0.001431 | 结构化版本优于 shuffled，但总增益小 |
| Chameleon | +0.006579 | +0.006398 | 有轻微机制信号 |
| Squirrel | +0.003522 | +0.006586 | 性能稳，但机制差距偏小 |

当前裁决：AOMPNV 暂时升级为 active-but-risky candidate。它修复了 full MPNV 固定加权的部分问题，并在小门控中不弱于 semantic-only/spatial-only dense 分支；但 shuffled control 偏强，不能包装成“结构化 dense positives 已被证明”。下一步必须做 splits 0-9 × seeds 1/2 的 normal/shuffled 硬门控；若 normal 与 shuffled 接近，则降级为 regularization ablation。

## 2026-06-29 追加：AOMPNV 硬门控与放弃

已完成 Texas/Actor/Chameleon/Squirrel x splits 0-9 x seeds 1/2 x 50 epoch 的 AOMPNV normal 与 shuffled-positive control，复用 `runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50/` 中已有的 `gcn_mlp_gcl` baseline。

执行与汇总：

```bash
RUNS_DIR="runs/mpnv_gate_ta_wiki_s1-2_splits0-9_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="aompnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="1 2" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="aompnv_shuffled" EXTRA_ARGS="--aompnv-shuffle-positives" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```

Aggregate：

| Dataset | AOMPNV ΔF1Mi | AOMPNV ΔF1Ma | Positive/Zero/Negative F1Mi | Shuffled ΔF1Mi | Normal - Shuffled ΔF1Mi | 裁决 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Texas | +0.010811 | +0.024572 | 9/5/6 | +0.002703 | +0.008108 | 均值正向但 split-level 不稳 |
| Actor | -0.002829 | -0.006672 | 7/1/12 | -0.010592 | +0.007763 | normal 优于 shuffled，但低于 baseline |
| Chameleon | +0.000658 | +0.000853 | 12/1/7 | +0.011952 | -0.011294 | shuffled 明显更强，机制失败 |
| Squirrel | +0.018348 | +0.017634 | 16/1/3 | +0.004755 | +0.013593 | 唯一较清楚正例 |

裁决：

- AOMPNV 不再作为 active main idea；
- 只有 Squirrel 同时满足较清楚的 normal > baseline 与 normal > shuffled；
- Texas 有正均值但 split-level 正负混杂，不能作为稳定主证据；
- Actor 的 normal-vs-shuffled 差值不能弥补低于 baseline 的事实；
- Chameleon shuffled control 明显强于 normal，直接击穿结构化 positive routing 叙事；
- AOMPNV 保留为 regularization / negative-result ablation，不再继续调 router、branch weight 或 confidence threshold；
- 下一代 candidate 必须换机制，继续以 `gcn_mlp_gcl` 为 strong foundation，并保留 shuffled/random/no-structure control。

## 2026-06-29 追加：SRGNV-GCL split0 early gate 与放弃

已实现 `--method srgnv_gcl`：Structure-Residual Gated Natural-View GCL。该方法保留 GCN-MLP bootstrap，将 graph view 中与 ego view 正交的 structure residual 作为额外蒸馏目标，并用 raw feature propagation residual score 做节点级 gate。`--srgnv-shuffle-residual` 用于打乱 residual target，作为 no-structure control。

执行：

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
| Actor | +0.001974 | +0.016069 | +0.002632 | +0.020925 | shuffled 更强 |
| Chameleon | -0.041667 | -0.076563 | -0.021930 | -0.019895 | 明显失败 |
| Squirrel | -0.005764 | -0.019545 | -0.028818 | -0.022141 | 失败 |

裁决：

- SRGNV 不进入 splits 0-2 扩展；
- graph residual 蒸馏可以提高 residual cosine，但不稳定改善下游节点分类；
- Actor 的唯一正向被 shuffled residual 超过，机制证据失败；
- Chameleon 大幅退化，直接触发停止条件；
- SRGNV 保留为 negative-result ablation，不继续调 residual weight、threshold 或 temperature。

## 2026-06-29 追加：PCNV-GCL prototype calibration 实现与 early gate

已实现 `--method pcnv_gcl`：Prototype-Calibrated Natural-View GCL。该方法保留 `gcn_mlp_gcl` 的 Natural-View bootstrap，并加入 trainable prototypes，让 ego view 与 graph view 在 prototype assignment 空间做双向 stop-gradient consistency。`--pcnv-shuffle-assignments` 用于打乱 assignment target 与节点对应关系，作为机制 control。

执行：

```bash
RUNS_DIR="runs/pcnv_split0_s0_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/split_study_runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/split_study_aggregate_vs_gcn_mlp.csv"
```

Default PCNV vs `gcn_mlp_gcl`：

| Dataset | PCNV ΔF1Mi | PCNV ΔF1Ma | Shuffled ΔF1Mi | Shuffled ΔF1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | +0.027027 | +0.085552 | +0.000000 | -0.036472 | normal 明显更好 |
| Actor | +0.007895 | +0.007423 | +0.013158 | +0.022509 | shuffled 更强 |
| Chameleon | +0.028509 | +0.029835 | +0.030702 | +0.033567 | shuffled 更强 |
| Squirrel | -0.014409 | -0.019669 | -0.016330 | -0.014506 | baseline 失败 |

Sharpened PCNV 追加执行：

```bash
RUNS_DIR="runs/pcnv_sharp_split0_s0_e50"
EXTRA_ARGS="--pcnv-prototype-weight 0.5 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.1 --pcnv-target-temperature 0.03"
DATASETS="Texas Chameleon" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Chameleon" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="$EXTRA_ARGS --pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Actor Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Actor Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="$EXTRA_ARGS --pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
```

Sharpened 结果：

| Dataset | Normal F1Mi/F1Ma | Shuffled F1Mi/F1Ma | 观察 |
| --- | ---: | ---: | --- |
| Texas | 0.729730 / 0.459091 | 0.702703 / 0.414448 | 当前最强 Texas 单点，normal > shuffled |
| Actor | 0.351316 / 0.306585 | 0.348026 / 0.319835 | micro normal 略高，macro shuffled 更高 |
| Chameleon | 0.449561 / 0.442837 | 0.438596 / 0.429895 | normal > shuffled，但弱于 default PCNV |
| Squirrel | 0.295869 / 0.273211 | 0.315082 / 0.298631 | 明显失败，且 shuffled 更强 |

裁决：

- PCNV 暂时保留为 active-but-risky candidate；
- default PCNV 的性能信号强于 SRGNV，但 Actor/Chameleon 的 shuffled control 不干净；
- sharpened PCNV 在 Texas 上给出强信号，但固定尖锐 assignment 导致 Chameleon/Squirrel 存在 prototype collapse 风险；
- 不能声称 PCNV 已足以作为顶会/顶刊主方法；
- 下一步只能实现 entropy-guarded / adaptive prototype calibration；如果 normal-vs-shuffled 与 usage entropy 问题不能同时改善，应放弃 prototype calibration 主线。

## 2026-06-29 追加：Guarded PCNV 复核与降级

已新增 PCNV guarded 参数：

- `--pcnv-min-target-confidence`
- `--pcnv-confidence-power`
- `--pcnv-entropy-guard`
- `--pcnv-min-usage-entropy-frac`
- `--pcnv-entropy-guard-temperature`
- `--pcnv-min-view-agreement`
- `--pcnv-view-agreement-power`

Sharp guarded PCNV：

```bash
RUNS_DIR="runs/pcnv_guarded_split0_s0_e50"
EXTRA_ARGS="--pcnv-prototype-weight 0.5 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.1 --pcnv-target-temperature 0.03 --pcnv-min-target-confidence 0.2 --pcnv-confidence-power 1.0 --pcnv-entropy-guard --pcnv-min-usage-entropy-frac 0.5 --pcnv-entropy-guard-temperature 0.1"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="$EXTRA_ARGS --pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
```

| Dataset | Normal ΔF1Mi/ΔF1Ma | Normal - shuffled ΔF1Mi/ΔF1Ma | Usage entropy | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | +0.054054 / +0.125000 | +0.081081 / +0.125427 | 1.259613 | 强正向 |
| Actor | -0.007895 / +0.001567 | -0.006579 / -0.006120 | 0.707327 | 失败 |
| Chameleon | +0.024123 / +0.020505 | +0.024123 / +0.024941 | 0.396174 | 性能正，但坍塌严重 |
| Squirrel | -0.012488 / -0.010562 | +0.007685 / +0.015068 | 0.311565 | baseline 失败 |

Soft guarded PCNV：

```bash
RUNS_DIR="runs/pcnv_soft_guarded_split0_s0_e50"
EXTRA_ARGS="--pcnv-prototype-weight 0.2 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.2 --pcnv-target-temperature 0.1 --pcnv-min-target-confidence 0.12 --pcnv-confidence-power 1.0 --pcnv-entropy-guard --pcnv-min-usage-entropy-frac 0.65 --pcnv-entropy-guard-temperature 0.15"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="$EXTRA_ARGS --pcnv-shuffle-assignments" OVERWRITE=1 bash scripts/run_split_study.sh
```

| Dataset | Normal ΔF1Mi/ΔF1Ma | Normal - shuffled ΔF1Mi/ΔF1Ma | Usage entropy | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | +0.000000 / +0.105497 | +0.054054 / +0.176984 | 2.520100 | macro 强正向 |
| Actor | +0.006579 / +0.011453 | +0.007895 / +0.002728 | 2.200305 | 小幅正向 |
| Chameleon | +0.004386 / +0.007861 | -0.013158 / -0.012230 | 1.083819 | shuffled 反超 |
| Squirrel | -0.015370 / -0.023046 | -0.014409 / -0.012190 | 0.947448 | 失败 |

View-agreement gated PCNV：

```bash
RUNS_DIR="runs/pcnv_view_guarded_split0_s0_e50"
EXTRA_ARGS="--pcnv-prototype-weight 0.2 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.2 --pcnv-target-temperature 0.1 --pcnv-min-target-confidence 0.12 --pcnv-confidence-power 1.0 --pcnv-min-view-agreement 0.08 --pcnv-view-agreement-power 1.0 --pcnv-entropy-guard --pcnv-min-usage-entropy-frac 0.65 --pcnv-entropy-guard-temperature 0.15"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl pcnv_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" EXTRA_ARGS="$EXTRA_ARGS" OVERWRITE=1 bash scripts/run_split_study.sh
```

| Dataset | Normal ΔF1Mi/ΔF1Ma | Usage entropy | 裁决 |
| --- | ---: | ---: | --- |
| Texas | +0.000000 / +0.007042 | 2.103996 | 弱于 soft/sharp |
| Actor | -0.002632 / -0.004816 | 1.674498 | 失败 |
| Chameleon | -0.010965 / -0.008340 | 0.551141 | 失败 |
| Squirrel | -0.038425 / -0.033737 | 0.721000 | 明显失败 |

裁决：

- PCNV 不再作为 active main idea；
- soft guarded PCNV 是当前最健康的 prototype 变体，但仍未过 Chameleon shuffled control 与 Squirrel baseline gate；
- sharp guarded PCNV 的 Texas/Chameleon 正向依赖较低 usage entropy，不适合作为通用机制；
- view-agreement gate 明确失败，不补 shuffled；
- 后续不再继续调 PCNV temperature、confidence、entropy 或 view-agreement 参数；
- 下一代方法必须换机制，优先考虑节点级局部结构条件下的 objective selection，或更直接的 downstream separability proxy。

## 2026-06-29 追加：LCOS-GCL local-conflict objective selection 与放弃

已实现 `--method lcos_gcl`：Local-Conflict Objective Selection GCL。该方法使用 raw feature local conflict gate，在完整 graph view alignment 与 high-pass view alignment 之间做节点级 objective selection，并让 final representation 使用 `[ego, (1-gate) graph + gate high]`。

已新增参数：

- `--lcos-route-temperature`
- `--lcos-route-threshold`
- `--lcos-min-branch-weight`
- `--lcos-degree-weight`
- `--lcos-shuffle-gate`

已修复 raw residual 数值问题：从 `||x-Px|| / ||x||` 改为 `||x-Px|| / (||x|| + ||Px||)`，避免 Chameleon 等数据集中零特征/极小范数节点导致 residual 爆炸。

执行：

```bash
RUNS_DIR="runs/lcos_split0_s0_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl lcos_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="lcos_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--lcos-shuffle-gate" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```

结果：

| Dataset | Normal ΔF1Mi | Normal ΔF1Ma | Normal - shuffled ΔF1Mi | Normal - shuffled ΔF1Ma | 裁决 |
| --- | ---: | ---: | ---: | ---: | --- |
| Texas | -0.054054 | +0.072502 | +0.000000 | +0.056593 | micro 失败 |
| Actor | +0.009211 | +0.006041 | +0.001974 | +0.001804 | 弱正向 |
| Chameleon | -0.004386 | -0.006271 | +0.015351 | +0.012920 | baseline 失败 |
| Squirrel | +0.013449 | +0.006619 | +0.050913 | +0.041619 | 唯一清楚机制线索 |

裁决：

- LCOS 第一版不进入 splits 0-2 扩展；
- Squirrel 显示局部冲突 gate 有机制线索，但 Texas/Chameleon baseline gate 失败；
- 直接对齐 high-pass target 的 objective selection 不成立；
- 后续不再继续调 LCOS threshold、temperature 或 degree weight；
- 下一代若继承局部冲突线索，应转向 loss reliability、negative suppression 或 downstream separability proxy，而不是 graph/high alignment 切换。
