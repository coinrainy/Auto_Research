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
