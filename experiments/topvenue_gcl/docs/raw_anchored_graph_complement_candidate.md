# Raw-Anchored Graph Complement GCL 候选备忘录

## 研究判断

RAGC-GCL 的核心判断是：在异配图节点分类中，Graph SSL embedding 不应默认替代 raw features。更稳妥的问题是：能否显式保留 raw feature separability，同时让 Natural-View GCL 学到 raw 之外的 graph-context complement。

相关文献边界：

- GRAPE/TheWebConf 2024 已经把 expansive/adaptive hard negative mining 做成强方向，且指出普通 hard negative 思路在 GCL 中会受 message passing 影响。
- 2026 年 SPGCL 重新讨论 positive samples 与 message passing pre-alignment，说明仅继续微调 positive/negative 权重很容易落入已有叙事。
- 因此本候选从“再设计一个 reliability 权重”转向 output-level raw-anchor complement：先保护 raw，再检验 learned graph context 是否有稳定增量。

## 当前实现

入口：

```bash
python train.py --dataset Chameleon --method ragc_gcl --epochs 50 --split-index 0 --seed 0
python train.py --dataset Chameleon --method raw_features --epochs 50 --split-index 0 --seed 0
```

实现方式：

- `raw_features`：不训练模型，直接用 normalized raw features 做 linear probe。
- `ragc_gcl`：训练目标与 `gcn_mlp_gcl` 一致，使用 ego/graph Natural-View bootstrap。
- 最终表示：`concat(ragc_raw_weight * normalize(raw_x), ragc_learned_weight * normalize(learned_embedding))`。
- 默认 `ragc_raw_weight=1.0`，`ragc_learned_weight=1.0`。
- `--ragc-control shuffle`：保留 learned embedding 分布但打乱节点对应关系。
- `--ragc-control random`：用同维随机向量替代 learned branch。

## 初筛结果

### split0 seed0 50 epoch

输出目录：`runs/ragc_split0_e50/`

RAGC vs `raw_features`：

- Texas：持平，0.783784 vs 0.783784。
- Actor：+0.004605 F1Mi。
- Chameleon：+0.026316 F1Mi。
- Squirrel：+0.012488 F1Mi。

RAGC vs `gcn_mlp_gcl`：

- Texas：+0.108108 F1Mi。
- Actor：+0.016447 F1Mi。
- Chameleon：+0.030702 F1Mi。
- Squirrel：+0.023055 F1Mi。

### splits0-2 seed0 50 epoch

输出目录：`runs/ragc_splits0-2_e50/`

RAGC vs `raw_features`：

| Dataset | Delta F1Mi mean | Delta F1Ma mean | Positive/negative splits |
| --- | ---: | ---: | ---: |
| Actor | +0.009649 | +0.008234 | 3/0 |
| Chameleon | +0.031433 | +0.032426 | 3/0 |
| Squirrel | +0.013128 | +0.020016 | 3/0 |
| Texas | -0.009009 | -0.057181 | 0/1 |

### learned-branch control gate

输出目录：`runs/ragc_control_splits0-2_e50/`

RAGC normal vs `raw_features`：

| Dataset | Delta F1Mi mean | Delta F1Ma mean | Positive/negative splits |
| --- | ---: | ---: | ---: |
| Actor | +0.011184 | +0.009989 | 3/0 |
| Chameleon | +0.027047 | +0.029025 | 3/0 |
| Squirrel | +0.016651 | +0.024313 | 3/0 |
| Texas | -0.018018 | -0.070973 | 0/1 |

Control vs `raw_features`：

| Dataset | Shuffle ΔF1Mi | Random ΔF1Mi | Normal - Shuffle | Normal - Random |
| --- | ---: | ---: | ---: | ---: |
| Actor | -0.007675 | -0.020614 | +0.018860 | +0.031798 |
| Chameleon | -0.024123 | -0.073099 | +0.051170 | +0.100146 |
| Squirrel | -0.015690 | -0.039065 | +0.032341 | +0.055716 |
| Texas | -0.027027 | -0.117117 | +0.009009 | +0.099099 |

解释：

- Actor/Chameleon/Squirrel 上，normal 同时超过 raw、shuffle 与 random；shuffle/random 均低于 raw-only。
- Chameleon/Squirrel 的 normal-control gap 很大，支持 learned graph context 具有节点对应的互补信号，而不是高维拼接或验证集 C 搜索造成的假增益。
- Texas 仍低于 raw-only，但 normal 也高于 shuffle/random；该数据集主要暴露 safety selector 问题，而不是 learned branch 完全无效。

### splits0-9 seed0 50 epoch

输出目录：`runs/ragc_s0_splits0-9_e50/`

RAGC normal vs `raw_features`：

| Dataset | Raw F1Mi | RAGC F1Mi | ΔF1Mi | ΔF1Ma | Positive/negative splits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor | 0.351711 | 0.361118 | +0.009408 | +0.003346 | 9/1 |
| Chameleon | 0.457895 | 0.474781 | +0.016886 | +0.018412 | 9/1 |
| Squirrel | 0.330740 | 0.338136 | +0.007397 | +0.009380 | 9/1 |
| Texas | 0.808108 | 0.813514 | +0.005405 | +0.000600 | 5/2 |

解释：

- 固定 RAGC 在四个目标异配数据集的 10-split 均值上均高于 raw-only。
- Actor/Chameleon/Squirrel 的 split 级方向较稳定，均为 9/10 split 正向、1/10 split 负向。
- Texas 从 splits0-2 的负向转为 10-split 均值微正，但只有 5/10 split 正向、2/10 负向、其余持平；因此 WebKB safety 仍不能忽略。

### Wiki 10-split learned-branch controls

输出目录：`runs/ragc_controls_wiki_s0_splits0-9_e50/`

| Dataset | Normal F1Mi | Raw F1Mi | Shuffle F1Mi | Random F1Mi | Normal - Shuffle | Normal - Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chameleon | 0.474781 | 0.457895 | 0.434211 | 0.383114 | +0.040570 | +0.091667 |
| Squirrel | 0.338136 | 0.330740 | 0.319500 | 0.299712 | +0.018636 | +0.038425 |

解释：

- Chameleon/Squirrel 上，normal learned branch 明显优于 shuffle 与 random controls。
- random control 显著低于 raw-only，排除“额外 256 维随机特征/维度扩张”解释。
- shuffle control 保留 learned embedding 分布但破坏节点对应关系；normal-shuffle gap 说明节点对应的 graph complement 是核心信号。
- Squirrel split3 中 shuffle 高于 normal，是需要保留的局部反例；当前机制主张应基于均值与大多数 split，而不是声称逐 split 全胜。

## Safety selector 尝试

已实现 `--method ragc_auto_gcl`：

- 训练阶段与 RAGC 相同。
- 输出阶段在 `raw_features`、learned-only Natural-View embedding 与 `raw+learned` 三个候选表示之间用验证集 F1Mi 选择。
- 新增 `ragc_auto_min_val_margin`，默认 0.02；只有当 learned-only 或 RAGC 验证 F1Mi 至少超过 raw 0.02 时才允许选择非 raw 候选。

裁决：

- 无 margin 的 validation selector 在 Actor split5 上失败：验证集选择 RAGC，但测试低于 raw。
- margin=0.02 修复两个已知负例：Actor split5 与 Texas split1 均回退 raw。
- 三候选版本在 Texas split1 smoke 中可用：raw val=0.796610，RAGC val=0.779661，learned val=0.644068，选择 raw，测试 F1Mi=0.918919。
- Planetoid 当前在 `eval_mode=auto` 下走 random linear-probe evaluation，不使用 mask validation；因此 `ragc_auto_gcl` 对 Cora/CiteSeer/PubMed 会落入 `no_validation_mask` fallback，不能作为 Planetoid auto-selection 证据。
- margin 可能牺牲多个小幅正向 split，因此 `ragc_auto_gcl` 暂作为 safety ablation，不替代固定 RAGC 主方法。

## Homophily safety

输出目录：`runs/ragc_homophily_s0-4_e50/`

设置：Cora/CiteSeer/PubMed × split0 × seeds0-4 × 50 epoch。当前 Planetoid 在本工作区使用 random linear-probe evaluation，因此这里的 seed 表示 linear-probe split seed 与模型 seed共同变化，不是多官方 split。

RAGC vs `raw_features`：

| Dataset | Raw F1Mi | RAGC F1Mi | ΔF1Mi | ΔF1Ma | Positive/negative seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.634783 | 0.711348 | +0.076565 | +0.080923 | 5/0 |
| CiteSeer | 0.648837 | 0.690640 | +0.041803 | +0.032818 | 5/0 |
| PubMed | 0.846936 | 0.858868 | +0.011931 | +0.011212 | 5/0 |

RAGC vs learned-only `gcn_mlp_gcl`：

| Dataset | GCN-MLP F1Mi | RAGC F1Mi | ΔF1Mi | ΔF1Ma | Positive/negative seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.763905 | 0.711348 | -0.052557 | -0.065799 | 0/5 |
| CiteSeer | 0.692933 | 0.690640 | -0.002293 | +0.003085 | 2/3 |
| PubMed | 0.843033 | 0.858868 | +0.015835 | +0.016020 | 5/0 |

解释：

- RAGC 明确通过 homophily non-degradation against raw-only，且三组 Planetoid 数据均是 5/5 seed 正向。
- 但 RAGC 并不在所有 homophily 数据上优于 learned-only；Cora 上 learned-only Natural-View 明显更强。
- PubMed 上 RAGC 同时超过 raw-only 与 learned-only，说明 raw+graph complement 在大同配图上仍可能有价值。
- 论文表述应避免“homophily SOTA”口径，更适合说：RAGC 在 homophily 上对 raw-only 安全，且在 PubMed 上展示 complement gain；Cora 暴露了 fixed concatenation 的上界问题。

## 当前裁决

RAGC-GCL 升级为当前最强 active candidate，但仍不是最终成功主方法。

保留理由：

- 在 Actor/Chameleon/Squirrel 三个异配数据集上，RAGC 对 raw-only 的增量在 splits0-2 全部为正。
- 在 Actor/Chameleon/Squirrel/Texas 的 10-split seed0 均值上，RAGC 对 raw-only 全部为正。
- 在 Cora/CiteSeer/PubMed 的 5-seed random-probe safety 中，RAGC 对 raw-only 全部为正。
- Chameleon/Squirrel 正是许多前序候选的失败边界，本候选在这两个数据集上同时超过 raw-only 与 learned-only。
- learned-branch shuffle/random controls 在 Chameleon/Squirrel 的 10-split 上明显失败，说明 normal learned branch 的节点对应关系有机制价值。
- 论文切口比“再调 reliability weight”更清楚：raw feature anchor 负责安全性，graph SSL branch 只证明 complement value。

主要风险：

- Texas 10-split 均值微正但 split 级不稳定，WebKB 小图仍需要 safety selector 或 learned-branch weight control。
- 当前主要是 seed0 split study，仍需多 seed 与 homophily safety。
- Cora 上 learned-only 明显强于 RAGC，说明 fixed concatenation 不是全局最优；若要冲顶会，需要协议一致的 raw/learned/RAGC selector 或更细的 learned-branch scaling。

## 下一步硬门槛

必须补做：

- Cora/CiteSeer/PubMed homophily safety。
- Actor/Texas 的 10-split shuffle/random controls。
- 多 seed 或更标准的 split/seed 分离复核。
- 与 `gcn_mlp_gcl`、GRACE、SSPNV/BSPNV 等已实现强候选固化为同表比较。
- 设计可用于 Planetoid random-probe 协议的 selector，或明确将 selector 只用于具有 validation mask 的 transductive split。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Actor Chameleon Squirrel Texas" METHODS="raw_features ragc_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/ragc_s0_splits0-9_e50" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir runs/ragc_s0_splits0-9_e50 --baseline-method raw_features --out runs/ragc_s0_splits0-9_e50/runs_vs_raw.csv --aggregate-out runs/ragc_s0_splits0-9_e50/aggregate_vs_raw.csv
DATASETS="Cora CiteSeer PubMed" METHODS="raw_features ragc_gcl" SPLITS="0" SEEDS="0 1 2 3 4" EPOCHS=50 RUNS_DIR="runs/ragc_homophily_s0-4_e50" OVERWRITE=1 bash scripts/run_split_study.sh
```
