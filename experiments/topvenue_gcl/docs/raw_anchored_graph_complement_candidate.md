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

## 当前裁决

RAGC-GCL 升级为新的 active candidate，但仍不是最终成功主方法。

保留理由：

- 在 Actor/Chameleon/Squirrel 三个异配数据集上，RAGC 对 raw-only 的增量在 splits0-2 全部为正。
- Chameleon/Squirrel 正是许多前序候选的失败边界，本候选在这两个数据集上同时超过 raw-only 与 learned-only。
- 论文切口比“再调 reliability weight”更清楚：raw feature anchor 负责安全性，graph SSL branch 只证明 complement value。

主要风险：

- Texas splits0-2 平均为负，尤其 macro 退化明显，WebKB 小图需要 safety selector。
- 当前还没有 learned-branch shuffled/random control，无法排除高维拼接或验证集 C 搜索带来的偶然收益。
- 当前没有 homophily safety；Cora/CiteSeer/PubMed 上若 raw-anchor 拼接拖累性能，方法仍不能作为通用 GCL。

## 下一步硬门槛

必须补做：

- `ragc_gcl` vs `raw_features` 的 splits0-9 seed0。
- `ragc_gcl` vs learned-branch shuffled/random control，确认 learned context 的节点对应关系有效。
- Texas/WebKB safety selector：当 learned complement 不可靠时回退 raw-only 或降低 learned weight。
- Cora/CiteSeer/PubMed homophily safety。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Actor Chameleon Squirrel Texas" METHODS="raw_features ragc_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="runs/ragc_s0_splits0-9_e50" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir runs/ragc_s0_splits0-9_e50 --baseline-method raw_features --out runs/ragc_s0_splits0-9_e50/runs_vs_raw.csv --aggregate-out runs/ragc_s0_splits0-9_e50/aggregate_vs_raw.csv
```
