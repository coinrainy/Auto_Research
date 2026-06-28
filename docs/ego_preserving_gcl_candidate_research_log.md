# Ego-Preserving / Graph-Usage Calibrated GCL 候选研究日志

日期：2026-06-28

## 候选问题

前面的 reliability weighting、false-negative attenuation、spectral mix、prototype objective、RR/CBR 与多种 gate 都只带来小幅或条件性收益。新的证据显示，很多异配图失败并非主要来自 contrastive loss 的样本权重，而是来自 GRACE 默认 GCN encoder 的低通邻域聚合：节点自身特征中的类别语义被异类邻居混合冲淡。

本轮候选假设：

> 在异配图节点分类的 GCL 中，encoder 必须显式保留 ego-feature channel，并校准 graph propagation 的使用强度；否则标准 GCN-based GRACE 会在少数类/弱类上产生系统性类别覆盖失败。

## 当前实现

实现位置：

- `experiments/grace_idea/model.py`
- `experiments/grace_idea/train.py`

新增方法：

- `--method ego_grace`：纯 ego-feature MLP encoder，忽略 edge_index，只保留 GRACE 的 feature-drop 双视图和 InfoNCE。
- `--method residual_grace`：GCN branch + ego MLP branch，使用可学习全局 scalar gate 融合。
- `--method gated_ego_graph_grace`：GCN branch + ego MLP branch，使用节点级 local feature-neighborhood agreement gate 融合。

`gated_ego_graph_grace` 的当前 gate：

- 计算每个节点与邻居均值特征的 cosine agreement；
- 对 agreement 做图内标准化；
- `graph_gate = sigmoid((score - threshold) / temperature)`；
- graph gate 越高，越偏向 GCN branch；越低，越偏向 ego branch；
- 当前默认：`--graph-gate-temperature 0.5 --graph-gate-threshold 0.0`。

## 关键实验结果

### 1. `residual_grace` 异配 10 split 复核

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-9 × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/residual_grace_splits0-9_seed0_e100`
- `experiments/grace_idea/runs/summaries/residual_grace_splits0-9_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta | F1Mi positive/zero/negative | F1Ma positive/zero/negative |
| --- | ---: | ---: | --- | --- |
| Actor | +0.064868 | +0.090895 | 10/0/0 | 10/0/0 |
| Cornell | +0.172973 | +0.183641 | 10/0/0 | 10/0/0 |
| Texas | +0.102703 | +0.188203 | 9/1/0 | 10/0/0 |
| Wisconsin | +0.180392 | +0.199546 | 10/0/0 | 10/0/0 |

诊断：这是目前所有候选中最强、最稳定的异配信号。尤其 Actor 也稳定正向，解决了 CBR/RR 系列一直存在的 Actor 退化问题。

最终 `ego_gate` 基本停在 0.5 附近：

| Dataset | ego_gate mean |
| --- | ---: |
| Texas | 0.498319 |
| Cornell | 0.497999 |
| Wisconsin | 0.498898 |
| Actor | 0.492705 |

说明收益主要来自稳定的 ego + graph 双通道表示容量，而不是 scalar gate 学到了强数据集特化。

### 2. `residual_grace` 同配 quick sanity

实验设置：Cora/CiteSeer/PubMed × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/summaries/residual_grace_homophily_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.008477 | -0.009624 |
| CiteSeer | +0.006678 | +0.012899 |
| PubMed | +0.012003 | +0.014253 |

诊断：`residual_grace` 没有系统性同配退化，但 Cora 有约 0.8-1.0 个点下降，后续需要 10 seeds/splits 复核，并加入 homophily-safe gate 或 residual schedule。

### 3. `ego_grace` MLP-only ablation

实验设置：Texas/Cornell/Wisconsin/Actor × splits 0-2 × seed0 × 100 epochs。

输出：

- `experiments/grace_idea/runs/summaries/ego_grace_splits0-2_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | +0.067763 | +0.084266 |
| Cornell | +0.252252 | +0.261006 |
| Texas | +0.135135 | +0.221388 |
| Wisconsin | +0.228758 | +0.345109 |

与 `residual_grace` 的 split0-2 对比：

| Dataset | ego - residual F1Mi | ego - residual F1Ma |
| --- | ---: | ---: |
| Actor | +0.006140 | +0.007301 |
| Cornell | +0.036036 | +0.000088 |
| Texas | +0.045045 | +0.071625 |
| Wisconsin | +0.039216 | +0.099954 |

诊断：当前异配收益的核心不是“多加一点 RR/weighting”，而是 ego-feature preservation。纯 MLP-only 在小异配图上非常强，这既是机会也是风险：论文不能简单声称 graph branch 本身带来全部收益，必须证明 graph usage 的校准何时有益。

### 4. `gated_ego_graph_grace` 初版

异配 split0-2：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | +0.066009 | +0.084107 |
| Cornell | +0.261261 | +0.280904 |
| Texas | +0.180180 | +0.298539 |
| Wisconsin | +0.209150 | +0.200034 |

和 ego/residual 对比：

| Dataset | Best current note |
| --- | --- |
| Texas | gated 最高 |
| Cornell | gated 略高于 ego/residual |
| Wisconsin | ego macro 明显最高，gated micro 仍强 |
| Actor | gated 与 ego 接近 |

graph gate 诊断：

| Dataset | graph_gate mean | graph_gate std |
| --- | ---: | ---: |
| Texas | 0.501034 | 0.342162 |
| Cornell | 0.503500 | 0.349249 |
| Wisconsin | 0.507854 | 0.342581 |
| Actor | 0.427728 | 0.159707 |

同配 quick sanity：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.201805 | -0.241269 |
| CiteSeer | -0.056984 | -0.053311 |
| PubMed | -0.037173 | -0.031199 |

诊断：当前 `gated_ego_graph_grace` v1 异配很强，但同配安全性失败，不能直接作为最终主方法。它说明 local feature-neighborhood agreement gate 有异配收益，但 homophily-safe graph usage calibration 还没解决。

## 当前判断

当前最有潜力的方向不是继续改 contrastive pair weighting，而是转向：

> Ego-Preserving Graph Contrastive Learning / Graph-Usage Calibrated GCL。

但本轮新增 raw-feature baseline 后，必须进一步收缩说法：

- `residual_grace` 是当前最稳的 encoder-level GCL diagnostic candidate：异配 10 split 全面正向，同配 quick sanity 基本安全。
- `ego_grace` 是必须保留的强 ablation：它证明 ego-feature preservation 是关键机制，但也会被 reviewer 质疑“这还是 graph learning 吗”。
- `gated_ego_graph_grace` 是下一版方法雏形：异配强，但同配失败，必须加入 homophily-safe fallback 或 dataset/region-level graph usage calibration。
- `raw_features` 线性分类是必要强 baseline：WebKB 上 raw features 明显强于 ego/residual，Actor 上也与 ego 接近。因此当前 ego/residual 不能直接作为 SOTA 方法写，只能作为暴露 graph propagation failure 的机制证据。

不应声称：

- 不能说当前 calibrated gate 已经 SOTA-ready；
- 不能说 graph branch 当前一定优于 ego-only；
- 不能忽略 MLP-only baseline，否则论文很容易被 reviewer 击中。

## 下一步

优先级从高到低：

1. 跑 `ego_grace` 的 10 split 异配复核，与 `residual_grace` 的 10 split 对齐；
2. 为 `gated_ego_graph_grace` 设计 homophily-safe gate，例如 graph gate lower bound / homophily detector / validation-free global graph-use prior；
3. 实现 `mlp_only supervised/linear` 或 feature-only SSL baseline，确认不是简单特征分类器已经足够；
4. 对 `residual_grace` 做 Cora/CiteSeer/PubMed 多 seed 复核；
5. 加入 Chameleon/Squirrel，测试这个 encoder-level idea 是否能扩展到更大异配图；
6. 机制诊断：按 class、degree、local feature agreement 分桶，证明 ego-preserving channel 恢复了低图一致性节点或少数类覆盖。

## 2026-06-28 追加压力测试

### 5. `ego_grace` 异配 10 split 复核

输出：

- `experiments/grace_idea/runs/summaries/ego_grace_splits0-9_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta | F1Mi positive/zero/negative | F1Ma positive/zero/negative |
| --- | ---: | ---: | --- | --- |
| Actor | +0.070855 | +0.099576 | 10/0/0 | 10/0/0 |
| Cornell | +0.172973 | +0.141093 | 10/0/0 | 9/0/1 |
| Texas | +0.124324 | +0.232101 | 10/0/0 | 10/0/0 |
| Wisconsin | +0.225490 | +0.274924 | 10/0/0 | 10/0/0 |

与 `residual_grace` 对齐：

| Dataset | ego - residual F1Mi | ego - residual F1Ma | ego better F1Mi splits | ego better F1Ma splits |
| --- | ---: | ---: | ---: | ---: |
| Texas | +0.018919 | +0.043232 | 7/10 | 7/10 |
| Cornell | -0.008108 | -0.045063 | 4/10 | 2/10 |
| Wisconsin | +0.043137 | +0.074671 | 7/10 | 8/10 |
| Actor | +0.006053 | +0.007432 | 8/10 | 7/10 |

判断：ego-only 在 Texas/Wisconsin/Actor 更强，Cornell 上 residual 更稳。这说明最终方法不应固定为 ego-only 或 residual，而应解决“何时使用 graph propagation”的 calibration 问题。

### 6. `gated_ego_graph_grace --graph-gate-min 0.5` 同配压力测试

输出：

- `experiments/grace_idea/runs/summaries/gated_ego_graph_grace_min05_homophily_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.184167 | -0.224984 |
| CiteSeer | -0.052198 | -0.045933 |
| PubMed | -0.038056 | -0.031795 |

判断：简单提高 graph gate 下界不能修复同配退化。当前 local feature-neighborhood agreement 的节点级路由方向暂时判为失败，不继续调 `graph_gate_min/max`。

### 7. raw-feature baseline

新增脚本：

```bash
python evaluate_raw_features.py --dataset Texas --split-index 0 --out-dir runs/raw_features/Texas_split0
```

输出：

- `experiments/grace_idea/runs/summaries/raw_features_heterophily_splits0-9.csv`

raw feature 10 split 平均：

| Dataset | raw F1Mi | raw F1Ma |
| --- | ---: | ---: |
| Texas | 0.808108 | 0.643405 |
| Cornell | 0.735135 | 0.513742 |
| Wisconsin | 0.837255 | 0.610948 |
| Actor | 0.351711 | 0.328359 |

raw - ego：

| Dataset | F1Mi | F1Ma | raw better F1Mi splits | raw better F1Ma splits |
| --- | ---: | ---: | ---: | ---: |
| Texas | +0.094595 | +0.149131 | 10/10 | 10/10 |
| Cornell | +0.072973 | +0.074007 | 9/10 | 8/10 |
| Wisconsin | +0.068627 | +0.069676 | 9/10 | 8/10 |
| Actor | -0.007237 | +0.000659 | 1/10 | 5/10 |

raw - residual：

| Dataset | F1Mi | F1Ma | raw better F1Mi splits | raw better F1Ma splits |
| --- | ---: | ---: | ---: | ---: |
| Texas | +0.113514 | +0.192363 | 10/10 | 10/10 |
| Cornell | +0.064865 | +0.028944 | 8/10 | 8/10 |
| Wisconsin | +0.111765 | +0.144347 | 10/10 | 10/10 |
| Actor | -0.001184 | +0.008090 | 4/10 | 6/10 |

判断：WebKB 上 raw features 明显强于当前 SSL encoder；Actor 上 ego/residual 仍有一定 micro 增益但 macro 与 raw 接近。因此当前 idea 不能以 WebKB accuracy SOTA 作为主卖点。下一步必须转向：

1. feature-anchored GCL：显式保留 raw features，并证明 SSL embedding 提供 raw 之外的增量；
2. 更强/更大异配数据集，如 Chameleon/Squirrel；
3. 机制论文路线：证明 GCN-based GCL 在强 feature heterophily 数据集上会损害 raw-feature separability，并提出 graph usage calibration 作为诊断/修正框架。

### 8. 未完成的 concat post-hoc

尝试评估 `raw features + SSL embedding` 拼接表示，但高维 logistic regression 评估明显变慢并出现收敛 warning，已中断。本结果不作为证据。下一步若继续 concat，应实现专门的快速 evaluator，并明确 solver/max_iter/normalization。

建议下一步命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python evaluate_raw_features.py --dataset Actor --split-index 0 --out-dir runs/raw_feature_smoke/Actor_split0
```

研发下一步不是重复跑 ego 10 split，而是实现一个快速、可控的 `raw + SSL embedding` / residual-to-raw evaluator，判断 GCL 表示是否能在 raw feature 之外提供增量；若不能，该路线应转为机制诊断论文或继续换 idea。

## 2026-06-28 raw + SSL fusion 增量评估

新增脚本：

```bash
python evaluate_feature_fusion.py --runs-dir runs/ego_grace_splits0-9_seed0_e100 --include-methods ego_grace
```

脚本功能：

- 递归读取 `artifacts.pt`；
- 用同一 dataset/split 的 raw `data.x` 与已保存 SSL embedding 重新评估；
- 支持 `raw`、`ssl`、`concat` 三种表示；
- 对 concat 采用 raw block 与 SSL block 分别 L2 normalize 后拼接；
- 输出 per-run 表、paired delta 表与 dataset aggregate 表。

### 完整 C 网格：split0-2 校准

输出：

- `experiments/grace_idea/runs/summaries/feature_fusion_ego_residual_splits0-2_aggregate.csv`
- `experiments/grace_idea/runs/summaries/feature_fusion_residual_splits0-2_aggregate.csv`

`ego_grace` concat - raw：

| Dataset | F1Mi | F1Ma | F1Mi positive/zero/negative | F1Ma positive/zero/negative |
| --- | ---: | ---: | --- | --- |
| Actor | +0.019298 | +0.014279 | 3/0/0 | 3/0/0 |
| Cornell | +0.018018 | +0.006552 | 2/1/0 | 1/1/1 |
| Texas | +0.018018 | +0.036499 | 1/2/0 | 1/2/0 |
| Wisconsin | -0.013072 | -0.020250 | 0/2/1 | 0/2/1 |

`residual_grace` concat - raw：

| Dataset | F1Mi | F1Ma | F1Mi positive/zero/negative | F1Ma positive/zero/negative |
| --- | ---: | ---: | --- | --- |
| Actor | +0.016228 | +0.018583 | 3/0/0 | 3/0/0 |
| Cornell | -0.009009 | -0.008758 | 1/1/1 | 2/0/1 |
| Texas | -0.036036 | -0.049192 | 0/1/2 | 0/1/2 |
| Wisconsin | -0.026144 | -0.027646 | 0/1/2 | 0/1/2 |

判断：完整 C 网格下，后验 concat 不是稳健方法。`ego_grace` 在 Actor/Cornell/Texas 有增量，但 Wisconsin 负；`residual_grace` 只有 Actor 稳定正向，在 WebKB 小图上反而更容易损伤 raw separability。

### 固定 C=1：10 split 快速筛查

输出：

- `experiments/grace_idea/runs/summaries/feature_fusion_ego_splits0-9_fast_aggregate.csv`
- `experiments/grace_idea/runs/summaries/feature_fusion_residual_splits0-9_fast_aggregate.csv`

固定 `C=1` 的快速筛查中，concat - raw 在 4 个异配数据集上均为正：

| Dataset | ego F1Mi/F1Ma | residual F1Mi/F1Ma |
| --- | ---: | ---: |
| Actor | +0.014013 / +0.016407 | +0.006645 / +0.008116 |
| Cornell | +0.045946 / +0.051876 | +0.064865 / +0.084247 |
| Texas | +0.040541 / +0.072808 | +0.062162 / +0.125084 |
| Wisconsin | +0.017647 / +0.028493 | +0.019608 / +0.047159 |

判断：这说明 SSL embedding 可能包含 raw 之外的互补信息，但该信号对 evaluation C 搜索敏感，不能把 post-hoc concat 当成最终方法。更合理的新候选是：

> Raw-Anchored Residual/Complement GCL：显式把 raw feature 作为 anchor，把 GCL encoder 学到的表示约束为 raw 之外的补充通道，并通过稳定的 validation-free 或 light-validation 融合机制避免破坏 raw separability。

当前路线取舍：

- 放弃把 `ego_grace` / `residual_grace` 单独包装成 SOTA encoder；
- 放弃继续手调 `gated_ego_graph_grace` 的 local agreement gate；
- 保留 ego/residual 作为机制证据：GCN-based GRACE 损害异配图上的 raw-feature separability，ego preservation 能恢复部分语义；
- 下一步应实现显式 residual-complement 训练或融合，而不是只做后验 concat。

## 2026-06-28 Raw-Anchored Complement GCL 原型

新增方法：

- `--method raw_complement_gcl`
- encoder 输出 hidden `[raw_anchor, complement]`；
- `raw_anchor` 来自 ego MLP；
- `graph_context` 来自 GCN encoder；
- `complement = LayerNorm(graph_context - stop_gradient(raw_anchor))`；
- 训练损失为 GRACE InfoNCE + `raw_complement_weight * corr(raw_anchor, complement)^2`；
- final representation 的默认 `anchor` 模式为 `[normalized raw features, normalized learned complement]`。

新增参数：

- `--raw-complement-weight`，默认 `0.05`；
- `--raw-complement-detach-anchor / --no-raw-complement-detach-anchor`；
- `--raw-complement-eval-mode anchor|hidden|graph`，用于测试 raw-anchor、hidden concat 与 graph fallback。

### 异配 split0-2 初筛

输出：

- `runs/summaries/raw_complement_anchor_splits0-2_seed0_e100_aggregate.csv`
- `runs/summaries/raw_complement_anchor_vs_raw_splits0-2_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta | positive splits |
| --- | ---: | ---: | --- |
| Actor | +0.072807 | +0.094350 | 3/3 |
| Cornell | +0.297297 | +0.331202 | 3/3 |
| Texas | +0.252252 | +0.376704 | 3/3 |
| Wisconsin | +0.300654 | +0.416371 | 3/3 |

相对 raw feature full-C baseline：

| Dataset | F1Mi delta | F1Ma delta | note |
| --- | ---: | ---: | --- |
| Actor | +0.014035 | +0.006336 | 小幅正向 |
| Cornell | ~0.000000 | -0.008090 | 基本持平 |
| Texas | +0.009009 | -0.044913 | micro 持平/略正，macro 仍低 |
| Wisconsin | -0.006536 | -0.003619 | 基本持平/略负 |

### 异配 split0-9 扩展

输出：

- `runs/summaries/raw_complement_anchor_vs_grace_raw_splits0-9_seed0_e100.csv`
- `runs/summaries/raw_complement_anchor_vs_grace_raw_splits0-9_seed0_e100_aggregate.csv`

相对 GRACE：

| Dataset | F1Mi delta | F1Ma delta | positive splits |
| --- | ---: | ---: | --- |
| Actor | +0.074737 | +0.106628 | 10/10 |
| Cornell | +0.237838 | +0.211600 | 10/10 |
| Texas | +0.221622 | +0.360388 | 10/10 |
| Wisconsin | +0.298039 | +0.353723 | 10/10 |

相对 raw feature full-C baseline：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg |
| --- | ---: | --- | ---: | --- |
| Actor | +0.011118 | 8/0/2 | +0.006394 | 8/0/2 |
| Cornell | -0.008108 | 3/5/2 | -0.003500 | 6/2/2 |
| Texas | +0.002703 | 4/3/3 | -0.020844 | 4/2/4 |
| Wisconsin | +0.003922 | 3/4/3 | +0.009123 | 4/3/3 |

判断：这是目前最强的异配方法信号。它在 4 个异配数据集上相对 GRACE 全 split 正向，并且相对强 raw baseline 基本持平到小幅正向；但它仍不能被称为全面 SOTA，因为 WebKB raw features 本身极强，方法主要贡献应表述为“在不牺牲 raw baseline 的前提下修复 GCN-GRACE 的类别覆盖失败”。

### Homophily safety

输出：

- `runs/raw_complement_anchor_homophily_seed0_e100`
- `/tmp/raw_complement_cora_graph`

结果：

- Cora: GRACE 0.8224/0.8015；`anchor` mode 0.6524/0.6047，灾难性退化；
- CiteSeer: GRACE 0.7171/0.6563；`anchor` mode 0.6897/0.6401，小幅退化；
- PubMed: 本轮旧 `anchor` run 在 GRACE 后中断，尚未补跑；
- Cora `graph` fallback: 0.7997/0.7655，明显修复 anchor 退化，但仍低于 GRACE。

判断：当前原型只能作为 heterophily-focused active candidate，不能声称 homophily non-degradation。下一步应实现 validation-based 或更可靠的 graph/raw representation selection；如果无法修复 Cora 退化，论文定位必须收缩为异配图专门方法。

## 2026-06-28 表示选择诊断

新增脚本：

```bash
python select_representation.py --run-dir /tmp/raw_complement_cora_graph/Cora_raw_complement_gcl_seed0 --selection-eval-mode random
```

脚本功能：

- 递归读取 `artifacts.pt`；
- 在同一 dataset/split 下比较 `raw`、`saved`、`anchor`、`graph`、`complement`、`hidden` 等候选表示；
- 使用验证集选择候选表示，再报告测试集 F1；
- 支持固定 mask 协议与随机 train/val/test 协议；
- 支持 `--candidate-names`、`--solver`、`--c-min-power/--c-max-power`，用于控制诊断成本。

### Cora safety 诊断

输出：

- `runs/summaries/raw_complement_representation_selection_cora_random_seed0_fullc.csv`
- `runs/summaries/raw_complement_representation_selection_cora_random_seed0_fullc_aggregate.csv`

结果：

- 完整 C 网格、3 次随机划分，验证集均选择 `saved`；
- `saved` 在该 graph fallback run 中等价于 graph-context 表示；
- 平均 F1Mi/F1Ma 为 `0.800738/0.777029`；
- 相比 anchor mode 的 `0.6524/0.6047` 明显修复同配退化；
- 但仍低于同设置 GRACE 的约 `0.8224/0.8015`。

判断：validation-based representation selection 可以避免 Cora 上最严重的 raw-anchor 崩溃，但还不能证明 homophily non-degradation。若论文方法要覆盖同配图，需要继续做更强的 graph-context fallback 或训练期 gate；否则应明确定位为 heterophily-focused。

### Heterophily raw-vs-saved 诊断

输出：

- `runs/summaries/raw_complement_raw_vs_saved_selection_heterophily_splits0-9_seed0_e100.csv`
- `runs/summaries/raw_complement_raw_vs_saved_selection_heterophily_splits0-9_seed0_e100_aggregate.csv`

为控制成本，本诊断只比较 `raw` 与 `saved`，并使用窄 C 网格；因此它用于选择趋势分析，不作为最终性能表。

| Dataset | selected counts | selected F1Mi | selected F1Ma | 判断 |
| --- | --- | ---: | ---: | --- |
| Actor | saved:10 | 0.359803 | 0.327280 | 学到的互补表示有稳定增量 |
| Cornell | raw:7; saved:3 | 0.637838 | 0.379415 | 多数 split 仍退回 raw |
| Texas | raw:9; saved:1 | 0.700000 | 0.414278 | 多数 split 仍退回 raw |
| Wisconsin | raw:8; saved:2 | 0.752941 | 0.412862 | 多数 split 仍退回 raw |

判断：当前 raw-complement 表示的真实增量最清楚地出现在 Actor；WebKB 三个小图多数 split 仍由 raw feature 主导。该证据削弱了“通用 SOTA 表示学习方法”的叙事，但保留了一个更合理的论文切口：Graph SSL 在异配图上常被 raw-feature separability 支配，方法贡献应证明何时能学习 raw 之外的互补信号，而不是只赢弱 GRACE baseline。

### 工程结论

全量候选、完整 C 网格的 sklearn logistic probe 在 Actor 上明显变慢，已中断两次。`select_representation.py` 当前保留为单 run / 小批量诊断工具；正式大规模汇总应继续使用训练脚本自带 evaluator 或后续实现更快的 torch linear probe。

## 2026-06-28 Homophily graph fallback 补充

新增命令：

```bash
python train.py --dataset CiteSeer --method raw_complement_gcl --raw-complement-eval-mode graph --seed 0 --epochs 100 --save-dir /tmp/raw_complement_citeseer_graph --overwrite --log-every 100
python train.py --dataset PubMed --method raw_complement_gcl --raw-complement-eval-mode graph --seed 0 --epochs 100 --batch-size 4096 --save-dir /tmp/raw_complement_pubmed_graph_b4096 --overwrite --log-every 100
```

PubMed 全量 InfoNCE 首次运行出现 CUDA OOM；使用 `--batch-size 4096` 后完成训练。这说明 PubMed 必须纳入单卡 12GB 约束下的 batched InfoNCE 实验协议，不能默认全量相似度矩阵。

### GRACE vs raw-complement graph fallback

| Dataset | GRACE F1Mi/F1Ma | raw-complement graph F1Mi/F1Ma | Delta F1Mi/F1Ma | 判断 |
| --- | ---: | ---: | ---: | --- |
| Cora | 0.822395 / 0.801539 | 0.799699 / 0.765548 | -0.022696 / -0.035991 | 仍明显退化 |
| CiteSeer | 0.717084 / 0.656260 | 0.723984 / 0.651930 | +0.006900 / -0.004330 | micro 略升，macro 略降 |
| PubMed | 0.844247 / 0.840378 | 0.839513 / 0.835653 | -0.004734 / -0.004725 | 小幅退化 |

### CiteSeer representation selection

输出：

- `runs/summaries/raw_complement_representation_selection_citeseer_random_seed0_fullc.csv`
- `runs/summaries/raw_complement_representation_selection_citeseer_random_seed0_fullc_aggregate.csv`

结果：完整 C 网格、3 次随机划分均选择 `saved`/graph-context 表示，F1Mi/F1Ma 为 `0.726749/0.633206`。micro 上 selection 与 train-time graph eval 一致偏正；macro 偏低，说明 graph fallback 修复 anchor 退化，但未稳定改善类别均衡。

### 判断更新

raw-complement 的 homophily safety 风险从“全面失败”收缩为“Cora 明显失败，CiteSeer/PubMed 基本可控”。这让该 idea 仍值得继续，但不能把默认 `anchor` 输出作为最终方法。下一步应实现一个训练期或评估期的安全输出策略：

- 对 heterophily/Actor 类场景，保留 raw+complement 的增量；
- 对 Cora 类同配小图，自动退回 graph-context 或更接近 GRACE 的表示；
- 对 PubMed 必须固定使用 batched InfoNCE，保证单卡 12GB 可复现。

## 2026-06-28 Cora graph fallback 多 seed 复核

新增命令：

```bash
python train.py --dataset Cora --method raw_complement_gcl --raw-complement-eval-mode graph --seed 1 --epochs 100 --save-dir /tmp/raw_complement_cora_graph_seed1 --overwrite --log-every 100
python train.py --dataset Cora --method raw_complement_gcl --raw-complement-eval-mode graph --seed 2 --epochs 100 --save-dir /tmp/raw_complement_cora_graph_seed2 --overwrite --log-every 100
python train.py --dataset Cora --method grace --seed 1 --epochs 100 --save-dir /tmp/grace_cora_seed1 --overwrite --log-every 100
python train.py --dataset Cora --method grace --seed 2 --epochs 100 --save-dir /tmp/grace_cora_seed2 --overwrite --log-every 100
```

### 同 seed 对照

| Seed | GRACE F1Mi/F1Ma | raw-complement graph F1Mi/F1Ma | Delta F1Mi/F1Ma |
| ---: | ---: | ---: | ---: |
| 0 | 0.822395 / 0.801539 | 0.799699 / 0.765548 | -0.022696 / -0.035991 |
| 1 | 0.806125 / 0.792402 | 0.809270 / 0.792088 | +0.003145 / -0.000314 |
| 2 | 0.822259 / 0.803730 | 0.819114 / 0.797644 | -0.003145 / -0.006087 |
| mean | 0.816926 / 0.799223 | 0.809361 / 0.785093 | -0.007565 / -0.014130 |

判断更新：Cora graph fallback 不是稳定灾难，但仍平均低于 GRACE，尤其 macro 更明显。seed0 的退化偏大，seed1/2 接近 GRACE，说明当前问题可能来自训练随机性和 raw-complement regularizer 对图上下文的扰动，而不是 graph fallback 表示本身完全错误。

### Cora representation selection seeds1-2

命令：

```bash
python select_representation.py --run-dir /tmp/raw_complement_cora_graph_seed1/Cora_raw_complement_gcl_seed1 --run-dir /tmp/raw_complement_cora_graph_seed2/Cora_raw_complement_gcl_seed2 --selection-eval-mode random --random-repeats 3 --candidate-names raw saved anchor graph --c-min-power -8 --c-max-power 8 --max-iter 3000 --out runs/summaries/raw_complement_representation_selection_cora_random_seeds1-2_fullc.csv --aggregate-out runs/summaries/raw_complement_representation_selection_cora_random_seeds1-2_fullc_aggregate.csv
```

结果：6 次随机划分全部选择 `saved`/graph-context，F1Mi/F1Ma 为 `0.815883/0.797750`。这支持 validation-based selection 能稳定避开 `anchor`，但它仍不是无标签方法，也不足以作为最终方法机制。

### 设计取舍

当前不应放弃 raw-complement 主线，但必须停止把 `anchor` mode 作为默认最终输出。更合理的下一步是实现一个 self-supervised safety gate 或训练期 graph-context preservation：

- `graph` 输出作为 homophily-safe fallback；
- `anchor` / raw+complement 输出只在互补信号明确时启用；
- Cora 的目标不是大幅超 GRACE，而是把平均退化压到 0.5 个百分点以内，同时保持 Actor/WebKB 异配收益。

## 2026-06-28 raw-complement weight 小消融

目标：检查 Cora graph fallback 的退化是否能通过简单调节 `raw_complement_weight` 修复。如果可以，后续可走轻量超参/默认值路线；如果不行，应停止把问题归因于单一 loss 权重。

新增命令：

```bash
python train.py --dataset Cora --method raw_complement_gcl --raw-complement-eval-mode graph --raw-complement-weight 0.01 --seed 0 --epochs 100 --save-dir /tmp/raw_complement_cora_graph_w001_seed0 --overwrite --log-every 100
python train.py --dataset Cora --method raw_complement_gcl --raw-complement-eval-mode graph --raw-complement-weight 0.1 --seed 0 --epochs 100 --save-dir /tmp/raw_complement_cora_graph_w01_seed0 --overwrite --log-every 100
python train.py --dataset Actor --method raw_complement_gcl --raw-complement-eval-mode anchor --raw-complement-weight 0.1 --seed 0 --split-index 0 --epochs 100 --save-dir /tmp/raw_complement_actor_w01_seed0_split0 --overwrite --log-every 100
python train.py --dataset Texas --method raw_complement_gcl --raw-complement-eval-mode anchor --raw-complement-weight 0.1 --seed 0 --split-index 0 --epochs 100 --save-dir /tmp/raw_complement_texas_w01_seed0_split0 --overwrite --log-every 100
```

结果：

| Setting | Dataset/split | F1Mi/F1Ma | 对照 |
| --- | --- | ---: | --- |
| `weight=0.01`, graph | Cora seed0 | 0.7931 / 0.7565 | 低于 default `0.05` 的 0.7997 / 0.7655 |
| `weight=0.1`, graph | Cora seed0 | 0.8050 / 0.7725 | 高于 default，但仍低于 GRACE 0.8224 / 0.8015 |
| `weight=0.1`, anchor | Actor split0 | 0.3730 / 0.3379 | 略高于 default 0.3704 / 0.3281 |
| `weight=0.1`, anchor | Texas split0 | 0.7838 / 0.5979 | 低于 default 0.8108 / 0.6200 |

判断：

- 降低 `raw_complement_weight` 到 `0.01` 不能修复 Cora，反而更差；
- 提高到 `0.1` 对 Cora seed0 有轻微缓解，对 Actor split0 不伤甚至略好；
- 但 `0.1` 明显损伤 Texas split0，说明单一全局权重无法同时满足 homophily safety 与 WebKB 异配收益；
- 停止继续做朴素 `raw_complement_weight` 网格搜索。

下一步必须转为结构性改法：安全输出 gate、graph-context preservation regularizer、或 dataset/region-adaptive complement usage，而不是继续调一个全局 loss 权重。
