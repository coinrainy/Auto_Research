# GRACE Idea 工作副本说明

本目录由 `baselines/GRACE` 复制而来，用于承载当前研究 idea 的代码改动。

- 来源 baseline：`baselines/GRACE`
- 来源版本：`b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- 协作约束：不要直接修改 `baselines/GRACE`；所有方法改动写在本目录。

当前路线调整：

- 删除此前自建的 RW-GCL scaffold 与批跑脚本。
- 以后以 GRACE 原实现为基础做最小侵入式修改。
- 优先保持 baseline 逻辑可对照，再逐步加入 reliability / weighting / diagnostics。

## 当前已实现的方法入口

训练入口仍为 `train.py`，但新增 `--method` 参数：

- `--method grace`：保留原始 GRACE 训练路径。
- `--method es_weighted`：embedding-stability weighted GRACE，保留为历史/对照路线。
- `--method sgfn`：Stability-Guided False-Negative attenuation，已降级为负结果/诊断资产。
- `--method spectral_mix`：Adaptive Spectral Mix GCL，已降级为失败原型/诊断资产。
- `--method pbcl`：Prototype-Balanced Contrastive Learning，已降级为失败原型/诊断资产。
- `--method pccl`：Prototype Consistency Contrastive Learning，已降级为失败原型/诊断资产。
- `--method rr_gcl`：Redundancy-Reduced GCL，保留为条件性线索/诊断资产，尚非 active SOTA candidate。
- `--method hybrid_rr_gcl`：GRACE InfoNCE + RR 小权重正则，已降级为条件性诊断资产，提示 RR 可能改善 macro/少数类覆盖但全局固定权重不稳。
- `--method cbr_gcl`：Cluster-Balanced Redundancy-Reduced GCL，当前最值得保留的 RR 条件性候选，但仍需 anti-degradation gate 后才可能成为主方法。
- `--method gated_cbr_gcl`：基于 RR diagonal confidence 的 CBR gate，已降级为失败 gate / 消融资产。
- `--method stable_cluster_cbr_gcl`：基于 cluster compactness / separation 的节点级 CBR 稳定权重，已降级为失败 gate / 诊断资产。
- `--method ego_grace`：纯 ego-feature MLP encoder 的 GRACE ablation，当前显示异配图强信号，是 ego-preserving 机制的关键对照。
- `--method residual_grace`：GCN branch + ego MLP branch 的 residual encoder，是当前最稳 active candidate。
- `--method gated_ego_graph_grace`：基于 local feature-neighborhood agreement 的节点级 graph usage gate；异配强，但当前 v1 同配退化严重，不能作为最终主方法。

`es_weighted` 的设计边界：

- reliability 只来自 encoder embedding stability，不再使用 projection head softmax distribution consistency。
- 通过 EMA teacher 在原图上产生稳定参照 embedding。
- warm-up 后，对两个增强视图下的 student embedding 与 EMA teacher embedding 做余弦相似度，得到节点级 reliability。
- 默认将 reliability 用作 positive anchor weighting。
- 可通过 `--negative-weighting` 将 reliability 同时用于 InfoNCE denominator candidate weighting，低可靠节点作为负样本时贡献更小。
- reliability 只作为 stop-gradient 权重使用，不把权重估计路径反传回模型。
- 可通过 `--shuffle-weights` 做分布保持的 reliability-node 对应打乱控制。
- 可通过 `--random-weights` 做 uniform-random 权重压力测试；它不保留 normal reliability 的分布，不应当作主随机化 control。
- 默认拒绝写入已有非空 run 目录；如确需覆盖，显式添加 `--overwrite`。

最小 smoke 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python train.py --dataset Cora --method grace --epochs 2 --skip-eval
python train.py --dataset Cora --method es_weighted --epochs 2 --warmup-epochs 1 --negative-weighting --skip-eval --save-dir runs/smoke --overwrite
python train.py --dataset Cora --method es_weighted --epochs 2 --warmup-epochs 1 --shuffle-weights --skip-eval --save-dir runs/smoke_controls --overwrite
```

`sgfn` 的设计边界：

- 使用 EMA teacher 在 clean graph 上产生稳定参照 embedding。
- 对 teacher embedding 相似度做 row-standardized scoring，估计 pair-level false-negative risk。
- 在 InfoNCE denominator 中对疑似 false negative 做有界 attenuation，而不是只做 node-wise anchor weighting。
- 支持 `--shuffle-weights` 做 pair 映射打乱控制，用于检验权重结构是否有效。
- 支持 `--fn-consensus feature`、`--pair-normalization row_mean|blend_row_mean`、`--fn-attraction-weight`，但当前实验显示这些扩展不适合作为默认主线，应作为消融或负结果保留。

当前研究判断：

- 默认 `sgfn` false-negative attenuation 已不再作为最终主候选。10-split 复核显示 Texas 稳定正向，但 Wisconsin 退化、Actor 近零、Cornell normal 不稳定优于 shuffled。
- `--fn-consensus feature` 在 Texas/Cornell 上有局部改善，但 Wisconsin/Actor 仍失败，也不能作为主方法。
- 当前保留的是 pair-level denominator attenuation、shuffled pair mapping control 与 label-only false-negative pressure 诊断；下一代方法必须先判断“何时不应 attenuation”，而不是继续增强全局 attenuation。
- `row_mean` reallocation 机制更干净但性能弱；`blend_row_mean` 与 `fn_attraction_weight=0.1` 在 4 个 heterophily 数据集 sanity 中整体负向。
- 已实现并筛选 context-gated false-negative calibration。`local_feature_degree + product` 在 Texas split0 early stop；`degree_inverse + anchor` 在 Texas/Cornell 有正向，但 Wisconsin 三个 split 全部 micro 退化、Actor 近零且 normal 低于 shuffled。
- 当前决策：停止 teacher-similarity false-negative attenuation 主线。保留代码和诊断作为负结果资产，下一轮重新构思 GCL idea。

`spectral_mix` 的设计边界：

- 将 GRACE 的 view construction 从纯随机 feature drop 扩展为局部自适应 low/high-pass feature mix。
- 用邻域均值近似低频特征，用节点特征减邻域均值近似高频残差。
- `adaptive` gate 来自局部特征一致性；一致性高的节点更偏向低频视图，一致性低的节点保留更多高频残差。
- 两个 view 通过 `--spectral-mix-jitter` 施加相反方向的小扰动，避免两个 spectral view 完全一致。
- 当前推荐候选参数是 `--spectral-high-scale 0.5`，用于限制高频残差强度。

`spectral_mix` 当前研究判断：

- `high_scale=1.0` 在 Wisconsin macro 上明显正向，但 Cornell macro 退化大，不适合作为默认设置。
- `high_scale=0.5` 在 split0-2 上曾在 Texas/Cornell 出现正向，但 split0-9 复核后 Cornell 明确转负，Texas/Wisconsin 仅弱正且不稳定，Actor 近零。
- 已新增 `--spectral-residual-alpha` 保留原始特征 residual anchor；`alpha=0.5` 保留 Texas split0-2 收益，但 Actor/Wisconsin 仍不稳定，不能作为主候选。
- Cora/CiteSeer quick sanity 有小幅下降，因此暂不能声称 homophily non-degradation。
- 当前定位是失败原型和机制线索：谱扰动可能帮助部分少数类，但当前 gate 没有可靠对齐下游语义。

`pbcl` 的设计边界：

- warmup 后用两个增强 view 的 encoder embedding 均值做 consensus embedding。
- 对 consensus embedding 做轻量 KMeans，默认原型数为数据集类别数，也可用 `--pbcl-num-prototypes` 指定。
- 按原型簇大小的逆频率生成 node-wise anchor weight，提高低密度原型节点在 InfoNCE 聚合中的权重。
- 支持 `--shuffle-weights` / `--random-weights`，用于检验节点-原型密度对应关系是否真的有效。

`pbcl` 当前研究判断：

- Texas/Cornell split0-2 有局部正向，但 Actor 三个 split 全部负向，Wisconsin macro 负向。
- normal 不稳定优于 shuffled；Cornell 与 Wisconsin 上 shuffled 经常接近或超过 normal。
- 当前结论：简单 prototype-density anchor reweighting 不能作为主候选；保留为 density/prototype 诊断资产。

`pccl` 的设计边界：

- warmup 后用两个增强 view 的 encoder embedding 均值做 consensus embedding。
- 对 consensus embedding 做 KMeans 得到 prototype centers。
- 用 consensus-to-prototype soft target 监督两个 view 的 prototype assignment。
- 增加 prototype usage balance loss 抑制塌缩。
- `--shuffle-weights` 在 PCCL 中表示打乱节点-prototype target 对应关系，用作机制 control。

`pccl` 当前研究判断：

- Cornell macro 有局部正向，但 Texas/Wisconsin micro 为负，Actor 近零略负。
- normal 不稳定优于 shuffled；Cornell 与 Texas 上 normal-shuffled 均值为负。
- 当前结论：当前 KMeans soft target 不可靠，prototype consistency objective 不能作为主候选；保留为 prototype-level objective 诊断资产。

`rr_gcl` 的设计边界：

- 使用 GRACE 原增强视图，但不用 InfoNCE。
- 对两个 view 的 projection features 做 cross-correlation redundancy reduction。
- 对角项逼近 1，非对角项逼近 0。
- `--shuffle-weights` 在 RR-GCL 中表示打乱 positive node correspondence，用作机制 control。

`rr_gcl` 当前研究判断：

- Cornell split0-2 上相对 GRACE 有 F1Mi/F1Ma +0.036036/+0.082584，且 normal 优于 shuffled。
- Actor/Texas/Wisconsin 不稳，尤其 macro 退化明显。
- 当前结论：negative-free redundancy reduction 是比 prototype 路线更有价值的条件性线索，但裸 RR objective 不能作为主候选；下一步应尝试 GRACE InfoNCE + small RR regularizer 或 adaptive mixing。

`hybrid_rr_gcl` 的设计边界：

- 保留 GRACE InfoNCE 主损失；
- 额外加入 Barlow/CCA 风格 RR 正则：`loss = InfoNCE + hybrid_rr_weight * RR`;
- `--shuffle-weights` 只打乱 RR positive correspondence，InfoNCE 不变；
- 支持 `--hybrid-rr-weight` 控制正则强度，默认 `0.01`。

`hybrid_rr_gcl` 当前研究判断：

- `hybrid_rr_weight=0.01` 在 Cornell/Texas 上有 macro 亮点，但 Texas normal 明显低于 shuffled，Cornell micro 低于 GRACE。
- `hybrid_rr_weight=0.001` 保留 Cornell/Wisconsin macro 信号，但 Texas micro/macro 均退化，Cornell normal 仍未稳定优于 shuffled。
- 当前结论：固定全局 RR 正则不能作为 active SOTA candidate；它揭示的有效线索是 RR 可能改善类别覆盖/少数类，但必须改成 adaptive / class-sensitive / region-sensitive RR 才值得继续。

`cbr_gcl` 的设计边界：

- warm-up 前等同 GRACE；
- warm-up 后使用 GRACE InfoNCE 主损失；
- 用两个 view 的 consensus embedding 做无监督 KMeans；
- 通过 inverse cluster-size 权重计算 cluster-balanced RR cross-correlation；
- `--shuffle-weights` 只打乱 RR positive correspondence，InfoNCE 和 cluster weights 不变；
- 默认 `--cbr-rr-weight 0.001`，cluster 数默认等于 dataset.num_classes。

`cbr_gcl` 当前研究判断：

- split0-2 sanity 显示 Texas/Cornell/Wisconsin 均为正，Actor 小负；Texas normal-vs-shuffled 从 Hybrid RR 的负信号转为正信号。
- split0-9 复核显示：Texas F1Mi/F1Ma +0.005405/+0.010329；Wisconsin +0.003922/+0.029514；Cornell +0.000000/+0.008028；Actor -0.000789/-0.002619。
- normal-vs-shuffled：Texas +0.005405/+0.008799；Wisconsin +0.009804/+0.032024；Cornell -0.002703/+0.013165；Actor -0.000329/-0.001300。
- 当前结论：CBR 是目前 RR 方向最值得继续的条件性候选，支持“cluster-balanced RR 改善少数/弱类覆盖”的机制线索；但 Actor 近零略负、Cornell micro control 不稳，不能作为 SOTA-ready 主方法。
- 下一步必须实现 anti-degradation gate，而不是继续全局调 `cbr_rr_weight`。

`gated_cbr_gcl` 的设计边界：

- 在 `cbr_gcl` 的基础上，用 weighted cross-correlation diagonal mean 作为 positive correspondence confidence；
- gate scale 为 `sigmoid((diag_mean - cbr_gate_min_diag) / cbr_gate_temperature)`；
- 实际 RR loss 为 `raw_CBR_RR * gate_scale`；
- 支持 `--cbr-gate-min-diag`、`--cbr-gate-temperature`、`--cbr-gate-min-scale`；
- `--shuffle-weights` 仍只打乱 RR positive correspondence。

`gated_cbr_gcl` 当前研究判断：

- 默认 `cbr_gate_min_diag=0.82` 在 split0-2 上保护了 Actor，但削弱 Texas，并使 Wisconsin micro 明显退化。
- 宽松 `cbr_gate_min_diag=0.78` 恢复 Texas，但 Cornell F1Mi/F1Ma 变为 -0.036036/-0.080680，normal 远低于 shuffled。
- 当前结论：单一 RR diagonal confidence 不是可靠 anti-degradation gate，停止继续调该阈值；该方法只保留为 gate 消融资产。

`stable_cluster_cbr_gcl` 的设计边界：

- 在 `cbr_gcl` 基础上记录 cluster margin、assigned similarity 与 stability scale；
- 用 KMeans consensus embedding 的 top1-top2 cluster margin 生成节点级 stability scale；
- 只降低不稳定簇边界节点在 CBR-RR 正则中的贡献，InfoNCE 主损失保持不变。

`stable_cluster_cbr_gcl` 当前研究判断：

- Texas/Wisconsin split0-2 有弱正向，但 Cornell/Actor 全负或明显不稳。
- normal-vs-shuffled control 不干净：Cornell normal 低于 shuffled，Texas/Wisconsin 也不是稳定优于 shuffled。
- 当前结论：cluster compactness/separation 不等于下游语义可靠性，停止继续调该 gate；保留日志字段用于机制诊断。

`ego_grace` / `residual_grace` / `gated_ego_graph_grace` 的设计边界：

- `ego_grace` 完全忽略 `edge_index`，只用 MLP encoder 做 feature-drop 双视图 GRACE，用于检验 ego-feature preservation 是否是异配收益来源；
- `residual_grace` 使用 GCN branch + ego MLP branch，通过可学习 scalar gate 融合；
- `gated_ego_graph_grace` 使用 GCN branch + ego MLP branch，通过节点级 local feature-neighborhood agreement gate 融合；
- 三者都保留 GRACE 的 InfoNCE 目标，不引入标签、不使用验证集调参。

`ego_grace` / `residual_grace` / `gated_ego_graph_grace` 当前研究判断：

- `residual_grace` 在 Texas/Cornell/Wisconsin/Actor × split0-9 上全面正向：Actor +0.064868/+0.090895，Cornell +0.172973/+0.183641，Texas +0.102703/+0.188203，Wisconsin +0.180392/+0.199546。
- `residual_grace` 同配 quick sanity：Cora -0.008477/-0.009624，CiteSeer +0.006678/+0.012899，PubMed +0.012003/+0.014253；当前基本安全，但 Cora 需要复核。
- `ego_grace` split0-2 比 `residual_grace` 更强，说明 ego-feature preservation 是核心机制，也意味着必须保留 MLP-only baseline，否则论文叙事不成立。
- `gated_ego_graph_grace` split0-2 在 Texas/Cornell/Actor 很强，但 Cora/CiteSeer/PubMed 同配 quick sanity 严重退化；当前 v1 不能作为最终主方法。
- 当前 active candidate 收缩为 Ego-Preserving / Graph-Usage Calibrated GCL：以 `residual_grace` 作为稳健主线，`ego_grace` 作为必要强 baseline，下一步设计 homophily-safe graph usage gate。

正式实验前仍需补齐：

- reliability 与 downstream error、degree、local homophily 的独立诊断；
- 与 ProGCL / GRAPE / GraphRank 等 false-negative 或 hard-negative 方法的公平对照。

## 当前实验入口能力

- 支持 Planetoid/CitationFull：`Cora`、`CiteSeer`、`PubMed`、`DBLP`。
- 支持异配数据集：`Texas`、`Cornell`、`Wisconsin`、`Actor`。
- 支持 `--split-index` 选择 PyG 提供的二维 split mask。
- `--eval-mode auto` 对异配数据集默认使用固定 `train/val/test` mask；对原 GRACE 数据集保持随机 linear probe。
- `--eval-mode mask` 可强制使用 mask 评估。
- `--eval-mode random` 可强制使用原 GRACE 风格随机 linear probe。
- `scripts/run_split_study.sh` 可通过 `DATASETS`、`SPLITS`、`SEEDS`、`METHODS`、`ES_CONTROLS` 做 split-aware 批跑。
- `train_log.csv` 记录权重均值、方差、min/max 与 effective sample size ratio，用于判断 reliability 权重是否实质上接近等权。
- `summarize_runs.py` 可从 `runs/` 目录生成 matched paired summary 与 dataset aggregate summary，并兼容 `es_weighted` / `sgfn` / `spectral_mix` 的 normal、shuffled、uniform_random 控制组。
- `analyze_pair_weights.py` 可对 `sgfn` run 做 label-only false-negative pressure 诊断；该诊断不参与训练，只用于机制验证。
- `sgfn` 支持 `--fn-context-gate local_feature|degree_inverse|local_feature_degree` 与 `--fn-context-pair-mode product|min|anchor`，但当前筛选结果显示这些 gate 未能救活主线。
- `spectral_mix` 支持 `--spectral-mix-mode adaptive|low|high|random`、`--spectral-mix-temperature`、`--spectral-mix-jitter` 与 `--spectral-high-scale`。
- `pbcl` 支持 `--pbcl-num-prototypes`、`--pbcl-kmeans-iters`、`--pbcl-weight-power`、`--pbcl-min-weight`、`--pbcl-max-weight`。
- `pccl` 支持 `--pccl-num-prototypes`、`--pccl-kmeans-iters`、`--pccl-prototype-temperature`、`--pccl-target-temperature`、`--pccl-consistency-weight`、`--pccl-balance-weight`。
- `rr_gcl` 支持 `--rr-offdiag-weight` 与 `--rr-loss-scale`，并复用 `--shuffle-weights` 做 positive correspondence control。
- `hybrid_rr_gcl` 支持 `--hybrid-rr-weight`，并复用 RR 参数与 positive correspondence control。
- `cbr_gcl` 支持 `--cbr-rr-weight`、`--cbr-num-clusters`、`--cbr-kmeans-iters`、`--cbr-min-weight`、`--cbr-max-weight`，并记录 cluster balance diagnostics。
- `gated_cbr_gcl` 支持 CBR 的所有参数，并额外支持 `--cbr-gate-min-diag`、`--cbr-gate-temperature`、`--cbr-gate-min-scale`。
- `stable_cluster_cbr_gcl` 支持 CBR 的所有参数，并额外记录 `cbr_cluster_margin_*` 与 `cbr_stability_scale_*`。
- `ego_grace` 支持纯 MLP ego encoder。
- `residual_grace` 支持 `--ego-gate-init`，并记录 `ego_gate`。
- `gated_ego_graph_grace` 支持 `--graph-gate-temperature`、`--graph-gate-threshold`、`--graph-gate-min`、`--graph-gate-max`，并记录 `graph_gate_*`。

示例 split-aware 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell" SPLITS="0 1 2" SEEDS="0" METHODS="grace sgfn" ES_CONTROLS="normal shuffled" SAVE_DIR="runs/sgfn_split_control_sanity" scripts/run_split_study.sh
python summarize_runs.py --runs-dir runs/sgfn_split_control_sanity --target-method sgfn --paired-out runs/summaries/sgfn_split_control_sanity_paired.csv --aggregate-out runs/summaries/sgfn_split_control_sanity_aggregate.csv
python analyze_pair_weights.py --runs-dir runs/sgfn_split_control_sanity --out runs/summaries/sgfn_split_control_sanity_pair_weights.csv --aggregate-out runs/summaries/sgfn_split_control_sanity_pair_weights_aggregate.csv --control-paired-out runs/summaries/sgfn_split_control_sanity_pair_weights_controls.csv
```

近期需要补齐：

- 暂停 naive `spectral_mix` 扩展，不继续补 Chameleon/Squirrel。
- 若继续 spectral 方向，只做小消融定位失败机制：`adaptive` vs `low` vs `high` vs `random`。
- 暂停 PBCL 扩展，不继续跑 10 splits；简单 anchor reweighting 已被 shuffled control 证伪。
- 暂停 PCCL 扩展，不继续跑 10 splits；简单 KMeans prototype soft-target consistency 已被 shuffled control 证伪。
- 暂不扩裸 RR-GCL 到 10 splits；先实现 hybrid objective 或 adaptive mixing，检验能否保留 Cornell 的 class-level 收益同时减少 Actor/Texas/Wisconsin 退化。
- 暂停固定全局 hybrid RR 权重搜索；`0.01` 和 `0.001` 均未通过 Texas / normal-vs-shuffled 机制压力测试。
- 已实现并复核 `cbr_gcl`，不建议继续简单调 `cbr_rr_weight`。
- 已实现并筛选基于 RR diagonal confidence 的 `gated_cbr_gcl`，该单信号 gate 失败，不建议继续调 diagonal threshold。
- 已实现并筛选 `stable_cluster_cbr_gcl`，该 cluster compactness/separation gate 未通过 Cornell/Actor 压力测试，不建议继续沿 CBR gate 小修小补。
- 当前应优先推进 Ego-Preserving / Graph-Usage Calibrated GCL：先扩展 `ego_grace` 10 split，与 `residual_grace` 对齐；再设计 homophily-safe graph usage gate，修复 `gated_ego_graph_grace` 在 Cora/CiteSeer/PubMed 的同配退化。
- 本目录中的 SGFN / context-gated SGFN 只作为负结果、诊断工具和消融资产保留。
