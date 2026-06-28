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
- `--method ego_grace`：纯 ego-feature MLP encoder 的 GRACE ablation，当前显示异配图强信号，是 ego-preserving 机制的关键对照；但 WebKB raw-feature baseline 更强，不能直接包装为 SOTA 方法。
- `--method residual_grace`：GCN branch + ego MLP branch 的 residual encoder，是当前最稳的 encoder-level diagnostic candidate；但同样未越过 WebKB raw-feature baseline。
- `--method gated_ego_graph_grace`：基于 local feature-neighborhood agreement 的节点级 graph usage gate；异配强，但当前 v1 与 `--graph-gate-min 0.5` 同配退化严重，不能作为最终主方法。
- `--method raw_complement_gcl`：Raw-Anchored Complement GCL 原型，训练 hidden `[raw_anchor, complement]`，final representation 默认使用 `[normalized raw features, normalized learned complement]`；当前异配 10 split 相对 GRACE 全正向，但 homophily safety 尚未解决。当前最新机制消融显示，收益更准确地来自 raw-relative graph complement，而不是普通 graph context 或简单 raw+graph 拼接。
- `select_representation.py`：raw-complement 的表示选择诊断工具，可从 `artifacts.pt` 中比较 `raw/saved/anchor/graph/complement/hidden`，用验证集选择候选表示；当前发现 Cora 可通过 graph/saved fallback 避免 anchor 崩溃，但仍低于 GRACE，Actor 上 saved 表示有清楚增量，WebKB 三小图多数 split 仍由 raw feature 主导。

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
- `ego_grace` split0-9 相对 GRACE 仍稳定强正向：Actor +0.070855/+0.099576，Cornell +0.172973/+0.141093，Texas +0.124324/+0.232101，Wisconsin +0.225490/+0.274924。
- `ego_grace` 与 `residual_grace` 的对齐结果显示：Texas/Wisconsin/Actor 上 ego-only 更强，Cornell 上 residual 更稳；最终方法不应固定为纯 ego-only 或固定 residual，而应学习/诊断“何时使用 graph propagation”。
- `gated_ego_graph_grace` split0-2 在 Texas/Cornell/Actor 很强，但 Cora/CiteSeer/PubMed 同配 quick sanity 严重退化；`--graph-gate-min 0.5` 仍未修复同配问题，Cora/CiteSeer/PubMed 分别为 -0.184167/-0.224984、-0.052198/-0.045933、-0.038056/-0.031795。
- 新增 `evaluate_raw_features.py` 后，WebKB raw-feature baseline 明显强于当前 SSL embeddings：Texas raw - ego +0.094595/+0.149131，Cornell +0.072973/+0.074007，Wisconsin +0.068627/+0.069676；Actor micro 上 ego 略优 raw，但 macro 基本持平。
- 当前 active candidate 必须从“encoder 改法直接 SOTA”收缩为 Feature-Anchored / Graph-Usage Calibrated GCL：以 raw features 为硬 baseline，证明 GCL embedding 或 graph branch 能提供 raw 之外的增量，或者转向机制论文路线解释 GCN-based GCL 何时破坏 raw-feature separability。

正式实验前仍需补齐：

- reliability 与 downstream error、degree、local homophily 的独立诊断；
- 与 ProGCL / GRAPE / GraphRank 等 false-negative 或 hard-negative 方法的公平对照。

## 当前实验入口能力

- 支持 Planetoid/CitationFull：`Cora`、`CiteSeer`、`PubMed`、`DBLP`。
- 支持异配数据集：`Texas`、`Cornell`、`Wisconsin`、`Actor`、`Chameleon`、`Squirrel`。
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
- `raw_complement_gcl` 支持 `--raw-complement-weight`、`--raw-complement-detach-anchor/--no-raw-complement-detach-anchor`、`--raw-complement-eval-mode anchor|hidden|graph|anchor_graph|raw_graph`，并记录 raw/complement correlation diagnostics；当前 `--raw-complement-weight` 默认值已改为 `0.0`，`0.05` 仅作为附录消融/robustness check。
- `evaluate_raw_features.py` 支持对原始 `data.x` 使用当前同一套 mask/random linear evaluation 协议，作为 ego/residual/GRACE 的 feature-only 硬 baseline。
- `evaluate_feature_fusion.py` 支持递归读取 `artifacts.pt`，在同一 split 下评估 `raw`、`ssl`、`raw+ssl concat`，并输出 concat 相对 raw/ssl 的 paired delta 与 aggregate summary。
- `select_representation.py` 支持读取 `artifacts.pt` 并用验证集选择 raw/saved/anchor/graph/complement/hidden 候选表示；当前定位为单 run / 小批量诊断工具，全候选完整 C 网格在 Actor 上过慢，不作为正式大规模评估主入口。
- `summarize_raw_complement_probe.py` 支持按 dataset/split/seed 对齐 raw、GRACE、Raw-Complement 的 `eval_summary.csv`，支持从多个结果根目录搜索 GRACE 与 Raw-Complement run，输出 paired delta 与 aggregate。

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
- 当前应优先推进 Raw-Anchored Residual/Complement GCL：`evaluate_feature_fusion.py` 已显示 SSL embedding 可能包含 raw 之外的互补信息，但 post-hoc concat 对 C 搜索敏感，不能作为最终方法。
- 已实现 `raw_complement_gcl` 原型，异配 Texas/Cornell/Wisconsin/Actor × splits0-9 相对 GRACE 全部 10/10 正向；相对 raw feature baseline 基本持平到小幅正向/负向，说明该方向有真实方法潜力。
- `raw_complement_gcl` 当前最大风险是 homophily safety：Cora anchor mode 从 GRACE 0.8224/0.8015 降到 0.6524/0.6047；graph fallback 回升到 0.7997/0.7655 但仍低于 GRACE。
- 表示选择诊断显示：Cora 完整 C 网格随机选择会选择 saved/graph-context，F1Mi/F1Ma 约 0.800738/0.777029；异配 raw-vs-saved 快速诊断中 Actor 10/10 选择 saved，Cornell/Texas/Wisconsin 多数选择 raw。当前叙事应继续收缩为 raw-feature anchored complement learning，而不是通用 Graph SSL SOTA。
- Homophily graph fallback 补充显示：Cora 相对 GRACE 仍明显退化（F1Mi/F1Ma -0.022696/-0.035991），CiteSeer micro 略升但 macro 略降（+0.006900/-0.004330），PubMed 小幅退化（-0.004734/-0.004725）。因此风险不是全面同配失败，而是 Cora 类小图安全输出仍没解决。
- Cora graph fallback seed0-2 同 seed 对照显示：平均相对 GRACE 为 F1Mi/F1Ma -0.007565/-0.014130；seed1/2 接近 GRACE，seed0 退化偏大。结论从“Cora 稳定失败”修正为“Cora 小幅但不稳定退化”。
- Cora seed1/2 的 representation selection 6 次随机划分均选择 saved/graph-context，F1Mi/F1Ma 为 0.815883/0.797750；说明验证集选择可以稳定避开 anchor，但不是最终无标签机制。
- PubMed 全量 InfoNCE 会在 12GB GPU 上 OOM，当前需使用 `--batch-size 4096`；后续所有 PubMed raw-complement/GRACE 公平对照都应固定 batch 协议。
- `raw_complement_weight` 小消融显示：`0.01` 在 Cora seed0 更差（0.7931/0.7565），`0.1` 能轻微缓解 Cora（0.8050/0.7725）且 Actor split0 不伤（0.3730/0.3379），但 Texas split0 明显低于默认（0.7838/0.5979 vs 0.8108/0.6200）。因此停止朴素全局权重搜索。
- `--no-raw-complement-detach-anchor` 消融显示：Cora seed0 graph 仅 0.8020/0.7651，Texas split0 anchor 降到 0.7838/0.6147；detach/no-detach 不是核心修复方向。
- `anchor_graph=[raw, complement, graph_context]` 并联输出消融显示：Cora seed0 仅 0.7726/0.7265，Actor split0 0.3638/0.3482，Texas split0 0.8108/0.6200；简单拼接不能替代显式 gate/selection。
- `anchor/graph/anchor_graph` output selection 诊断显示：Cora seeds0-2 的 9 次随机选择全部选 graph，graph 平均 0.810834/0.790843，明显优于 anchor 与 anchor_graph；Actor/Texas split0 均选 anchor_graph，但 Actor 上 anchor 的 test micro 反而略高，说明单一 val micro 选择还不够稳。
- 加入 raw 候选后，Cora 仍 9/9 选择 graph，raw 只有 0.645910/0.600536；Actor raw 0.348026/0.332063 低于 anchor/anchor_graph；Texas raw 与 anchor_graph 测试持平 0.810811/0.619968。说明 raw 不总是支配，但 WebKB 小图必须严守 raw baseline。
- `select_representation.py` 已加入 `--random-selection-repeats` 控制；minimal candidates `raw/graph/anchor_graph` 中 validation selection 相对 random selection：Cora 0.810834/0.790843 vs 0.766123/0.732582，Actor 0.364474/0.343554 vs 0.338651/0.310091，Texas 0.810811/0.619968 vs 0.770270/0.533084。selection/gate 方向值得方法化，但当前仍依赖标签验证集。
- CiteSeer/PubMed 补充同配 selection-control：CiteSeer selected 0.721868/0.639671 vs random 0.702766/0.634546，PubMed selected 0.847903/0.847778 vs random 0.825259/0.825074。Cora 偏 graph、CiteSeer 在 graph/anchor_graph 间摇摆、PubMed 偏 raw/anchor_graph，说明没有单一固定 fallback 能同时解决同配安全问题，必须方法化 output safety selection。
- 文献边界更新：HLCL/HeterGCL/H3GNNs 已覆盖异配图高低频、结构语义和 homophily/heterophily self-supervised 调和；当前 Raw-Complement 若继续，创新点必须收缩到 raw-feature anchored complement learning + safety selection，而不是泛化地声称 heterophily GCL SOTA。
- 已新增 `summarize_selection_controls.py`，用于把多个 `select_representation.py` aggregate 表统一成 selected-vs-random delta 表；当前 5 个 dataset 的 selected-random test micro delta 均为正，范围约 +0.019 到 +0.045。
- 已新增 `select_representation_proxy.py`，实现 label-free output safety proxy v1：effective-rank 过滤 collapsed candidate，按 `edge_random_contrast + 0.08 * raw_similarity_correlation - raw_penalty` 选择候选，小图上取消 raw penalty 作为 WebKB raw-baseline safety rule。完整 C 网格 sanity 中，proxy 在 Cora/PubMed/Actor/Texas 达到 validation selection 上界，CiteSeer micro 低约 0.010 但 macro 高约 0.013；5 个数据集均优于 random selection。
- WebKB/Actor splits0-9 扩展显示：一旦 `raw` 纳入候选且使用完整 C 网格，Actor/Cornell/Texas/Wisconsin 的 validation selection 与 proxy selection 都 100% 选择 raw。这说明 Raw-Complement 的 WebKB/Actor 收益主要是 raw baseline safety，而不是 learned complement 稳定超过 raw。
- 已新增 Chameleon/Squirrel loader 与配置。Chameleon/Squirrel split0-2 的 50 epoch sanity 显示 Raw-Complement `anchor_graph` 全部同时超过 raw baseline 与 GRACE：Chameleon raw-complement - raw F1Mi 为 +0.032895/+0.010965/+0.059211，Squirrel 为 +0.018252/+0.012488/+0.014409。
- Chameleon/Squirrel splits0-9 的 seed0 50 epoch 对照已经完成并由 `summarize_raw_complement_probe.py` 汇总：Raw-Complement 相对 raw baseline 在 Chameleon 上 F1Mi/F1Ma 平均 +0.033772/+0.033688，10/10 split 为正；在 Squirrel 上 +0.008742/+0.011555，10/10 split 为正。相对 GRACE 的均值提升更大：Chameleon +0.066228/+0.068241，Squirrel +0.065514/+0.076369。
- Chameleon/Squirrel splits0-9 × seeds0-2 的 50 epoch 多 seed 复核已经完成：Chameleon 相对 raw 为 +0.037208/+0.037554，30/30 pair 为正；相对 GRACE 为 +0.073319/+0.078816，30/30 pair 为正。Squirrel 相对 raw 为 +0.010086/+0.010904，28/30 pair 为正；相对 GRACE 为 +0.062184/+0.078077，30/30 pair 为正。
- Chameleon/Squirrel splits0-2 × seeds0-2 机制消融显示：`graph_only` 相对 raw 全负；`raw_graph=[raw, graph_context]` 明显弱于 residual complement，Squirrel 相对 raw F1Mi 变为 -0.006084；`anchor_only=[raw, complement]` 保留大部分收益但 Squirrel 有 2/9 个相对 raw 负例；`anchor_graph_weight0` 与默认几乎持平。
- Chameleon/Squirrel 完整 10 split seed0 复核显示：`raw_graph` 在 Chameleon 上 RC-raw F1Mi/F1Ma 为 +0.014035/+0.015452，但低于默认 +0.033772/+0.033688；在 Squirrel 上 RC-raw F1Mi/F1Ma 为 -0.006340/-0.000734，0/9 split 为正。`anchor_graph_weight0` 与默认持平甚至略高：Chameleon +0.034868/+0.035164，Squirrel +0.009702/+0.013379。
- No-penalty Chameleon/Squirrel splits0-9 × seeds0-2 复核已经完成：no-penalty 在 Chameleon 上 RC-raw F1Mi/F1Ma 为 +0.036769/+0.036744，30/30 pair 为正；Squirrel 为 +0.010471/+0.012456，相对 GRACE 30/30 为正。与早期 `0.05` 版本基本持平，且 Squirrel macro 略高。
- No-penalty homophily safety 检查显示：Cora seeds0-2 的 graph fallback 相对 GRACE 均值为 F1Mi/F1Ma -0.012351/-0.018962，label-based output selection 仍只有 0.812372/0.791086，不能解决 Cora 退化；CiteSeer seed0 micro 略升 macro 降约 1.0 个点，PubMed seed0 小幅退化约 0.28 个点。
- No-penalty Cora proxy selection 复核显示：label-free proxy v1 在 Cora seeds0-2 上 9/9 选择 graph，F1Mi/F1Ma=0.814473/0.794776，优于 random selection 与 label-based validation selection，但仍低于 GRACE seeds0-2 均值约 0.824948/0.810003。因此当前 proxy 不能作为同配 safety gate。
- 已新增 `--raw-complement-graph-loss-weight` 与 auxiliary graph-context projector，用于给 `graph_context` 分支增加可选 projected InfoNCE。未投影 direct graph loss 在 Cora 上失败；projected `0.1` 将 Cora seeds0-2 从 no-penalty graph 0.812597/0.791041 提升到 0.818567/0.794162，但仍低于 GRACE 0.824948/0.810003，并在 Chameleon/Squirrel split0 seed0 上分别低于 no-penalty 约 0.004/0.004 和 0.0067/0.0065。
- Auxiliary graph-context projector `0.1` 已完成 Chameleon/Squirrel splits0-9 × seed0 判定：Chameleon 相对 no-penalty F1Mi/F1Ma 约 +0.000000/+0.000056，Squirrel 为 +0.000672/-0.000118；它没有明显伤害主战场，但也没有主战场增益。
- 当前决策：Raw-Complement 在 WebKB/Actor 上降级为机制诊断与 output safety selection 资产，但在 Chameleon/Squirrel 上仍是当前 active candidate。它不是通用 heterophily SOTA claim，而是 raw-feature anchored complement learning 在 WikipediaNetwork-style heterophily graphs 上的条件性强候选。当前主贡献应收缩为 no-penalty raw-relative graph complement parameterization；correlation penalty 已从主方法移出，作为 optional regularizer / appendix ablation。Cora safety 是当前最大风险，现有 proxy 已证实不能解决该问题；auxiliary graph-context preservation 只能作为 optional safety refinement / appendix ablation，不作为核心创新。
- 完整 C 网格 split0-2：`ego_grace` concat - raw 在 Actor/Cornell/Texas 为正、Wisconsin 为负；`residual_grace` 仅 Actor 稳定正向，Cornell/Texas/Wisconsin 为负。
- 固定 C=1 的 10 split 快速筛查：ego/residual concat - raw 在 Actor/Cornell/Texas/Wisconsin 均为正，但该证据只能说明存在互补信号，不足以支撑 SOTA claim。
- 下一步应实现显式 raw-anchored residual/complement objective 或 light-validation fusion，而不是继续把 `ego_grace` / `residual_grace` 单独包装为方法。
- 暂停继续调 `gated_ego_graph_grace` 的 `graph_gate_min/max`；`--graph-gate-min 0.5` 已显示同配退化仍严重。
- 加入 Chameleon/Squirrel 前，先确认当前 loader/evaluator 能支持对应固定 split；若 WebKB raw-feature baseline 已远强于 SSL，WebKB 只能作为机制诊断而非 accuracy SOTA 主战场。
- 本目录中的 SGFN / context-gated SGFN 只作为负结果、诊断工具和消融资产保留。
