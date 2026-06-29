# 实现原则

## 必须满足

- 主方法代码必须在 `experiments/topvenue_gcl/src/` 内独立实现；
- 训练入口必须可通过 `scripts/*.sh` 或顶层 Python CLI 重现；
- 每个实验必须记录 dataset、split、seed、config、metric 与输出路径；
- linear evaluator 和训练逻辑必须使用同一套数据 split；
- 所有 early gate 都必须包含 random/shuffled 或 no-structure control。

## 禁止作为主线

- 只导出第三方模型 embedding 再做 post-hoc feature trick；
- 只在一个数据集或一个 split 上报结论；
- 只靠 dataset-specific 参数让结果成立；
- 只和弱 GRACE scaffold 比较；
- 只超过 learned-only GCL 但没有超过 `raw_features`；
- 继续在 `experiments/grace_idea/` 内堆叠新方法。

## 参考代码范式

- PolyGCL：spectral polynomial views + heterophily benchmark scripts；
- S3GCL：MLP inference + spectral biased views + semantic/spatial positives；
- GraphECL：fast inference + structure encoder / MLP encoder + heterophily scripts。

## 当前主线状态

当前 active foundation 是 `gcn_mlp_gcl`。它是必须击败的 strong control，但不是论文主贡献。

当前 active candidate 是 `ragc_gcl`，Raw-Anchored Graph Complement GCL。它沿用 `gcn_mlp_gcl` 的 Natural-View bootstrap 训练，但最终表示拼接 normalized raw features 与 learned Natural-View embedding；`raw_features` 是必须报告的强基线。Actor/Chameleon/Squirrel/Texas × splits0-9 × seed0 × 50 epoch 下，RAGC 相对 `raw_features` 的 F1Mi mean delta 分别为 +0.009408、+0.016886、+0.007397、+0.005405。Chameleon/Squirrel 的 10-split learned-branch controls 已通过：normal 显著高于 shuffle 与 random，且 random 显著低于 raw-only。`ragc_auto_gcl` 已实现验证集 safety selector，但无 margin 版本在 Actor split5 失败，margin=0.02 只作为 safety ablation。当前状态是最强 active candidate，不是最终成功方法；下一轮必须补 homophily safety、Actor/Texas 10-split controls、多 seed 与统一 paper table。

`tns_gcl` 已降级为失败/诊断资产：它尝试在 Natural-View bootstrap 上加入 trusted-negative repulsion，但 split0 seed0 只有 Actor 正向，Texas/Chameleon/Squirrel 均低于 `gcn_mlp_gcl`。该路线不继续调 margin、threshold 或 weight。

`bprrnv_gcl` 已从 active-but-risky candidate 降级为失败/弱正则资产：它把 RRNV 从主目标降级为 bootstrap-preserving auxiliary regularizer，并用 graph density 与 graph/high energy conflict 控制正则强度，但 10 split seed0 只有 overall +0.001398 F1Mi，且 Chameleon targeted controls 不支持干净机制。`rwirrnv_gcl` 已从 active-but-risky candidate 降级为机制线索：10 split seed0 中 Texas/Squirrel/Chameleon 有正向性能信号，但 Chameleon 的 constant-weight control 最好、Squirrel 的 shuffled-weight control 最好，说明当前 per-node reliability 排序不能作为主贡献。`eairrnv_gcl` 已验证 graph-level energy attenuation 有局部收益但不能过 Squirrel safety。后续不再继续围绕 RRNV auxiliary regularization 调小参数。

当前已经停止的主线：

- `er_cache_gcl`；
- `er_residual_gcl`；
- `energy_spgcl`；
- `danv_gcl` / `danv_degree_gcl`；
- `fdnv_gcl`；
- `sspnv_gcl` / `afpnv_gcl` / `bspnv_gcl` 作为最终主方法。
- `mpnv_gcl` 作为最终主方法。
- `aompnv_gcl` 作为最终主方法。
- `srgnv_gcl` 作为最终主方法。
- `dsp_gcl` 作为最终主方法。
- `darrnv_gcl` 作为最终主方法。
- `dirrnv_gcl` 作为最终主方法。
- `dprrnv_gcl` 作为最终主方法。
- `nprrnv_gcl` 作为最终主方法。
- `rwirrnv_gcl` 的当前 per-node reliability 排序作为最终主方法。
- `eairrnv_gcl` 的单一 graph-level energy attenuation 作为最终主方法。
- `bprrnv_gcl` 作为最终主方法。
- `tns_gcl` 作为最终主方法。

后续所有新候选必须同时报告：

- vs GRACE；
- vs `gcn_mlp_gcl`；
- vs shuffled/random/no-structure control；
- 失败数据集和停止条件。

## 历史候选：Energy-SPGCL 与 Natural-View foundation

工作假设：

用 raw low-pass / propagation signature 构造 positive sampler，但在 high-energy propagation residual 表示上执行 sampled InfoNCE；GCN-MLP Natural View GCL 是必须击败的强对照。

low-pass positive cache 在 Texas split0 seed0 early gate 中 normal 低于 shuffled，当前已降级为失败消融，不作为主贡献。

high-energy residual bootstrap 在 Texas/Chameleon 失败，只在 Actor 清楚正向，当前也降级为失败/条件性消融。

GCN-MLP Natural View GCL 已在 Texas/Actor/Chameleon/Squirrel × splits 0/1/2 × seed0 × 50 epoch 中相对 GRACE 全正向，因此升级为下一阶段 strong control / architecture foundation。它本身仍不够创新；后续方法必须稳定超过它。

第一版不要追求复杂模型，先实现：

- MLP online encoder；
- lightweight graph teacher；
- high-energy residual sampled InfoNCE；
- raw low-pass / propagation-signature positive sampler；
- GCN-MLP bootstrap strong control；
- high-energy residual 与 low-pass positive cache 仅作消融；
- 10 split node classification evaluator。

下一步强制要求：

- 新模块必须同时报告 vs GRACE 与 vs GCN-MLP；
- 若只超过 GRACE 而不超过 GCN-MLP，不保留为 active candidate；
- 若只在 WebKB/Actor 正向，需要明确边界，不能包装成通用 heterophily SOTA。

## RRNV 当前裁决

- `rrnv_gcl` 入口为 `--method rrnv_gcl`；
- 它在 MLP ego view 与 GCN graph view 的 predictor 输出上执行 VICReg/CCA 风格 redundancy reduction；
- `--rrnv-shuffle-pairs` 是必须保留的机制 control；
- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 mean ΔF1Mi 分别为 +0.099099、+0.002412、+0.008041、-0.008325；
- normal-vs-shuffled mean ΔF1Mi 分别为 Texas +0.054054、Actor +0.006360、Chameleon +0.007310、Squirrel +0.002241；
- 当前解释：Texas 是最强证据，Chameleon 有小正与较干净 control，Actor 是弱辅助，Squirrel 是明确风险；
- RRNV 升级为 active-but-risky candidate，但不是最终成功方法；
- DS-RRNV 已取代 RRNV 成为当前更优候选。

## DS-RRNV 当前裁决

- `dsrrnv_gcl` 入口为 `--method dsrrnv_gcl`；
- 它保留 RRNV 训练目标，但在 final representation 中根据图平均度 gate 混合 graph/high residual view；
- 默认 gate：Texas 约 0.000061、Actor 约 0.000617、Chameleon 约 0.076343、Squirrel 约 0.744020；
- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 mean ΔF1Mi 分别为 +0.090090、+0.001974、+0.013158、+0.006724；
- normal-vs-shuffled mean ΔF1Mi 分别为 Texas +0.072072、Actor +0.005482、Chameleon +0.014620、Squirrel -0.012168；
- 当前解释：Texas/Chameleon 是主要证据，Squirrel safety 有均值改善但机制 control 失败；
- DS-RRNV 是 active-but-risky candidate，不是成功主方法；
- 后续必须解释或修复 Squirrel 上 shuffled 更强的问题。

## DIRRNV 当前裁决

- `dirrnv_gcl` 入口为 `--method dirrnv_gcl`；
- 它在 DS-RRNV 基础上对高密度图衰减 true-pair invariance；
- split0 seed0 未救回 Squirrel，Texas 也弱于 DS-RRNV；
- DIRRNV 降级为失败 safety 变体，不进入 splits 0-2。

## DPRRNV 当前裁决

- `dprrnv_gcl` 入口为 `--method dprrnv_gcl`；
- 它沿用 DS-RRNV final representation，并在 RRNV invariance target 中按图平均度 high gate 混入 shuffled graph target；
- split0 seed0 normal vs `gcn_mlp_gcl`：Texas +0.027027、Actor +0.003289、Chameleon +0.002193、Squirrel +0.026897；
- normal-vs-full-shuffled：Texas +0.027027、Actor -0.005263、Chameleon -0.015351、Squirrel +0.019212；
- DPRRNV 修复 Squirrel，但 Actor/Chameleon control 不干净，且 Texas/Chameleon 弱于 DS-RRNV；
- DPRRNV 只保留为高密度图配对可靠性机制线索，不进入 splits 0-2。

## NPRRNV 当前裁决

- `nprrnv_gcl` 入口为 `--method nprrnv_gcl`；
- 它将 DPRRNV 的图级 target perturbation 改成节点级，gate 由 degree、raw residual、raw agreement 与 ego-graph view cosine 估计；
- 默认 split0 seed0 normal vs `gcn_mlp_gcl`：Texas +0.000000、Actor -0.011184、Chameleon -0.039474、Squirrel +0.014409；
- strict `--nprrnv-min-local-scale 0.0`：Chameleon -0.013158、Squirrel +0.028818；
- NPRRNV 对 Squirrel 有正信号，但默认和 strict 都无法避免 Chameleon 退化；
- NPRRNV 降级为 negative/diagnostic asset，不进入 splits 0-2；后续不再调 `nprrnv_min_local_scale`。

## RWIRRNV 当前裁决

- `rwirrnv_gcl` 入口为 `--method rwirrnv_gcl`；
- 它复用 NPRRNV 的节点级不可靠性估计，但不扰动 target，只对 RRNV invariance 做 per-node reliability weighting；
- `--rwirrnv-shuffle-weight` 是必须保留的 reliability 排序 control；
- split0 seed0 normal vs `gcn_mlp_gcl`：Texas +0.081081、Actor +0.009211、Chameleon +0.043860、Squirrel -0.008646；
- normal-vs-shuffled-weight：Texas +0.108108、Actor -0.003289、Chameleon +0.028509；Squirrel 因 normal 已失败未跑 control；
- splits 0-2、seed0 normal vs `gcn_mlp_gcl`：Texas +0.117117、Chameleon +0.013889、Squirrel +0.014089、Actor +0.002851；
- splits 0-2、seed0 normal-vs-shuffled-weight：Texas +0.063063、Chameleon +0.008772、Squirrel +0.001601、Actor +0.002193；
- 已新增 `--rwirrnv-constant-weight` 同均值常数权重 control；
- splits 0-9、seed0 normal vs `gcn_mlp_gcl`：Texas +0.075676、Chameleon +0.013158、Squirrel +0.022574、Actor -0.001513；
- splits 0-9、seed0 shuffled-weight vs `gcn_mlp_gcl`：Texas +0.032432、Chameleon +0.014254、Squirrel +0.026705、Actor -0.004934；
- splits 0-9、seed0 constant-weight vs `gcn_mlp_gcl`：Texas +0.072973、Chameleon +0.015132、Squirrel +0.018636、Actor -0.001184；
- 当前解释：invariance attenuation 有信号，但 per-node reliability 排序没有通过 control；Chameleon 常数最好，Squirrel shuffled 最好，Actor 失败；
- RWIRRNV 当前形态降级为机制线索，后续应设计 graph-level / schedule-level attenuation，而不是继续调 reliability score。

## EAIRRNV 当前裁决

- `eairrnv_gcl` 入口为 `--method eairrnv_gcl`；
- 它在 RRNV 中用 graph/high view energy ratio 估计图级 conflict，并据此衰减 true-pair invariance：`scale = 1 - strength * energy / (energy + threshold)`；
- 默认参数：`eairrnv_energy_threshold=0.15`、`eairrnv_strength=0.6`、`eairrnv_power=1.0`、`eairrnv_min_invariance_scale=0.25`；
- 已完成 Texas/Chameleon/Squirrel/Actor × splits 0-2 × seed0 × 50 epoch 的 strength sweep；
- strength=0.6 vs `gcn_mlp_gcl`：Texas +0.126126、Chameleon +0.031433、Actor +0.001754、Squirrel -0.005123；
- strength=0.3 vs `gcn_mlp_gcl`：Texas +0.099099、Chameleon +0.021930、Actor +0.003509、Squirrel -0.008005；
- strength=0.9 vs `gcn_mlp_gcl`：Texas +0.099099、Chameleon +0.013889、Actor +0.003509、Squirrel -0.008005；
- 当前裁决：EAIRRNV 证明 graph-level energy attenuation 是有用机制线索，但单一全图 scale 无法修复 Squirrel，且 strength sweep 不存在稳健全局最优；不进入 10 split 扩展，不作为主方法；
- 对照 DARRNV：`darrnv_gcl` 在同一 splits 0-2 seed0 下 Texas -0.027027、Chameleon +0.023392、Actor +0.003728、Squirrel -0.000640，说明 density auxiliary gate 能保护 Squirrel 但会损失 Texas 主信号；
- 下一代若继续 RRNV，必须从“替代式 RRNV”改成“bootstrap-preserving selective regularization / selector”，显式判断何时使用 RRNV、何时回退到 `gcn_mlp_gcl`，而不是继续调 `eairrnv_strength`。

## BPRRNV 当前裁决

- `bprrnv_gcl` 入口为 `--method bprrnv_gcl`；
- 它保留 `gcn_mlp_gcl` 的 bootstrap loss，不再让 RRNV 取代主目标；
- 辅助正则为 `bprrnv_rr_weight * aux_gate * (invariance + 0.1 variance + 0.01 covariance)`；
- `aux_gate = density_factor * energy_factor`，其中 density factor 在高平均度图上衰减，energy factor 在 graph/high energy conflict 高时衰减；
- `--rrnv-shuffle-pairs` 是 pair correspondence control，`--bprrnv-uniform-gate` 是去掉 density/energy selector 的 control；
- 默认强度已从失败的 `bprrnv_rr_weight=0.25` 下调到 `0.1`；
- 0.25 在 Texas/Chameleon/Squirrel/Actor × splits0-2 × seed0 × 50 epoch 下整体失败：相对 `gcn_mlp_gcl` 的 overall mean micro delta 为 -0.005938；
- 0.1 normal vs `gcn_mlp_gcl`：Texas +0.027027、Chameleon +0.003655、Squirrel +0.000640、Actor +0.005044，overall +0.009092；
- 0.1 uniform gate vs `gcn_mlp_gcl`：Texas +0.000000、Chameleon +0.006579、Squirrel -0.005123、Actor +0.002632，overall +0.001022；
- 0.1 shuffled pair vs `gcn_mlp_gcl`：Texas -0.027027、Chameleon -0.010234、Squirrel -0.007365、Actor -0.001754，overall -0.011595；
- normal-vs-shuffled mean delta：Texas +0.054054、Chameleon +0.013889、Squirrel +0.008005、Actor +0.006798，overall +0.020687；
- 已执行 Texas/Chameleon/Squirrel/Actor × splits0-9 × seed0 × 50 epoch normal/baseline 硬门槛；
- 10 split normal vs `gcn_mlp_gcl`：Texas +0.000000、Chameleon +0.005044、Squirrel -0.000768、Actor +0.001316，overall +0.001398；
- 10 split macro delta overall 为 -0.000840，说明 micro 小正不伴随宏平均改善；
- 正/平/负 split 数：Texas 2/4/4、Chameleon 7/1/2、Squirrel 6/0/4、Actor 6/0/4，overall 21/5/14；
- Chameleon targeted controls：normal +0.005044、shuffled +0.003728、uniform +0.001974 vs `gcn_mlp_gcl`；
- Chameleon normal-vs-shuffled mean 只有 +0.001316，且 normal 只在 3/10 split 高于 shuffled、5/10 低于 shuffled；
- 当前解释：BPRRNV 的 density/energy selective RR regularizer 只带来噪声级或弱正则级别收益；Chameleon 有小均值正，但 pair correspondence control 不干净；Texas/Squirrel/Actor 不支持扩展；
- 当前裁决：BPRRNV 降级为失败/弱正则资产，不再作为 active candidate，不补多 seed、homophily safety 或 no-density/no-energy controls。后续如果继续 natural-view 路线，必须换成直接面向 false-negative / downstream separability / negative suppression 的目标，而不是继续调 RRNV auxiliary strength。

## DSP / DARRNV 停止记录

- `dsp_gcl` 用 downstream separability proxy 进行节点级 loss weighting；split0 seed0 中 Texas 被 shuffled 反超，Chameleon/Squirrel 低于 baseline，已降级；
- `darrnv_gcl` 用图平均度 gate 将 RRNV 作为 BYOL 的小辅助项；Texas 丢失 RRNV 主信号、Actor 低于 baseline，已中止；
- 后续不再继续调 DSP weight temperature、DARRNV degree threshold 或 auxiliary weight。

当前下一代方法假设：

- 名称：Disagreement-Aware Natural-View GCL (DANV-GCL)；
- 底座：GCN-MLP natural views；
- 核心：对一致节点做 alignment，对冲突节点保留/去相关互补信息；
- gate：raw-neighbor agreement、view cosine、feature propagation residual energy；
- 控制：必须超过 `gcn_mlp_gcl`，否则放弃。

DANV splits 0/1/2 early gate：

- Texas/Chameleon/Squirrel 的 mean micro 超过 GCN-MLP，Actor mean micro 仅小幅超过但 3 个 split 中 2 个为负；
- Chameleon/Squirrel 共 6 个 split micro 全正，macro 也为正，是当前最干净的 DANV 信号；
- Texas mean micro 正向但 macro 下降，split0 macro 退化尤其明显；
- 当前状态仍是 active-but-risky，不是成功方法；
- 下一步只允许做 gate/penalty 消融与 safety fallback：`danv_disagreement_weight=0`、`0.02`、gate temperature、min-align；
- 若消融不能缓解 Texas macro 与 Actor split instability，应放弃 DANV 主线或收缩为 WikipediaNetwork 条件性机制。

DANV 消融裁决：

- `danv_disagreement_weight=0.0`、`0.02` 和 `danv_degree_gcl` 均未形成稳定主方法；
- 固定全局 disagreement penalty 没有稳定有效区间；
- degree-aware disagreement gate 在 split0 上未超过 early gate；
- DANV 家族降级为失败/条件性消融资产，不再作为当前 active idea；
- 下一代方法仍可继承 `gcn_mlp_gcl` natural-view foundation，但必须换机制。

FDNV 第一版裁决：

- `fdnv_gcl` 显式学习 low/high filtered targets，比 DANV penalty 更贴近 S3GCL/PolyGCL 的 filter 方向；
- `fdnv_route_weight=0.5` 与 `0.1` 都在 Chameleon split0 上失败；
- FDNV 第一版不进入 splits 0/1/2，不作为 active main idea；
- 后续若继续 filter 方向，必须改 objective，而不是继续调 route weight；
- 下一步优先考虑 semantic/spatial positive split 或 high/low branch complementarity。

SSPNV / AFPNV / BSPNV 当前裁决：

- `sspnv_gcl` 的固定完整版本保留为机制原型，但不再作为最终主方法包装，入口为 `--method sspnv_gcl`；
- 方法使用 semantic positives 监督 high-pass target，spatial positives 监督 low-pass target，并保留 GCN-MLP Natural-View bootstrap；
- 10 split / seed0 / 50 epoch 下相对 `gcn_mlp_gcl` 在 Texas、Actor、Chameleon、Squirrel 的 mean micro/macro 均为正；
- Chameleon 10/10 split micro 正向，Squirrel 9/10 split micro 正向，是当前最强实验信号；
- Actor 仅弱正且 4/10 split 为负，必须作为边界而非主成功证据；
- component ablation 与 random-positive control 已完成 Chameleon/Squirrel 10 split；
- Chameleon 上 semantic-only、spatial-only 与 random semantic 均接近或超过完整 SSPNV，说明固定 semantic-spatial 双分支同权相加不是必要机制；
- Squirrel 上 random semantic / random spatial 失败，说明结构化 positives 有条件性价值，但 full SSPNV 增益仍偏小；
- `afpnv_gcl` 已实现为置信度加权版本，入口为 `--method afpnv_gcl`；它在 Chameleon/Squirrel 上均未超过对应最强 SSPNV 变体，当前只保留为 ablation；
- `bspnv_gcl` 已实现为 semantic / spatial / bootstrap branch selection，入口为 `--method bspnv_gcl`；它强于 AFPNV，但仍未同时超过 Chameleon semantic-only 与 Squirrel full SSPNV；
- SSPNV 家族已触发停止条件，全部降级为失败/条件性消融资产；
- 不再继续调 SSPNV/AFPNV/BSPNV 的 threshold、temperature、branch bias；
- 下一代 active idea 必须换训练目标或参考范式，继续保留 `gcn_mlp_gcl` 作为 strong foundation，但不要再把 filter-specific positives 的小变体包装成主线。

MPNV 当前裁决：

- `mpnv_gcl` 入口为 `--method mpnv_gcl`；
- 它不再做单一 semantic/spatial positive 采样，而是构造 dense semantic mask 与 dense spatial mask；
- semantic mask 来源于 raw propagation signature KNN，监督 high-pass target；
- spatial mask 来源于原图一跳邻居，监督 low-pass target；
- 保留 `gcn_mlp_gcl` 的 Natural-View bootstrap；
- `--mpnv-shuffle-positives` 是必须保留的机制 control，用于打乱 positive mask 与节点对应关系；
- Chameleon/Squirrel × splits 0-9 × seed0 × 50 epoch 中，MPNV 相对 `gcn_mlp_gcl` 曾分别取得 +0.017105/+0.019132 与 +0.015082/+0.014767 的 F1Mi/F1Ma mean delta；
- 但 Texas/Actor/Chameleon/Squirrel × splits 0-9 × seeds 1/2 × 50 epoch 复核失败：ΔF1Mi 分别为 +0.002703、-0.001776、+0.000219、-0.000288；
- Squirrel seed1/seed2 不再保持 normal 10/10 split micro 正向，seed0 核心机制证据失效；
- 当前裁决是降级为失败/条件性消融资产，不再作为 active main idea；
- 后续不再继续跑 MPNV shuffled seed1/2，因为 normal gate 已失败；
- 下一代方法必须加入 label-free objective activation / node-level fallback，或彻底更换训练目标。

AOMPNV 当前裁决：

- `aompnv_gcl` 入口为 `--method aompnv_gcl`；
- 它复用 MPNV 的 dense semantic/spatial mask，但不再固定相加 semantic/spatial/bootstrap loss；
- 每个节点根据 semantic dense InfoNCE、spatial dense InfoNCE、bootstrap negative cosine 的相对自监督 loss，以及 raw-signature positive confidence，路由到 semantic / spatial / bootstrap objective；
- `--aompnv-shuffle-positives` 是必须保留的机制 control；
- Texas/Actor/Chameleon/Squirrel × splits 0-2 × seeds 1/2 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.022523、+0.001864、+0.015351、+0.018892；
- normal-vs-shuffled 的 ΔF1Mi 分别为 Texas -0.000000、Actor +0.009539、Chameleon +0.006579、Squirrel +0.003522；
- 已完成 splits 0-9 × seeds 1/2 的 normal/shuffled 硬门控；
- 相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 Texas +0.010811、Actor -0.002829、Chameleon +0.000658、Squirrel +0.018348；
- normal-vs-shuffled 的 ΔF1Mi 分别为 Texas +0.008108、Actor +0.007763、Chameleon -0.011294、Squirrel +0.013593；
- 当前解释：只有 Squirrel 同时给出较清楚的 baseline 增益与 shuffled control 差距；Texas 正向但 split-level 不稳，Actor 低于 baseline，Chameleon shuffled 明显更强；
- AOMPNV 已降级为 regularization / negative-result ablation，不再作为主线；
- 后续不再继续调 AOMPNV 的 router temperature、branch weight 或 confidence threshold；
- 下一代方法必须换训练机制，同时保留 `gcn_mlp_gcl` strong foundation 与 shuffled/random/no-structure control。

SRGNV 当前裁决：

- `srgnv_gcl` 入口为 `--method srgnv_gcl`；
- 它尝试将 graph view 分解为 ego/feature 可解释部分与 structure residual，并用 raw feature propagation residual score 做节点级 gate；
- `--srgnv-shuffle-residual` 是必须保留的 no-structure control；
- Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.000000、+0.001974、-0.041667、-0.005764；
- shuffled residual 的 ΔF1Mi 分别为 -0.027027、+0.002632、-0.021930、-0.028818；
- 当前解释：SRGNV 能优化 residual cosine，但下游分类不稳定受益；Actor 唯一正向还被 shuffled control 超过，Chameleon/Squirrel 明显失败；
- SRGNV 已降级为 negative-result ablation，不进入 splits 0-2 扩展；
- 后续不再继续调 residual weight、threshold 或 temperature；
- 下一代方法必须直接约束或诊断 downstream separability / neighborhood conflict，而不是只蒸馏 representation residual。

PCNV 当前裁决：

- `pcnv_gcl` 入口为 `--method pcnv_gcl`；
- 它保留 `gcn_mlp_gcl` 的 Natural-View bootstrap，并加入 trainable prototypes；
- ego view 与 graph view 在 prototype assignment 空间做双向 consistency，target stop-gradient；
- `--pcnv-shuffle-assignments` 是必须保留的机制 control；
- default PCNV 在 Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 +0.027027、+0.007895、+0.028509、-0.014409；
- default shuffled 的 ΔF1Mi 分别为 +0.000000、+0.013158、+0.030702、-0.016330，说明 Actor/Chameleon 机制 control 不干净；
- sharpened PCNV 使用 `--pcnv-prototype-weight 0.5 --pcnv-balance-weight 0.1 --pcnv-assignment-temperature 0.1 --pcnv-target-temperature 0.03`；
- sharpened PCNV 在 Texas 达到 F1Mi/F1Ma=0.729730/0.459091，强于 default 与 shuffled，是当前最强 Texas 单点；
- 但 sharpened PCNV 在 Squirrel 明显失败，且 Chameleon/Squirrel 的 prototype usage entropy 过低，提示原型坍塌；
- 已新增 entropy-guarded / confidence-weighted PCNV 与 view-agreement gated PCNV；
- sharp guarded PCNV 在 Texas 达到强正向，但 Actor/Squirrel 失败，Chameleon/Squirrel usage entropy 明显过低；
- soft guarded PCNV 是最健康变体：Texas macro 与 Actor 正向，但 Chameleon shuffled control 反超，Squirrel 仍失败；
- view-agreement gated PCNV 在 Texas/Actor/Chameleon/Squirrel split0 seed0 上全面弱于 soft guarded，并使 Squirrel 明显退化；
- 当前裁决：PCNV 已降级为 conditional / diagnostic asset，不再作为 active main idea；
- 后续不再继续调 PCNV 的 temperature、confidence threshold、entropy guard 或 view-agreement gate；
- 下一代方法必须换机制，优先考虑节点级局部结构条件下的 objective selection 或更直接的 downstream separability 代理。

LCOS 当前裁决：

- `lcos_gcl` 入口为 `--method lcos_gcl`；
- 它使用 raw feature local conflict gate 在完整 graph view alignment 与 high-pass view alignment 之间做节点级 objective selection；
- `--lcos-shuffle-gate` 是必须保留的机制 control；
- 已修复 raw residual 数值问题：使用 `||x - P x|| / (||x|| + ||P x||)`，避免零特征节点导致 residual 爆炸；
- Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 -0.054054、+0.009211、-0.004386、+0.013449；
- normal-vs-shuffled 的 ΔF1Mi 分别为 +0.000000、+0.001974、+0.015351、+0.050913；
- 当前解释：Squirrel 给出最清楚的局部冲突 gate 机制线索，但 Texas micro 和 Chameleon baseline gate 失败；
- LCOS 第一版已降级为失败/条件性诊断资产，不进入 splits 0-2 扩展；
- 后续不再继续调 LCOS route threshold、temperature 或 degree weight；
- 已实现 `lcm_gcl` 作为 LCOS 的 final-only 后续变体：训练目标保持 `gcn_mlp_gcl`，只在 final representation 中使用 local-conflict graph/high mix；
- LCM 在 Texas/Actor/Chameleon/Squirrel × split0 × seed0 × 50 epoch 中，相对 `gcn_mlp_gcl` 的 ΔF1Mi 分别为 -0.054054、+0.004605、+0.006579、+0.000961；
- LCM normal-vs-shuffled 的 ΔF1Mi 分别为 -0.135135、+0.015132、+0.024123、+0.001921，Texas shuffled 明显更强；
- LCM 也已降级，不进入 splits 0-2 扩展；
- 若继承 LCOS/LCM 线索，必须改变目标设计：局部冲突 gate 应用于 loss reliability、negative suppression 或 downstream separability proxy，而不是直接对齐 high-pass target，也不是简单 graph/high final mix。
