# Energy-Routed Cache GCL 研究日志

日期：2026-06-28

## 当前裁决

旧的 reliability-weighted GCL / projection-consistency 路线不再作为主方法继续推进。原因：

- split-aware 复核后，combined reliability 的 accuracy 增益不稳定；
- projection distribution consistency 不是分类语义一致性，不能作为强机制证据；
- positive anchor weighting 没有真正处理 denominator 中的 false negative；
- 该路线难以支撑 2026 顶会/顶刊级 SOTA claim。

SPARC-GCL / patched official SP-GCL residual branch 也不作为 active 主线。它保留了有价值现象：propagation residual 对 SP-GCL embedding 有稳定增益，但 patch third-party official code 的实现方式不适合作为主论文方法。

## 当前 active candidate：Energy-SPGCL

2026-06-28 third gate 设计：`gcn_mlp_gcl` 在 Texas 基本追平 GRACE、Chameleon micro 小正但 macro 下降、Squirrel 失败、Actor 正向，因此只能作为强对照而非新主线。

新的 active candidate 是 `energy_spgcl`：使用 raw low-pass / propagation signature 只负责 positive sampling，而不负责 bootstrap 拉近；真正的 InfoNCE 学习发生在 high-energy propagation residual 表示上，并显式加入随机负样本。这个设计更贴近“low-energy for sampling, high-energy for positive signal”的 2026 最新理论线索。

2026-06-28 fourth gate 更新：`energy_spgcl` 在 Texas/Chameleon split0 seed0 20 epoch 下均低于 GRACE，当前实现不再作为 active 主线。详见 `early_gate_summary_2026-06-28.md`。

## 已实现强对照：GCN-MLP Natural View GCL

2026-06-28 second gate 更新：`ER-Residual-GCL` 在 Texas/Chameleon 失败，只在 Actor 明显正向、Squirrel micro 弱正但 macro 下降，未满足保留标准。因此它降级为失败/条件性消融。

新的必要强对照是 `gcn_mlp_gcl`：不再强行使用 high-energy residual，而直接对齐 ego/MLP branch 与 graph/GCN branch，默认表示为 `[ego, graph]`。这既是当前 2025/2026 “天然双视图”趋势下必须击败的强基线，也用于判断 residual route 是否方向错误。

## 已放弃主线：ER-Residual-GCL

2026-06-28 early gate 更新：`ER-Cache-GCL` 的 normal positive cache 在 Texas split0 seed0 20 epoch 下低于 shuffled cache 和 self-only control，违反预设停止标准。因此 low-pass positive cache 不再作为主线；保留为失败消融。

随后主线曾收缩为 **ER-Residual-GCL**：只使用 same-node ego/MLP ↔ high-energy propagation residual bootstrap，不使用额外 positive cache。但四数据集 split0 seed0 early gate 后未通过保留标准。

## 已放弃子主张：ER-Cache-GCL

核心想法：

1. 用 graph encoder 产生结构表示 `z_g`；
2. 用一次 row-normalized propagation 得到 low-pass 表示 `P z_g`；
3. 用 residual `z_g - P z_g` 得到 high-energy 表示；
4. 用 low-pass 表示构建 label-free positive cache；
5. 用 high-energy residual 与 ego/MLP 分支执行 negative-free contrastive/distillation；
6. 最终表示默认使用 `[ego, high-residual]`，同时保留 MLP fast-inference 潜力。

## 与近年工作的区分

- 相对 GraphECL：同样重视 MLP fast inference，但 positive cache 和学习信号由 Dirichlet-energy/propagation residual 路由，而不是普通 cross-model alignment。
- 相对 PolyGCL / S3GCL：不学习全局 spectral polynomial filter，而是用 low-pass 负责采样、high-pass residual 负责 positive learning signal。
- 相对 2026 SPGCL positive-sample revisit：吸收“message passing 会 trivialize positives”的观点，但把它落到 cache construction 与 residual distillation 的训练框架中。
- 相对旧 reliability idea：不再用 projection-head softmax 一致性当语义可靠性；shuffled cache 是硬停止标准。

## 第一轮早筛标准

保留 ER-Residual-GCL 的最低标准：

- ER-Residual-GCL 在 Texas/Chameleon/Squirrel/Actor 的至少 2 个数据集上优于 GRACE；
- ER-Cache-GCL 若要恢复，normal cache 必须优于 shuffled cache；
- homophily Cora/CiteSeer/PubMed 不出现明显退化；
- 若 self-only / shuffled cache 接近或超过 normal，则放弃 cache 主张，转向 residual/ego 双视图机制。

## 已实现

- 标准入口：`experiments/topvenue_gcl/train.py`
- 数据加载：Planetoid、WebKB、Actor、WikipediaNetwork、HeterophilousGraphDataset
- 方法：`grace` baseline、`er_residual_gcl` 与失败消融 `er_cache_gcl`
- 控制：`--shuffle-cache`、`--disable-cache`
- 输出：每个 run 的 `run.json`、`artifacts.pt`、`train_log.csv`，以及 `runs/summary.csv`
