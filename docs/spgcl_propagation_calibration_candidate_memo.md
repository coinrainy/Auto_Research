# SP-GCL Propagation-Residual Calibration 候选备忘录

日期：2026-06-28

## 当前候选

新的 active candidate 暂命名为：

> SPARC-GCL: Strong Propagation-residual Adaptive Representation Calibration for Graph Contrastive Learning

核心想法：

- 不再从弱 GRACE 变体重新发明 single-pass objective；
- 以 official SP-GCL 这类强 augmentation-free GCL embedding 为 backbone；
- 在训练后对 learned embedding 做轻量传播校准；
- 关键不是简单 raw concat，而是拼接 `SSL embedding + propagation residual / propagation-calibrated embedding`；
- 目标是保留 SP-GCL 的强判别信息，同时显式暴露“局部传播会改变什么”的残差信号。

当前最有前景的 mode：

- `ssl_resid1`: `[normalize(z), normalize(z - Pz)]`
- `ssl_prop2`: `[normalize(z), normalize(P^2 z)]`

其中 `P` 是带 self-loop 的 row-normalized graph propagation operator。

## 为什么这条线不同于 Raw-Complement / PGSP

Raw-Complement 失败点：

- 只赢 GRACE/raw，不能赢 SP-GCL；
- raw concat / raw complement 在强 SP-GCL embedding 上没有自然优势；
- homophily safety 不稳。

PGSP 失败点：

- 直接用 propagation signature 做 pseudo-positive training 明显弱于 official SP-GCL；
- 说明“传播签名作为训练目标”不是核心增益。

SPARC-GCL 的新信号：

- official SP-GCL embedding 已经很强；
- 对强 embedding 做 propagation residual calibration 后，Squirrel 出现跨 split 稳定提升；
- Chameleon 也有小幅正向，但幅度较小，需要继续验证。

## 当前证据

输入 embedding：

- 来自本地 official SP-GCL 克隆；
- 配置：`reset_epochs=100`、`reset_hidden=256`、`reset_seed_num=32`、`reset_max_size=512`、`reset_subg_num_hops=2`；
- 只保存 SSL embedding，后续用当前项目的 mask linear probe 统一评估。

评估脚本：

- `experiments/grace_idea/evaluate_propagation_calibration.py`
- `experiments/grace_idea/scripts/run_spgcl_embedding_export.sh`

后者会自动导出 Geom-GCN 数据、给本地 ignored SP-GCL 克隆注入 `SPGCL_EMBED_OUT` 保存钩子、运行 official SP-GCL，并转换为当前评估脚本可读取的 `artifacts.pt`。

快速 gate：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python evaluate_propagation_calibration.py \
  --runs-dir /tmp/spgcl_embedding_artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_prop2 ssl_resid1 \
  --max-hop 2 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 16 \
  --max-iter 300 \
  --out runs/summaries/spgcl_propagation_calibration_splits0-9_fast.csv \
  --aggregate-out runs/summaries/spgcl_propagation_calibration_splits0-9_fast_aggregate.csv
```

结果：

| Dataset | Mode | F1Mi mean | F1Ma mean | Delta vs SSL F1Mi | Delta vs SSL F1Ma | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chameleon | ssl | 0.623246 | 0.623299 | 0.000000 | 0.000000 | - |
| Chameleon | ssl_prop2 | 0.625439 | 0.626038 | +0.002193 | +0.002739 | 6/4 |
| Chameleon | ssl_resid1 | 0.632237 | 0.631764 | +0.008991 | +0.008465 | 7/2 |
| Squirrel | ssl | 0.440154 | 0.434061 | 0.000000 | 0.000000 | - |
| Squirrel | ssl_prop2 | 0.459270 | 0.454152 | +0.019116 | +0.020091 | 10/0 |
| Squirrel | ssl_resid1 | 0.468588 | 0.463756 | +0.028434 | +0.029695 | 10/0 |

解释：

- Squirrel 的 propagation residual calibration 信号非常清楚；
- Chameleon 的提升较小，但 `ssl_resid1` 仍为正均值；
- 这只是 fixed `C=16` 的 fast gate，不是正式结果；
- 当前证据足以把 SPARC-GCL 作为下一轮 active candidate，但还不足以声称 SOTA。

## C-grid 复核

为排除 fixed `C=16` 偶然性，已用 `C={4,16,64}`、`max_iter=500` 复核 Chameleon/Squirrel 10 splits：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir /tmp/spgcl_embedding_artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_prop1 ssl_prop2 ssl_resid1 ssl_resid2 \
  --max-hop 2 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/spgcl_propagation_calibration_splits0-9_cgrid.csv \
  --aggregate-out runs/summaries/spgcl_propagation_calibration_splits0-9_cgrid_aggregate.csv
```

结果：

| Dataset | Mode | F1Mi mean | F1Ma mean | Delta vs SSL F1Mi | Delta vs SSL F1Ma | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chameleon | ssl | 0.629167 | 0.629475 | 0.000000 | 0.000000 | - |
| Chameleon | ssl_prop1 | 0.630263 | 0.631050 | +0.001096 | +0.001575 | 5/4 |
| Chameleon | ssl_prop2 | 0.634430 | 0.635059 | +0.005263 | +0.005584 | 7/3 |
| Chameleon | ssl_resid1 | 0.637500 | 0.637427 | +0.008333 | +0.007952 | 5/5 |
| Chameleon | ssl_resid2 | 0.636404 | 0.636519 | +0.007237 | +0.007044 | 7/2 |
| Squirrel | ssl | 0.451201 | 0.446531 | 0.000000 | 0.000000 | - |
| Squirrel | ssl_prop1 | 0.478482 | 0.475293 | +0.027281 | +0.028761 | 10/0 |
| Squirrel | ssl_prop2 | 0.483093 | 0.479413 | +0.031892 | +0.032882 | 10/0 |
| Squirrel | ssl_resid1 | 0.485495 | 0.481700 | +0.034294 | +0.035169 | 10/0 |
| Squirrel | ssl_resid2 | 0.479923 | 0.476676 | +0.028722 | +0.030144 | 10/0 |

C-grid 复核后，Squirrel 的提升幅度更强且 10/10 split 稳定为正；Chameleon 仍是小幅正向，`ssl_resid1/2` 与 `ssl_prop2` 均高于原始 SSL embedding。

## 项目内可复现 artifacts 复核

已使用 `scripts/run_spgcl_embedding_export.sh` 重新生成项目内 artifacts：

```bash
DATASETS="Chameleon Squirrel" \
OUT_DIR="runs/spgcl_official_embeddings_seed42_e100" \
RESET_EPOCHS=100 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=256 \
RESET_SEED_NUM=32 \
RESET_MAX_SIZE=512 \
RESET_SUBG_NUM_HOPS=2 \
SEED=42 \
bash scripts/run_spgcl_embedding_export.sh
```

随后在新 artifacts 上复跑同一 C-grid：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/spgcl_official_embeddings_seed42_e100/artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_prop1 ssl_prop2 ssl_resid1 ssl_resid2 \
  --max-hop 2 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_seed42_cgrid.csv \
  --aggregate-out runs/summaries/sparc_seed42_cgrid_aggregate.csv
```

结果：

| Dataset | Mode | F1Mi mean | F1Ma mean | Delta vs SSL F1Mi | Delta vs SSL F1Ma | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chameleon | ssl | 0.631798 | 0.632267 | 0.000000 | 0.000000 | - |
| Chameleon | ssl_prop1 | 0.637500 | 0.638438 | +0.005702 | +0.006171 | 7/3 |
| Chameleon | ssl_prop2 | 0.640570 | 0.641855 | +0.008772 | +0.009588 | 8/2 |
| Chameleon | ssl_resid1 | 0.639254 | 0.639301 | +0.007456 | +0.007034 | 6/4 |
| Chameleon | ssl_resid2 | 0.634211 | 0.634237 | +0.002412 | +0.001969 | 4/6 |
| Squirrel | ssl | 0.450817 | 0.445798 | 0.000000 | 0.000000 | - |
| Squirrel | ssl_prop1 | 0.474832 | 0.471390 | +0.024015 | +0.025592 | 10/0 |
| Squirrel | ssl_prop2 | 0.480307 | 0.476469 | +0.029491 | +0.030670 | 10/0 |
| Squirrel | ssl_resid1 | 0.488953 | 0.485416 | +0.038136 | +0.039618 | 10/0 |
| Squirrel | ssl_resid2 | 0.474736 | 0.471220 | +0.023919 | +0.025421 | 10/0 |

该结果将 SPARC-GCL 的证据从 `/tmp` 临时 artifacts 推进到项目内可复现 artifacts。当前最强模式仍是 Squirrel 上的 `ssl_resid1`，Chameleon 上更偏向 `ssl_prop2`。

## 机制诊断：class / degree / local homophily bucket

已新增诊断脚本：

- `experiments/grace_idea/analyze_sparc_diagnostics.py`

该脚本在同一 official SP-GCL embedding 上重训相同协议的 mask linear probe，并记录相对 `ssl` baseline 的逐节点正确性变化：

- split 级整体 gain/loss；
- degree low/mid/high bucket；
- local label homophily low/mid/high bucket；
- class 级 correct-rate delta。

运行命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python analyze_sparc_diagnostics.py \
  --runs-dir runs/spgcl_official_embeddings_seed42_e100/artifacts \
  --include-methods spgcl_official \
  --modes ssl_prop2 ssl_resid1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out-prefix runs/summaries/sparc_seed42_diagnostics
```

输出文件：

- `runs/summaries/sparc_seed42_diagnostics_splits.csv`
- `runs/summaries/sparc_seed42_diagnostics_splits_aggregate.csv`
- `runs/summaries/sparc_seed42_diagnostics_buckets.csv`
- `runs/summaries/sparc_seed42_diagnostics_buckets_aggregate.csv`
- `runs/summaries/sparc_seed42_diagnostics_classes.csv`
- `runs/summaries/sparc_seed42_diagnostics_classes_aggregate.csv`

整体 split 级结果与 C-grid 复核一致：

| Dataset | Mode | SSL correct | Mode correct | Delta | Gain/Loss | Positive/Negative splits |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chameleon | `ssl_prop2` | 0.631798 | 0.640570 | +0.008772 | 167/127 | 8/2 |
| Chameleon | `ssl_resid1` | 0.631798 | 0.639254 | +0.007456 | 267/233 | 6/4 |
| Squirrel | `ssl_prop2` | 0.450817 | 0.480307 | +0.029491 | 843/536 | 10/0 |
| Squirrel | `ssl_resid1` | 0.450817 | 0.488953 | +0.038136 | 1082/685 | 10/0 |

bucket 级关键发现：

- Squirrel 上 `ssl_resid1` 对 degree high/low/mid 三个桶均为正，delta 分别约 +0.0536、+0.0361、+0.0244。
- Squirrel 上 `ssl_resid1` 对 local homophily high/low/mid 三个桶也均为正，delta 分别约 +0.0205、+0.0387、+0.0550。
- Squirrel 上 `ssl_prop2` 同样在所有 bucket 上为正，说明收益不是单一 high-degree 或单一 local-homophily 区域造成的。
- Chameleon 上 `ssl_prop2` 较稳，degree 三个桶均为非负，local homophily low/high 为正，但 mid 略负。
- Chameleon 上 `ssl_resid1` 明显帮助 local-homophily low/mid 桶（约 +0.0349/+0.0250），但伤害 local-homophily high 桶（约 -0.0386，9/10 split 为负）。

class 级关键发现：

- Squirrel 上 `ssl_resid1` 在 5 个 class 上均为正，delta 约 +0.0331、+0.0378、+0.0368、+0.0462、+0.0373。
- Squirrel 上 `ssl_prop2` 也在 5 个 class 上均为正，delta 约 +0.0186 到 +0.0402。
- Chameleon 上 `ssl_prop2` 在 class 2/0/4 为正，class 3 略负，class 1 接近 0。
- Chameleon 上 `ssl_resid1` 对 class 3/1/0/4 为正，但 class 2 为负；这进一步说明 Chameleon 需要 mode/gate，而不是固定 residual 模式。

机制解释更新：

- Squirrel 的提升具有跨 class、degree bucket 与 local-homophily bucket 的普遍性，支持继续把 SPARC 作为 active candidate。
- Chameleon 的收益更像局部条件性收益：residual branch 能帮助低/中 local-homophily 区域，但会损害高 local-homophily 区域；propagation branch 更平滑、更稳。
- 因此下一步不应简单固定 `ssl_resid1`，而应设计 label-free mode selection 或 node/dataset-level gate，在 `ssl`、`ssl_prop2` 与 `ssl_resid1` 之间自适应选择。

## Label-free mode selection v1

已新增脚本：

- `experiments/grace_idea/select_sparc_mode_proxy.py`

该脚本评估 SPARC 候选模式，并输出：

- 每个 candidate mode 的 mask linear probe 结果；
- 无标签 balanced proxy selector；
- 无标签 edge-contrast selector；
- 无标签 effective-rank / low-anisotropy selector；
- 新增 feature-adaptive selector；
- validation selector 与 oracle selector 只作为诊断上界，不作为无标签方法。

运行命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python select_sparc_mode_proxy.py \
  --runs-dir runs/spgcl_official_embeddings_seed42_e100/artifacts \
  --include-methods spgcl_official \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_mode_proxy_seed42.csv \
  --aggregate-out runs/summaries/sparc_mode_proxy_seed42_aggregate.csv
```

`feature-adaptive` 的当前规则：

- 先计算原始特征 `data.x` 的 edge-random contrast；
- 若 `feature_edge_random_contrast < 0`，说明边上原始特征比随机节点对更不相似，图呈现 feature-disassortative 状态，选择 effective-rank / residual route；
- 否则选择 edge-contrast / propagation route。

项目内 official SP-GCL seed42 artifacts 上的早筛结果：

| Dataset | Feature edge-random contrast | Selector | Selected mode | F1Mi | F1Ma |
| --- | ---: | --- | --- | ---: | ---: |
| Chameleon | +0.008638 | `selected_feature_adaptive` | `ssl_prop2` | 0.640570 | 0.641855 |
| Chameleon | +0.008638 | `selected_ssl` | `ssl` | 0.631798 | 0.632267 |
| Chameleon | +0.008638 | `selected_oracle` | mixed | 0.652412 | 0.652424 |
| Squirrel | -0.001591 | `selected_feature_adaptive` | `ssl_resid1` | 0.488953 | 0.485416 |
| Squirrel | -0.001591 | `selected_ssl` | `ssl` | 0.450817 | 0.445798 |
| Squirrel | -0.001591 | `selected_oracle` | mostly `ssl_resid1` | 0.491162 | 0.487506 |

对照结果：

- balanced proxy v1 固定选择 `ssl_prop1`，在两个数据集上都超过 `ssl`，但低于 feature-adaptive；因此不作为当前主 selector。
- edge-contrast selector 固定选择 `ssl_prop2`，在 Chameleon 最强，在 Squirrel 低于 `ssl_resid1`。
- effective-rank / low-anisotropy selector 固定选择 `ssl_resid1`，在 Squirrel 最强，在 Chameleon 接近但略低于 `ssl_prop2`。
- validation selector 在当前设置中反而低于 feature-adaptive，说明简单用少量 val label 选 mode 不一定稳定。

当前方法化假设更新：

> Feature-disassortativity Adaptive SPARC: 当原始特征在图边上呈现正 edge-random contrast 时，传播分支 `P^k z` 更可能补充平滑邻域信息；当原始特征 edge-random contrast 为负时，残差分支 `z - Pz` 更可能保留被传播抹平的判别信号。

保守裁决：

- 该 selector 是 SPARC 从 post-hoc analysis 走向 label-free method 的第一个正向证据。
- 目前只覆盖 Chameleon/Squirrel 与一个 official SP-GCL seed42 embedding；不能宣称最终 SOTA。
- 下一步必须补不同 SP-GCL seed / epoch artifacts，并尝试把该 gate 扩展到更多 heterophily 数据集或至少补足同配图 safety。

## Official seed7 复核与 selector 裁决修正

为检查 seed42 是否偶然，已重新导出 official SP-GCL seed7 embedding：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Chameleon Squirrel" \
OUT_DIR="runs/spgcl_official_embeddings_seed7_e100" \
RESET_EPOCHS=100 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=256 \
RESET_SEED_NUM=32 \
RESET_MAX_SIZE=512 \
RESET_SUBG_NUM_HOPS=2 \
SEED=7 \
bash scripts/run_spgcl_embedding_export.sh
```

随后复跑 C-grid：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/spgcl_official_embeddings_seed7_e100/artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_prop1 ssl_prop2 ssl_resid1 ssl_resid2 \
  --max-hop 2 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_seed7_cgrid.csv \
  --aggregate-out runs/summaries/sparc_seed7_cgrid_aggregate.csv
```

结果：

| Seed | Dataset | `ssl` F1Mi/F1Ma | `ssl_prop2` F1Mi/F1Ma | `ssl_resid1` F1Mi/F1Ma | Best |
| --- | --- | ---: | ---: | ---: | --- |
| seed42 | Chameleon | 0.631798 / 0.632267 | 0.640570 / 0.641855 | 0.639254 / 0.639301 | `ssl_prop2` |
| seed42 | Squirrel | 0.450817 / 0.445798 | 0.480307 / 0.476469 | 0.488953 / 0.485416 | `ssl_resid1` |
| seed7 | Chameleon | 0.631360 / 0.631923 | 0.641009 / 0.641697 | 0.645175 / 0.645380 | `ssl_resid1` |
| seed7 | Squirrel | 0.448991 / 0.443961 | 0.476753 / 0.473270 | 0.483093 / 0.479099 | `ssl_resid1` |

跨 seed 的 selector 汇总：

| Selector | Rule | Mean F1Mi | Mean F1Ma | Mean delta vs `ssl` |
| --- | --- | ---: | ---: | ---: |
| `ssl` | 不校准 | 0.540741 | 0.538488 | 0.000000 |
| `prop2` | 固定 `ssl_prop2` | 0.559660 | 0.558323 | +0.018918 |
| `feature_adaptive_v1` | Chameleon 选 `ssl_prop2`，Squirrel 选 `ssl_resid1` | 0.563406 | 0.562017 | +0.022665 |
| `resid1` | 固定 `ssl_resid1` / effective-rank route | 0.564119 | 0.562299 | +0.023378 |

裁决修正：

- seed7 继续支持 SPARC：`ssl_prop2` 与 `ssl_resid1` 在 Chameleon/Squirrel 上均高于 `ssl`。
- feature-adaptive v1 仍然稳定正向，但不是最强 selector；它在 seed7 Chameleon 上选 `ssl_prop2`，而 `ssl_resid1` 实际更强。
- 当前主线应从 “feature-disassortativity adaptive selector” 收缩为 “propagation-residual calibration, with residual/effective-rank route as the most robust default”。
- feature contrast 可以保留为解释变量和未来 gate 信号，但不应作为当前主 selector claim。
- 更强的下一步不是继续手调 selector，而是把 `ssl_resid1` 形式方法化为训练时 residual calibration branch，并补同配图 safety 与更多 official seed。

运行策略记录：

- 本轮曾启动 `select_sparc_mode_proxy.py` 的完整 seed7 selector 评估，但它与 C-grid 重复训练同一批 linear probes，耗时过长，已中断。
- seed7 selector 所需的核心结论可由 C-grid 直接得到；后续应重构 selector 脚本，使其可读取已有 C-grid candidate scores，避免重复 fit logistic regression。

已新增轻量汇总脚本：

- `experiments/grace_idea/summarize_sparc_selectors.py`

该脚本读取已有 C-grid aggregate CSV，不重新训练 linear probe，直接生成 selector 对照表：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python summarize_sparc_selectors.py \
  --cgrid runs/summaries/sparc_seed42_cgrid_aggregate.csv \
  --seed-label seed42 \
  --cgrid runs/summaries/sparc_seed7_cgrid_aggregate.csv \
  --seed-label seed7 \
  --out runs/summaries/sparc_selector_seed42_seed7.csv \
  --aggregate-out runs/summaries/sparc_selector_seed42_seed7_aggregate.csv
```

输出文件：

- `runs/summaries/sparc_selector_seed42_seed7.csv`
- `runs/summaries/sparc_selector_seed42_seed7_aggregate.csv`

该脚本当前支持的 selector：

- `ssl`: 不做 SPARC 校准；
- `prop2`: 固定 `ssl_prop2`；
- `resid1`: 固定 `ssl_resid1`；
- `feature_adaptive_v1`: Chameleon 选 `ssl_prop2`，Squirrel 选 `ssl_resid1`。

当前正式 cross-seed selector 表以 `summarize_sparc_selectors.py` 输出为准；后续新增 seed 时应先生成各 seed C-grid aggregate，再用该脚本合并，避免重复训练探针。

## Official seed0 最小复核

为检查 `ssl_resid1` 路线是否只是 seed42/seed7 的偶然现象，已补跑 official SP-GCL seed0 embedding：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Chameleon Squirrel" \
OUT_DIR="runs/spgcl_official_embeddings_seed0_e100" \
RESET_EPOCHS=100 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=256 \
RESET_SEED_NUM=32 \
RESET_MAX_SIZE=512 \
RESET_SUBG_NUM_HOPS=2 \
SEED=0 \
bash scripts/run_spgcl_embedding_export.sh
```

随后只评估最小硬门槛 `ssl` vs `ssl_resid1`：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/spgcl_official_embeddings_seed0_e100/artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_resid1 \
  --max-hop 1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_seed0_resid1_cgrid.csv \
  --aggregate-out runs/summaries/sparc_seed0_resid1_cgrid_aggregate.csv
```

结果：

| Seed | Dataset | `ssl` F1Mi/F1Ma | `ssl_resid1` F1Mi/F1Ma | Delta F1Mi/F1Ma | Split sign |
| --- | --- | ---: | ---: | ---: | --- |
| seed0 | Chameleon | 0.632456 / 0.632515 | 0.636623 / 0.636698 | +0.004167 / +0.004183 | 5/10 正、5/10 负 |
| seed0 | Squirrel | 0.446013 / 0.440950 | 0.479443 / 0.475803 | +0.033429 / +0.034853 | 10/10 正 |

三 seed 合并后，完整可比的 selector 是 `ssl` 与固定 `ssl_resid1`：

```bash
python summarize_sparc_selectors.py \
  --cgrid runs/summaries/sparc_seed42_cgrid_aggregate.csv \
  --seed-label seed42 \
  --cgrid runs/summaries/sparc_seed7_cgrid_aggregate.csv \
  --seed-label seed7 \
  --cgrid runs/summaries/sparc_seed0_resid1_cgrid_aggregate.csv \
  --seed-label seed0 \
  --out runs/summaries/sparc_selector_seed42_seed7_seed0.csv \
  --aggregate-out runs/summaries/sparc_selector_seed42_seed7_seed0_aggregate.csv
```

完整三 seed 结果：

| Selector | Rows | Mean F1Mi | Mean F1Ma | Mean delta F1Mi | Mean delta F1Ma |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ssl` | 6 | 0.540239 | 0.537903 | 0.000000 | 0.000000 |
| `resid1` | 6 | 0.562090 | 0.560283 | +0.021851 | +0.022380 |

按数据集看：

| Dataset | `ssl_resid1` mean F1Mi/F1Ma | Mean delta F1Mi/F1Ma | Seed-level micro deltas |
| --- | ---: | ---: | --- |
| Chameleon | 0.640351 / 0.640460 | +0.008480 / +0.008224 | +0.004167, +0.007456, +0.013816 |
| Squirrel | 0.483830 / 0.480106 | +0.035223 / +0.036536 | +0.033429, +0.038136, +0.034102 |

注意：seed0 本轮只跑了 `ssl` 与 `ssl_resid1`，因此三 seed selector aggregate 中的 `prop2` 与 `feature_adaptive_v1` 行不是完整可比证据；它们只能继续作为 seed42/seed7 的辅助观察。当前完整证据支持把固定 `ssl_resid1` / effective-rank route 作为 SPARC 的最稳默认路线。

## Official seed1 复核与方法化 artifact

为继续排除 official seed 偶然性，已补跑 seed1：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Chameleon Squirrel" \
OUT_DIR="runs/spgcl_official_embeddings_seed1_e100" \
RESET_EPOCHS=100 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=256 \
RESET_SEED_NUM=32 \
RESET_MAX_SIZE=512 \
RESET_SUBG_NUM_HOPS=2 \
SEED=1 \
bash scripts/run_spgcl_embedding_export.sh
```

最小 C-grid 复核：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/spgcl_official_embeddings_seed1_e100/artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_resid1 \
  --max-hop 1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_seed1_resid1_cgrid.csv \
  --aggregate-out runs/summaries/sparc_seed1_resid1_cgrid_aggregate.csv
```

seed1 结果：

| Seed | Dataset | `ssl` F1Mi/F1Ma | `ssl_resid1` F1Mi/F1Ma | Delta F1Mi/F1Ma | Split sign |
| --- | --- | ---: | ---: | ---: | --- |
| seed1 | Chameleon | 0.628289 / 0.628816 | 0.639474 / 0.639231 | +0.011184 / +0.010416 | 7/10 正、2/10 负 |
| seed1 | Squirrel | 0.454755 / 0.448902 | 0.482037 / 0.478395 | +0.027281 / +0.029493 | 10/10 正 |

四个 official seeds（seed42/seed7/seed0/seed1）中，完整可比的 `ssl_resid1` 结果为：

| Dataset | `ssl_resid1` mean F1Mi/F1Ma | Mean delta F1Mi/F1Ma | Seed-level micro deltas |
| --- | ---: | ---: | --- |
| Chameleon | 0.640132 / 0.640153 | +0.009156 / +0.008772 | seed42 +0.007456; seed7 +0.013816; seed0 +0.004167; seed1 +0.011184 |
| Squirrel | 0.483381 / 0.479678 | +0.033237 / +0.034775 | seed42 +0.038136; seed7 +0.034102; seed0 +0.033429; seed1 +0.027281 |
| Overall | 0.561756 / 0.559915 | +0.021196 / +0.021774 | 8 dataset-seed rows |

方法化入口已新增：

- `experiments/grace_idea/build_sparc_artifacts.py`

该脚本把 strong SSL embedding artifact 显式转换为 SPARC artifact，例如把 official SP-GCL 的 `z` 转为 `spgcl_official_sparc_resid1` 的 `[normalize(z), normalize(z-Pz)]` 表示：

```bash
python build_sparc_artifacts.py \
  --runs-dir runs/spgcl_official_embeddings_seed42_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed7_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed0_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed1_e100/artifacts \
  --out-dir runs/sparc_resid1_artifacts_seed42_seed7_seed0_seed1 \
  --mode ssl_resid1 \
  --method-suffix sparc_resid1 \
  --overwrite
```

生成的 artifacts 可被现有评估脚本作为独立 method 读取：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/sparc_resid1_artifacts_seed42_seed7_seed0_seed1 \
  --include-methods spgcl_official_sparc_resid1 \
  --modes ssl \
  --max-hop 1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_resid1_artifact_method_seed42_seed7_seed0_seed1.csv \
  --aggregate-out runs/summaries/sparc_resid1_artifact_method_seed42_seed7_seed0_seed1_aggregate.csv
```

artifact method 结果与四 seed residual summary 对齐：

| Method | Dataset | Runs | F1Mi/F1Ma |
| --- | --- | ---: | ---: |
| `spgcl_official_sparc_resid1` | Chameleon | 40 | 0.640132 / 0.640153 |
| `spgcl_official_sparc_resid1` | Squirrel | 40 | 0.483381 / 0.479678 |

这一步的意义是把 SPARC 从 “evaluation mode” 固化为可比较的 artifact-level method。下一步仍需推进到训练时 residual projector / calibration objective，否则论文贡献会被质疑为 post-hoc linear-probe trick。

## Propagation branch 对照与训练时原型早筛

已补 `ssl_prop2` 的 artifact-level method 对照：

```bash
python build_sparc_artifacts.py \
  --runs-dir runs/spgcl_official_embeddings_seed42_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed7_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed0_e100/artifacts \
  --runs-dir runs/spgcl_official_embeddings_seed1_e100/artifacts \
  --out-dir runs/sparc_prop2_artifacts_seed42_seed7_seed0_seed1 \
  --mode ssl_prop2 \
  --method-suffix sparc_prop2 \
  --overwrite
```

并使用同一套 C-grid 评估：

```bash
python evaluate_propagation_calibration.py \
  --runs-dir runs/sparc_prop2_artifacts_seed42_seed7_seed0_seed1 \
  --include-methods spgcl_official_sparc_prop2 \
  --modes ssl \
  --max-hop 1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/sparc_prop2_artifact_method_seed42_seed7_seed0_seed1.csv \
  --aggregate-out runs/summaries/sparc_prop2_artifact_method_seed42_seed7_seed0_seed1_aggregate.csv
```

四 seed artifact-level 对照：

| Dataset | `sparc_resid1` F1Mi/F1Ma | `sparc_prop2` F1Mi/F1Ma | `resid1 - prop2` |
| --- | ---: | ---: | ---: |
| Chameleon | 0.640132 / 0.640153 | 0.639254 / 0.640057 | +0.000877 / +0.000095 |
| Squirrel | 0.483381 / 0.479678 | 0.478410 / 0.474747 | +0.004971 / +0.004931 |

结论：`ssl_prop2` 仍应保留为 ablation / optional branch，但当前主路线继续选择 `ssl_resid1`。Squirrel 上 residual branch 明显更强；Chameleon 上两者接近，说明后续如果做 node-level gate，主要收益空间不在全局 mode selector，而在避免 high-local-homophily 区域被 residual branch 伤害。

已实现训练时最小原型：

- `train.py --method sparc_gcl`
- 主损失：GRACE InfoNCE；
- 附加损失：view-specific residual `z-Pz` 的 auxiliary InfoNCE；
- 默认评估表示：`[normalize(z), normalize(z-Pz)]`；
- 关键参数：`--sparc-residual-weight`、`--sparc-eval-mode hidden|resid|hidden_resid`。

smoke 已通过：

```bash
python train.py --dataset Cora --method sparc_gcl --epochs 2 --skip-eval \
  --save-dir runs/sparc_gcl_smoke --overwrite --log-every 1 \
  --sparc-residual-weight 0.1

python train.py --dataset Chameleon --method sparc_gcl --epochs 2 \
  --eval-mode mask --split-index 0 --save-dir runs/sparc_gcl_smoke \
  --overwrite --log-every 1 --sparc-residual-weight 0.1
```

但 50 epoch early gate 失败：

| Method | Dataset | Split/Seed | F1Mi/F1Ma |
| --- | --- | --- | ---: |
| `sparc_gcl` | Chameleon | split0 seed0 | 0.4276 / 0.4282 |
| `sparc_gcl` | Squirrel | split0 seed0 | 0.3333 / 0.3077 |

这些结果远低于 official SP-GCL embedding 与 SPARC artifact-level method。因此当前训练时 `sparc_gcl` 只能作为工程原型和后续消融入口，不能作为主方法结果。主方法应继续围绕 strong SP-GCL embedding 的 residual calibration，下一步若要摆脱 post-hoc 质疑，应修改/扩展 official SP-GCL 训练流程本身，而不是在当前 GRACE scaffold 上继续小调 residual loss。

## Official SP-GCL 内部 residual branch

为避免 SPARC 被质疑为 post-hoc linear-probe trick，已新增可复现 patcher/runner，把 residual branch 注入 official SP-GCL 训练流程：

- `experiments/grace_idea/patch_spgcl_sparc.py`
- `experiments/grace_idea/scripts/run_spgcl_sparc_residual_export.sh`

patch 内容：

- 在 `SPGCL` 中新增 residual embedding：`r = z - Pz`；
- 新增 residual auxiliary projector；
- 在 official SP-GCL 的 sample top-k positive / negative loss 上，对 residual projection 复用同一批 positive/negative index，加入 `sparc_residual_weight * residual_loss`；
- 新增 `--sparc_residual_weight` 与 `--sparc_embed_mode hidden|resid|hidden_resid`；
- runner 默认把 artifact method 标记为 `spgcl_sparc_residual`，避免和 `spgcl_official` 混淆。

验证命令：

```bash
python patch_spgcl_sparc.py --spgcl-root ../../third_party_baselines/SPGCL

DATASETS="Chameleon" \
OUT_DIR="runs/spgcl_sparc_residual_smoke_v2" \
RESET_EPOCHS=1 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=64 \
RESET_SEED_NUM=4 \
RESET_MAX_SIZE=64 \
RESET_SUBG_NUM_HOPS=2 \
SEED=0 \
SPARC_RESIDUAL_WEIGHT=0.1 \
SPARC_EMBED_MODE=hidden_resid \
bash scripts/run_spgcl_sparc_residual_export.sh
```

smoke 产物：

- `runs/spgcl_sparc_residual_smoke_v2/artifacts/Chameleon_spgcl_sparc_residual_seed0_split0/artifacts.pt`
- embedding shape 为 `(2277, 128)`，对应 hidden 64 的 `hidden_resid` 双分支输出；
- metadata 中 `method=spgcl_sparc_residual`、`sparc_embed_mode=hidden_resid`、`sparc_residual_weight=0.1`。

进一步执行 seed0 / 100 epoch early gate：

```bash
DATASETS="Chameleon Squirrel" \
OUT_DIR="runs/spgcl_sparc_residual_seed0_e100" \
RESET_EPOCHS=100 \
LINEAR_EPOCHS=10 \
RESET_HIDDEN=256 \
RESET_SEED_NUM=32 \
RESET_MAX_SIZE=512 \
RESET_SUBG_NUM_HOPS=2 \
SEED=0 \
SPARC_RESIDUAL_WEIGHT=0.1 \
SPARC_EMBED_MODE=hidden_resid \
bash scripts/run_spgcl_sparc_residual_export.sh

python evaluate_propagation_calibration.py \
  --runs-dir runs/spgcl_sparc_residual_seed0_e100/artifacts \
  --include-methods spgcl_sparc_residual \
  --modes ssl \
  --max-hop 1 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 4 16 64 \
  --max-iter 500 \
  --out runs/summaries/spgcl_sparc_residual_seed0_e100_cgrid.csv \
  --aggregate-out runs/summaries/spgcl_sparc_residual_seed0_e100_cgrid_aggregate.csv
```

seed0 C-grid 结果：

| Dataset | Patched `spgcl_sparc_residual` | Post-hoc `spgcl_official_sparc_resid1` | Patched - post-hoc |
| --- | ---: | ---: | ---: |
| Chameleon | 0.661404 / 0.662057 | 0.636623 / 0.636698 | +0.024781 / +0.025359 |
| Squirrel | 0.484534 / 0.479518 | 0.479443 / 0.475803 | +0.005091 / +0.003715 |

当前解释：

- 这是 SPARC 从 post-hoc calibration 向训练时方法推进的第一条强正向证据；
- Chameleon 的提升尤其重要，因为此前 residual post-hoc 在 Chameleon 上只是小幅正均值且 split-level 不稳；
- 但该结果目前只有 seed0，不能据此声称 SOTA 或稳定改进；
- 下一步硬门槛是补 seed1/seed7/seed42 patched official SP-GCL，并与原 official SP-GCL、post-hoc residual、prop2 artifact 做同 C-grid 对齐。

## 研究假设

SP-GCL 已经通过局部子图相似性学到强 embedding，但这个 embedding 仍混合了两类信息：

- 局部传播后保持稳定的 class-relevant signal；
- 传播后被改变或抹平的 heterophily-sensitive residual signal。

在 Squirrel 这类更强异配/复杂局部结构数据上，`z - Pz` 能恢复被平滑传播压掉的判别成分。因此 `[z, z - Pz]` 比单独 `z` 更适合下游分类。

## 下一步硬门槛

1. 复现级评估：
   - 保存 official SP-GCL embeddings 的可复现实验入口，不依赖 `/tmp` 临时路径；
   - 跑 Chameleon/Squirrel 多 seed 或多 official seed embedding；
   - 使用完整 C 网格确认 fast gate 不是 C 选择假象。

2. 机制诊断：
   - 第一轮 class、degree、local homophily bucket 诊断已完成；
   - 继续比较 `ssl_prop1/2/3`、`ssl_resid1/2/3` 的更大 mode 空间；
   - 证明 raw concat 不等价于 propagation residual calibration。

3. 方法化：
   - `sparc_gcl` 在当前 GRACE scaffold 上的训练时原型 early gate 失败，不应继续小调；
   - official SP-GCL 内部 residual branch seed0 early gate 已正向，下一步应补多 seed 复核；
   - 若继续做 selection/gate，应优先做 node-level local-homophily/degree proxy，避免 Chameleon high-local-homophily 区域被 residual branch 伤害。

4. 强基线：
   - 以 official SP-GCL 为 baseline；
   - 后续再补 PolyGCL/HLCL 表格或官方结果对齐。

## 当前裁决

SPARC-GCL 是当前最值得继续的 active candidate。

继续理由：

- 直接建立在已验证强 baseline SP-GCL 上；
- 在 Squirrel 的 four official seeds × 10 split 上有稳定正向信号；
- 在 Chameleon 的 four official seeds 上均为正均值，但 split-level 稳定性弱于 Squirrel；
- 已有 artifact-level method 入口 `spgcl_official_sparc_resid1`，不再只依赖临时 evaluation mode；
- 已有 official SP-GCL 内部 residual branch patch/runner，seed0 early gate 强于 post-hoc residual；
- 创新点更清楚：不是再做 augmentation，也不是简单 high/low-pass filter，而是 strong GCL embedding 的 propagation residual calibration；
- 算力友好，适合 RTX 3060。

主要风险：

- 当前只是 post-hoc representation calibration；
- 当前 GRACE scaffold 上的训练时 `sparc_gcl` 原型早筛失败，不能作为主结果；
- official SP-GCL 内部 residual branch 目前只有 seed0 正向证据，需要多 seed 复核；
- Chameleon 增益小且 best mode 可能不同于 Squirrel；
- 需要证明不是 validation/C-grid 偶然；
- 需要扩展到更多数据集和更多 seed。
