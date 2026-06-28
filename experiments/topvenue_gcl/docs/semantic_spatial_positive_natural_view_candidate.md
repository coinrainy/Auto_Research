# Semantic-Spatial Positive Natural-View Candidate

日期：2026-06-28

## 当前裁决

SSPNV-GCL 的固定完整版本不再作为最终主方法包装，但它保留为当前最有价值的机制原型与 ablation 基线。

第一轮 10 split / seed0 / 50 epoch 结果显示，完整 SSPNV 稳定超过 `gcn_mlp_gcl` strong foundation，尤其在 Chameleon 与 Squirrel 上明显好于 DANV/FDNV 早期版本。但后续 control 表明：Chameleon 上 random semantic positives 与 semantic-only/spatial-only 均可取得接近甚至超过完整 SSPNV 的表现，因此“semantic-spatial 双正样本拆分本身就是必要机制”的主张被削弱。Squirrel 上 random semantic / random spatial 明显弱于完整 SSPNV，说明结构化 positives 仍有条件性价值。

因此当前裁决是：放弃“固定双分支 SSPNV = 终稿 SOTA idea”的包装，转向更保守的问题定义：**filter-specific positives 的价值具有数据/局部结构条件性，下一代方法必须学习何时使用 semantic、spatial 或 bootstrap-only 目标，而不是固定同权相加。**

## 方法定义

名称：Semantic-Spatial Positive Natural-View GCL  
入口：`--method sspnv_gcl`

核心假设：

- GCN-MLP Natural-View foundation 已经证明 ego/raw-feature view 与 graph/message-passing view 是异配图上更稳的双视图底座；
- 失败的 FDNV 说明不能简单让 ego view 追逐 routed high/low target；
- 更合理的做法是把 positive relation 也 filter-specific 化：语义相似正样本主要监督 high-pass target，空间邻接正样本主要监督 low-pass target；
- 保留 GCN-MLP bootstrap，避免纯 sampled InfoNCE 把训练变成不稳定的 hard-mining trick。

当前实现：

- semantic positives：用 raw propagation signature 做 KNN，默认 `sspnv_semantic_topk=5`；
- spatial positives：每个节点取一个确定性一跳 incident neighbor，孤立节点回退到 self；
- semantic loss：`pred_ego` 对比 `high[semantic_positive]`；
- spatial loss：`pred_ego` 对比 `low[spatial_positive]`；
- bootstrap loss：保留 `pred_ego -> graph` 与 `pred_high -> ego` 的 Natural-View alignment；
- 默认权重：`sspnv_bootstrap_weight=1.0`、`sspnv_semantic_weight=0.1`、`sspnv_spatial_weight=0.1`。

## Early Gate 结果

命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Texas Actor Chameleon Squirrel" \
METHODS="gcn_mlp_gcl sspnv_gcl" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
SEEDS="0" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_sspnv_s0_splits0-2_e50" \
bash scripts/run_split_study.sh

python summarize_split_study.py \
  --runs-dir runs/split_study_sspnv_s0_splits0-2_e50 \
  --baseline-method gcn_mlp_gcl \
  --out runs/split_study_sspnv_s0_splits0-2_e50/runs_vs_gcn_mlp.csv \
  --aggregate-out runs/split_study_sspnv_s0_splits0-2_e50/aggregate_vs_gcn_mlp.csv
```

Aggregate vs GCN-MLP：

| Dataset | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | ---: | ---: | --- | --- |
| Texas | +0.032432 | +0.069760 | 6/2 | 均值强正，macro 尤其明显；有 2 个负 split |
| Actor | +0.000658 | +0.004324 | 6/4 | 弱正且不稳定，只能作为边界数据集 |
| Chameleon | +0.031140 | +0.032431 | 10/0 | 当前最强证据 |
| Squirrel | +0.011720 | +0.009843 | 9/1 | 稳定小中幅正向，但 macro 有波动 |

## Control 与 AFPNV 结果

新增 control 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
RUNS_DIR="runs/sspnv_controls_wiki_s0_splits0-9_e50"

DATASETS="Chameleon Squirrel" METHODS="gcn_mlp_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="sspnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="full" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="sspnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="semantic_only" EXTRA_ARGS="--sspnv-spatial-weight 0.0" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="sspnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="spatial_only" EXTRA_ARGS="--sspnv-semantic-weight 0.0" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="sspnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="random_semantic" EXTRA_ARGS="--sspnv-random-semantic" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="sspnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="random_spatial" EXTRA_ARGS="--sspnv-random-spatial" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Chameleon Squirrel" METHODS="afpnv_gcl" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" OVERWRITE=1 bash scripts/run_split_study.sh
```

Aggregate vs GCN-MLP：

| Dataset | Variant | ΔF1Mi | ΔF1Ma | Positive/Negative F1Mi | 裁决 |
| --- | --- | ---: | ---: | --- | --- |
| Chameleon | full SSPNV | +0.027412 | +0.029084 | 9/1 | 正向，但不是最强 |
| Chameleon | semantic-only | +0.037281 | +0.038611 | 9/0 | 当前 Chameleon 最强，提示 semantic/high-pass auxiliary 足够解释大部分收益 |
| Chameleon | spatial-only | +0.033553 | +0.036636 | 9/1 | 也强，说明收益不专属于 semantic KNN |
| Chameleon | random semantic | +0.029825 | +0.029735 | 10/0 | major warning：随机 semantic 也强，削弱结构化 semantic 必要性 |
| Chameleon | random spatial | +0.023465 | +0.025592 | 9/1 | 仍正向，说明 Chameleon 对正样本来源不敏感 |
| Chameleon | AFPNV | +0.025000 | +0.024495 | 8/1 | 可运行但弱于 full/semantic-only，不升级 |
| Squirrel | full SSPNV | +0.007397 | +0.003500 | 6/4 | 当前 Squirrel 最强，但稳定性一般 |
| Squirrel | semantic-only | +0.004803 | +0.000484 | 7/3 | 接近 full，但 macro 较弱 |
| Squirrel | spatial-only | +0.000480 | -0.000961 | 5/4 | 基本无效 |
| Squirrel | random semantic | -0.004131 | -0.009298 | 4/6 | 明确失败，说明 Squirrel 需要结构化 semantic positives |
| Squirrel | random spatial | -0.000096 | -0.003287 | 6/4 | 基本无效 |
| Squirrel | AFPNV | +0.004995 | -0.000219 | 6/4 | 未超过 full SSPNV，不升级 |

AFPNV-GCL 已实现，入口为 `--method afpnv_gcl`。它在 SSPNV 上加入 raw propagation signature 正样本置信度加权，避免固定同权使用 semantic/spatial loss。但本轮 Chameleon/Squirrel 10 split 结果没有超过更简单的 SSPNV 消融，因此当前只保留为可复现 ablation，不作为 active main idea。

## 风险与放弃条件

当前不能宣称 SOTA，原因：

- 只跑了 seed0 的 10 个 split，还未覆盖多个 model seed；
- random semantic positive 在 Chameleon 上接近/超过完整 SSPNV，结构化 semantic positive 必要性不足；
- semantic-only / spatial-only 在 Chameleon 上超过完整 SSPNV，固定双分支同权相加不是必要机制；
- AFPNV 置信度加权没有超过完整 SSPNV，当前 adaptive weighting 第一版未成功；
- 还没有与 S3GCL / SP-GCL / GraphECL / PolyGCL 等强基线同协议对齐；
- Actor 增益过小且 4/10 split 为负；
- Squirrel 有少数 macro 负 split，需要解释是否偏向多数类。

下一轮停止条件：

- 若下一代自适应方法仍不能同时超过 Chameleon semantic-only 与 Squirrel full SSPNV，应放弃 SSPNV 家族主线；
- homophily 数据集明显退化；
- Chameleon/Squirrel 在多 seed 或强基线对齐后失去主要增益。

## 下一步

优先做下一代自适应选择，而不是继续包装固定 SSPNV：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Chameleon Squirrel" \
METHODS="gcn_mlp_gcl sspnv_gcl afpnv_gcl" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
SEEDS="1" \
EPOCHS=50 \
RUNS_DIR="runs/afpnv_s1_splits0-9_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

若 AFPNV 在 seed1 仍弱于 full SSPNV 或 semantic-only，应停止当前置信度加权路线，转向更明确的 branch selection / mixture-of-objectives，而不是继续调阈值。
