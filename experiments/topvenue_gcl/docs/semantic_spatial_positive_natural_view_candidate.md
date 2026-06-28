# Semantic-Spatial Positive Natural-View Candidate

日期：2026-06-28

## 当前裁决

SSPNV-GCL 是当前 active candidate，但还不是可宣称 SOTA 的最终方法。

第一轮 10 split / seed0 / 50 epoch 结果显示，它稳定超过 `gcn_mlp_gcl` strong foundation，尤其在 Chameleon 与 Squirrel 上明显好于 DANV/FDNV 早期版本。下一步应围绕 SSPNV 做 ablation、strong baseline 与 homophily safety，而不是继续随机发明新模块。

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

## 风险与放弃条件

当前不能宣称 SOTA，原因：

- 只跑了 seed0 的 10 个 split，还未覆盖多个 model seed；
- 还没有 semantic-only、spatial-only、bootstrap-only、random semantic positive、random spatial positive 消融；
- 还没有与 S3GCL / SP-GCL / GraphECL / PolyGCL 等强基线同协议对齐；
- Actor 增益过小且 4/10 split 为负；
- Squirrel 有少数 macro 负 split，需要解释是否偏向多数类。

下一轮停止条件：

- random semantic positives 接近完整 SSPNV；
- semantic-only 或 spatial-only 与完整方法几乎相同，说明双正样本拆分不是必要机制；
- homophily 数据集明显退化；
- Chameleon/Squirrel 在多 seed 或强基线对齐后失去主要增益。

## 下一步

优先做 ablation，而不是继续扩大模型复杂度：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
DATASETS="Texas Chameleon Squirrel Actor" \
METHODS="gcn_mlp_gcl sspnv_gcl" \
SPLITS="0 1 2 3 4 5 6 7 8 9" \
SEEDS="1 2" \
EPOCHS=50 \
RUNS_DIR="runs/split_study_sspnv_s1-2_splits0-9_e50" \
OVERWRITE=1 \
bash scripts/run_split_study.sh
```

随后实现并运行 semantic-only、spatial-only、random semantic positive、random spatial positive、homophily non-degradation 与强基线同协议对齐。
