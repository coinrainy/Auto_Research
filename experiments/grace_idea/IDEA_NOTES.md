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
- `--method sgfn`：Stability-Guided False-Negative attenuation，当前主候选。

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
- 当前方向应转为 context-gated false-negative calibration；若门控版仍不能避免 Wisconsin/Actor 退化，则放弃 false-negative attenuation 主线，重新构思 GCL idea。

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
- `summarize_runs.py` 可从 `runs/` 目录生成 matched paired summary 与 dataset aggregate summary，并兼容 `es_weighted` / `sgfn` 的 normal、shuffled、uniform_random 控制组。
- `analyze_pair_weights.py` 可对 `sgfn` run 做 label-only false-negative pressure 诊断；该诊断不参与训练，只用于机制验证。

示例 split-aware 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell" SPLITS="0 1 2" SEEDS="0" METHODS="grace sgfn" ES_CONTROLS="normal shuffled" SAVE_DIR="runs/sgfn_split_control_sanity" scripts/run_split_study.sh
python summarize_runs.py --runs-dir runs/sgfn_split_control_sanity --target-method sgfn --paired-out runs/summaries/sgfn_split_control_sanity_paired.csv --aggregate-out runs/summaries/sgfn_split_control_sanity_aggregate.csv
python analyze_pair_weights.py --runs-dir runs/sgfn_split_control_sanity --out runs/summaries/sgfn_split_control_sanity_pair_weights.csv --aggregate-out runs/summaries/sgfn_split_control_sanity_pair_weights_aggregate.csv --control-paired-out runs/summaries/sgfn_split_control_sanity_pair_weights_controls.csv
```

近期需要补齐：

- context-gated false-negative calibration 的最小实现；
- gate 与 downstream error、degree、local structure 的独立诊断；
- 若 gate 版通过 Texas/Cornell/Wisconsin/Actor 的 3-split 筛选，再接 Chameleon/Squirrel 和 ProGCL / GRAPE / GraphRank 对照。
