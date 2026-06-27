# GRACE Idea 工作副本说明

本目录由 `baselines/GRACE` 复制而来，用于承载当前研究 idea 的代码改动。

- 来源 baseline：`baselines/GRACE`
- 来源版本：`b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- 协作约束：不要直接修改 `baselines/GRACE`；所有方法改动写在本目录。

当前路线调整：

- 删除此前自建的 RW-GCL scaffold 与批跑脚本。
- 以后以 GRACE 原实现为基础做最小侵入式修改。
- 优先保持 baseline 逻辑可对照，再逐步加入 reliability / weighting / diagnostics。

## 当前已实现的第一版改进

训练入口仍为 `train.py`，但新增 `--method` 参数：

- `--method grace`：保留原始 GRACE 训练路径。
- `--method es_weighted`：新增 embedding-stability weighted GRACE。

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

正式实验前仍需补齐：

- reliability 与 downstream error、degree、local homophily 的独立诊断；
- negative weighting 的 false-negative pressure 诊断。

## 当前实验入口能力

- 支持 Planetoid/CitationFull：`Cora`、`CiteSeer`、`PubMed`、`DBLP`。
- 支持异配数据集：`Texas`、`Cornell`、`Wisconsin`、`Actor`。
- 支持 `--split-index` 选择 PyG 提供的二维 split mask。
- `--eval-mode auto` 对异配数据集默认使用固定 `train/val/test` mask；对原 GRACE 数据集保持随机 linear probe。
- `--eval-mode mask` 可强制使用 mask 评估。
- `--eval-mode random` 可强制使用原 GRACE 风格随机 linear probe。
- `scripts/run_split_study.sh` 可通过 `DATASETS`、`SPLITS`、`SEEDS`、`METHODS`、`ES_CONTROLS` 做 split-aware 批跑。
- `train_log.csv` 记录权重均值、方差、min/max 与 effective sample size ratio，用于判断 reliability 权重是否实质上接近等权。
- `summarize_runs.py` 可从 `runs/` 目录生成 matched paired summary 与 dataset aggregate summary，并兼容 `es_weighted_shuffled` / `es_weighted_uniform_random` 控制组；旧的 `es_weighted_random` 历史目录仍可解析。

示例 split-aware 命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell" SPLITS="0 1 2" SEEDS="0" ES_CONTROLS="normal shuffled uniform_random" SAVE_DIR="runs/split_control_sanity" scripts/run_split_study.sh
python summarize_runs.py --runs-dir runs/split_control_sanity --paired-out runs/summaries/split_control_sanity_paired.csv --aggregate-out runs/summaries/split_control_sanity_aggregate.csv
```

近期需要补齐：

- reliability 与 downstream error、degree、local homophily 的独立诊断；
- negative weighting 的 false-negative pressure 诊断。
