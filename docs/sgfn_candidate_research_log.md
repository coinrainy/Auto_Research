# SGFN 候选方法研究日志

日期：2026-06-28（UTC）

## 当前候选 idea

候选名称：SGFN（Stability-Guided False-Negative attenuation for Graph Contrastive Learning）

核心问题：在 GRACE/InfoNCE 式节点级图对比学习中，所有非自身节点都会进入 negative denominator；其中同类节点会被错误推开，形成 false-negative contamination。已有 hard-negative / false-negative mining 工作很多，因此本项目不应只声称“发现假负样本”，而要强调：

- 使用 EMA teacher 在 clean graph 上形成稳定参照；
- 对每个 anchor 做 row-standardized teacher similarity，得到 pair-level false-negative risk；
- 在 InfoNCE denominator 中对疑似 false negative 做有界 attenuation；
- 用 shuffled-pair control 和 label-only false-negative pressure 诊断证明不是简单弱化负样本或随机重加权。

## 文献边界

当前快速核查到的强相关工作包括：

- ProGCL：指出图上 hard negatives 往往是 false negatives，并估计 negative 为 true negative 的概率。
- FD4GCL：使用 attribute / structure-aware 方式检测 false negative。
- GraphRank：通过 ranking objective 规避 InfoNCE 把同类节点当 negative 的问题。
- GRAPE：从 subspace preserving 角度做 expansive/adaptive hard negative mining。
- HLCL / AS-GCL：从 heterophily 与 spectral view 方向改造 GCL。

因此 SGFN 的潜在新意目前只能保守表述为：轻量 teacher-stability pair risk + pair-denominator attenuation + 可证伪机制诊断；不能表述为“首次解决 false negative”或“通用 heterophily SOTA”。

## 已实现内容

代码位置：`experiments/grace_idea/`

- `train.py`
  - 新增 `--method sgfn`。
  - 新增 pair-level denominator weighting：`pair_denominator_weights`。
  - 新增 false-negative risk 参数：
    - `--fn-risk-margin`
    - `--fn-risk-temperature`
    - `--fn-attenuation-power`
    - `--fn-consensus none|feature`
  - 新增控制组：
    - `--shuffle-weights`：打乱 pair weight 映射。
    - `--random-weights`：uniform random pressure test。
    - `--pair-shuffle-mode column|row`。
  - 新增实验性扩展：
    - `--pair-normalization row_mean|blend_row_mean`：denominator mass reallocation 消融。
    - `--fn-attraction-weight`：疑似 false negative attraction 消融。
- `model.py`
  - InfoNCE 支持 pair-specific denominator weights，含 full-batch 与 batched 路径。
- `summarize_runs.py`
  - 支持 `--target-method sgfn`，可汇总 SGFN vs GRACE 与 normal-vs-control。
- `analyze_pair_weights.py`
  - 新增 label-only 机制诊断，计算同标签 false-negative keep weight、异标签 true-negative keep weight、weighted/unweighted false-negative pressure share，并输出 aggregate 与 normal-vs-control 对照。
- `scripts/run_split_study.sh`
  - 支持 `sgfn` 批跑。

## 小规模结果

所有结果仅为 sanity，不是正式论文结果。设置均为 Texas/Cornell/Wisconsin/Actor × splits 0/1/2 × model seed 0 × 50 epochs。

### SGFN attenuation（当前主候选）

命令核心：`METHODS="grace sgfn" ES_CONTROLS="normal shuffled" EPOCHS=50 WARMUP_EPOCHS=20`

SGFN normal - GRACE（F1Mi mean）：

- Texas：+0.027027
- Cornell：+0.009009
- Wisconsin：-0.013072
- Actor：-0.002412

SGFN normal - shuffled（F1Mi mean）：

- Texas：+0.018018
- Cornell：+0.009009
- Wisconsin：-0.013072
- Actor：+0.001316

机制诊断（weighted - unweighted FN pressure，全图 mean）：

- Texas normal：-0.016206；shuffled：约 -0.000233
- Cornell normal：-0.008706；shuffled：约 -0.000138
- Wisconsin normal：-0.015739；shuffled：约 +0.000146
- Actor normal：约 -0.000385；shuffled：约 -0.000018

判断：SGFN attenuation 是当前唯一同时具备一定性能苗头与机制诊断信号的候选，但还远不到 SOTA。

### row-mean reallocation

命令核心：`TRAIN_EXTRA_ARGS="--pair-normalization row_mean"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.009009
- Cornell：+0.018018
- Wisconsin：0.000000
- Actor：-0.004825

判断：机制更干净，权重均值固定为 1，但整体性能不优；适合作为审稿人质疑“只是弱化 denominator”的消融，不适合作为默认主方法。

### blend reallocation

命令核心：`TRAIN_EXTRA_ARGS="--pair-normalization blend_row_mean --pair-reallocation-alpha 0.5"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.027027
- Cornell：-0.018018
- Wisconsin：0.000000
- Actor：-0.007237

判断：失败消融；不应继续扩大。

### attraction

命令核心：`TRAIN_EXTRA_ARGS="--fn-attraction-weight 0.1"`

SGFN normal - GRACE（F1Mi mean）：

- Texas：-0.027027
- Cornell：-0.009009
- Wisconsin：-0.006536
- Actor：-0.008553

判断：疑似 false negative attraction 不稳定，且 shuffled 在部分 split 更强；不应作为主线。

## 当前决策

主线收敛为：SGFN attenuation + 机制诊断，而不是 row-mean reallocation、blend reallocation 或 attraction。

论文定位若继续推进，应保守设为“false-negative pressure calibration / diagnostic-aware GCL method”，而不是通用 heterophily SOTA。下一步必须用更严格协议验证是否有足够强的投稿潜力。

## 下一步建议命令

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" METHODS="grace sgfn" ES_CONTROLS="normal shuffled" EPOCHS=100 WARMUP_EPOCHS=20 SAVE_DIR="runs/sgfn_attenuation_hetero_splits0-9_seed0_e100" MANIFEST_PATH="runs/sgfn_attenuation_hetero_splits0-9_seed0_e100/run_manifest.csv" OVERWRITE=1 LOG_EVERY=100 scripts/run_split_study.sh
python summarize_runs.py --runs-dir runs/sgfn_attenuation_hetero_splits0-9_seed0_e100 --target-method sgfn --paired-out runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_paired.csv --aggregate-out runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_aggregate.csv
python analyze_pair_weights.py --runs-dir runs/sgfn_attenuation_hetero_splits0-9_seed0_e100 --out runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_pair_weights.csv --aggregate-out runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_pair_weights_aggregate.csv --control-paired-out runs/summaries/sgfn_attenuation_hetero_splits0-9_seed0_e100_pair_weights_controls.csv
```

停止标准：

- 若 10 split 下 Texas/Cornell/Wisconsin/Actor 的平均 F1Mi/F1Ma 仍无至少 2 个数据集正向，或 normal 不稳定优于 shuffled，则停止 SGFN 方法线，转向机制/负结果论文或重新构思。

