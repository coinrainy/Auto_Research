# GRACE Idea 代码与理论系统审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / code-and-theory audit
- Origin Date: 2026-06-27
- Verification Status: ANALYZED
- Evidence Scope: 当前可运行代码限定在 `experiments/grace_idea/`，历史 `src/rwgcl/`、`configs/`、`scripts/`、`results/` 框架已经删除，旧文档中的相关结论只作为研究记录保留。
- No New Training: 本轮未继续扩展训练实验，只做静态代码审计、当前结果表复核与理论证据链检查。

## 当前真实实现边界

当前工作副本是从 `baselines/GRACE` 复制出的 `experiments/grace_idea/`。原始 `baselines/GRACE` 是 submodule，未被修改。

当前方法只有两个入口：

- `--method grace`：保留 GRACE 原始训练逻辑。
- `--method es_weighted`：warm-up 后使用 EMA teacher 的 clean-graph encoder embedding 作为参照，对两个增强视图下的 student encoder embedding 做余弦稳定性评分，再把该节点级分数用于 weighted InfoNCE。

重要边界：

- 当前没有 `prediction_consistency` 或 `projection_distribution_consistency` 训练信号。
- 当前没有 shuffled / random reliability control 入口。
- 当前没有独立 `diagnose.py`、failure analysis、false-negative pressure、downstream error bucket 诊断脚本。
- 当前支持的数据集是 Cora、CiteSeer、PubMed、DBLP、Texas、Cornell、Wisconsin、Actor；尚未接入 Chameleon、Squirrel 或新的 PyG heterophily benchmark。
- 当前已有实验结果主要是 Cora sanity 与 Texas/Cornell/Wisconsin/Actor 的 split 0-2、model seed 0 小规模对照。

## 核心结论

当前代码已经修掉了早期最危险的“projection head softmax 被叫作分类 prediction consistency”的定义错位，因为现版本只保留 embedding-stability-only。这个方向更干净，但论文叙事必须同步收缩：

> 现在不能再声称方法由 embedding stability + prediction consistency 共同估计 pair reliability；更准确的说法是：基于 EMA teacher 的节点级增强稳定性，对 GRACE 的正样本 anchor 训练信号和可选 denominator candidate 贡献进行轻量重加权。

当前最大缺口不是实现不能跑，而是证据链还不够独立：

- 它更像 view/augmentation stability weighting，不足以直接等价于分类语义 reliability。
- 可选 `--negative-weighting` 是节点候选级权重，不是 pair-specific false-negative suppression。
- 当前没有 normal vs shuffled/random 的同协议反事实对照。
- split 与 model seed 的论文级协议还没有自动化。
- 旧文档记录里有大量删除前框架的实验与脚本，不能混同为当前代码可复现能力。

## 代码审计发现

### P0. 旧研究记录与当前可运行代码脱节

`docs/reliability_weighting_decision_memo.md` 与 `AGENTS.md` 仍保留了旧框架时期的 split-aware runner、projection consistency、component ablation、false-negative pressure 等记录。但当前文件树中已经没有根目录 `train.py`、`src/rwgcl/`、`scripts/run_small_reliability_study.sh`、`diagnose.py` 或相关分析脚本。

影响：

- 后续如果按旧文档命令复现实验，会直接找不到入口。
- 论文证据整理时容易把“历史探索结果”误写成“当前 GRACE 副本可复现实验”。

处理建议：

- 保留旧文档作为历史记录，但所有新实验以 `experiments/grace_idea/` 为唯一代码事实源。
- 新增或更新文档时显式标注“当前 GRACE 副本可复现”与“旧框架历史结果”。

### P0. 当前没有 shuffled/random reliability control

当前 `train.py` 的参数只有 `--method grace|es_weighted`，没有 `--shuffled-reliability` 或 `--random-reliability`。这意味着最关键的反事实诊断还没有在新 GRACE 副本里恢复。

影响：

- 不能排除收益来自任意 reweighting、训练噪声或正则化效应。
- 不能支撑“reliability 非随机有效”的机制主张。

处理建议：

1. 在 `es_weighted` 中增加 `--shuffle-weights` 和 `--random-weights`。
2. 保存 `final_weights`、shuffled 标记、随机种子与置乱索引摘要。
3. 汇总脚本按 dataset + split + model seed 同时对齐 GRACE、normal、shuffled、random。

### P0. split 协议还不是论文级

`train.py` 已有 `--split-index`，并且 WebKB/Actor 会使用固定 mask 评估。但当前批量能力只在 `summarize_runs.py` 层面做 run 目录汇总，没有正式 runner 保证 `split_index=0..9` 和 `model_seed` 的区分。

影响：

- 目前 split 0-2、seed 0 只能视作 sanity，不是标准异配图实验。
- 若继续只循环 seed 而不循环 split，会重犯此前“10 seeds 不是 10 splits”的问题。

处理建议：

- 先实现一个极小 runner，显式循环 `SPLITS` 和 `SEEDS`。
- run 目录命名建议加入 `modelseed{}`，避免把 split seed 与模型 seed 混在一起。
- 汇总输出至少包含 `dataset, split_index, model_seed, method, F1Mi, F1Ma`。

### P1. run 目录命名会碰撞，重复实验会污染日志

`prepare_save_dir()` 使用 `{dataset}_{method}_seed{seed}_split{split}` 作为目录名。如果重复运行同一配置，`train_log.csv` 会继续 append，而 `eval_summary.csv`、`metadata.json`、`artifacts.pt` 会覆盖。

影响：

- `summarize_runs.py` 读取最后一行训练日志，表面上还能跑，但同一目录已经混入多次实验历史。
- 复现实验时很难判断产物来自哪一次命令。

处理建议：

- 默认拒绝写入已存在且非空的 run 目录，除非显式 `--overwrite`。
- 或在目录名中加入时间戳 / 短 run id。
- metadata 中保存完整命令、git commit、baseline submodule commit、依赖版本。

### P1. `negative-weighting` 的语义需要改名或补诊断

当前 `Model.semi_loss()` 中 denominator weighting 是按 candidate node reliability 统一缩放分母贡献，并保留 positive 项不被缩放。它不是 pairwise `w_ij_neg`。

影响：

- 它可以降低低可靠节点作为所有 anchor 的负样本贡献。
- 它不能识别“anchor i 与 negative j 是否可能同类”，所以不能直接声称解决 false negative / hard negative imbalance。
- 低可靠节点如果本身是有用 hard negative，也可能被过度降权。

处理建议：

- 文中称为 `denominator candidate weighting`，不要称为 false-negative weighting。
- 若要支撑 false-negative 机制，至少补 label-based 离线诊断：同标签节点在 denominator softmax mass 中的占比变化。
- 若要做真正的训练机制，需要 pair-specific negative reliability，例如基于 teacher embedding 相似度、prototype agreement 或局部结构上下文的 `q_ij`。

### P1. 权重区分度过弱

已有 split 0-2 实验中，`es_weighted` final weight mean 多在 0.946 到 0.977，std 约 0.014 到 0.021。结合 `min_weight=0.05` 和余弦相似度映射，当前权重实际只是在非常窄的范围内微调样本贡献。

影响：

- 这解释了为什么 Cora 与 Wisconsin 基本无变化。
- 如果继续作为方法论文，reviewer 可能质疑“改动太轻，是否只是训练噪声”。

处理建议：

- 做 `weight_power` 与 `min_weight` 的小范围敏感性，但不要过度调参。
- 报告权重分布、effective sample size、top/bottom bucket 的下游错误率。
- 若权重始终饱和，理论叙事应转为 stability diagnostic，而不是强方法。

### P1. 数据集覆盖低于研究计划

当前 `get_dataset()` 未接入 Chameleon、Squirrel、Roman-empire、Amazon-ratings 等计划中的异配数据集。`config.yaml` 也只补了 WebKB 与 Actor。

影响：

- 当前不能复现旧记录中关于 Chameleon/Squirrel 的判断。
- 当前异配证据过窄，且 WebKB 小图方差很高。

处理建议：

- 优先补 Chameleon/Squirrel，因为它们能检验 degree-stability 偏置。
- 第二步再考虑 PyG `HeterophilousGraphDataset` 的新 benchmark。

### P1. 诊断能力不足

当前新 GRACE 副本只有 `summarize_runs.py`。没有以下独立诊断：

- reliability bucket 的 downstream test error；
- reliability 与 degree/local homophily/class distribution 的关系；
- label-based false-negative pressure；
- confusion matrix delta 自动摘要；
- shuffled/random reliability 对照诊断。

影响：

- 只能看 performance delta 和 final weight stats。
- 无法回答“为什么 Texas 有效、Cornell/Wisconsin 无效”。

处理建议：

- 先实现 confusion matrix delta 与 downstream error bucket，成本低且直接利用现有 `eval_details.json` 和 `artifacts.pt`。
- 再实现 false-negative pressure 和 degree/local analysis。

### P2. 可复现性记录还不完整

`set_seed()` 设置了 Python 与 PyTorch seed，但没有设置 NumPy seed，也没有保存依赖版本、CUDA 信息、完整命令和 git commit。

影响：

- 当前结果能作为探索记录，但论文复现材料还不够。

处理建议：

- 增加 `np.random.seed(seed)`。
- metadata 保存 `sys.argv`、`torch.__version__`、`torch_geometric.__version__`、CUDA、git commit、submodule commit。
- 需要时再打开 deterministic flags，但要记录性能影响。

### P2. requirements 需要现代化

`experiments/grace_idea/requirements.txt` 使用了 `sklearn`，现代 pip 环境更推荐 `scikit-learn`。依赖也没有版本上限或 CUDA/PyG 安装说明。

影响：

- 新机器复现可能因为 PyG wheel / CUDA 版本不匹配失败。

处理建议：

- 改为 `scikit-learn`。
- 在 README 中记录当前实际环境版本，而不是只沿用 GRACE 旧 requirements。

## 理论审计发现

### T0. 当前方法不是严格的 pair reliability

GRACE 的 positive pair 是同一节点在两个增强视图中的表示。节点 identity 不变，所以 positive pair 的标签一致性天然成立。当前 reliability 衡量的主要是：

- 该节点在增强扰动下的 encoder 表示是否稳定；
- student 是否接近 EMA teacher 的 clean-graph 表示；
- 节点作为 anchor 的对齐损失是否应该被强调。

因此更准确的概念是 `node-wise view reliability` 或 `augmentation-stability-aware anchor weighting`，而不是完整的 positive/negative pair reliability。

### T1. embedding stability 不等于分类语义可靠性

EMA teacher clean-graph embedding stability 很可能捕捉以下因素：

- 节点 degree 较高，增强后邻域仍稳定；
- 局部结构对 edge drop / feature mask 不敏感；
- teacher 与 student 的表征已形成自确认稳定区域。

这些因素可能与分类准确率相关，也可能在 heterophily 图上相反。当前理论需要把“稳定性”与“语义可靠性”分开，不能直接等号。

建议表述：

> embedding stability is used as an unlabeled proxy for view reliability, not as a guaranteed proxy for class semantics.

### T1. false-negative 机制仍未闭合

如果主张是解决 false negative / hard negative imbalance，则当前方法还缺两块：

- 训练中没有 pairwise negative reliability；
- 诊断中没有证明加权后同标签负样本的 denominator pressure 稳定下降。

当前 `--negative-weighting` 只能作为候选节点级 denominator attenuation。它可能是有价值的启发式，但不是 false-negative-specific 方法。

### T1. view consistency 诊断需要独立化

如果 reliability 由 embedding stability 定义，再按 reliability 分桶统计 embedding stability，高低桶差异就是定义内循环。当前新实现尚未恢复该诊断，但后续若恢复，必须避免把它作为独立机制证据。

更独立的诊断应优先使用：

- bucket-wise downstream test error；
- bucket-wise false-negative denominator pressure；
- bucket-wise class entropy / label agreement，仅作离线分析；
- reliability 与 degree/local structure 的偏相关或分层统计。

### T2. 当前实验结果支持保守方向

当前 split 0-2、seed 0 的 heterophily sanity 结果：

- Texas：mean delta F1Mi +0.0180，mean delta F1Ma +0.0107；
- Cornell：mean delta F1Mi -0.0090，mean delta F1Ma -0.0114；
- Wisconsin：0；
- Actor：mean delta F1Mi +0.0009，mean delta F1Ma +0.0003。

这不支持“稳定提升 heterophily”的强叙事，只支持：

- 代码路径可运行；
- Texas 存在弱正向信号；
- 其他数据集不稳定或无效；
- embedding-stability-only 值得继续做更规范的 split-aware 小规模验证。

## 建议优先级

### 立即做，低风险

1. 修复 run id / overwrite 机制，避免重复实验污染。
2. 加 `--shuffle-weights` 与 `--random-weights`。
3. 写 split-aware runner：`DATASETS × SPLITS × SEEDS × METHODS`。
4. 写 confusion matrix delta 摘要，先解释 Texas/Cornell。
5. 保存完整 reproducibility metadata。

### 第二批做，决定理论路线

1. 做 downstream error bucket 诊断。
2. 做 label-based false-negative pressure 诊断。
3. 做 degree / local structure / reliability 关系分析。
4. 接入 Chameleon/Squirrel。
5. 扩展 Texas/Cornell/Wisconsin/Actor 到 `split_index=0..9`，每个 split 至少 `model_seed=0`，再决定是否加更多 model seeds。

### 暂缓

- degree gate；
- closed-loop augmentation；
- high/low-pass gate；
- SOTA baseline 大扩展；
- 把方法包装为通用 heterophily 提升器。

## 当前推荐路线

当前最稳妥的下一步不是继续加模块，而是先把 GRACE 副本补成“可证伪”的实验平台：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python train.py --dataset Texas --method es_weighted --seed 0 --split-index 0 --epochs 100 --warmup-epochs 20 --negative-weighting --save-dir runs/audit_next
```

在此基础上优先实现 shuffled/random control 和 split-aware runner。等这些补齐后，再跑：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python summarize_runs.py --runs-dir runs/hetero_splits0-2_seed0_e100 --paired-out runs/summaries/hetero_splits0-2_seed0_e100_paired.csv --aggregate-out runs/summaries/hetero_splits0-2_seed0_e100_aggregate.csv
```

论文叙事建议先收缩为：

> We study whether node-wise embedding stability under graph augmentations can serve as an unsupervised view reliability signal for reweighting GRACE-style contrastive training. Current evidence suggests conditional benefits and important failure modes, rather than a universal heterophily improvement.

