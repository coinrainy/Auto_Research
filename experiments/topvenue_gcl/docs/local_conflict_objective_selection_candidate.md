# Local-Conflict Objective Selection GCL 候选记录

## 当前裁决

`lcos_gcl` 已降级为 **失败/条件性诊断资产**，不进入 splits 0-2 扩展。

核心原因：

- Texas micro 明显低于 `gcn_mlp_gcl`，Chameleon 也低于 baseline；
- Actor 仅小幅正向；
- Squirrel 同时满足 normal > baseline 与 normal > shuffled，是唯一有机制线索的数据集；
- 因此局部冲突 gate 可能有信息量，但“直接在 graph view 与 high-pass view 之间切换对齐目标”的 objective 设计不成立。

## 方法定义

LCOS 保留 `gcn_mlp_gcl` 的 MLP/GCN natural-view foundation，但加入无标签局部冲突 gate：

- raw agreement：`cos(x, P x)`；
- raw residual：`||x - P x|| / (||x|| + ||P x||)`；
- degree context：`log(1 + degree)`；
- gate score：标准化 residual - 标准化 agreement + degree weight；
- high gate：`sigmoid((score - threshold) / temperature)`，并用 `lcos_min_branch_weight` 防止分支权重为 0。

训练目标：

- 低冲突节点更多使用完整 graph view alignment；
- 高冲突节点更多使用 high-pass view alignment；
- final representation 使用 `[ego, (1-gate) graph + gate high]`。

机制 control：

```bash
--lcos-shuffle-gate
```

该 control 打乱 gate 与节点对应关系。如果 normal 不强于 shuffled，则局部冲突信号本身不成立。

## 实验记录

执行：

```bash
RUNS_DIR="runs/lcos_split0_s0_e50"
DATASETS="Texas Actor Chameleon Squirrel" METHODS="gcn_mlp_gcl lcos_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="normal" OVERWRITE=1 bash scripts/run_split_study.sh
DATASETS="Texas Actor Chameleon Squirrel" METHODS="lcos_gcl" SPLITS="0" SEEDS="0" EPOCHS=50 RUNS_DIR="$RUNS_DIR" RUN_TAG="shuffled" EXTRA_ARGS="--lcos-shuffle-gate" OVERWRITE=1 bash scripts/run_split_study.sh
python summarize_split_study.py --runs-dir "$RUNS_DIR" --baseline-method gcn_mlp_gcl --out "$RUNS_DIR/runs_vs_gcn_mlp.csv" --aggregate-out "$RUNS_DIR/aggregate_vs_gcn_mlp.csv"
```

结果：

| Dataset | Normal ΔF1Mi/ΔF1Ma | Normal - shuffled ΔF1Mi/ΔF1Ma | High gate mean | 裁决 |
| --- | ---: | ---: | ---: | --- |
| Texas | -0.054054 / +0.072502 | +0.000000 / +0.056593 | 0.517036 | micro 失败 |
| Actor | +0.009211 / +0.006041 | +0.001974 / +0.001804 | 0.505916 | 弱正向 |
| Chameleon | -0.004386 / -0.006271 | +0.015351 / +0.012920 | 0.465479 | baseline 失败 |
| Squirrel | +0.013449 / +0.006619 | +0.050913 / +0.041619 | 0.464667 | 唯一清楚机制线索 |

## 解释

LCOS 的 positive evidence：

- Squirrel 上 normal 同时超过 baseline 与 shuffled，说明局部冲突 gate 在某些异配结构中确实可能有用；
- Actor 有小幅正向，但机制差距很小。

LCOS 的 negative evidence：

- Texas micro 大幅下降，说明直接把高冲突节点推向 high-pass target 会损害主类别判别；
- Chameleon 低于 baseline，即使 normal > shuffled 也不能构成有效方法；
- gate 均值集中在 0.46-0.52，说明第一版 gate 更像半软切换，不足以形成清晰的节点级 objective selection。

## 后续判断

不继续调第一版 LCOS 的 threshold、temperature 或 degree weight。

下一代 idea 如果继承 LCOS 线索，应改变目标设计：

- 不再直接对齐 high-pass target；
- 将局部冲突 gate 用于 **loss reliability / negative suppression / representation mixing**，而不是硬切 graph/high alignment；
- 需要加入 downstream separability proxy 或 cluster-margin proxy，避免只优化 view alignment。
