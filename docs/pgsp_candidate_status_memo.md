# PGSP-GCL 候选状态备忘录

日期：2026-06-28

## 设计动机

Raw-Complement 被 SP-GCL 强基线压住后，下一候选尝试转向 single-pass / propagation-guided 方向。文献边界如下：

- SP-GCL：single-pass、augmentation-free，对 homophily/heterophily 都给出理论和实证保证，且已在当前 Chameleon/Squirrel 半正式基线中明显强于 Raw-Complement；
- PolyGCL / HLCL：已经覆盖 spectral polynomial / low-pass high-pass graph filter 叙事；
- HeterGCL：已经覆盖 heterophily 下结构-语义双模块对比；
- PROP/PROPGCL：强调 propagation 与 transformation 解耦，提示简单传播本身可能比复杂 GCL 更强。

因此本轮新原型不是继续做双增强 GRACE，而是实现 `pgsp_gcl`：

> Propagation-Guided Single-Pass GCL：单次前向得到 embedding，用多跳传播签名或 embedding 自举在采样子图内构造 pseudo-positive 排序，再用 learned embedding 的相似度优化 positive attraction 与 random/low-sim negative dispersion。

## 已实现能力

代码入口：

- `experiments/grace_idea/train.py`
- `experiments/grace_idea/model.py`

新增方法：

- `--method pgsp_gcl`

核心参数：

- `--pgsp-hops`
- `--pgsp-topk`
- `--pgsp-neg-topk`
- `--pgsp-max-size`
- `--pgsp-target-blend`
- `--pgsp-neg-selection random|low_target|low_embedding`
- `--pgsp-anchor-sampling random|tree`
- `--pgsp-seed-num`
- `--pgsp-anchor-hops`
- `--pgsp-square-sample`
- `--pgsp-hidden`
- `--pgsp-dropout`
- `--pgsp-use-bn`

实现要点：

- `SinglePassEncoder` 使用 GCNConv + optional BatchNorm + dropout；
- `pgsp_propagation_signature` 构造 raw、multi-hop propagated feature、residual blocks 的无标签传播签名；
- `pgsp_sample_anchor_index` 支持 random 与 tree-style k-hop anchor sampling；
- `propagation_guided_single_pass_objective` 支持 square-sample 内部 top-k positive / negative 训练。

## 早筛结果

主筛查都使用 Chameleon/Squirrel split0 seed0 50 epoch，目标是判断是否值得扩展到 10 split。

| Variant | Chameleon F1Mi/F1Ma | Squirrel F1Mi/F1Ma | 判断 |
| --- | ---: | ---: | --- |
| PGSP v1, propagation target blend=0.7, full candidates | 0.4057 / 0.4019 | 0.2632 / 0.2198 | 失败 |
| SP-like, embedding target, full candidates | 0.3947 / 0.3913 | 0.2988 / 0.2526 | 失败 |
| SP-like + BN/dropout/hidden256, full candidates | 0.3728 / 0.3709 | 0.3007 / 0.2899 | 失败 |
| SP-like + BN/dropout/hidden256 + tree/square sample | 0.4430 / 0.4409 | 0.2882 / 0.2531 | Chameleon 接近 raw，Squirrel 失败 |
| tree/square sample + propagation blend=0.3 | 0.4276 / 0.4252 | 0.2959 / 0.2790 | 失败 |

对照线索：

- Raw feature split0：Chameleon 0.4408 / 0.4365；Squirrel 0.3285 / 0.3253。
- Official SP-GCL embedding 在当前项目的 split0 mask probe：Chameleon 0.6382 / 0.6408；Squirrel 0.4428 / 0.4400。
- Official SP-GCL + raw concat 反而低于 SP-GCL embedding：Chameleon 0.6031 / 0.6012；Squirrel 0.4217 / 0.4129。

## 当前裁决

`pgsp_gcl` v1 不作为 active top-venue candidate 继续推进。

原因：

- 单独 embedding 明显弱于 official SP-GCL；
- propagation-guided pseudo-positive 没有带来增益，反而削弱 Chameleon；
- Squirrel 明显低于 raw feature baseline 与 SP-GCL；
- raw concat 虽然能让弱 PGSP embedding 对 raw 有一点互补增益，但 official SP-GCL 已经证明简单 raw concat 会伤害强 embedding。

保留价值：

- 当前代码提供了一个可复用的 single-pass objective scaffold；
- tree/square sampling、传播签名、raw/ssl fusion 诊断可作为后续实验工具；
- 负结果提示：若要超过 SP-GCL，不能只把传播签名当作 pseudo-positive 排序；需要更强的机制，例如可学习 propagation operator、class/cluster-aware propagation depth、或结构化 output selection，而不是静态多跳特征相似度。

## 下一步建议

停止继续调 `pgsp-target-blend`、`pgsp-topk` 或 `pgsp-hidden` 这类小参数。下一轮应重新构思：

1. 以 official SP-GCL 为强核心，寻找能在不伤其 embedding 的情况下改进 evaluation/selection 的机制；
2. 或转向 learnable propagation / propagation-depth selection，而不是静态 propagation signature；
3. 若继续 single-pass 路线，必须先复现 official SP-GCL 的 embedding 质量，再谈创新模块。
