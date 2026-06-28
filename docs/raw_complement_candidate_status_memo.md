# Raw-Complement GCL 候选状态备忘录

日期：2026-06-28

## 当前候选

当前 active candidate 是 no-penalty Raw-Relative Graph Complement GCL：

- raw anchor：由 ego/raw-feature MLP 得到；
- graph context：由 GCN encoder 得到；
- complement：`LayerNorm(graph_context - stop_gradient(raw_anchor))`；
- 默认主输出：`anchor_graph = [raw x, complement, graph_context]`；
- 默认不使用 correlation penalty：`raw_complement_weight=0.0`；
- auxiliary graph-context loss 仅作为 optional safety refinement，不作为主方法。

## 最强证据

最终候选汇总文件：

- `experiments/grace_idea/runs/summaries/raw_complement_final_candidate_wiki_s0-2_paired.csv`
- `experiments/grace_idea/runs/summaries/raw_complement_final_candidate_wiki_s0-2_aggregate.csv`

Chameleon/Squirrel × 10 splits × seeds 0/1/2：

| Dataset | RC - raw F1Mi | RC - raw F1Ma | pos/zero/neg vs raw F1Mi | RC - GRACE F1Mi | RC - GRACE F1Ma |
| --- | ---: | ---: | --- | ---: | ---: |
| Chameleon | +0.036769 | +0.036744 | 30/0/0 | +0.072880 | +0.078006 |
| Squirrel | +0.010471 | +0.012456 | 26/2/2 | +0.062568 | +0.079629 |

解释：

- Chameleon 的信号很强：所有 30 个 split/seed pair 均超过 raw 与 GRACE；
- Squirrel 相对 GRACE 全部为正，相对 raw 均值为正，但有少量非正 pair；
- 这支持“Raw-Complement 在 WikipediaNetwork-style heterophily graphs 上稳定超过 GRACE，并能在多数情况下超过 raw-feature baseline”；
- 这不支持“通用 heterophily GCL SOTA”。

## 已放弃或降级的模块

- `raw_complement_weight=0.05` correlation penalty：与 no-penalty 持平，不是核心贡献；
- `raw_graph=[raw, graph_context]`：在 Squirrel 上相对 raw 为负，不能解释收益；
- `graph_only`：相对 raw 全负；
- label-free proxy selection：能优于 random，但不能解决 Cora safety；
- auxiliary graph-context preservation：能缩小 Cora gap，但不能修复 homophily non-degradation，也无主战场增益，仅保留为 optional appendix ablation。

## 最大风险

Homophily safety 尚未解决。

Cora seeds0-2：

| Variant | F1Mi | F1Ma | vs GRACE |
| --- | ---: | ---: | --- |
| GRACE | 0.824948 | 0.810003 | - |
| no-penalty graph fallback | 0.812597 | 0.791041 | -0.012351 / -0.018962 |
| label-free proxy selection | 0.814473 | 0.794776 | -0.010475 / -0.015227 |
| auxiliary graph loss 0.1 | 0.818567 | 0.794162 | -0.006381 / -0.015842 |

判断：

- Cora micro gap 可缩小，但 macro gap 仍明显；
- 当前不能声称 homophily non-degradation；
- 如果论文目标要求通用 GCL 方法，该风险会成为 major weakness；
- 若定位为 WikipediaNetwork-style heterophily 条件性方法，该风险可作为 boundary analysis 处理。

## 文献边界

近期/相关 heterophily GCL 已经覆盖以下方向：

- HLCL：通过 homophilic/heterophilic subgraph 与 low-pass/high-pass graph filters 处理 heterophily；
- PolyGCL：从 spectral polynomial filters 构造 low-pass/high-pass contrastive views；
- SP-GCL：声称单次前向的 GCL 可同时适配 homophily 与 heterophily；
- heterophily handbook/survey：强调 heterophily 数据集类型、benchmark 和模型边界。

参考入口：

- HLCL: <https://openreview.net/forum?id=khvJM3uFk8>
- PolyGCL: <https://openreview.net/forum?id=y21ZO6M86t>
- SP-GCL: <https://openreview.net/forum?id=244KePn09i>
- Heterophilic Graph Learning Handbook: <https://arxiv.org/abs/2407.09618>

因此 Raw-Complement 不能包装为“又一个高低频 heterophily GCL”。更稳妥的创新点应限定为：

> 在强 raw-feature baseline 下，显式学习 raw feature 之外的 graph-context complement，并证明简单 graph-context 或 raw+graph 拼接无法解释该收益。

## 2026 投稿可行性判断

当前证据达到“值得继续”的条件，但尚未达到“足以顶会/顶刊”的条件。

可以继续的理由：

- Chameleon/Squirrel 多 split 多 seed 结果稳定；
- 机制消融支持 residual complement，而非普通 graph context；
- 方法简单，适合构造清晰的 problem diagnosis -> method -> boundary analysis 论文。

暂不足的理由：

- 强 baseline 不够，目前主要对 GRACE 与 raw baseline；
- homophily safety 未解决；
- WebKB/Actor 上 raw baseline 支配，方法不能声称通用异配提升；
- 需要和 HLCL、PolyGCL、SP-GCL 或其可复现实验结果直接对齐。

## 下一步硬门槛

1. 强 baseline gate：
   - 至少补一个 heterophily-specific GCL baseline：HLCL、PolyGCL 或 SP-GCL；
   - 若无法复现官方实现，则至少写清楚不可比原因，并补公开表格/同协议近似对照。
   - 当前进展：SP-GCL 官方实现已在本地 smoke 跑通，见 `docs/spgcl_baseline_integration_note.md`。

2. Safety gate：
   - 不继续围绕 `raw_complement_graph_loss_weight` 做大网格；
   - 优先设计 dataset-level fallback 或 graph-context preservation 的结构性版本；
   - 若 Cora macro 仍低于 GRACE 超过 0.01，则不能写 homophily non-degradation。

3. Mechanism gate：
   - 补 per-class / degree / local homophily 诊断；
   - 证明 complement 帮助的是哪些节点/类别，而不只是 aggregate accuracy。

4. Scope gate：
   - 题目、摘要和贡献声明必须避免“general heterophily GCL SOTA”；
   - 推荐定位：raw-feature anchored complement learning for WikipediaNetwork-style heterophily graphs。

## 当前建议

继续保留 Raw-Complement 作为 active candidate，但进入“强 baseline 与机制诊断”阶段。若 HLCL/PolyGCL/SP-GCL 对齐后 Raw-Complement 只赢 GRACE、不赢 heterophily-specific GCL，则应放弃顶会主方法路线，转为机制诊断/short paper/附录负结果路线。
