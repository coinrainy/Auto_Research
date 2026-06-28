# GCL 研究方向重启备忘录：放弃 SPARC-GCL，转向顶会实现范式

日期：2026-06-28

## 当前裁决

放弃当前 SPARC-GCL / patched official SP-GCL residual branch idea，不再继续扩展实验。

放弃原因不是因为已有数值完全失败，而是因为该方向的实验实现方式与近年顶会/顶刊 Graph Contrastive Learning 论文的主流实现范式存在差异：

- 当前方法依赖对 third-party official SP-GCL 代码进行 patch，再导出 embedding 做外部 linear probe；
- 方法贡献容易被审稿人质疑为 post-hoc / wrapper-style modification，而不是完整、正统、可复现实验框架；
- 当前实验主线过度围绕 Chameleon/Squirrel 与单一 official baseline 展开，不足以支撑 2026 顶会顶刊级 claim；
- 后续若继续沿 patch-based 方式推进，工程投入会越来越偏离标准论文实现方式。

因此，SPARC-GCL 保留为 abandoned candidate / negative experience，不作为 active top-venue candidate。

## 已拉取的参考代码

以下代码仅作为实现范式参考，位于 ignored 目录 `third_party_baselines/reference_gcl/`，不提交第三方源码。

| 论文/方法 | Venue | 本地路径 | 参考重点 |
| --- | --- | --- | --- |
| PolyGCL: Graph Contrastive Learning via Learnable Spectral Polynomial Filters | ICLR 2024 | `third_party_baselines/reference_gcl/PolyGCL` | spectral polynomial views、heterophily datasets、10 split evaluator、dataset-specific scripts |
| S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning | ICML 2024 | `third_party_baselines/reference_gcl/S3GCL` | MLP inference、spectral biased views、semantic/spatial positives、cross-pass GCL |
| GraphECL: Efficient Contrastive Learning for Fast and Accurate Inference on Graphs | ICML 2024 | `third_party_baselines/reference_gcl/GraphECL` | fast inference GCL、teacher/moving-average style structure encoder、homophily/heterophily split scripts |

外部来源：

- PolyGCL official repository: https://github.com/ChenJY-Count/PolyGCL
- PolyGCL ICLR page: https://proceedings.iclr.cc/paper_files/paper/2024/hash/6faf3b8ed0df532c14d0fc009e451b6d-Abstract-Conference.html
- S3GCL official/public repository: https://github.com/GuanchengWan/S3GCL
- S3GCL ICML paper PDF: https://raw.githubusercontent.com/mlresearch/v235/main/assets/wan24g/wan24g.pdf
- GraphECL official repository: https://github.com/tengxiao1/GraphECL
- GraphECL ICML page: https://proceedings.mlr.press/v235/xiao24g.html
- Heterophily paper/code index: https://github.com/gongchenghua/Papers-Graphs-with-Heterophily

## 从参考实现中提取的实验范式

近年顶会 GCL / heterophily SSL 代码通常具备以下共同点：

1. 完整训练入口，而不是对第三方模型做一次性 patch；
2. 方法自身拥有独立 `model.py` / `training.py` / `run.sh`；
3. 明确区分 homophilous 与 heterophilous dataset protocol；
4. 使用固定 splits，通常对每个 dataset 跑 10 splits；
5. 使用 dataset-specific hyperparameter scripts，并在论文表中透明报告；
6. linear evaluator 嵌入主训练流程，或至少由同仓库统一实现；
7. 贡献点落在 view construction、positive pair construction、encoder/inference efficiency、spectral filter design 或 semantic/spatial neighbor modeling，而不是下游 embedding 后处理；
8. 对 heterophily 的方法通常直接在 Roman-empire、Amazon-ratings、Minesweeper、Tolokers、Questions 或 WebKB/Actor/WikipediaNetwork 等数据上建立主表，而不是只依赖单一老式小图。

## 下一阶段 idea 选择标准

新的 active candidate 必须满足：

- 直接以 PyG/DGL 标准训练入口实现，不依赖 patch third-party official code；
- 从第一版开始就包含 `train.py`、dataset loader、固定 split evaluator、config/run scripts；
- 至少对齐一个近年强 GCL baseline：PolyGCL、S3GCL、GraphECL、HeterGCL、HLCL 或 SP-GCL；
- 机制贡献必须能写成算法本体，而不是 analysis-only 或 post-hoc trick；
- 第一轮实验优先覆盖：
  - heterophily: Roman-empire、Amazon-ratings、Minesweeper、Tolokers、Questions，或 Texas/Cornell/Wisconsin/Actor/Chameleon/Squirrel；
  - homophily safety: Cora/CiteSeer/PubMed；
- 早筛标准必须严格：
  - 若只在 1 个数据集小幅正，不保留；
  - 若 shuffled/random control 同样有效，不保留；
  - 若需要过多 dataset-specific hack，不保留；
  - 若无法在单卡 RTX 3060 12GB 上完成 10 split 小实验，不保留。

## 新方向候选池

### Candidate A: Distribution-aware positive pair construction

问题：GraphECL/S3GCL 类方法大量依赖 semantic/spatial neighbors，但现有 positive pair 通常是 top-k 或 fixed sampling，缺少对 local distribution shift、feature heterophily 与 class-conditional uncertainty 的显式建模。

可行贡献：

- 用 label-free local distribution statistics 为 semantic positives 设置信任区间；
- 将 positive pair 从 hard top-k 改为 distribution-calibrated soft positives；
- 与 GraphECL/S3GCL 的 MLP-inference / fast-inference 范式兼容。

风险：容易和 hard positive mining / false positive literature 重叠，需要非常清楚地区分。

### Candidate B: Spectrum-conditioned augmentation-free GCL

问题：PolyGCL/S3GCL 已经证明 high/low-pass view 对 heterophily 有用，但现有方法多使用固定或 learnable global filters，缺少 node-wise spectrum routing。

可行贡献：

- 以 node-level feature-edge spectral statistics 选择 contrastive branch；
- 不是 post-hoc 拼接，而是在训练目标内决定 low/high/identity view 的正负对；
- 以 PolyGCL/S3GCL 的实现为主参考。

风险：和 SPARC 的失败路线有概念接近，必须避免再变成后处理 residual calibration。

### Candidate C: Inference-efficient heterophily GCL with neighbor-cache distillation

问题：GraphECL/S3GCL 共同关注 fast inference，即训练期利用图结构、推理期使用 MLP。现有方法仍可能依赖固定 neighbor sampling 或 EMA teacher，缺少对 heterophily-aware neighbor cache 的可靠性控制。

可行贡献：

- 训练期维护 semantic/spatial neighbor cache；
- 将结构 encoder 的信息蒸馏到 MLP encoder；
- 用 cache staleness / feature disagreement 控制 positives；
- 推理期完全 MLP，与 ICML 2024 fast-inference 叙事一致。

风险：实现复杂度高，但实验范式最贴近顶会。

## 当前推荐

优先探索 Candidate C，其次 Candidate A。

理由：

- 它更接近 ICML 2024 GraphECL 和 S3GCL 的实现范式；
- 不依赖对第三方代码打补丁；
- 有明确工程落点：neighbor cache、EMA/moving-average structure encoder、MLP inference、10 split evaluator；
- 贡献叙事更现代：训练期图结构知识 -> 推理期轻量 MLP，同时处理 heterophily positive pair 可靠性。

下一步应先建立一个标准化 `experiments/topvenue_gcl/` 实验骨架，而不是继续复用 `experiments/grace_idea/` 中累积了大量失败原型的代码。

## 已创建的新实验目录

已创建：

- `experiments/topvenue_gcl/README.md`
- `experiments/topvenue_gcl/.gitignore`
- `experiments/topvenue_gcl/configs/`
- `experiments/topvenue_gcl/scripts/`
- `experiments/topvenue_gcl/src/`
- `experiments/topvenue_gcl/docs/implementation_principles.md`
- `experiments/topvenue_gcl/runs/.gitkeep`

该目录将作为后续 active candidate 的唯一主工作区。`runs/` 已设置为本地输出目录，不提交实验产物。

## 下一步建议命令

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
find ../../third_party_baselines/reference_gcl -maxdepth 3 -name 'run*.sh' -o -name '*train*.py' -o -name 'main*.py'
```

随后将 GraphECL/S3GCL/PolyGCL 的 dataset loading、10 split evaluator、run script 风格抽象成新的 minimal baseline scaffold。
