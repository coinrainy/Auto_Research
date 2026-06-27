# Adaptive Spectral Mix GCL 候选研究日志

日期：2026-06-28

## 候选问题

在 Graph Contrastive Learning 中，随机 feature masking 与 edge dropping 默认所有局部区域都适合相同扰动强度。但在 heterophily 图上，邻域聚合既可能提供有用低频语义，也可能引入跨类噪声。当前候选 idea 是把增强从“随机删除特征”推进到“局部自适应的低频/高频特征混合”。

核心设定：

- 对每个节点估计局部特征一致性；
- 一致性高的区域更偏向邻域均值低频视图；
- 一致性低的区域保留更多节点自身相对邻域的高频残差；
- 两个 contrastive view 通过轻微 gate jitter 形成差异；
- 默认仍保留 GRACE 的 edge drop 与 feature drop，以保证 baseline 对照清晰。

候选方法暂命名为 `spectral_mix`。当前最保守、也最值得继续扩展的设置是 `--spectral-high-scale 0.5`，用于避免高频残差过强。

## 文献边界

已快速核查的相关方向包括 HLCL、GCL-JAM、AS-GCL、ProGCL、GRAPE、GraphRank 等。前几类已经覆盖 heterophily 或 spectral GCL，后几类主要覆盖 false negative、hard negative 或 adversarial negative mining。

因此当前候选不能宣称“首次做 spectral augmentation”或“首次处理 heterophily GCL”。更合理的创新边界是：提出一种轻量、局部自适应、可直接插入 GRACE 视图构造的 low/high-pass feature mix，并通过同一协议验证它是否比纯随机增强更适合部分 heterophily 节点分类场景。

## 当前实现

实现位置：`experiments/grace_idea/train.py`

新增入口：

```bash
python train.py --dataset Texas --method spectral_mix \
  --spectral-mix-mode adaptive \
  --spectral-mix-jitter 0.1 \
  --spectral-high-scale 0.5
```

关键参数：

- `--method spectral_mix`：启用 adaptive spectral feature mix view。
- `--spectral-mix-mode adaptive|low|high|random`：选择 gate 来源。
- `--spectral-mix-temperature`：控制局部一致性 gate 的锐度。
- `--spectral-mix-jitter`：给两个 view 加相反方向的小扰动。
- `--spectral-high-scale`：控制高频残差强度；当前候选默认建议 `0.5`。

汇总脚本 `experiments/grace_idea/summarize_runs.py` 已支持 `--target-method spectral_mix`。

## 初筛结果

### 异配图，adaptive，high scale = 1.0

命令目录：`experiments/grace_idea/runs/spectral_mix_adaptive_splits0-2_seed0_e100`

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | -0.012061 | -0.008575 |
| Cornell | +0.009009 | -0.056450 |
| Texas | +0.009009 | -0.002293 |
| Wisconsin | +0.013072 | +0.112712 |

解释：高频残差不受限时，Wisconsin macro 有明显正向，但 Cornell macro 退化大，不适合作为默认设置。

### 异配图，adaptive，high scale = 0.5

命令目录：`experiments/grace_idea/runs/spectral_mix_adaptive_hs05_splits0-2_seed0_e100`

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Actor | -0.003509 | -0.001734 |
| Cornell | +0.099099 | +0.051104 |
| Texas | +0.036036 | +0.016846 |
| Wisconsin | -0.006536 | +0.065409 |

解释：这是当前最值得保留的候选设置。Texas/Cornell 有明显正向，Wisconsin macro 正向但 micro 轻微负向，Actor 近零负向。相比 SGFN，该路线在小样本初筛中更有生命力。

### 同配图 quick sanity，adaptive，high scale = 0.5

命令目录：`experiments/grace_idea/runs/spectral_mix_adaptive_hs05_homophily_seed0_e100`

相对 GRACE 的 split0 seed0 delta：

| Dataset | F1Mi delta | F1Ma delta |
| --- | ---: | ---: |
| Cora | -0.017364 | -0.026928 |
| CiteSeer | -0.006678 | -0.012850 |

解释：没有出现灾难性退化，但同配图仍有小幅下降。因此当前候选不能写成 universal non-degradation，需要继续做 safety gate 或只定位 heterophily-targeted augmentation。

## 当前判断

`spectral_mix --spectral-high-scale 0.5` 保留为当前 active candidate，但尚不能称为 SOTA idea。它比 SGFN 更值得继续，因为：

- performance sanity 在 Texas/Cornell 上显著强于 GRACE；
- Wisconsin 虽有 micro 下降，但 macro 正向，提示少数类或类别不均衡区域可能受益；
- Actor 负向很小，存在通过 gate 或 safety fallback 改善的空间；
- 机制上不是继续小修 false-negative attenuation，而是改变 GCL 的 view semantics。

当前不能声称：

- 通用 heterophily 提升；
- homophily non-degradation；
- 已超过 HLCL / AS-GCL / GCL-JAM 等 spectral GCL 方法；
- 已达到 2026 顶会顶刊投稿强度。

## 下一步

优先级从高到低：

1. 扩展 `spectral_mix high_scale=0.5` 到 Texas/Cornell/Wisconsin/Actor × splits 0-9，确认 split 稳定性。
2. 增加 Chameleon/Squirrel，并观察 high-degree 图上是否过度平滑或过度高频。
3. 做 ablation：`adaptive` vs `low` vs `high` vs `random`，证明不是任意滤波都有效。
4. 加入 homophily safety：当局部一致性极高时保留更接近 GRACE 的 feature drop，而不是强制 low/high mix。
5. 与 HLCL / AS-GCL / GCL-JAM 做协议级对照或至少复现实验表边界。

建议下一条正式命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2 3 4 5 6 7 8 9" SEEDS="0" METHODS="grace spectral_mix" EPOCHS=100 SAVE_DIR="runs/spectral_mix_adaptive_hs05_splits0-9_seed0_e100" MANIFEST_PATH="runs/spectral_mix_adaptive_hs05_splits0-9_seed0_e100/run_manifest.csv" OVERWRITE=1 LOG_EVERY=100 TRAIN_EXTRA_ARGS="--spectral-mix-mode adaptive --spectral-mix-jitter 0.1 --spectral-high-scale 0.5" scripts/run_split_study.sh
```
