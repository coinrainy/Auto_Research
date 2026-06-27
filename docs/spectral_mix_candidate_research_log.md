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

候选方法暂命名为 `spectral_mix`。早期最保守的设置是 `--spectral-high-scale 0.5`，用于避免高频残差过强；后续 10 split 复核表明该设置仍不够稳定。

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

更新于 2026-06-28：`spectral_mix --spectral-high-scale 0.5` 已从 active candidate 降级为被证伪的原型。split0-2 的早期正向在 10 split 复核中没有稳定保持，尤其 Cornell 从强正向转为明显负向。

### 10 split 复核，adaptive，high scale = 0.5

命令目录：`experiments/grace_idea/runs/spectral_mix_adaptive_hs05_splits0-9_seed0_e100`

相对 GRACE 的 split0-9 mean delta：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg |
| --- | ---: | ---: | ---: | ---: |
| Actor | +0.000526 | 5/0/5 | +0.003657 | 6/0/4 |
| Cornell | -0.018919 | 3/1/6 | -0.067351 | 2/0/8 |
| Texas | +0.010811 | 5/2/3 | -0.001730 | 4/1/5 |
| Wisconsin | +0.005882 | 6/0/4 | +0.019962 | 5/0/5 |

关键解释：

- Texas/Wisconsin 只有弱正向，且 split 稳定性不足；
- Actor 基本等于零；
- Cornell 明确失败，macro 8/10 split 为负；
- 因此 naive adaptive spectral mix 不能作为主方法继续扩展到 Chameleon/Squirrel 或多 seed。

class-level 诊断显示，该方法可能改善部分少数类，但会伤害另一部分类别。例如 Wisconsin 的 `F1Class4` 平均提升 +0.225714，但 `F1Class0/F1Class3` 出现负向；Cornell 的 `F1Class2` 平均下降 -0.202979。这说明当前 low/high gate 没有可靠对齐下游类别语义。

### Residual safety sanity，adaptive，high scale = 0.5，residual alpha = 0.5

为测试失败是否来自谱增强过度替换原始特征，已新增 `--spectral-residual-alpha`。默认值为 `1.0`，保持旧实验可复现；设为 `0.5` 时使用 `0.5 * original_feature + 0.5 * spectral_feature`。

命令目录：`experiments/grace_idea/runs/spectral_mix_residual_a05_hs05_splits0-2_seed0_e100`

相对 GRACE 的 split0-2 mean delta：

| Dataset | F1Mi delta | F1Mi pos/zero/neg | F1Ma delta | F1Ma pos/zero/neg |
| --- | ---: | ---: | ---: | ---: |
| Actor | -0.003289 | 0/0/3 | -0.011317 | 0/0/3 |
| Cornell | +0.000000 | 2/0/1 | +0.011631 | 1/0/2 |
| Texas | +0.036036 | 2/1/0 | +0.055615 | 3/0/0 |
| Wisconsin | -0.019608 | 1/1/1 | +0.038815 | 2/0/1 |

Residual anchor 保留了 Texas 的收益，并改善了 Cornell split0-2 的均值，但 Actor/Wisconsin 仍不稳定，不能作为新的主候选。

## 当前保留价值

`spectral_mix` 原型不再作为 active candidate，但保留以下资产：

- Texas 在 naive 与 residual 的短程 sanity 中都保留正向，说明谱扰动并非完全无效；
- Wisconsin 的 class-level 结果提示部分少数类或类别不均衡区域可能受益；
- Actor 基本零附近，说明该扰动不是强灾难性模块，但也没有贡献；
- 机制上从 false-negative attenuation 转向了 view semantics，为下一轮更保守的 view selection 提供了失败边界。

需要修正为更保守的表述：

- 该原型提供了“部分少数类可能受益于谱扰动”的线索；
- 当前 gate 与下游语义不可靠对齐，不能直接形成方法论文主线；
- 后续若继续 spectral 方向，应从“替换式 spectral feature mix”转向“语义保守的 spectral perturbation selection”，例如只在检测到 GRACE 视图不稳定或局部类别偏置风险时启用。

当前不能声称：

- 通用 heterophily 提升；
- homophily non-degradation；
- 已超过 HLCL / AS-GCL / GCL-JAM 等 spectral GCL 方法；
- 已达到 2026 顶会顶刊投稿强度。

## 下一步

优先级从高到低：

1. 暂停 naive `spectral_mix` 扩展，不继续跑 Chameleon/Squirrel。
2. 若继续 spectral 方向，先设计更强的 safety gate：不是固定 low/high 混合，而是判断何时应该保留原始 GRACE view。
3. 做 `adaptive` vs `low` vs `high` vs `random` 的小消融，只用于定位失败机制，不作为主论文实验。
4. 重新寻找更有 SOTA 潜力的 idea：优先选择能改变 contrastive objective 或 view selection decision 的机制，而不是单纯替换特征。

建议下一条诊断命令：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
DATASETS="Texas Cornell Wisconsin Actor" SPLITS="0 1 2" SEEDS="0" METHODS="grace spectral_mix" EPOCHS=100 SAVE_DIR="runs/spectral_mix_mode_ablation_splits0-2_seed0_e100" MANIFEST_PATH="runs/spectral_mix_mode_ablation_splits0-2_seed0_e100/run_manifest.csv" OVERWRITE=1 LOG_EVERY=100 TRAIN_EXTRA_ARGS="--spectral-mix-mode low --spectral-high-scale 0.5 --spectral-residual-alpha 0.5" scripts/run_split_study.sh
```
