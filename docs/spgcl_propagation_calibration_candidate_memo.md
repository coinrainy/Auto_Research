# SP-GCL Propagation-Residual Calibration 候选备忘录

日期：2026-06-28

## 当前候选

新的 active candidate 暂命名为：

> SPARC-GCL: Strong Propagation-residual Adaptive Representation Calibration for Graph Contrastive Learning

核心想法：

- 不再从弱 GRACE 变体重新发明 single-pass objective；
- 以 official SP-GCL 这类强 augmentation-free GCL embedding 为 backbone；
- 在训练后对 learned embedding 做轻量传播校准；
- 关键不是简单 raw concat，而是拼接 `SSL embedding + propagation residual / propagation-calibrated embedding`；
- 目标是保留 SP-GCL 的强判别信息，同时显式暴露“局部传播会改变什么”的残差信号。

当前最有前景的 mode：

- `ssl_resid1`: `[normalize(z), normalize(z - Pz)]`
- `ssl_prop2`: `[normalize(z), normalize(P^2 z)]`

其中 `P` 是带 self-loop 的 row-normalized graph propagation operator。

## 为什么这条线不同于 Raw-Complement / PGSP

Raw-Complement 失败点：

- 只赢 GRACE/raw，不能赢 SP-GCL；
- raw concat / raw complement 在强 SP-GCL embedding 上没有自然优势；
- homophily safety 不稳。

PGSP 失败点：

- 直接用 propagation signature 做 pseudo-positive training 明显弱于 official SP-GCL；
- 说明“传播签名作为训练目标”不是核心增益。

SPARC-GCL 的新信号：

- official SP-GCL embedding 已经很强；
- 对强 embedding 做 propagation residual calibration 后，Squirrel 出现跨 split 稳定提升；
- Chameleon 也有小幅正向，但幅度较小，需要继续验证。

## 当前证据

输入 embedding：

- 来自本地 official SP-GCL 克隆；
- 配置：`reset_epochs=100`、`reset_hidden=256`、`reset_seed_num=32`、`reset_max_size=512`、`reset_subg_num_hops=2`；
- 只保存 SSL embedding，后续用当前项目的 mask linear probe 统一评估。

评估脚本：

- `experiments/grace_idea/evaluate_propagation_calibration.py`

快速 gate：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python evaluate_propagation_calibration.py \
  --runs-dir /tmp/spgcl_embedding_artifacts \
  --include-methods spgcl_official \
  --modes ssl ssl_prop2 ssl_resid1 \
  --max-hop 2 \
  --split-indices 0 1 2 3 4 5 6 7 8 9 \
  --c-values 16 \
  --max-iter 300 \
  --out runs/summaries/spgcl_propagation_calibration_splits0-9_fast.csv \
  --aggregate-out runs/summaries/spgcl_propagation_calibration_splits0-9_fast_aggregate.csv
```

结果：

| Dataset | Mode | F1Mi mean | F1Ma mean | Delta vs SSL F1Mi | Delta vs SSL F1Ma | Positive/Negative F1Mi |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Chameleon | ssl | 0.623246 | 0.623299 | 0.000000 | 0.000000 | - |
| Chameleon | ssl_prop2 | 0.625439 | 0.626038 | +0.002193 | +0.002739 | 6/4 |
| Chameleon | ssl_resid1 | 0.632237 | 0.631764 | +0.008991 | +0.008465 | 7/2 |
| Squirrel | ssl | 0.440154 | 0.434061 | 0.000000 | 0.000000 | - |
| Squirrel | ssl_prop2 | 0.459270 | 0.454152 | +0.019116 | +0.020091 | 10/0 |
| Squirrel | ssl_resid1 | 0.468588 | 0.463756 | +0.028434 | +0.029695 | 10/0 |

解释：

- Squirrel 的 propagation residual calibration 信号非常清楚；
- Chameleon 的提升较小，但 `ssl_resid1` 仍为正均值；
- 这只是 fixed `C=16` 的 fast gate，不是正式结果；
- 当前证据足以把 SPARC-GCL 作为下一轮 active candidate，但还不足以声称 SOTA。

## 研究假设

SP-GCL 已经通过局部子图相似性学到强 embedding，但这个 embedding 仍混合了两类信息：

- 局部传播后保持稳定的 class-relevant signal；
- 传播后被改变或抹平的 heterophily-sensitive residual signal。

在 Squirrel 这类更强异配/复杂局部结构数据上，`z - Pz` 能恢复被平滑传播压掉的判别成分。因此 `[z, z - Pz]` 比单独 `z` 更适合下游分类。

## 下一步硬门槛

1. 复现级评估：
   - 保存 official SP-GCL embeddings 的可复现实验入口，不依赖 `/tmp` 临时路径；
   - 跑 Chameleon/Squirrel 多 seed 或多 official seed embedding；
   - 使用完整 C 网格确认 fast gate 不是 C 选择假象。

2. 机制诊断：
   - 比较 `ssl_prop1/2/3`、`ssl_resid1/2/3`；
   - 分析提升来自哪些 class、degree、local homophily bucket；
   - 证明 raw concat 不等价于 propagation residual calibration。

3. 方法化：
   - 设计无标签 mode selection：何时用 `z`、`[z, P^kz]`、`[z, z-P^kz]`；
   - 或把 propagation-residual branch 写成训练时 regularizer / projector，而不是仅 post-hoc eval。

4. 强基线：
   - 以 official SP-GCL 为 baseline；
   - 后续再补 PolyGCL/HLCL 表格或官方结果对齐。

## 当前裁决

SPARC-GCL 是当前最值得继续的 active candidate。

继续理由：

- 直接建立在已验证强 baseline SP-GCL 上；
- 在 Squirrel 10 split 上有稳定正向信号；
- 创新点更清楚：不是再做 augmentation，也不是简单 high/low-pass filter，而是 strong GCL embedding 的 propagation residual calibration；
- 算力友好，适合 RTX 3060。

主要风险：

- 当前只是 post-hoc representation calibration；
- Chameleon 增益小；
- 需要证明不是 validation/C-grid 偶然；
- 需要扩展到更多数据集和更多 seed。
