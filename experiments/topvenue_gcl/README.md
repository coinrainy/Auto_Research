# Top-Venue GCL 工作区

本目录用于从头实现新的图对比学习候选方法，避免继续依赖 patch 第三方官方代码或旧 `grace_idea/` 中累积的失败原型。

当前 active candidate：**Energy-SPGCL**，`GCN-MLP Natural View GCL` 是必须击败的强对照，`ER-Residual-GCL` 已降级为失败/条件性消融。

核心假设：

- 高能量/propagation-residual 表示保留更有效的 positive learning signal；
- MLP/ego 分支用于保留 raw-feature 与 fast-inference 潜力，graph 分支只在训练期提供结构教师信号；
- low-pass positive cache 已在 early gate 中失败，保留为消融而非主线。

最小 smoke：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/topvenue_gcl
bash scripts/run_smoke.sh
```

单次运行示例：

```bash
python train.py --dataset Texas --method gcn_mlp_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method energy_spgcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_residual_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0
python train.py --dataset Texas --method er_cache_gcl --epochs 5 --split-index 0 --seed 0 --shuffle-cache
python train.py --dataset Texas --method grace --epochs 5 --split-index 0 --seed 0
```
