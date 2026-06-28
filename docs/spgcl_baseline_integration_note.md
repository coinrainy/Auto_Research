# SP-GCL Baseline 接入记录

日期：2026-06-28

## 目标

为 Raw-Complement GCL 建立 heterophily-specific GCL 强 baseline gate。当前第一优先级是 SP-GCL，因为其官方实现可在当前环境导入并运行。

## 来源

- 官方仓库：<https://github.com/haonan3/SPGCL>
- 论文入口：<https://openreview.net/forum?id=244KePn09i>

## 当前状态

已在本地克隆官方仓库：

```bash
git clone --depth 1 https://github.com/haonan3/SPGCL.git third_party_baselines/SPGCL
```

第三方源码不提交进主仓库；`third_party_baselines/*` 已加入 `.gitignore`。

## 数据适配

SP-GCL 官方 loader 读取 Geom-GCN 风格数据：

- `dataset/non_homophilous_benchmark_data/{dataset}.mat`
- `dataset/non_homophilous_benchmark_data/splits/{dataset}-splits.npy`

已新增 exporter：

```bash
cd /root/autodl-tmp/Auto_Research/experiments/grace_idea
python export_spgcl_geom_data.py \
  --datasets Chameleon Squirrel \
  --pyg-root ../../data/WikipediaNetwork \
  --out-root ../../third_party_baselines/SPGCL
```

导出结果：

- Chameleon: 2277 nodes, 2325 features, 10 splits
- Squirrel: 5201 nodes, 2089 features, 10 splits

## 本地兼容补丁

当前环境 NumPy 为 1.24+，SP-GCL 官方代码中的 `np.int` 会报错。已在本地克隆中将以下文件的 `np.int` 替换为 `int`：

- `third_party_baselines/SPGCL/data_loader_src/dataset.py`

这是第三方本地工作区 patch，不提交进主仓库。后续若重新 clone，需要重新应用该兼容修改。

## Smoke 验证

命令：

```bash
cd /root/autodl-tmp/Auto_Research/third_party_baselines/SPGCL
mkdir -p results/logs saved_models
PYTHONPATH=$(pwd) python src/main.py \
  --dataset chameleon \
  --neg_selection random \
  --load_params 1 \
  --log_type '' \
  --save_folder logs \
  --reset_epochs 1 \
  --linear_epochs 10 \
  --reset_hidden 64 \
  --reset_seed_num 4 \
  --reset_max_size 64 \
  --reset_subg_num_hops 2
```

Smoke 结果：

- 数据加载成功，使用导出的 benchmark splits；
- subgraph cache 构建/读取成功；
- 1 epoch SSL 训练成功；
- 10 epoch linear evaluation 成功；
- 一键 smoke runner 最新输出 `[Test] Acc Mean=0.2083333284`。

该 smoke 结果不代表正式 baseline 性能，只证明 SP-GCL 官方代码可以在当前项目内跑通。

## 半正式强基线结果

为快速判断 Raw-Complement 是否值得继续作为主方法候选，已在官方 SP-GCL 代码上运行 Chameleon/Squirrel 半正式配置：

```bash
cd /root/autodl-tmp/Auto_Research/third_party_baselines/SPGCL
PYTHONPATH=$(pwd) python src/main.py \
  --dataset chameleon \
  --neg_selection random \
  --load_params 1 \
  --save_folder logs \
  --reset_epochs 100 \
  --linear_epochs 300 \
  --reset_hidden 256 \
  --reset_seed_num 32 \
  --reset_max_size 512 \
  --reset_subg_num_hops 2

PYTHONPATH=$(pwd) python src/main.py \
  --dataset squirrel \
  --neg_selection random \
  --load_params 1 \
  --save_folder logs \
  --reset_epochs 100 \
  --linear_epochs 300 \
  --reset_hidden 256 \
  --reset_seed_num 32 \
  --reset_max_size 512 \
  --reset_subg_num_hops 2
```

结果：

| Dataset | Test Acc Mean | Test Acc Std | Last Acc Mean | Best Acc Mean |
| --- | ---: | ---: | ---: | ---: |
| Chameleon | 0.557456 | 0.028776 | 0.558991 | 0.559868 |
| Squirrel | 0.360423 | 0.018587 | 0.361287 | 0.363016 |

Raw-Complement 最终候选均值：

| Dataset | Raw-Complement F1Mi | Raw-Complement F1Ma |
| --- | ---: | ---: |
| Chameleon | 0.494664 | 0.489314 |
| Squirrel | 0.341210 | 0.333616 |

解释：

- 这不是完整官方复现实验，但已比 smoke 更接近有效 baseline；
- 指标格式不完全一致，SP-GCL 输出 accuracy，Raw-Complement 汇总为 F1Mi/F1Ma；
- 在同一批导出的 Chameleon/Squirrel benchmark splits 上，SP-GCL 已明显高于 Raw-Complement；
- 因此 Raw-Complement 未通过当前 strong baseline gate，应降级为机制/负结果资产。

## 下一步

1. 若后续需要论文级对照，再写正式 SP-GCL runner，固化日志解析与配置记录。
2. 当前研究决策上，Raw-Complement 已经不值得继续作为主方法微调。
3. 下一轮应优先搜索或设计能正面超过 SP-GCL/PolyGCL/HLCL 的新机制。
