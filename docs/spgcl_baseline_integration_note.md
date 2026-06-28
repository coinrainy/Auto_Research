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

## 下一步

1. 写正式 SP-GCL runner，记录 dataset、seed、epochs、linear_epochs、hidden、max_size、subg hops 与输出日志路径。
2. 先跑 Chameleon/Squirrel seed0 正式或半正式配置。
3. 若 SP-GCL 正式结果显著高于 Raw-Complement，则当前 idea 必须降级或重新设计；若 Raw-Complement 仍有优势，再补 PolyGCL/HLCL 对照。
