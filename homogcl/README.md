# 同配图 GCL / 传播证伪原型

本工作区用于快速证伪和孵化“优先面向同配图”的图对比学习研究想法。当前结论很明确：最初的 `homogcl` 候选没有通过 smoke test，不应继续包装为主方法；当前保留的更有价值方向是把自动多阶传播作为强证伪器，再寻找真正能超过该边界的 GCL 机制。

## 当前方法入口

- `raw`：原始节点特征 + frozen linear probe。
- `prop`：GCN 式训练免费特征传播。
- `propcat`：多阶传播银行 `[X, SX, ..., S^KX]`。
- `autopropcat`：用无标签传播残差平台期自动选择 `K` 的传播银行；当前作为所有 GCL 候选必须击败的强证伪器。
- `grace` / `gracecat`：随机增强 InfoNCE 及其传播拼接诊断。
- `homogcl`：失败候选；同配保真增强 + 多正样本 InfoNCE。
- `horp`：HoRP 教师表示；节点级传播残差门控 + 传播轨迹/残差拼接。
- `horpgcl`：失败候选；HoRP 教师排序 + 多正样本对比 + 相对排序 margin loss。
- `propcca` / `propccat` / `ccassg` / `ccacat`：CCA/去相关类诊断方法；当前未超过传播证伪器。

## 已验证的早筛结论

- `homogcl` 在 Cora/CiteSeer/PubMed smoke test 中未超过 `propcat`，尤其 CiteSeer 明显落后，判定为失败主线。
- `horpgcl` 在 Cora 快速测试中 test accuracy 为 0.796，未超过传播基线，暂不作为主线。
- `propccat` 在 Cora `K=10` 快速测试中 test accuracy 为 0.821，未超过 `propcat/autopropcat`。
- `autopropcat` 使用无标签残差平台期选择传播深度，在单 seed public split 上得到：
  - Cora：0.831，selected `K=6`
  - CiteSeer：0.726，selected `K=6`
  - PubMed：0.789，selected `K=7`
- 这些结果只能作为早筛，不足以支撑 SOTA 或顶会投稿结论。

## 快速运行

```bash
python -m homogcl.train --dataset Cora --method autopropcat --max-prop-steps 10 --probe sklogreg --logreg-c-grid 0.25,1,4,16
python -m homogcl.summarize --input-dir results/autoprop --output-csv results/autoprop_summary.csv
```

完整 smoke test：

```bash
bash scripts/run_homogcl_smoke.sh
bash scripts/run_autoprop_smoke.sh
```

## 当前协议

- 任务：节点分类。
- 主指标：test accuracy。
- 评估：自监督/无训练表征冻结后训练线性 logistic regression probe。默认 `--probe sklogreg` 参考 GRACE/BGRL 的 sklearn `LogisticRegression` + One-vs-Rest + C 网格；`--probe torchlogreg` 参考 GCA/CCA-SSG 的单层 `nn.Linear` + 交叉熵；`--probe ridge` 仅作为普通线性回归诊断，不作为主表默认。
- 数据划分：Planetoid public split，`split_index=0`。
- 标签使用：自监督训练不用标签；标签仅用于 linear probe train/val/test 和 edge homophily 诊断元数据。
- 结果文件名必须包含关键超参签名，避免传播阶数、排序权重等实验被覆盖。
