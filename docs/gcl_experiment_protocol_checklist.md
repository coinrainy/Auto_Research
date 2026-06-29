# 图学习实验协议确认清单

日期：2026-06-29

用途：在重新开始研究方向或实现新方法前，先固定实验协议，避免数据划分、评估流程、随机种子和调参策略不一致造成不可比结果。

## 1. 数据集与划分

- 每个数据集必须明确 split 类型：公开固定 split、随机 split、Geom-GCN split 或自定义比例 split。
- 如果使用随机 split，必须记录生成规则、类别分层方式、`split_seed`、`split_index` 和实际 train/val/test 数量。
- 如果使用公开固定 split，必须记录来源、mask 形状、可用 split 数量和当前使用的 `split_index`。
- 同一主表中不要混用不可比 split；如确实需要混用，应分表展示并明确说明。
- 小数据集必须报告 split-level 波动，不能只报告单次划分结果。

## 2. 随机性记录

- 必须区分 `model_seed`、`split_seed` 和 `split_index`。
- 多 seed 实验不能替代多 split 实验；二者含义不同。
- 每个 run 的输出目录或 metadata 中应写入完整随机性字段。
- 主表建议报告 mean、std、paired delta 和 win/tie/loss count。

## 3. 评估口径

- 必须明确任务类型：节点分类、聚类、链接预测、鲁棒性或迁移。
- 节点分类需要明确是 frozen encoder + linear probe，还是端到端 fine-tuning。
- 如果使用 linear probe，应记录分类器类型、训练 epoch、学习率、weight decay、early stopping 和 validation criterion。
- 自监督训练阶段不得使用测试标签；标签用途必须限于协议允许的 probe、validation、diagnostic 或最终评估。
- 主指标默认使用 accuracy；如使用 F1、AUC 或其他指标，需要说明原因并保持全表一致。

## 4. Baseline 与调参公平性

- 每个 baseline 必须记录来源：官方实现、公开框架、第三方复现或本项目自实现。
- 新方法与 baseline 应使用相同数据划分、相同评估入口和可比调参预算。
- 需要记录调过哪些超参、候选集合是什么，以及最终选择依据。
- 如果因为显存或时间限制修改 baseline 设置，必须记录替代设置和潜在偏差。

## 5. 运行记录

- 每个 run 必须保存 metadata：数据集、split 协议、`split_index`、`split_seed`、`model_seed`、完整命令、依赖版本、git commit/dirty 状态和关键超参数。
- 结果表必须包含足够字段以按 dataset、split、seed、method 对齐。
- smoke test、早筛实验和论文主表实验必须明确区分。
- 汇总脚本应优先做 paired comparison，避免只比较独立均值。

## 6. 新方向启动前需要确认

1. 本轮研究问题是什么？
2. 目标任务和主指标是什么？
3. 使用哪些数据集？每个数据集采用哪个 split 协议？
4. 使用多少 split、多少 model seed？主表如何汇总？
5. baseline 清单是什么？每个 baseline 来源是什么？
6. 是否调参？调参预算如何保证公平？
7. 哪些结果只作为 smoke test，哪些可以进入论文主表？
8. 显存或时间不足时，哪些实验可以降级，哪些不能降级？
