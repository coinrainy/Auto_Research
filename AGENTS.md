# 项目协作记录

## 固定协作规则

- 回答与项目文档尽量使用中文。
- 在当前工作目录内工作，不参考其他目录的项目代码。
- 新项目需要维护 `AGENTS.md`，每次任务结束同步更新当前状态。
- 如当前目录是 GitHub 仓库，任务结束后按用户要求同步上传；当前目录已初始化为 Git 仓库，远程目标为 GitHub 公开仓库 `coinrainy/Auto_Research`。
- 每轮任务结束时给出建议的后续命令。

## 当前状态

- 日期：2026-06-29（UTC）。
- 用户要求：完全忘记此前已经实现过的研究内容，并直接删除大量旧方法路线、旧代码和旧实验结论。
- 当前研究状态：候选方向清空；不再继承旧方法、旧实验结果、旧代码框架或旧论文叙事。
- 保留内容仅限协议层信息：数据划分、评估口径、随机种子记录、metadata、baseline 公平性、GitHub 协作规则等。
- 协议细节见：
  - `docs/gcl_experiment_protocol_checklist.md`
  - `docs/context_reset_protocol_only_2026-06-29.md`

## 已清理内容

- 删除旧 baseline submodule 与第三方 baseline 目录。
- 删除旧实验工作区与旧运行结果目录。
- 删除旧候选方法、旧文献路线、旧实验结论和旧决策备忘录文档。
- 重写 `AGENTS.md`，只保留当前协议和重置状态。

## 后续原则

- 下一轮如果重新构思研究方向，需要先定义新的 research question、目标失败模式、baseline 清单、数据划分协议和最小可发表证据。
- 如果需要写代码，应新建清晰隔离的工作区；不得默认复用旧方法实现。
- 允许复用或重写协议工具，但必须检查其中不包含旧方法假设。
- 旧实验结论不再作为新方向判断依据。

## 建议后续命令

```bash
git status --short
git add -A
git commit -m "chore: reset project to protocol-only state"
git push origin main
```
