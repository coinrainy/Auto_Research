# 项目协作记录

## 固定协作规则

- 回答与项目文档尽量使用中文。
- 在当前工作目录内工作，不参考其他目录的项目代码。
- 新项目需要维护 `AGENTS.md`，每次任务结束同步更新当前状态。
- 如当前目录是 GitHub 仓库，任务结束后按用户要求同步上传；当前目录已初始化为 Git 仓库，远程目标为 GitHub 公开仓库 `coinrainy/Auto_Research`。
- 每轮任务结束时给出建议的后续命令。

## 当前状态

- 日期：2026-06-29（UTC）。
- 当前仓库处于协议保留状态。
- 当前没有已选定的研究方向或方法实现。
- 保留内容仅限项目协作规则与实验协议：数据划分、评估口径、随机种子记录、metadata、baseline 公平性和 GitHub 协作规则等。
- 协议细节见：
  - `docs/gcl_experiment_protocol_checklist.md`
  - `docs/context_reset_protocol_only_2026-06-29.md`

## 后续原则

- 新一轮研究需要先定义 research question、目标任务、主评估指标、baseline 清单、数据划分协议和最小可发表证据。
- 如果需要写代码，应新建清晰隔离的工作区。
- 允许复用或重写协议工具，但需要确认它们只包含通用实验流程。

## 建议后续命令

```bash
git status --short
ls -R docs
```
