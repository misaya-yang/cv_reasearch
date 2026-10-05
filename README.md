# CVPR 2027 研究工作区

目标：一篇能在 CVPR 2027（截稿 2026-11-16 AoE）拿到 solid accept 的方法论文。
当前只有一条在做的方向：**单样本上下文分割**（[demo9](demo_lists/demo9_transductive_ics/README.md)），做到哪了以 STATUS.md 为准。

## 从哪里看起

| 想知道什么 | 看哪里 |
|---|---|
| 现在做到哪了、下一步是什么、等你决定什么 | [docs/harness/STATUS.md](docs/harness/STATUS.md) |
| 已确认的数、失败账本 | [demo9 README](demo_lists/demo9_transductive_ics/README.md) |
| 方法和理论 | [HANDOFF.md](demo_lists/demo9_transductive_ics/HANDOFF.md) |
| 还没跑的实验 | [PLAN.md](demo_lists/demo9_transductive_ics/PLAN.md)，Codex 的当日计划在 `docs/codex_plan/` |
| agent 的工作守则 | [AGENTS.md](AGENTS.md)，全局守则在 `~/.codex/AGENTS.md` |
| 已经停掉的方向和原因 | [docs/harness/ARCHIVE.md](docs/harness/ARCHIVE.md) |
| 服务器用法 | [docs/harness/SERVER.md](docs/harness/SERVER.md) |

## 目录

```
AGENTS.md            agent 守则
docs/harness/        STATUS.md、SERVER.md、ARCHIVE.md
docs/reference/      文献表、数据核查
demo_lists/
  demo9_transductive_ics/   主线
  demo4_incontext_seg/      主线依赖的工具库（icx/common.py）和 INSID3 误差账本
```

什么不进仓库只由 `.gitignore` 决定：数据集、权重、特征缓存、第三方源码，以及 `results/` 下文本以外的文件。
已停方向和失败实验的代码在 git 历史里（提交 `b9efc2b`）。
