# CVPR 2027 研究工作区

目标：一篇能在 CVPR 2027（截稿 2026-11-16 AoE）拿到 solid accept 的方法论文。
当前只有一条在做的方向：**转导式上下文分割**（[demo9](demo_lists/demo9_transductive_ics/README.md)）。

## 从哪里看起

| 想知道什么 | 看哪里 |
|---|---|
| agent 的工作守则 | [AGENTS.md](AGENTS.md) |
| 现在做到哪了、谁在跑什么、等你决定什么 | [docs/harness/STATUS.md](docs/harness/STATUS.md) |
| 当前方法、思路和下一步实验 | [HANDOFF.md](demo_lists/demo9_transductive_ics/HANDOFF.md)、[PLAN.md](demo_lists/demo9_transductive_ics/PLAN.md) |
| 做科研的方法（可复用的 skill） | [.claude/skills/measure-first-research](.claude/skills/measure-first-research/SKILL.md) |
| 已经停掉的方向和原因 | [docs/harness/ARCHIVE.md](docs/harness/ARCHIVE.md) |
| 服务器用法、仓库规范 | [SERVER.md](docs/harness/SERVER.md)、[REPO.md](docs/harness/REPO.md) |

## 目录

```
AGENTS.md            agent 守则
docs/harness/        agent 用的操作文档（INDEX.md 是索引）
docs/reference/      文献表、数据核查
.claude/skills/      可复用的 skill
demo_lists/
  demo9_transductive_ics/   主线
  demo4_incontext_seg/      主线依赖的工具库和 INSID3 误差账本（冻结）
```

数据集、权重、特征缓存和第三方源码不进仓库；`results/` 下只保留小的文本结果。已停方向的代码在 git 历史里（提交 `09ea15f`）。
