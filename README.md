# CVPR 2027 — single-reference segmentation

当前工作方式：**整合已有有效部件，冻结完整流程，完成有限验证，再据实组织论文。**
新完整流程尚未验证；约 60.1 是工作预测。当前不再自动寻找新机制，也不把原创技巧作为继续交付的前提。

## 入口与职责

Codex 和 Claude Code 共用 [AGENTS.md](AGENTS.md)。先读 STATUS、CLAIM，需要行动再读 PLAN。

| 文件 | 唯一职责 |
|---|---|
| [STATUS](docs/harness/STATUS.md) | 当前阶段、已完成事项和仍缺的事实 |
| [CLAIM](docs/research/CLAIM.md) | 论文目标、已有数值及可以支持的结论 |
| [PLAN](docs/research/PLAN.md) | 唯一待办清单、有限验证及结束条件 |
| [HANDOFF](HANDOFF.md) | 接续入口、已有工作保护和本次变更范围 |
| [SERVER](docs/harness/SERVER.md) | 需要远程执行时查阅的环境记录与操作说明 |

## 代码与证据

- `src/ics/`、`scripts/`：共享数据、编码器、基线、推理与统计实现；[代码导航](scripts/README.md)。
  目录中存在脚本不等于它属于当前方法或待执行清单。
- [本地结果账本](evidence/local/RESULTS.md)：完整结果及失败记录；正文结论绑定各自报告。
- `evidence/local/research_20261005/`、`research_20261006/`：现有实验、理论、预测和实现来源。
- [INSID3 历史](evidence/insid3/RESULTS.md)、[导入研究记录](evidence/dots-2026-10-05/IMPORT.md)：按需查证。
- [历史教训](docs/research/LESSONS.md)、[关闭方向](docs/harness/ARCHIVE.md)：历史背景，不产生待办。

旧状态、旧计划、旧交接和重复导航已收进一个
[整理前快照](docs/archive/2026-10-06-workflow-snapshot.md)，包括整理前未提交的文档修改。
当前入口不再宣称旧 GPU 队列仍在运行。项目 Claude 自动记忆保持关闭；没有另建 CLAUDE.md 或第二套规则。
