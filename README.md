# CVPR 2027：冻结 DINOv3 的单参考分割

输入一张参考图、它的完整 mask 和一张查询图，冻结 DINOv3-L/16，整个方法免训练，输出查询图的完整 mask。目标是在十个公开基准上同协议超过已发表方法。

当前方法 **SH+** 不使用 FoRIS 的分数、投票和聚类先验；新照片留出比完整 FoRIS 高 **+3.13 [2.28, 3.74] pp**。官方协议已完成六域，另四域未完成；最近邻方法的同协议对比仍待补齐。

- [研究总览](docs/research/OVERVIEW.md)：唯一入口，包括方法、证据、进行中的任务、规则和术语
- [工作约定](AGENTS.md)
- [流水记录](docs/research/STATUS.md)
- [论文主张与证据](docs/research/PAPER_EVIDENCE_MAP_20261014.md)
- [实验计划与报告](docs/research/x1012_tasks/)
- [下一步：ISIC 范围读出](docs/research/P1_ISIC_LOG.md)：S 半首轮整数账目与接续问题
- [服务器](server/README.md) · [代码](scripts/README.md) · [证据](evidence/README.md)
- [历史文档](docs/research/archive/README.md)：10-13 及以前的计划、结果账本和旧地图，只为查证据

模型、图像、mask 和生成数据都在仓库外的 `../cv_data/`。

没有公司的模型和数据时，也可用 Python 标准库核对当前结果：`python3 -B scripts/read_sh_evidence.py`。这只重汇总已评分的整数，不运行模型或重新评分确认集。
