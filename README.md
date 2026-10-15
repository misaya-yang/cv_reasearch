# CVPR 2027：冻结 DINOv3 的单参考分割

输入一张参考图、它的完整 mask 和一张查询图，不训练、不微调 DINOv3，输出查询图的完整 mask。目标是在十个基准上超过 FoRIS。

- [研究总览](docs/research/OVERVIEW.md)：唯一入口，包括方法、证据、进行中的任务、规则和术语
- [工作约定](AGENTS.md)
- [流水记录](docs/research/STATUS.md)
- [论文主张与证据](docs/research/PAPER_EVIDENCE_MAP_20261014.md)
- [实验计划与报告](docs/research/x1012_tasks/)
- [服务器](server/README.md) · [代码](scripts/README.md) · [证据](evidence/README.md)
- [历史文档](docs/research/archive/README.md)：10-13 及以前的计划、结果账本和旧地图，只为查证据

模型、图像、mask 和生成数据都在仓库外的 `../cv_data/`。
