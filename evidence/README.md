# 可随Git携带的研究证据

高价值研究结果、失败原因及理论依据要能在另一台机器独立阅读。这里保留精选原报告/准确摘要、方法规格与必要逐例I/U；完整特征、图像、Mask和反复执行日志外置。

- [results.json](research/results.json)：34家族结论、协议、边界与用途。
- [primary_reports.json](research/primary_reports.json)：主要原报告的完整JSON或明确投影，覆盖所有家族依据。
- [mechanism_reports.json](research/mechanism_reports.json)：可辨识性、source-query oracle、S阶段/负曝光、tangent/context等关键机制与纠错。
- [method_specs.json](research/method_specs.json)：固定构造、控制和实验范围。
- `research/*_iu.jsonl.gz`：COCO4000、S600、N6200、C1和B1共同查询的必要整数。`python -B scripts/read_research_results.py`可无外部资产复算C1/B1。
- [retained_work.json](research/retained_work.json)：正文和源码为何保留，含恢复SHA与并发文件保护记录。

便携JSON包含原路径、原文件SHA、保留内容和省略字段；路径是出处，不要求家里存在这台Mac的目录。它们是历史报告的有界副本，不是新实验或新的完成收据。

`local/`中只保留少量原路径的高价值研究正文与必要producer/冻结源码，避免破坏当前动态依赖；没有恢复GB级预测和特征。小型失败证据不因失败而删除，重复过程材料不因可追溯就全部带回。

[人可读结果](../docs/research/archive/RESULTS.md) · [研究导读](../docs/research/archive/RESEARCH_GUIDE.md) · [跨机器接续](../docs/research/archive/CONTINUING_ELSEWHERE.md) · [清理记录](../docs/research/archive/CLEANUP.md)。
