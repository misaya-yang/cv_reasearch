# 可随Git携带的研究证据

高价值研究结果、失败原因及理论依据要能在另一台机器独立阅读。这里保留精选原报告/准确摘要、方法规格与必要逐例I/U；完整特征、图像、Mask和反复执行日志外置。

当前 SH 的证据从这里读：

- `research/w30_E1_iu.jsonl.gz`、`w30_E1_report.json`：新照片留出 3,211 例；FoRIS/MFU 的配对整数在 `photo_holdout_iu.json.gz`。
- `research/w30_E2_iu.jsonl.gz`、`w30_E2_summary.json`：官方协议六域 9,459 例。仓库内逐例文件只有 SH/SH+，FoRIS/MFU 的配对基线整数仍在公司资产目录。
- `research/w27_sh_single_line_S.jsonl.gz`、`w28_sh_second_batch_S.jsonl.gz`：两批 S 半的严格单线消融。
- `research/w54_fundus_fresh.jsonl.gz`、`w69_fundus_archive_accounting.json`：细结构异源复测与已有机制账目。
- `research/w58_score_repair_cases.jsonl.gz`：更正后的逐例诊断；旧的表示/分辨率硬上限解释已撤回。

`python3 -B scripts/read_sh_evidence.py` 用标准库复核 W30 点估计和两批消融；`--json` 输出完整汇总，`--isic` 只读取 P1 的 S 半账目。原报告的区间原样引用，不重新估计；不读取模型、图像、查询 GT 或外部路径。历史 C1/B1 的复算仍用 `read_research_results.py`。

10-05～10-13 的研究依据：

- [results.json](research/results.json)：34家族结论、协议、边界与用途。
- [primary_reports.json](research/primary_reports.json)：主要原报告的完整JSON或明确投影，覆盖所有家族依据。
- [mechanism_reports.json](research/mechanism_reports.json)：可辨识性、source-query oracle、S阶段/负曝光、tangent/context等关键机制与纠错。
- [method_specs.json](research/method_specs.json)：固定构造、控制和实验范围。
- `research/*_iu.jsonl.gz`：COCO4000、S600、N6200、C1和B1共同查询的必要整数。`python -B scripts/read_research_results.py`可无外部资产复算C1/B1。
- [retained_work.json](research/retained_work.json)：正文和源码为何保留，含恢复SHA与并发文件保护记录。

便携JSON包含原路径、原文件SHA、保留内容和省略字段；路径是出处，不要求家里存在这台Mac的目录。它们是历史报告的有界副本，不是新实验或新的完成收据。

`local/`中只保留少量原路径的高价值研究正文与必要producer/冻结源码，避免破坏当前动态依赖；没有恢复GB级预测和特征。小型失败证据不因失败而删除，重复过程材料不因可追溯就全部带回。

[人可读结果](../docs/research/archive/RESULTS.md) · [研究导读](../docs/research/archive/RESEARCH_GUIDE.md) · [跨机器接续](../docs/research/archive/CONTINUING_ELSEWHERE.md) · [清理记录](../docs/research/archive/CLEANUP.md)。
