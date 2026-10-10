# 04 数据集差异研究任务

状态：执行中。只写本目录和 `evidence/local/parallel_signal_20261010/04_dataset_differences/`；原始输入只读，数值计算单 CPU，不构建编码器或新 raw，不派生子代理。

## 可回答问题

1. 实际 loader、manifest 和已保存 mask 如何定义 Deep、PACO、LVIS、COCO，以及有可用证据的旧 PASCAL-Part/SUIM 的 query/reference 目标、父对象 crop、尺度与 ignore 区域？
2. 当前同批 Deep100/PACO100 的 APD ridge local4 相对 whole 在困难 ROI 的相反 AUC 方向，是否与合法 reference 可观测量（coverage、纯度、尺寸、形状、连通、parent crop）或既有 query 可观测量相关？哪些只是 query GT 诊断？
3. 数据能支持统一方法处理哪些条件；哪个最小下一方案可在不使用同 query GT 选部署公式的前提下检验？

## 输入与身份

- 主入口：`../INDEX.md`、`../ROOT_RESEARCH.md`。
- 固定逐例信号：`evidence/local/difficult_region_signal_20261010/{raw,existing}/episodes.jsonl`；APD/raw 身份：`raw/sealed.json`。
- 已存在 episode manifest、官方 loader 定义、reference/query mask、已保存完整输出逐例结果；优先元数据和小字段。
- `docs/research/EXISTING_EVIDENCE_AUDIT_20261010.md`、`docs/research/DIFFICULT_REGION_SIGNAL_STUDY_20261010.md`、`evidence/local/RESULTS.md`，及 cache mechanism / joint role pilot 的已有研究 MD。

## 固定对照与统计

- 主比较固定为同 episode 的 `source_apd.ridge.local4` 减 `source_apd.ridge.global`，ROI 分别是 FoRIS 已有前景与 P0 新增区域；保留有效样本数，不能当 mIoU。
- 将 episode 中合法可观测指标与 GT 诊断指标分栏保存。reference 的完整标注可用于方法；query 标签只用于评估与解释。
- 先报告分布和配对均值/中位数，再做少量预先定义的秩相关及跨数据集共同分层，避免按 query GT 搜阈值。探索性统计明确标注为已曝光样本诊断。
- 如有必要，从既有 manifest 对应的 mask 读取面积、连通、bbox 和实例/父框尺度；不扫描全量图像和大 tensor 缓存。
- LVIS/COCO/旧数据集只报告可溯源且同口径的已测结果；缺资产或协议不同明确标注，不混减。

## 交付物

- 本目录 `FINDINGS.md`：阶段性重要事实及待整合事项。
- 本目录 `REPORT.md`：数据定义、证据表、当前统计、限制、可反驳条件和下一可检验方案。
- 对应 evidence 目录：单 CPU 分析脚本、逐例小字段表、聚合 JSON/TSV、输入路径与 SHA256 收据（以实际读取范围为准）。

## 不执行

不重新编码、不修改原缓存或全局 PLAN/state/default 方法；不按数据集名字选视图，不用同 query 标签决定生产公式；不把历史字段上界、token AUC、不同类别聚合的 mIoU 混成端到端收益。
