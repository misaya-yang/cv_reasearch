# 01：真实局部观察与融合

2026-10-10。状态：已完成1400份source-arm的同像素核对、困难区诊断、唯一冻结融合及两个控制的完整输出验收；结论和可复核产物见`REPORT.md`。新增规则未通过强控制，不采用。

## 可回答问题

1. LVIS600 有局部 CRF 的区域法、Deep100 原/快九窗的现有正结果，分别来自真实局部支持、query context 组件筛选还是严格多数；需要读取保存的局部 mask/score、coverage 和完整输出检验。
2. PACO 快九窗中“真目标只在部分视图可见”和“相似干扰跨视图重复”的分布是否不同；严格多数擦除的真目标是否具有合法、无需 query GT 的区分证据。
3. 若存在清楚的合法信号，只冻结一个最小融合规则及匹配控制，再在同批完整输出上检验；不扫阈值，不按 query GT 挑选方法，不把 oracle 空间当可兑现增益。

## 已有数据与对照

只读 `cv_data/a` 下 LVIS native600 的三个200例来源、Deep100 原/快九窗、PACO fast9 600、其既有 analysis 与冻结 runner。优先保存的小 mask/score/组件字段，不重编码，不扫描整个 raw 池。完整 FoRIS、未筛选九窗、组件区域法、同 K 几何、MEAN 按实际同批可用性匹配。

原图 class/fold mIoU、canonical1024 像素收支和局部组件发生量分别报告；Deep道路使用总I/U。PACO旧600官方303槽位与新100观察87槽位不能混减。LVIS原区域法有局部CRF，PACO快版无局部CRF，PACO原慢版未测，不能混称。

## 计划计算

- 在覆盖像素中记录原始/接受局部支持数、覆盖视图数、支持比例、最长相邻支持结构；按完整 FoRIS FG/BG、真实 TP/FP/FN 分解，仅在评价阶段使用 GT。
- 比较严格多数丢失的局部并集与跨视图持续干扰，按已有合法分数（source response、组件 query context / reference evidence）报告条件区分力或像素收支。
- 对实际可实现候选保存冻结规则、输出 mask、逐例I/U与编辑账本、源文件 SHA、CPU和墙钟成本。若字段缺失，明确缺失范围和结论边界。

## 资源与交付

所有数值任务1 CPU；必要矩阵最多2 CPU。不创建子代理或进程池，不改默认入口、PLAN/state或其他线文件。任务结果只写本目录 `FINDINGS.md`、`REPORT.md`；量化代码/产物只写 `evidence/local/parallel_signal_20261010/01_multi_observation/`。仅通过读取共享MD交换证据。

参考记忆只用于沿用 `codex_m4`、Strong完整输出及口径区分要求；当前事实以现存资产重核为准（MEMORY.md 18–24）。
