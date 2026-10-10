# 02：目标身份与 query 上下文

状态：研究已完成；见 `REPORT.md`。只写本目录和对应 `evidence/local/parallel_signal_20261010/02_task_identity/`，新增单个候选900例完整验证后关闭，未采用。

问题：PACO600 whole FG/BG 竞争相对完整 FoRIS 的 +1.179400 点，具体来自哪些目标修复与干扰删除？该证据在 COCO200、Deep100 哪些困难区域失效？LVIS query-context presence 的已知正结果是否支持同一种身份机制？

可用输入：PACO/COCO whole-role competition 保存的 source score、h、最终 mask、逐例及逐类 I/U；Deep100 原始与 whole-role 对照；LVIS context/component presence200 的完整输出与区域账本；PACO fast9 的同父部件及 query 伪 BG 根因账本。先验证现存路径、episode/标签/分辨率身份，不以历史描述替代当前文件。

预定实证：

1. 从完整最终 mask/逐例 I/U 重算同批结果与逐类净贡献；统计补真、补假、删真、删假及正负类/episode。
2. 从保存小字段计算 whole h 和 FoRIS score 在已有预测前景、新增及删除区域的区分力，比较同信号跨任务的方向、支持覆盖和代价。GT 只用于事后评价。
3. 将 PACO 修复/伤害对应同父混杂、目标尺度和伪 BG 污染；LVIS 与整图竞争的机制只在资产允许的共同观测下比较。
4. 只有困难区证据支持一个明确的最小合法改进时，冻结该单一规则并完成同条件 mask 检验；不扫阈值、不按 query GT 选规则，不承诺凭诊断必然产生新方案。

口径：原图、canonical1024、token/fine AUC 分开；PACO600 官方303槽位包含缺类；Deep 以道路总I/U；LVIS200 仅当前 cohort。全部已曝光开发样本。CPU 1线程、无新编码/MPS/进程池、raw 池只读。新脚本和数值产物留在本线目录，`FINDINGS.md` 记录关键结果，`REPORT.md` 给可反驳结论及可运行复核命令。
