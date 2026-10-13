# 科研代码：当前执行、完成研究与高价值负结果

保留标准是研究价值及可解释/可复现性。当前执行闭包只是其中一部分；重要失败、理论诊断和成功强控的代码同样随分支保留。完整选择与理由见[保留清单](../evidence/research/retained_work.json)。

|用途|主要入口|状态读法|
|---|---|---|
|当前R1/R2/R3|[r1_reliability_graph.py](x1012/r1_reliability_graph.py)、[r2_within_patch.py](x1012/r2_within_patch.py)、[r3_compose.py](x1012/r3_compose.py)|由当前实验会话维护；不覆盖其修复|
|独立基座完整P0/A/B/C suite|[x1012/](x1012/)、[对应报告](../docs/research/x1012_tasks/)|保留完成研究的完整实现，不重建旧输出目录|
|范围/排序诊断|x1012的d/e/dump脚本|有限经验边界；TOKEN旧总论已被B1限定|
|完整基线与强控|[run_m4_baselines.py](run_m4_baselines.py)、[run_foris.py](run_foris.py)、[run_insid3_complete.py](run_insid3_complete.py)、[MEAN](../src/ics/methods/mean_control.py)|保留精确对照和实际预算|
|任务可辨识性/源空间|[audit_task_identifiability.py](audit_task_identifiability.py)、[task_transfer_source_span_audit.py](task_transfer_source_span_audit.py)|理论与oracle诊断，不能冒充部署方法|
|任务核/context/层级等负结果|[方法库](../src/ics/methods/)、对应run与score入口|固定构造与反例保留，失败不等于整个方向无解|
|S阶段/错误恢复机制|sol_strong、sol_negative、sol_restored等选定脚本|必要的阶段归因；不恢复成批重复verify或旧队列|
|仅用仓库复算核心表|[read_research_results.py](read_research_results.py)|Python标准库，无GT/图像/特征/GPU依赖|
|工作树大产物检查|[check_repository.py](check_repository.py)|包含ignored文件，防GB级产物回流|

实际运行依赖的模型、图像/Mask、raw O24、basis、FoRIS/CRF及平台二进制在仓库外 `../cv_data`。本机环境只是版本快照；跨机器布局、绝对路径绑定和Git同步边界见[接续说明](../docs/research/CONTINUING_ELSEWHERE.md)。

恢复历史源码保留原字节；只有不存在的文件从归档复制，当前R2等已修改版本没有被覆盖。保留旧CLI是为了研究可检查，不保证在缺资产的新机器直接执行，也不自动重跑它的默认输出目录。

R2使用的像素描述子源码在 `evidence/local/autonomous_20261010/physical_pixel_evidence/frozen/diagnose.py`，部分冻结producer也以原路径保留。这里是小型源码，不是恢复大批生成物。新运行输出统一写 `../cv_data/a/<run>/`。
