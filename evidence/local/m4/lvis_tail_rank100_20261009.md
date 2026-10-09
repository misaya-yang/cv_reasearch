# 同100：局部参考排序没有胜过同量幅度控制

CLI局部rank47.187467，相对原MEAN+0.078097，低于中心化同L1幅度控制47.473022、完整FoRIS47.704029和FG-anchor50.047216。原图结论相同，没有完成Strong涨分目标。

|方法|CLI1024|原图|
|---|---:|---:|
|完整FoRIS|47.704029|47.873899|
|原MEAN|47.109370|47.182508|
|graph_only|46.531773|46.627895|
|FG-anchor+原CRF|50.047216|50.182680|
|局部rank|47.187467|47.266615|
|关闭G内参考校正|47.112631|47.185973|
|去均值后同L1 global校正|47.473022|47.551189|

固定同100、原H/A/s及原参考guide。G=(s>.5)是64²token候选集，不使用GT。局部方法仅在G内y=s+.25*(rank_G(g)−rank_G(s))；G外保留原MEAN unary。去参考控制在G内y=s。幅度控制将原G内global校正去均值后放大，使其绝对校正L1等于局部方法；保留原global方向的形状，不clip、不选系数。G<2回退原unary。所有方法同CG与原bilinear>.5，无新增CRF。

G外仅unary逐位保持，图解可以向外传播。局部CDF是经典秩重组，没有增加参考信息。中心化控制既控制幅度，又去掉原平均偏移，避免把恢复总unary质量误算成新参考能力。

局部相对MEAN51涨41跌8平，3折涨7折跌，零交集仍7。相对幅度控制净−0.285554，精确点数：补真+0.019708、补假−0.062219、删真−0.386504、删假+0.143461。幅度控制较MEAN+0.363651，仍低于FoRIS0.231008。局部排序的重组不比简单保留原方向后放大更好。

完整100独立审查通过：actual unary/外G保持/同H/A/最终mask/IU和全部四baseline身份一致；同L1最大误差1.42e−14，净校正近零，差异仅FP32舍入。原parent场与两帧mask全100逐位复现。没有模型、编码、新视图或新query；逐例推断合计3.179秒，单次启动日志创建至完整report为10.301秒，含导入与评分。

代码：`scripts/lvis_tail_rank_pilot.py`、`src/ics/methods/tail_rank_control.py`。完整产物：`cv_data/a/lvis_tail_rank100_20261009/`，保留config/source/producer_snapshots、fields、predictions、sealed、逐例I/U、execution_receipt与reviews。PID68969/session92519已结束。标签已曝光100的开发试验，不支持新样本泛化、扩大样本或独立原创性主张。
