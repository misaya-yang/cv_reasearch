# 固定100：匹配背景参考校准

同100完整输出已完成：匹配背景reference在CLI上48.961873，低于其同连续L1校正量uniform49.633312，差−0.671439；原图分别49.109783、49.765735，差−0.655952。相对完整FoRIS虽有+1.257844，仍未超过统一标定或既有FG-anchor，不能将它归因于有效参考关系。

|方法|CLI mIoU|原图 mIoU|
|---|---:|---:|
|完整FoRIS|47.704029|47.873899|
|原MEAN|47.109370|47.182508|
|FG-anchor+原CRF|50.047216|50.182680|
|匹配背景reference+原CRF|48.961873|49.109783|
|匹配背景同L1 uniform+原CRF|49.633312|49.765735|

方法仅替换参考倾向p：保持完整前景kernel cloud，背景以参考FG相关性加权，排除参考自相似diagonal；τ=.07，全部系数与1024 envelope、原CRF不变。连续字段和全部100预测封存后才评分。没有DINO构造/前向，原始输入、FoRIS分数及MEAN输出复用；推理八worker墙钟176.391秒。

相对uniform的CLI逐例43涨、50跌、7平；补TP27390、补FP100018、删TP469、删FP4539。CLI零交集reference6、uniform7、anchor9、FoRIS1。参考保护恢复目标时仍引入干扰，AUC改善不等价于最终净涨分。

产物：`cv_data/a/lvis_matched_background_calibration100_20261009/`中的config、sealed、inference、逐例I/U、report和独立reviews。入口：`scripts/lvis_reference_protected_pilot.py infer|score --matched-background`。旧实验producer保留在旧root/source，防止源码演进混淆历史。

这是旧标签已经暴露、100类别每类1例的继续开发探索。结果不代表LVIS1400或其他数据集泛化，不新增样本、不恢复停掉的6000或官方全量。

独立审查全部100通过，连续预算最大相对误差4.78e−8。reference相对uniform精确CLI点数贡献：补真+0.486167、补假−1.158588、删真−0.003296、删假+0.004277；4折涨6折跌。全部输入、字段、mask、I/U与两帧身份通过。
