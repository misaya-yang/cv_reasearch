# 同100：参考协方差方向没有迁移收益

新判别方向CLI46.427086，低于同精度均值差47.024882、原MEAN47.109370与完整FoRIS47.704029；原图结论相同。该候选按失败实验保留。

|方法|CLI1024|原图|
|---|---:|---:|
|完整FoRIS|47.704029|47.873899|
|原MEAN|47.109370|47.182508|
|FG-anchor+原CRF|50.047216|50.182680|
|FP32均值差|47.024882|47.132327|
|类内协方差判别方向|46.427086|46.512628|

只新增一个candidate。原FP32 O24缓存经原APD处理，q保持原FP16舍入后FP32单位化；完整coverage的FG/BG均值和δ逐位保持。以同μ32中心计算FP64类平衡类内Σ，ridge=trace(Σ)/1024，解(Σ+ridge I)v=δ并单位化。固定原H/A/s、rank系数.25、graph λ16与原CG/双线性>.5；没有新BG筛选、tail/amp/CRF组合或参数搜索。空角色/零δ/零trace显式回退。

源参考正则化判别目标提高不等于可迁移：目标函数相对raw中位3.381倍、最小1.548倍，但查询整体加权token AUC .963065→.943016，实际FoRIS前景mask内 .773993→.738614。小目标组40例中39个有效mask的AUC .816300→.778689。这里AUC是条件有效episode宏平均、不是mIoU；源目标函数也不是参考分类准确率。方向与raw的余弦中位.500，单参考的低方差区别在这些query中损害了目标排序。

相对raw净−.597796，2折增8折降，零交集9 vs6。精确点数贡献：补真+.187600、补假−.283994、删真−1.139833、删假+.638431；去掉背景的收益不足以偿还误删目标。

独立审查通过完整100原始cache/profile/payload/tensor与q/R处理、raw guide/均值/δ逐位重放、同H/A/s、最终mask/IU和四旧baseline。3个Σ独立重建完全一致，100方向方程最大相对残差2.49e−15。原始新view不需要，本次encoder0、推理图像解码0，8worker逐例时间总56.634秒、cov solve总11.468秒；启动日志创建至完整report10.310秒，包括启动与评分。

代码：`scripts/lvis_reference_discriminant_pilot.py`、`src/ics/methods/reference_discriminant.py`。产物：`cv_data/a/lvis_reference_discriminant100_20261009/`，保留covariance、vectors、source身份、fields/predictions、封存与逐例I/U、execution_receipt、source_transfer_diagnostics和reviews。PID70573/session74730已结束，默认入口不变。只在已曝光100上开发，不能宣称未见样本泛化、原创关系或最优LDA。
