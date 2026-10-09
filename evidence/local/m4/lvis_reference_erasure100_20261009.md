# 固定100：参考编码前擦除没有带来完整收益

真实FG擦除的最终CLI mIoU为46.560291，低于原MEAN47.109370、原始FG−BG控制47.024882和完整FoRIS47.704029。真实擦除比移位擦除46.335384高0.224907，但不能据此认为该机制有效：简单控制仍更好，新增编码没有换来目标涨分。

|方法|CLI1024 mIoU|原图 mIoU|
|---|---:|---:|
|完整FoRIS（复用）|47.704029|47.873899|
|原MEAN（逐位重放）|47.109370|47.182508|
|真实FG擦除guide|46.560291|46.707378|
|同面积移位擦除guide|46.335384|46.354881|
|原始FG−BG guide|47.024882|47.132327|

使用提前冻结的同100：原1024归一化参考图中的FG擦成归一化0；控制mask沿两轴各roll512，保持面积和形状，记录实际FG重叠而不称完美背景控制。新增视图由同冻结DINOv3-L/16在MPS以FP32 B2编码。原参考与查询的O24缓存复用，保留原APD决策与query坐标；full−view在FP32计算，用未移位的完整coverage对FG/BG加权均值作差并归一化。三guide进入实际父MEAN的同一H/A/s、rank系数.25、λ16与CG；采用原双线性1024>.5，无新增CRF。空角色/零差分有显式回退，不用GT选阈值。

100例、200个新参考视图、100次实际B2前向；原R/Q前向0。原始FP32新视图O24保留在输入寻址cache，后续同视图可复用。推断486.966秒，编码与cache读写合计436.152秒；不称为零模型CPU重放。全部100原MEAN场和最终mask、reference coverage、原R处理都逐位复现。所有预测封存后才评分。源码独立审查通过，完整输出审计另存reviews。

CLI真实擦除相对原MEAN46例涨、47例跌、7平，净−0.549079；相对raw控制44涨50跌6平，净−0.464591。相对raw补TP18106、补FP85819、删TP142103、删FP550380：减少大量背景的同时也丢了更多目标，必须按逐类I/U计价。

封存后的标签诊断仅检查源guide排序，不生成新mask或挑系数。按GT面积加权的token排序宏AUC：parent .953763、raw .963065、真实擦除 .769623、移位 .506884。40个query<1%案例的真实擦除AUC .692334。full−erase在参考中产生的方向，与原始语义方向不同，却更难直接迁移到未经擦除的query；这是该具体差分guide的反证，不是证明所有编码前mask利用都无效。整体高AUC也不等于高分干扰区分或最终mIoU。

代码：`scripts/lvis_reference_erasure_pilot.py`、`src/ics/methods/reference_erasure_guide.py`。完整产物：`cv_data/a/lvis_reference_erasure100_20261009/`中的source/config、raw_cache、fields、predictions、sealed、report、逐例I/U、execution_receipt及source_diagnostics。PID66689/session70922已退出，不重启；没有新增照片或运行其他数据集。这是标签已暴露100上的开发探索，尚无新样本泛化证据。

独立完整审查通过：100个H均独立重建逐位一致，三臂同H/A/s；所有预测、I/U、四项编辑和两帧一致。真实擦除相对raw的精确CLI点数贡献为补真+0.106250、补假−0.223612、删真−4.604178、删假+4.256948，3折涨7折跌。零交集MEAN7、raw6、真实擦除12、shift16；删除真目标的代价未回本。报告见root/reviews/independent_audit.md/json。
