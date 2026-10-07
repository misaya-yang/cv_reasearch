# 固定灰度 DINO：预定 FIRST4 完整实测

**四例已完成六行掩码统一封存与评分。灰度主方法原尺寸 mIoU36.979714，高于同MEAN1.284501pp，但低于同信息RGB128匹配0.225398pp、简单亮度纹理0.525776pp及真实H20匹配0.805492pp。** 因此本批没有建立灰度新增路径的必要性或总体质量收益；没有自扩24/600、调参或据首例GT决定扩四例。[完整评分](actual_first4/report.json)、[核对摘要](actual_first4/result_status.json)

**源码绑定：** 首四例封存配置记录的模块SHA256为6cf47db241f9bd8a47dcc9075a5f6c27c2a80007c1b10dc13df4e36993981eb4，runner SHA256为968b701eb5ae086814d67c588c02e53806cbd0c9f26b67b080d3080b0f212464。本仓库当前模块SHA256为62b6fcaff692022d7a2638dcf7922fada482dc60d7bf17a3ee4be63fbc778245，因此该实测不验证当前模块版本；没有当前SHA的运行记录。原配置含服务器路径，仍保留本地。

|固定行|原尺寸class-summed mIoU|1024 mIoU|原尺寸新增TP|新增FP|删除TP|删除FP|
|---|---:|---:|---:|---:|---:|---:|
|同MEAN|35.695213|35.684576|0|0|0|0|
|灰度DINO主方法|36.979714|36.960566|120|225|42|5246|
|同RGB128 DINO|37.205113|37.194597|140|358|36|5382|
|亮度纹理匹配|37.505490|37.465652|0|692|450|6036|
|既有raw H20匹配|37.785206|37.752921|29|651|104|3528|
|cached native原尺寸适配|36.360804|36.316536|582|4962|318|11041|

四例固定为0_0_72/1_0_73/2_0_74/3_0_75，均是已暴露开发例；首例所有行IoU0、未恢复目标。分数是按类累计I/U后取均值，不是独立确认或总体增益。更廉价控制本批分数更高，同时其误删TP也更多，应保留全部四类错误，不能只用删FP包装优势。[逐例与完整错误](actual_first4/report.json)

**固定主路径与区别：** 原R/Q RGB逐像素变换为uint8 `floor((299R+587G+114B+500)/1000)`，重复三通道后进行实际冻结Eva128整图前向，取final-LN patch5:69，绝不取CLS、移除区域或旧crop视图。旧对应核对只发现brightness×0.75压力及RGB瓶颈路径，与这项灰度物理变换不同；有界核对不是全球原创性检索。[灰度/视图源码](../../../../src/ics/methods/grayscaled_dino.py:20)、[实际producer](../../../../src/ics/methods/grayscaled_dino.py:173)、[旧亮度压力](../../RESULTS.md:80)、[旧RGB路径](../../../../src/ics/methods/color_bottleneck.py:97)

整原图物理aspect最长边128，居中同中性RGB padding；灰度仅物理图像像素，padding与RGB一致。完整MR以同resize几何nearest转换，16×16面积形成FG权重，valid content−FG为已知BG权重，padding不当BG。单位patch的加权FG/BG均值产生cosFG−cosBG margin；其类别语义/校准仍未知。Q 8×8 token按 `canvas_x=pad_x+(j+.5)*resized_w/64`、相同y公式，在FP64 grid_sample、align_corners=False/border规则下映到全图64×64再unit，不把8×8直接拉伸忽略padding。实际R resizedHW依次128×85、96×128、128×69、128×96；Q依次128×128、96×128、128×96、96×128。[几何及匹配源码](../../../../src/ics/methods/grayscaled_dino.py:34)、[映射源码](../../../../src/ics/methods/grayscaled_dino.py:86)、逐例receipt留在本地运行输出中

所有生成场均为同MEAN+0.05×[tanh(channel margin/0.07)−tanh既有native q/r参考均值margin/0.07)]；灰度与RGB使用完全相同监督、几何、读出，强对照另含同RGB图像亮度/纹理unit5描述[1,meanL,stdL,mean|dxL|,mean|dyL|]及既有actual-after-block20 raw H20 unit匹配。**推导**每个场对MEAN及gray−RGB的绝对改动≤0.1；无minmax、概率或语义不变性保证。场bilinear64→1024严格>0.5，再binary1024 bilinear→实际原尺寸严格>0.5。native行是既有完整1024二值缓存的第二级renderer适配，不是原native原尺寸CRF原值。[读出源码](../../../../src/ics/methods/grayscaled_dino.py:109)、[完整driver](../../../../scripts/run_grayscaled_dino.py:78)

**表示实测：** 灰度/RGB并非数值相同。四例Q单位patch平均cos依次0.9374/0.8624/0.8930/0.7794；margin最大差0.1381/0.1107/0.03275/0.1869；完整64场最大差0.05882/0.04848/0.02227/0.08663。区别不能当作类别信息或因果去色收益；封存后token-purity相关只作辅助，不把4096插值token算独立样本。[逐例已封存表示量](actual_first4/result_status.json)、[辅助诊断](actual_first4/report.json)

**实测成本：** CPU2、8GiB上限，4例各4次实际128新前向共16，无复用CLS/removal视图；既有M5 H20提取8.862s、新增1024编码0，另列。含模型加载的完整推断进程25.368s、supervisor28.025s、峰值3,131,088,896 bytes，exit0；逐例4.959/4.182/3.875/4.234s。只用46466现存模型，无GPU/下载/新实例。[统一seal](actual_first4/sealed4.json)、supervisor与H20来源绑定收据留在本地（含服务器路径）

**验证与封存：** uint8灰度公式、padding排除FG/BG标签、非恒定坐标ramp的精确映射、相同灰度输入的RGB/gray精确退化、完整渲染与.1界均通过合成operator检查，不是DINO质量证明。实际四例全6行masks、5个生成fields、margins、tokens、receipts先统一seal，score核全部hash/source后才开原annotation与work packet.truth；原GT没有从1024逆插值，也没有GT选候选/配方。评分报告记录运行时source/prediction hash校验通过；但当前module revision与运行module SHA不同，详见上方源码绑定说明。[check](check.json)、[seal](actual_first4/sealed4.json)、source config收据留在本地（含服务器路径）、[核对摘要](actual_first4/result_status.json)

**候选计数决定：** root认可计 **1个distinct candidate implementation**：灰度物理变换进入实际冻结DINO、合法参考patch匹配及完整主读出是此前对应CLS/removal/M4构造未覆盖的路径，不只是控制行、参数值或共享helper。科学原创性未知。实际source/config已冻结临时计数0，保持不改；计数sidecar记录0→approved1及源码SHA/UTC时间，比较行与16views不另计数。[计数决定](candidate_count_decision.json)

完整逐例预测、特征与运行收据仍留在本地；本次提交仅包含汇总报告、路径无关摘要和seal。FIRST4已结束，所有owned计算退出；首四例中被测灰度路径未胜强简单控制，不自动扩大。是否继续由root依据本完整事实决定，单个/四个暴露例不否定整个灰度方向。
