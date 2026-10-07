# Object-removal response revision0：完整首例阴性关闭

**关闭固定构造，不扩4/24/600，不改参数，独立方法增量0。** 已暴露固定例 `0_0_72` 的六行原尺寸/1024完整掩码均已封存并评分：全部 IoU0、新增TP0。ΔCLS确实不同于同信息Δpatch，却没有质量收益，按完整错误计数比较：原尺寸ΔCLS新增FP75、删除FP790，弱于同信息all-patch的0/1085和masked-patch的0/1333。[report.json:24-76](actual_single/report.json)

旧定向核对仅读对应源码/报告，未广扫失败账本：旧CLS是FG/BG绝对crop表示，旧1200 GT-box AUROC0.433959保留为严重反证；M4是canonical点贴片配对及环境协方差。二者未计算本项同上下文原/移除单位表示差方向，但这只支持已有re-encoding family的修订，不证明独立性或语义因果。[CLS源码:39-55](../../../../scripts/diagnose_object_cls_dev241.py:39)、[旧结果:1342-1353](../../RESULTS.md:1342)、[M4源码:198-225](../../../../src/ics/methods/pro_paired_environment.py:198)

固定完整合同：整张原R/Q作为共同crop、物理aspect最长边128、中性padding，移除图只把合法区域的原尺寸RGB填(124,116,104)，同几何重编码；原尺寸ROI外逐值不变，resize边界可混合邻域。RFG完整mask、RBG纯BG Ward最多8区；Q固定4/8/16/32四完整Ward，不用GT筛选。实际FP32 Eva `forward_features` 的69×1024 final norm token0与同次patch产生 `d=unit(unit(original)-unit(removed))`；all-patch及同ROI coverage加权patch是强简单控制。差范数≤1e−6/geometryzero明确弃权，逐ROI范数、覆盖与改动像素均保存。本例三个channel全活动、无阈值或geometryzero弃权。[源码:20-81](../../../../src/ics/methods/object_removal_response.py:20)、[源码:115-235](../../../../src/ics/methods/object_removal_response.py:115)、[receipt:42-101](actual_single/receipt.json:42)

margin为cosFG−maxBGcos，完整场为同MEAN+0.05×(四尺度平均tanh(m/0.07)−nativeROI证据)。**推导**绝对改动≤0.1，[0.4,0.6]外coarse token不能翻转；非概率/优化/鲁棒保证。两级renderer为FP32场bilinear到1024>0.5，再binary1024 bilinear到真实原H/W>0.5。`zero-original-CLS` 的原RFG/BG图相同，margin恒0，明确是常数证据控制，不能冒称旧isolated CLS。[源码:84-112](../../../../src/ics/methods/object_removal_response.py:84)、[源码:195-217](../../../../src/ics/methods/object_removal_response.py:195)、[renderer](../../../../src/ics/methods/object_crop_cls.py:123)

**实测成本：** CPU2/8GiB，71逻辑view中12个Q成员重复，实际48唯一Q ROI；57移除图+2原图=59新增128前向。成员复用后114个token-cache请求中55次精确RGB hash复用；两种复用没有重复扣减，旧isolated view未复用。方法41.084s、同MEAN重算1.080s、完整进程51.078s、supervisor53.546s、峰值3,238,912,000 bytes，exit0。首次部署的python别名不存在，只影响未启动的部署；用现存解释器修正后无方法错误/重复前向。[receipt:42-64](actual_single/receipt.json:42)、[receipt:2779-2796](actual_single/receipt.json:2779)、[supervisor](actual_single/supervisor.json)

R640×425→128×85，左padding21；RFG原像素112485，view覆盖4499.4、实际改RGB4727。Q640×640，各ROI view覆盖176–9240、改RGB248–9400。ΔCLS/Δallpatch/Δmasked的单位化前范数范围0.0862–0.9072 / 0.0622–0.8183 / 0.1656–1.1595，均远高于弃权阈值；逐ROI详见[receipt](actual_single/receipt.json)、[计算摘要](actual_single/result_status.json)。

|固定行|原尺寸IoU|新增TP|新增FP|删除TP|删除FP|1024IoU|1024新增FP|1024删除FP|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|MEAN|0|0|0|0|0|0|0|0|
|ΔCLS|0|0|75|0|790|0|205|1826|
|Δall-patch|0|0|0|0|1085|0|0|2568|
|Δmasked-patch|0|0|0|0|1333|0|0|3207|
|zero-original-CLS|0|0|971|0|0|0|2373|0|
|nativeROI mean|0|0|301|0|311|0|700|800|

全场ΔCLS vs Δallpatch/Δmasked最大差0.021999/0.022703；Q差方向夹角余弦均值0.5484/0.2438，证明实测数值不同，**不证明新增类别信息**。48重叠层级ROI的封存后purity rho −0.0295/−0.2657/−0.0680，仅辅助诊断，不是48独立样本。此前isolated revision同例CLS删FP1511、无新增FP，本项未修复旧阴性；五项R/Q输入hash一致，视图构造不同。[评分:115-159](actual_single/report.json:115)、[旧同例评分](../object_cls_revision0_01a1100b/actual_single/report.json:24)、[核对摘要](actual_single/result_status.json)

**合成检查：** 确定性非DINO RGB encoder、patch人为混淆；正例恢复64目标token、删14干扰，完整work TP12996；同类180°负例把原先正确64token全部误删，work TP0。几何零ROI、近零差弃权、aspect/全覆盖/场界通过，不能据此声称DINO优越性。[check.json](check.json)、[检查源码](check_removal.py)

所有六行预测/场、config/source/input hashes与region scores先封存，之后score核hash才读真实原尺寸annotation及既有workGT；无GT选参数/候选，原尺寸GT没有从1024逆插值。完整本地[actual_single/](actual_single/)已有predictions/fields、seal、receipt、report、source config、binding及supervisor；own module/driver和全sealed文件hash已复核。冻结Eva源码hash也与绑定一致，binding历史名称带RGB1024不替代实际128尺寸断言。远端namespace为 `runs/object_removal_revision0_single1_v1/single`，所有owned计算结束。[driver:99-162](../../../../scripts/run_object_removal_response.py:99)、[seal](actual_single/sealed.json)、[核对摘要](actual_single/result_status.json)

**解释与未知：** 这个固定完整读出在首个暴露例无目标恢复，ΔCLS弱于同信息控制，满足预设停止条件；关闭不等于证明所有removal response不可能。单例不估计总体mIoU或泛化，不继续核查/诊断/变体。未改共享算法、PLAN、Git或旧object_cls method.json。

本次Git提交包含方法/检查代码、摘要、评分与receipt/seal；生成的预测、field NPZ及verbose日志仍保留在本地/服务器。
