# 同100参考支持保真：完整结果失败

两臂已封存、评分并通过独立核查。只改变原正响应共识G=(s>.5)&(y>.5)的保真权重：候选Δa=16*d*rank_G(g)，对照在同G使用degree-weighted统一η并匹配ΣΔa。原W、guide、unary、CG及两帧读出保持；H与RHS同步更新。无新模型、图像特征、query照片或参数扫描。

| 方法 | CLI mIoU | 原图 mIoU | CLI零交集 |
|---|---:|---:|---:|
| 完整FoRIS | 47.704029 | 47.873899 | 1 |
| 原MEAN | 47.109370 | 47.182508 | 7 |
| 上轮条件排序图 | 47.701403 | 47.789883 | 6 |
| degree-weighted保真对照 | 45.935585 | 46.034884 | 2 |
| 参考支持保真 | 46.268881 | 46.374766 | 1 |

CLI候选较MEAN−.840490、FoRIS−1.435148。虽然较同量对照+.333296（6折涨、4折跌），整体没有涨分。原图结论一致。

相对MEAN的实际像素编辑为补真291522、补假625189、删真1670、删假3188。精确CLI分数贡献为补真+3.599458、补假−4.451327、删真−.006685、删假+.018064，合计−.840490。零命中7→1没有转化成总体优势：pixel precision .640786→.607350，recall .686082→.725393。这两个面积汇总比例不是class mIoU。

40个面积<1%的query，按原fold/class聚合候选20.142093、MEAN20.424772、FoRIS22.139022；仍比MEAN低.282679。另60例比MEAN低.927406。不能将子组按例数平均还原总体，因为fold/class权重不同。

封存后实际新增像素诊断：补真区域原guide全图rank面积加权均值.799694，补假区域.893074；补真区域约49.9%质量在G内，补假约55.2%，其余来自图传播。这些是实际编辑区域的加权读数，不是AUC或概率；它们支持继续查参考身份不足，不能将高guide直接当正确前景证据。

独立核查全100 G/rank/actual degree/Δa、200个真实H差量与同步RHS、200个场方程、2000份I/U、1600份基线mask、4000份编辑和1000份原图renderer均通过。空/单节点/常数guide/零degree的显式identity检查通过。ΣΔa最大匹配误差3.64e−12；100例中100例的ΣΔa*y都不同，因此不是RHS或逐节点fidelity完全匹配，候选对照差不能独立证明新身份信息。

源码由主代理完成。aidemo、codex_m4；controller PID77992/session16165 exit0，已退出。8worker逐例累计4.373秒，日志创建到report5.533秒（启动/推断/评分），原始模型前向0、推理图像解码0；全100预测封存后读取query GT评分。旧100标签已经曝光，仍为开发探索。

本组件不进入默认方法。不调整强度补跑，不根据减少zero-IU宣布成功，不叠加上轮图门控。下一研究重点是目标/干扰判别信息：仅保住已有高响应会同时保住高guide错误前景。

产物：[report.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_reference_fidelity100_20261009/report.json)、[独立审查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_reference_fidelity100_20261009/reviews/independent_audit.md)、[实际新增区域诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_reference_fidelity100_20261009/added_region_diagnostic.json)、[执行收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_reference_fidelity100_20261009/execution_receipt.json)。
