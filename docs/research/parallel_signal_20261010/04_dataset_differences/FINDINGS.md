# 04 数据集差异：阶段实证

2026-10-10。已完成单CPU读取：6组既有cohort共1,740个episode，3,431个唯一mask路径/hash对逐一验证；当前200的raw小分数字段和既存预测SHA全部核对。数值采集56.97秒，DINO/原O24读取/缓存写入均0。结果位于 `evidence/local/parallel_signal_20261010/04_dataset_differences/{mask_profiles.jsonl,main200_features.jsonl,summary.json,input_receipt.json}`。

## 直接影响主线的结论

**Deep/PACO的相反方向稳健存在，但目前不能用reference面积/纯度或简单query一致性替代数据集名构造可靠视图开关。** 同200的新增区，local−whole APD ridge宏AUC差：Deep +.065235（80/100例正）；PACO −.092367（21/96例正）。已有FoRIS前景内分别+.086266（85/100正）、−.061725（32/95正）。都是fine128分数+canonical1024质量的ROI诊断，不是完整mask收益。

- reference像素质量纯度 `sum(c²)/sum(c)`：Deep中位.5236、PACO100 .9406；但它和新增区局部增益的组内Spearman仅+.016/+.148。混合FG质量占比在合并200中rho+.477，拆开Deep/PACO后+.108/−.138；跨组相关不能当可迁移的选择器证据。
- 同一reference覆盖率<5%的新增区，Deep70例平均+.0744，PACO26例−.0563；覆盖率5–20%为Deep30例+.0439、PACO41例−.1311。共同纯度.5–.9区间也仍相反（Deep56例+.0629；PACO26例−.0812）。这是粗分层，不能认定已完全匹配所有混杂。
- local与whole全图rank一致性中位数近似相同：Deep .8197/PACO .8147。新增区局部优势与一致性的组内rho −.027/+ .004；“越一致越可信”没有这里的收益证据。local seam相对jump虽Deep1.626/PACO3.810，却同样只有组间区别，新增区组内rho−.096/−.060。
- 当前固定16项合法指标里，新增区组内关联没有通过各组/ROI内BH q<.05。**已有前景内有一个可观测候选**：local/whole score标准差比与local−whole AUC，Deep rho+.294（q=.014）、PACO+.444（q≈.0001）。同指标在新增区仅+.038/+.130。因此它只支持后续检查视图响应退化，不能给新增区的部署开关或阈值背书。
- GT诊断有明显关联：Deep新增区query连通数rho+.376、最大连通块质量rho−.379；但reference相应量与FoRIS伪mask相应量没有复现这种关联。不能把query GT形状偷换成合法选择条件。

## 实际协议与形状

- PACO官方loader先选父实例，再union同父同部件annotation，R/Q都按父bbox裁。当前PACO600的R/Q父crop各600；loaded query面积中位19,251px。局部512窗是在这个已裁的父视野上继续切，不能说PACO还没用合法参考crop。
- Deep保留1024²道路场景，自定固定抽样协议；当前100不是官方未公开配对的复现。参考coverage中位2.94%，bbox内部填充4.57%，每例≥90%coverage参考token中位仅3个。LVIS600 reference coverage更小（1.26%），但纯度.8498、bbox填充45.53%，所以“目标小”不足以等同道路薄结构。
- LVIS600真实localCRF区域法52.005，与PACO600无localCRF快版34.971不是同版本。COCO200只有fold0/1、39个观察类；本cohort没有九窗结果，不虚构whole/local比较。
- 旧PASCAL120/SUIM120取已运行explore600的输入；均为legacy PNG pack。PASCAL父crop的原身份未保存，manifest crop=null不等于没裁。SUIM包声明resize过，不能将其与fresh官方原尺寸协议混称。
- 本次PACO600实际观察到264个fold/class，旧评分官方303槽位含39缺类；PACO100仅87观察类。LVIS600为600个观察fold/class，每类一例。不能混用绝对分数或把600例当重复类的通用样本结构。

## 待整合

建议主线继续完成同一reference标定下whole/local/固定融合完整输出；这里没有证据支持加入面积、纯度、seam或一致性数据依赖开关。若使用score标准差比，只可写成已有前景内的探索性可靠性候选，阈值/融合权重不得从这批query标签选择。完整REPORT补充定义、旧数据集边界、父内最终唯一FP与下一可检验条件。

## 主线 candidate_reference_v1 的追加核查

已读取主线新封存结果，独立逐例I/U重聚合与主报告全部一致；代码和额外统计在本线 `crosscheck.py / crosscheck.json`。

**视图排序差与anchored完整输出差并不等价。** Deep `anchor.local4`逐例原图IoU比`anchor.global`好94例、差5例、平1例；最终总I/U差+2.273753点。PACO则51好/45差/4平，最终观察class/fold差−.312216点，远小于困难区的AUC差所暗示的视觉印象。

scene selector在Deep选全whole的55例中，52例的固定local完整IoU其实更高；PACO选全whole86例中43例local更高。这是同批事后诊断，不能取这些GT较优臂做方法。它具体支持：以FoRIS自身伪角色拟合视图权重会偏向重现whole来源；缺少的不是更好的数据集名字代理，而是允许局部纠正原guide错误的可靠性依据。

PACO600的最终唯一原图FP中，完整FoRIS父内比例68.34%、快区域66.99%；PACO100分别62.07%/59.47%。它们与窗口发生量的79.19%不是同一分母。父内非目标确是主要错误来源，但最终快版同时删TP423,241、FP870,408并新增TP83,902、FP317,837，不能写成单向扩张失败。
