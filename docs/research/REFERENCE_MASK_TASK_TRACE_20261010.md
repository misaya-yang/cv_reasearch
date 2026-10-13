# 固定图像、切换真实参考 Mask：任务信息在哪里丢失

用户最新研究问题：固定同一查询图和参考 RGB，只改变参考 Mask，内部判断如何随指定任务改变，首次失真在哪里，是否存在跨数据集重复的机制。先定位，再以定点干预验证；不把数值变化或单域涨分当原创理论。

已有普通 joint COCO150/LVIS150、共同 APD300 和 mask 编码条件探针40均未获得预定的目标身份判别收益。150例用于检验编码假设，当前不继续机械扩大。完整情况见 [OPUS55_STATUS_20261010.md](OPUS55_STATUS_20261010.md)。reference 放大强读出仅有规格，尚未运行，现由本追踪优先。

## 冻结的最小真实实验

程序：`scripts/opus55_reference_mask_trace.py`。资产：`../cv_data/a/reference_mask_task_trace_20261010/`。COCO、PACO-Part、SUIM各20对，60张互异域内查询 RGB；每对固定 R/Q RGB、既有 crop、DINO O/24、预处理和完整 FoRIS。A是原任务，B是同一参考与查询中均实际存在的另一类别或同对象兄弟部件。按既有 manifest 顺序取首20个合格查询，B在共享真实标签中取参考面积最接近A者，完全不按结果选例。GT只用于准备阶段验证真实任务身份与非空互斥区域，以及全部轨迹封存后的评分。

SUIM匿名 PNG 已与原数据集 RGB逐像素 hash匹配，并核对原任务二值 mask完全相同，随后才取另一原生类别 mask。PACO固定原父对象与 crop，不移动到另一个对象。COCO使用原80类 indexed annotation。没有新照片或新的编码。

每个真任务另有一个原始面积完全相同的半宽/半高循环平移 mask，共四变体A/B/shiftA/shiftB。平移变体是错支持的阴性对照，不是具有合法 query GT的第三、第四任务。其目的为区分正确参考支持与面积/位置刺激。它不是纯面积匹配的随机化试验，保留平移重叠及控制局限。

## 逐步记录与判据

使用未经修改的官方 FoRIS与只读 return-frame taps：原始O24、实际APD门控、最近邻和stage2实际mask token化、FG/BG原型、FG匹配、BG抑制、候选投票、query种子聚类先验、语义惩罚、聚类修正、归一化阈值、原二值化、原CRF完整掩码。原任务A有现存score时要求逐位相同。DINO不存在实例和兜底，缺失/损坏 raw直接失败。只缓存本对两张 raw数组，四变体共同复用。

重要源码事实：APD/特征门控用nearest参考mask；stage2、候选投票和种子先验用bilinear threshold，并在空前景时fallback。这些路径不能被混称成同一token mask。raw阶段诊断为相同原型分类器作用于APD前特征，不是官方另一个顺序prefix。

主要指标是任务切换AUC：A场在查询A独占区域相对B独占区域的排序AUC，与B场反方向排序AUC的平均。相同场在两个任务中固定不变时，该值严格为0.5；因此不会把“看见了同一显眼物体”误当跟随mask。每个原生64×64 token按1024评分帧中两独占区域像素质量加权，另存二值化/CRF全分辨率AUC与每任务I/U。参考内部的同类任务区分、原型余弦、两种mask碰撞、APD切换、query聚类同一性和面积平移对照也保存。每域以20个不同查询图为配对bootstrap单位，报告2000次固定seed0区间。

AUC只检验两个真实任务的区分，不能替代背景精度、完整mIoU或官方全基准。独占区域以外的像素仍计入完整I/U。样本及原任务分数已曝光，本轮是机制开发，不能称独立确认。

## 什么证据才支持机制

首先区分三个位置：参考mask是否在token化后不可区分；参考原型仍能分清R上的两个任务却不能迁移Q；Q上的任务证据存在，但投票、query聚类或最终读出造成确定的下降。阶段间下降只给定位候选，不能等同不可恢复的信息损失。

对跨域重复的下降算子，下一步在同一输入上做唯一的移除/替换干预，冻结其他算子及评价：检查任务切换AUC是否恢复、背景误报是否代价过大、完整掩码是否改善。若仅原任务A受益、B失效或平移对照同样受益，拒绝“理解指定目标”的解释。机制需在未用于选择的缓存照片中重复，并保留反例；目标最终仍是多个数据集的完整结果稳定超过FoRIS。

## 执行与交接

```sh
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/bin/python scripts/opus55_reference_mask_trace.py prepare --per-dataset 20
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/bin/python scripts/opus55_reference_mask_trace.py run --workers 4
```

准备已完成：COCO检查38个既有图对取20；PACO检查29取20；SUIM检查30取20。`config.json`与`prepared.json`固定源码、parent manifests、权重profile与顺序；`pairs.json`是无query label的推理输入；`query_truth.json`/`query_labels/`隔离评分标注；`inference.jsonl`、`traces/`、`sealed.json`与`diagnosis.json`分别保存执行、完整场、封存与评分。当前只使用CPU缓存重放，新增DINO编码0，不需要训练头授权。按真实PID、记录增长和hash核验完成，不能仅凭退出或文件名宣称成功。

## 首轮完成与唯一干预

60对、240完整轨迹已封存，COCO/PACO原任务A共40个score逐位复现；240变体均无query label访问。整个trace仅51.10MiB，没有新raw。初次评分遇到1个PACO参考mask包含关系导致参考独占区AUC未定义，已将该辅助指标标为缺失，主任务60对均保留；原推理源码封存于`frozen/trace_source_v1.py`，修复只作用于post-seal评分，没有重复推理。

| 域 | APD FG/BG | 加投票 | 加种子聚类 | 加语义惩罚 | 加delta | 二值化 | CRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| COCO | .9042 | .9138 | .9248 | .9236 | .9180 | .8643 | .8620 |
| PACO-Part | .8911 | .8960 | .8623 | .8642 | .8599 | .8110 | .8117 |
| SUIM | .8423 | .8474 | .8702 | .8676 | .8653 | .7898 | .7819 |

这些是任务切换AUC，不是完整mIoU。真mask最终score相对等面积平移控制的AUC优势为+.4736/+.5398/+.4549，说明FoRIS确实利用正确参考支持，不能概括成“完全不理解或不响应Mask”。60对没有两种真实mask的hard-token碰撞。query种子聚类在57/60对相同，语义聚类55/60对相同，但同一partition不自动等于有害：种子先验在COCO/SUIM平均有益、PACO有害。

delta的连续任务AUC平均变化三域均负：−.00562/−.00424/−.00229，只有COCO的配对95%区间排除0。因此选择一个可证伪的因果实验：仅移除末尾additive delta_map，冻结全部上游和原renderer，不扫权重，执行同60对的A/B完整mask。入口`scripts/opus55_mask_trace_followup.py`，资产`semantic_delta_followup/`。保存完整误报/漏检、两任务均值IoU和task AUC，完整IoU若不恢复就拒绝“移除即可解决任务损失”。选择已用这60对结果，不能称独立确认。

另在该干预预测全部封存后计算阈值诊断：区分minmax64排序、bilinear1024连续排序、原>.5二值化与CRF；用GT计算最佳二值任务AUC上界，只有上界诊断意义，不能取其阈值推理或声称可部署涨分。二值化必然降低排序表示能力，只有原阈值明显低于该二值上界，才支持继续追查校准，而不是泛称“minmax丢失任务”。

## 完成的反证与已知条件通道消去

去delta完整两任务均值IoU点差COCO−.772、PACO−.372、SUIM−.265，配对区间都跨零，不采用。原renderer重放起初漏了目标tensor批次维度，被native CRF逐位校验拒绝；57个未封存、未评分错误输出保存在`semantic_delta_followup_rejected_v1/`，修正后的三域六个native CRF掩码逐位一致。原240完整轨迹不受此错误影响。

minmax64排序AUC差严格为0，插值平均改善.0132/.0058/.0063；GT最优二值任务阈值相对原阈值的AUC上界差为.0717/.0736/.0831。然而这些最优任务阈值忽略其余背景，FP/GT从.294/.508/.252扩大到1.057/.931/.548；完整pre两任务均值IoU变为49.144/48.245/56.827，原为54.776/46.772/59.562。故不能把两目标身份切换AUC改善当完整分割成功。GT最优完整IoU阈值的上界分别66.255/59.094/67.622，仍不可部署，已有PACO参考定阈值失败不被推翻。见`readout_audit/`。

三角色分析将当前目标、另一个真实目标和其余背景区分：误报中另一个真实目标占11.1%/30.9%/37.1%，大部分误报是其余背景。最终场在原CRF mask内部的FG/FP AUC为.718/.741/.696，共同有效照片19/18/19，不能用全图双目标AUC替代。Query种子cluster-constant GT任务oracle为.978/.755/.918，PACO明显有部件混合瓶颈，但不能据此宣称所有域都是同一聚类问题。见`role_audit/`，其中其余背景包括所有非目标、非另一个任务像素，不能概称未见类别。

已知标量gate消去机制在三域真实Mask切换中完成因果重放。Mask改变的gate RMS为.188/.170/.169；正标量gate进入unit normalization后方向最大差仅3.9e−7。默认source stage2/3打分、聚类用raw-query而跳过gate；stage4逐token normalize，满足`normalize(g*x)=normalize(x), g>0`。去gate后的score、query semantic cluster、pre和CRF均在118/120任务逐位相同；PACO剩余一任务差252个最终像素，SUIM差11个，完整任务均值IoU变化+.000884/−.000728点，是float32聚类的微小舍入差。此机制原来自`MEAN_SIMPLIFICATION_20261009.md`的五例LVIS验证，本轮仅扩大验证，不包装成新发现或新算法。它指出一条有效条件通道被消去，而其他reference prototypes与投票仍保留Mask信息。见`gate_channel_audit/`。

## 直接对应的单一方向度量探针

从消去机制推出两个可检验必要条件：条件化必须改变单位化后的方向，且真实打分/聚类路径必须消费该变化。为此仅新增一个固定算子、一个matched control：源APD后，对R/Q共同应用`normalize((I+(sqrt(2)-1)*d*dT)x)`。该PSD度量把一方向的平方权重增至2；不扫强度、不改变阈值或CRF。

Task轴为`d=normalize(mu_FG-mu_hardBG)`，使用当前合法reference mask和原source20%难背景均值；matched轴为全reference mean，不用mask构造轴，成本和强度相同。源APD及其少数mask依赖gate保持，两臂共享，因此“无mask”只描述新增axis而非整个pipeline。没有跨episode训练，没有全局学习头，也没有query GT输入；不依赖未回答的训练口径澄清。

入口`scripts/opus55_task_metric_probe.py`，资产`directional_metric_probe/`。固定同60对、两真实任务、两轴，原完整FoRIS直接复用，共240个新的CPU缓存完整前向、零DINO编码。必须同时胜native与matched的完整IoU；仅有方向响应或身份AUC提升不能通过。该样本已用于机制选择，正结果也须在未用于选择的缓存照片确认。结果未知时不预设采纳。

该探针已完成：COCO两任务完整均值IoU55.854→56.603（+.749），PACO47.682→50.807（+3.124），SUIM60.239→58.252（−1.987）。同强度whole-reference-axis为55.478/41.766/61.095，task axis相对matched +1.125/+9.041/−2.843。查询单位方向RMS确实变化.0017量级，但跨域完整分数未成立，不推广、不扫强度隐藏失败。该均值是两任务／照片配对，不是官方class mIoU。完整bootstrap见目录REPORT/diagnosis。

reference label/feature partition与query membership的两个后续检验现已完成，结果见下文。不按数据集名加selector，不从query GT选阈值；机制和算法都还没有达成主goal。

推理原版和只用于post-seal评分修复的base helper分别保存在`frozen/trace_source_v1.py`与`frozen/trace_source_postdiag_v2.py`。之后增加的“新cohort自动存snapshot／已封存不重推”仅修复后续流程，不改变本次任一推理产物。独立验算入口`scripts/verify_opus55_mask_trace.py`与`verification.json`，覆盖hash、真实像素I/U、gate微小差、oracle上下界及40个原score身份。

家中Opus无需先加载大型raw，即可读取仓库内`evidence/local/false_alarm_floor_20261010/execution/reference_mask_trace_snapshot.json`，包含60对身份／原cache绑定、全部逐阶段与逐任务结果、反证和误报分解，完整大数组保留在cv_data。快照由`report_opus55_status.py`更新，大小以实际文件为准。[机制总览图](../../evidence/local/false_alarm_floor_20261010/execution/reference_mask_task_mechanisms.png)与生成脚本`plot_opus55_mask_trace.py`也已保存；该图仅覆盖首轮与早期干预，后续以本报告与JSON为准。所有结论仅限已曝光开发样本与所述算子通道，尚未形成可推广的原创算法。

## 参考拟合与查询聚类：完成的两个检验

`reference_partition_audit/`仅用合法R mask与既有O24，计算原cosine average-linkage在tau=.6的参考语义分区。它是新诊断分区，不能冒称原FoRIS用该reference partition作判决。gamma为FG token所在cluster中的平均BG比例；`E Var(M|C)`是仅用cluster ID重建mask的最小平方误差。这是已知条件期望恒等式，不是原创定理。descriptor先封存，再关联既有query结果。

| 域 | 原gamma | 任务轴gamma | 匹配轴gamma | 原gamma与query收益rho | R拟合改善、Q IoU恶化 |
|---|---:|---:|---:|---:|---:|
| COCO | .4110 | .3342 | .5422 | .128 | 13/40任务 |
| PACO-Part | .6370 | .4447 | .7902 | .065 | 15/40任务 |
| SUIM | .3128 | .2725 | .4746 | −.089 | 21/40任务 |

task轴在三域均显著改善这个参考拟合指标，却没有稳定迁移收益。49/120任务是直接反例；不从跨域均值推导reference自适应强度或事后selector。面积匹配随机mask的精确期望gamma与平移控制均保留，不能把小mask天然更低混合当作语义适配。

`query_partition_intervention/`只把task轴下的两种query cluster membership固定为原native分区，重新计算方向条件化后的cluster均值、affinity、seed权重及修正。两臂FG/BG/vote逐位相同；原task-axis的120个score/pre/CRF与父实验全部逐位一致，固定partition也逐位等于原分区。全部轨迹先封存再评分，零新增DINO。

| 域 | 原FoRIS两任务均值IoU | task轴 | 固定原query分区 | 固定分区−原FoRIS及95%区间 |
|---|---:|---:|---:|---|
| COCO | 55.854 | 56.603 | 56.283 | +.429 [−.968, +1.809] |
| PACO-Part | 47.682 | 50.807 | 51.463 | +3.780 [+.179, +7.686] |
| SUIM | 60.239 | 58.252 | 58.850 | −1.389 [−3.333, +.459] |

改变membership的因果增量task轴−固定分区为+.320/−.656/−.598，区间均跨零。PACO收益不需要改变query grouping；SUIM损失也未因固定partition而消失。PACO固定分区的任务切换AUC反而比完整task轴低，进一步说明双目标AUC不能代替含背景的完整IoU。label编号不影响的ARI审计见`label_invariant_partition_audit.json`，不用label数组不同就断言实际partition改变。

## 排序与分数分布交换：已完成

入口`scripts/opus55_copula_readout_intervention.py`，资产`copula_readout_intervention/`。从原native及固定query分区的task-axis最终score取两场，各自沿原minmax/bilinear到1024后，再做两种交换：原空间排序＋候选分数分布；候选空间排序＋原分数分布。随后直接`>.5`和原CRF，不再次归一化。这样可拆分空间身份排序和阈值下的预测面积，候选选择依据是去除membership混淆，不是事后最高分。

相同source分数采用平均rank，同值仍同值；ties会使目标直方图／预测面积不能完全相等，逐任务保存误差而不按像素位置破tie。原120个pre全部逐位校验，三域首对的native与candidate CRF共12个mask另行重放校验。只读封存score及query RGB，DINO与raw读取均0。全预测封存后计算TP/FP/FN、pre与CRF面积、照片配对IoU区间和原任务A／辅助B分开结果；不把诊断交换本身当可部署算法。

| 域 | 原FoRIS | 固定query分区候选 | 原排序＋候选分布 | 候选排序＋原分布 |
|---|---:|---:|---:|---:|
| COCO | 55.854 | 56.283 | 56.679 | 55.546 |
| PACO-Part | 47.682 | 51.463 | 50.724 | 47.809 |
| SUIM | 60.239 | 58.850 | 59.272 | 59.905 |

这些仍是完整CRF后每照片两任务均值IoU。PACO候选+3.780，但原排序＋候选分布可得+3.042，而候选排序＋原分布只有+.126 [−.233, +.595]。因此当前PACO收益大部分无需更换空间排序便可产生，不能归为新增语义身份理解。不是精确可加归因：CRF与IoU存在非线性，三个点差不能相加分摊。COCO和SUIM的点差也主要随分布变化；pre面积匹配误差最多2/4/1像素，远小于实际面积变化。全目录约4.15MiB，控制器已退出且sealed/diagnosis齐全。

## 逐阶段原极值桥：已完成

`prefix_range_audit/`保留候选所有连续场和原query membership，仅把各阶段min/max固定为对应原native场，再沿原bilinear及>.5读出。六个prefix的pre是假想诊断mask，不是原网络中间mask；最终score桥才执行原CRF。原native/candidate score-pre和CRF均逐位等于父产物。60对全预测先封存再评分，新增DINO和raw读取0。

| 域 | 原FoRIS完整IoU | 候选自极值 | 候选＋原极值 | 原极值桥−原FoRIS及95%区间 |
|---|---:|---:|---:|---|
| COCO | 55.854 | 56.283 | 54.554 | −1.301 [−4.318, +1.014] |
| PACO-Part | 47.682 | 51.463 | 53.329 | +5.646 [+1.621, +10.345] |
| SUIM | 60.239 | 58.850 | 56.997 | −3.242 [−6.396, −.469] |

固定极值不能修复三域，还显著扩大SUIM损失。FG-only的自极值pre在COCO/PACO已有+3.097/+4.236点变化，后续BG和vote在COCO削弱这项收益，PACO seed-prior前后则改变覆盖范围。这个轨迹只定位各算子的读出效应，不证明FG、BG或prior是统一的首次不可恢复信息损失位置。报告按冻结input manifest顺序作bootstrap，不把不同prefix中的最高分挑作方法。

## 同一FG/BG判据在参考与查询：已完成

`role_transfer_audit/`仅分析封存native场和隔离GT，没有新推理、raw读取或编码。R按source实际hard mask64评分，Q按1024目标/非目标像素质量加权；这里的非目标包含所有其他像素，区别于主双任务独占区域AUC。

| 域 | R FG AUC | R FG/BG AUC | Q FG AUC | Q FG/BG AUC | Q背景方向AUC<.5 |
|---|---:|---:|---:|---:|---:|
| COCO | .9938 | .9970 | .9473 | .9666 | 3/40任务 |
| PACO-Part | .9793 | .9884 | .9121 | .9169 | 8/40任务 |
| SUIM | .9724 | .9874 | .7623 | .8802 | 5/40任务 |

参考上的自洽判别不能保证迁移。背景抑制平均仍提升query角色AUC，尤其SUIM，不能从16个异常任务推出整体删除该项。hardBG与FG的原cosine中位数为.683/.892/.761，正交残差norm中位数.730/.451/.649；单位化增益与query背景项效果的域内rho为.198/−.164/−.035，未支持“残差放大是统一失真原因”的解释。R背景margin为正而Q反向只有2/5/4任务。方向反转和相关性均是diagnostic，不能直接推广成算法或原创定理。

当前持续研究须满足三层要求：有效参考条件通道被实际打分消费；改进保留在固定原分数分布的空间排序检验中；再用完整含背景输出验证迁移。已知scalar gate消去可精确定位，但方向metric未带来统一空间判别收益，reference fit与统一极值校准都不足。主goal保持进行，任何单批DIAGNOSED均不是主goal完成；后续优先检查现有真实参考观察能否改变有效任务判据，而不是再扫相似度权重或扩失败样本。

## 2026-10-11：机制证据与当前九窗研究的关系

当前原始问题仍是固定R/Q图像、只切换参考Mask后的有效任务判断。冻结的无Mask查询编码本来就相同；应检查参考条件如何进入实际查询读出，及其对当前任务、另一个任务与其余背景的区分。原60对的240轨迹和因果干预已完成，150例早期编码假设不继续机械扩大。6380参考校准也已完整验收；当前只在已有真实九窗缓存内检验一个固定参考判据在不同查询观察中的表现，不新增编码、下载或数据队列。

|位置|现有确定证据|能推出什么|保留的反证或缺口|
|---|---|---|---|
|条件gate到单位方向/实际打分|正标量满足normalize(gx)=normalize(x)；原默认路径还有raw bypass；去gate118/120任务最终mask逐位不变|这一条件通道几乎未改变实际查询判断，是有因果重放的局部通道消去|prototype/投票等其它参考通道仍存在；60对均有正确Mask相对平移控制的响应。不能称整个FoRIS在此失去所有任务信息，也不能将已知归一化恒等式当原创算法|
|单参考拟合到校准支持|质量采样取走特定小目标99.240%–99.875% FG质量，剩余分离度为负时调用者清零新增H；query ridge场本来仍非恒定|具体子类在校准处丢失了已有新增证据，而非冻结编码完全不可表达|同样低剩余支持仍有大量正分离度参考。固定6380逐删干预未稳定跨域改善；不能凭参考自拟合分离度推广修复|
|参考角色分布到查询绝对坐标|既有四域600例中，参考自身FG/BG坐标均值为1/0，查询FG均值仅.547–.814、BG均值.127–.197；条件角色分布不保持|此原型坐标不能仅靠未知FG比例解释查询差异；高排序AUC与准确绝对零点仍是不同要求|它是这个坐标的必要假设失败，不能外推为所有任务读出不可迁移或任意校准都失败；oracle角色标签只用于诊断|
|真实局部提案到严格多数|在4覆盖区，真实支持1–2票被映成BG；同时四窗一致接受仍有大量FP，三域均重复|多数算子丢失了已有提案支持，接受一致性也不足证明任务身份|这是输出融合的确定映射与开发数据中的任务误差，不证明其它窗原生特征没有任务信息，也不独自构成新身份算法|
|whole任务场到争议区动作|已存场1400 source-arm审计完成；full-reference H正号若直接删M，会损失LVIS/Deep快/PACO已有TP46.11%/39.96%/50.30%|整图参考场具有一定排序，却不能用现零点同时保护九窗真覆盖和拒绝干扰|正分离度的校准仅改变同一ridge场的仿射零点/截断；新的query排序必须来自真实不同观察或不同有效信息，不由校准本身产生|

因此，“某分支在某步对Mask不再敏感”与“最终任务信息不可恢复”必须分开：下游仍可再次读取参考prototype或其它参考条件，不能把有旁路的计算图当成单链，再由一处响应下降推出全模型的信息论结论。现有证据还未给出跨数据集统一的首次不可恢复位置；去delta、方向度量、统一极值和校准修正的完整反证均保留，不再重跑。

角色分布600的闭合恒等式为`π_coordinate−π_GT = π_GT(E_QF s−1)+(1−π_GT)E_QB s`；大量背景的小偏移足以造成显著FG比例误差。该结果不能被改写成“参考信息完全消失”，也不能用错误的纯label-shift前提给新组合选择query阈值。真实R-focus200的既有equal配方也已失败，PACO匹配插值正差并未伴随更好的排序；此旧版与当前固定whole头/真实九窗不同，不以它替代此次诊断，也不直接重演。完整证据见[REFERENCE_QUERY_MIXTURE600_RESULTS.md](autonomous_20261010/REFERENCE_QUERY_MIXTURE600_RESULTS.md)和[REFERENCE_FOCUS200_RESULTS.md](autonomous_20261010/REFERENCE_FOCUS200_RESULTS.md)。

当前新增诊断严格保持旧whole-R ridge头/APD/校准，只将Q替换为缓存中的真实九窗观察；逐例复原whole场<1e−12后，记录接受支持窗的最大H及其争议ROI表现。原M、M超出whole-guide的独有TP、少数支持、四票一致区域都保留；GT仅用于固定诊断，候选公式仍未冻结。规格在[NINE_WINDOW_COMPOSITION_20261011.md](NINE_WINDOW_COMPOSITION_20261011.md)，完整结果由该规格和信号报告接续，不将区域AUC当Strong成绩。

主Sol另核对1300条执行输入：LVIS600/Deep100/PACO600同R/Q RGB、crop、尺寸且异R-mask的组均为0；PACO570张照片对应580种实际Q输入和595种R/Q输入。该普通九窗benchmark可以检验跨域观察/融合假设，Mask-only因果证据继续使用本报告的三域60对，不能将两种样本范围合称一次任务切换实验。无GT输入覆盖收据在[nine_window_mask_switch_coverage.json](../../evidence/local/sol_validation_20261010/nine_window_mask_switch_coverage.json)，6380完整控制与独立验收在[REFERENCE_CALIBRATION6380_RESULTS.md](../../evidence/local/sol_validation_20261010/REFERENCE_CALIBRATION6380_RESULTS.md)。尚未验证原创算法或多数据集完整结果稳定领先。


## [主Sol全量证据] 真实九窗同参考头1300诊断及配对区间完成

1300真实R-head复原、11655次真实窗读出已结束（五个LVIS无正校准参考不作无效窗计算），完整whole ridge误差≤5.11e−15、旧cal误差0；86875计算检查及独立静态来源/标量验证通过。成功墙钟1862.58秒、CPU4520.92秒，峰值RSS1.586GiB，14255次既有payload加载213.99GiB；0DINO/CRF/候选mask/dense保存。失败首例的额外工作单列在信号报告，不计为已知wall/CPU0。主goal仍active，尚未冻结新完整候选。

主Sol只读封存标量做2000次配对photo/RQ-connected区间，耗时0.476秒，不重复GT/raw/fit/prediction。accepted-minority中native L_A−whole H宏AUC：LVIS+.025730，RQ区间[+.011818,+.038851]；Deep+.114402，[+.089341,+.138674]；PACO−.068046，[−.081575,−.053934]。query-photo同方向；PACO570查询照片、232 RQ组件、最大43例，未按episode假装独立。

L_A正号若替换M会损TP39.29%/30.23%/57.00%，同时四票一致FP仍保留1191816/197286/1980719。PACO固定0/14恢复207831TP/1350FP，是实际互补；固定0/0拒绝干扰却只保97/23513原M TP，不能用单例推广。新Q观察确有不同有效排序，但其跨域方向不统一；后续保留whole与native的互补，评价一条统一完整组合及matched控制，不要求单模块独立胜FoRIS，不用域名选场或旧zero点替换所有M。

全部源、成本、反例和独立验证见[信号报告](NINE_WINDOW_SIGNAL_20261011.md)，机器完整结果 `evidence/local/nine_window_signal_20261011/native_fixed_head/report.json`；主Sol[配对区间收据](../../evidence/local/sol_validation_20261010/nine_window_paired_auc_intervals.json)及[入口](../../scripts/sol_nine_window_auc_intervals.py)。固定R/Q仅Mask切换的因果边界仍是既有三域60对，这1300普通benchmark输入没有异Mask同R/Q组。
