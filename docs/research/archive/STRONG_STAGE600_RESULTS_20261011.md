# 原 Strong S 的窗口、区域门控和最终多数：同600逐位复原

600例原始窗口Mask和区域收据全部存在。只用这些已封存资产复原旧S、同区域数量控制和原始窗口多数，全部逐位一致；没有运行encoder、原FoRIS/CRF、特征读取或新视图。随后读600份query标签归因，另146953项root整数、旧逐例I/U、class和区域质量核验通过。**Strong目标未完成，本项是阶段诊断，没有新算法成绩。**

最重要的纠正：原S误删完整FoRIS真前景的大头，在区域门控之前的局部观察和覆盖多数路径已经发生。原区域门控整体作用很强，直接推翻它会破坏现有强处理。

## 同一原600及完整输出

沿用[原600身份清单](../../evidence/local/strong600_conflict_alignment_20261011/membership.json)，不是LVIS1400头600。三批各200：`lvis_component_presence200_20261009`、`lvis_component_replication200_20261009/component`、`lvis_component_completion600_20261009/component`。对应原窗口目录为`lvis_fixedwindows200_20261009`及后两批父目录。

原S处理：合法reference Mask决定参考RGB裁剪；旧真实局部query完整FoRIS生成9张窗口Mask；对每张Mask作8连通区域、pool局部特征，与完整query FoRIS伪FG/BG角色比较；margin>0才接受整个区域；**所有覆盖窗口计入分母，严格多数，平票为BG**。当前只使用历史选择，不重算margin。区域不是真实实例。

| 同完整输出，600 observed-class mIoU | 原图 | CLI1024 |
|---|---:|---:|
| 完整FoRIS B | 47.779371 | 47.619569 |
| 原局部窗口strict多数，未用区域gate | 34.128915 | 33.977336 |
| 原S，query-context区域gate后strict多数 | **52.005138** | **51.702409** |
| 原同保留区域数量、按global agreement选择 | 49.744750 | 49.479666 |
| 任一局部窗口前景的并集，仅诊断 | 23.320457 | 23.243340 |

任一窗口并集是诊断反事实，不是候选。官方920 class/fold零填未观测槽的指标、逐例整数I/U保存在结果JSON；不把600 observed均值当全920完成成绩。

## S删除B前景的最早可见失败阶段

设原局部前景票数V、已接受区域票数A、总覆盖数D。U=(V>0)，R=(2V>D)，S=(2A>D)，A≤V。各Boolean图按旧FP32 bilinear/align_corners=False/>.5分别render到原图，仍有S⊆R⊆U。因而B∩¬S可互斥、完备分为以下三项。

| 原图阶段 | 删除B真前景 | 占所有误删 | 删除B假前景 |
|---|---:|---:|---:|
| 任一局部FG支持图渲染后为零，¬U | 248303 | 36.75% | 1553675 |
| 有局部FG，但未过原始覆盖多数，U∩¬R | 413377 | 61.18% | 856943 |
| 原始多数本成立，被区域gate改变，R∩¬S | 14014 | 2.07% | 28665 |
| 总数 | **675694** | 100% | **2439283** |

第一项的局部输出已经包含参考裁剪、FoRIS和CRF；原图¬U还包括any图的FP32>.5渲染不足，这里不能进一步归责encoder、reference或某个head。第二项包含原图render路径，gate也可能减少同一像素的支持。因此这三项是“最早可见失败阶段”，不是互斥的全模型因果机制。第三项在局部Mask、coverage和readout固定的accept-all反事实下，gate是最终拒绝发生所需的变化。

S还比B新增197153真前景和589455假前景。上述四编辑与此前封存600逐例收据全部相同；本次没有换同名集合或评价口径。

## 有效区域否决必须保住

区域gate相对原始局部多数，原图删除**113193真前景、4475364假前景**；其中B内部只有14014真/28665假，其主要收益是阻止局部窗口新增的大量假前景。完整mIoU提升17.876223pp。仅恢复B被误删处的归因，不能代表gate全局作用。

CLI逐窗口区域统计：拒绝27146区域，其中26307完全不含GT，249同时含真/假像素，590全在GT内；拒绝的逐视图像素1546487真/68338371假。接受20217区域，逐视图像素65263558真/43468211假。**逐视图重复计数不是原图新增/删除，不能相加成目标面积**。区域数也不能当物体数。

## 已接受支持与平票冲突

S删除B真前景中，411399原图像素仍有被gate接受的局部支持；其中337857属于已接受票数canonical覆盖平票图的原图渲染成员域。对应删B假前景分别864170和681776。原始未gate票数的平票图渲染域包含338498删B真/678558删B假。原图成员域不是原图重新计算的整数等票。

这定位了一个具体冲突：已接受的局部任务证据在最终覆盖平票被读为BG。但有支持/平票也覆盖大量假前景，**不能直接宣布补回它们有效，不能把它叫任务语义证书**。已有完整`S | (B & acceptedAny)`的历史版本同600原图50.681860<S52.005138，补411411真/864146假（[历史完整结果](parallel_signal_20261010/01_multi_observation/REPORT.md)）。它先在canonical旧roundtrip guide融合整Mask再render，与本次分别render条件域的411399/864170口径略不同，不能偷换新增账本。

仅再约束平票`S | (B & acceptedTie)`没有新任务身份信息，仍属人类关闭的九窗融合变体，**未冻结、未执行，不因本次归因启动**。同预算若pool恰是整个新增域就会与主臂退化为同一结果；rawTie/acceptedTie也非包含关系，不能把其差异当纯gate消融。后续真正不同的完整任务约束仍须先固定同600 full、必要完整消融和非退化实际预算控制，全seal后GT，无明显领先即毙。当前诊断没有新的候选成绩。

## 资源、协议和资产

准备inventory 0.669s，诊断50.629s/CPU54.225s，RSS733904896B（约0.684GiB）。先全部600逐位复原17.550s，再读GT。两次复原共4800项S/区域控制双frame逐位核验，另两次各600原native原图逐位核验；保存源码不含任何模型调用。

0 raw/MPS/encoder/FoRIS/CRF/训练/新view/新candidate/新最终Mask/dense写入。4200次Mask容器payload请求36.22MB，inventory加诊断5400次Mask文件SHA请求42.50MB，均为逻辑容器请求，物理磁盘IO未知。仅新增JSON收据、源码和文档。

入口：[protocol](../../evidence/local/strong_stage600_20261011/protocol.json)、[全部2400旧Mask容器身份与region/source收据hash](../../evidence/local/strong_stage600_20261011/inventory.json)、[GT前逐位复原seal](../../evidence/local/strong_stage600_20261011/reconstruction_seal.json)、[结果和成本](../../evidence/local/strong_stage600_20261011/report.json)、[600逐例整数](../../evidence/local/strong_stage600_20261011/episodes.json)、[逐视图区域TP/FP/margin收据](../../evidence/local/strong_stage600_20261011/components.jsonl)、[root核验](../../evidence/local/strong_stage600_20261011/root_verification.json)。

复原源码：[sol_strong_stage600.py](../../../scripts/sol_strong_stage600.py)，独立核验：[sol_verify_strong_stage600.py](../../scripts/sol_verify_strong_stage600.py)。原大缓存不复制；各原窗口Mask、完整B、S以及reference裁剪/窗口raw-key身份继续保留。

本结果限LVIS600、这条历史处理链。不是同query换reference Mask的新增因果干预，不是跨五域重复机制或冻结DINO统一不可恢复定理。
