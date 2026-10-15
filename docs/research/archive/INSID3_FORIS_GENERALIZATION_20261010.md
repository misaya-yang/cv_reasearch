# 从 INSID3 / FoRIS 的成功解释下一步研究

2026-10-10起的共同研究记录。用户要求解释 INSID3 / FoRIS 为什么简单且有效、尚未解决什么，以及已有失败与局部成功的具体原因。后续人类明确纠正：方法应作完整组合，可以保留或借鉴 FoRIS 的高质量处理，不要求每个孤立模块先赢完整 FoRIS。目标仍是有实质贡献、冻结 DINOv3、单参考、不训练的完整跨域收益。

**2026-10-11最新人类约束：** 后续方案限已有整图/真实九窗的原始DINO缓存，不新增前向、下载、扩队列或大体积中间产物。研究重心回到保住真实九窗的覆盖优势并解决误报，只提出能由既有字段执行的CPU逐例组合/消融；缺视图、层或Q/K/V就明确标缺，插值不是新的真实编码。清理由Sol统一，所有原始DINO特征及最小复现证据保留。已在进行的6380校准实验共102,080张完整mask已封存，当前只收尾独立评分，不新开后续实验。49318全官方仍未启动，之前的缓存执行准备不是启动授权。

**已完成结果：** 原生PASCAL/SUIM各2000七臂双帧已评分；N的CLI为55.677669/59.632460，相对FoRIS+.888713/+.449685。PASCAL的两类照片区间均正，SUIM的R/Q连通组区间跨0；旧SUIM40中MEAN明显领先的排序没有在此2000例延续。两种2380 CRF链均完成，单次链避免Deep的二次损伤，但仍未替代真实多视图强结果，新增来源筛选不稳定胜同数量控制。真实九窗及组件筛选的旧同批LVIS600/Deep100优势必须保留，PACO快九窗失败则是后续组合需解释的反例。五集完整官方、独立泛化与充分创新性仍未证实；seed-2清单至少322任务与已评分2380重复，不能称照片独立或全标签未见。下文保留各阶段证据，最新人类方向和完整结果优先。

## Astra 综合：目前可以站住的解释

**它们成功的关键是分工合适，而非某个公式简单。** 大部分跨域知识已经在 DINOv3 的预训练表示中；reference mask 指定要找什么，跨图对应提供目标线索，query 自身的区域组织提供覆盖与边界。INSID3 不要求一张参考图学出的二元分界直接切开全部 query；FoRIS 保留多个连续证据，让一次对应或一个 seed 的错误不必立即否决整个目标。这是源码和现有消融支持的解释，仍不能推成无训练方法必然泛化。

**已有失败不能用一个病因概括。** 参考拟合、ridge/NTK、CCA 和几何一致性各自优化的条件，并不保证“在另一图的真目标与相似背景间排序正确”；真实局部观察能找回部分目标，但严格多数、错误伪 BG 或错误区域归属会再次擦除它；query 内 min–max 与参考零阈值有不同的面积偏差，改变预测面积有时涨分却没有改善目标识别。因此“参考更可分”“多个视图更一致”“全图 AUC 更高”和“完整 IoU 提高”是四个不同结论。下文账本保留每类方法的正例和反例。

**本轮新增证据直接否定了一个容易误读的消融。** 同60对、120个真实任务的官方 INSID3 重放中，去掉 seed 内部相似项的平均 IoU 有小幅上移，但120个输出全部只发生扩张；COCO 的困难候选内排序反而下降。该项确实提供了部分目标排错能力，也确实会抑制同一概念的其他外观。下一方法需要解决这种任务依赖的归属，不能把“删掉这一项”或“扩大输出”当作更好的身份理解。

**当前研究约束应落在能力上。** 保留预训练语义度量及 query 的区域结构，在真实错种子、同父非目标和漏检区域证明更稳定的任务相对排序，并用同一完整读出核查收益。利用同一份缓存中尚未用好的信息也可构成贡献，不要求新增编码或强行换表示；同样不能先认定必须保留 FoRIS 的具体模块。当前没有独立方法在多域完整端点超过强基线，不把本轮诊断替换为算法成功。

本轮交付：两基线源码/原文核查；11类失败与局部成功账本；120任务官方 INSID3 与一个固定机制控制、240完整输出；tangent 三分支4800完整输出复核；两张同图换任务反例。所有新增推理都复用已有特征，新增 DINO 编码为0。

## 协作与证据约定

当前由 Astra 负责理论及只读证据分析，Sol负责实验与统一清理。沿用户此前最多3名Sol的明确授权，既有 `sol_experiments` 仅收尾已经封存的6380校准评分；其余两名此前任务已完成。新方向只核查已有真实九窗及整图缓存的组合能力，不新编码、扩队列或生成大体积中间产物。原单次CRF和原生4000进程已结束，原主Sol及其他线程复用已完成产物，不重复启动同批生产或评分。共同记录即本文件。各代理只在文件末尾追加带作者和主题的完整章节，不改写别人的段落；根维护首段当前状态及PLAN入口。通信限需要决策或完成的节点。数值结果必须标明样本、口径、真实控制和资产路径；原图、1024、token排序、oracle、局部编辑与完整输出不能相互代替。

保留三层措辞：**事实**来自真实源码或已核验结果；**解释**是有支持但仍有竞争解释的归因；**待证伪预测**明确下一项最小验证会支持或否定什么。不因公式复杂或拟合参考图好就推荐新方法，也不因一次局部涨分就称泛化。新增实验优先复用封存 O24/其他特征和已保存字段；不重复编码、不扫描失败方法参数。未完成的验证明确标记。

## Astra：已经从实际源码确认的起点

- INSID3 在去位置特征上做跨图对应，在原始单位特征上形成 query 内部区域；对应确定目标身份，query 结构决定区域覆盖。实现中的区域接纳同时使用 cross similarity、intra similarity、候选占簇比例，而非把参考 FG/BG 风险最小化后的一个线性函数直接迁移。
- FoRIS 在这类结构上保留 dense foreground 多原型响应，以 hard-BG 方向、参考最近邻票、query 区域 prior 及有界一致性项修正连续场，最后在 query 自身范围内读出。它的实际源码比“三阶段”标题包含更多启发式常数；不能把成功简单归为“参数少”。
- 二者没有跨数据集任务训练，但仍然依赖 DINOv3 的预训练先验、参考的任务标签和 query 的内部组织。训练自由不意味着没有先验，也不意味着任意无训练读出都能泛化。
- 已有独立 ridge / gradient kernel 的失败不能单独证明表示中没有任务信息：它们同时缺少两基线的若干结构与读出机制。需把可分性、目标识别、区域覆盖、绝对零点和完整输出分开验证。

源码核对位置：`../cv_data/third_party/INSID3/models/insid3.py:144-189,275-364`；`../cv_data/third_party/foris_official/models/foris.py:295-361,369-554,561-758,764-832`。以下各节继续追加可复核分析。

## Sol 缓存实验：已冻结 null-rank 反例与 INSID3 机制实验设计

**事实，已完成：**旧 Sol 的17个 PACO 真 focus 实验已完整封存并独立重放170个1024掩码。真 focus 无约束更新平均 IoU 37.019，原 R 读出33.124，完整 FoRIS42.634；安全 nullspace 真 focus 33.190，匹配插值对照33.162，配对差 `+0.028`，95% bootstrap `[-0.369,+0.454]`。17例中无约束更新7例倒退，`paco_part_007` AUC `.9804→.9368`，FP `59,374→167,530`；`paco_part_001` 则AUC `.6706→.8563`。安全臂保持全部4096 R token符号，必要安全系数中位仅`.0274`。这条保符号 nullspace 候选关闭，不扫步长；它没有跨域成绩。完整证据和复核脚本在 `SOL_CACHE_EXPERIMENTS_20261010.md`、`evidence/local/sol_cache_experiments_20261010/`。

**待证伪预测，先冻结再执行：**独立线性/梯度读出与 INSID3 的差异可能来自 query 内部组织而非只来自跨图身份分数。用已固定的 COCO/PACO/SUIM 各20对，同图A/B两任务、同一 O24 raw、同一原生黑图 positional basis，调用未改公式的官方 `INSID3.predict_mask`。只以已核验 raw 缓存替换 encoder forward，不做DINO编码；保留官方 APD、forward/backward candidate、raw-query agglomerative clustering、seed、cross/intra/area 与默认 bilinear 1024 输出。中途只读钩子记录 candidate、cluster label、seed、cross/intra/area，并在同一个已算状态上作唯一预设干预 `intra_sim=1`；candidate、cluster、cross、area、阈值及其余公式逐值保持相同。该臂仅为机制控制，不称新方法。固定无CRF；不把官方默认 bilinear 与另方法CRF混成一个终点。若日后补 INSID3-CRF，须单独注明图像/CRF路径及额外成本。

预测前冻结源码与输入 SHA，120个完整输出、阶段字段、逐任务记录全封存后才打开 query GT。评分以完整1024 I/U/FP/FN为主，同域20对A/B按任务及照片汇总；另报告仅诊断的64 token candidate 目标覆盖、seed纯度、cluster-label oracle 上限，区分候选先丢目标、seed错身份、cluster混合和聚合扩错。与同批 FoRIS 旧 trace 的 `pre` 和 `crf` 分列比较；在 FoRIS 原CRF与INSID3 bilinear之间的绝对差只说明最终链差，不归因于 query 聚合。另按冻结的 FoRIS 高低分分组，不按本实验结果重选。需要独立重算部分密集 mask 和 hook 同官方函数输出的一致性；所有组都是已曝光开发照片，非官方全量mIoU或独立确认。

## [Sol 基线机制] 从可迁移身份到图内区域的分工

**事实（论文、固定源码）**。INSID3 在参考 mask 选出的 FG 上取均值；用去位置的特征做跨图候选（FG 原型正相似且目标 token 的参考最近邻为 FG），在 query 原始特征上做细聚类与簇间自相似。候选重叠簇中取跨图相似度最高的种子，再以`跨图均值相似 × 种子图内相似 × 候选占簇比例 > 0.2`接纳区域；源码虽将种子 area 权重置 1，**种子仍须过同一阈值**。因此 seed/候选一旦错、漏，后续不能凭空恢复。位置基于固定空白输入的 SVD 前 500 维估计，属于预定义修正，不是在每个参考上拟合任意高维分界。见[INSID3 原文 §3、Table 3、附录 C.4](https://arxiv.org/html/2603.28480v1)与`../cv_data/third_party/INSID3/models/insid3.py:144-189,245-271,275-364`。原文 Table 3 的**公开协议**：直接阈值 COCO/PASCAL-Part 44.2/35.4，跨图簇聚合 54.6/48.5，再加图内自相似 57.6/50.5；这是区域组织带来增益的证据，不是本机成绩，也不证明单个部件跨域必然增益。

**事实（论文、固定源码）**。FoRIS 的参考 mask 同时定义 FG 多模态簇及 BG 集；BG 取与 FG 均值最相近的 20%，其均值去掉 FG 平行分量后才作为负方向。FG 用温度 0.07 的 log-sum-exp 聚合，初始响应减 `0.55 × BG`。目标最近邻参考标签票保留连续比例；候选给 query 聚类选种子，还每两格加入网格点以免候选过稀；query 聚类结合特征、RGB、位置，生成连续 prior。后续不一致/FG-BG 共高的惩罚有不确定度调制及上限，簇级修正也有限幅；最终 score 在当前 query 内 min-max，阈值 0.5，再可选 CRF。APD 仅当参考 FG 与目标均值语义相似小于 0.8 时启用。默认 Stage-1 正标量 gate 对 Stage-2 参考和 raw target 评分不起作用，进入 FC 前的 gated target 又被单位化而消去；不能把该 gate 的独立贡献当作当前源码事实。见[FoRIS 原文 §3–5、附录 F](https://arxiv.org/pdf/2609.03384v1)与`../cv_data/third_party/foris_official/models/foris.py:239-268,295-361,369-554,561-758,764-832`。FoRIS 原文消融支持组合流程，但与此固定源码的 gate 生效性不能混同。

**解释（有竞争解释）**。这两法的简洁不等于“参数最少”，而是把三种不同统计任务分开：参考标签只确定**目标身份**；跨图相似提供可迁移但有位置/背景混淆的身份线索；query 图内结构决定**覆盖与边界**。INSID3 把身份作为硬候选/seed，再利用目标图自相似补全，避免要求一个参考 FG 均值精确线性分开所有 query 像素；FoRIS 则把多原型、难 BG、参考 NN 票和 query prior 当作相对证据连续合成，减轻错误硬截断。有限种原型与相对比较降低了“在一张参考上学习高维判别器”的自由度，但**不是不自拟合的定理**：mask 若含背景、只有一种外观或挑错难 BG，低维原型仍会系统偏移；query 聚类若跨真实边界，图内组织同样会扩散错误。相似度高、同色或同父物体的无标签局部细部，可能在所有这些证据中仍同号；单参考 mask 未直接编码“为何这一部分而非相邻部分”的跨视角规则。

**边界（事实与推断分开）**。目标不存在时，INSID3 论文附录 C.4 报告 COCO 4000 缺目标配对 85% 正确空 mask；源码机制允许种子不达 0.2 而空输出。这是有限协议结果，非任意域的可靠拒识。FoRIS 当前 min-max 读出在非恒定 score 下总会令某处得 1，缺少同类绝对零点；故缺目标时易强制选局部，这是由源码推出的风险，**未据此声称实际缺目标准确率**。thin vessel/road 常小于或混入 token、跨枝外观分散，图内簇或 CRF 不能创造被编码尺度抹去的真值；FoRIS 原文附录 F 的默认 1024/layer23 Fundus/DeepGlobe-18 one-shot 19.7/13.3 mIoU，层/分辨率敏感性支持该边界，但这组公开数与本地自建道路协议不可直接相减。位置先验在稳定解剖中可有用，永远去位置也非普适；FoRIS §4.2 的 APD 消融体现了这点。

**机制级原则**。独立新算法若要超越“FoRIS 的校正器”，应在**参考 mask 的任务含义如何稳定地跨图绑定到 query 的局部区域**上产生可测的新信息，同时保留 query 区域组织和可拒识/校准的完整输出；不能仅对 FoRIS 最终 score 重排、追加图传播或调系数。判别创新应看它是否在同输入、同编码、同完整读出下解决 FoRIS/INSID3 都丢失的身份或区域信息，而非只修复某个后处理阈值。

**待证伪预测（最多两问）**：① 对同父细部与错误种子例，若固定 query 区域，只交换参考 mask 中目标/相邻非目标的标签，真正读懂任务的核心证据应让正确区域相对排序稳定反转；若两种标签仍指向同一高相似簇，则缺的是任务区分信息，不能靠末端阈值补。② 对缺目标与细线断裂例，记录正确区域在**原始特征响应、跨图票、query prior、完整 score**逐阶段是否曾成为可分的高秩区域；若从未可分，需改变特征观察尺度/任务绑定，若曾可分后丢失才有理由改区域聚合或读出。两问均可先从已封存 raw/字段做固定样本、配对检查，不启动新候选或权重扫描。

## Sol 失败与局部成功账本

本节只综合已执行的开发证据，不提出新候选。**共同机制判断（解释，不是因果定理）**：INSID3/FoRIS 借冻结 DINOv3 的跨图语义相似性找目标，再借 query 内部区域或连续场、参考负例与有限的空间约束控制覆盖和边界；这种组合对“一个参考标签如何落到另一图”比单个参考拟合函数更有约束。源码层机制见本文 Astra 段。训练自由仍使用大规模预训练语义先验；简单有效不是“参考样本足以唯一确定任务函数”。反面缺口集中在部件相对父物体的身份、细长/小面积目标的真实像素 footprint、原 guide 漏检后被伪 BG 二次拒绝，以及跨图分数零点/负证据稳定性。下表“原因”均为现有控制支持的**解释**；带 GT 的分组、oracle 和 AUC 只作诊断。

| 机制家族 | 事实：实际收益或失败（各行口径独立） | 证据支持的原因；反证/未解决点 | 应保留什么 |
|---|---|---|---|
| FoRIS 连续场的 RCG/rank 图/fine 读出 | 旧 self-sampled 原图：COCO1800 FoRIS 60.517→RCG 61.798→fine 62.179；PASCAL598 53.868→55.328→55.352；SUIM582 60.950→60.872→61.250；PACO599 41.445→41.745→41.691。旧 COCO4000 的真实 shifted-fine64 比同核 coarse 控制 +.352 点。 | 图能修剪错误 unary、真实细观察能改善部分边界，属于已有 FoRIS 场的读出增量；旧 COCO4000 scalar residual 相对 MEAN +.332 点的编辑主要调整背景范围。PACO 部件收益很小；LVIS 条件 rank 图在100例仅比匹配边量控制 +.229，仍低 FoRIS。不能把这些叠成独立语义能力，也没有 scalar residual 跨域完整确认。 | 保留可归因的图/真实细观察增量与同量控制；不把旧 processed-cache 数值和当前原生协议混减。证据：[03 报告](parallel_signal_20261010/03_graph_readout/REPORT.md)。 |
| 原/快九窗与组件 query-context | 原图同批：LVIS 原600 FoRIS47.779→区域52.005，Deep 原100 29.378→37.024、快100→37.344；PACO 快600 38.461→34.971，虽比未筛选多数29.539强。 | 多视野给 Deep 正确覆盖，组件拒绝改善 LVIS；但严格多数会擦掉 PACO 已接受少数支持（全体 GT 的18.92%），同时四窗一致仍有大量父内错误。LVIS 原区域还在236/600例改变了 R crop，故 Q 九窗并非单因素原因；PACO 原慢版未执行。 | 保留“真局部观察有用”及“身份/融合另需解决”两个结论；先做固定 Q 九窗、撤去 LVIS 额外 R crop 的匹配对照。证据：[01 报告](parallel_signal_20261010/01_multi_observation/REPORT.md)、[04 报告](parallel_signal_20261010/04_dataset_differences/REPORT.md)。 |
| query 伪 BG 与 whole FG/BG 竞争 | PACO600 原图 FoRIS38.461→whole39.640（+1.179）；净减少 FP 的93.87%来自父实例内部。COCO200 63.359→60.896（−2.463）；Deep100 +.152。 | 最强负例有同父排错价值，却会删真；PACO 新增区旧竞争 h 的 fine128 AUC 仅.4875，对应 source .6032。LVIS guide recall<20%的17例里 query component 只保留局部 TP 发生量28.01%，reference gate 保留正确支持。GT 条件不可作为部署开关；新 source-agreement 的 Deep +.719 仍低快九窗，PACO 未保住 whole 收益。 | 保留同父负证据，但必须允许推翻伪 BG，且检验实际新增区而非全图排序。证据：[02 报告](parallel_signal_20261010/02_task_identity/REPORT.md)。 |
| APD-ridge 参考方向与有界 anchor | 冻结 global 原图：PACO600 官方303槽 FoRIS38.461→40.775（+2.314），剩余500亦 +1.705；COCO200 官方40槽 FoRIS63.359→62.460（−.899），低 MEAN64.007。Deep100 local anchor34.289胜 FoRIS29.378但低快九窗37.344。 | 参考判别方向有真排序信号（Deep 新增区 local AUC 比 P0 +.100；PACO whole +.089），有界 anchor 把它部分兑现；但 reference 单独阈值在 PACO100 把 whole 零阈值39.090降到24.300。PACO 与 COCO的编辑符号不同；COCO−.899区间跨0，不能称普遍显著退化。scene selector Deep 选全 whole 的55例中52例固定 local 更好，简单面积/纯度/接缝开关未获组内支持。 | 保留 source 方向和 anchor 的局部成功，始终同时报 uniform/固定视图/完整 FoRIS/MEAN；不要按域挑 best。证据：[主线](parallel_signal_20261010/ROOT_RESEARCH.md)、[COCO 固定验证](parallel_signal_20261010/02_task_identity/COCO_RIDGE_REPORT.md)、[05 标定审计](parallel_signal_20261010/05_reference_readouts/REPORT.md)。 |
| R1-G/joint geometry 与参考场校准 | 独立重数同一原图 pilot：Deep100 R1-G 12.597 对 direct unary 12.631，PACO100 29.275 对29.237；Deep/PACO 原图像素 XOR 仅.6162%/.8072%。旧 LVIS100 matched-reference 校准49.110虽高 FoRIS47.874，却低 uniform 校准49.766。                                 | 改变很少像素或仅在同图校准更好，不能证明跨图任务识别。reference held-out patch 与拟合 patch 同图且空间接近；Deep/PACO 补集的多数 FG patch 还被量化采样抽走。FoRIS 场与 R/Q 分布不共享可靠绝对零点。 | 保留匹配 uniform 控制与 reference coverage 质量；R1-G 具体比例及直接因果归因暂未知。证据：[05 标定审计](parallel_signal_20261010/05_reference_readouts/REPORT.md)。 |
| 场景基底与 reference→query mixture | 600例 CLI1024 新 scene＋CRF 均低同批 FoRIS：Deep15.986/29.378、PACO44.100/45.630、COCO41.629/63.170、LVIS28.064/46.680。reference→query 条件均值推前景比例的平均绝对误差为 Deep17.528、PACO12.563、COCO15.471、LVIS12.484个百分点，虽方向全图 AUC .906/.911/.954/.959。 | query 组织可影响预测范围，却不能把 source 角色分布直接当 query 角色分布；PACO scene 相对匹配 R-basis 增 TP 也增 FP，不能称身份更准。mixture 只否定纯 label-shift 必要条件，不能否定所有无监督校准。 | 保留 query 结构作为待验证覆盖信息、区分 AUC 与绝对读出。证据：[scene600](autonomous_20261010/SCENE600_RESULTS.md)、[mixture600](autonomous_20261010/REFERENCE_QUERY_MIXTURE600_RESULTS.md)。 |
| Copula 邻域关系 | Deep/PACO 真 context 的 whole 节点 AUC .518/.461，低同 ROI support-ridge .925/.911；PACO 四状态平衡风险变差，原频率 NLL 改善受 BB 占83.25%影响。 | 真实物理邻接表有统计效应，但现有条件依赖项没提供可靠目标身份；同边际 shuffle 对照和稀有状态平衡反驳“自然频率 loss 降=新身份信息”。它仍只是 saved-table 诊断，无完整 mask。 | 保留真实配对与 shuffle、balanced-state 控制；不升级成 segmentation claim。证据：[copula200](autonomous_20261010/COPULA200_RESULTS.md)。 |
| 真实 R focus 与 reference null-rank | 实际 focus200 的固定 equal+CRF CLI1024：Deep25.620低 FoRIS29.378，PACO39.057低45.630；PACO比插值focus +3.079、丢全目标例21→11，仍低基线。独立核验 PACO17：原 R33.124，无约束真 focus37.019，安全真33.190、插值安全33.162，FoRIS42.634；安全真−插值+.028，CI跨0。 | 真观察可恢复部分 footprint，却不稳定提供身份/零点：001 AUC .6706→.8563，007 .9804→.9368；009 IoU 改善而 AUC 不升。保留全部4096 R token 决策迫使安全步长中位.0274，吞掉新增信息。 | 保留真实观察、插值匹配与逐例反例；关闭当前 nullspace 安全步长候选。证据：[focus200](autonomous_20261010/REFERENCE_FOCUS200_RESULTS.md)、[独立复核](SOL_CACHE_EXPERIMENTS_20261010.md)。 |
| 独立跨图任务函数：R→Q重建、全局物理先验、同位置绑定 | COCO/PACO/SUIM各20对 A/B 的 CLI1024 双任务/照片均IoU：R-only 45.851/41.879/47.104，R→Q约束36.371/42.548/46.455；物理先验 Deep20/PACO20 13.471/26.494，低 FoRIS29.289/44.916；同位置绑定16.665/23.759，也低相同 FoRIS。 | R/Q 共同 oracle 用 query 标签可到 Q留出 AUC .995/.992/.988（并非可部署），证明“可表达”与“仅凭 R 能选中”不同；物理先验内 oracle 仍强。绑定虽胜部分同视图控制，未稳定胜保边际打乱的困难区排序，更未胜完整外部强控；不能说真实配对本身已提供足够任务约束。 | 保留 query-label oracle 仅作上界、同边际物理打乱作必要对照；不宣称原创完成。证据：[task core](TASK_TRANSFER_CORE_20261010.md)、[physical](TASK_PHYSICAL_PRIOR_20261010.md)、[binding report](../../../../cv_data/a/cross_view_binding40_20261010/report.json)。 |
| finite NTK、hidden、linearized 与 source-span | 同60对、A/B、CLI1024 双任务/照片均IoU：task tangent 41.172/32.661/46.205，O24 ridge45.851/41.879/47.104、位置控制48.802/39.417/50.044、FoRIS55.854/47.682/60.239。共同 midrange 给 tangent42.334/45.285/53.057；linearized-task 零阈值44.315/39.359/46.458，仍低 FoRIS。 | 真实末块梯度核经数学/autograd检查，结果只拒绝这套核+读出，不拒绝 DINO 表示有任务信息；阈值变化可大幅改结论。密集 R 监督对 Q 的 soft IoU几乎不补，query 标签 oracle 的增益可藏于采样 R nullspace，故 source 自拟合不能辨别竞争解释。当前 mask 的独立复核状态以另线收据为准，不把数学核验写成全流程独立核验。 | 保留相同字段/同阈值的强控制、A/B任务切换与 source-span 诊断；不再叠核公式找事后最佳。证据：[kernel report](../../../../cv_data/a/task_tangent_kernel60_20261010/report.json)、[source-span](../../../../cv_data/a/task_transfer_source_span60_20261010/report.json)。 |

**三条最强的可证伪问题（供 Astra 选，不等于实验授权或算法结论）**：

1. **身份是否能在同一参考任务下由合法多观察唯一约束？** 固定 R mask/Q 视图与完整输出，比较真实同物理位置配对、保每个角色/视图边际的置换、因子化和同成本单视图；事先指定父内 BG 与 guide 漏检区。若真实配对不胜置换的困难区排序和完整强控，不能归因于新增身份信息。
2. **Deep/LVIS 的局部收益到底来自 Q footprint 还是 R 尺度？** 在相同冻结 episode 上只替换 LVIS 那236例额外 R crop，并另以匹配插值控制真实 Q crop；保留每个原图 TP/FP 与同 K 几何。若撤 R crop 近乎无影响，现有参考纯度提升只是伴随现象；若真 Q 与插值无差，不能称新增物理观测必要。
3. **可用的跨图负证据与零点从何而来？** 固定 reference 方向，分别测目标对父内 BG、父外 BG的排序，和不读 query GT 的阈值/有界读出；要求同一规则在 PACO、COCO、Deep 及 A/B 换任务下胜 FoRIS/MEAN、uniform/固定视图控制。若排序有益而完整输出不增，先定位阈值或 query 内覆盖；若困难区排序也不增，先承认参考任务解释未识别。

截至这些已曝光开发证据，尚无独立于 FoRIS、在多域完整原图端点领先强控制的 frozen-DINOv3 单参考方法。旧原图、CLI1024、fine/token AUC、query-label oracle、局部编辑和缺类槽位各有不同分母；本账本从不跨层相减或把各行的最好臂拼成新方法。

## [Sol 缓存实验] 官方 INSID3 全链路重放与图内相似干预结果

**事实（同输入、同任务、完整输出）：**设计保存在运行的 `config.json` 并由 `sealed.json` 绑定；官方 `INSID3.predict_mask` 用同60对 COCO/PACO/SUIM 的原始 O24 和真实 A/B 参考mask重放，原生黑图 positional basis，官方 APD、candidate、raw-query聚类、seed、cross/intra/area及默认 bilinear 原式保留。仅用 adapter 注入已核验 raw，新增 DINO 编码0。60 pair/120任务的预测和中间状态在读 query GT 前全封存，wall time 73.69秒（MPS+CPU缓存推理，非冷端到端）；官方 sklearn 聚类每个Q算1次，三域各1个Q独立重聚类后，A/B label逐位一致。完整资产 `../cv_data/a/insid3_task_structure60_20261010/`，源码/独立核验在 `evidence/local/sol_cache_experiments_20261010/`。封存后从64² mask独立重建240张1024²输出逐位相同，120个聚合场重建等于官方输出；60个状态、60个预测、120个GT的SHA全部核对。

| 同批40 A/B任务/域的完整1024 IoU均值 | 官方 INSID3 bilinear | `intra_sim=1` 控制 | FoRIS pre | FoRIS CRF |
|---|---:|---:|---:|---:|
| COCO | 46.513 | 48.824 | 54.776 | 55.854 |
| PACO-Part | 44.637 | 44.640 | 46.772 | 47.682 |
| SUIM | 54.141 | 55.070 | 59.562 | 60.239 |

**机制干预的限定结论：**固定 intra 只改变官方聚合中的一个因子，candidate/cluster/seed/cross/area/阈值逐值相同。相对官方的20照片配对均值差为 COCO `+2.310`（95% bootstrap `[-2.026,+6.797]`）、PACO `+0.003`（`[-1.945,+1.735]`）、SUIM `+0.929`（`[-1.456,+3.547]`）；三域区间均跨0，不能推荐固定 intra 作为改进。它确实改变了23/13/17个任务的完整mask，并随同一Q的A/B身份翻转：`coco_000`官方 A/B IoU `72.41/1.16`，固定后 `46.83/54.92`；`paco_part_001`官方 `33.90/43.32`，固定后 `80.74/7.37`。所以图内相似既能抑制错误簇也会排除正确目标，并非恒定平滑项。FoRIS原CRF与INSID3 bilinear只作分列终点，差值不能单独归因为该项。

**阶段限制（仅64 token 软像素质量，GT封存后诊断）：**candidate 对目标的 recall 均值 `.709/.587/.559`，seed 簇目标纯度 `.629/.548/.695`（COCO/PACO/SUIM）。以旧 FoRIS CRF 分数最低10个任务/域预定困难层，recall `.52/.31/.23`、seed纯度 `.29/.21/.29`；最高10任务分别 `.90/.78/.75` 与 `.95/.87/.90`。低层用 query GT 事后选择任意簇并集的 token-mass IoU 乐观上限 `43.63/41.40/58.95`，只准候选触及的簇则 `32.63/24.95/53.79`。这说明候选漏目标、seed错身份与区域混合都可能限制读出，但低/高关联不是单独因果证明；oracle的64-token上限也不能与上表完整1024 IoU直接相减。更细逐例/FP-FN、SHA和复核见 [`SOL_CACHE_EXPERIMENTS_20261010.md`](SOL_CACHE_EXPERIMENTS_20261010.md) 与封存 `report.json`。当前没有跨域领先的新算法。

## Sol 补充：参考反向裁决与联合观察

此节更新 [误报底噪报告](FALSE_ALARM_FLOOR_20261010.md) 第9节的时间边界：该节当时只有 COCO 原顺序3例联合编码；工作区现在已有 COCO150、LVIS150 的真实联合/分开768输入封存评分，以及 raw/APD 两分支。以下先看已完成的150例，不能继续拿3例（其中 `joint.ridge_all` 80.635 对 FoRIS78.451，但候选内AUC .646低于分开.723）当作联合观察的总体结论。150例仍是**曝光开发队列**；原图指标是实际观察 fold/class 槽位平均，不是全官方槽位，也不是 CLI1024。联合画布同时改变跨图上下文和位置几何，因而即使联合胜分开，也不能单独归因于 cross-attention。

|同批原图完整 FoRIS / 观察槽 | 固定联合 vs 分开：可保留的效果 | 反例与解释边界 |
|---|---|---|
| COCO150，65.974 / 38槽 | raw `back_auc` 66.187（+0.212 对 FoRIS）、分开65.635；联合命中保留.995、误报保留.879；候选内AUC联合.6831、分开.6837。 | 联合相对 FoRIS 的已曝光配对区间[−.251,+.369]跨0，收益很小；`ridge_all` 联合59.893、分开59.185，都远低 FoRIS。raw `two_sided` 联合62.129、分开61.294，也未解决。 |
| LVIS150，47.648 / 150观察槽 | raw `ridge_all` 联合44.879、分开42.880，局部读出确有上移；APD `ridge_all` 联合45.286、分开43.938。 | 这些臂都低 FoRIS；raw `back_auc` 联合47.455、分开47.456，APD联合47.425、分开47.478，也都略低 FoRIS。联合未建立可迁移完整强基线优势。 |
| APD 同批控制 | COCO150 APD `joint.ridge_all`62.772，比 raw joint59.893高，但仍低65.974；APD `joint.back_auc`65.922低 FoRIS。 | APD 改变特征分支与读出表现，不能将 raw/APD 中每项最高臂拼成方法；LVIS150 APD联合 `ridge_all`仍45.286<47.648。 |

这些150例是各自真实封存结果，不是前3例的简单外推。COCO150 [报告](../../evidence/local/false_alarm_floor_20261010/inmask/joint_coco150/report.json)、[4800个原图 mask/IU独立核验](../../evidence/local/false_alarm_floor_20261010/inmask/joint_coco150/final_verification.json)；LVIS150 [报告](../../evidence/local/false_alarm_floor_20261010/inmask/joint_lvis150/report.json)；[COCO APD报告](../../evidence/local/false_alarm_floor_20261010/inmask/joint_coco150_apd/report.json)、[LVIS APD报告](../../evidence/local/false_alarm_floor_20261010/inmask/joint_lvis150_apd/report.json)。报告均保存完整字段和 seal；在本节只核读报告，不宣称我对 LVIS/APD mask 做了额外独立重放。

“候选回到参考图裁决”的已测固定 `back_auc` 也有严格边界：第9节先冻结设计 PACO600/COCO200，再到 LVIS200/PASCAL40 开发迁移；同源原图观察槽的设计收益仅 +.0348/+ .0103，LVIS −.1283，PASCAL +.0221。LVIS 的汇总像素IoU虽 +.3671，类mIoU却 −.1283，因此不能拿单一总误报交换律推出每类收益。回投找到混淆反例的比例在 PACO/COCO/LVIS 为89.2%/52.0%/55.0%，不等于能安全删它们；`back_auc_hard` 在三组为 −1.88/−6.19/−2.60。主PACO/COCO/LVIS 外来参考上半段覆盖误报30.5%/23.3%/31.5%，覆盖命中34.4%/29.5%/30.6%；这也不支持“所有误报都是任意参考都会亮的空响应”。这些数值和封存路径在[第9节](FALSE_ALARM_FLOOR_20261010.md#9-m4-实际执行结果2026-10-10)。

[误报底噪报告](FALSE_ALARM_FLOOR_20261010.md)前半的 `IoU=I/(G+F)`、召回与误报面积账是**恒等式/诊断工具**；`F=0` 和无漏检是使用 query GT 的上界，不能解释成可实现增益或唯一病因。`F` 与目标面积在若干旧包相关低，不代表误报大小普遍与目标无关；SUIM、Deep 等已有例外，不同输入裁剪、父部件身份和像素 footprint 也影响错误。0.88 AUC 是该报告双正态近似与指定 K95 目标下的换算，不是跨域理论门槛；主队列候选内 AUC 与 K95、完整类mIoU必须一起检查。更好的相似度/编码不是已经被排除：已见 APD、真 focus、joint 对某些排序或输出有局部作用，只是目前未兑现为共同规则的完整跨域领先。其它层或更多联合参数尚无本节所读的匹配完整证据，不能用“误报地板”直接断言应做或不应做。

R1-G 的直接结果来源是 [独立原图计数](autonomous_20261010/evidence_validation.md#4-joint-role-repair-substantial-model-failure-small-wrapper-drift)：Deep100 joint12.597 对 unary12.631、PACO100 joint29.275 对 unary29.237；原图像素 XOR .6162%/.8072%。这只说明几何 wrapper 改动小，不能证明联合关系模型的任何可行容量都无效。

前表 finite tangent 行的验证状态已更新：[独立输出复核收据](../../evidence/local/task_transfer_core_20261010/tangent_output_verification.json) 为 `PASSED`，对 normalized、common-midrange、linearized 三次已封存研究重建4800完整mask、重数5520条I/U/FP/FN/IoU记录并核对 source/hash；没有新拟合或编码。因此三次报告的**完整输出计数**已有独立重建支持，仍只回答这三套固定核/读出，不能推成全部任务函数先验失败。


## Astra：把图内相似的排序效应与预测面积效应分开

这项只读诊断核对全部120任务的 sealed mask/state、FoRIS trace、真实GT与RGB身份，没有新模型推断或编码。逐像素证明 `intra=1` 在120任务上均只增加预测，不删除任何像素；实际改变mask的任务数为 COCO23、PACO13、SUIM18（上节17是按IoU点值变化计数，不能替代mask变化数）。公式也解释了方向：在这些实际正相似区域中，`cross × area × intra > .2` 相当于 `cross × area > .2 / intra`；去掉不超过1的因子会放宽接纳，不能仅用涨分证明排序更好。

| 同一ROI上“无intra−原版”的AUC差；两任务/照片均值 | COCO | PACO-Part | SUIM |
|---|---:|---:|---:|
| candidate触及的query簇内 | −.00919 [−.01713,−.00284] | −.00076 [−.00576,+.00516] | −.00120 [−.01264,+.00964] |
| 原完整FoRIS前景内 | −.02098 [−.05056,−.00284] | −.00303 [−.00980,+.00331] | +.01306 [−.00317,+.03977] |

这是64格常量场按1024真实FG/BG像素质量的排序诊断，并非完整输出mIoU。有效成对数第一行20/19/20、第二行19/18/19；区间按连接的R/Q照片组重采样2000次。COCO当前证据支持intra具有排错排序价值，但固定阈值下召回损失可超过这项价值；另两域没有相同稳定方向。不能从公开论文平均消融推断本批每个任务都受益，也不能从本批平均IoU小涨推翻该机制。

两张可视反例使用实验报告已经指出的A/B相反效应，不是总体代表性抽样：

- [COCO同图两个任务](../../evidence/local/insid3_foris_generalization_20261010/coco_000_task_switch.png)：停车计费器任务新增351 TP、28,065 FP，IoU72.41→46.83；车辆任务新增125,915 TP、2,945 FP，1.16→54.92。相同query的区域结构在两个任务中承担不同的排错/召回作用。
- [PACO同一容器两个部位](../../evidence/local/insid3_foris_generalization_20261010/paco_part_001_task_switch.png)：任务A新增361,079 TP、2,369 FP，33.90→80.74；任务B新增611 TP、507,060 FP，43.32→7.37。不是“PACO应整体少用图内项”，更不是仅按参考mask面积就能推出query目标面积。

对于只扩张的任一已有mask，加入区域的真实精度若超过 `旧IoU/(1+旧IoU)`，IoU才上升；这是由I/U得到的事后恒等式，不能作为推理时使用GT的规则。它解释了为什么同样扩张能呈现局部成功与灾难性反例，也说明收益必须同时检查TP/FP及目标排序。

入口 [insid3_factor_evidence.py](../../../scripts/insid3_factor_evidence.py)，[全部逐任务及区间](../../evidence/local/insid3_foris_generalization_20261010/factor_evidence.json)。图片经过实际查看，绿色TP、红色FP、蓝色FN，显示的是模型使用的方形输入视角。

**统计修正。** 同一诊断发现此前tangent producer的photo grouping把SUIM缺失的photo_id转成字符串None，从而错误合成一个照片组，产生退化区间。已以真实photo_id或已核验RGB SHA回退重算：COCO/PACO/SUIM实际20/18/20组；没有改变任何mask、GT、I/U或点估计。原sealed报告及生产源码保留；受影响的不确定度以[修正附录](../../evidence/local/task_transfer_core_20261010/tangent_uncertainty_correction.json)为准，未形成跨域强结果的结论不变。

## Sol 快速既有实验核对：reference 锚定的 R+Q 调和图

**结论：相近的真正联合 Dirichlet 图已经失败；当前提出的 APD 跨图邻接 + raw-DINO query 邻接这一精确配方，未找到已执行证据。** 不能把它写成全新问题，也不能把旧负结果扩大为所有联合图都不可能。证据身份要分三层：

|层级|实际查到什么|对本问题的约束|
|---|---|---|
|真正相近的已执行图|[COCO DEV241 旧 report](../../evidence/local/results/joint_reference_graph_v1/report.json)明确写 `Joint reference Dirichlet inference`、真实 R 标签、FG/BG 参考边质量平衡、Q native 32-NN、带宽 .07；无 CRF 的1024模型尺寸，joint 23.170、同批 FoRIS pre-CRF58.562、reference-density35.169、FoRIS-graph31.085。241例中27升212降；新增 FP 77,297,228，参考→查询边质量中位比例仅 .000161，70例预测面积超过整图95%。|这不是仅在 FoRIS score 上的 MEAN/RCG 平滑，确实把 R 标签作为图边界；其败因有可见的弱锚定与强 Q 混合/背景扩张，而非“线性 ridge 不会读出”。但本地只找到封存汇总 `report.json`，未找到此 v1 的实际 runner/逐 mask，可核对的是报告契约和数字，不能声称本轮源码重放。|
|另一历史近似|[旧 reference_absorption 账本](../../evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md#old-reference_absorption)记 processed-cache、work1024 的 first-hit/harmonic R 吸收 + residual：600例37.8215；同批 MEAN62.9315、nearest51.8913、one-step61.5380。|它含 MEAN/base、union 图与 residual，已严重失败，但输入、初值和读出与上述纯 Dirichlet 控制及这次 APD/raw 提案不同；只算相关反例。源码路径在账本中声明，当前工作区未找到 `reference_absorption.py`，不补称当前独立验证。|
|尚无精确已试证据|未找到把 **R 所有真实mask标签固定为 Dirichlet 边界、R↔Q 边由 APD semantic、Q↔Q 边由 raw DINO** 分别构造，再在相同帧/样本上与上述 joint/reference-density/图控制配对的封存报告。|旧 joint v1 的 bandwidth 与图关系不能自动视为 APD 异构邻接；INSID3 的跨图身份/图内区域分工只是动机。若以后考虑此方向，必须先解释如何避免旧 v1 的极弱边界质量及错误背景扩散，并证明不是靠阈值/seed 避错；本节不授权启动。|

[旧 query recurrence 的600例对照](parallel_signal_20261010/05_reference_readouts/REPORT.md#已执行旧正结果及强控制)相对 single-seed 有+.749点、相对 all-seed +.848点，却与 MEAN 近乎持平；这支持“单个错误 Q seed 可能伤及其它实例”作为**机制担忧**，不等于上述 Dirichlet 图已解决多实例身份。对现有 RCG/MEAN：它们在 FoRIS 已有连续场上修图，R 标签并非调和方程的固定边界，因此不能拿其正差替代 R+Q 图证据。当前决定只是登记精确差别及旧失败，不追加新候选或运行。


## [Sol 基线近邻审查] 吸收式参考标签图传播的研究位置

**直接近邻，而非原创性主张。** [LPOSS（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Stojnic_LPOSS_Label_Propagation_Over_Patches_and_Pixels_for_Open-vocabulary_Semantic_CVPR_2025_paper.pdf)已用 DINO patch 亲和图及线性系统做 training-free 标签传播，还以像素图修边界；但其种子来自 VLM 文本预测、作用是单图预测的软保真细化，不是把单张真实参考 mask 的 FG/BG 作吸收边界，在跨图与 query 图内两套几何上同时求未知标签。[Pyramid Graph Networks（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Pyramid_Graph_Networks_With_Connection_Attentions_for_Region-Based_One-Shot_Semantic_ICCV_2019_paper.pdf)更直接把 support/query 作为图并逐元素传标签，说明“参考图传播到查询图”早已提出；其跨图边权由训练的 graph attention 学得，不能把 frozen-DINOv3 的 Dirichlet 解本身当作方法创新。[GF-SAM（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/3a2ef31a1e45908901adc0ca853a8faf-Paper-Conference.pdf)也是无训练、参考驱动的图法，但图节点是 prompt 点及 SAM 生成的 mask，边是 mask 对点的覆盖关系，最终质量依赖 SAM；它不是 DINO token 上的吸收标签扩散。这三篇尚未显示已有**完全同设定的成熟全法**，但已经覆盖了“图传播”“跨图 support-query 传标签”“无训练图推理”各个基本构件。若更全面文献核查找到同设定全法，应把它作直接基线而停止重复命名。

**为何仍可作机制控制。** R 的真实 FG/BG 标签固定、Q 无标签，调和解只由冻结特征的跨图边与图内边决定，不再从单 R 学自由高维分类面；多个互不相邻的 Q 支路可以分别受 R 标签到达，无单 seed 必选约束。它检验的是“固定几何里的任务标签能否跨图到达所有真区域且避开相似假区域”，而非图拉普拉斯算法新颖性。与之相近的 [RePRI（CVPR 2021）](https://openaccess.thecvf.com/content/CVPR2021/papers/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.pdf)已指出仅用 support 监督的线性读出会在 query 上缩成小响应、单加熵仍会退化，需要考虑 query 的前景比例；其特征主干经过 base-class 训练且做测试时优化，不能把其数值直接当 frozen-DINO 控制，但警示“无自由分类面”并不自动解决标签密度偏置。

**唯一尚待解释的假设。** 即使跨图边采用 APD、Q 内边采用 raw 特征，二者的相对导通量若未按源/目标采样密度约束，吸收概率可能主要反映 R 的 FG/BG token 数与高连通背景，或在 Q 内越过同父细部边界过度平滑，而不是 reference mask 所指定的概念。最小判别是固定图构造与同一 Q，只交换 R 目标/相邻区域标签并记录各 Q 连通区的 FG/BG 吸收质量与真/假目标排序；若排序随任务翻转且多支均被覆盖，该机制值得进入完整输出对照。若在源端面积、背景密度或跨边权微扰下即塌为全空/全满，说明首先要解释图的质量守恒与边界，而非加新层或扫传播系数。此处是**预测/控制建议**，未运行新的分割实验。

## [Sol 缓存实验] 冻结：分尺度吸收式 query 图机制对照

只检验旧 joint Dirichlet 图弱参考锚定的具体原因，不命名新算法。固定同一60对 COCO/PACO/SUIM、每对 A/B 两真任务、原始 O24 和原生黑图 basis；无新 DINO 编码、无跨episode拟合。参考 mask 用 INSID3 官方 `load_mask`/`downsample_mask` 变成64²硬标签，APD 的 query token 到 reference token 1-NN 得到 `z_Q∈{0,1}`。query 图只用原始单位 O24 cosine，去self，固定32-NN与温度.07的行 softmax，得到行随机 `P`。固定点 `f=.5z_Q+.5Pf`，到最大残差 `<1e-7`；不读 query 标签，不加别的 unary。唯一三臂是 `cross_only=z_Q`、真实 `P` 的 `f`、同一图以 seed0 全局节点置换得到 `P_null=ΠᵀPΠ` 的 `f_null`（A/B共用置换，`z_Q`不动，同谱/同权/同吸收率）。置换操作和全0/全1/常量固定点、概率界、残差须先经合成检验。

预先固定两种独立读出：64²字段 bilinear1024 后 `>0.5` 为主；同字段先64格 minmax，再 bilinear1024 后 `>0.5` 为零点影响控制，常量字段保持原值。不得按域挑最佳，不扫 k/温度/吸收率/阈值。完整 INSID3 官方 bilinear、FoRIS pre/CRF 只读复用上轮同输入封存，CRF 不与无CRF主臂混为同机制差。所有字段、完整mask、原始输入和源码 SHA 先封存，之后才打开 query GT；报告同域40任务及按实际 R/Q 照片身份成组的配对区间、完整1024 I/U/TP/FP/FN，并在64格按真实像素质量报告全图、旧FoRIS前景内、旧FoRIS漏检区对非目标背景的排序。最后一项是事后诊断，非输入。真实图若不胜同谱置换与 cross-only 的困难区排序，就不称发现稳定 query 结构信号；即使胜也须在完整输出超过强基线才考虑推广。旧 COCO241 joint23.170、FoRIS pre58.562 及 reference边质量中位.000161是既有反例，本实验不复述为未试的新方法。

## [Sol 缓存实验] 分尺度吸收图的完整结果与机制边界

冻结的唯一经典控制已完成同60对、A/B共120任务，零新DINO编码。合成常量固定点/概率界/残差/置换共轭和谱先通过；首个 v1 真实试跑因 MPS FP32 softmax→CPU FP64 行和约1e−7的偏差触发严格界，保留失败局部记录，另开 v2 精确归一化行权重后冻结并全量运行。v2 缓存 MPS+CPU 推理19.626秒，MPS driver观测峰1.480GB；全120任务240固定点最大残差9.99e−8、至多17轮。60图/字段/预测、120 GT的SHA，60置换图、720完整1024mask逐位重建、240固定点均后验验证。路径 `../cv_data/a/heterogeneous_harmonic60_v2_20261010/`，完整证据见 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)。

|40 A/B任务/域的完整1024 IoU平均；均无CRF，FoRIS CRF另列|跨图1-NN z|真实Q图 f|同谱置换 f_null|INSID3默认bilinear|FoRIS pre|FoRIS CRF|
|---|---:|---:|---:|---:|---:|---:|
|COCO|42.583|43.835|38.564|46.513|54.776|55.854|
|PACO-Part|43.788|44.687|39.296|44.637|46.772|47.682|
|SUIM|46.545|47.075|41.818|54.141|59.562|60.239|

真实图对同谱置换图主读出完整IoU差 `+5.271/+5.391/+5.256`，按 R/Q 实际照片连通组件20/18/20组的95% bootstrap区间 `[+1.959,+8.923]/[+3.610,+7.104]/[+3.436,+7.226]`；对仅跨图 z 差 `+1.252/+.898/+.529`，SUIM该差区间跨0。真实图相对 z 的平均1024 TP增656/1160/1179，FP减2570/3301/2292，但仍比FoRIS pre低10.941/2.085/12.488 IoU。64-token、按真实1024像素质量加权的困难区排序另看：在FoRIS CRF前景内真实图对置换图AUC差 `+.135/+.099/+.083`，在FoRIS漏检目标对其余FoRIS外背景差 `+.281/+.324/+.280`，这些组件区间均为正。z是二值且大量并列，同谱置换才是更有力的传播/连续分数对照。

**解释：**分开归一化跨图身份与query内部几何，确实避免旧 joint v1 的近乎无参考导通并恢复可测真实邻接结构信号；它没有解决可靠覆盖和绝对零点。固定 minmax读出会大幅提高置换图，真实图仍仅高置换 `+1.534/+1.389/+1.075` IoU；FoRIS在PACO/SUIM的平均TP仍明显更高。此实验仅是经典图法的必要机制控制，已曝光任务上的完整分数未胜强基线，不构成独立创新算法。

## [Sol 缓存实验] 硬锚定解析限制与实测纠正

对上节的效用解释作关键限定。因 `f=.5z+.5Pf`、`z∈{0,1}`、`0≤Pf≤1`，`z=1` 必有 `f≥.5`，`z=0` 必有 `f≤.5`。实际120任务的真实图与同谱置换图在全部64² token 上 `f>.5` 与原 `z` **逐位完全相同**：两臂均0个token改变、无恰好等于.5；真实图最小FG分数.501923、最大BG.498246。收据 `evidence/local/sol_cache_experiments_20261010/harmonic_token_lock.json`。所以前节真实图的 AUC优势反映硬标签两侧的连续 margin 排序，完整1024 IoU差来自 bilinear 边界/形状和 minmax读出，**不证明修正了任何1-NN token身份**。这也解释为何主臂难补大量 FoRIS FN；该硬锚定控制至此结束，不事后改变吸收率/阈值。若检验图能否改 token 身份，必须另冻结有软跨图锚定且强匹配对照的实验；当前结果不可替代它。

## [Sol 缓存实验] 冻结：一次软跨图锚定的必要对照

这是在硬锚定结果与代数锁定被看到后提出的**曝光开发对照**，只为关闭图机制问题，不是新的方法搜索。同60 pair/120 A-B任务、同O24和官方 hard reference token labels，复用上一轮逐SHA封存的真实 `P` 与同谱 `P_null`，绝不重建或改图。唯一变更：APD Q→R cosine 的32-NN（去不去self无对应同图self）用固定温度 `.07` 行softmax，以硬参考标签加权成连续 `p∈[0,1]`；保存跨图邻居和权重以复核。三臂为 `p`、`f=.5p+.5Pf`、同一 `p` 上的 `f_null=.5p+.5P_null f_null`；沿用主 bilinear1024 `>.5` 与64格minmax后 bilinear1024 `>.5` 两读出。无新编码/CRF/参数搜索。合成先验证 `.6` anchor 与0邻域可跨`.5`翻成背景、概率界、固定点残差；所有场与完整mask先封存后评分。评分要分别报告真实图与null的全图、FoRIS前景和漏检区排序，以及64 token 相对 `p>.5` 的GT面积加权纠正/误改、完整1024 IoU/TP/FP/FN；照片连通组件20/18/20组。若真实图不稳定优null或仍低完整强基线，就关闭，不扩量/扫k/阈值。


## [Sol 主实验验证] 任务核的完整阈值能力：已封存后GT上界

**事实，已独立复算。** 本项没有新模型或字段；对已有三域60对、120任务、五臂共600个signed场，在原bilinear1024完整像素上处理所有ties，独立选取GT最佳IoU level-set阈值。query GT只作隔离oracle，不能把oracle阈值拿去推理，也不能视为方法分数。

|域|任务核原完整IoU|任务核GT上界|去位置O24上界|隐藏激活上界|无mask梯度上界|
|---|---:|---:|---:|---:|---:|
|COCO|41.172|58.613|62.714|57.431|58.663|
|PACO|32.661|59.069|62.207|59.202|59.637|
|SUIM|46.205|64.608|65.758|64.781|64.654|

**解释边界。** 原零点损失确实较大，但即使放开每任务阈值，任务核相对去位置O24的COCO/PACO点差为−4.101/−3.138，照片连通组95%区间分别[−7.818,−1.056]/[−5.296,−.782]；SUIM−1.150区间跨零。不能把任务核劣势全部归给校准，也没有显示出比mask-free tangent更好的固定排序上限。任务核oracle均值超过完整FoRIS，并不等于可用增益；19/40、13/40、16/40任务即使oracle仍低于完整FoRIS，单阈值不能补齐所有任务。

**统计更正。** 本项及Sol得失分报告使用真实photo ID，空ID回退已核验RGB SHA；三域20/18/20个连通照片组。原producer将SUIM空ID变成str(None)误并1组，是代码问题，不是数据独立性事实。旧mask／GT／I/U和点估计保持不变，受影响区间按修正附录及本报告为准。

生产1980检查，独立ascending-deletion扫描2580检查通过，最优IoU误差1.42e−14。生产38.667秒、独立复算37.424秒，CPU只读字段，无encoder、核拟合、CRF或新预测。完整逐任务cut、TP/FP/FN、与原IoU／完整FoRIS差和强简单对照区间见 [Sol上界报告](../../../evidence/local/sol_validation_20261010/TANGENT_THRESHOLD_CEILING.md)；JSON与独立验证收据在同目录。此事实用于Astra提出下一可证伪方案，不由Sol据此扫描核参数或挑各域最佳renderer。

## [Sol 缓存实验] 软锚定对照结果：真图有信息，完整方法仍未成立

该对照**在硬 z 结果及代数锁定后设计，属于曝光开发实验**。合成 `.6` 跨图anchor在0邻域下可传播到`.3`并翻转 token。首个soft producer在近纯FG的 FP64加权行和略超1时停下，保留局部试跑；v2只将该浮点误差截到合法 `[0,1]`，新目录完成同60对/120任务。直接复用硬实验封存的60个真实P/P_null，保存每Q的APD跨图32-NN索引及温度`.07`权重，source仍为INSID3官方硬reference mask。新增编码0，预测读query GT 0；缓存MPS+CPU15.369秒、MPS driver观测峰1.480GB。封存后核对父图及cross/field/pred各60 SHA、120 GT SHA，重建120个soft `p`、720张1024mask逐位一致，240固定点最大残差9.97e−8。证据在 `../cv_data/a/heterogeneous_soft_harmonic60_v2_20261010/` 和 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)。

|40 A/B任务/域，完整1024 IoU均值；主读出均无CRF|soft cross `p`|真实Q图|同谱置换|INSID3默认bilinear|FoRIS pre|FoRIS CRF|
|---|---:|---:|---:|---:|---:|---:|
|COCO|42.447|42.804|32.374|46.513|54.776|55.854|
|PACO-Part|46.388|46.069|29.974|44.637|46.772|47.682|
|SUIM|50.117|50.286|35.126|54.141|59.562|60.239|

**真实token身份可变，但收益没有兑现为稳定完整胜利。** 真实图相对 `p>.5` 每任务平均翻转COCO/PACO/SUIM 21.4/37.9/38.6个64格；按真实像素面积加权的正确编辑（新增TP+删FP）3787/6322/5985，错误编辑（新增FP+删TP）1698/3367/3897像素。20/18/20实际R/Q照片连通组件重采样：真实图相对 `p` 主读出 IoU `+.357/−.319/+.169`，区间均跨0。真实图对置换图主读出虽高10.429/16.095/15.160点，但预声明minmax读出使置换图大幅回升；同minmax下真实图仍只高1.019/1.247/1.200点。这区分了图结构与零点/面积效应。minmax真实图 COCO/PACO/SUIM 47.013/48.177/51.526，PACO比FoRIS CRF47.682高+.495，但18组区间[−6.498,+7.979]，COCO/SUIM仍低FoRIS CRF8.842/8.713，不能择域宣布领先。

64格、以真实1024像素质量加权的困难区排序：真实图对置换图在旧FoRIS漏检目标对其余背景AUC高 `+.164/+.166/+.171`，组件区间均为正；旧FoRIS前景内部目标对误报高 `+.047/+.020/+.032`，仅COCO区间为正，PACO/SUIM跨0。说明真实Q邻接确有局部目标相关信息，却不足以在三域困难区稳定排错或在完整输出上超过FoRIS；到此关闭本轮固定图机制对照，不扩样本、不扫k/温度/吸收率/阈值，更不包装为独立创新算法。


## [Sol 主实验验证] 参考Mask切换的图传播与读出定位

本项只读已封存60对、120个合法A/B任务的三臂字段和完整1024掩码，无新模型、编码、图构建、CRF或阈值搜索。样本已曝光，GT仅用于事后诊断。

评价两个目标的独占区域，重叠区域不计；每照片对A-only与B-only等权，随后按实际R/Q照片连通组件20/18/20组统计区间。该身份/覆盖指标不包含其他背景，不能代替完整IoU。

连续切换AUC是A场区分A-only/B-only、B场反向区分的平均；响应AUC是同一Q的场差s_A−s_B区分A-only/B-only。完整掩码的“唯一正确”要求该目标对应的mask选中、另一任务mask不选中；另列唯一错误、两任务皆前景及皆背景，四者逐任务合为1。

|域|跨图场切换AUC|真实图切换AUC|真实图−跨图及95%区间|跨图场唯一正确|真实图唯一正确|真实图两任务皆漏|
|---|---:|---:|---|---:|---:|---:|
|COCO|0.89438|0.92359|+0.02922 [+0.01298,+0.05101]|66.56%|66.03%|27.42%|
|PACO-Part|0.86721|0.90625|+0.03904 [+0.00744,+0.06508]|59.19%|58.13%|38.84%|
|SUIM|0.81134|0.86303|+0.05168 [+0.02219,+0.08252]|57.76%|57.60%|34.81%|

三域真实图的连续身份排序均优于跨图场，照片组区间均为正；但固定概率读出下唯一正确覆盖均未增加，PACO下降1.059个百分点，95%区间[−1.752,−.434]。跨图起点本身已有26.86%/37.49%/34.10%的独占目标在两个任务mask中都漏掉，故不能把原始失败全归于图传播。真实图默认概率完整IoU仍是42.804/46.069/50.286，完整FoRIS为55.854/47.682/60.239，来自父封存报告，未产生新算法分数。

预声明minmax控制的唯一正确覆盖为71.63%/61.05%/58.81%，比跨图控制高.940/1.466/.787个百分点；SUIM区间跨0，且该指标排除了所有其他背景，因此不能据此把minmax挑成完整分割方案。真实图的响应AUC为.95535/.95014/.86283；对同谱置换图的响应AUC余差在COCO/PACO区间为正，SUIM跨0。这里存在参考条件排序信号，也存在覆盖失败，不能概括为模型完全不响应Mask。

固定P下，保存的f满足f=.5p+.5Pf，故连续跨图场可由p=2f−Pf复原；同一Q两参考任务的差同样满足此恒等式。180个保存场均保留非零任务响应，真实/置换传播差逆算跨图响应最大误差3.70e−7，符合父固定点容差。该可逆性只针对已知P和未二值化的连续场，是经典线性算子的事实，不是原创定理，也不证明语义排序可靠；它排除了此实验中“传播精确抹去全部Mask响应”的解释。

2040项输入绑定、像素四态分区、父全图AUC一致性和场逆算检查通过；其中540次AUC用独立tie-group pairwise-win公式核对sklearn ROC积分，最大误差2.22e-16。CPU只读分析及组bootstrap4.946秒，新增编码和模型推理均0。

下一验证应区分：跨图起点缺少正确目标支持、连续分数有身份但无可靠绝对零点、以及其他背景误报。任何候选都须在相同完整输出及强对照上验证；此报告不授权按GT调阈值、扩样本或改传播参数。


完整资产见 [任务切换报告](../../evidence/local/sol_validation_20261010/GRAPH_TASK_SWITCH_AUDIT.md)。本节只追加Sol核验，没有新预测或公式候选。

## [Sol 缓存实验] 冻结：soft vote丢弃的绝对匹配幅度诊断

只做后验**证据分析**，不生成新候选mask或做DINO编码。旧 role-transfer 同批Q FG/BG方向 AUC COCO/PACO/SUIM `.9666/.9169/.8802`，LVIS1400 mean-guide全图 AUC`.9546`但FoRIS前景内`.7435`；它们都没有回答给定soft membership后绝对幅度是否还有信息。本次同60对/120真任务，复用已封存soft跨图32-NN索引/权重、p、真实图f与同批FoRIS CRF ROI。原生INSID3 APD的R/Q单位token和官方hard reference mask构成 `c_F=cos(unit_mean(R_APD[FG]),Q_APD)`；`c_any=.07·logmeanexp(top32_APD_cos/.07)`只衡量到**任意**参考token的绝对匹配，不解释成FG语义。top32 cosine与保存权重重构比对；无source-BG高维训练、无阈值/公式搜索。p和f只是固定已有字段。

先在不碰query GT的阶段封存 c_F/c_any以及原p/f绑定，随后才评分。全图、旧FoRIS CRF前景内（TP对FP）、旧FoRIS漏检目标对其外TN三ROI用64格真实1024像素质量报告同任务 AUC。主要端点：按每任务 p 的4096 query token排序，等量切为10箱，分箱完全不看GT；每ROI/每箱若FG或BG质量为0则不贡献。箱内分别重算 c_F、c_any、p和f的FG/BG加权AUC；以该箱 FG质量×BG质量作pair-mass权重，先聚合每任务，再两任务/照片，最后按真实R/Q照片连通组件20/18/20组 bootstrap。记录10箱有效数、有效task、有效pair-mass与可比任务集；不筛p>.5，不挑有利箱。若 c_F 无稳定增量，不构造产品分割公式；若有，明确ROI/域/反例，仅请Astra决定下一步。此诊断使用queryGT只在seal后，不能当推理阈值。

## [Sol 缓存实验] 绝对FG匹配幅度的条件排序结果

同60对/120真任务的**只读证据诊断**已完成，零新mask、零DINO编码。旧soft cross文件仅存APD32-NN索引/权重，无绝对cosine；本轮从同raw按原矩阵乘法重建top32cos并封存。`c_F=cos(unit_mean(R_APD[hardFG]),Q_APD)`严格对应INSID3固定参考FG原型；`c_any=.07·logmeanexp(top32cos/.07)`只表示匹配任意R的幅度。按邻居einsum的首个部分试跑因数值路径令权重误差3.18e−6超过预设2e−6，保留不评分；v2原矩阵乘法的权重重建误差最大3.43e−8。v2 `../cv_data/a/absolute_match_condition60_v2_20261010/` 全字段在GT前seal，预测GT读取0；缓存MPS+CPU11.316秒、driver观测峰1.150GB。评分校验60字段/父cross/父field、120GT SHA，重放120个p/f及60个c_any。完整证据见 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)。

关键端点只按每任务p排序等量切10箱，不看GT选箱；每箱在全图、原FoRIS CRF前景内TP对FP、FoRIS漏检目标对其外TN，用真实1024像素质量算加权AUC，箱内FG×BG pair-mass加权到每任务，再A/B照片和R/Q连通照片组件20/18/20组 bootstrap。有效任务：全图40/40/40；FoRIS前景39/38/39；漏检38/38/40。全图有效箱均数4.68/5.53/7.18，有效pair-mass仅占所有跨类pair `.067/.059/.057`，因此下面是**同p局部排序**，不是完整IoU或所有像素对。

|条件AUC均值，COCO/PACO/SUIM|p|真Q图f|c_F|c_any|
|---|---:|---:|---:|---:|
|全图同p箱|.756/.674/.657|.848/.755/.788|.882/.813/.778|.724/.658/.536|
|FoRIS前景 TP对FP|.597/.647/.612|.655/.679/.654|.690/.729/.612|.652/.566/.542|
|FoRIS漏检目标对其外TN|.647/.598/.592|.826/.782/.805|.831/.754/.796|.604/.582/.499|

`c_F−p`在全图同箱AUC `+.126/+.134/+.122`、漏检区 `+.194/+.167/+.204`，组件区间均为正；在真正候选误报区 `+.095/+.089/+.005`，COCO/PACO仅小幅正，SUIM区间 `[−.063,+.072]`。更严格地，`c_F−f`在全部三ROI及三域的组件区间**全部跨0**：相对已用真实Q结构的f没有稳定额外排序优势。`c_any`普遍低于`c_F`，任意参考匹配幅度不能当FG语义。反例 `suim_003/A` 在FoRIS前景内 `c_F−p=−.614`（7有效箱），同一图B为+.444。故被softmax丢掉的FG定向幅度在固定p下仍含局部信息，但现有证据不足以推断它能稳定解决FoRIS候选误报或补足f；不据此构造部署公式/阈值。

**[Sol 统计补正，条件幅度诊断]** 上节首版 `report.json` 在一个ROI中A/B只剩一项有效时，直接以有效task均值形成组件，而冻结设计要求先A/B成照片再按连通组件。所有字段、箱内AUC和有效任务不变；规范[逐照片收据](../../evidence/local/sol_cache_experiments_20261010/absolute_match_photo_weighted.json)已按此重算。PACO FoRIS前景内的 `c_F−p` 组件均值由+.089修为 **+.086**，95%区间由[+.012,+.178]修为 **[+.008,+.175]**；漏检区+.167修为 **+.166**。COCO/SUIM和总体判断不变。逐照片条件AUC表与这项补正以[Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)为准，旧 `report.json` 原样保留初版任务账，不伪改生产封存。

## [Sol 缓存实验] 以真实 query 图字段 f 条件化的逻辑核查

前一节 `p` 等量箱内 `c_F` 与 `f` 的AUC差区间跨0，**不能证明 `c_F` 对 `f` 冗余**：同AUC的两个字段可能错在不同像素。最后的只读核查只将分箱变量改成 `f`，固定同10个Q token rank等量箱、同stable tie、同全图/FoRIS前景/FoRIS漏检-vs-TN ROI与FG×BG箱内pair-mass；箱号仅依赖f、不看标签。全部字段来自原seal，无新编码、mask、预测或参数。双任务A/B均有效的照片才进入主均值，单任务有效单列；之后按真实R/Q照片连通组件bootstrap。检验60字段/父字段SHA、120 GT、120个f绑定和120套10箱。完整逐task及统计在 `../cv_data/a/absolute_match_condition60_v2_20261010/report_condition_on_f.json` 与 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)。

|f等量箱条件AUC，A/B均有效照片均值；COCO/PACO/SUIM|p|f|c_F|c_any|
|---|---:|---:|---:|---:|
|全图|.732/.645/.622|.785/.689/.683|.856/.791/.732|.762/.637/.544|
|FoRIS CRF前景 TP对FP|.587/.646/.603|.636/.663/.627|.697/.730/.617|.654/.560/.521|
|FoRIS漏检目标对其外TN|.613/.551/.553|.704/.642/.656|.770/.665/.714|.681/.592/.516|

双有效照片数全图20/20/20，FoRIS前景19/18/19，漏检18/18/20；对应有效连通组件全图20/18/20、前景19/17/19、漏检18/16/20。全图同箱有效pair-mass只占跨类pairs `.065/.059/.056`，前景`.703/.453/.462`，漏检`.078/.078/.086`，因此仍是局部排序。`c_F−f` 组件均值/95%区间：全图COCO `+.072/[+.025,+.117]`、PACO `+.095/[+.039,+.154]`、SUIM `+.048/[−.007,+.107]`；漏检区仅COCO `+.066/[+.032,+.102]` 稳定；**FoRIS前景TP对FP**三域 `+.062/[−.028,+.150]`、`+.066/[−.005,+.141]`、`−.010/[−.087,+.076]`，均跨0。反例 `coco_010/B` 的FoRIS前景 `c_F−f=−.447`（2有效箱），`suim_003/A` `−.631`（4有效箱）。所以给定f的粗箱后，FG定向绝对幅度在部分域/ROI仍有补充排序信号；关键候选误报ROI尚无跨域稳定优势。此10箱比较不是精确条件独立检验，不构造融合公式，亦不把先前AUC差跨0解读为已证明无额外信息。


## [Sol 主实验验证] 32近邻支持不足与参考硬标签footprint

只读已封存三域60对/120合法A/B任务的32近邻、R硬标签、三个场及完整1024输出；不构建新模型、图或预测，不新增编码。Q GT仅事后评分。本批已曝光开发样本，不能称独立确认。

每query16×16像素格按32个实际参考邻居分为无硬FG、混合且FG票≤.5、混合且FG票>.5、全硬FG。GT与保存完整mask的FN均以真实像素计数，四类精确合为总量，不用格点数量代替目标面积。

|域|目标无硬FG近邻|纯目标格无硬FG近邻|soft完整FN来自无FG类|soft完整FN来自混合低票类|R真FG面积落在硬BG格|
|---|---:|---:|---:|---:|---:|
|COCO|12.28%|10.04%|35.00%|53.01%|12.54%|
|PACO-Part|18.88%|17.68%|30.95%|60.23%|6.58%|
|SUIM|21.36%|18.68%|38.04%|52.75%|6.30%|

以上为两任务/照片平均，区间依实际R/Q照片连通组件20/18/20组。比例分母为0时标缺失：纯目标格指标有效照片18/20/19；完整FN来源COCO有效19对，其余20对。逐项有效照片数、组数和CI均在JSON，不能把不同分母合成一个统一比例。

多数漏检不是没有任何前景近邻，而是有前景支持但票不超过一半。没有硬FG的query目标又分为其参考邻居footprint存在真实FG、以及完全不触及真实FG：前者占全部目标2.18%/3.76%/4.35%，后者10.10%/15.12%/17.01%。参考硬标签不等于整个16×16特征footprint：R真实FG面积12.54%/6.58%/6.30%落在标成BG的格。但这些数只描述物理重叠；不能把邻居含一像素FG解释成正确语义对应，也不能推出软coverage必定修复。

纯目标格仍有10.04%/17.68%/18.68%没有32-neighbour硬FG支持，故不能把全部问题解释成query边界混格。另一方面，“无32-neighbour FG”只针对这一截断对应，不等于DINO没有可用任务信息。完整FoRIS作为封存后外部对照，在这些区域的像素合池召回为24.60%/36.61%/29.80%，且找回soft输出FN的24.28%/36.58%/29.73%；没有把FoRIS字段用作新算法输入。

条件照片平均的FoRIS召回分别32.87%/37.55%/29.52%，有效17/17/20对、17/15/20组；soft-FN找回比例30.33%/38.63%/28.23%，有效17/16/20对、17/14/20组。它们与像素合池分母不同，不可混用。两种量具都说明全局角色/区域证据可以跨过本地近邻限制，仍未给出可部署独立修复。

生产2220项绑定、官方CPU硬mask重建、邻居票重建、四类及原FN/TP一致性检查通过；独立将类别直接铺到1024像素后2580项重算完全一致，并逐位核对参考FG落入硬BG格的真实面积。生产1.790秒，独立1.608秒，过程峰RSS0.496GiB，无MPS、raw读取或新encoder。

下一固定控制只检验既有源密度假设：在相同32邻居、同P和同固定读出下校正合法参考FG/BG数量比例，并保留原cross/graph、平衡cross及参考标签数量不变的置换控制。不扫k、温度、传播系数或GT阈值；若只是改变覆盖而完整输出/FP不改善，关闭这一解释。不把类平衡这一传统操作称为原创算法。


完整数据见 [支持报告](../../evidence/local/sol_validation_20261010/CROSS_SUPPORT_AUDIT.md)。本节为Sol固定缓存量具，无新算法或编码。


## [Sol 窄先验审查] 参考 mask 进入冻结主干的跨 token 注意力

**最近先例，限定原创性。** [DiffewS 原文](https://arxiv.org/html/2410.02369)已经把 support 的 K/V 拼进 query 自注意力，并比较过“support mask 限制可见 K”等输入方式；但模型是扩散 UNet，做了带 query-mask 监督的任务训练（论文报告约 10k/30k 迭代），不是冻结 DINO 单主干的免训练结果。[EMAT 原文](https://arxiv.org/html/2507.23642v1)以冻结 DINOv2 提特征，让 support mask 直接筛选后续相关性 transformer 的注意力条目；该 transformer 训练 80 epoch，mask 并未进入冻结 DINO 主干。[INSID3](https://arxiv.org/html/2603.28480v1)与[FoRIS](https://arxiv.org/pdf/2609.03384v1)是当前更贴近的一主干免训练基线，但都在编码后用 mask 选择参考标签/原型/候选，未让 mask 条件化 DINO 的 token 交互。上述原文已覆盖 KV 交互或 mask 参与注意力这一一般构件；本次窄查未发现“冻结 DINOv3 的真实参考 mask 在主干内改变 Q↔R 注意力、无需训练、以完整单参考 ICS 评估”的成熟全法，**不把未查到等同于不存在**。

**已有资产允许和不允许的结论。** 当前封存 `../cv_data/a/task_tangent_features60_20261010/COMPLETE.json` 只列 119 个唯一输入的 `mlp_input / mlp_preactivation / mlp_hidden / block_output / O24`（均为 4096 patch token）；采集源码 `scripts/prepare_task_tangent_features.py:79-125` 明确 `mlp_input` 是末块 `norm2` 输出，位于注意力**之后**，且缓存删除了 5 个 prefix token。末 MLP 的核或这些后态不能重构先前 QKV、注意力 softmax 或受 mask 干预后的残差。`src/ics/data.py:13-22` 使用 timm `vit_large_patch16_dinov3` 的 24 块 `forward_intermediates(...,norm=True)`；末块真正干预至少需要该块输入的**全 token（含 prefix）**、`norm1`/QKV 路径与固定权重，最好从真实前向捕获并核对未干预输出逐值一致。O23/pre-last-block 输入若未保存，就必须重新真实编码；不得把现有 `norm2`/O24 伪称 pre-attention。旧 `joint_coco150/lvis150` 是无 mask 的 R+Q RGB 并排前向：原图 back_auc 候选内排序约 `.6837→.6831`、`.6855→.6580`（分开→联合），不支持“光并排就改善身份”；旧 `maskcue` 将 R 的 FG/BG RGB 保留、其余置零后重新联合编码，虽有 mask 影响输入，却同时改变颜色/能量/上下文，20例控制未给稳定胜基线的身份结果，不能作为纯注意力因果证据。见 `docs/research/OPUS55_STATUS_20261010.md:41-132`、`scripts/opus55_mask_cue_probe.py:220-229`。

**可检验前提与决定。** 唯一合理的小干预是预先固定**末块**这一位置（最少新捕获，前23块保持原始语义），仅使 query token 对参考 patch key 的可见性随合法 mask 切换；Q→Q 路径、原图 RGB、参考/查询画布与模型权重固定，并保留等面积错误/平移 mask 控制。必要预测不是“Q 随 mask 变化”，而是同 Q 两个真实任务切换后，目标相对同父细部/误报的**候选内排序**按正确方向翻转，且完整输出相对原 FoRIS 和错误 mask 控制不退化。需同时量真实 Q→R 注意力质量；若原本几乎不看 R，或仅产生位置/面积效应，单末块干预应直接止步，不扫层/注入强度。这是一个**可做但必须新捕获的窄机制实验**，不是从现存 cache 可无编码完成的严格干预，也不是已成立的新算法。若研究不接受这一预先固定位置与真假 mask 配对预测，应否决此方向，避免任意选择层和注入系数。


## [Sol 主实验验证] 固定类平衡验证源密度假设：未形成跨域方法

本项落实共同文档中已有的参考FG/BG密度假设，固定同60对、120合法A/B任务。只把已有32-neighbour posterior的源类比例校正为等类先验，原P、吸收率.5、残差容差1e−7和FP64 bilinear1024 >.5均不变。主臂事先固定balanced_graph，简单控制balanced_cross；原cross/graph逐位重放，另有保参考FG数量与A/B重叠的同seed0标签置换。没有新DINO/raw、query标签或FoRIS输入，没有扫参数；经典控制不是原创算法。

合法参考pi=hard64前景比例，平衡票 `b=p(1-pi)/(p(1-pi)+(1-p)pi)`。无FG/BG的退化参考回退原p并记录。该公式等价于分别除以R两类数量后归一化的参考邻居票；单图/照片先验改变不意味着query真实前景比例相同。

|域|原cross|原graph|平衡cross|预定主臂平衡graph|平衡参考置换|完整FoRIS|O24 ridge|去位置O24 ridge|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|COCO|42.447|42.804|39.381|41.120|4.939|55.854|45.851|48.802|
|PACO-Part|46.388|46.069|44.691|45.811|10.863|47.682|41.879|39.417|
|SUIM|50.117|50.286|53.943|54.814|18.107|60.239|47.104|50.044|

口径：同一照片两个合法任务的CLI1024完整IoU平均，不是官方全class mIoU。外部FoRIS与两种O24控制只在预测封存后读取经过核验的同输入结果，均核对当前GT面积及I/U/FP/FN恒等式。

|域|平衡主臂−原graph及95%区间|平衡主臂−完整FoRIS及95%区间|平衡主臂−平衡cross及95%区间|
|---|---|---|---|
|COCO|-1.684 [-6.944,+4.806]|-14.735 [-20.183,-9.672]|+1.739 [+1.048,+2.511]|
|PACO-Part|-0.258 [-5.615,+4.170]|-1.872 [-8.392,+4.640]|+1.120 [+0.431,+1.945]|
|SUIM|+4.528 [+0.302,+9.654]|-5.424 [-10.500,-0.477]|+0.872 [+0.376,+1.397]|

照片连通组件COCO/PACO/SUIM为20/18/20。主臂仅SUIM相对原graph有区间为正的收益，但仍低完整FoRIS5.424点；COCO/PACO相对原graph点差为负，不能按域挑最佳臂或扩大这套配方。真reference优于数量匹配的置换（分别+36.180/+34.948/+36.708）说明标签身份必须正确，不能把涨分全部归给前景面积。

COCO平均FN从20,373降至9,480，但FP从36,674升至91,969；召回.660→.865，精度.512→.462，完整IoU反降。PACO平均FN51,419→30,628、FP34,861→84,105，IoU也未提高。SUIMFN87,356→61,165、FP46,444→71,766，有局部面积/召回收益，完整FoRIS仍有更好精度与IoU。平衡后graph的全图目标/背景AUC相对原graph为COCO−.007207、PACO−.001761，两域照片组区间为负；SUIM+.000121区间跨0，故未形成新的稳定判别收益。

**源面积方向的解析与完整实测。** pi<.5时b≥p，pi>.5时b≤p；固定P的调和解具有非负算子，故同一固定读出只会按该方向扩张或收缩。完整120任务实测COCO40扩/0缩、PACO39扩/1缩、SUIM37扩/3缩，全部与原graph满足严格mask包含关系。这解释了小reference mask校正时无力删掉已有误报，不能把扩张自动写成理解改善。

**无支持锚定的读出锁定。** p=0仍有b=0，而f=.5b+.5Pf、0≤f≤1使对应token的f≤.5；它不能在>.5下变FG。120任务354,549个零票节点实际0个翻FG。p=1节点相应f≥.5，33,566个节点没有翻BG，实测最小f=.595987。这不是连续场精确擦除信息（已知P可以逆算），而是固定锚定和二值读出阻止未知支路纠错；要验证纠错须另有明确的非局部任务支持/不确定性依据，不能只调传播系数。结合支持量具，纯目标格仍有10.04%/17.68%/18.68%无32近邻FG，当前类平衡不会解决这个缺口。

**数值边界。** 点状类平衡在精确算术下单调，保存场排序实际0次反转、90组新增tie；端点有限精度会把少量不同p合成同一值。因此直接cross AUC不是逐位严格不变，120任务最大差.0009263；不能把该舍入差称为新信息。主臂graph的真实排序变化另行报告。

推理前配置/源码冻结，600完整mask封存后评分，其中240为父控制逐位重放、360为新固定控制输出。query GT访问尝试0，encoder/raw新增0。缓存推理3.918秒，峰RSS0.478GiB；评分5.627秒。1920项绑定、直接类计数票重建、固定点残差、600mask重建及完整计数检查通过；直接平衡票最大误差9.99e-16，最大残差9.96e-08。另120方向包含核验通过。总资产18.68MiB，不复制原raw、P或DINO权重。

本控制未通过，关闭此配方；不扩大、不扫类先验或按域挑renderer。研究目标仍未达到。下一规格应处理“没有局部FG支持”与“有确切BG证据”的区别，并检验全局参考任务证据能否正确触及新支路且排除相似背景；原单纯类比例和单参考自拟合都不足。

源码：[固定实验](../../../scripts/sol_class_balanced_cross_control.py)。机器大数组：`../cv_data/a/class_balanced_cross60_20261010/fields/`、`predictions/`，输入指向原soft_harmonic的cross及hard_harmonic的P；`config.json`、`sealed.json`、`records/`、`report.json`保存输入/源码/输出SHA。复算需要该源码SHA及原两组父缓存；不要覆盖已存在的目录。

完整机器资产和家中轻量快照见 [固定控制报告](../../evidence/local/sol_validation_20261010/CLASS_BALANCED_CROSS_REPORT.md)。这是对既有源密度假设的固定传统对照，不称新算法或整体goal完成。

## Astra：本轮理论收敛与下一项密度控制的解析预测

这部分只解释已封存结果和下一项传统对照，不产生新 mask 或编码。后续执行可以直接使用本节及父资产；本轮子代理图与幅度诊断均已结束，已有图、近邻和字段无需重算。

**三个信息层必须分别保留。** 参考 mask 改变条件场、连续场区分两个目标、最终完整 mask 增加目标并减少其余背景，是不同的要求。Sol 的[任务切换核验](../../evidence/local/sol_validation_20261010/GRAPH_TASK_SWITCH_AUDIT.md)已发现真实图使三域的连续切换AUC都上升，但完整唯一正确覆盖未增；已知P时还能以 `p=2f−Pf` 从连续图场逆算原soft场。因此本次失效不能概括为“图把全部mask信息抹掉”。同理，task tangent 的[完整阈值oracle](../../../evidence/local/sol_validation_20261010/TANGENT_THRESHOLD_CEILING.md)高于原零阈值输出，却低于同输入去位置O24的阈值oracle；既有读出问题，也有排序差距，不能把全部失败归给零点。

**硬锚定的限制是解析事实。** 若 `z∈{0,1}`、P非负且每行和为1，`f=αz+(1−α)Pf` 的解在[0,1]。当α≥.5时，z=0的位置满足f≤.5、z=1的位置满足f≥.5；本轮α=.5、120任务的实测严格阈值标签完全相同。因此其1024收益只能来自连续场插值的边界变化，不能证明纠正64格身份。即使换成soft p，只要某格p=0、α=.5，该格仍不可能通过f>.5。缺少正支持被当成了强负锚，这限制了漏检恢复；它不证明降低α就能解决语义错误。

**目标相似与角色偏好有不同含义。** softmax邻居投票p对每个query位置所有cross logits的共同加法偏移不变；它表达邻居标签的相对偏好，未保留该位置与参考的绝对匹配幅度。这个性质是选定读出的事实，不是DINO原始表示缺少该信息的证明。已测c_F给定f后的补充排序只在部分域/ROI稳定，关键FoRIS前景内TP/FP三域区间均跨零；两个字段AUC相近也不能证明它们信息冗余。因此当前不能据此直接合成一个融合公式或宣布统一病因。

一个相关但有严格适用边界的几何恒等式：对单位前景均值F、单位难背景均值B，令ρ=F·B，则单位差分方向 `(F−B)/||F−B||` 在F轴上的分量为 `sqrt((1−ρ)/2)`。同60对的源端ρ中位为COCO .683、PACO .892、SUIM .761，对应前景轴分量约.398/.232/.346。差分会把大量权重转到二者的差异方向；FoRIS的正交BG惩罚在单FG原型的简化情形下保留独立的F项。**F轴未必纯粹是有益语义**，它也含场景因素；已有role-transfer读数不支持把ρ或正交残差放大当成普遍原因。此恒等式只说明应检验共同语义与差异判据各自的作用，不支持按ρ设置新gate。

**给Sol已规划的源标签数量平衡对照：先规定它能与不能证明什么。** 设合法参考硬FG比例为π_R，传统平衡变换为

`p_bal = (p/π_R) / [p/π_R + (1−p)/(1−π_R)]`，其中0<π_R<1。

1. 对单张query它严格单调，故cross-only全图/任意固定ROI的AUC必须保持不变；`.5`读出等价于原p以π_R阈值读出。若仅cross端涨分，应归为源比例引起的覆盖变化，不能称新身份排序。
2. 若π_R<.5，则p_bal≥p。固定同一P后，`H=.5(I−.5P)⁻¹`的每项非负，因此f_bal≥f；在固定`.5`、正权bilinear读出下只能扩张，不能删除既有FP。π_R>.5时方向相反。min–max读出不受此集合单调保证，必须另列。
3. p=0仍为0。它不能在cross端恢复完全没有FG近邻的格；图端α=.5的主阈值也有上述零支持上界。不能把“没有近邻FG”直接判成表示不可识别，FoRIS在这些格已有外部召回反例。
4. 源比例平衡只有在核密度/类条件假设下才有概率解释，并不直接估计query目标面积。完整收益须记补真/补假/删真/删假，特别核查是否只是扩大预测；不扫π修正强度、k、温度或GT阈值。

本批源标签比例预先决定了COCO/PACO/SUIM各40任务中主要扩张与收缩的方向，详细计数及源绑定在[只读理论收据](../../evidence/local/insid3_foris_generalization_20261010/source_geometry_facts.json)。此项只是固定传统控制，不作为原创方案；必要验证结束后不延长为参数搜索。

**当前不启动冻结注意力的新捕获。** 将合法mask作用于真实跨token交互，与终层原型/标量gate确实不同；但本地仅有末MLP后态，缺真实末块attention输入和prefix，无法从现存缓存伪造严格干预。最近先例依赖任务训练，新方向尚欠足够明确的机制证据。若后续提出固定末块的真实干预，必须先说明跨图位置、注意力质量、真假mask配对和完整读出；仅让Q随mask变化不算进展，不能借此扫层或扩编码。

## [Sol 缓存实验] 冻结：top32零FG票的证据语义

只读已封存 `absolute_match_condition60_v2`、soft图跨图邻居/权重/字段、FoRIS trace/CRF与隔离GT；零新mask、零编码、零公式或吸收率搜索。每任务以**真实top32硬reference FG标签计数严格等于0**定义no-support token（4096格中的子集），不以`p<ε`近似；先核对计数与已存p=0，再封存统计脚本/绑定。主要三ROI是no-support全域、其与原FoRIS CRF前景交集（TP对FP）、其与FoRIS外交集（漏检FG对TN）。按GT1024像素质量投到64格，记录零票token数、FG/BG质量、FG中FoRIS已检出/漏掉质量及其占全任务FG比例；不能把queryGT用到推理。

固定比较 `c_F`、仅任意R匹配幅度`c_any`、原真图场`f`、原FoRIS最终连续`score`的各ROI FG/BG加权AUC，并另列FoRIS原`apd.fg`与`apd.bg`两个已存阶段以观察全局FG项是否已有排序；阶段关联不等于单独因果贡献。`c_any`在no-support的漏检FG-vs-TN AUC及其FG−BG加权均值差是“锚强度是否在漏目标上较高”的预声明读数；低AUC不自动说明它不适合做置信度。每任务有效须该ROI FG/BG质量均>0；主估计只保留A/B两任务均有效照片，再按真实R/Q照片连通组件bootstrap，单任务有效单列；报告有效照片、组件、ROI内可比pair-mass和有效质量比例。只解释无局部FG票是未知还是可靠负证据，不产生完整算法IoU。

## [Sol 缓存实验] 严格零局部FG票的只读判定

冻结后仅读旧absolute-match、soft图top32邻居/字段、FoRIS trace/CRF与queryGT，没有新mask、编码或参数。no-support定义是**32个实际R邻居中硬FG计数严格0**，与已有p==0在120任务逐位同一；COCO/PACO/SUIM计数131,180/120,136/103,233，总计354,549个64格，域内占比.801/.733/.630。按原1024 GT像素质量，这些格平均承载每任务真实FG的12.28%/18.86%/21.39%，合计351,863/872,612/1,673,523个FG像素质量。FoRIS CRF在其中实际检回86,554/319,479/498,777，微平均回收比例.246/.366/.298。故“无局部FG票”**不是可靠背景证据**，只说明固定top32相对标签尚无正支持。冻结配置、源码及60 cross/soft/absolute字段、120 GT SHA、120个p零集复核见 [no_support_report.json](../../evidence/local/sol_cache_experiments_20261010/no_support_report.json)。

在no-support内用64格字段和真实1024 FG/BG质量评AUC，主口径只纳A/B均有两类的照片、再按R/Q照片组件bootstrap；单任务有效另列。FoRIS外漏检FG-vs-TN ROI的双有效照片COCO/PACO/SUIM为16/16/20、组件16/14/20，有效pair-mass占相应FoRIS外ROI全部跨类pairs `.326/.399/.378`。各字段照片均值：`c_F .821/.707/.783`、`c_any .538/.478/.407`、旧真图`f .921/.904/.858`、FoRIS最终score`.816/.755/.759`、FoRIS原`apd.fg .804/.718/.717`。f各域组件95%区间 `[.891,.946]/[.867,.939]/[.802,.908]`，说明p=0格依旧存在稳定连续身份排序，固定`.5`锚定的token阈值无法兑现。`c_any`在此ROI的SUIM AUC CI `[.338,.477]`，对漏目标反而**较低**；COCO/PACO区间都跨.5。预声明的c_any漏检FG−TN加权均值差 `+.0156/−.0022/−.0252`，三个区间均跨0。不能把到**任意**R的高匹配强度当作漏目标正置信，低AUC亦不否定它的其它可能用途。

原FoRIS前景内的同zero ROI双有效照片15/13/16、组件15/12/16，pair-mass仅原前景ROI的`.005/.039/.059`；c_F/f/FoRIS score AUC分别 COCO `.512/.547/.451`、PACO `.584/.586/.540`、SUIM `.506/.545/.523`，样本对很少，不能外推为全部候选误报。FoRIS外zero ROI中最终score减旧`apd.fg`配对AUC差 `+.012/+.037/+.042`的组件区间均跨0；`apd.bg−apd.fg`在COCO/SUIM点差+.046/+.066且区间正。已存前缀场只能说明FG、BG及后续读出各有相关排序线索，不能单独归因“FoRIS恢复只来自全球FG”或宣告某模块无作用。完整数字与有效质量账在 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)；本节没有完整新算法IoU。


## [Sol 末块 Q/K 量具规范] 真正预训练关系与 RoPE 坐标

**本机源码事实。** `aidemo` 的 timm 1.0.30 将 `vit_large_patch16_dinov3` 实例化为 `Eva`、24 个 `EvaBlock`、5 个 prefix（cls+4 register）、16 个注意力头/头维64、`RotaryEmbeddingDinoV3`、`rotate_half=True`、fused SDPA。末块 `EvaBlock.forward` 是 `x_attn=attn(norm1(x),rope)`，乘 `gamma_1` 加回残差，再进入 `norm2→MLP→gamma_2`；`EvaAttention.forward` 从 `qkv(norm1(x))` 切出 q/k/v，过 `q_norm/k_norm`（此型号均为 Identity），**仅 patch token**用传入的 RoPE 旋转 q/k，prefix 不旋转，然后 `softmax(qkᵀ/√64)·v`。本型号 `gate=None`、attention 内 `norm=Identity`；但重放应调用真实模块而非手写假设。源码位置：`/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/lib/python3.13/site-packages/timm/models/eva.py:1001-1055`（原生前向及 RoPE 传递）、同文件 `EvaBlock/EvaAttention.forward`、`.../timm/layers/pos_embed_sincos.py:1122-1320`（DINOv3 的按网格中心归一化坐标与 `get_embed(shape)`）。本地包装 `src/ics/data.py:13-22` 调 `forward_intermediates(...,norm=True)`；`scripts/prepare_task_tangent_features.py:79-125` 只钩 `norm2` 以后并剔除 prefix。已有 119 个 raw 的 O24/MLPinput/preactivation/hidden/block_output 均不能逆推末块输入、QKV 或未旋转/旋转后的注意力。

**最小真实捕获，不做分割候选。** 对固定 R|Q 并排 RGB（若复用旧 joint 协议就是 768×1536、48×96 patch），只新增一次原生前向，捕获进入 `blocks[-1]` 前的**全 token** `X23∈R^{1×(5+4608)×1024}`，保存原 RGB/input tensor 哈希、参考 mask 在 48×48 R 网格的真实标签、canvas 与 R/Q patch 索引、模型/权重/timm 版本、`rope.get_embed(shape=(48,96))` 的参数/网格语义，及未干预 O24 校验。用 `blocks[-1].norm1(X23)` 与同一块 `attn.qkv` 重算 Q/K，严格经过真实 `q_norm/k_norm`、对 patch 施加 `apply_rot_embed_cat(...,half=True)`，prefix 原样；逐值对钩到的真实 qkv 输出、未干预末块输出核对。无需保存所有 16×4608² 注意力矩阵；按 Q token 块计算 Q→R logits，若要原生相对**全部**注意力质量则块内还须包括 Q/prefix keys 的 softmax 分母。A/B 真实 mask 在同一 R/Q RGB 与同一 `X23,Q,K` 上只改变参考 key 的标签分组，因而是无新增编码的成对任务读取；首次生成 `X23` 本身**必须新编码**，旧联合 O24 无法补造。一次联合前向后多 mask 共用该状态，比每任务重跑 24 块小。

**两种不同关系，不可混称。** 原生联合坐标 logits 为 `Lᵍ_{hij}=<RoPE_48×96(q_{h,Q_i}),RoPE_48×96(k_{h,R_j})>/8`：R 在左48列、Q 在右48列，prefix 未旋转，且 `X23` 已经历前23块的同一联合画布交互。它才是该无 mask 联合前向末块真正会使用的 Q→R logits；给 R key 划 FG/BG 只是事后合法读数，**不是** mask 已进入网络。`L⁰=<q,k>/8` 用同一 `X23` 去掉**末块** RoPE，隔离末块显式相位影响，但前23块的位置/联合上下文仍在；它不是“彻底去位置”，也不是原生前向。若改用 separate-R/Q 的 `X23` 和各自 RoPE 再交叉点积，则 Q 从未在前23块看过 R、坐标各自在单图归一化；无论旋不旋都只是反事实跨图检索，不可叫 joint 原生注意力。DINOv3 RoPE 相位随**同一画布的相对 patch 坐标**变化，原生横拼可能惩罚跨半幅距离；pre-RoPE 可能改善语义跨图匹配，也可能丢失有效结构，必须由同输入配对结果判别。

**唯一主量与代数风险。** 不按头/层挑选：每个 Q token 定义 16 头平均的类平衡关系 `d_i=(1/16)Σ_h[logmeanexp_{j∈R_F} L_{hij}−logmeanexp_{j∈R_B} L_{hij}]`，分别报 `Lᵍ` 与 `L⁰`；另报真实 softmax 下的 R_F/R_B/全部 R 注意力质量，以显示它是否实际读 R，且单列 FG/BG token 数。前者消掉简单集合大小优势，后者保留原生物理流量；两者不能互换。对同 Q 的 A/B 真任务，检验目标独占区域相对另一目标/同父细部/旧 FoRIS TP-FP ROI 的排序是否按 mask 翻转，并与相同输入的 post-O24 cosine 与合法错 mask 控制对照。仅有 `d_i` 的全图 AUC 或 Q 随 mask 改变不够；需在原困难 ROI 上给出增量，后续完整输出另验。若改做 FG-only value 平均注入，令 `v_F=|F|^{-1}Σ_{j∈F}v_j` 对所有 Q 等量相加，则 `y_i'=y_i+c(M_R)`；同一图中点积到固定方向只会出现共同偏置，归一化/簇分配/面积可变化却未新增局部身份排序。即使按注意力权重注入，也可能主要反映 FG 面积及 RoPE 位置；先做纯 Q/K 诊断，不把“可响应 mask”误报为分割信息。若原生/去末块 RoPE 的 `d_i` 都不能在旧失败 ROI 稳定优于 O24，止于量具，不扫头、层或注入强度。


## Astra：下一项只检验预训练关系，先固定输入与解释边界

新的零支持审计把两个问题分开了。现有图场在没有任何局部FG近邻的漏检区域仍有较高排序，说明这些目标并非全部缺少可用信号；但FoRIS前景内同类位置的任务/误报排序很弱，不能只把锚定放松或把低阈值当成语义创新。`c_any`只是匹配任意参考内容，其与漏目标的关系随域改变。现阶段没有依据把它直接变成负锚置信度。此前源类平衡的完整结果也确认：扩大范围不是统一修复。

**可证伪的新问题是：预训练主干用于token交互的关系，是否保留了O24余弦读出没有用好的任务判别？** 这与终层MLP参数梯度核不同：后者仍从逐token支路构造关系，没有观测末块的Q/K。也不预设attention会更好；它在预训练中用于同图上下文，跨图使用存在分布差异。

为单独检验这个问题，采用同60对/119个唯一输入的原1024单图前向，补取末块前全token状态X23（含5个prefix）。选择separate前向是为了保持全部原RGB、前23块上下文与旧O24一致；**跨R/Q的Q/K点积称为反事实关系检索，不称原生联合attention**。不同时改变画布、分辨率或前23块交互。前述联合捕获规范仍是正确的另一实验，但本项不执行联合前向。当前O24和末MLP缓存缺X23，必须为这一新观测做一次真实补充前向；不是重复提取已有O24用于冒充新信息。119次上限来自输入去重清单，完成项不再编码。

固定最后一个真实block及全部16头，不选层/头、不拟合参考度量。通过真实norm1/qkv/q_norm/k_norm得Q/K；主关系用pre-RoPE跨图logit，控制用两个单图原生64×64坐标下的RoPE logit。后者有同坐标位置偏好，前者仅去掉最后块显式相位，二者都保留前23块的位置/上下文。对每头分别取合法R前景与背景的log-mean-exp差，再对16头等权平均；这是固定的经典类别密度读数，不声称它等于DINO原生输出或可靠后验概率。额外参照是同X23经norm1后的单位余弦，以及旧raw/APD O24单位余弦，余弦温度统一用既有.07。它们用于区分新关系、较早特征与位置修正，不组成参数搜索。

所有字段先于query GT封存。同Q两个合法任务、保两任务面积/重叠的同seed0源标签置换、全图、旧FoRIS TP/FP与FN/TN、严格零局部FG票区域均保留。主要判断看真正困难区域的配对排序及任务切换，而非原R拟合或mask响应幅度；完整1024零阈值和既有统一midrange读出均预先保存，后者仅作刻度控制，不能按域挑选。完整FoRIS和原O24同输入结果只在评分后比较。若这项固定关系没有新判别，就关闭本项，不转成注入强度/头/层搜索；若有，也只得到下一独立算法的依据，不自动获得原创性或多域领先结论。

执行前记录输入/模型/依赖与源码SHA；每个原生O24必须逐值等于旧cache，未干预末block重放必须一致；小块真实Q/K点积另用CPU FP64复算。只保存可重放的新X23和固定参数，不复制旧20GiB MLP cache。单MPS producer、FP32/B1、20GB预算，无权重更新。根负责理论与复核，Sol在独占新目录实现/执行，避免与主Sol已封存的class-balance资产混写。


## Astra：图场携带的是折扣路径证据，不能由高AUC直接推出解锁规则

对固定非负行随机P与0≤p≤1，已有场满足 `f=αΣ_{t≥0}(1−α)^t P^t p`。设d_i是沿P有向边到任一p>0节点的最短步数，则d_i之前各项为0，故 `0≤f_i≤(1−α)^d_i`；不可达时f_i=0。这是经典随机游走/Neumann展开的直接结果，不是新算法。α=.5时，一步外至多.5、两步外至多.25。字段很低可以表示到正支持的路径受折扣，不能直接读作确定背景；反过来，取消折扣也会让相连背景获得正响应，并不识别任务身份。

只读全部120任务的旧P/p/f，逐任务用独立SciPy最短路核对自写BFS，240个graph/field/cross/trace绑定、120个GT、120个固定点残差及深度上界都通过。旧场最大残差9.97e−8，全部满足上界（允许的解误差为残差/α）；未产生新场、mask或编码，CPU3.04秒。本批所有Q节点都可达某个正支持，最大深度COCO/PACO/SUIM16/14/15，故仅凭“存在连通路径”不能拒绝背景。入口 [graph_support_reachability_audit.py](../../scripts/graph_support_reachability_audit.py)，[逐任务收据](../../evidence/local/insid3_foris_generalization_20261010/reachability/report.json)。

|严格零局部FG票且位于旧FoRIS外；照片对均值|COCO|PACO-Part|SUIM|
|---|---:|---:|---:|
|原f的目标/背景AUC|.921|.897|.858|
|仅最短步数的反向排序AUC|.877|.865|.832|
|原f减步数排序AUC|+.044|+.032|+.026|
|**相同步数内部**原f的条件AUC|.769|.689|.679|
|同层可比跨类pair质量占本ROI全部pairs|.146|.179|.218|

原f相对纯步数的配对区间三域都为正，同层条件AUC的区间也都高于.5，说明信息不完全是离已有正支持更近；实际边权、路径分布及端点值仍有判别线索。统计为A/B先照片对、有效照片等权、按实际R/Q连通组件重采样：有效照片16/16/20、组件16/14/20。这与前节组件先等权的PACO .904有估计量差异，本节明确采用照片对均值.897，未改变任何字段/GT或逐任务AUC。

然而在旧FoRIS前景内的同zero小ROI，相同步数内部AUC仅`.543/.564/.567`，三域区间都跨.5；其质量和样本范围沿前节限制。图场确有漏检排序，但没有由这项证据得到稳定排误报能力或可部署零点。因而不据此实施“全部p=0改unknown”、按路径长度反除衰减或调α；这些都会改变大量背景，尚缺辨别依据。

## Astra：注意力的先例与反例，限定下一量具的意义

已查阅的原始研究表明，内部Q/K并非一个未经研究的创新入口。[Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814)已经比较DINO的token/query/key/value并将key描述子用于免训练对应、物体与部件共分割；因此换成内部facet本身不构成新贡献。它研究的DINO版本与本次v3不同，不能直接移用最佳facet/层的结论。

更关键的是，[Dissecting Query-Key Interaction in Vision Transformers（NeurIPS 2024原文）](https://proceedings.neurips.cc/paper_files/paper/2024/file/6216515a5e0b3257c49dcb1647e497d1-Paper-Conference.pdf)说明Q/K交互可编码物体与背景、不同部位或不同物体之间的上下文关系。令输入为x，忽略bias时其logit是 `x_i^T W_q^T W_k x_j`；这个矩阵一般既不对称也不半正定，高交互不必表示“同一目标”。论文也观察到DINO等模型的层间趋势存在例外，不能把“末层只看不同对象”当普遍定理。

因此本次固定Q/K检验的失败可能暴露**上下文关系不等于角色同一性**，而不证明主干不含语义；成功则仍须在同mask/同输入的困难ROI和完整输出中成立。输出随合法mask改变、优于置换mask、或SVD模式可解释，都不能单独证明更好的任务迁移。当前不增加facet/层/头扫描，也不以这些先例给负结果翻号。

## [Sol 缓存实验] 固定末块Q/K关系量具：完整负结果

**真实新增观测和核验。** 同60对119唯一原1024单图输入补捕末块前全`(4101,1024)` X23（含5 prefix），119次FP32/B1冻结DINO新编码，非0成本；用时191.890秒，MPS driver峰6.648GB，无权重更新、queryGT读取0。119项原O24逐位同旧cache、未干预末block与QKV/Qnorm/Knorm逐值重放误差0；保存norm1/qkv/原RoPE64与源码/权重SHA。独立NumPy FP64对16头144个小块QK logits核验：pre-RoPE/单图RoPE相对FP32最大误差6.85e−6/8.28e−6。资产 `../cv_data/a/task_attention_relation60_20261010/`，入口 `scripts/prepare_task_attention_features.py` 与 `scripts/task_attention_relation_probe_v3.py`，独立收据见 [Sol报告](SOL_CACHE_EXPERIMENTS_20261010.md)。原relation配置因tuple/list JSON机械不等停止，v2因FP64标量不能送MPS停止，均未预测；v3只作list配置/FP32载入修复后冻结、首例pilot通过并完成60对，未变五臂或阈值。

**固定读出。** separate-R/Q X23经真实norm1/qkv/q_norm/k_norm后，用16头平均、每头FG/BG logmeanexp差的pre-RoPE QK主臂和各自单图原生64坐标RoPE臂；同X23 unit norm1余弦、旧unit O24余弦、旧APD500 O24余弦（余弦/.07）作固定同读出控制。QK跨图点积是**反事实检索，不是模型原生跨图attention**。主臂另用seed0同一reference位置置乱A/B标签，保counts/overlap。全字段及1320完整1024mask先于GT封存，固定bilinear`>0`主读出与minmax64→bilinear`>.5`刻度控制；null只存主zero mask。字段推理105.632秒、MPS峰1.111GB、无额外编码。后验60字段/预测、120GT SHA及1320张mask逐位重建、120官方hardmask/zero子集/置乱验证通过。

|完整1024 IoU，同域40 A/B任务均值|QK pre主臂|QK单图RoPE|norm1 X23 cosine|旧O24 raw cosine|旧O24 APD cosine|QK置乱|INSID3 bilinear|FoRIS pre/CRF|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|COCO|14.714|13.392|24.609|26.996|27.874|5.420|46.513|54.776/55.854|
|PACO-Part|29.207|29.074|30.702|31.086|33.262|7.820|44.637|46.772/47.682|
|SUIM|31.407|27.656|46.510|51.688|55.036|15.276|54.141|59.562/60.239|

主QK的统一minmax臂14.448/24.293/32.527也未救完整端点。与旧O24 APD相比，主QK在FoRIS前景内TP/FP AUC COCO `.617<.729`、PACO `.738≈.744`、SUIM `.627<.733`；前者与后者组件配对差COCO−.112、SUIM−.106的95%区间全负，PACO−.006跨0。FoRIS外漏检对TN AUC主QK `.677/.811/.709`，旧O24 APD `.872/.875/.870`；严格top32零票且FoRIS外主QK `.669/.775/.663`，O24 APD `.807/.806/.821`，没有稳定新的困难区判别。A/B独占任务切换AUC主QK `.649/.851/.739`，旧O24 APD `.917/.910/.871`，连通照片组差−.268/−.059/−.132区间全负。真标签明显优于置乱只证明关系读数响应参考身份，不证明比旧特征更准确。pre-RoPE相对单图RoPE的完整收益仅SUIM +3.751（组件区间[+1.237,+6.278]），仍低强控。

结论仅拒绝**这一固定末块Q/K、FG/BG logmeanexp、跨separate单图检索与读出**；不证明预训练attention普遍无信息。已有同输入O24关系更强，完整输出远低FoRIS，停止该项，不扫层、头、温度或注入。逐task I/U/FP/FN、AUC、任务切换及20/18/20连通照片组件区间在 `../cv_data/a/task_attention_relation60_20261010/relation_v3/report.json`。


## Astra：末块关系的独立复核与本轮决策

独立CPU NumPy实现总体LayerNorm、affine Q/K、half-RoPE和log-mean-exp，在预先固定的三域首对、每图9个均匀query位置上重算主QK、RoPE及主置乱场：18组162个字段值相对MPS封存值最大差`1.7632e−6`。全程未读queryGT、未跑encoder/MPS，未复用生产QK/LLR函数。见 [task_attention_interaction_audit.py](../../scripts/task_attention_interaction_audit.py) 与 [复核收据](../../evidence/local/insid3_foris_generalization_20261010/attention_interaction/geometry_and_readout.json)。完整1320掩码重建由Sol另行完成，二者核验范围不同。

同一份实际末块权重的16头小矩阵SVD给出了源码层风险的量化：`W_q^T W_k` 的奇异值加权左右方向对齐度均值`.166`，头间范围`−.229～.485`；反对称部分占Frobenius平方范数均值`.455`。对称部分本身的负特征值平方能量比例均值`.358`、范围`.108～.706`。这些量排除“所有QK都是同向正定相似度”的代数假设，但忽略bias/RoPE/实际输入分布，**不能把35.8%解释成背景注意力比例或失败因果份额**。它们与真实任务排序劣化相容，仍不能证明一切attention读出都会失败。

实际成本补充：119条唯一输入的原生DINO前向合计174.988秒，每输入捕获/重放/写盘记录合计193.960秒。上一节191.890秒是完整续跑的wall时间，首两例smoke的3.404秒单列，不能把它当包含所有初始化/试跑的端到端总时长。新X23文件合计1,998,941,056字节；旧MLP cache未复制。关系读出105.632秒另计，之后全部运算可复用该新缓存。

**决定不变：关闭固定QK分类路线。** 真实预训练交互会响应合法mask，但这次并未提供比O24更好的跨图身份；不通过翻号、挑头或挑域救结果。仍可复用的正证据是：原DINO语义及query内部连接已有迁移能力，局部近邻缺支持并不等于确定背景，而困难候选内的角色区分仍是缺口。下一独立算法须为这个缺口提供可检验的任务绑定依据，并保留完整输出收益；单独增加表示层次、参考拟合自由度或传播范围均未得到本轮支持。主goal未达到。


## Astra：从已有得失分冻结一套完整组合，交指定Sol执行

**当前状态：推理规则及900例顺序已固定，尚未实施或评分这套组合。** 后续实验由指定主Sol执行；根的子代理任务均结束。机器协议及所有源清单SHA在 [composition_bg_admission_protocol.json](../../evidence/local/insid3_foris_generalization_20261010/composition_bg_admission_protocol.json)（SHA256 `4aae752889a4316b9686dbaaacaaf09adeadaf38ca47cb515ede31fca775f8b5`）。这次验证完整FoRIS宿主、独立参考ridge及二者的有界组合；不再要求任何一个模块单独击败完整FoRIS，也不为名义独立性删除已有识别、区域组织和误报抑制能力。Q/K直接分类的负结果不构成“该模块在任何组合都无用”的结论；但目前没有它的互补收益证据，所以这套链不加入它。

### 已有收益与损失决定职责

现有 `anchor.global` 已是一条完整链：FoRIS提供身份与query区域组织，真实reference coverage拟合的APD-ridge提供另一任务判据，reference合法阈值把它变成有界证据，最后修改FoRIS低幅度区域。原图PACO600为40.775对FoRIS38.461；Deep100为32.015对29.378；COCO200为62.460对63.359。Deep仍弱于已有快九窗37.344，必须保留这个强控制，不能把三域点差直接写成Strong成功。COCO损失包含新增误报和误删目标，PACO则有明确的同父误报删除收益；不能把整条负分支撤去。

本次只读同一旧Deep100/PACO100与COCO200的400份ridge字段、完整FoRIS及旧组合输出。先只用合法reference/query输出封存描述量，之后才读取GT分解旧编辑，未生成新mask。定义不受既有`.5*h`有界校正翻转的query核心：完整FoRIS的BG/FG标签且原score幅度 `|2*minmax(S)−1|>.5`。用它在128格的真实像素质量求ridge分数的加权中位数，得到背景 `b_Q` 与前景 `f_Q`。全部400例两个核心都存在且 `b_Q<f_Q`。收据：[core_interval/report.json](../../evidence/local/insid3_foris_generalization_20261010/core_interval/report.json)；入口 [composition_core_interval_audit.py](../../../scripts/composition_core_interval_audit.py)。封存合法描述后，旧编辑评分CPU4.53秒，无新编码或新候选。

|可观测条件及其旧编辑（原图像素池；不是macro分数）|COCO200|Deep100|PACO100|
|---|---:|---:|---:|
|参考阈值 `t_R<b_Q` 的例数|6|0|0|
|这组旧新增 TP / FP|73,003 / 431,376|0 / 0|0 / 0|
|`t_R>f_Q` 的例数|85|28|59|
|这组旧删除 TP / FP|273,608 / 263,690|72,558 / 570,212|13,983 / 78,390|

COCO仅6例就贡献旧组合84.33%的新增FP；其中query背景核心的平均真背景比例仍为93.72%。这一**背景阈值冲突**是新增前景过宽的具体、可观测风险。反向条件不能对称处理：`t_R>f_Q`时，Deep/PACO仍大量删对FP；这两组所谓前景核心平均真实精度仅48.50%/60.65%，COCO也只有80.20%。因此不能强行降低阈值去迎合FoRIS前景标签，否则可能撤销参考判据正在完成的纠错。这里的核心纯度是GT后验解释，推理只使用核心本身及ridge分数，不读取纯度。

### 固定完整链及每步为何存在

1. **保留完整FoRIS宿主**，包括当前APD、目标多模态、难BG、对应、query区域组织、有界修正和原CRF，输出完整mask `B` 与连续score `S64`。它负责已证明有效的语义、覆盖与边界，不把新的ridge直接阈值结果当完整替代品。
2. **保留同一真实reference ridge**，不改coverage目标、128/role压缩、lambda `.01`、whole观察或APD分支。取已封存 `r128`、合法source阈值 `t_R` 及类均值分离度 `Δ_R`。它负责用参考任务判据修正宿主，尤其保留已证实的拒绝相似非目标能力；本项不换loss/表示或重做训练。
3. **只约束新增前景的准入**。从完整FoRIS的受保护query背景核心求 `b_Q`，定义 `t_A=max(t_R,b_Q)`；若背景核心为空，则 `t_A=t_R`。没有query前景上界，不用reference面积、数据集名称或GT选择分支。`t_A`是避免参考零点把典型query背景当正类的最小单向修正，不声称是后验概率或IoU最优阈值。
4. **按编辑方向组合，并保持原有界读出**。在128格计算 `h_R=clip((r−t_R)/Δ_R,−1,1)`、`h_A=clip((r−t_A)/Δ_R,−1,1)`；两者都沿原bilinear到原图。保持旧 `g0=sign(B)*|2*minmax(S)−1|`、`.5`权重和精确tie继承B，得旧完整组合 `M_R` 与准入读出 `M_A`。最终

   `M_full = M_R ∩ (B ∪ M_A)`。

   在原B内部完整保留旧ridge的删除决定；在B外只接受通过背景约束的旧新增前景。无需额外CRF，避免后处理破坏这一作用边界。非正source分离度沿旧零证据回退，不另造规则。

这是同一算法对每个任务的条件处理，不是“PACO选旧方法、COCO选新方法”。400例描述量预告Deep/PACO pilot逐位保持旧完整组合；COCO仅6例进入新增区域约束。后续PACO剩余500必须按同样规则实测，不能根据其结果再改条件。

### 为什么这一组合比直接平均或双向校准更可检查

`t_A≥t_R`，clip和正权插值都单调，所以 `M_A⊆M_R`。最终的**原FoRIS前景删除集合完全等于旧组合**，新增集合只会缩小；不会为了压新增误报，额外撤销宿主里原本保住的目标。`t_R≥b_Q`或无背景核心时逐位还原旧组合；零secondary证据逐位还原FoRIS。上述是可验证的集合性质，**不保证净涨分**：被拒的旧新增仍可能是真目标，背景核心也可能包含漏检目标。

主要反例也固定：若FoRIS把整个目标当高幅度BG，核心会被污染，而且旧有界宿主本就无法恢复该区；若新增目标与背景在ridge中重叠，背景约束会误删新增TP；若COCO损失主要来自旧删除，此规则保留该损失，不声称全治。因而当前收益预判只针对“源阈值低于同query可靠背景”的过宽新增，不能推广成所有高AUC读出都能统一校准。

### Sol的最小完整验证，不增参数菜单

固定旧清单**Deep100＋COCO200＋PACO600，共900例，PACO pilot只计一次**。全部原始特征、FoRIS输出、ridge字段已存在，不需新增DINO或重新拟合。PACO600中复制的99个pilot条目需从原raw signal字段按episode ID取回，不能因扩展目录没有重复field就重算或跳过；协议绑定了这两份父资产。

只新生成两臂：上面的完整组合，以及**新增面积匹配的原source排序控制**。对每张原图，该控制保留和候选同样数量K的旧新增像素，按原S64上采样分数从高到低选取；B内部仍保留旧ridge输出。它检查净收益是否只来自少加一些前景。精确ties按固定row-major顺序处理并报告其规模，绝不读GT打破tie。完整FoRIS、旧 `anchor.global`、MEAN及已有whole/快九窗直接复用同输入封存结果；没有必要把弱模块单独扩跑。

实现及配置先冻结，900例完整预测全部seal后评分。原图主端点保持PACO303、COCO40官方class/fold槽位及缺类零值，Deep总I/U；不混用本次PACO100的87观察槽位。报告完整IoU、同批差及照片依赖区间、补真/删真/删假/补假；单列阈值约束触发、旧新增被拒的TP/FP与其实际分数代价、原B内删除完全不变、同K控制数量相同、旧组合/零证据重放身份和缓存增量成本。新方法高于旧组合却未高于完整FoRIS/MEAN仍不算主goal达成；候选与同K控制相当时，不把收益归因于新的任务识别。

这是一套有明确职责和反例的**完整组合实验**，并非已证明原创性充分或已经跨域领先的方法。根不执行新预测，指定Sol可以直接按协议实施；共同文档保留所有此前负结果及本次尚未实测的状态。


## Astra：900例输入已绑定，组合可直接实施

[推理输入绑定表](../../evidence/local/insid3_foris_generalization_20261010/composition_inputs900.json)已按冻结协议的900例顺序完成；每例包含已有r128字段、reference标定、完整FoRIS原图/1024结果、source score和旧ridge组合的准确路径及SHA。PACO扩展目录未重复保存字段的99例，已按episode ID指回原pilot字段，全部文件身份核对通过；不需要重拟合或补编码。[仅合法信息的900例描述量](../../evidence/local/insid3_foris_generalization_20261010/composition_descriptors900.json)也已封存：协议真正的触发条件`t_R<b_Q`在COCO6/200、PACO5/600、Deep0/100成立。剩余889例应逐位保持旧完整ridge组合；触发不等于输出一定改变，也不等于收益已成立。

PACO另外两例有`b_Q≥f_Q`，但`t_R`都高于`b_Q`，本规则仍不触发。**前景中位数与`condition`分类字符串只作诊断，不是推理输入或回退规则**；实现只能依据协议里的背景中位数、source阈值和空背景核心分支，不能加入“核心顺序反转即停用ridge”的额外selector。

入口 [prepare_composition_cache_manifest.py](../../scripts/prepare_composition_cache_manifest.py)只读既有小缓存，5.51秒，queryGT/新mask/新拟合/新DINO均0。输入表SHA `f4a6c6ddbb5211034368c25a528c6992c830aa995f95cb442ddfcc3ffdbb6c36`，描述量SHA `fb2efe1e5fb27540c7fc6933847ad2406580ca0f601d25fecd9d994950b4609c`；原冻结协议没有修改。指定Sol可直接从这份输入表生成两臂完整预测并封存，再进入评分。


## [Sol 主实验验证] 按最新人类方向：模块互补、冲突与完整组合

最新人类方向已从监督线程直接用户消息核实：复用已有得失分和反例，允许保留/借鉴FoRIS高质量处理；完整组合及少量消融决定去留，不能要求每个模块单独超过FoRIS。Astra负责统一机制及冻结规格，Sol验证固定同批完整输出和条件收益。

本项只读固定60对/120任务的已封存Q/K及同读出强控制、完整FoRIS CRF和隔离GT。没有新encoder、预测、融合公式、GT门控或参数搜索；Q/K直接读出低分只拒绝该读法，不否定模块的一切用途。

对候选M与强输出B，逐像素记录新增TP/FP、删掉TP/FP，并在候选union精确计价四项IoU收支。合并(B∪M)或取交(B∩M)的完整IoU差由这些计数直接推得，只是后验账本，不生成候选组合mask，更不选择哪一照片该并或交。

|主QK/zero对完整FoRIS，每任务平均像素|新增TP|新增FP|删掉TP|删掉FP|有益新增任务/40|有益删除任务/40|
|---|---:|---:|---:|---:|---:|---:|
|COCO|6417|257138|25047|11536|2|11|
|PACO-Part|8573|123910|46157|34507|4|24|
|SUIM|34275|171118|69507|27741|12|5|

“有益”仅指用真实GT计价的该新增或删除集合净收益为正，不是可部署判断。三域有不同数量的补回/排错成功，但大量冲突掩盖这些局部作用，故模块既不是全部无用，也不能直接整体采用。

|主QK/zero的简单并集后验差及95%CI|简单交集后验差及95%CI|
|---|---|
|COCO -34.825 [-41.872,-27.595]|-17.667 [-27.296,-9.305]|
|PACO-Part -12.910 [-18.633,-7.781]|-6.866 [-12.872,-1.111]|
|SUIM -19.662 [-28.366,-12.110]|-12.391 [-18.677,-6.836]|

组件仍按真实R/Q照片身份20/18/20组重采样。以上否定无条件整并/整交，没有对连续信息、局部使用或真正统一组合做总体否定。作为匹配强控制，旧O24 APD/zero取交的COCO/PACO/SUIM点差−.699/+1.037/−2.789，COCO/PACO区间跨0，SUIM为负，也不能按PACO挑一个统一筛选器。

**可观测条件只给线索，尚不足作统一门控。** 在QK所提议的新增/删除集合内，用旧O24 APD零阈值读出是否也选中分层；分层来自保存预测，没有读GT选择。两读出都否定的拟删除区域，其真实前景精度COCO/PACO/SUIM为57.08%/17.42%/41.51%（像素合池）：PACO约82.58%是背景，COCO却仍含大量真目标。两者都选中的拟新增区域真实前景精度4.14%/8.07%/44.12%，只看模块同意也不能确认正确任务。单张照片、两个合法任务和其它简单对照的全部分层计数在JSON。此类条件必须借助已识别的有效任务语义/结构信息，不能按dataset名切换。

3960项SHA、完整I/U、TP/FP编辑恒等式及IoU计价检查通过，CPU6.230秒，新增encoder/预测0。1200个已有方法/readout任务结果均覆盖；范围是完整1024真像素，指标为两个任务/照片平均，非官方全class mIoU。

新X23缓存绑定独立核验839项通过：119全token文件、对应旧输入/O24绑定、checkpoint norm1/QKV参数及原RoPE等。未重复encoder/MPS；producer的119次O24/末块逐位重放是producer收据，本核验验证其绑定，没有冒称重新前向119次。原RoPE FP64单位范数误差2.24e-07，来自保存FP32 sin/cos；独立检查使用8 eps32容差，不改模型、字段或评分。Astra已有三域首例字段CPU复核，未重复运行本脚本的字段抽样模式。

下一动作：依据这些局部得失及反例，按Astra的完整统一方案保留有效处理、只检验有互补依据的模块和少量关键消融；不能因为Q/K独自低于FoRIS就关闭所有使用，也不能把并/交账本变成按GT或域名选择的推理规则。


完整逐任务与收据见 [组合得失报告](../../evidence/local/sol_validation_20261010/ATTENTION_EDIT_COMPLEMENTARITY.md)。所有对已有Q/K直接读法的否定仍保留，不扩展为模块在所有组合中无用。


## Astra：900例完整组合已完成，区分原组合收益与新增约束失败

指定主Sol已执行并封存 `../cv_data/a/bg_admission_composition900_v1_20261010/` 的900例两臂，推断7.253秒、峰RSS约.904GB，无新DINO、raw读取或reference拟合。全预测封存后评分4.443秒，原图PACO303/COCO40槽位及Deep总I/U保持不变。根另用 [background_admission.py](../../../src/ics/methods/background_admission.py) 的独立函数实现重放全部1800个新mask，逐位一致；900个旧parent、背景中位数、原B内删除不变、同K数量等核验全部通过，6.53秒，未读queryGT。此前同模块的10个合成fixture只检查阈值、B保护和tie行为；不能把它们当质量证据。见 [输出复核收据](../../evidence/local/insid3_foris_generalization_20261010/background_admission_output_verification.json)。

|同批原图主指标|FoRIS|原ridge完整组合|背景准入完整组合|同新增面积source排序|MEAN|快九窗|
|---|---:|---:|---:|---:|---:|---:|
|Deep100|29.3775|32.0150|32.0150|32.0150|26.4654|37.3440|
|COCO200|63.3589|62.4598|62.4132|62.4295|64.0068|未执行|
|PACO600|38.4609|40.7749|40.7656|40.7646|39.3520|34.9714|

新增约束相对原组合Deep不变、COCO−.0466、PACO−.0093，不能接受为已有组合的改进。COCO6例实际拒掉旧新增TP57,753、FP121,762；PACO5例拒TP499、FP689。原FoRIS前景内部删除集合在900例完全未变。约束虽然发现一类新增误报集中的**episode条件**，却没有在这些episode内选出有利的删除集合；病例风险不等于像素动作的有效性。宏mIoU还必须按原class/fold的I/U计价，不能用总删除FP较多或84.33%的像素集中性直接预判涨分。与同K控制也没有有意义的稳定优势。

完整组合中仍存在真实互补，不能从这条约束的负差淘汰全部模块。按当前完整输出相对FoRIS的实际编辑分开核算：Deep只加为30.3094、只删为31.0482，二者组合32.0150；PACO只加38.4786、只删40.7508，组合40.7656，主要价值仍是参考判据的拒识；COCO只加62.6795、只删63.0830，都未高于FoRIS63.3589。数值是按原class/fold重构的事后动作账，非可部署GT选择或可线性相加的模块贡献。

下一步只读已经保存的完整FoRIS、MEAN、ridge，检查**编辑是否被另一机制支持**这一更贴近像素动作的可观测条件：分别统计两者共同改、仅ridge改、仅MEAN改的补真/补假/删真/删假及class计价。此处考察完整链的互补与共同错误；不追加背景百分位、阈值强度或新编码，也不因某单模块分数低于全FoRIS就淘汰它。当前完整多数据集领先目标仍未达到。


## [指定主Sol] 原冻结900完整组合已执行：新增准入未获净收益

按Astra已冻结的完整方案执行Deep100+COCO200+PACO600，PACO pilot只计一次，复用99个原pilot字段，不补编码或重拟合。保留完整FoRIS识别/区域组织/CRF及原合法reference ridge有界修正，只对FoRIS外的旧新增前景施加同query受保护背景核心的下限；原FoRIS前景内删除完全保留。主臂与同新增面积、原source score排序控制事先固定，全部1800原图完整mask先封存后query GT。

新增准入 `t_A=max(t_R,b_Q)`，`b_Q`为CLI1024受保护背景核心投到128格真实像素质量的lower weighted median；无背景核心回退`t_R`。h先128格clip后原bilinear到原图，沿旧g0+.5h、精确tie继承B。最终`M_full=M_R AND (B OR M_A)`，无额外CRF、前景上限、dataset名开关或参数搜索。

|原图官方端点|完整FoRIS|旧anchor.global|新完整组合|同K source排序|MEAN|快多视图|
|---|---:|---:|---:|---:|---:|---:|
|Deep100|29.377542|32.014976|32.014976|32.014976|26.465369|37.344028|
|COCO200|63.358900|62.459822|62.413184|62.429538|64.006821|未有匹配输出|
|PACO600|38.460868|40.774944|40.765619|40.764630|39.352045|34.971403|

PACO采用303个官方fold/class槽位、COCO两fold共40槽，包括实际缺类的零值；Deep为总I/U。不能换成观察到的87个PACO pilot槽位或CLI1024。COCO MEAN为同输入已封存原图输出，未重跑当前raw；所有对照仅复用，不吞掉其历史编码成本。

|新组合点差；连接R/Q照片bootstrap95%区间|对完整FoRIS|对旧组合|对同K控制|对MEAN|
|---|---|---|---|---|
|Deep100|+2.637434 [+1.159475,+4.353686]|+0.000000 [+0.000000,+0.000000]|+0.000000 [+0.000000,+0.000000]|+5.549607 [+2.993267,+8.339395]|
|COCO200|-0.945716 [-2.586436,+0.354037]|-0.046638 [-0.430903,+0.203964]|-0.016354 [-0.044310,+0.011134]|-1.593637 [-3.432206,-0.397151]|
|PACO600|+2.304750 [+0.815296,+2.227035]|-0.009325 [-0.016929,+0.010600]|+0.000988 [-0.000080,+0.002406]|+1.413574 [+0.088978,+1.757333]|

连接R/Q照片组数Deep/COCO/PACO为100/196/232，2000次固定cohort cluster重采样；官方槽位始终保留，bootstrap中无draw的槽位亦为0。因此该百分位区间可能偏离原点估计，不能当未见照片/类别的确认。原存档query-photo-within-fold/class10000次区间同时保留在JSON，两个估计范围不同；PACO新组合相对旧组合该区间[−.016168,+.001004]，相对同K[+.000283,+.002102]，连接组区间则跨0。微小对同K差不作为新身份识别贡献。

完整组合相对旧链COCO−.046638、PACO−.009325、Deep0；新的新增准入耦合未成立。现有旧FoRIS+ridge职责仍在Deep/PACO带来真实收益：新组合相对FoRIS分别+2.637434/+2.304750，但COCO−.945716，且Deep仍低快多视图5.329052；没有达到跨域强基线目标。此次负结果只否定这套背景下限耦合，不把ridge或FoRIS模块整体判为无用。

**触发与实际代价。** COCO6/200、PACO5/600、Deep0/100触发且改变mask，889例与旧组合逐位同一。COCO拒绝旧新增TP57,753/FP121,762，PACO499/689；减少FP的像素总数并不保证官方mIoU收益。COCO `dev_s1/coco/0/13`（loader class60）拒TP50,172、FP21,143，episode IoU21.248→.291，同K2.426；该fold/class池化损失−7.937点。另class8主要拒FP95,034、TP127，池化+6.943；两个类贡献经40槽位等权后使净差略负。高纯BG核心的典型ridge分数仍可高于一部分真实新增目标，故“源阈值低于背景核心”不足以识别哪些新增可安全拒绝。

PACO `dev_s1/paco_part/0/91`拒TP157/FP11，IoU14.569→11.280；`dev_s1/paco_part/2/75`只拒171 FP、0 TP，64.391→68.052。这些原序列反例与所有11触发记录已保存，不据此重设中位数分位、阈值、域名条件或追加selector。

**核验范围。** 原B内删除集合900/900完全等于旧ridge链；原B外新增始终是旧新增子集；同K控制900/900新增数完全相同、B内部相同；旧公式与零secondary证据900/900逐位重放。所有源码/输入/field/pred SHA、无queryGT推理guard和顺序绑定保留。评分22,200项真实I/U、编辑重建与旧对照计数检查通过；独立NumPy11,756项检查通过，900真像素计数/集合性质及官方类池化独立一致，并对全部11触发案例重新以NumPybilinear/lexsort重建两臂，完整mask差0。

**实际新增成本。** CPU缓存组合推理7.253秒，峰RSS0.842GiB；评分及两种bootstrap4.443秒，独立复核3.456秒。DINO新增0、raw读取0、reference拟合0、queryGT推理访问0。完整数组/原图pack约11.72MiB，旧大缓存未复制。此为新增缓存成本，不是完整冷启动端到端时延。

主goal未达到，不推广该耦合。下一完整方案应依据已存编辑和Source/Query可观测信息区分可拒新增与真实目标，保留已证实的ridge拒假及宿主能力；由Astra冻结统一规则，再验证关键消融，不因一次组合失败重新轮换孤立公式。

源码：[两臂推理](../../../scripts/sol_bg_admission_composition.py)、[评分](../../../scripts/sol_score_bg_admission900.py)、[独立验证](../../../scripts/sol_verify_bg_admission900.py)。机器资产`../cv_data/a/bg_admission_composition900_v1_20261010/`含config/manifest/sealed/predictions/records/report及独立收据；原输入来自Astra冻结900绑定。

完整官方类池化、逐任务触发反例、区间、成本及家中快照见 [900组合报告](../../evidence/local/sol_validation_20261010/BG_ADMISSION_COMPOSITION900.md)。源码和输出独占新目录，未修改Astra原协议或其它实验资产。


## Astra：完整链的编辑重合给出稳定共同部分，但唯一编辑不能一概删除

只读900例已经封存的完整FoRIS、原ridge组合与MEAN，先封存不含GT的共享/独有编辑集合，再以16状态像素联合计数核对三方法全部I/U、六类编辑及其重构恒等式，9,900项通过，后验计数1.70秒。未产生新的分割候选。MEAN的COCO使用同输入归档mask，PACO的外部mask也逐文件SHA绑定。来源、逐例与class计价见 [edit_overlap/report.json](../../evidence/local/insid3_foris_generalization_20261010/edit_overlap/report.json)，入口 [composition_edit_overlap_audit.py](../../scripts/composition_edit_overlap_audit.py)。

表中数值是**在该完整链中，保留某个编辑集合相对于撤销该集合的mIoU价值**；正值表示该集合在该上下文有益。它不是整链新方法成绩，各行不能相加。

|实际编辑集合；计价所在完整链|Deep100|COCO200|PACO600|
|---|---:|---:|---:|
|ridge与MEAN都新增；ridge链|+.023|+.182|+.024|
|ridge与MEAN都删除；ridge链|+.932|+.549|+1.710|
|只有ridge新增；ridge链|+.938|−.818|+.0003|
|只有ridge删除；ridge链|+.862|−.782|+1.109|
|只有MEAN新增；MEAN链|−.408|+.129|−.501|
|只有MEAN删除；MEAN链|−3.233|−.204|+.158|

共同新增/删除在MEAN自己的完整链中也均为正。相反，ridge独有编辑对Deep有重要覆盖及拒识价值、对PACO有拒识价值，却在COCO两方向都亏损；MEAN独有删除则在Deep大量误删真路。**另一个方法保持原mask，可能只是没有足够幅度改变，并不自动表示它拥有强反证。** 因而共同同意可作为可检验的保守部分，不能把全部不一致直接定义为错误，更不能按数据集名保留各自最好分支。

下一项仍是完整组合的固定控制，先量出共同部分的真实端点：`C=majority(B_FoRIS,R_ridge,G_MEAN)`，等价于两条校正都提出同方向编辑时才改变宿主；不同意处保留宿主。三条完整链及其原图读出均不变，无新编码/拟合/图/CRF。增加一个**同新增数与同删除数**的source排序控制，在两链编辑并集且原ridge可变范围内选同数量像素，分别检验是否仅靠少改像素获益；ties固定row-major。900例顺序和两臂规格已写 [consensus_composition_protocol.json](../../evidence/local/insid3_foris_generalization_20261010/consensus_composition_protocol.json)，SHA `d9942f89a465ff883eef54b10a260ad7c9c9e8fc38c1c46e838d2507d165cb03`，由指定主Sol执行，先seal再评分。

这项布尔同意规则是已知组合操作，不作为原创性结论。它预计牺牲部分Deep/PACO独有正收益，必须与原ridge和快九窗一并保留。其用途是先验证能否形成跨域有益的共同修改，再研究如何用参考/查询可观测条件保留那些没有第二条链确认、但实际有价值的修改；总体目标没有缩成一个更容易通过的保守子集。


## [指定主Sol] 共同编辑900完整端点及独立验证已完成

固定同一majority完整规则，900原顺序、两臂1800原图mask先seal后GT，评分17,100、独立NumPy12,615检查通过；source控制逐位差0。原图官方Deep/COCO/PACO共同编辑30.225795/64.080288/39.664615，对完整FoRIS+.848253/+.721388/+1.203747，两种照片bootstrap区间均正。对同新增/删除预算source排序差+.093246/+.315654/+.432451，Deep区间跨0，COCO/PACO两种区间均正。

对旧ridge完整差−1.789181/+1.620466/−1.110329：Deep撤回独有新增的后验仅动作代价−.938119、恢复独有删除−.862054；PACO主要损失拒假−1.108576，撤回新增仅−.000304。COCO对MEAN+.073467但两种区间跨0，Deep仍弱快多视图7.118232。共同修改在三域有实益，但不一致≠可靠反证；该已知布尔组合及已曝光开发池不构成原创性/未见泛化确认，主goal未达到。

CPU缓存推理2.793秒、峰RSS.637GiB，评分4.321秒、独立验证6.156秒，0新编码/raw/拟合/queryGT推理访问。另以直接bitmap撤销编辑复核Astra900重合账本，18,024检查通过，最大mIoU误差2.14e−14、2.565秒。所有失败与强对照保留；大型源缓存不复制。

[完整报告及家中轻量快照](../../evidence/local/sol_validation_20261010/CONSENSUS_COMPOSITION900.md)、[独立编辑重合收据](../../evidence/local/sol_validation_20261010/edit_overlap900_verification.json)、[前一背景准入四个相反反例图册](../../evidence/local/sol_validation_20261010/bg_admission_counterexamples/INDEX.md)。下一理论规格应能保留可靠独有修改并解释共同错误；不将总体目标缩为保守少改的共同子集。


## [指定主Sol] 独有编辑余量900：可读线索仍在，但高余量并非可靠性

先封存三项原链descriptor再GT，未生成新mask。ridge独有删除的abs(h)正确性AUC Deep/COCO/PACO=.665708/.574426/.659125，RQ照片组区间均高于.5；原判决abs(g0+.5h)降至.623527/.537678/.591390，COCO/PACO配对下降区间排除0，Deep尚跨0。低宿主置信对独有删除的AUC均约.49，不能替代task证据。3,491,016实际修改像素满足abs(g0+.5h)=.5abs(h)-abs(g0)，误差0；分量保留时可恢复，不能称整条任务信息不可恢复。

反例：COCO独有新增402,916个饱和abs(h)>=1像素，其中348,830个误报；船例0/73有235,969个错误饱和新增。故现有reference校准高余量并不自动意味query可靠，也不能根据排序正信号就恢复全部不一致或扫描cut。Deep独有新增task AUC区间跨.5；空/全正确/全错误池都保留，未混成完整mIoU。连续MEAN源场尚未在这900绑定中齐全，二值保留B的原因不能补造。

准备11,700、评分24,300项通过，pair-count AUC与sklearn最大差2.22e−16。CPU准备5.927秒、峰RSS.925GiB、评分5.021秒；0新编码/raw/拟合/prediction。约81.2MiB合法descriptor及报告，家中轻量快照无需加载数组。详见 [900余量审计与资产](../../evidence/local/sol_validation_20261010/unique_edit_evidence900/REPORT.md)。主goal未达到；此证据支持保留独有task线索，同时要求分离宿主推翻幅度和独有修改的可靠性条件。


## Astra：共同编辑完整组合的正结果与下一步范围

指定Sol已完成 `../cv_data/a/consensus_composition900_v1_20261011/` 两臂900例的完整推断、封存和评分：缓存推断2.793秒、峰RSS约.684GB，评分4.321秒，无新编码/拟合。根用独立8状态真值表及lexsort控制重放全部1800个mask，再从先前独立16状态像素账推导900个主臂I/U，全部一致，9,903项检查通过。来源：[consensus_output_verification.json](../../evidence/local/insid3_foris_generalization_20261010/consensus_output_verification.json)，入口 [verify_consensus_composition_outputs.py](../../scripts/verify_consensus_composition_outputs.py)。该复核未重新读取GT图像文件，但I/U推导明确使用了封存后的GT计数，不能说完全不使用标签信息。

|原图开发口径|FoRIS|MEAN|共同编辑完整组合|相对FoRIS|同编辑预算控制|
|---|---:|---:|---:|---:|---:|
|Deep100，自定义道路|29.3775|26.4654|30.2258|+.8483|30.1325|
|COCO200，40槽位|63.3589|64.0068|64.0803|+.7214|63.7646|
|PACO600，303槽位|38.4609|39.3520|39.6646|+1.2037|39.2322|

统一规则在这三组已曝光开发输入上均高于完整FoRIS；两种照片统计口径的区间下界也都为正。COCO/PACO相对同编辑预算控制分别+.3157/+.4325，支持共同同意确实改善了部分修改的选择，而不只是少改一些像素；Deep相对同预算仅+.0932、区间跨零。COCO相对MEAN仅+.0735、区间跨零，不能称已明确优于这个强控制。

同时，共同部分牺牲的价值已完整兑现为损失：比原ridge链Deep低1.7892、PACO低1.1103，Deep比快九窗低7.1182。不能把这些损失隐藏成“全面更优”。当前得到的是一条可复核的跨域共同修改机制及其边界，布尔同意操作本身不是充分的原创性主张。

**优先保持规则，检查剩余官方数据集迁移，避免继续只在900例上调组合。** 现行主goal在PLAN中的明确范围是冻结DINOv3-L、单参考、不训练，COCO/LVIS/PASCAL-Part/PACO-Part/SUIM按官方协议的完整class mIoU超过FoRIS且95%区间下界>0；开发选择与一次性确认分开。当前只有COCO/PACO两个官方数据集的有限开发子集，Deep是额外的自定义道路研究，不能替代剩余三个官方数据集，也不能将原图成绩直接当 released CLI1024 结果。

已现场确认低成本下一入口：`../cv_data/a/lvis_foris1400_score_reuse_20261009/` 的 `run/` 与 `mean1400/` 都有同序1400例manifest、inference与sealed以及完整预测，`run/fields/`保留source字段；`prepared/manifest.json`为现有固定开发清单。下一步可绑定其raw whole R/Q、保持同一ridge拟合与共同编辑规则，再由Sol完成整套LVIS迁移。PASCAL/SUIM的本地完整官方paired source目前只核到smoke各1例；主raw缓存的82/83条相关输入不等于有完整分割输出，更不等于598/582例历史转移包已在本地。须先核对实际缓存覆盖，缺少的原始O24不能由已处理FP16特征伪造。

后续按官方CLI1024主帧与原图辅帧明确冻结完整读出，不根据哪一帧分数更好选口径；确认集先核查照片暴露与读取账本。独有编辑的研究继续围绕参考任务与query图先验是否相容，但此刻不新增层/头/阈值扫描，也不把900例的共同核心当五集goal完成。


## [指定主Sol] 原完整规则迁移到其余三域120：已绑定并执行

主Sol已按不变原ridge/majority/同编辑预算规格接续LVIS/PASCAL/SUIM各40例，不改变参数或按域选择；来源为现有冻结zoom panel，顺序保留，GT仅全部mask封存后。原同输入FoRIS、MEAN、source64和236原始O24输入均已校验RGB/真实transform/tensor/payload SHA，4200项通过，复用约3.42GiB原payload但不复制、不补编码。绑定入口`evidence/local/sol_validation_20261010/remaining_composition120/manifest.json`及`receipt.json`，曝光状态/legacy输入保留。

由指定主Sol唯一执行`sol_transfer_consensus120.py`，新目录`../cv_data/a/consensus_remaining120_v1_20261011/`。新R字段原来不存在，因此只做120次同一合法reference-only拟合；先复算已冻结PACO sentinel，完整mask、128场、校准必须逐位/数值完全相同。之后原chain生成anchor、共同编辑、同新增/删除数量控制三个完整mask；0新DINO/raw写入/视图/CRF，未使用新置信度诊断挑门控。配置已先固定到out/config.json；实际PID/完成度见Sol state。请避免其它代理在这120例重复拟合或输出。该小已曝光面板不能充当五集全基准或未见照片验证，结果仍未知。


[主Sol迁移120校验更正] 首次哨兵完整mask已逐位相同，但脚本错误要求历史128浮点场也逐位相同，启动在此停止，120新拟合/输出均未开始。原已验证COCO sentinel实际field_equal=false、maxdiff=3.1086244689504383e−15、mask及校准exact。修正只将哨兵数值差要求恢复为与该历史收据完全一致；保留旧拒绝config/源码至`consensus_remaining120_v1_20261011_sentinel_rejected_v1`，不改算法、绑定或参数。


## Astra：LVIS1400实际缓存绑定完成，双帧读出已固定

[binding receipt](../../evidence/local/insid3_foris_generalization_20261010/lvis1400_binding/receipt.json) 已覆盖现有同序LVIS1400全部输入，2464个不同whole R/Q transform key无缺失，FoRIS/MEAN两帧2800份预测及1400份source字段SHA核对通过。30,184项输入/结构检查，32.345秒，无queryGT、拟合、新预测或DINO；约38.34GB原payload只引用不复制，本次未重算大payload/tensor hash，原frozen reader在实际推理读取时继续验证。入口 [bind_composition_lvis1400.py](../../scripts/bind_composition_lvis1400.py)。此为已曝光开发1400，不能称全LVIS或未见确认。指定主Sol正在完成剩余三域120评分，此绑定只消除后续数据准备障碍，没有重复其120拟合或启动1400推断。

[双帧固定规格](../../evidence/local/insid3_foris_generalization_20261010/consensus_dual_frame_transfer_protocol.json) 将现有同一原图公式逐项定义到CLI1024：每帧使用该帧真实完整FoRIS与MEAN mask，同一S64和同一clip后的h128分别bilinear到该帧，再沿相同g0+.5h、tie继承B和共同编辑规则输出。CLI不能由缩放原图二值candidate冒充，亦不能选择涨分更多的一帧。官方CLI1024为后续主端点，原图保留；原图重放、zero身份、受保护核心与同补/删预算逐项不变。只冻结坐标一致的读出，不增加背景门控、阈值或参数。后续计分须按当前官方loader的真实class映射和全部槽位统一重算配对方法，同时单列观察类端点；旧1400配置中观察类清单不能默认为全LVIS的完整类集合。


## [指定主Sol] 其余三域120原图完整输出已核验：PASCAL未获益，SUIM低于MEAN

沿完全同一900规则和lambda=.01参考拟合完成120，360原图mask先封存后GT。独立NumPy120 ridge、majority及预算排序控制均差0；3375真实像素/池化/身份检查通过。LVIS/PASCAL/SUIM官方全槽位共同编辑1.883415/36.352448/62.533676，对FoRIS+.042603/−.121415/+.682824；LVIS仅39/920槽有样本，全槽位绝对数不能解释为整个LVIS成绩，观察类次级44.922550对FoRIS43.931299。PASCAL共同删除−.129264造成净损失；SUIM对MEAN−2.771487两种区间负，相对同预算−.002077。多域稳定强基线主goal仍不成立。

输入绑定4200检查、236缓存O24共3.42GiB只读取不复制。成功CPU缓存生产40.335秒、RSS1.247GiB，评分1.982秒；120新R-only拟合，原PACO哨兵完整mask/校准exact，场误差与已验证历史收据精确相同3.1086e−15。首次过严哨兵检查与首次SUIM null-ID分组错误已修正且保留拒绝源码，不重新编码或重复生产120；成功队列计时不吞掉未计时拒绝尝试。

[完整120报告及家中逐例快照](../../evidence/local/sol_validation_20261010/CONSENSUS_REMAINING120.md)。此队列config在新的dual-frame规格发布前固定original主端点，后者将作为新固定frame readout，120已有R字段必须复用，不能重新拟合或根据涨分挑frame。


## Astra：不一致的下一解释，先检验参考任务是否适合语义图

剩余120已由Sol完成：原图共同编辑相对FoRIS的官方完整槽位差为LVIS+.042603、PASCAL−.121415、SUIM+.682824；连接照片区间分别[+.000173,+.052426]、[−.289341,+.561053]、[−.944356,+1.949772]。LVIS只观察39/920槽，观察类辅助差+.991251；PASCAL27/56槽辅助差−.051235，SUIM覆盖7/7槽。不能混用辅助/全槽指标，也不能将LVIS几乎全singleton的类内窄区间视作未见泛化。主组合仍需要改进，尤其共同同意会否决SUIM的有效MEAN独有编辑。

固定下一项只读假设：如果真实reference mask与预训练语义图的连接关系冲突较大，则query语义平滑可能不适合这个任务粒度；这应在**同一数据集内**预测ridge相对MEAN及相对共同编辑的价值。图使用既有MEAN同族的reciprocal20NN/adaptive距离权重，参考侧构图；合法描述量为reference coverage的归一化图能量 `cᵀLc / Σd(c−mean_d(c))²`。相同20NN物理网格图、前景面积、coverage混合度为控制；它们用于排除仅反映目标大小或边界长度的解释。此量不是query正确率，source图相容也不保证跨图迁移。

[scripts/reference_graph_task_compatibility.py](../../scripts/reference_graph_task_compatibility.py) 正在由Astra读取原1020例参考缓存生成描述量，先封存再关联既有GT计数；没有query特征读取、新encoder、reference拟合、候选mask或门控。固定分析仅为各域内与三项旧episode-IoU差的秩相关，并控制面积/混合度/物理图能量，按实际R/Q连接照片组给区间。若只在跨域均值上成立、域内消失或方向相反，就不据此设计selector。既有reference pair-conflict改query边的负结果仍保留；本项只检验任务与图先验相容性的解释，不将普通图能量命名为新理论。


## [指定主Sol] Dual-frame现有1020已绑定；字段全部复用

按Astra新固定规格，将已完成900+剩余120所有来源统一绑定为1020双frame输入。11,220项源field/原ridge/parent/FoRIS/MEAN/source64 SHA、实际CLI和原图键、形状/有限值通过，准备3.413秒、0GT/新预测/新拟合/新raw读取。入口`evidence/local/sol_validation_20261010/dual_frame_readout1020/manifest.json`及`receipt.json`，固定releasedCLI1024主、原图次，不resize原二值mask造CLI。下一执行无需新原始特征或重复拟合：仅沿旧h/G0读出、majority及分frame同预算控制，并逐位重现已有1020原图mask。扩展LVIS1400时须按Astra绑定、复用重叠40个任务的已验字段。此处尚无CLI新成绩，主goal仍未达成。


## Astra：1020例参考图相容性检验完成，不进入推断门控

全部合法描述量已先封存，随后关联旧完整输出I/U；0新编码/拟合/mask/query原始特征。描述准备93.095秒，MPS driver峰约1.329GiB；评分6.834秒。六域各首例CPU与MPS图边、图能量完全一致；全1020图的edge-sum/二次型恒等式通过；18项普通/偏秩相关用Scipy独立重算，36项通过。见 [完整报告](../../evidence/local/insid3_foris_generalization_20261010/reference_graph_compatibility1020/REPORT.md)。

预设“reference语义图越不适合任务，ridge相对MEAN越有价值”的域内预测未成立。控制面积、coverage混合和物理图能量后，ridge−MEAN偏相关Deep+.057、COCO−.181、PACO−.049、LVIS−.037、PASCAL−.017、SUIM−.289；只有COCO区间排除0且方向相反，其余均跨0。ridge−共同编辑在COCO同样反向−.181，其余也没有稳定正关系。描述量仅说明source图与标签关系，不能替代query中真实修改的可靠性。

这还修正了“部件任务必然与语义图冲突更大”的粗略解释：PACO/PASCAL中位能量.0453/.0436，接近COCO.0360/SUIM.0386；Deep较大的.2356同时伴随更高物理边界能量和混合token质量。不能用跨域均值伪装可部署选择器。按预设决定不加graph-energy门控、不扫cutoff；保留全部正负结果。下一条可检查的信息仍是原完整MEAN连续场：二值不改B不足以区分主动反对与同向但未越过阈值，目前只核对字段来源和旧输出身份，不由本次弱相关直接组新公式。


## Astra：连续MEAN字段已找到720例，剩余300需按原链重放

[连续字段绑定收据](../../evidence/local/insid3_foris_generalization_20261010/mean_continuous1020_binding/receipt.json) 实际覆盖PACO600及LVIS/PASCAL/SUIM各40；720个mean64与source64均通过原推理SHA绑定，两帧1440完整MEAN mask逐位重建一致，source64与当前宿主字段720/720完全一致。6480检查，1.875秒，没有新推断文件/拟合/编码或GT。**MEAN实际读出不重新minmax：mean64直接FP32 bilinear到1024后>.5，再将该二值CLI mask投到原图。** 因而 `mean64−minmax(source64)` 的零点有明确含义，可以区分向某修改方向移动与真正越过分割阈值。

Deep100的joint-role包只保存了MEAN完整mask，连续字段未存；COCO200的连续字段在旧dev6000中已不在本地。COCO归档MEAN来自旧编码，而当前ridge/FoRIS使用后来准备的whole raw；同RGB、同参数和已验证FoRIS mask相同，不自动证明MEAN连续场相同。不能从二值图反推分数，也不能用现有FP16处理特征伪造raw。

[固定300例重放规格](../../evidence/local/insid3_foris_generalization_20261010/mean_continuous_replay300_protocol.json) 给指定主Sol使用：先完成当前双帧/迁移，再仅从既有whole O24重放原MEAN，保存实际source64/mean64并比较全部两帧旧mask；不用新DINO、参数或视图。若COCO发生数值路径造成的mask差异，旧档案和当前raw同公式结果分开保留，先seal再评分相同完整组合，不按分数挑数值版本。Astra不重复执行该推断。

下一只读诊断已经固定：在ridge独有修改内，检验MEAN的实际连续变化是否同向；在MEAN独有修改内，检验既有h_R是否同向。两者都保留零变化、空集合和全正确/全错误池，完整1020描述先封存再GT计价。该项不预设“同向就可靠”，不把尚未产生的方向门控当算法；先区分二值不一致究竟是反证还是幅度不足，保留真正的相反案例。


## [指定主Sol] 双frame1020已核验，LVIS1400沿冻结规格实际执行

1020例双frame全部6120mask先seal后queryGT，其中3060原图逐位复现旧输出；78418真实I/U/编辑/NumPy bilinear/sign/lexsort及官方槽位检查通过，三臂两frame全差0。CLI共同编辑COCO63.888679/PACO39.492237/Deep30.225795/SUIM62.193643/PASCAL36.379279；PASCAL仍低FoRIS.044046，SUIM比FoRIS+.776747但区间跨0且低MEAN3.198866。三域正差没有变成五集稳定结果，不能选frame报更高分。

主Sol已唯一启动`sol_consensus_lvis1400.py`，原Astra1400绑定按原顺序保留，40个已有同RQ/参考mask字段复用，1360新合法reference-only拟合；0DINO/raw写入/新视图/CRF。两frame用实际各自FoRIS/MEANmask，各自产生ridge/共同编辑/同预算控制，预计8400mask在任何query GT前统一封存。目录`../cv_data/a/consensus_lvis1400_dual_frame_v1_20261011/`和实际PID/进度已记Sol state。避免其它代理重复同批拟合。结果尚未知，完整主goal继续。


## Astra：LVIS1400遇到缺失参考角色，定义零证据延拓

1400迁移在833份完整记录后遇到 `dev_s1/lvis/0/148`：合法reference mask为480×640全前景，背景像素为0；1024→64 coverage仍为FG4096/BG0。原二角色平衡ridge及FG/BG分离度在此未定义，当前尚无1400 seal或query评分。该输入不能删除、改标签或按GT换方法。

[明确处理规格](../../evidence/local/insid3_foris_generalization_20261010/composition_missing_reference_role_fallback.json)：任一reference角色质量为0时不拟合该head，令h_R=0，按既有零修正身份得到R=B；不变majority(B,R,G)及同编辑预算控制都等于B。这是对未定义输入的宿主保留延拓，沿用原invalid-source/zero-evidence语义，没有新增阈值或domain selector。记录真实缺失角色与fit_performed=false，不伪造reference score/IoU。指定主Sol保留已完成833例和原冻结源码，补记新的producer hash及评分前修订后只续缺失后缀；全1400封存后再GT。


## Astra：官方原生PASCAL/SUIM各2000的实际缺口已核清

完整保留早已固定的 `dev_s1_all` 中PASCAL2000、SUIM2000顺序，逐张真实RGB/官方1024 transform绑定whole O24 key，未按缓存命中挑样本，未读query标签。PASCAL共有2928不同整图输入，42已缓存、2886缺失，只有2个episode两输入都齐；SUIM为1310不同输入，72已缓存、1238缺失，18个episode双输入齐。全部RGB存在且实际decoded hash通过，暂无下载阻塞。4124个缺失FP32 O24按未压缩计算约64.44GiB，当前盘余205GiB；不是编码耗时或压缩后大小的实测承诺。已在缓存的114个输入必须复用。

[覆盖报告](../../evidence/local/insid3_foris_generalization_20261010/official_pascal_suim4000_raw_coverage/REPORT.md) 及完整输入绑定表已保存，45.006秒、无编码/拟合/新预测。这里只查raw，不把“raw可用”写成完整FoRIS/MEAN对照已有。之前各40的legacy导出面板不能替代这些原生官方输入；2000开发也不是PASCAL8459/SUIM3859的全官方基准。

固定1020的双帧验证已由Sol完成：78,418项重建/计数检查、原图3060mask逐位复现。CLI主帧共同编辑相对FoRIS为COCO+.7184、PACO+1.0542、Deep+.8483、LVIS40+.04317、PASCAL40−.04405、SUIM40+.77675。PASCAL/SUIM区间仍跨0，SUIM比MEAN低3.1989；不能通过换帧把它写成五集成功。后续先完成当前LVIS1400的固定迁移及缺角色延拓，再以原生官方开发队列验证剩余空白，比继续只用legacy40或旧900调规则更能回答目标；公式、双帧和类口径保持固定，新增DINO只用于真正缺失的4124个输入，不重启旧停止任务。


## [指定主Sol] LVIS1400的合法全前景参考：零证据回退与833例原样接续

首次1400队列完成833例后，在`dev_s1/lvis/0/148`遇到原R mask307200/307200像素全FG，1024帧亦1048576/1048576全FG，没有BG。这不是缓存缺失或Mask在预处理中消失；平衡R-only ridge及两角色mean separation不可定义。按原dual-frame无效separation的h=0回退，不拟合此参考，R及C均保留完整FoRIS，全部case继续计入。无queryGT决定或域名分支；相应字段标source-role-degenerate，不伪装成正常拟合。

原833字段/完整输出按SHA直接接续，0重复拟合；首次源码和cfg留frozen/config_initial_before_role_fix.json，边界原生像素计数及295.180秒执行下界留interruption.json。新执行仅补退化边界和恢复已有记录，所有非退化拟合/阈值/系数/读出不变。原40复用保证继续。一次中断的最终小段未独立计时，最终成本将报告下界/续跑时间，避免吞掉失败尝试。实际新handle/PID/进度见Sol state；仍未读取1400 queryGT用于评分。


[Astra补充核查] 全1400 reference mask的文件SHA及角色可用性已只读核对，只有 `dev_s1/lvis/0/148` 这一例缺背景；所有原生参考尺寸≤640×640，nearest放大到1024不会丢失其余非空角色。缺角色例已实际重放1024→64质量，FG4096/BG0。收据 [lvis1400_reference_role_audit.json](../../evidence/local/insid3_foris_generalization_20261010/lvis1400_reference_role_audit.json)；没有queryGT/拟合/预测。零修正延拓适用于可观测缺角色条件，不按该episode ID特判。


## [指定主Sol] 双frame1020报告与逐例资产已归档

[完整双frame报告](../../evidence/local/sol_validation_20261010/CONSENSUS_DUAL_FRAME1020.md)含CLI主/原图次、官方固定槽位与观察类次级、全部强对照、两种区间及家中轻量快照。78,418评分/独立重建检查、20,400生产检查通过；原图全部输出及I/U/FP/FN身份复现，三臂两frameNumPy重建差0。0拟合/编码/raw/CRF，读出22.869秒/RSS1.011GiB，评分94.699秒。CLI仍PASCAL未修复、SUIM弱MEAN3.198866且相对FoRIS不确定；COCO/PACO正差不扩大成官方五集稳定成功。LVIS1400结果单独接续，仍不称未见确认。


[Astra固定迁移规格] [官方PASCAL/SUIM开发4000协议](../../evidence/local/insid3_foris_generalization_20261010/composition_official_pascal_suim4000_protocol.json) 已将完整既有清单、双帧规则及缺参考角色的零证据处理固定；指定主Sol可在当前LVIS迁移后沿已验证实现推进。完整FoRIS、MEAN、ridge必须共享同一批实际raw，每个dataset完整输出先seal再queryGT，先完成的数据集不用于改另一个的公式。保存mean64以免重复本次缺字段问题。只补4124缺失whole输入，复用114现有输入；Astra没有启动编码/候选任务。开发4000不充作两集完整官方或未见确认。


[Astra执行顺序说明] LVIS1400封存/评分完成后，先补上面已有raw的300个MEAN连续场及COCO当前raw身份，便于马上完成1020例连续证据分析；这项不增加encoder。随后/独立准备官方4000的新raw生产，不因字段缺失再次重编码。两项均由指定主Sol执行，Astra只消费封存字段分析。


## Astra：LVIS1400正迁移成立，仍保留同批更强的九窗对照

指定Sol完整封存/评分1400，两帧输出74,210项检查通过；新增reference拟合1359、复用40、缺角色零修正1，新增DINO0。CLI全920槽位FoRIS43.762466→共同编辑44.398755，+**.636289**；原图43.912229→44.519548，+.607319。实际观察876槽，按旧观察类辅助口径CLI45.948066→46.614507（+.666441），不是与全920口径混用的另一成绩。主类内query照片配对区间CLI[+.424757,+.754398]，仍限当前开发池；不是独立或23000例全LVIS确认。原任务未定义的全前景reference只此1例，已在读queryGT前补为h=0并保留整例。

根另外把旧九窗区域600与新1400逐个匹配，**600/600输入完全一致**。两帧真实像素重计，18,600检查，原图旧FoRIS/区域/几何控制1800份I/U完全复现。按这600观察类：CLI FoRIS47.619569、MEAN47.580880、旧ridge45.925918、共同编辑48.445810、同编辑预算48.205904、九窗区域51.702409。共同编辑相对FoRIS+.826241，仍落后九窗3.256599。不能把新增1400正差写成已超过全部强控制；也不能将九窗在PACO失分当作其在任何完整组合都无用。该精确比较见 [lvis600_matched_strong_control/REPORT.md](../../evidence/local/insid3_foris_generalization_20261010/lvis600_matched_strong_control/REPORT.md)。

这里另明确一项统计限制：600中每类只有一个任务。全照片组重采样再将没抽到的类置0，会把差值向0压缩，区间甚至不包含原点估计；这只是该再抽样算法的敏感性范围，不能凭“区间为正/负”宣称完整类别总体的名义95%置信。原封存结果不改写；后续全官方多例/类确认须沿冻结统计协议，并报告实际照片/类别覆盖，不能用这种稀疏开发区间替代最终要求。


[Astra合法连续证据准备720/1020，尚未GT评分] `prepare_continuous_edit_direction.py` 已将可用720例的CLI独有编辑位置、peer同向量及own margin保存，12,960核验、37.705秒、queryGT0；没有新候选/拟合/编码。定义固定：ridge独有修改的peer为朝该编辑方向的 `bilinear(mean64−minmax(source64))`，MEAN独有修改的peer为同方向既有h_R；零点不搜索。PACO ridge独有删除有7,033,353/20,435,405像素（34.42%）收到MEAN同向变化，MEAN独有新增仅535,522/10,449,822（5.12%）收到h_R同向支持。说明二值不一致确实混合了弱同向与真正反向两种情况，但**尚没有这些分组的正确性或分数收益**。剩余Deep100/COCO200明确列缺，不把720改称新评测子集；全1020补齐并封存才进入GT分析。收据 [continuous_edit_direction720_preparation/receipt.json](../../evidence/local/insid3_foris_generalization_20261010/continuous_edit_direction720_preparation/receipt.json)。已有720的descriptor可以直接复用，后续只补缺300的合法字段，不重复准备全套。


[Astra接续原输出核验，避免字段依赖停住分析] 目前未发现活动MEAN重放进程或相应Sol脚本。Astra以独立旧输出核验方式执行 `scripts/verify_mean_continuous_recovery300.py`：仅existing raw、缓存FoRIS的原始tap和不变MEAN函数，先要求PACO sentinel连续场及两帧mask逐位相同；然后恢复Deep100/COCO200的source64/mean64。输出只存连续字段和旧mask差异计数，0新candidate mask文件/encoder/queryGT；若COCO不一致，记录当前raw重放差异，不生成新组合或读标签，其后matched完整候选仍由指定Sol处理。目录 `../cv_data/a/mean_continuous_recovery300_20261011/`，owner收据见 [mean_continuous_replay_audit_owner.json](../../evidence/local/insid3_foris_generalization_20261010/mean_continuous_replay_audit_owner.json)。请复用核验产物，勿重复300旧baseline计算；官方4000生产职责仍归指定Sol。


[Astra旧MEAN核验完成，Sol无需再重放300] `mean_continuous_recovery300_20261011` 已封存全部300连续字段。PACO sentinel的原mean64及两帧mask逐位一致；Deep100/COCO200的current source64全相同、全部600旧mask逐位相同，差异案例0。222.982秒、DINO0、candidate mask文件0、queryGT0。未知的历史连续浮点数组已丢失，不能声称与其逐位比较；这是同公式、同当前缓存且完整输出相同的连续重放。全1020绑定见 [mean_continuous1020_complete_binding/receipt.json](../../evidence/local/insid3_foris_generalization_20261010/mean_continuous1020_complete_binding/receipt.json)，COCO无需额外数值版本候选评分。

缺300的合法独有编辑descriptor已补齐9.227秒，与旧720通过硬链接合并，不重复复制/计算。全1020的peer量、own margin、正/负/零分组，以及每个独有动作内同数量、按own margin排序控制均已在GT前封存；[continuous_edit_direction1020/sealed.json](../../evidence/local/insid3_foris_generalization_20261010/continuous_edit_direction1020/sealed.json)。现在才进入完整六域统一的旧编辑正确性和class计价分析，原完整预测保持不变；主Sol可直接接官方4000准备。


## [指定主Sol] LVIS1400完整验证完成：优于FoRIS/MEAN但未超过多视图强控制

8400mask统一先seal后GT，74,210像素/NumPy全输出/lexsort预算/官方池化核验通过，差0。CLI920槽C44.398755，FoRIS43.762466、MEAN43.806129、ridge42.679952、预算44.214342；C对FoRIS+.636289、MEAN+.592627，两种照片区间均正。原图C44.519548对FoRIS43.912229、MEAN43.872534也正。覆盖876/920、1093真实RQ照片组；仍是开发1400，不称未见确认。

C新增−.263068、删除+.907344，主要增益仍拒假。Astra同输入600强控制已核验：CLI同600观测类C48.445810，九窗query-context51.702409；全920槽C31.595094对强控33.718963，负连接区间[−1.772904,−.911699]。不同历史预算和子集端点都保留，不拿1400绝对分与600混减。新参考拟合1359、原40字段复用、合法全FG无BG参考1例零证据回退；原833输出原样接续。生产耗时下界501.943秒（初次末段未计时）、续跑峰RSS1.570GiB、评分112.682秒，0编码/raw写入/CRF。

[完整报告/紧凑摘要/逐例快照](../../evidence/local/sol_validation_20261010/CONSENSUS_LVIS1400.md)已归档。PASCAL/SUIM原生2000各的新冻结规格接续，不重复已有1400或改参数。主goal继续。


## Astra：连续支持只有明确的非对称用途，冻结下一套完整组合

完整1020六域的方向审计已经完成，35,700项检查、48个独立AUC核对通过；28.562秒，无新encoder/拟合/candidate mask。四类独有编辑都检查，空/全对/全错不从账本删去，先封存方向及同K own-margin控制再GT。结果见 [continuous_edit_direction1020/REPORT.md](../../evidence/local/insid3_foris_generalization_20261010/continuous_edit_direction1020/REPORT.md)。

在当前完整C上下文，**MEAN独有新增且h_R>0**的精确class计价Deep/COCO/PACO/LVIS40/PASCAL40/SUIM40均正：+.150775/+.073358/+.025624/+.001256/+.130090/+.195566。同K按MEAN自身余量选也均正，且PASCAL/SUIM略好，所以当前不能称源证据普遍改善像素排序；source条件控制准入数量也可能解释主要收益。PACO/Deep/LVIS的peer正确性AUC优于own margin，COCO/PASCAL/SUIM的AUC差则不能排除0，仍不可替代完整分数。

对称处理明确不成立：同向ridge独有新增仍使COCO−.436；同向ridge独有删除使PASCAL−.171/SUIM−.614；同向MEAN独有删除使Deep−.504/PASCAL−.374。因而这次不是把所有“同向”一律放回，更不按域名选动作。

[完整组合协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_supported_foreground_protocol.json) 已固定给主Sol：保留原C=majority(B,R,G)；只在 `U=(~B)&(~R)&G` 中恢复 `H>0` 的前景，H为既有reference合法校准、128clip后到目标帧的同一ridge证据，得到 `N=C | (U & (H>0))`。它要求MEAN已经给出完整前景、reference任务也支持该方向，但不再要求ridge先单独越过宿主幅度。原C的全部删除和其余决策不变；H=0停留C，缺角色仍保留B。新增允许进入旧宿主的受保护背景，不能偷偷再套confidence<=.5。

固定Deep100/COCO200/PACO600/LVIS1400/PASCAL40/SUIM40共2380，LVIS40不重复计数；只新两臂：N和同新增K的ownMEAN余量排序控制（同U、row-major ties）。全部字段与母mask均已有，1400的mean64可由atomic tasks索引回原三个source runs，先核对旧两帧mask，无需重算MEAN/拟合/编码。所有9520新mask先seal再GT，CLI主/原图次，保留旧C、完整FoRIS、MEAN、ridge和已匹配九窗强对照。1020的上述数字只是旧编辑的精确定价，完整新mask和更大LVIS迁移尚待主Sol实施。该组合操作本身不作为原创性已充分或五集goal已达成的声明。


## [指定主Sol] 原生PASCAL/SUIM4000：缺失O24单MPS producer实际启动

按Astra新`composition_official_pascal_suim4000_protocol.json`推进各2000完整原生开发输入。已核对磁盘204GiB，补缺4124唯一O24原始输入的未压缩算术64.44GiB，不编码已有key，不增加view/layer。唯一MPS producer`sol_native4000_encode_missing.py`已启动，batch2实际keys/批次/CPU及driver峰/耗时/写入bytes保留`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/encoding/`；实际PID/进度在Sol state。当前driver约8.06GiB，尚无4000新分割成绩。

完整两数据集source manifests、native56/7类槽位和已验证FoRIS+MEAN cache-only baseline调用已固定在同out的sources/、expected_classes.json、baseline_commands.json。原raw编码全seal后用`--no-score`执行完整各2000匹配B/G，然后原R-only/C/预算两frame全seal才queryGT；不因先前数据集结果换公式/阈值/样本或frame。编码器唯一，不重启旧已停止control，不把40例legacy或native2000开发叫官方8459/3859全基准/未见确认。


## Astra：单一Sol缓存实验执行者已接2380冻结组合

沿用户此前明确授权的最多3名Sol子代理，现只恢复既有 `sol_experiments` 执行上面2380冻结方案；另外两个子任务保持结束，Astra仍单代理理论。其唯一新输出目录为 `../cv_data/a/consensus_supported_foreground2380_v1_20261011/`，先检查现有同协议工作并持非阻塞producer锁，避免与原主Sol重复。该子任务仅字段组合/封存/评分/独立验证，不启动encoder、raw生产、新拟合或官方4000。原主Sol可继续官方队列准备，看到此接续记录后请直接复用2380产物。执行owner收据：[supported_foreground2380_execution_owner.json](../../evidence/local/insid3_foris_generalization_20261010/supported_foreground2380_execution_owner.json)。


[Astra机制边界补充] 新分支中h_R>0只是合法reference支持条件，绝不是已校准query正确率。被它暂不接纳的MEAN独有新增（h_R<0）在Deep/PACO/LVIS旧池计价分别−.679/−.404/−.0129，却在COCO/PASCAL/SUIM仍有+.103/+.358/+1.420的价值；尤其SUIM仍丢失大量有益query提议。故不能将h_R<0普遍解释成可靠背景，更不能称这一步已经解决所有不一致。该规则只检验一个跨六域同向正价值的新增职责，保留全部完整强对照与明确的剩余损失。它的可检查差异是：MEAN负责实际前景提议，reference只补充方向支持；反过来用MEAN的弱同向变化批准ridge独有新增，已在COCO亏损，不能互换这两个职责。


## [指定主Sol] 已有Sol助手2380新组合的独立输出核验

按执行owner `/root/sol_experiments`避免重复生产，主Sol对其实际v2已封存9520mask独立只读核验。58,298方向/集合/删集/零回退/数量检查通过；NumPy H和N逻辑逐位一致，1779非平凡own-margin控制截止排名/row-major同分证书通过。两frame新增2,053,502像素、2117 frame-case触发，仅集合数不作GT收益。90.967秒、0queryGT/encoder/新mask/拟合；当前尚无helper report，不声称完整成绩。收据和方法限制见 [2380输出核验](../../evidence/local/sol_validation_20261010/SUPPORTED_FOREGROUND2380_VERIFICATION.md)。

原生4000已在标签评分前固定增加N及其own-margin附加臂，保留原parentC/原预算/FoRIS/MEAN/R共七方法两frame；参数及两数据集规则固定，新的归档`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/composition_plan.json`，无需为N新编码。既有唯一MPS producer继续，不启动第二个encoder。

## [指定Sol] 冻结的2380例 supported-foreground 组合已封存、评分和独立复算

[协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_supported_foreground_protocol.json) SHA `5fd1f4be5abc6473af2b59c60d75a521fb10211269820c093e2949893166f2e2`；纠正v1在首张mask前因manifest键名停止的问题后，独占v2目录 `../cv_data/a/consensus_supported_foreground2380_v2_20261011/` 封存Deep100/COCO200/PACO600/LVIS1400/PASCAL40/SUIM40共2380例、CLI与原图两臂共9520张新mask，读queryGT前封存，新增encoder/raw/fit=0。独立评分器逐例重建N和same-K控制，132040项检查通过、四组掩码像素差异均为0；旧1020子集六域CLI的N-C与此前解析加法一致，最大绝对差`1.43e-14` pp。[完整报告](../../../evidence/local/sol_cache_experiments_20261010/supported_foreground2380/REPORT.md)和[核验收据](../../evidence/local/sol_cache_experiments_20261010/supported_foreground2380/verification.json)。

固定官方class/fold槽位CLI mIoU中，候选N相对旧C的pp增量依次为Deep `+.150775`、COCO `+.073358`、PACO `+.025624`、**LVIS1400 `-.081136`**、PASCAL40 `+.130090`、SUIM40 `+.195566`；原图LVIS1400也为`-.081546`。匹配LVIS600 CLI N`31.470808`、C`31.595094`、fast region强对照`33.718963`；该组合没有超过C或强对照。same-K余量控制在LVIS CLI及PASCAL CLI/SUIM两帧持平或更好，因此不能称H方向有独有排序收益。这是暴露开发池，不是未见验证或完整五集官方结论。

## Sol：seed-2 确认清单的只读暴露资格核查（2026-10-11）

只读取两份 seed-2 val 清单、seed-1 dev_s1_all10000、已完整评分2380及可直接绑定评分报告的旧COCO952/LVIS1400清单中的**已存 RGB/mask hash**；不打开 val query mask/图像、不推断或评分。详细逐域双角色交叉、唯一照片数、输入SHA与覆盖边界见[收据和报告](../../evidence/local/sol_cache_experiments_20261010/confirmation_exposure/REPORT.md)、[机器结果](../../evidence/local/sol_cache_experiments_20261010/confirmation_exposure/receipt.json)。比较均跨 COCO/LVIS 文件路径用 RGB hash 去重；“完全同任务”要求有序R/Q RGB、两侧mask hash及类别/域相同。

seed-2 各域2000、共10000个合法抽样：相对**仅已准备**的 seed-1 dev10000，val逐episode的 Q→旧Q、R→旧R、Q→旧R、R→旧Q 分别4315/4316/4318/4355，任一照片角色命中7269，完全同任务704。相对**已评分2380**则为1467/1358/1349/1490，任一角色3268、完全同任务322；旧已评分 COCO952/LVIS1400 与2380取并集后任一角色3473、完全同任务仍322。上述为重复episode数，非唯一照片数：相对2380的唯一Q→旧Q/R→旧R/Q→旧R/R→旧Q分别808/741/761/823。PACO2000与已评分2380完全同任务168、LVIS154，其余三域0；准备的dev10000相应 PACO479、LVIS207、SUIM15、COCO2、PASCAL1，不能把“准备”都写成曾用GT评分。

val_s2 的 prepared.json 明确是 FRESH_MANIFEST_FROZEN、query标签仅用于**协议准备**；准备过程已物化mask，因此不是“从未接触标注”。本次局部资产检查没有找到 val_s2 的候选编码、预测或评分收据，故只能称当前已准备，不能证明全历史从未编码/读标签。seed-2 与 seed-1 不具照片独立性，且至少322个val任务已在直接核验的2380中按同一R/Q+mask评分；后续若用这批清单，不能称全部样本/标签未见。该审计不删重、不改清单、不另设审批门槛，旧processed/public等缺统一哈希绑定的评分史仍是覆盖边界；当前官方raw补齐不计作queryGT评分。


## [指定主Sol] 2380新新增规则成绩已独立核验，LVIS扩展反证保留

助手v2已完成2380两frame9520mask评分，主Sol全部GT整数nearest、七方法I/U/FP/FN、动作新增数及官方类池化66,724检查通过，最大差1.42e−14；此前58,298完整输出核验也通过。CLI N对C Deep+.150775/COCO+.073358/PACO+.025624/PASCAL40+.130090/SUIM40+.195566；完整LVIS1400−.081136，两种区间负。旧40在完整1400分母贡献+.000702，额外1360−.081834，source方向本身不足以保障查询新增可靠性。PASCAL原图仍低FoRIS，SUIM仍弱MEAN约3点，强目标/原创性均未完成。

[完整成绩、扩展类得失及独立收据](../../evidence/local/sol_validation_20261010/SUPPORTED_FOREGROUND2380_RESULTS.md)。原生4000在标签评分前已保留父C和N附加臂共七方法，不因本次开发结果改配置。主Sol完整CPU cache-only baseline/七臂生产程序已备`sol_native4000_compose.py`，两2000原生配置检查及12实际frame60mask身份复现通过，但未执行整4000；唯一MPS producer继续补缺，实际状态在Sol state。


## Astra：LVIS回退是已有目标附近的几何外扩，下一步保留同等质量处理

全2380已封存输出的CLI只读几何审计完成，16,221检查、114.36秒，0新candidate/fit/encoder。LVIS N−C=−.081136中，距C≤4像素的新增贡献−.082750，其余+.001614；384,857新增FP中89.00%在此窄带。与C不连通的MEAN组件只有+.000060，不能把这次损失主要解释成选错新对象。原C≥75IoU的样本贡献−.063904，但该GT分组仅用于解释。其它五域的近边界新增仍有正值，因此没有依据把所有边界新增剪掉或按域切换。完整精确class计价、正负反例及源证据：[geometry2380/REPORT](../../evidence/local/insid3_foris_generalization_20261010/supported_foreground_geometry2380/REPORT.md)。

独立审计原FoRIS半径5的实际N边界带：覆盖LVIS新增FP84.91%，同时也覆盖新增TP71.25%，不能由可达性宣称CRF必然改善。当前N/C/MEAN/ridge在修改最终mask后都未补原RGB细化步骤；这提供一个直接可证伪的完整组合检查。已冻结[final_refinement_protocol](../../evidence/local/insid3_foris_generalization_20261010/consensus_final_refinement_protocol.json)：全部2380，六个已有完整输入B/G/R/C/N/sameK全部经过同一个原版CPU CRF，同时保留原输出，CLI主、按原FoRIS导出规则原图次。重复B的CRF用于控制double-pass效果，MEAN/R同suffix作为完整强控制，不能把普通细化收益归于source创新。由既有缓存实验Sol唯一执行，0MPS/encoder/raw/refit；主Sol的官方4000冻结比较不改。

## [指定Sol] 冻结2380例统一原版CRF后处理执行归属

主Sol的官方4000生产独立运行；指定现有Sol子代理独占 `../cv_data/a/consensus_final_refinement2380_v1_20261011/`，执行[冻结协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_final_refinement_protocol.json) SHA `856c8e838032f62b59666f66830792f634a59c917607e494d062ea3545abbfa7`。只消费旧mask、score64和query RGB，CPU最多2 worker×2 threads，无encoder/MPS/raw/新拟合。六个完整输入同后缀，全2380双帧与未细化CLI导出控制先封存后读GT；执行进展记录在新目录 `activity.json`，评分与核验完成后回填结果。


## Astra：把当前组合化简，避免把相关判断误写成三份独立证据

令B为完整FoRIS标签、G为MEAN标签、d=|2s−1|为原连续场置信度、H为已校准并clip后插值的reference证据，R由`(2B−1)d+H/2`取号，零点保留B。则C在B背景上新增的充要条件是`G=1且H>2d`；在B前景上删除的充要条件是`G=0且H<−2d`。N只把前者放宽成`G=1且H>0`，后者完全不变。d≥.5时C不能推翻B；N仍保留这一前景保护，却允许越过背景保护。这个等价关系包含严格不等号及tie，100个离散边界组合已核验，[代数收据](../../evidence/local/insid3_foris_generalization_20261010/composition_algebra_receipt.json)。

因此这里的majority不是三个统计独立专家投票：R明确含B，G也继承FoRIS原场。真实增量是reference方向与query图修改共同控制宿主更新。N的新增收益不能被写成已学会一种新的跨图身份关系；它更多释放了已选目标附近的弱支持范围，这与本次几何账本一致。后续统一边界细化检验的是这条完整链是否正确保留了已有像素级处理，不改变该贡献边界。


[Astra覆盖口径核对] 当前2380报告与released seed0元数据逐域核对：[收据](../../evidence/local/insid3_foris_generalization_20261010/current2380_official_coverage.json)。COCO200仅fold0/1、固定40槽实际39类，完整官方是4fold/80类/4000draw；PACO600固定303槽实际264类，完整官方10000draw；LVIS1400固定920槽实际876类，官方23000draw；PASCAL40固定56槽实际27类，官方8459draw；SUIM40已有7类但仅40/3859draw；Deep100是额外自定义道路研究。历史paired点差仍成立于各自完整已测池，但不能将未测fold/类别视为验证过，也不机械缩放出虚构的完整官方成绩。当前4000扩大开发覆盖、完整官方raw成本审计及统一质量处理实验均继续，原冻结任务不删改。

## Sol：官方 s0 全五集 raw O24 只读资源边界（2026-10-11）

按已冻结[官方 s0 清单](../../cv_data/a/official_s0_manifest_20261008/manifest.json)原顺序盘点全部49,318个draw：COCO4000、LVIS23000、PASCAL-Part8459、PACO-Part10000、SUIM3859。使用官方crop/1024变换真实核对每个decoded RGB与FP32 tensor哈希，按同一profile重建key；仅查entry metadata、O/24分支和payload存在性，不打开raw payload、任何mask/queryGT或预测。详细输入SHA、可复核脚本、快照时间和机器收据见[只读盘点](../../evidence/local/sol_cache_experiments_20261010/official_full_raw_inventory/REPORT.md)。

全局去重后只有**28,481个唯一whole/cropped RGB输入及O24 key**；2026-10-11 07:31:10–07:31:12 UTC 固定的cache entry快照中，4,640已缓存、23,841缺失。逐域独特key/缺失：COCO6722/6301、LVIS11213/9016、PASCAL6734/5444、PACO3248/2355、SUIM1503/1428；COCO/LVIS等共享输入，**这些域数不得直接相加**。并发补齐的官方PASCAL/SUIM开发4000计划共4124keys，但在该时点仅2303个属于这份s0清单的尚缺key；若4124全部按同profile完成，s0仍缺**21,538**，并非4124都能从该缺口扣掉。此为条件预测，与当前缓存快照分列，不等于完整FoRIS/MEAN预测或评分可用。

每个FP32 O24数组4096×1024×4字节，当前缺口的未压缩算术量**372.516 GiB**，计划完成后仍**336.531 GiB**；稍后磁盘取样可用**172.187 GiB**。未压缩需求超过取样空间，但NPZ实际压缩与并发写入尚未量化，不能凭此断言最终压缩缓存一定放不下。全部RGB存在，当前缺失key均为快照时entry不存在；未删文件、改精度、筛draw或启动实验。


## Astra：完整官方清单的磁盘边界可用一次编码的临时生命周期解决

Sol已核验全49318draw、28481唯一输入；当前4124key计划完成后仍缺21538key，FP32 raw全留存需336.53GiB。根进一步核对实际writer为`np.savez/ZIP_STORED`，当前编码记录每payload确为16MiB+262字节，不应继续假设压缩会自动消除这个缺口。

但全留存不是完成冻结比较的必要条件。按原官方顺序，以每个输入第一次/最后一次R或Q使用作为存续区间，**全量冷缓存最坏同时仅3586输入，即56.03125GiB raw临时磁盘**。直接加载/释放模拟与独立事件扫描在所有49318位置一致，所有R/Q访问命中、每身份只编码一次、最终存续集合为空。它连已有共享缓存命中也悲观算进临时占用，实际只新建缺失输入会更低。旧研究缓存完全保留；未来只释放该次运行明确拥有的新临时raw，且必须在该例所有冻结方法输出及raw校验和原子提交后才释放。详见[storage plan](../../evidence/local/insid3_foris_generalization_20261010/official_streaming_storage_plan/REPORT.md)。

这是只读已复算调度方案，尚未实现adapter、启动官方全量或删除任何文件；已有4000/2380两条生产不改。持久输出另计：全官方2/3/6臂的CLI+原图bitpack未压缩约14.35/21.52/43.05GiB，示例源字段16.56GiB，最终方法/字段/磁盘余量需按实际冻结配方核定。没有靠降精度、删旧缓存或重复编码来缩小任务。


## [Sol bounded raw replay 适配] 临时缓存接口已核验，未接入官方生产

前节磁盘计划后的状态更新：新增独立 [`scripts/bounded_raw_replay.py`](../../scripts/bounded_raw_replay.py)，**没有修改**旧 `RawFeatureCache`/`CachedDINO` 或活跃4000/2380 runner。固定有序 R/Q identity 与 first/last-use 后，新 run 只把现有 immutable raw payload hardlink 到自己的 scratch，`entry.json` 独立复制；完全缺失的 unique 输入才进调用者回调，之后交给 `CachedDINO(encoder=None)`。冻结每例应有的输出名；逐个文件校验 SHA256 后原子提交该例收据，才能释放自己 last-use 的 scratch。恢复时逐条核对已提交前缀的输出并从首个未提交例续跑；陌生目录拒绝接管。接口不包含新方法、评分或实际 DINO 编码。

[核验报告](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/REPORT.md)及[机器收据](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/report.json)：2项 CPU synthetic 测试通过，覆盖冷热混合、R/Q 跨例重复、提交前/后中断续接、先提交后释放、旧文件字节不变和陌生目录拒绝；另用现有真实 COCO R/Q 的 `O/24` 在无 encoder 模式逐位 replay，通过且旧 payload/entry 哈希不变，模型调用0、query GT读取0。接受记录在 `real_probe_run_v3`；未跑完整49,318，也未证明最终方法冻结配方的输出预算或实际生产吞吐。前节“尚未实现adapter”的时间状态由本节更新，磁盘56.03125GiB仍只是调度上界。


## [Sol bounded raw replay v2 修复] 旧版路径隔离与释放后raw证据缺口已关闭

独立复核在**一次性synthetic shared**上发现 v1 可被替换 `run/raw_cache/<profile_id>` 为 symlink，导致 release 删除假 shared/key；未触碰真实cache。[复现收据](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/root_review_pre_fix.json)。已修 `scripts/bounded_raw_replay.py`：构造、prepare、commit、release 均重新核查 run/raw_cache/profile/64hex key 各段无 symlink 或路径重定向，绑定只接受单层 lowercase 64hex；旧共享cache仍只读。v1 原[报告](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/report.json)/真实收据保留为修复前版本。

v2 每例原子 commit 在输出 SHA 之外，保存每个 R/Q 的 raw key、profile/input hash、entry/payload SHA、所用分支的 shape/dtype/tensor SHA；提交前 fsync 输出与 raw 文件，resume 同冻结分支和identity绑定校验，未到 last-use 的 raw 还核物理文件。5项synthetic测试覆盖构造前/后嵌套 symlink、key symlink、恶意binding、冷热重复与中断续接、释放后raw收据篡改拒绝；只用临时假primary。[v2报告](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/REPORT_v2.md)、[机器收据](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/report_v2.json)。新的真实R/Q无encoder重放在`real_probe_run_v5`逐位一致，旧entry/payload SHA不变，编码/GT读取均0。仍**未接入**全官方producer，不推断49318例运行或最终方法收益。


[Astra v2独立复查] 固定适配器SHA `cd019595bdca93326373b412b24fb200fbb5d9708bcf80ecf86aa9e1c34aab62` 上重新运行5项synthetic回归全部通过；另直接核对真实 `real_probe_run_v5` 的两份R/Q raw收据、当前共享entry/payload SHA及输出SHA，均与提交一致，已释放scratch不存在。没有新模型/GT/生产集成。[复查收据](../../evidence/local/sol_cache_experiments_20261010/bounded_raw_replay/root_review_post_fix.json)。


## Astra：PASCAL2000完整缓存已就绪，可接既有CPU生产

2026-10-11 08:43:47 UTC重新核查完整固定PASCAL2000，所需 **2928个唯一raw输入全部有已提交entry**，profile/input hash、O24 FP32形状及payload存在性一致；SUIM仍由同一MPS producer补缺。已记录[只读就绪收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_pascal_readiness.json)，未读raw payload/queryGT、未开新预测。按主Sol已有per-dataset衔接方案，现在可由指定主Sol运行 `sol_native4000_compose.py baseline --dataset pascal_part` 接 `compose`，无需等SUIM全部编码完；实际reader仍逐payload/tensor核验，冻结规则与两域各2000范围不变。


[Astra执行衔接] 为利用已齐的PASCAL输入，已安排既有 `sol_baseline_mechanisms` Sol子代理仅接 **PASCAL2000 CPU baseline**，原样执行既有 `sol_native4000_compose.py baseline --dataset pascal_part`，使用 native.infer 的非阻塞 `run.lock`；若原主Sol已持锁则不重复启动。该助手不执行compose/score，不开encoder或改方法，完整基线封存后交回原主Sol；原主Sol继续唯一SUIM补缺及后续组合/评分。实际启动PID由执行助手追加，此段是分工，尚不充当启动或封存证据。


## [Sol PASCAL CPU baseline 尝试] PID 1722 已失败，未封存

2026-10-11，按已冻结 `scripts/sol_native4000_compose.py baseline --dataset pascal_part` 原样以 `aidemo` Python 启动唯一 CPU 生产者 PID **1722**；输出目标 `../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/baseline/`，非阻塞锁为该目录 `run.lock`。启动前核读[2928唯一输入就绪收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_pascal_readiness.json)及绑定/profile SHA，未见另一个 native4000 baseline 进程或现成 baseline config。进程持锁后在 `frozen_dino_plan.py:_infer` 的 `from ics import fine_readout` 处以 ImportError/exit1 停止：实际已载入的 `ics` 是 `../cv_data/a/joint_role_pilot200_20261010/frozen/src/ics/__init__.py`，该快照缺 `fine_readout`；`sol_native4000_compose.py` 顶部动态导入的旧 ridge `run.py:23` 先把该 frozen/src 置入 `sys.path`，而后 `frozen_dino_plan.py` 插入主 `src` 不能替换 `sys.modules['ics']`。这是入口导入顺序冲突，不是 raw 缺失或锁冲突。现场只有 `run.lock` 留文本1722，PID已结束，**没有config、字段、预测、seal，也无活动锁主**。未改源码/manifest/公式，未重启、未编码、未执行compose或score；该baseline尚未开始可接受生产，需原主Sol/根决定最小入口修复后再按原锁恢复。


## [Sol PASCAL CPU baseline 恢复] 唯一 PID 2330 已持锁生产

保留失败前driver SHA `f3ee03fb4d416de0d6dec7547b61379b74e784db647f843717f3aa0a1525b335` 的[原副本](../../evidence/local/insid3_foris_generalization_20261010/native4000_driver_before_baseline_import_fix.py)和[exit1收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_pascal_baseline_import_failure.json)。入口只隔离导入：`baseline`不提前载入旧ridge快照，`compose`仍按原顺序载入；原算法、manifest、锁、线程与缓存路径不变。修后driver SHA `cffe8e4b3e0875fd9f671477fef02a010d7dd71c0ae289bb84025451d0c5c352`。[无模型修复核验](../../evidence/local/insid3_foris_generalization_20261010/native4000_baseline_import_fix_check.json)确认 baseline 当前repo/src `ics/fine_readout`、compose旧ridge路径，并重放6域12实际frame/60旧归档mask逐位相同；[SHA绑定](../../evidence/local/insid3_foris_generalization_20261010/native4000_driver_import_fix_receipt.json)。

2026-10-11 已按原命令恢复 **唯一 PASCAL2000 CPU baseline PID 2330**：`aidemo/bin/python scripts/sol_native4000_compose.py baseline --dataset pascal_part`。输出 `../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/baseline/`，`run.lock` 当前文本2330，进程活跃；config/manifest/raw_cache/initialization已写，`inference.jsonl`初始0，**尚未seal**。这是该段唯一锁主；原主Sol勿重复启动。固定 `foris.crf+mean`、CPU2threads、cache-only encoder=None、双frame、no_score/queryGT guard继续；本助手只等此2000例完整封存，不做compose/score或新编码。


[首例无GT读出检查] 首条 `dev_s1/pascal_part/0/0` 已提交，`inference.jsonl` 的该例 `encoder_calls=0`；baseline config为CPU2threads、`encoder_initialized=false`、`query_mask_in_inference=false`，预测NPZ实有 `cli/original × foris.crf/mean` 四个bitpack及原图尺寸，字段和预测SHA均入记录。新PASCAL2000与旧remaining120的PASCAL40在**同R/Q RGB**键上的交集为0/2000（旧40唯一对数40），因而不存在可诚实复用的旧40同输入双帧逐位哨兵；未虚称首例等于旧例。入口修复的已有6域12frame/60旧mask原样重放仍是跨路径读取语义控制。


## Astra：LVIS新误报主要重新打开了原CRF已拒绝的位置

旧2380完整N−C新增按原FP32 FoRIS pre-CRF中间mask分组，19,691项核验、51.108秒，0新candidate/encoder/CRF，未读正在运行的细化输出。LVIS新增FP中284,821/384,857（74.01%）原本在pre-CRF前景内、但被原完整FoRIS的CRF移除；该组在固定最终class分母的可加分解中贡献−.080958 pp，总N−C为−.081136 pp，其它新增−.000178 pp。这比“靠近已有边界”更具体：当前组合重新引入了原质量处理排除的像素。[完整分组与限制](../../evidence/local/insid3_foris_generalization_20261010/supported_foreground_crf_reversal2380/REPORT.md)。

同一组在Deep/PACO/PASCAL/SUIM贡献+.140385/+.004985/+.009103/+.112520，COCO为−.025694，因此不据LVIS新增统一禁止规则。这是旧结果的可加归因，并非移除该组后的反事实方法分数。当前统一F(N)/F(C)/全部强控制实验继续；组合移到原唯一一次CRF之前是另一种完整链位置，可能恢复零证据→原B及一次CRF成本，但不与现double-pass等价，尚未冻结或执行。


[Astra两域基线衔接] 在PASCAL CPU baseline已活跃后，执行助手 `sol_baseline_mechanisms` 的归属扩展为这份既已冻结4000清单的两域baseline：等原唯一MPS producer的encoding seal通过、SUIM全部1310输入ready且无SUIM锁主，再按相同入口启动完整SUIM2000 CPU baseline；资源允许时与PASCAL并行。每域各用原native非阻塞run.lock，若主Sol先启动则不重复。助手仍不做compose/score、不改数学或输入；原主Sol承接后续七方法组合与评分。实际启动/结束证据由助手另记。


## [Sol SUIM CPU baseline 启动] 唯一 PID 4415，PASCAL 2330 继续

原MPS PID90256已结束，`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/encoding/sealed.json` 为 `ALL_NATIVE4000_WHOLE_INPUTS_AVAILABLE`，config/records SHA复核吻合；SUIM2000 的1310唯一O24 entry在[只读就绪收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_suim_readiness.json)与完整编码seal绑定，metadata/profile/input/shape及payload存在性核对通过（raw内容由native reader使用时校验）。确认SUIM无旧进程/锁且系统memory_pressure报告78% free，才用同一修复后driver启动 **唯一SUIM CPU baseline PID 4415**。它持 `../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/suim/baseline/run.lock`（文本4415），已有首4例提交；config为CPU2threads、`foris.crf+mean`、`encoder_initialized=false`、`query_mask_in_inference=false`。PASCAL PID2330保持原锁和进程、已249/2000；两域并行互不抢锁，均尚未seal。此助手仅负责这两条完整baseline封存，不执行compose/score、不新编码或改原方法/manifest。


[SUIM首例无GT读出检查] 首例 `dev_s1/suim/-1/0` 的预测/字段SHA与已提交记录吻合，`encoder_calls=0`；NPZ含两帧 FoRIS/MEAN 四个实际bitpack及原图480×640尺寸。SUIM新2000与旧remaining120的SUIM40同R/Q RGB交集也是0，故没有同输入旧40逐位对照可合法宣称；入口修复仍由预先12frame/60旧mask归档重放检验。


## Astra：原生4000的raw已封存，双域CPU基线接续

原唯一MPS进程90256已实际退出。encoding seal记录完整4238个唯一输入可用，其中新编码4124、2062次B2调用，耗时9976.602秒，新增payload 69,190,319,272字节，driver采样峰8,687,730,688字节，queryGT尝试0、head/CRF/prediction0。根独立核查完整4000输入、4238entry/profile/input/FP32形状与文件大小、4124新key集合及生产收据，33,916项通过；原有114份payload共1,773,792,300字节重新hash未变。新payload未重复全量扫描，实际native reader继续逐文件/tensor核验。[独立收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_encoding_completion_root_review.json)。

两个固定2000域的CPU baseline均已接上并有实际提交：PASCAL PID2330，SUIM PID4415，各持自己的native非阻塞run.lock，由同一现有Sol助手跟进。都使用同一修复后入口、CPU2threads、encoder=None和no_score/queryGT guard；原主Sol不要重复启动，后续七方法compose/score待各域完整baseline封存。2380统一细化仍独立继续；编码完成只证明输入可用，不替代方法完整评分或五集目标。


## [Sol 未运行scorer入口修复] baseline目录变量不再被编辑计数覆盖

根发现 `scripts/sol_score_native4000.py` 的 `main` 原先把baseline目录放在局部`base`，首例尾部 `for base,e in bases.items()` 会将同一函数作用域改成方法名字符串，第二例 `base/'predictions'/...` 必以TypeError中断。原未执行脚本SHA `0fb4720741f35d0b30d4f00edd9ea31b7f7a24ef6f2f9fe7721a7cf87c7c49c3` 及原frozen scorer/receipt已单独保留在 `evidence/local/insid3_foris_generalization_20261010/native4000_*before_shadow_fix*`。只把目录变量改`baseline_dir`、编辑循环变量改`base_arm`，未动统计/排序/分帧公式；新版 [`scripts/sol_score_native4000.py`](../../scripts/sol_score_native4000.py) 与机器`frozen_scorer.py` SHA同为 `52a9440d38f1ef8095caab300098b6839ca8ba8c2e8a80137049f7f53110d1ad`，更新原scorer_receipt并另存[修复SHA收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_scorer_shadow_fix_receipt.json)。[纯synthetic两例及AST检查](../../evidence/local/insid3_foris_generalization_20261010/native4000_scorer_shadow_check.json)复现旧TypeError、新路径两例不变；py_compile通过。**未读新4000 queryGT、未执行score或compose、未改两条活跃baseline。** 评分归属仍待首域baseline seal后由根明确。


## Astra：原生4000后续组合与评分交由现有基线Sol连续完成

2026-10-11 10:07 UTC核查：原主Sol线程仍显示active，但最新实际动作仍是准备driver/scorer，系统无原生4000 compose/score进程。为避免完成的基线等待交接，现将既有 `sol_baseline_mechanisms` 的执行归属扩展为这份固定PASCAL/SUIM各2000的完整CPU链：各域baseline进程正常结束、2000条seal与queryGT guard核验后，继续现有七臂双帧compose，再在该域全部输出seal后用已修复scorer评分。两域规则保持既有冻结版本，不从先得的成绩修改后一个域。

执行者每阶段重新检查实际进程与输出目录，并在对应composition目录持非阻塞进程锁；已有活跃producer时不重复启动，已有完整seal/report直接核验复用。原主Sol此后复用这些产物，不另启动同批生产或评分。此交接不改仍活跃的baseline入口、方法公式、输入、原raw或2380细化执行归属；不新增DINO编码。当前仅确认了接续责任，七臂组合与最终评分仍未执行，实际开始/完成由执行者记录。


## [Sol PASCAL baseline 完整封存] 首域交接检查通过

PID2330 原生 baseline 已正常退出0；`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/baseline/sealed.json` 状态 `ALL_PREDICTIONS_SEALED`、n2000。2000条`inference.jsonl`与manifest同序且episode ID唯一，所有预测/字段SHA和seal所指config/manifest/inference索引SHA核验通过；逐例`encoder_calls=0`、config `encoder_initialized=false`/CPU2threads/`foris.crf+mean`，`query_labels_opened=false`、`query_gt_guard.json`访问尝试空，未评分。首域封存当刻全局ps仅见SUIM baseline PID4415，未见compose/scorer/lock wrapper进程；两域composition目录均尚不存在，系统memory_pressure约80% free。根已授权本助手连续接两域 compose→scorer；下一步只用各自composition非阻塞锁调用原冻结CLI，PASCAL可与未完成的SUIM baseline并行。


[Sol PASCAL七臂compose唯一生产] 已启动独立非阻塞composition锁包装PID **9276**，其子进程PID **9277** 原样执行 `aidemo/bin/python scripts/sol_native4000_compose.py compose --dataset pascal_part`；锁 `../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/composition/run.lock` 文本绑定9276。配置/manifest/expected_classes及records/fields/predictions目录已出现，进程活跃，尚未seal；SUIM baseline PID4415继续。包装仅持锁并调用原CLI，不改公式、不运行scorer。


## [Sol SUIM baseline 完整封存] 两域各2000原生B/G已先于GT封存

SUIM baseline PID4415 正常退出0，`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/suim/baseline/sealed.json` 为 `ALL_PREDICTIONS_SEALED`、n2000；2000条inference与manifest同序且唯一，所有预测/字段SHA与seal所指config/manifest/index SHA均核验通过。config CPU2threads、`foris.crf+mean`、`encoder_initialized=false`，逐条`encoder_calls=0`；`query_labels_opened=false`且query-GT guard尝试空。此刻全局ps仅有PASCAL compose包装/子进程9276/9277，无SUIM compose/scorer；SUIM composition目录尚不存在，memory_pressure约79% free。两域原生B/G完整输出都已封存，未评分。根据根更新授权，此助手接两域各自锁定compose→已修复scorer，SUIM可与PASCAL compose并行，保持同公式、双帧与固定输入。


[Sol SUIM七臂compose唯一生产] 非阻塞锁包装PID **9867**、子进程PID **9868** 已启动原CLI `aidemo/bin/python scripts/sol_native4000_compose.py compose --dataset suim`，锁在`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/suim/composition/run.lock`；SUIM已写首15记录，PASCAL独立包装9276/子进程9277仍在自己的目录写到1725记录。两个composition锁互不重叠，均尚未seal；没有执行score或新编码。

## [指定Sol] 冻结2380例六臂统一原版CRF后缀：全量封存、评分与独立验证

按[冻结协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_final_refinement_protocol.json)完成全部Deep100/COCO200/PACO600/LVIS1400/PASCAL40/SUIM40，不筛例。六个既有CLI完整mask（B/G/R/C/N/ownK）统一走不变原版CPU CRF，再从refinedCLI导出原图，并保存每臂未细化CLI的对应原图导出控制；全部14280张refinedCLI、14280张refined原图、14280张未细化原图控制先seal才读queryGT。2 CPU worker各2thread，真实CRF调用11420次、同query逐位相同输入复用2860次，wall10730.17秒，新增encoder/raw/fit=0。六域首例score64→原FoRIS严格重放旧B均逐位相同，首例plain/prepared逐位一致；后封存独立评分91380项检查、六域32次plain CRF重放像素差0、旧父I/U再核28560项。结果与哈希：[完整报告](../../evidence/local/sol_cache_experiments_20261010/final_refinement2380/REPORT.md)、[核验收据](../../evidence/local/sol_cache_experiments_20261010/final_refinement2380/verification.json)。

CLI固定官方class/fold槽位，N相对**同后缀C**的pp差为Deep `+.789135`、COCO `+.167571`、PACO `+.028203`、LVIS `+.081560`、PASCAL `+.085719`、SUIM `+.288369`，六域query-photo配对95%区间均正；LVIS未细化N-C为`-.081136`，符号翻转，支持边界处理敏感性。然而Deep完整N从30.376570降至26.889854（同输入后缀效应`-3.486716`pp，区间`[-4.611614,-2.518003]`），低于旧完整FoRIS29.377542；同后缀MEAN在COCO64.240747、LVIS44.858072、SUIM65.948667均高于N的64.060310/44.644789/62.912770。N相对同后缀ownK六域区间全部含0；原图帧方向相同。匹配LVIS600 CLI N31.594662仍低于既有九窗强控制33.718963，Deep既有fast37.344028远高于N26.889854。旧direct-original与未细化CLI导出已分开报告，不能混充纯CRF增益。两帧N/ownK对同后缀C在本2380例零单调性违例，不外推普遍性质。此开发池实验没有证明原创性、独有ridge排序收益或Strong/五完整官方集目标达成。

[指定Sol调用成本补注] 上述11420次是2380例全量生产的实际CRF调用数；另有封存前7次首例sentinel调用（细化耗时18.65秒）和封存后独立plain CPU重放32次。生产wall10730.17秒不含这些核验调用。


## [Sol PASCAL七臂双帧封存→评分唯一生产]

PASCAL compose 包装9276/子进程9277 退出0，`../cv_data/a/consensus_official_pascal_suim4000_v1_20261011/pascal_part/composition/sealed.json` 为 `ALL_NATIVE2000_SEVEN_METHODS_DUAL_FRAME_SEALED`、n2000。2000条按manifest顺序，2000个record/预测/字段SHA及config/manifest/expected-classes SHA全核通过；CLI主/原图次、`new_fits=2000`、`reused_fits=0`、缺角色0、`query_gt_access_attempts=[]`、`new_encoder_calls=0`，compose用时585.330秒、peak RSS1.564GB。评分开始前全局ps仅SUIM compose包装9867/子进程9868，无PASCAL重复stage；按根连续授权，用PASCAL composition的非阻塞锁启动**评分包装PID10110/子进程PID10111**，原CLI `scripts/sol_score_native4000.py --dataset pascal_part`，此时报告尚未出现。它仅消费已seal的两帧七臂，评分入口修复SHA `52a9440d...`；不影响SUIM compose，未改任一数据集公式或输入。

[Sol PASCAL评分数值重放修复] 首次scorer包装10110/子进程10111在2000例第352例 `dev_s1/pascal_part/0/351` 原图 `source_budget` 独立重建差2像素后exit1；没有写report/scored文件，composition封存原样。定位到NumPy独立双线性与生产端torch FP64插值在删点预算cutoff处相差1 ULP，排序选中相邻的2个像素；非预测或输入污染。评分脚本只在预算排序得分上使用生产端FP64插值，并逐帧要求独立NumPy双线性最大差≤1e-12；独立NumPy排序、Boolean重建、GT/指标和其余检查不变。旧评分脚本/receipt快照与[诊断收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_scorer_numeric_replay_fix_receipt.json)保留，修后脚本及frozen副本SHA `dec26ebdaa0b09239dd261d2100e80ece6bb98f4f322d26e3eb4602625f79f1d`；失败例定向重放0差、py_compile通过。确认无PASCAL评分锁主后用原非阻塞包装重启，包装PID10407/子进程10408；SUIM compose9867/9868持续且目录独立。仍未更改任何生产mask/公式/manifest或新编码。


## [指定主Sol] 原生4000生产与评分完整入口已接续

在唯一MPS producer持续填缺期间，主Sol已补完七方法两frame完整scorer（独立NumPy/readout/rank证书/真实像素/官方池化/两种照片区间）。API全2000源契约、12真实frame60既有mask身份复现与语法检查通过，尚无native4000新score。已去除无必要的global4000raw等待门槛：完整PASCAL2000所需2928key到齐即可CPU baseline/compose，与MPS继续填SUIM所需1310key重叠；仍保留全部2000样本/参数/最终统一seal。readiness元数据不冒称完整payload验证，reader逐使用SHA。source程序及明确测试范围见 [4000接续入口](../../evidence/local/sol_validation_20261010/NATIVE4000_EXECUTION_READY.md)。新cached2380 refinement由原指定Sol助手唯一生产，不重复其CRF或碰官方4000方法规格。


## Astra：二次细化的完整结果不接纳，下一步只移动到原唯一CRF之前

2380统一细化已完整封存、评分和原生重放。根从逐例整数I/U独立重聚合两帧全部198个完整域点估计，最大差1.42e-14；核查seal/score绑定、六域32次plain重放零像素差及全2380两帧N/ownK包含C的零违例。见[根复核](../../evidence/local/insid3_foris_generalization_20261010/final_refinement2380_root_review.json)与[完整结果](../../evidence/local/sol_cache_experiments_20261010/final_refinement2380/REPORT.md)。LVIS N-C从−.081136翻为+.081560，但Deep N因suffix下降3.486716、最终比原FoRIS低2.487688；同后缀MEAN仍在COCO/LVIS/SUIM更好，六域N-ownK的配对区间均含0。边界处理敏感性得到支持，完整跨域质量和新增source排序贡献没有据此成立，不接纳二次CRF为最终方法。

下一项仅改变完整链中的位置：[single-final-CRF固定协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_single_final_crf_protocol.json)。用原生FP32 score64重建第一次CRF之前的B0，保留既有G/H/d及相同R/C/N/ownK公式，把宿主B换为B0后只过原一次F。主臂F(N0)，完整控制F(C0)/F(R0)/F(K0)，复用原FoRIS=F(B0)、已封存F(G)及输入/RGB/参数完全相同的旧F结果。H=0时自然还原原完整FoRIS；这是另一条链，不能叫等价优化，也不保证能修复SUIM的身份/召回差距。全2380顺序和双帧不变、无新编码/raw/拟合/参数搜索，全部新预测先seal后评分。

该项继续由已有 `sol_experiments` 独占新输出目录；原生4000由另一个Sol按既有七臂协议连续完成，规则和产物不因新实验更改。当前是固定规格，实际启动和封存由执行者另记。

## [指定Sol] 单次最终CRF链位置测试执行归属

指定现有 `sol_experiments` 独占新目录 `../cv_data/a/consensus_single_final_crf2380_v1_20261011/`，按[冻结协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_single_final_crf_protocol.json) SHA `a07ff76f8e5d1cfe9ce9814583bc7404561028b6f54858f8397514a3dc529f72` 执行同序2380完整案例。只把B宿主换成原FP32 score64重建B0，再一次原版CPU F；先逐例检查旧F输入/RGB/参数/输出可复用来源，四臂R0/C0/N0/K0与双帧全部封存后评分。已有原生4000生产/方法/目录不动；日常状态写新目录activity.json。此处是执行起点，尚无新成绩。

[Sol PASCAL完整评分/SUIM评分启动] PASCAL 修后唯一scorer包装10407/子进程10408 正常退出0，`pascal_part/composition/report.json` 为 `COMPLETE_NATIVE2000_SEVEN_ARMS_SCORED_AND_VERIFIED`，双帧七臂各2000，228014项检查、重建像素差0、56/56官方class/fold槽位，report/seal/scored文件SHA互相绑定，scorer SHA `dec26ebd...`。CLI七臂mIoU（%）：B54.788956、G55.249268、R55.785157、C55.508974、source-budget55.261308、supported-N55.677669、ownK55.659050；原图对应B55.496178、G55.715834、R56.534176、C56.255810、source-budget55.985842、N56.394769、ownK56.386686。SUIM compose包装9867/子进程9868 正常退出0，seal `ALL_NATIVE2000_SEVEN_METHODS_DUAL_FRAME_SEALED`、2000条按manifest同序唯一、所有预测/字段及config/manifest/class SHA全核，GT guard[]、new encoder0、new fits2000、reused0、缺角色0、wall615.698s、peak RSS1.749GB。全局ps无另一个Sol compose/scorer后，SUIM评分独占composition非阻塞锁，包装PID10792/子进程10793，原CLI同修后SHA；报告尚未出现。

## [Sol PASCAL/SUIM native4000 最终封存评分] 固定各2000例、七臂双帧

两域B/G各2000例均先于GT封存；随后各2000例七臂CLI及原图mask封存，最后在各自非阻塞锁下独立评分。PASCAL/SUIM `composition/report.json` 都为 `COMPLETE_NATIVE2000_SEVEN_ARMS_SCORED_AND_VERIFIED`；各228014项检查、重建像素差0、2000个双帧评分记录同源顺序唯一、report↔sealed↔scored SHA绑定、scorer SHA `dec26ebdaa0b09239dd261d2100e80ece6bb98f4f322d26e3eb4602625f79f1d`。评分使用官方固定class/fold槽位，PASCAL 56/56、SUIM 7/7均观测到。两域baseline `query_labels_opened=false`、inference guard空、逐例encoder_calls=0；composition guard空、新encoder/CRF=0，2000次新reference-only fit、复用0、缺角色0。SUIM scorer唯一包装10792/子进程10793退出0。固定输入为暴露的官方development各2000原生子序列，**不是**完整PASCAL8459/SUIM3859官方集合或未见确认。

主CLI分数为官方固定槽位mIoU百分数；原图分数为同预测链的次要帧，不混作CLI增益。七臂顺序B=`foris.crf`、G=`mean`、R=`anchor.global`、C=`shared_edits_complete`、K=`same_edit_budget_source_order`、N=`supported_mean`、ownK=`own_margin_sameK`。

| 域/帧 | B | G | R | C | K | N | ownK |
|---|---:|---:|---:|---:|---:|---:|---:|
| PASCAL CLI | 54.788956 | 55.249268 | 55.785157 | 55.508974 | 55.261308 | 55.677669 | 55.659050 |
| PASCAL 原图 | 55.496178 | 55.715834 | 56.534176 | 56.255810 | 55.985842 | 56.394769 | 56.386686 |
| SUIM CLI | 59.182775 | 59.260172 | 58.426355 | 59.419885 | 59.261611 | 59.632460 | 59.623091 |
| SUIM 原图 | 59.660311 | 59.549329 | 59.169070 | 59.907918 | 59.758717 | 60.095403 | 60.083051 |

对主CLI的N−C：PASCAL +0.168695 pp，query-photo配对95%区间[+0.135975,+0.202707]，R/Q-photo连通组区间[+0.119118,+0.234361]；SUIM +0.212575 pp，对应[+0.152789,+0.277691]、[+0.082028,+0.366537]。N−B、N−G在PASCAL的两类CLI区间均为正；SUIM中query-photo配对为正，但R/Q连通组区间跨0。PASCAL N−R点值−0.107488 pp、两类区间跨0；SUIM N−R点值+1.206105 pp，但R/Q连通组区间[−0.129879,+1.740515]跨0。PASCAL N−ownK CLI +0.018619 pp，两类区间均>0，效果很小；原图+0.008082 pp、两类区间跨0。SUIM N−ownK CLI +0.009369 pp、两类区间跨0；原图+0.012352 pp，query-photo区间>0但R/Q连通组区间跨0。两域N−C原图也为正且R/Q连通组区间>0：PASCAL +0.138959 [+.074351,+.271005]、SUIM +0.187485 [+.091156,+.373066] pp。所有区间只是给定development图池条件下的重抽样，不代表未见数据不确定性。

真实额外成本按域：PASCAL composition 585.330秒、peak RSS1.564GB，评分245.102秒；SUIM composition 615.698秒、peak RSS1.749GB，评分330.756秒。每域composition新fit2000、encoder/CRF=0；baseline是各2000例已有raw cache上的CPU2thread新B/G预测，未在baseline seal中记录wall/RSS，故不伪报baseline用时。旧PASCAL/SUIM40与此native2000的同R/Q RGB对交集均为0，不能虚称旧例逐位重放哨兵。评分入口曾有变量遮蔽与1-ULP插值排序边界两处验证程序修复，旧源码/失败收据均留存；没有改变任何已封存生产预测、组合公式、输入或manifest。


## [主Sol独立归档] 原生4000七方法双frame完整结果

两域各2000原生baseline/compose/score均完整结束，主Sol逐文件SHA及整数/官方池化860048项通过，mIoU最大差7.11e-15；没有重复像素GT评分或新增编码/预测。CLI N PASCAL55.677669/SUIM59.632460，FoRIS54.788956/59.182775、MEAN55.249268/59.260172；PASCAL R仍55.785157，不能宣称最佳或参考筛选独有收益。所有官方槽位56/56、7/7保留。详细矩阵、两种区间、既有反证及真实成本见 [4000完整报告](../../evidence/local/sol_validation_20261010/NATIVE4000_RESULTS.md) 和同目录native4000_summary.json。主goal仍未完成，当前单次最终CRF2380由既有指定助手唯一生产。


## [主Sol只读审计] 原生4000相同补点数量下的位置收益

固定N与MEAN own-margin K每例都保留C且补点数相同。封存整数账本40,252项独立恒等式/官方计价检查通过，0新GT/预测/fit/encoder，CPU0.231秒；精确class贡献公式误差≤1.24e-14。CLI PASCAL N-C+.168695、K-C+.150076、N-K仅+.018619；SUIM分别+.212575、+.203207、+.009369。PASCAL N多TP417例、K多TP258例、相同1325；SUIM511/457/1032。PASCAL主frame微小N-K增益有正区间，但原图区间跨0；SUIM主frame两照片区间均跨0。不能把C到N的主要扩张收益全部归给参考方向选点，也不能把净多TP直接代替class/fold mIoU。详情、真实post-seal正负反例ID和逐class可加贡献见 [同预算审计](../../evidence/local/sol_validation_20261010/NATIVE4000_BUDGET_SELECTION.md)。当前单次CRF冻结链不改，等待其完整输出。


## [主Sol无GT核验] 单次最终CRF提交前缀与真实复用

已对当时689/2380提交记录完成27,616项检查、2775文件SHA，3.667秒，原FP32 B0及四臂输入/输出/旧缓存身份/新memo归属逐位相同；0核验CRF/encoder/GT。2756逻辑结果对应1696真实新F、28旧完成F精确复用、1032同query输入相同memo。未观测到zero-H，最终六域全部输出/导出/退化例/GT/scorer/plain replay仍未完成。现有PID11028/CPU workers继续，不重启/改公式。[前缀核验报告](../../evidence/local/sol_validation_20261010/SINGLE_FINAL_CRF_REUSE_PREFIX.md)。


## Astra：扩大原生队列后的结论与单次质量链的边界

原生4000两域全部七臂、双帧完成后，根从原整数I/U独立重聚合28点，并核查56,000组I/U/FP/FN恒等式、全序列身份、全部56/7槽位、report/seal/scored/scorer SHA；最大点差7.11e-15。[独立收据](../../evidence/local/insid3_foris_generalization_20261010/native4000_results_root_review.json)。PASCAL CLI N−B为+.888713，query-photo区间[+.692135,+1.122047]、RQ连通组[+.397702,+1.066238]；SUIM CLI N−B为+.449685，前一种区间正而RQ连通组跨0。SUIM有48个RQ连通组，最大组覆盖1941/2000个episode，这一重抽样的有效结构必须一并报告，不能把2000次draw说成2000张独立照片，也不在看到结果后改用另一组法来恢复显著性。

旧40例和新各2000例没有相同R/Q RGB对；原生SUIM2000上N59.632460高于MEAN59.260172，旧40中MEAN明显更好的排序没有在此队列延续。只能保留各自同批比较，不能把绝对分数跨池相减或将变化全归因于一个已证明的病因。PASCAL N−ownK的CLI小正差+.018619有两类区间支持，原图+.008082区间跨0；这个新证据与旧2380中同后缀N−ownK均不确定并存，不偷换cohort/frame/后处理。

单次末端CRF的动机还可以直接从完整控制读出：F并非幂等，Deep完整FoRIS B29.377542再次通过F后变25.601976，所以“多做一次原质量处理”本身就会破坏质量。把组合移到原唯一F前，可在H=0时严格回到原完整FoRIS，并恢复一次CRF的算法调用次数。这个结构身份只保护零任务更新；非零H是否真正改善完整输出，仍以当前全部2380固定实验和匹配强对照为准。原生4000结果不改变该已冻结实验。


## [主Sol机制覆盖边界] 4000普通benchmark例不等于4000次Mask-only干预

精确保持裁剪后R/Q RGB hash、尺寸/裁剪相同而参考Mask和任务类别变化，PASCAL2000只有1组2例，SUIM2000无组；每个hash均非空，未用None照片ID分组。唯一匹配episode /3/312与/3/422尚未控制forward随机状态，只是机制候选绑定。4000完整成绩和同预算审计不能外推为4000次同图任务切换机制验证；原同图60对三域因果证据分列，后续其它域需冻结同输入/随机状态成对协议并复用现有raw，不换参考RGB冒充Mask-only。当前2380完整链位置试验不改变。[覆盖报告](../../evidence/local/sol_validation_20261010/NATIVE4000_MASK_SWITCH_COVERAGE.md)。

## [Sol 全官方执行前置] 逐episode有界cache原子消费，六例真实差分通过

新 [逐例核心](../../scripts/sol_episode_cache_core.py) 在已冻结row上将原生FoRIS CRF+MEAN、旧reference-only ridge及已冻结七臂双帧读出接到同一`BoundedRawReplay`持有的run-owned O/24 scratch；先写齐baseline mask、score64/mean64、ridge字段、七臂双帧mask、row record五项，再由原adapter对全部文件与raw收据原子commit，**commit后**才`release_committed`按last-use释放scratch。缺raw无encoder callback而失败。保留当前repo `ics`先导入以避开旧ridge快照的同名包冲突；原native4000完整脚本、已seal生产目录、共享raw、方法公式均未改。原adapter SHA仍`cd019595bdca93326373b412b24fb200fbb5d9708bcf80ecf86aa9e1c34aab62`。

[六例真实差分](../../evidence/local/sol_cache_experiments_20261011/episode_atomic/REPORT.md)只运行已有封存身份：旧序COCO/PACO/LVIS首例，新native PASCAL/SUIM首例，LVIS `dev_s1/lvis/0/148` 缺reference BG零证据例。各例绑定R/Q解码RGB、crop、reference mask、原raw profile/key/input/tensor hash及归档文件SHA；score64、mean64、校准H、七臂CLI和原图mask全部逐位一致，非退化5例calibration JSON完全一致。零证据例原旧LVIS与较新native4000的fallback文字/schema不同，但阈值=分离度=0、reference IoU未定义、无fit、H=0及全部mask一致；不虚称该元数据字节相同。六例五项输出均有commit收据，release后owned raw entry为0、旧shared entry/payload SHA不变、encoder/queryGT访问0；第二次resume核验也通过。核心SHA`200c3fd0338d875c9e68e9d2b4cefce2221b66676153bc6821dca3b5d85fbb31`，[机器收据](../../evidence/local/sol_cache_experiments_20261011/episode_atomic/receipt.json)。这仅补全量比较所需的逐例执行能力；没有启动49318全官方队列、新编码、GT评分或新one-CRF公式，也不据此选择最终方法。


## [主Sol机制审计] 非缺失参考也可在校准处失去新增任务通道

6380条metadata及5400套保存reference字段48,642项检查通过，0新GT/fit/encoder/候选/CRF。LVIS7、PASCAL1、SUIM4个非缺失episode（共10不同R/Mask；SUIM3例重复同R）h=0。实际原Mask都有FG/BG，全参考分离度+1.0248至+1.4805，但拟合128质量分位点后校准仅余0.125%–0.760% FG质量，分离度−.1155至−.01115，于是调用者清零整个新增ridge证据，保存query ridge场仍非恒定。引理：未入fit的coverage原子c_i≤sum(c)/128；FG面积<1/64时，剩余没有前景占多数patch。正分离度病例也有低支持反例，full-reference指标含训练数据，不能据此自动换阈值或宣称query修复。这是条件机制候选，不是完整模型Mask无响应/原创性/收益证明。现有2380链与所有参数不变。[报告及重现](../../evidence/local/sol_validation_20261010/REFERENCE_CALIBRATION_SUPPORT.md)。

[Sol 逐例跨行恢复/last-use闭环] 再用**同一个**`BoundedRawReplay`真实三行日程：旧COCO首例→LVIS缺reference BG零证据例→同一COCO首例。第0行五项输出commit后重建adapter/core；恢复`next_index=1`且COCO两项owned raw仍保留；第1行所有字段与七臂双帧继续对封存结果逐位一致，仅到期LVIS两项释放；第2行COCO同输入重放一致，五项输出commit后才释放COCO两项。三个commit输出及raw收据SHA核对，最后owned raw entry 0，旧shared entry/payload SHA不变、encoder/queryGT调用0。独立[三行收据](../../evidence/local/sol_cache_experiments_20261011/episode_atomic/three_row_resume_receipt.json)和[说明](../../evidence/local/sol_cache_experiments_20261011/episode_atomic/REPORT.md)；未重跑原六个单例、未改adapter/公式/封存目录或开启全量队列。


[主Sol零H实际身份补注] 当前已提交7个零Hcase（6个正常角色负校准+1个实际缺BG），四个pre与B0及四个final与旧完整FoRIS在CLI/原图全部逐位相同；每case真实新F0、旧F精确复用4。只核已提交case，不提前读GT或称完整2380通过。收据 `evidence/local/sol_validation_20261010/single_final_crf_zero_H_identity.json`。


## [主Sol全量验收] 单次最终CRF2380已封存，scorer接续

producer PID11028及workers正常结束，9520 finalCLI及native输出全部先seal，wall2825.153秒，真实新F5845、旧完成F复用180、同query memo3495；encoder/MPS/raw/fit/queryGT尝试0。主Sol完整2380输入/旧输出/memo95,471项及9665文件SHA通过，8个零H四臂两frame还原旧完整FoRIS。seal/config/manifest/preflight/完整ledger↔ordered index绑定已核。现有指定助手已启动唯一scorer PID15911，尚无最终成绩；原图导出/GT/六域plain native replay仍待该scorer验收。[生产和源核验报告](../../evidence/local/sol_validation_20261010/SINGLE_FINAL_CRF_EXECUTION_2380.md)。

## [指定Sol] 冻结单次最终CRF链位置测试：2380例双帧已封存、评分和plain重放

沿[固定协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_single_final_crf_protocol.json)把宿主从旧post-CRF B移到FP32 score64重建B0，只对R0/C0/N0/K0各过原一次F；全部2380×4最终CLI/原图9520+9520张、B0及四臂pre控制各11900张双帧均先seal后queryGT。生产新CPU CRF调用**5845**、旧封存F精确输入命中**180**（原FoRIS89、旧F(anchor)91）、同query先前新臂结果复用**3495**；两CPU worker各2thread，wall2825.15秒，新CRF worker-seconds5315.76，新增encoder/MPS/raw/fit=0。独立评分110732项检查通过；六域17次新输入plain CPU重放零像素差，294个完整域/帧/臂官方槽位点估计再聚合最大误差1.42e-14pp。8个H=0的LVIS例（含缺reference角色1例）四臂pre=B0、final逐位等于原完整FoRIS。证据：[完整报告](../../evidence/local/sol_cache_experiments_20261010/single_final_crf2380/REPORT.md)、[核验收据](../../evidence/local/sol_cache_experiments_20261010/single_final_crf2380/verification.json)。

CLI固定官方槽位N1=F(N0)的mIoU为Deep29.685895/COCO64.113892/PACO39.631330/LVIS44.872437/PASCAL40 36.520735/SUIM40 62.844617。相对原完整FoRIS依次为+.308353/+.943662/+1.193277/+1.109971/+.097410/+1.427721pp；Deep相比二次CRF N26.889854恢复，但对旧FoRIS的配对区间跨0，且既有fast强控37.344028仍远高。LVIS相比同后缀C1仅+.012168（配对区间跨0），比同K控制K1 **-.005780**；匹配九窗LVIS600强控33.718963>N1 31.925710。PACO虽比F(G)+.186222，既有task强控39.727550仍高于N1 39.631330；SUIM40比F(G)低3.104050。LVIS相对F(G)仅+.014366，query-photo percentile区间与点估计偏离，连接R/Q照片区间跨0，不能宣称稳健超越。旧40例SUIM不代表另行完整执行的原生2000例。本实验既未改变原生4000，也未建立独有来源排序收益、所有强控制胜出、原创或Strong目标完成。


## [主Sol完整结果] 单次最终CRF2380仍未达到Strong目标

完整scorer110732项、六域17次plain replay差0；主Sol95,471输入/复用及565,488整数/合池检查通过，312点误差≤1.42e-14。CLI N Deep29.685895/COCO64.113892/PACO39.631330/LVIS44.872437/PASCAL36.520735/SUIM62.844617。Deep比二次N+2.796041但仍低旧N.690675/强多视图7.658133；COCO和SUIM低同F MEAN，SUIM差−3.104050两区间负。所有CLI N-K RQ区间含0，原图LVIS N-K两区间负；D/P单次R还高于N。旧40PASCAL原图仍低FoRIS，完整原生4000分列不混。8零H结构身份通过，完整链未接纳。详见 [完整结果和家中摘要](../../evidence/local/sol_validation_20261010/SINGLE_FINAL_CRF2380_RESULTS.md)。已结束进程不再称运行，主goal保持进行。

## Astra：下一次只检验参考校准支持，先恢复支持再判定跨图作用

两种CRF位置比较已经回答了质量链问题，不再重复这两套生产或按域挑后处理。新的具体证据来自[参考校准支持审计](../../evidence/local/sol_validation_20261010/REFERENCE_CALIBRATION_SUPPORT.md)：确定性质量分位采样把大部分前景拿去拟合，余下补集不能代表原参考的角色分布；12个完整角色episode甚至因补集分离度为负而清零新增通道。更广泛的低支持正分离度反例表明，这不是一个可以按面积门控的充分失效条件。全参考含拟合点的高分离度也不能直接证明修复有效。

保持原拟合索引、方向、FP64 query ridge128、APD分支、post-CRF FoRIS宿主及R/C/N/同数量控制不变。拟比较两个明确的参考标定输入：全4096点的原参考预测（含拟合点的自拟合控制），以及只把拟合点换成其固定权重逐点删除预测的全参考预测。未拟合参考点保留原值，两者都用原strict pixel-mass IoU阈值和FG/BG面积均值分离度；没有新的阈值网格、系数或query标签。缺完整参考角色仍保留H=0语义。

具体数学：固定原weights w、S=sum(w)、加权中心化Xc和旧lambda=.01，K=Xc Xc^T、A=K+lambda diag(1/w)。不惩罚截距的线性平滑矩阵为 H=1(w/S)^T+K A^-1，因为K A^-1 1=0。拟合点的删除预测为 `s_minus_i = y_i - (y_i-s_i)/(1-H_ii)`。计算的是移去一个损失项、其余weights和lambda保持原值的解析反事实；删除后截距仍重新拟合。它不重新抽样或重算FG/BG归一权重，不声称整个自适应采样过程的无偏交叉验证，也不消除同图空间/主干上下文相关。真实直接删除重拟合和独立实现必须先吻合，才进入候选生产。

这条恒等式以及用预验证预测校准ridge均有明确先例：[Dempster等，Prevalidated Ridge Regression，§3/Theorem 1](https://arxiv.org/html/2401.15610v2)。当前只借用经典工具检验已发现的校准支持偏差；既不把它命名为新核心算法，也不把参考分离度恢复当作query提升。完整输出若不胜旧同链N、全参考自拟合控制或必要强控制，应保留负结果；不再据结果扫描留出比例、权重或阈值。

执行顺序固定为已完成单次CRF manifest 的原2380例，再原生PASCAL2000和SUIM2000，共6380，保留重复draw、原frame和每个cohort独立的class/fold槽位；不能把旧40和新2000相减后归因于本干预。现有 `sol_experiments` 仅做源绑定及旧fit/calibration复现准备；`sol_baseline_mechanisms` 独立做公式与真实参考验证，均不读新query GT或生成候选。完成数学与源身份核验后，再封存最终协议和生产代码，所有6380完整预测先于任何本轮评分。原主Sol复用这些产物，无需另启同一实验；当前也不启动49318全官方。

## [Sol 固定权重ridge逐点删除] 参考校准数学与既有路径核验，未生产候选

在原`ridge_only`的**同一批**量化拟合patch上，令`X`为固定APD分支后的单位参考O24特征，`y=2c−1`，`w=.5c/C+.5(1−c)/B>0`，`S=Σw`，`Xc=X−(wᵀX/S)`，`K=XcXcᵀ`，`A=K+.01 diag(1/w)`。原拟合最小化`Σw_i(y_i−b−x_iβ)²+.01||β||²`，截距`b`不惩罚。其线性平滑矩阵可写为`H=1(w/S)ᵀ+K A⁻¹`：因为`Xcᵀw=0`，故`Kw=0`、`Aw=.01·1`、`KA⁻¹1=0`，加权中心化后的第二项无需额外均值交叉项。因此`H_ii=w_i/S+(KA⁻¹)_ii`。单独删除第`i`个损失项使正规方程矩阵减去`w_i z_i z_iᵀ`（`z_i=[1,x_i]`）；Sherman–Morrison给`1−H_ii>0`时` s₋ᵢ=y_i−(y_i−s_i)/(1−H_ii)`。这里的删除后截距**重新拟合**，其余patch的旧`w`及`.01`保持不变；不重新选择ids，也不按剩余FG/BG重平衡。对未在拟合集的参考patch仍用原保存分数，仅以`s₋ᵢ`替换拟合位置，再把全部4096点真实连续coverage交给**原**`reference_threshold`做strict `>`像素质量IoU阈值和class-mean分离度标定。这不是按所有4096点重新训练ridge。

[唯一核验脚本](../../evidence/local/sol_cache_experiments_20261011/reference_calibration_loo_math/verify.py)用17×7合成固定权重数据逐个直接删点并重算截距/加权ridge，全部17点公式最大差`4.47e−14`；真实7例各取含最高杠杆在内的3–4点直接删点，最大差`1.55e−15`，`1−H_ii`最小`0.71747`。读取的是真实reference RGB/mask及已存R raw O24；APD布尔值沿用封存记录，没有读取query raw/GT。原完整拟合分数经原strict-IoU补集标定，在7例的阈值、分离度、两类均值和IoU上与封存calibration**数值完全相同**；可得归档reference_score的例子也逐位相同。原profile/key/input/tensor及R RGB/crop/mask哈希均核对。0 encoder、新query场、新mask、GT读或CRF。

|参考例|拟合点|原未拟合补集FG质量占全部|补集`c≥.9`数|原补集分离度|4096点逐删标定分离度|原补集IoU / 新同图全参考IoU|
|---|---:|---:|---:|---:|---:|---:|
|Deep首例|249|73.99%|3|0.46999|0.48214|0.28117 / 0.30926|
|COCO首例|256|15.00%|6|1.29781|1.64434|0.49449 / 0.81655|
|PACO首例|256|13.04%|7|1.31854|1.44010|0.30883 / 0.74449|
|LVIS首例|256|85.82%|633|1.64630|1.64951|0.89921 / 0.90588|
|新PASCAL首例|256|89.43%|1004|1.54546|1.54357|0.92510 / 0.92769|
|新SUIM首例|256|26.50%|4|0.81001|1.19617|0.17752 / 0.51072|
|PASCAL `dev_s1/pascal_part/3/258` 非缺角色旧零H|144|0.45%|0|−0.08280|1.21518|0.000014 / 0.62673|

末行reference的FG/BG总质量都严格>0，不是缺角色回退；原补集只有全部FG质量的0.45%、没有高FG patch，旧分离度负使H按原协议清零，封存field的H逐点全0已核。逐删后同图全4096点的标定分离度转正，仅证明所提校准计算在这个已知故障输入上定义且与原公式一致。表中两个IoU的**评价patch集合不同**，不能相减称泛化提升；7例都是既有暴露参考图，不是query完整输出。也不能把本7例当成六域效果估计，更不能据此改变ridge方向、query场、组合或CRF。

此前[reference LOO身份报告](../../../../cv_data/a/cache_mechanism_study_20261010/REFERENCE_IDENTITY_REPORT.md)研究的是FG/BG角色weighted-LSE和max的self patch删除、角色质量重新归一、固定零阈值，未测试这个固定设计/旧权重的ridge解析逐点删除或4096点IoU校准。[05线历史readout审计](parallel_signal_20261010/05_reference_readouts/REPORT.md)还记载4空间组prototype jackknife及量化补集FG质量不足；同样不等于本ridge校准干预。故本数学核验没有把旧LOO结果重命名成新实验。精确7例数值、来源SHA和界限见[机器收据](../../evidence/local/sol_cache_experiments_20261011/reference_calibration_loo_math/receipt.json)；尚未生成6380候选、查询预测或任何GT评分。


[Astra校准6380协议冻结] 独立数学与7例真实参考核验完成后，已冻结 [唯一6380协议](../../evidence/local/insid3_foris_generalization_20261010/consensus_reference_calibration6380_protocol.json)，SHA `2ef877783c1c5c0c58f7b445da230ebd09bdba341e005094a4bcfde6b591abe0`。现有 `sol_experiments` 完成源绑定及原链哨兵后连续独占生产、完整seal、独立评分；输出 `../cv_data/a/consensus_reference_calibration6380_v1_20261011/`。两种标定各R/C/N/K双帧，八个cohort独立报，最多2个CPU worker各1thread，0新encoder/query raw/CRF；真实参考拟合重建、因子分解和精确复用分别计数。当前这是生产授权及冻结规格，实际开始以PID和提交记录为准。原主Sol与其他线程不重复启动。


[Astra：逐点删除的事前反例] 完整参考只有一个纯FG特征原子、所有BG原子具有相同另一特征时，原ridge仍可得到正确的FG/BG方向；删去唯一FG损失项后，剩下只有目标−1的BG，带自由截距的最优解就是常数−1，故该FG的删除预测为−1。其余BG的删除模型仍含FG，其预测一般大于−1；全参考逐删分离度可能因此为负而清零H，即使原方向对同特征的query本可完全区分。这个解析反例说明本实验的留一项检验的是单点依赖，不是“真实参考包含任务信息”的充分必要条件。全参考直接标定控制必须保留；若其收益更好，不能为了留一叙事舍弃更简单合法的控制，也不能将新的零H都归为语义不可迁移。此反例不增加门控、留出比例或新参数，冻结6380协议不变。


[Sol参考校准6380实际生产启动] 原始参考拟合/校准逐行核验完成：6380行、5581个唯一reference配置、700个历史未存reference_score明确由已存参考O24重建；所有有存档数组逐位相同，旧calibration最大数值差0。fit阶段耗时209.8s；唯一缓存entry中的1个来自本轮先行row0哨兵，非旧封存免费复用；1个缺角色记录无ridge拟合，实际非退化unique ridge拟合5580次（包含该哨兵），其余精确重复row复用799次。source_bindings/fit_index/fit_receipt与生产代码已封存；唯一producer PID24674、两worker24676/24677在CPU各1thread执行，产物 ../cv_data/a/consensus_reference_calibration6380_v1_20261011；尚未seal，不读取新query GT。

[Sol参考校准6380掩码封存] 生产已完成：6380×2校准×4臂×2frame=102080张完整掩码，6380个prediction/record文件，生产149.9s；seal SHA `191be888b5dfd48b4518f21db409e5bd67aaf6d61952d73e7c980f4195f5c818`，query GT guard尝试[]、0 query raw/encoder/CRF。现在才打开本轮query GT进行逐像素重建与固定槽位评分。

## [Sol 只读资产核对] 真实九窗覆盖、组件误报与可直接绑定的旧字段

本节只核读[01真实多观察报告](parallel_signal_20261010/01_multi_observation/REPORT.md)、[04数据集差异报告](parallel_signal_20261010/04_dataset_differences/REPORT.md)及对应旧`sealed.json`/`inference.jsonl`/少量NPZ键名；没有生成候选、读取九窗raw payload、运行CRF或评分。真正的Q局部视图是把canonical1024图在`x,y∈{0,256,512}`的九个`512×512`方窗分别作为原生图送入同一1024 DINO预处理；每个canonical像素仅有1、2或4个覆盖窗，融合是`2×支持票数>覆盖数`，平票BG，拒绝的组件不减少覆盖分母。LVIS首批还保存了从**整图**O24插值得到的`window.interpolated`控制；它不是另一次真实局部编码，不能冒充原生九窗。

|既有cohort与seal|实际局部链和主原图结果|R视野与同输入边界|
|---|---|---|
|LVIS原600：[`fixedwindows200`](../../../../cv_data/a/lvis_fixedwindows200_20261009/sealed.json) + [`component_presence200`](../../../../cv_data/a/lvis_component_presence200_20261009/sealed.json) + [`replication200`](../../../../cv_data/a/lvis_component_replication200_20261009/sealed.json) + [`completion末200`](../../../../cv_data/a/lvis_component_completion600_20261009/sealed.json)，每份`ALL_PREDICTIONS_SEALED`|真实九窗各自完整FoRIS局部CRF，8连通组件query-context判定，再严格多数；区域原图52.005138、同批完整FoRIS47.779371（600观察槽）。|实际`reference_box`在84+74+78=**236/600**例另裁R；原官方LVIS row是整图R/Q。与whole原R的同输入Q九窗对照仅364例；其余236例同时改变参考视野，不可把全部差归于Q九窗或融合。|
|Deep原100 [`component100`](../../../../cv_data/a/deepglobe_component100_20261009/sealed.json) 与快100 [`single_final_crf100`](../../../../cv_data/a/deepglobe_single_final_crf100_20261009/sealed.json)，各`ALL_PREDICTIONS_SEALED`|原版真实九窗各自局部CRF+组件+多数37.023712；快版同九窗O24从实际pre-CRF mask重新分组件、无局部CRF，`fast.no_crf`原图37.344028。快版另测**一次最终CRF**但该臂仅32.987793，不能把37.344称作一次最终CRF收益。|100/100相同episode的R/Q RGB、crop、R-mask、九个raw key、window_boxes及reference_box逐项相同；两版完整mask是同输入配对，但局部mask拓扑、组件决定及CRF位置一并改变，非单一局部CRF因果对照。Deep R/Q均1024、无额外R crop。|
|PACO快600 [`fast9`](../../../../cv_data/a/paco_fast9_600_20261009/sealed.json)，`ALL_PREDICTIONS_SEALED`|真实九窗、**无局部CRF**，组件balanced whole-query伪FG/BG margin判定，再严格多数；原图区域34.971403、同批完整FoRIS38.460868，未筛选多数29.538809（固定303官方槽，264观察）。|官方row已按父实例bbox裁R/Q；快版另有`reference_box` **13/600**例作小mask的2×R局部裁。与原whole-R直接同输入的区域例为587；PACO同600的原慢版九窗+局部CRF没有封存结果，不能把快版负结果扩展到慢版。|

九窗缓存的**索引/存在性**已沿各`inference.jsonl:raw_keys[0..8]`与上述固定profile `paco_mean200_20261008/raw_cache/0a555915.../profile.json`核对，没有全盘raw扫描或payload SHA重算：LVIS三批各200×9=1800个键/条目，Deep原/快各100×9=900且两版按episode九键完全相同，PACO600有5400次窗口键引用、5220个唯一键。三组去重后共**11520**个真实Q九窗key，全部对应非缺失`entry.json`和payload路径，metadata的profile ID、文件名、`O/24`形状`[4096,1024]`及FP32均合；这些entry的branch集合**仅`O/24`**。这证明当前映射与文件存在，不是本次重新校验11,520个大payload内的张量SHA。Q/K/V、其它层或未实际编码的新视野均没有对应九窗entry，不能用整图O24插值伪装为真实view。

小字段同样有清晰边界：LVIS/Deep每例`window_masks/*.npz`保存九个packed 512²局部mask，`vote_fields/*.npz`保存canonical1024的`coverage`和局部投票；对应组件`inference`保存每窗8连通组件数/面积、`context_margin`及`selections`。PACO每例`window_masks`有`local/0..8`九个packed mask，`predictions`有`coverage`和完整FoRIS/快区域双帧mask，`inference.regions`有每组件面积、margin、`accepted`；它没有单独`vote_fields`目录，但[01线已封存重建](../../evidence/local/parallel_signal_20261010/01_multi_observation/reconstructed_seal.json)按episode映射保存`raw_votes`、`accepted_votes`、guide和原图/CLI基线（1400 source-arm结果含Deep两版）。上述每组实际window-mask/prediction文件和九组组件margin/决定metadata均逐episode存在。**逐窗连续FoRIS source score或逐窗ridge H没有保存在这些小产物中**；组件margin是区域级query-context量，不能当像素级source score。

PACO严格多数抹除真支持可直接在现有`coverage/raw_votes/accepted_votes`定位：`accepted_votes>0 且 2×accepted_votes≤coverage`是**合法组件已接受但票数不足**的canonical支持范围，保留它可限定后来同输入控制的改动区域，不必凭GT挑像素。01线已用封存GT仅作诊断：PACO快600中全体GT的21.27%有原局部支持却不过多数、18.92%有**接受**支持却不过多数；4覆盖且原局部至少一票的真目标中34.09%只有1–2票。同时4覆盖且四个接受视图一致仍有6,128,630个FP像素，不能把“少数支持”一律恢复或把四票当正确身份证明。已有PACO `0/14`是擦除真目标例，`0/0`是重复干扰反例；组件margin在接受少数区的宏AUC仅.6462，不能用已接受状态代替身份判别。

现有全图/四分视野ridge字段可以**同例同帧作控制性绑定**，不能冒称九窗自身score：Deep100及PACO首100的`evidence/local/difficult_region_signal_20261010/raw/fields`保存`source_apd.ridge.global/local4`；PACO余500的`paco_remaining500/fields`保存`global/local4`，其逐例校准在旧records；LVIS匹配600的`consensus_lvis1400_dual_frame_v1_20261011/fields`保存全图`ridge_global/evidence_global`及校准，**没有local4九窗ridge字段**。三组九窗共1300个不同episode ID在`consensus_supported_foreground2380_v2_20261011/manifest.json`均按R/Q RGB、crop、R-mask身份精确匹配；Deep100全部R视野也相同，PACO587与LVIS364无额外R裁可直接做同R/Q字段/组件对齐，其余13/236必须标注不同R视野。可将已保存的连续whole/local4 H依既有bilinear规则分别投到canonical1024和原图帧，与组件接受少数支持/一致误报区域同例同帧比较，并保留FoRIS、未筛选多数、组件、多视图同K等原完整mask控制；这只是**可执行的事先控制关系**，不是本轮新方法、域名best选择或新分数。local4是四分视野旧场，九个真实512窗若需新连续得分只能从对应真实O24在另行授权后计算；绝不以whole插值冒充。

因此目前可直接定位PACO支持被多数抹除、并可把已有whole/local4负证据与真实九窗成员放到相同像素坐标；**缺失**的是PACO同600慢局部CRF完整结果、已保存的逐窗连续score/H、LVIS同600的local4 H、九窗Q/K/V或其它层，以及236/13例与whole完全同R视野的区域结果。任何结论必须分别写明原图/CLI、完整mask/区域诊断、R视野变化和暴露开发cohort；不生成插值“新视图”，也不把旧局部覆盖优势或快版失败替换成统一域切换规则。

[Sol参考校准6380完整评分完成] 当前唯一score PID25326已退出，八cohort双帧229680次重建/整数核验完成，七个真实reference删点哨兵重放通过。CLI固定槽位逐删N-旧N：Deep−.03784、COCO+.00355、PACO−.00076、LVIS+.44342、旧PASCAL40+.01795、旧SUIM40−.09026、原生PASCAL2000−.00090、原生SUIM2000−.11525 pp。LVIS对全参考自拟合N仅+.00725（query-photo CI跨0），同K−.00598；精确匹配LVIS600强控 query-context 33.71896 > 逐删N31.70176。原生SUIM2000逐删N低旧N .11525（query-photo CI[−.208,−.033]），连接R/Q-photo区间跨0；Deep/PACO各仍低既有region/task强控。完整报告和清理边界在 [Sol6380报告](../../evidence/local/sol_cache_experiments_20261011/reference_calibration_loo6380/REPORT.md)，机器report在 `../cv_data/a/consensus_reference_calibration6380_v1_20261011/report.json`，SHA `8e41b19fca80d03d572164d4bf27134e5481d5ea7920a4f89f431d85e09f6b5b`。fit/production/scoring分别209.8/149.9/640.8s；0新encoder/query raw/CRF。旧40与原生2000独立报告，不作跨队列归因；不满足Strong、不推广逐删公式，后续不扩队列。score依赖已释放；不在本会话删除原始DINO raw或仍由其他任务使用的源字段。

### [Sol 七百例只读符号四格] 九窗支持/误报与旧whole/local4参考方向

在前述资产核对后，只对**旧Deep原100与PACO快600**做一次CPU逐例计数；不读取刚完成的6380新分数、不重跑`M∪(B∩A)`或生成候选。沿各episode原封存calibration把已有`source_apd.ridge.global/local4`（PACO余500为`global/local4`）转成clip后的H128，再按旧bilinear规则投到**同一canonical CLI1024帧**；正号定义`H>0`，零归非正。`++,+−,−+,−−`依次表示whole/local4符号。local4是真四角旧query场，**不是九个真实512窗的逐窗连续分数**。逐例以R/Q RGB、crop、R-mask与旧field清单、重建ROI封存SHA绑定；700例均匹配。GT只用于已曝光数据的TP/FP诊断，绝不进入旧校准或新方法。详细[14KB机器计数](../../evidence/local/sol_cache_experiments_20261011/multi_observation_sign_cells700.json)对每个ROI×符号格保存`tp/fp/pixels`，并按旧`mechanisms.json`核对M、接受少数、少数且guide=FG、4覆盖4票FP的总数，全部**整数一致**。

固定ROI为①`M:2×accepted_votes>coverage`，另标`coverage=4,accepted_votes=4,GT=BG`；②`A\M:accepted_votes>0`而不过多数，按旧whole guide FG/BG分层；③旧guide FG而`accepted_votes=0`。这都是旧窗/组件/guide的成员关系，没有从符号格生成新mask。下表给两组核心ROI的**总TP/FP**及每格TP/FP（格内像素数=两数之和，显式数见JSON）：

|组/ROI|总TP/FP|++|+−|−+|−−|
|---|---:|---:|---:|---:|---:|
|Deep M|2,415,571 / 2,016,376|1,295,723 / 539,458|263,603 / 187,258|258,282 / 117,030|597,963 / 1,172,630|
|Deep 4覆盖4票FP|0 / 307,677|0 / 108,949|0 / 31,110|0 / 18,741|0 / 148,877|
|Deep A\M|467,882 / 1,927,930|83,169 / 143,030|34,830 / 116,440|74,569 / 78,395|275,314 / 1,590,065|
|PACO M|62,317,217 / 41,022,467|21,175,152 / 6,751,989|9,610,290 / 2,725,273|1,475,639 / 799,408|30,056,136 / 30,745,797|
|PACO 4覆盖4票FP|0 / 6,128,630|0 / 1,428,242|0 / 633,150|0 / 123,561|0 / 3,943,677|
|PACO A\M|21,873,324 / 36,977,239|1,940,598 / 1,320,756|2,658,522 / 898,891|367,557 / 300,820|16,906,647 / 34,456,772|

PACO已接受少数且whole guide=FG的总数为15,680,429 TP / 14,267,316 FP；其中`−−`仍有**11,350,066 TP / 12,446,038 FP**。因此“两个参考场都非正就删少数支持”会丢掉大量已有真覆盖；反过来M内`−−`仍有30,056,136 TP，四票FP中`++`也有1,428,242，正号或一致票亦非可靠身份证明。Deep同样有混合格。`guide=FG`而零accepted支持的ROI、`A\M`分guide FG/BG的全部四格与逐格pixels均在JSON；它们只说明旧方向与旧错误共现，不提供部署阈值或因果收益。

PACO快版有13/600例另裁R：JSON同时列全600、**同原whole R的587**及另裁13的同一定义计数。587子群的PACO `A\M`仍为21,847,776 TP / 36,522,613 FP，四票一致FP为6,063,007；上表全600的方向不依赖把这13例误写成同R输入。此统计沿旧reference calibration，所有H仅在对应已保存字段上计算符号，不新保存dense数组；无DINO/raw读取、新query场、CRF或预测mask。旧字段的符号有诊断价值，但当前四格同时包含真支持和一致误报，不能据同批表选域best、补票规则或取代真实九窗完整控制。


## [主Sol独立接手] 6380已完整验收，继续缓存内真实九窗主线

6380的原评分229680检查已结束；主Sol另对封存整数和248个双帧固定槽位点完成1,845,711项归档核验，误差≤2.14e−14，没有重复预测/GT/拟合/编码。逐删校准未稳定改善八cohort，匹配LVIS600仍低九窗强控、原生SUIM退步，未接纳。完整结果、成本、资产及清理证据见[6380主Sol验收报告](../../evidence/local/sol_validation_20261010/REFERENCE_CALIBRATION6380_RESULTS.md)。

Astra已停，主Sol独立推进，不再等其规格；用户明确授权三个Sol子代理并行做信号诊断、统一完整组合设计、唯一缓存执行验证。实际原R/Q/crop/Mask与真实九窗原生缓存绑定已由执行线完成，缺缓存0；候选尚未冻结，暂未生成新候选。后续仅已有真实缓存、CPU逐例释放，不新增DINO/下载/队列/dense场，不能以whole插值冒充九窗。主goalactive，批次完成不终止研究。

已审计后清理明确失败R1-G200的200个派生fields，观测释放1.165GiB；448份保留资产SHA不变，原始特征、数据/权重、清单/配置/必要代码、最终mask和I/U均保留。依赖/打开文件重合0，详细收据同报告。state已更新为真实三个Sol当前阶段，旧章节的PID/等待Astra仅作历史记录。


[主Sol当前进程与因果覆盖] 已存whole场1400 source-arm诊断完成，107000检查、383.57秒；正号门控M会损失LVIS/Deep快/PACO已有TP46.11%/39.96%/50.30%，未冻结为候选。当前signals唯一真实CPU进程39719执行原固定whole-R判据作用真实九窗Q的L_A诊断，不新增DINO/候选或dense保存，具体进程存活需每轮再查。主Sol无GT核对1300条输入，三组同R/Q RGB、crop、尺寸且异R-mask的组都为0；PACO570照片对应580种实际Q输入、595种实际R/Q输入。九窗观察/融合结果不能单独声称Mask-only因果机制，既有三域60对控制追踪继续分列。收据 `evidence/local/sol_validation_20261010/nine_window_mask_switch_coverage.json`；诊断机器规格 `evidence/local/nine_window_composition_20261011/protocol.json`，候选仍null。


## [主Sol全量证据] 真实九窗同参考头1300诊断及配对区间完成

1300真实R-head复原、11655次真实窗读出已结束（五个LVIS无正校准参考不作无效窗计算），完整whole ridge误差≤5.11e−15、旧cal误差0；86875计算检查及独立静态来源/标量验证通过。成功墙钟1862.58秒、CPU4520.92秒，峰值RSS1.586GiB，14255次既有payload加载213.99GiB；0DINO/CRF/候选mask/dense保存。失败首例的额外工作单列在信号报告，不计为已知wall/CPU0。主goal仍active，尚未冻结新完整候选。

主Sol只读封存标量做2000次配对photo/RQ-connected区间，耗时0.476秒，不重复GT/raw/fit/prediction。accepted-minority中native L_A−whole H宏AUC：LVIS+.025730，RQ区间[+.011818,+.038851]；Deep+.114402，[+.089341,+.138674]；PACO−.068046，[−.081575,−.053934]。query-photo同方向；PACO570查询照片、232 RQ组件、最大43例，未按episode假装独立。

L_A正号若替换M会损TP39.29%/30.23%/57.00%，同时四票一致FP仍保留1191816/197286/1980719。PACO固定0/14恢复207831TP/1350FP，是实际互补；固定0/0拒绝干扰却只保97/23513原M TP，不能用单例推广。新Q观察确有不同有效排序，但其跨域方向不统一；后续保留whole与native的互补，评价一条统一完整组合及matched控制，不要求单模块独立胜FoRIS，不用域名选场或旧zero点替换所有M。

全部源、成本、反例和独立验证见[信号报告](NINE_WINDOW_SIGNAL_20261011.md)，机器完整结果 `evidence/local/nine_window_signal_20261011/native_fixed_head/report.json`；主Sol[配对区间收据](../../evidence/local/sol_validation_20261010/nine_window_paired_auc_intervals.json)及[入口](../../scripts/sol_nine_window_auc_intervals.py)。固定R/Q仅Mask切换的因果边界仍是既有三域60对，这1300普通benchmark输入没有异Mask同R/Q组。


## [主Sol接续] 唯一完整组合已冻结并真实运行

信号三阶段与主Sol区间已齐，最后B-only1300双帧也完成；正式protocol SHA `c620555f9b5ee3d30b3547ba3f5c6300db94d6dd8b919e1e183bbca6c844f83c`，[完整规格](NINE_WINDOW_COMPOSITION_20261011.md)。不再增加诊断前提，执行线唯一producer PID48206/exec59993已真实运行，主Sol核对前100例通过且queryGT访问0，候选尚未评分。

有效源的完整F为 `(M&B)|(B&~A&R0)|(A&(M xor B)&((W>0)|(L_A>0)))`；源校准无效时F=`M|(B&~A)`，三臂相同。保留whole/native任一正证据，不以PACO负AUC选择domain分支。两个控制为相同骨架whole-only证据，以及未裁whole-ridge排序的canonical同增删数量控制；B-only固定添加另计，原图预算不声称相同。明确共享FP会保留、部分M独有TP/whole-onlyTP会被删除，完整输出裁定而非单模块胜负。

全部1300×3臂×2帧=7800完整mask先seal后GT评分；原图必须F整体bilinear/strict>.5导出，旧R0原图不拼接冒充该结果。冻结源码/config/活动/最终mask和记录在 `evidence/local/nine_window_execution_20261011/`；主state及heartbeat据实际PID接续，当前先跟完整seal，再用已有scorer/independent verifier/photo区间收尾。0新增DINO/CRF/队列/dense场，真实新CPU/cache加载成本单计。不宣称Strong或原创目标达到。


[主Sol直接人类新增授权] 当前1300三臂健康生产不打断；核读monitor源原人类请求后，将Pro共同场原200固定六臂缓存验证交同一唯一execution代理排后顺序执行。数学/输入/六臂/常数保持，只必要几何适配与去除dense归档，不新增DINO/下载/新样本。详情 [SCENE_RECONSTRUCTION_20261011.md](SCENE_RECONSTRUCTION_20261011.md)。目前是已交付且已排后，未核实开始，不能称已运行或取得自然成绩。


## [主Sol最新人类方向与真实收尾] 九窗方法关闭，继续五域机制

1300三臂全部7800完整mask先seal后GT，score/直接核验/原R0整数控制/新臂敏感度均已完成；旧producer48206已退出。主Sol纯整数逐class与编辑归因93258项检查通过，误差≤1.31e-14，没有重读GT/raw或重跑bootstrap。原图LVIS49.748627低于原九窗52.005138，Deep36.426043低于原九窗37.344028；PACO38.348501低于FoRIS38.460868和原R0 40.774944。按最新直接人类指令关闭九窗方法，不扩批/变体，原资产保留。[1300结果、成本、口径与资产](../../evidence/local/sol_validation_20261010/NINE_WINDOW1300_CLOSURE.md)。

研究目标未完成，goal保持active。signals转向COCO200/PACO600/LVIS1400/原生PASCAL2000/SUIM2000五域6200的逐例得失及有效阶段；design依据这些正负反例、三域60对固定R/Q换Mask和既有容量/迁移证据推导一条统一低成本方案，保留FoRIS有效处理，不按域选规则。execution唯一顺序准备已授权Pro原200固定六臂机制验证，主Sol尚未核实启动，不宣称自然图成绩。Pro不扩展/改常数/新增前向或dense归档。state与十分钟heartbeat按该方向更新；批次完成不结束研究。


## [主SolPro200真实结果] 特征拟合改善与完整任务正确性分离

原200六臂1200原图mask已完整先封存后评分并direct verification；主臂Deep11.499408/PACO87槽29.657600，低于同批FoRIS29.377542/45.563915、MEAN及快九窗。Deep相对FoRIS补真1890234而补假27458704；PACO补真66889而补假689029，目标覆盖改善没有保住背景判别。global_repeat本身已弱，不据此单独认定物理配准无效或DINO没有任务信息。固定机制验证结束，未采用，不换head/rank/threshold或扩展。

主Sol读取已封存metadata：200/200真配对feature fit均优于shuffle、raw粗守恒≤1.55e-15、FG/BG合法非零，但最终任务仍失败。真实/打乱alpha强度不同，不能把优势完全归身份绑定。完整结果、成本、两次纯接口失败及unknown成本、原六臂/控制/IU/源码/最终mask与新增标量入口见[Pro200报告](../../evidence/local/sol_validation_20261010/PRO_SCENE200_RESULTS.md)。当前五域6200整数/身份/官方class贡献和10正负代表例已落盘，annotation/有效阶段机制由signals接续、统一推导由design接续；研究目标保持active。
