# FoRIS 低编码开销研究方向

2026年10月12日。当前要求是冻结 DINOv3，使用一张参考图和完整参考 Mask、一张查询图，在完整分割结果上明显超过 FoRIS，同时减少额外编码。五域共同领先及最终十域目标保留。

**按现有证据，最值得投入的主线是任务相关的按需局部观察：先用整图特征完成判断，只在需要时增加至多一个查询区域编码，参考区域特征提前缓存。** 已有 Z2 提供了真实增益与低观察量的依据；需要研究的是如何选中有用区域、如何接受局部纠正，才能把它推广成统一方法。若要求完全零额外编码，优先检验匹配的一致性和已有连续参考证据的读出；目前没有证据证明其中任何现成规则能在五域大幅共同领先。

同一次前向中分开利用语义和结构也是值得保留的零新增视图方向。它的风险低于不断增加固定图块，但仍属于待验证假设。两条路线都不能凭 oracle 阈值、局部 AUC 或缓存吞吐宣称 Strong 成功。

| 优先级 | 方向 | 相对整图基线的新增编码 | 当前依据和判断 |
|---|---|---|---|
| 先做因果对照 | 检验参考与查询匹配的归一化轴差异 | 0 | 发布代码存在明确差异；需要完整输出实验判断影响。工程修正本身不等于原创贡献。 |
| 主研究方向 | 根据任务证据决定是否观察一个局部区域，并保留可靠的局部纠正 | 查询每次0或1；合法参考裁剪可预缓存 | 已有低观察量 Z2 在 LVIS1400 上有完整增益；区域选择与跨域融合仍未解决。 |
| 零新增视图备选 | 一次前向保留末层语义和一个中间层的结构证据，分别读出 | 线上完整前向次数不变 | 原论文细结构结果支持中间层有信号；简单选层、混层和参考拟合已经有负证据，不能原样重复。 |
| 缓存上可先检验的读出问题 | 保留参考 FG/BG 连续排序，解决其符号和面积失配 | 0 | 同1000任务中密度场在 FoRIS 前景内的排序更好，但直接零阈值完整分割明显更差；存在局部信号，尚无可靠兑现规则。 |
| 仅在改变训练条件时考虑 | 冻结主干，训练轻量任务条件读出 | 推理时可仍为一次主干前向 | 有相关研究和旧监督头正结果，但改变当前免数据集训练条件，不作为当前主线的既成解法。 |

对延迟的判断需要同时看图像数量、分辨率和后处理。FoRIS 原论文的 RTX3090/1024 条件下，编码为702.27ms，完整推理为1620.77ms；因此无新增编码并不意味着额外匹配、聚类和读出免费。这些论文时延不能作为本机 M4 的时延。[FoRIS 附录D](https://arxiv.org/html/2609.03384v1#A4)

已有最接近这个计算目标的完整正结果如下。各行只在自己的输入、队列和指标内比较。

| 已有实验 | 完整 FoRIS | 对照或方法 | 同批变化 | 能支持什么 |
|---|---:|---:|---:|---|
| COCO4000，CLI1024 | 60.931741 | MEAN 62.512972 | +1.581231点 | 不增加图像编码的图与排序读出确有收益；不证明跨域迁移。 |
| LVIS1400，原图 | 46.105282 | MEAN 46.056276 | −0.049006点 | 同类无新增编码读出没有保留 COCO 优势。 |
| LVIS1400，原图 | 46.105282 | Z1 47.043175 | +0.937893点 | 仅在触发时增加查询裁剪已有有限完整收益。 |
| LVIS1400，原图 | 46.105282 | Z2 47.994001 | +1.888719点 | 查询裁剪和参考裁剪联合有增益；不是五域确认。 |
| PACO600，原图 | 38.460868 | global ridge 40.774944 | +2.314076点 | 参考背景竞争有局部正作用；同规则在 COCO200 对 FoRIS 为−0.899078点。 |

[图读出与强控制审查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/03_graph_readout/REPORT.md)、[Z2 合并1400原始评分](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_zoom800_20261009/score/cumulative1400.json)、[PACO及迁移总结](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/HISTORY_FAILURE_REVIEW_20261011.md)、[COCO ridge 完整负结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/02_task_identity/COCO_RIDGE_REPORT.md)。所有这些结果都来自已曝光开发池，不能称独立确认。

这次重新读取 Z2 的600和800份逐任务收据，共1400份：参考裁剪请求510次，查询裁剪请求220次，平均额外图像请求为730/1400=**0.521429**，查询触发比例为**15.7143%**。Z1 为220/1400=0.157143。它们是逻辑请求量，实际复用后的唯一编码数更少，不能当作冷运行实测时间。

若参考图固定且裁剪已缓存，旧策略每张新查询最多增加一次裁剪编码。按这批触发比例，查询图像请求量相对整图为1.157143倍。若每任务都从全新参考开始，不计任何复用，Z2 为2.521429张图像请求，相对原2张为1.260714倍。当前四图块新编码方案在冷输入下需要两张全图加八张局部图，即10张。**这是编码图像数量的比较，不是26%时延、速度倍数或下一策略的触发率预测。** [本次独立计数和差值核算](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/low_encoding_direction_research_20261012/saved_evidence_audit.json)

按需观察有收益依据，但旧选择器存在两个明显限制。Z2 从原 FoRIS 二值结果的包围框决定查询裁剪；完全漏掉的目标或把父物体整片当成目标时，这个框未必有用。查询裁剪后的输出还会清零框外区域，因而裁剪既改变特征也改变输出范围。新的区域选择应基于合法参考指定的任务冲突，例如前景相似度与背景竞争矛盾、局部提议与全图归属矛盾；仅对全图熵或面积设置触发阈值缺少现成收益证据。[冻结选择器及完整输出实现](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_zoom600_20261009/run_zoom.py:40)

旧五域各40的开发面板中，Z2对FoRIS在四域点差为正，COCO为−0.108点。该面板不是官方评测或独立确认，部分旧导出输入缺原照片身份。因此Z2是研究低观察量的起点，不能直接当作已满足统一五域目标的方法。[五域旧面板](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/zoom_crossdataset40_20261009/score/REPORT.md)

同200任务的控制保持参考视图、查询框、裁剪RGB和完整FoRIS/CRF，只用整图特征插值代替真实查询裁剪特征：Z2=49.259607，插值I2=46.713528，差+2.546079点。这支持真实局部观察提供了插值无法保留的有效表示。它不能分离尺度、上下文和位置变化，也不能把 Z2 的全部改善归因于新信息。[同框插值控制](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_zoom_interpolation200_20261009/score/paired_results.json)

所以主线应把研究问题定为：**如何用已有任务证据选择少量值得观察的区域，并允许这些观察纠正错误的全图判断？** 不是预先承诺所有目标都能靠一个框恢复。已有九窗口分析显示局部既找回真目标，也会让空窗口产生前景；全图错误背景角色还会否决正确局部目标。新方法必须检查真正争议区的身份，而不能只把裁剪结果并回原Mask。旧三窗、共享特征替代和少数支持恢复都有完整负结果；本建议不重启那些冻结配方。[局部观察与融合完整审查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/01_multi_observation/REPORT.md)、[共享计算与端到端时延](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SHARED_FEATURES_AND_LOCAL_READOUT_20261009.md)

完全零额外编码的首个具体检查是候选匹配。发布源码中 `ref_feats[0:1,m]` 为 `[1,C,H,W]`，沿 `dim=2` 归一化；查询则沿 `dim=1`。前者按高度、后者按特征通道。于是参考某位置的相似度不仅由该位置的特征方向决定，还受同列其它行的每个通道影响。远程发布源码和本地固定源码均有这个差异；既有眼底审计也记录过，没有看到修正轴后的完整收益结果。[发布代码](https://github.com/Xi-Mu-Yu/FoRIS/blob/main/models/foris.py)、[本地候选匹配](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/models/foris.py:582)、[既有审计](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/FUNDUS_MECHANISM.md)

一个独立代数例子说明其影响：固定某位置的参考特征和查询特征均为 `(1,0)`；把同列另一行从 `(0,1)` 改为 `(100,1)`，高度归一化下该固定位置的点积由1变为0.0099995，而通道归一化的余弦仍为1。这个例子证明两种度量不同，不证明哪一种在真实图像上更好。应先只改这一处，在其余APD、聚类、阈值、CRF和输入均保持一致时比较完整输出。如果有收益，修正后的FoRIS应成为后续强控制，不能把修正数值等价性本身包装成原创分割方法。[独立例子与源码SHA](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/low_encoding_direction_research_20261012/saved_evidence_audit.json)

另一个缓存上已有的机会是连续参考证据的读出。同五域各200、合计1000任务中，完整参考 FG/BG 密度场在 FoRIS 已预测前景内部的AUC均值分别比FoRIS连续场高 **0.021695、0.030702、0.020973、0.028073、0.036726**，顺序为COCO、LVIS、PASCAL-Part、PACO-Part、SUIM。有效任务数分别为196、195、199、193、200；这是有限ROI内逐例平均的点差，不是官方mIoU或已确认显著的跨域增量。然而该密度场的零阈值完整分割五域都更差。这说明在这个有限争议区域里有排序信号，但零点和输出范围不可靠；它不是一个已经成功的方法，也不是AUC改善能自动兑换成mIoU的证明。[五域完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_SEMANTIC_HIERARCHY_FIVE_RESULTS_20261012.md)、[本次原JSON差值核对](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/low_encoding_direction_research_20261012/saved_evidence_audit.json)

这里更值得研究的是保留连续得分和合法Mask覆盖，不因层级节点统计丢掉排序，并在最终判断中区分“更像目标”和“应该接受为目标”。不能再直接设置密度比零阈值、沿用参考图最优阈值、估计一个query面积后强制补点，或用强S的九窗缓存作为低成本方法输入却漏报生成它的成本。参考自拟合和背景补齐都已有反例；更好的局部排序尚缺可迁移决策规则。

一次前向中的语义与结构分工是另一个值得验证的表示方向：保留O24进行跨图任务匹配，同时取一个固定中间层，仅检验它是否对已有强场的结构恢复提供增量。中间层仍是64×64网格，不能称免费得到了128×128空间分辨率。已有参考Mask只能约束任务归属，不能保证另一幅图有相同面积、形状或局部邻接模式。

官方DINOv3分割实现支持在一次骨干前向中取多个block的输出。一个额外4096×1024的FP32特征图为16MiB，保存R/Q各一份增加32MiB；主要骨干计算次数不变，但LayerNorm、传输和读出仍有成本。现有339GiB库主要为O24，不能倒推出中间层；缓存研究若缺这些量，仍需补一次原始输入前向，线上再实现一次前向共同提取。[官方多层接口](https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/__init__.py)

FoRIS自己的附录在1024输入下给出同一个中间层index19的1-shot结果：Fundus33.4对末层19.7，DeepGlobe16.7对末层13.3。这是该论文既有配对与方法条件下的结果，支持不同层有结构相关信号，不证明本文的新分工。其精确道路协议也不能用我们的自定义道路池替代。[FoRIS 附录F 表10和11](https://arxiv.org/html/2609.03384v1#A6.T10)

这个方向应与“中间层直接匹配”“两个层分数平均”“仅换query分区”和同总平滑强度控制比较。已有层级1000因丢失排序而失败；FSSDINO也报告参考引导和无监督选层不能兑现oracle优势。若中间层只改变范围、增加平滑，或收益被同预算简单操作解释，就不能宣称新增了任务识别。[FSSDINO原论文](https://arxiv.org/html/2602.07550v1)、[已完成上下文相容性及分布交换](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_CONTEXT_COMPATIBILITY120_20261012.md)

外部方法的资源条件需要逐项辨别。检索结果里的“训练免费”“冻结骨干”“零样本”并不自动等于当前条件。

| 方法或原则 | 可借鉴的部分 | 为什么不能直接作为当前低成本胜出证据 |
|---|---|---|
| [FROST](https://arxiv.org/html/2606.31136v1) | 全FG/BG点云、连续密度比和统计解释 | 本地已有同族密度及LOO失败；发布实现增加参考翻转、白化和图读出，one-shot也不是单视图。其COCO/LVIS相对INSID3的劣势已在既有源码审查记录。 |
| [REBASE](https://arxiv.org/html/2607.09082v1) | 参考背景子空间抑制相似干扰 | 完整结果使用SAM；不能将其整套成绩归为纯DINO读出收益。 |
| [RePRI](https://arxiv.org/abs/2012.06166) | 查询统计参与推断和前景比例的作用 | 使用基类监督训练的特征；熵最小化和比例约束也不能保证纠正错身份。旧面积估计和自训练已有负证据。 |
| [SAOT](https://aclanthology.org/2026.alvr-main.3.pdf) | 部分匹配和未匹配质量 | 使用学习的传输代价、解码器与自训练；不满足当前免数据集训练条件。平衡传输强行复制参考面积也不适合多实例任务。 |
| [DINO多对多匹配](https://arxiv.org/html/2604.23670) | 避免把强余弦最近邻当唯一正确对应 | 研究几何匹配并使用SuperPoint、几何约束和RANSAC；其几何一致性不能直接跨不同实例、姿态或物体部件迁移成分割规则。 |
| [ViT-Up](https://arxiv.org/html/2606.14024) | 从同一次前向的多层状态恢复密集特征 | 公开模型含训练、LoRA适配和额外权重；现有公开backbone配置不证明与本机冻结ViT-L完全匹配。 |
| [PixelUp](https://arxiv.org/html/2608.02792) | 高分辨率语义指导 | 增加ConvNeXt语义编码器和已训练上采样器；不能当成免费O24插值。 |
| [UPLiFT](https://arxiv.org/html/2601.17950) | 轻量局部注意力上采样 | 有学习的上采样器；其密集任务收益不是单参考FoRIS配对收益。 |
| [SegDINO](https://arxiv.org/html/2509.00833) | 冻结骨干搭配轻量读出 | 是监督分割读出，不能直接宣称保持当前单参考、免数据集训练的研究条件。 |

因此，白化、普通KDE、参考拟合阈值、query自训练、无任务图传播、平均原型的边界环相减、原样候选层级或平衡OT都不应因新的名字再次排队。最新PACO600边界原型审计中，邻近内外边界相减对同父物体其余部分的AUC差为−0.0276，对兄弟部件也为负；参考Mask边界附近的对比不是现成的新身份信号。[PACO边界完整审计](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/PACO_BOUNDARY_PROTO_DIAGNOSTIC_20261012.md)、[40路线及正负证据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/HISTORY_FAILURE_REVIEW_20261011.md)

后续验证应按能够改变决策的顺序进行，具体算法在读取新query结果前冻结。

1. **先隔离归一化轴影响。** 使用已绑定五域缓存和相同完整FoRIS，不补编码；原始发布FoRIS、仅轴修正FoRIS及MEAN保持同任务、同输出帧。记录候选变化、最终四编辑和class/fold I/U。修正无完整收益，就不把它当涨分路线继续扩展。
2. **主线先回答一个局部观察能解决多少真实错误。** 优先复用已有真实窗口和组件收据；只比较一个候选区域预算，包含原Z1/Z2规则和同图像数量的固定区域控制。选择时只读R/Mask、Q及未看标签的场；oracle区域收益仅作诊断，不回流选择器。窗口内部必须允许无目标，框外应保护已有正确覆盖；不能延用无局部提案也计入多数分母的九窗规则。
3. **再验证低成本接受或拒绝条件。** 检查新局部场与whole场冲突时的补真、补假、删真、删假，保留全部连续证据。若新条件只重建原FoRIS排序或被同编辑预算控制追平，就没有新的选择能力。该步骤不能直接重用已失败的全局ridge、强S伪标签或简单并集作为答案。
4. **对零新增视图方案，先验证中间层在真实争议区的增量，再完成分割。** 固定末层任务场，不先用GT选层或调混合系数；原图完整结果、COCO/LVIS/PASCAL-Part/PACO-Part/SUIM的同规则比较与细结构压力测试分开报告。普通层平均、普通平滑和soft-mask/KDE是必要控制。
5. **最终以完整输出及同机延迟决定采用。** 同一规则必须对完整FoRIS、修正轴FoRIS、MEAN及适用的已有强控制比较；曝光开发结果之后需要照片隔离的确认。报告每域差值和区间，按任务记录触发率、实际编码图像数、分辨率、参考预计算与新query时间、完整后处理及p50/p95。缓存字段读出速度不能代替冷输入成本；不以单域收益或按域选赢家兑现统一目标。

“明显超过”应落实为相对这些完整强控制的稳定、可解释增量；目前只能提出优先级，不能诚实承诺固定涨分幅度。按需观察是现有准确率与编码量证据最好的起点；一次前向的语义和结构分工提供另一个低观察成本的研究机会；仅修阈值和边界插值目前不足以承担大幅跨域突破的承诺。

本次研究使用Exa的8个检索角度筛查64条结果，按URL去重为50个候选页面；版本、镜像和重复论文不计为独立证据。方法判断只采用阅读全文或作者代码的原始来源，与本地已封存实验交叉核对。[检索来源及原文列表](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/low_encoding_direction_research_20261012/literature_sources.json)

交付范围是研究结论、优先级与验证设计。新增encoder、预测和query标签访问均为0；Claude源码、日志、缓存与进程保持只读。低成本新算法的实际跨域增益尚待实验，本次没有把方向研究完成写成Strong方法已经成功。
