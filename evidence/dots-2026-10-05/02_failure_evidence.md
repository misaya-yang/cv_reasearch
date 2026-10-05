# 失败证据与仍然成立的推断边界

更新至 2026-10-05 07:50 UTC。仓库版本固定为 `0e1dea19716faec8cc4c623ce54c79f8a9df3173`。本文将失败定位到实际测试的信号、目标函数、选择规则、优化过程或输出作用域，不把负结果写成冻结表示的上限定理。重点是防止换名字重跑同一构造，也防止由于一个小样本区间跨零而过早否定尚未分辨的效果。

仓库结果以[固定版本 README](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)为主；云端独有结果在本包 `evidence/01_02/`。后者的[来源清单](evidence/01_02/manifest.json)保留原报告相对路径、SHA256 和摘录时间。表中样本、分辨率、对照不同时，数字不能横向排成统一榜单。

## 1 四层信息必须分清

1. **RGB 输入中的线索**：图像可能含有形状、纹理、对象部件关系与场景语义。输入有某种线索，不代表当前可用的 token 已保留它。
2. **预训练表示与内部计算**：最终层 q/r 只是模型输出的一种表示。没有测过的中间层、注意力、value、跨图内部交互，不能被当成已用信息，也不能由于最后层读出失败就宣布全都无用。
3. **算法实际使用的关系**：均值、近邻、核分布、图边、候选范围、源域分类等，都只利用一部分允许输入。改变这些运算可以重新利用已有信息，但不能据此宣称后处理新增了跨姿态、跨外观的识别能力。
4. **oracle 额外注入的信息**：真实 query mask 可用来挑阈值、挑候选、挑参考权重、清洗端点、指定覆盖率。这些标签或选择权是额外输入。较高 oracle 只说明带该特权的构造能输出好 mask，不证明 reference/query 本身能辨识它，也不证明所需语义规律必能从当前表示读出。

因此，“候选库里已有好 mask，所以只差一个选择器”“GT 可把一块区域修好，所以这块区域必有可提取语义证据”“当前方法失败，所以表示缺少信息”，三种推断均不成立。需要分别测候选存在性、允许输入的辨别能力和最终选择结果。

## 2 先修正比较关系 再读失败

### 2.1 弱宿主上的组件收益不能转移给强宿主

旧十例实验中，局部 patch→全局注意力读出约 42.78 对 42.68，增量 +0.10 [−0.37, 0.58]，而同批 FoRIS 为 65.00。它没有建立对强宿主的价值。旧实现的成对/分开编码差异也曾使 INSID3 逐例一致率只有 43%；这时绝对分数和“超过论文”的叙述均不可靠。[README 的七机制小实验、图池路线及 HANDOFF 缓存说明](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

### 2.2 pre-CRF 增益不是超过 native 的增益

DEV241 固定 matte 相对 pre-CRF 为 +0.8819，但相对 native 仅 +0.3691 [−0.1111, 1.0722]。旧120 RCG 相对 pre 为 +2.5673，相对 native 为 +1.7054。这里对照升级改变了研究判断；不是从正值变成负值，而是还未达到用户要求的稳定至少 2 点。[仓库 matte 记录](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)，[旧120固定比较](evidence/01_02/prior120_locked.json)。

### 2.3 数据名称与独立性

云端旧120由 old20/new40/new60 组成，已经暴露。统计文件名中的 `pooled100_full_work` 是旧 new40+new60 的共同臂，不是此次标签封存的新增100。新增100仅预测冻结，尚未评分。[云端家族定义](evidence/01_02/cloud_families.json)，[新增100冻结](evidence/01_02/new100_freeze.json)。

## 3 监督读出有效 但其无标注可迁移性没有成立

| 构造 | 实际观察 | 失败限制在哪里 |
|---|---|---|
| 基类标签拟合的 convctx:layers | CONFIRM600 63.33，对 FoRIS +3.54 [1.96,4.96]，原分辨率完整流程 | 这是正结果；限制是需要基类标注，不能归入零训练 |
| 同结构 score-only | patch级 +2.27 [1.59,3.20] | 容量与空间分数处理本身有贡献；不能把其他输入的全部收益归给新的参考条件推断 |
| 826对自造 same/paste 样本训练 D1 | DEV241 patch级，卷积 −7.40/−27.59/−21.18，线性约 +0.30/+1.00/+1.01，区间跨零 | 这一构造的同实例对应与目标占比不能直接迁移到真实同类别跨实例 episode；不是所有无监督学习的否定 |
| COCO读出直接迁移到PASCAL-Part | 299例 +0.97 [−1.23,2.42] | 当前样本未建立稳定迁移，不据区间跨零断言真实效果严格为零 |
| 直接迁移到PACO-Part | 299例 +0.08 [−1.08,1.45] | 同上，不能由输入“class-free”推出跨域普适 |

以上为[仓库已记录结果](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)，本文未重新训练。最重要的变化是：D1/T2 已有结果，不能继续写成“只等GPU就能验证”的未运行承诺。

## 4 面积阈值与身份排序不是同一个问题

仓库 patch 级诊断中，FoRIS 56.996，用真实目标占比切同一分数场可到 66.085；删去所有假阳性 patch 可到 76.969。真实占比带来 +9.089，但四种无标签面积估计的误差仍约 0.62–0.66 个 log 面积。全流程切分族中，最好的构造约 +1.53 [−1.29,2.99]，而往返式水平为 −12.66。[HANDOFF 诊断表](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/HANDOFF.md)，[README失败账](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

这些数允许说“固定决策水平存在可测损失”，不允许说“真实占比可以从分数分布唯一恢复”，也不允许将 +9.089 当成无标签方法可获得的预算。

HANDOFF 的 Bayes 叙述需要保留两个数学边界：其逐像素判决对应给定代价下的 0–1 风险，不直接等价于 class-mIoU 最优；真实前景先验 π 与最终预测 mask 的面积也不能直接写成同一对象。此处理论不严谨不抹去实测 oracle，但不能据公式断言学习头已学到正确的面积先验。

## 5 Query-own seed 纯度与对象范围

最新 DEV241 中，FoRIS最高1%种子的纯度≥0.9共有149例。在这些相对干净的例子上，query-own seed prototype 的最佳切分为0.816，FoRIS最佳切分为0.815，几乎没有新的排序空间；FoRIS实际IoU约0.732。纯度0.5–0.9的58例及低于0.5的34例表现不同，不能把149例的结论外推到所有错误身份。[README query-native replay 与 seed contamination 小节](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

自洽集合迭代出现 −10到−28；把整张query当负例的相关PU方向也没有建立目标增益。持续收紧种子或依靠更高分核心，不会自动排除一个整体被误认的相似对象。这个障碍是被选证据的身份错误，而非“token数量太少”的同义词：在34例污染种子中，没有一例少于21个真实目标token，且最高分token有29例是背景。这只解释该组实际错误，不证明这些正确token的标签能在测试时被识别。

源域验证也不能替代范围：在最高分token为背景的43例上，source FG/BG rank从候选中选中目标多数区域20次，对照10次；但全241为184对189，未建立总体增量。将这类候选接上source-IoU选择的cut后，完整输出39.05对58.56，−19.51 [−21.54,−14.64]；184个选中目标的病例中，96例范围超过真值1.5倍。跨折gate均选择不切换宿主。[同一README，`results/query_hypothesis_mask_v1/report.json`记录；本文转述，未重跑该241流程](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

## 6 参考分类器与协方差构造

### 6.1 Reference ridge 与 QDA

仓库 DEV241 的reference-only ridge为49.72，对pre-CRF58.56，−8.84 [−12.38,−6.44]，四折均负。它修对约584万像素，同时改坏约1071万像素。该数否定固定参考拟合与零点切分的安全转移，不能推出一切线性判别无效。[README reference-only discriminant](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

云端旧120的固定class-specific QDA为10.797，共享协方差为30.169，Euclidean均值判别49.454，FoRISpre61.156。QDA有75例全背景、6例全前景。参考重代入AUC几乎1，query全图AUC约0.885，并不等于其零点决策有效。它同时包含两个问题：相对Euclidean排序未建立优势，绝对密度阈值又严重失配。不能只凭source拟合很好就说可迁移；也不能只凭固定cut很差就说所有query排序都没有信息。[本包QDA实测](evidence/01_02/qda120.json)。

QDA与共享协方差的差还包括线性方向和常数项变化，不能将某个差值单独归因给二次项。单位特征上的各向同性二次项本来就是常数。

### 6.2 RePRI的事实纠正与这次适配结果

HANDOFF 把 RePRI描述为必须元训练、整批query，这是不准确的。原方法在冻结特征上对单query执行测试时分类器优化；mini-batch可并行多个独立任务。官方目标含support交叉熵、query熵和query预测比例的KL项。[原论文](https://arxiv.org/abs/2012.06166)，[官方固定提交classifier.py](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation/blob/aa54b19b696623ff8a072a19c7392d43f4c914f7/src/classifier.py)。

本次DINO缓存适配使用参考coverage≥0.5近似源标签，固定官方温度20、SGD学习率0.025、49次更新和第9次后的prior刷新，不使用query GT prior。它不是原论文ResNet结果的逐位复现，也不是闭式读出。

旧new40中INIT21.753、support-CE37.465、RePRI44.167，FoRISpre58.059。query转导项相对CE有+6.702，但完整输出仍明显落后。初始化的预测面积失配是一个真实因素；然而旧60只把初始面积匹配宿主后，初始读出56.968，继续同一RePRI优化降至46.464，对pre60.174为−13.710 [−20.127,−7.952]。因此“全局PU曾失败”不能代替RePRI测试，而完成这次测试后也不能把问题仅归为初始化。[云端重新汇总，`new40_RePRI_added`与`pooled60_RePRI_host_area`](evidence/01_02/cloud_families.json)。

## 7 局部结构 图平滑与关系残差

| 构造与范围 | 实测证据 | 不能越过的解释边界 |
|---|---|---|
| source局部D4模板，旧60完整输出 |40.060对native-pre60.174，−20.114 [−25.488,−15.062]；中心/池化/置换控制均需保留|局部模板匹配在这组跨实例任务未带来有效标签转移，不能说空间结构普遍无用|
| reference约束与query32-NN联合harmonic，仓库DEV241 |23.17对58.56，−35.39；相同参考密度无query图为35.17，同图FoRIS为31.08|具体图权重使query平滑压过参考约束，reference/query边质量比中位仅0.0001612；不是所有联合推断的否定|
| source边相容性J，真实旧20 |边conditional AUC约0.683，对endpoint-shuffle约0.682；纯anchor AUC约0.528，confidence约0.660，简单query四邻外观一致性约0.7311|未建立独立邻接增量；把它包装成身份选择器会跨任务归因|
| column hubness k10，旧120实际pre-FG区域 |host AUC0.714746，hub0.679306，原NN0.664402；hub胜NN但输host|不同区域AUC不能择优拼成新方法；条件化后的小NN增量尚未变成有效mask修正|
| exact leave-one-out残差，旧120的116可评例；RCG实际前景TP/FP像素质量、按原score分10箱 |AUC0.602180，对简单(Wz)/degree−u为0.602287，差−0.000106 [−0.000586,0.000300]|相对原残差的微小正值被简单同图控制覆盖；没有完整mask收益|

D4与完整harmonic分别见[云端家族结果](evidence/01_02/cloud_families.json)、[仓库最新联合图记录](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。J与hubness的来源及运行范围在[本包失败账](evidence/01_02/running_idea_ledger.csv)；J的纯anchor对照还见[真实20报告](evidence/01_02/J_trimap20.json)。LOO来自[同信息degree控制](evidence/01_02/loo_degree_control.json)，这是主结果后的机制核查，原区间使用default_rng(0)，没有在本文冒充RandomState版本。这些AUC是各自已定义条件/区域上的诊断，不能与全图AUC或最终mIoU混比。

## 8 边界混合比的成功部分与失败部分

仓库的query-local mixture在自己的固定两token未知带、近六邻居端点、alpha与score取均值下获得正的pre-CRF信号。云端早期的±16像素未知带、最近单一F/B端点、非负两端元cone比例，是另一个明确构造；旧20为66.260对66.603，差−0.343 [−0.840,0.166]。它不能称为仓库+0.88的逐位复现。[云端old20_own_trimap_mixture](evidence/01_02/cloud_families.json)。

同一云端配方在旧120为60.772，对pre61.156，差−0.384；只保语义方向的版本61.206，差+0.051。旧new40+new60共100例中，Fisher端点度量相对Euclidean均值控制为−0.341 [−1.113,1.288]。这些小效果不支持把协方差或风险区间包装成新的边界方法，但也不由跨零区间推出alpha信号严格不存在。[同报告的pooled120_common_work与pooled100_full_work](evidence/01_02/cloud_families.json)。

特别需要保留三点：

- GT trimap固定了带外前景/背景身份，90.14与score88.49的差不只是“该把未知带取多宽”。
- 固定端点对只保存最佳alpha点的有限候选集，可能无论放宽多少残差都不能包含0/1；完整连续残差profile则可以。早期有限集出现无穷容差，不能写成所有区间校准理论上不可能。
- source或query伪纯anchor上的误差校准没有天然的跨图覆盖保证。一个连贯错类区域可以在空间留出重建时极其稳定，依然属于错误身份。

最新仓库在同unknown集内分别做端点纯化和真实覆盖率替换，已将这两个问题分开；它们是用query标签得到的诊断，不赋予实际读出额外识别能力。[README fixed-unknown-set attribution](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

## 9 完整候选库 不能把可达性写成可识别性

同一新60（仍属于旧120）完整eligible-seed候选库的class-IoU oracle为69.248，原默认61.746，source_contrast_mean63.444。这个oracle逐类优化“每例选一个候选”的聚合IoU；它与[逐episode最佳候选再汇总](evidence/01_02/hypothesis_vs_rcg.json)的68.763不是同一量。候选排序的unique-mask质量一致性约0.8545，mean-dot约0.8341，差的区间跨零。[完整bank诊断](evidence/01_02/hypothesis_capacity.json)，[同候选排序诊断](evidence/01_02/hypothesis_pair_order.json)。

这只证明既定候选中存在部分好mask，不证明允许输入能识别它。两个已暴露难例被修复，也不能替代其余样本的证据：去掉这两例后，source_contrast_mean相对默认在118例为−0.597，58例子集为+0.145。[敏感性报告](evidence/01_02/hypothesis_sensitivity.json)。

另一个Ward64区域库在新60上的任意union oracle为79.429，但单region oracle仅38.935，覆盖90%目标需中位5块。这个差距说明输出需要组合多个区域；并不说明存在可从reference读出的正确组合规则。同bank的INSID3完整聚合实际55.977、mean44.367，但各自GT-bestprefix为70.083和69.801：实际11.61分差混合了阈值/范围与排序影响，不能全归因语义身份能力。[区域容量诊断](evidence/01_02/ward_capacity.json)。

## 10 全token RBF MMD真实负结果

这次测试保留了候选内部的query×query核项。若将FG和BG的两个MMD相减，该项会抵消，退化为逐token核相似度平均；本次没有做这种抵消。

固定公式为：候选H与参考FG之间的经验RBF-MMD²，使用所有token与对角项；带宽只取参考FG两两平方距离中位数。候选、renderer、阈值均保持不变。先两个真实无query-GT smoke，再冻结旧120的全部960候选分数，最后查原有I/U。单线程57.5秒，0fallback，未读新增100。

| 同一固定候选库 | 完整mask class-mIoU |
|---|---:|
| 主RBF-MMD |52.7444|
| 线性核MMD |57.6072|
| 只保跨图核相似的KDE均值 |61.1027|
| 同候选FG均值dot |60.9723|
| FoRIS native |62.0176|
| RCG |63.7230|

主相对线性MMD为−4.8629 [−7.4367,−0.9947]，四折负，6升23降91平；相对KDE为−8.3584 [−12.2324,−3.9497]。相对native为−9.2732 [−13.6227,−5.7138]。[MMD主报告](evidence/01_02/mmd120.json)，[native阶段比较](evidence/01_02/mmd_native.json)。

相对线性MMD实际换了29个mask，28个更大，中位token面积×1.752；相对KDE换70个，67个更大。自核项以更低的内部相似性补偿更低的跨图相似性。相对KDE时这一方向部分由argmax代数决定，不能另算独立身份信号；实际扩大范围和IoU损失才是错误证据。原始1/|H|对角项只占自核变化绝对量的约7%或10%，不能把失败简单归咎于显式对角大小偏置。[选择动作诊断](evidence/01_02/mmd_selection.json)。

失败只限制这个固定带宽、整个经验分布匹配的目标。姿态、遮挡和可见部件比例改变会使正确对象的分布离参考更远；具有相同token多重集合、不同空间排列的两个对象，任何bag分布距离都无法区分。这些是构造的预先局限，不是对所有预训练表示或所有高阶关系的上限定理。

## 11 任意参考方向的高oracle也不是身份信号

为避免把MMD失败简单归为“只要挑对参考部件就能解决”，另做了旧20的受限表达能力诊断。它不运行新编码器，不生成新候选：保存每个参考token对完整候选query均值的投影；冻结单token能选中的候选，以及线性规划认证的参考token非负凸组合可选候选；最后才用query标签评价这些候选的最佳mask。

| 允许的特权选择族 | class-mIoU oracle |
|---|---:|
| 原完整候选库 |74.693|
| 任意reference-FG单token，含均值基线 |73.792|
| 任意reference-FG凸组合 |74.513|
| 等数量reference-BG单token，含其均值 |73.826|
| 等数量reference-BG凸组合 |74.035|

实际FG均值为64.526，实际BG均值仅20.194。BG方向的高oracle不意味着BG均值是好方法，恰好说明：给query标签任意挑方向的自由度很大，几乎饱和该候选库，却缺乏前景身份特异性。[诊断定义](evidence/01_02/anchor_oracle20_plan.json)，[真实旧20结果](evidence/01_02/anchor_oracle20.json)。

因此，这个诊断没有产生一个可部署方法，也没有证明“表示中已包含可读出的正确语义”。它只显示：该受限线性方向族具有较大的输出表达自由度；正确权重如何从允许输入确定仍未建立。没有据此扩大为全120权重搜索或制造新的训练配方。

## 12 SAM3资源列的失败不能挪到DINO列

最新SAM3的自动名字与视觉路由远强于较早61.09基线。对它有效的下一步必须超过当前路由与同信息简单选择；不能在旧弱exemplar上获得增量后宣称解决了强路线。

仓库中，候选DINO过滤/聚类约−1.73到+0.22；反向检查AUC0.709低于正向分数0.776；候选级两路相互印证相对路由为−4.72 [−7.58,−1.76]。参考前8–12个名字都有机会包含正确名，但查询标量信号在fresh915上最好的路由变化仍约−0.61 [−2.56,1.12]。这些结果将问题定位到“现有候选与标量分数如何选择”，不证明所有实例token关系都没有信息；尚未取得的token信号不能写成已可用机制。[最新README的SAM3路线小节](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

真实类别名本身就是额外信息；自动词表清洗的改善也须与新的推断分开。只有删除错误候选的算子还不能承担增加遗漏目标的责任。即便删除法能够改善某些假阳性，也不能声称兑现了两个完整宿主的全部互补oracle。

## 13 从这些证据得到的继续条件

继续研究时，应先写清楚实际错误动作：选错整对象、正确对象范围过大/过小、漏一个部件，还是跨token边界的像素覆盖错误。然后问允许输入中哪个量对这类错误具有增量；同信息简单控制必须在同候选、同区域和同renderer上比较。

- 若只有GT oracle变高，记为候选/动作存在性，不建设一个声称可用的选择器。
- 若AUC提高但最终mask不提高，检查作用域、阈值和错误质量覆盖，不能把AUC当完整方法效果。
- 若源域重建很好而query失败，检查跨图迁移与校准，不把source fit称为身份识别证明。
- 若实际增量小且区间宽，保留未决与其效果大小；不按“显著/不显著”二分研究方向。
- 若在相同强对照上稳定负，记录具体构造和失效环节，不用改名称、扫参数或增加模块掩盖相同失败。

目标仍是可复现的完整方法，相对native稳定至少2点，并胜同信息简单控制；不是取得更高的特权上界、更多消融或更复杂的后处理。新增100目前未评分，不存在可引用的新泛化结论。
