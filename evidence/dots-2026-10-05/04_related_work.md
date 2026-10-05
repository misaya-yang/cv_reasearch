# 相关工作与公平比较

本文只整理直接改变本项目方法归属与对照选择的工作。检索与核对截至 2026-10-05。优先使用原论文和作者代码；公开分数用于定位资源与协议，不与项目 DEV、CONFIRM 或工作分辨率结果拼成一张 SOTA 排名。

最重要的边界是：“没有新增训练”不等于“没有 mask 预训练模型”；冻结 DINO、DINO 加 SAM、SAM3 加文本/MLLM、以及基类标注训练读出，应分别比较。仅保留同资源列，再核 backbone、分辨率、shot、列表、输出分辨率与后处理。

## 一、直接相关工作与资源差别

| 工作及原始出处 | 论文内 COCO-20i 1-shot 数字 | 主要资源与协议 | 对当前研究的作用 |
|---|---:|---|---|
| [INSID3，§4/Table 1](https://arxiv.org/html/2603.28480v1#S4) | 57.6 | 冻结 DINOv3-L，1024 输入，无新分割训练、无 SAM；原始/去偏双特征空间，57.6 含上采样至原分辨率与 CRF | 完整对象聚合基线，不能用单均值零阈值替代 |
| [FoRIS，Table 1](https://arxiv.org/html/2609.03384v1#S4) | 60.9 | 冻结 DINOv3-L，无新 mask 训练、无 SAM；含 RGB/位置辅助和完整 refinement | 当前已核纯 DINO 列的强公开完整基线 |
| [FROST，§4.3/Table 3](https://arxiv.org/html/2606.31136v1#S4) | 48.4；同表 INSID3 为 56.5 | 冻结 DINOv3-L、1024、额外 support 翻转编码、RGB 传播；无新 mask 训练 | 非参数密度比完整对照；不能与 QDA 画等号 |
| [FSSDINO，§V/Table I](https://arxiv.org/html/2602.07550v1#S5) | 46.99 | 冻结 DINOv3-B，512，5 个类原型；每折随机 1000 episodes | 原型/Gram 和选择缺口诊断的直接先例；骨干与分辨率不同 |
| [REBASE，Tables S1/S2](https://arxiv.org/html/2607.09082v1#A1) | DINOv2-L + SAM-H：49.92；DINOv3-L + SAM-H：53.75 | 无新增训练，但使用 SAM 的 mask 预训练与解码器 | 参考背景子空间消除先例；不能列入无 SAM 资源列 |
| [REBASE，Table S4](https://arxiv.org/html/2607.09082v1#A1) | INSID3 管线内基线 57.35；替换 51.41；叠加 51.33 | 作者的纯 DINO 四折实验 | 直接提醒：背景子空间操作未必改善强原管线 |

以上是论文各自报告值，不保证同一 episode 清单。尤其 FROST 表内 INSID3=56.5 与 INSID3 原文 57.6 不可混用；FSSDINO=46.99 也不能替换成 FROST 转引/复测的 39.9。FROST 发布物没有足够信息重建精确 COCO episode/seed 清单，不能擅补“4×1000、seed 0”。REBASE 的 495 次加 5 次 warm-up 是计时实验，不能当作准确率评测样本数。

在本次核实的公开工作中，FoRIS 60.9 是最接近当前限制的强结果：单个冻结 DINO、COCO-20i 1-shot、无新增 mask 训练、无 SAM 或额外图像池。这是限定范围的“目前核实”，不是对全部新论文的穷尽性最高分声明。2026 年 7 月之后直接相关的已核工作是 REBASE 与 FoRIS；六月 FROST 和更早工作因机制直接相关而列入，不伪装成符合日期窗口的新工作。

## 二、INSID3 与 FoRIS：应忠实保留的完整推断

### INSID3：参考定位和查询内部聚合各司其职

INSID3 在去偏特征空间做跨图匹配，在原始特征空间做 query 聚类和 seed 相似度。它先产生细粒度 query clusters，结合参考前景原型和 query→reference 近邻定位候选，再选 seed，以跨图相似度、与 seed 的图内相似度及候选占比聚合。原型不是完整方法的终点。[论文 §3](https://arxiv.org/html/2603.28480v1#S3)、[作者实现](https://github.com/visinf/INSID3/blob/main/models/insid3.py)、[聚类代码](https://github.com/visinf/INSID3/blob/main/utils/clustering.py)。

代码核对要点：候选占比对 seed 强制为 1；聚合主阈值为 0.2；0.9 分位只在特定前向候选为空时作为 fallback。把它概括为“固定分位切割”会漏掉完整对象聚合。名称中的 backward correspondence 指 query 找 reference 近邻，并不自动等于互为最近邻或闭环重建。

本项目 cached INSID3 使用按 FoRIS 缓存给定预处理状态的一套特征空间（旧 120 中 119 例去偏、3_1_3 未去偏），工作在 1024 且没有官方完整双空间与原分辨率 CRF 等价保证。其旧 120 分数 62.189 是适配版强对照，不能当作论文 57.6 的忠实复现，也不能据绝对分数宣称超过原论文。它的用途是揭示：忽略完整聚合后，弱均值基线会把已有推断能力误当成新贡献。

### FoRIS：完整公开入口是主比较对象

FoRIS 的公开方法分为自适应去偏、前景定位、前景巩固，包含 query seed、聚类/稠密候选和空间 refinement。它改进 INSID3 的完整流程，而非单独更换一个距离。原文 Table 1 的 60.9 应保留其配置和最终流程。[论文](https://arxiv.org/html/2609.03384v1)、[固定代码版本](https://github.com/Xi-Mu-Yu/FoRIS/tree/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e)。

项目检查还发现，公开代码的一些标量门控在随后的 L2 归一化下可能相消，另有具体归一化维度和入口状态差别。这些事实用于准确复现与解释消融，不能把修复默认行为或绕开弱入口当作方法创新。需同时记录 native pre-CRF、最终 native，以及候选方法实际输出阶段。

## 三、FROST：完整的非参数密度管线，不只是白化

FROST 于 2026-06-30 首发，早于七月筛选窗口。它与当前研究直接相关，因为完整推断使用所有合法 FG/BG 锚点构造核密度比，而非仅一个均值或每类一个高斯。[论文](https://arxiv.org/html/2606.31136v1)、[作者代码固定版本](https://github.com/jhpark-ai/FROST/tree/b9ece69d7495a698c298e7cc3d16efacd4497a43)。

实际代码的主要步骤是：support 原图与真实水平翻转共同编码，黑图基的 rank-250 去偏，收缩的类内协方差白化；每类保留全部锚点做 log-mean-exp 核密度，带宽由 support 留一准则选取；随后做 query 特征/RGB/位置/初始分数共同约束的传播，以及原型和 top-3 前景投票的候选门控；连续分数上采样后切零，无 CRF。[model.py](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/model.py)、[density.py](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/density.py)。

QDA 每类拟合一个高斯；FROST 在共同白化后保留多锚点核混合。两者不等价。因此 source-QDA 的失败不能代替完整 FROST 的结果。不过仓库历史确有完整 FROST head 试验，只有 encoder 做既有 timm 适配，不能说“完整 FROST 从未测试”。出处为 [native_membership/causal_v3](https://github.com/misaya-yang/cv_reasearch/tree/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/results/native_membership_v1/causal_v3)。该旧 40 例结果只限制那次适配，不能替代新 120 或官方全协议。

当前 q/r/cov 缓存也不足以忠实重建其全部输入：已经去掉的子空间、原始特征、真实翻转重新编码和 RGB 传播不能由单份去偏缓存恢复。可做受限 KDE 对照，但应明确命名并给均值/QDA 同样输入。论文与代码在位置基图像、LOO 权重和候选“互惠”描述上也有差别，执行公式以固定版本代码为准。

## 四、RePRI 与 PANet：不能通过错误边界排除老方法

### RePRI：单 query 的转导与区域占比已有先例

RePRI 使用 support 标签交叉熵、query 后验熵，以及预测 FG/BG 总体比例的 KL 正则，在测试时优化线性分类器。论文明确针对一张 query，基础特征训练用普通交叉熵，不要求 episodic meta-learning。它还研究已知真实面积比例的 oracle。[论文 §3](https://arxiv.org/html/2012.06166v2#S3)、[官方 classifier.py](https://github.com/mboudiaf/RePRI-for-Few-Shot-Segmentation/blob/aa54b19b696623ff8a072a19c7392d43f4c914f7/src/classifier.py)。

因此本项目不能声称首先发现单 query 占比或“固定阈值可被转导推断改进”。真正差别应是冻结自监督特征、是否使用基类 mask 训练、具体目标与实际效果。本地 DINO 适配可以是强同输入控制，但其失败不否定原法或所有传导推断；它与原论文使用的训练特征协议不同。

### PANet：原型与逆向监督的一手来源

PANet 通过 support 掩码池化构造类原型，匹配 query；再由预测 query 掩码形成原型，反向分割 support，作为 prototype alignment 训练正则。它属于有分割训练的算法，不是当前缓存上无训练 source reverse-IoU 的同一个方法。[ICCV 2019 论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)、[正确 arXiv 条目 1908.06391](https://arxiv.org/abs/1908.06391)。

它要求我们准确归属“参考—查询—参考”的思想；同时，训练正则有效不意味着测试时 source 重建高就能验证 query 身份。对应标签可能被许多错误 query 假设共同重建，这仍需单独检验。

## 五、FSSDINO：Gram 与“选择缺口”的直接先例

FSSDINO 在冻结 DINOv3 上使用多个类原型与参考二阶 Gram 能量，按 mean×max 类分数读出；它比较了 GT 选层、support 自 IoU、reverse-IoU、Gram 一致性等选层规则。论文 §V 是 ViT-B、512、每折 1000 episodes；主 COCO 四折表与 Table IV 的 fold-0 选层分析应分开。[原文 §III–VI](https://arxiv.org/html/2602.07550v1)、[作者代码](https://github.com/hussni0997/fssdino)。

因此“均值以外的二阶关系”“逆向重建挑选”“oracle 很高而无标签选择困难”都不能仅凭命名成为当前项目原创。作者将其观察解释为 semantic selection gap；这是作者的诊断解释，不是“给定 reference/query 已经能够识别 GT 最优层”的证明。

本项目更应采用严格表述：GT oracle 额外读取 query 标签，只证明某层或候选存在较好输出；原始 RGB、完整 backbone 内部状态、末层缓存、实际统计和 query GT 是不同的信息集合。算法失败与候选 oracle 都不能独自判断完整模型是否“理解了语义”。若今后测试新的关系统计，必须先排除它退化为 Gram 二阶能量、核均值、均值差或既有 NN。

## 六、RCG 的组件先例：CSLS 与图正则应分别归属

### CSLS：局部密度修正不是本项目发明

Conneau 等在跨语言词嵌入检索中提出 CSLS：对余弦做双侧近邻平均惩罚，抑制 hubness。将这一度量用于 DINO 跨图匹配是应用选择，不能写成首次提出密度修正相似度。[Word Translation Without Parallel Data，ICLR 2018](https://arxiv.org/abs/1710.04087)。

典型形式是 $2\cos(q,r)-r_Q(q)-r_R(r)$。部分项对某些固定 query 的 argmax 不影响排序；但经过全图排名、FG 最大值或后续融合后，等价关系需按实际代码推导。名称相同也不保证效果来自同一因果环节。RCG 的强均值控制改变了 guide 系数，能检验简单替代，却不能单独隔离 CSLS。

### 图平滑与置信加权二次解：成熟优化结构

图上的一致性正则、种子/软标签传播有长期先例，如 Zhou 等的 [Learning with Local and Global Consistency](https://papers.nips.cc/paper_files/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html)。对每点目标与置信做二次保真，再加边缘保持的 Laplacian 平滑，也是 [The Fast Bilateral Solver](https://arxiv.org/abs/1511.03296) 的基本结构；后者使用的图构造和求解压缩与 RCG 不同。

RCG 的全位置软 unary、正 A 与连续解，不等同于早先纯种子 PPR 再选 conductance cut 的构造。因此旧种子传播失败不逻辑上排除 RCG；反过来，RCG 的凸性、唯一解、最大值原理和 CG 也不能作为新理论本身。新意要落到参考指定的关系证据，以及它在完整任务中不可由更简单 guide 替代的增量。

## 七、SAM3、概念引导与文本资源必须另列

[Few-Shot Semantic Segmentation Meets SAM3](https://arxiv.org/html/2604.05433v1) 使用 SAM3 的视觉提示，文本版本又增加概念信息。该 v1 正文与表格数字不完全一致：无文本正文 66.1，Table 2/3/5 为 66.6；联合文本正文 75.4，Table 2/5 为 75.8。若引用，应明确表格/版本，不能把其中任一数当作无歧义统一结果。Table 6 又明确区分真类名 75.8、fold 候选词 72.4、dataset 候选词 67.8，三者的信息不同。SAM3 本身已有 mask/概念预训练，不能因推断时无训练就列为纯自监督 DINO。

[CG-ICS：Toward Robust In-Context Segmentation via Concept Guidance](https://arxiv.org/pdf/2606.28149v1#page=4) 已研究由参考指定概念、结合 query 存在性进行概念搜索，再驱动 SAM3 的路线；它加入 MLLM 与语言能力。项目自命名或视觉/命名路由必须面对这个机制先例。真类名则是另一个特权输入，不能作为参考单独识别目标的合法结果。

仓库最新标准 4000 列表的 SAM3 自命名/视觉路由 73.14，与更早缓存子阶段的 DEV 或 CONFIRM 数字有版本差异。它们说明额外基础模型/语言资源可以改变任务表现，但不构成纯 DINO 后处理增益，更不能与论文不同清单分数作直接显著性判断。

## 八、下一份方法稿应该怎样使用这些文献

相关工作应围绕具体缺口，而非把名字排列成目录：INSID3/FoRIS 给出完整对象获取与聚合；FROST 给出全锚点非参数判别；RePRI 给出单 query 转导与占比；PANet 给出原型及反向监督；FSSDINO 给出二阶读出与选择诊断；CSLS/图正则给出 RCG 的成熟组件；SAM3/REBASE/CG-ICS 定义额外预训练和语言资源的边界。

当前可提出、尚未证明的缺口是：这些具体固定统计在相似类别、遮挡和外观变化下，仍可能错选完整对象。下一条方法需找到实际可计算且有独立证据的参考条件判据，并在同信息最简控制及完整 native 基线上验证。它不能仅靠高 oracle、增加一个图目标、改换统计名称或扩大模型资源获得原创性的结论。
