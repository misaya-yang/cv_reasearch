# 05｜理论、可识别性与结论边界

**结论。** 仓库已有实测表明“固定表示上的判决规则有可改进空间”，并有基类监督读出在完整确认流程中的正结果；它尚未证明“查询占比可由当前证据无监督识别”“参考部件覆盖等同目标身份”或“某类固定图优化已抓住完整物体”。最需要修正的是把逐像素分类的贝叶斯规则直接用作IoU理论，以及把有限构造的失败写成表征上限。RCG的最大值原理成立，但其严格不可达条件只覆盖很少的实际深漏检。

本文的最新仓库来源截点为GitHub commit `0e1dea19716faec8cc4c623ce54c79f8a9df3173`；最新AGENTS/README/HANDOFF/STATUS全文见本包来源快照，`decision_heads.py`也按该commit单独核读。早期本地checkout `15786d6`仅用于历史结果副本，不混称最新HEAD。本文还纳入截至2026-10-05的云端真实结果。仓库内确认实验、工作目录中的暴露DEV实验、纯数学反例分别标注。未读取新增100例标签。旧120反复参与方法开发，不因重新封存、类留出或换一个统计口径而成为独立确认集。

## 0. 四层信息不能互相替代

| 信息层 | 实际提供什么 | 本轮能推出什么、不能推出什么 |
|---|---|---|
| RGB输入 | 图像像素，是原任务输入 | 多数CPU实验没有重新读取RGB；结果不能穷尽所有RGB关系 |
| 预训练表示与内部计算 | 缓存q/r只是所取层与编码方式的token；不是模型全部内部状态 | 冻结模型承载既有跨姿态/外观统计；这些实验没有重新训练或赋予模型新的类别识别能力 |
| 算法实际利用的关系 | 原型、cosine、图边、排名、候选均值或完整假设评分 | 只测该算子能否利用允许输入，不能由后处理变化断言模型新增识别能力 |
| 查询GT等额外标注 | oracle选阈值、纯seed、端点、部件/区域标签或最佳mask | 高oracle只证明指定动作空间含更好mask；不证明reference/query可选对，也不证明选择规律已在当前表示中可读出 |

反向推断也不成立：许多后处理失败，不证明预训练表示必缺信息；某一监督probe失败，也不证明其输入之外的全部表示不可识别。以下每条性质均应按所用信息层解释。

## 1. 应当先区分的四个问题

给定参考图及掩码(S,m)、查询Q和冻结表示φ，可以依次问：

1. **表示或观测够不够？** 所给q/r/cov及空间结构是否包含可用的目标身份信息；这些缓存不是整个预训练模型的内部计算。
2. **动作空间够不够？** 固定排序的阈值族、区域划分或完整候选库中，是否存在正确输出。
3. **合法判据能否挑出来？** 不使用查询GT的分数，能否在同一动作空间中优先正确答案。
4. **实际流程能否兑现？** 固定推断、渲染及CRF后，在相同资源、独立测试协议中是否超过强基线。

Oracle只回答第2问。一个手写指标失败，通常只限制第3问中的该指标；一个优化器的目标更低，不自动回答第4问。把这些问题合并，是此前多条路线反复绕回同一误差的原因。[仓库研究要求](evidence/repository_sources/AGENTS.snapshot.md)、[失败账本](evidence/repository_sources/demo9_README.md)。

## 2. HANDOFF中的判决理论需要哪些修订

### 2.1 似然比公式适用于什么

若目标/背景条件密度p_T(x)、p_B(x)已定义且适用于查询分布，类别先验π=P(Y=1)，等代价逐像素0–1风险的最优规则是

    p_T(x)/p_B(x) ≥ (1−π)/π。

这里π是生成分布中的真实先验或图像条件先验，不是“最优预测集合的大小除以N”。用预测集合大小反过来定义π，会变成尚需证明的自洽估计，不再是无条件贝叶斯结论。

而且，**最小化逐像素错误率不同于最大化IoU**。IoU将同一掩码中各像素的决策耦合，最优集合还依赖后验概率、集合大小与评价风险的定义。即便给出真实前景数量，选恰好相同数量的最高分像素，也不保证实现给定排序上的最优IoU。因此，GT面积切分是一个有用的特权干预，不是“恢复贝叶斯最优分割”的证明。Dice/IoU有专门的Bayes规则与校准理论，不能直接替换为分类阈值。[RankSEG原论文](https://jmlr.org/papers/volume24/22-0712/22-0712.pdf)。

还要区分“期望IoU”“期望I除以期望U”“每图IoU平均”及“每类先累加I/U再平均”。它们不是同一风险。代码实际报告的是最后一种；论文理论必须说明对应哪种量。

### 2.2 排序与切点可分，但不完全“正交”

固定分数s时，所有全局阈值形成嵌套掩码族。对s作严格单调变换不改变排序和该族，但会改变固定数值阈值选中的掩码。这足以解释“分数校准影响最终判决”。它并不意味着真实风险可唯一分解成互不影响的“排序误差+切点误差”，也不能证明一个卷积残差读出只修正切点。

仓库的DEV241、patch级GT面积干预，从56.996到66.085（+9.089）；误并删除oracle到76.969。这些数是实际动作空间诊断，应保留其GT条件，不写成可部署上界已被接近。[拟合报告](evidence/05_06/repo/demo_lists/demo9_transductive_ics/results/decision_v1/fit/report.json)、[理论原文](evidence/repository_sources/demo9_HANDOFF.md)。

### 2.3 “π不是分数自身的函数”应限于什么假设

如果只知道一个未知单调变换后的匹配排序，且未给定目标/背景类条件分布，目标占比通常不可由排序唯一识别。但若类条件分布已知、满足混合分布可识别条件，或额外结构能校准后验，未标注查询样本仍可能提供先验估计信息。因此不能把四种估计器失败扩张为“任何无标签占比推断都不可能”。

最新README中“FoRIS对参考的使用等价于一个均值向量”也应读成该缓存与消融设置下的经验结果，而非所有输入上的代数恒等。多原型、背景项和投票源码并未从公式上消失。

当前证据支持的表述是：**已测分数与手写规则没有提供稳定可迁移的面积判据；需要另外量到可用信息。** 这比一个未成立的普遍不可识别定理更准确。

### 2.4 RePRI的相关工作定位要纠正

HANDOFF称RePRI需要meta-training及“整批查询”，与原文不符。RePRI明确使用普通基类交叉熵训练而非episodic meta-learning，其转导推断针对given query image，利用该图的无标签像素与前景比例项。与本项目的真实区别是训练资源、主干/分割特征的监督来源和具体推断结构，而不是“RePRI不能单图推断”。[CVPR2021原文与摘要](https://openaccess.thecvf.com/content/CVPR2021/html/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.html)。

## 3. 监督读出：代码保证什么，实测又说明什么

`Head`的初值是8(s−.5)，shift为0，残差网络末层零初始化。因此在相同插值、阈值约定下，初始二值判决与宿主一致。这是**初始化性质**，不是训练后的无伤害保证。训练后scale、shift和逐位置残差都可改变，既能重切，也能改排序。[实现](evidence/05_06/decision_heads.py)。

`convctx`输入每张关系图及其图像均值、最大值、宿主前景内均值，并通过膨胀卷积产生局部修正。损失混合BCE和soft-IoU，早停留出类别，随后重拟合。代码没有仅输出一个全图阈值的限制。因此不应把它整体解释为“只估计π”。

CONFIRM600原分辨率、FoRIS refinement下，`convctx:layers`为63.325，对FoRIS59.783增益+3.542［+1.961，+4.957］，四折均正。这支持“基类标注拟合的读出能迁移到该协议的新类”。同一报告也有42例下降超过10点，直接排除了逐例不伤害说法。[确认报告](evidence/05_06/repo/demo_lists/demo9_transductive_ics/results/decision_v1/infer_1_confirm/report.json)。

四折线性系数余弦相近，是参数稳定性的观察。它不证明决策规律与类别无关，更不证明跨域到部件、水下或医疗图像仍成立。“类无关”最好明确为**输入/参数不显式索引类别、训练测试类分离**，将分布外泛化另列待验证性质。[系数报告](evidence/05_06/repo/demo_lists/demo9_transductive_ics/results/decision_v1/rule_probe.json)。

## 4. 零训练、无查询标签与无额外监督不是同义词

| 说法 | 所需证据 | 本项目中的边界 |
|---|---|---|
| 推断不读取query GT | 函数签名、调用路径、先冻结后评价 | 多个CPU候选已做到；不说明开发过程未用标签 |
| 无新增跨episode训练 | 不在额外episode上拟合共享参数 | 固定图/相似度规则可满足，但仍可能经DEV选参 |
| 无推断期权重拟合 | 单个测试episode也不优化classifier等权重 | 与前一项不同；episode内适配仍是参数拟合，须单独披露 |
| 无新增人工掩码监督 | 训练数据也不含额外人工mask | 基类监督读出不满足；按构造获得标签另需评估分布差距 |
| 参数无需验证集选择 | 配方在开发标签前固定 | RCG的alpha/lambda为开发选择，不能这么宣称 |
| 不使用mask预训练模型 | 资源中无SAM类分割先验 | 与“没有再训练SAM”不同 |
| 推断期只用一张参考 | 实际输入合同单参考 | 不排除主干已有大量预训练知识 |

最新README已报告D1/T2，不能沿用早期PLAN的待运行状态。用826个自造样本拟合，卷积读出在DEV241分别−7.40、−27.59、−21.18；线性为+.30、+1.00、+1.01，区间均跨零。COCO监督读出不重训迁移到PASCAL-Part299例仅+.97［−1.23,+2.42］，PACO-Part299例+.08［−1.08,+1.45］；另外三个数据包未完成，无读数。这支持“当前合成生成器未教会真实任务规则、当前COCO读出跨域收益弱”，不证明一切无标注训练或跨域读出不可能。最新README列出了服务器结果位置，本包未将未随包提供的原始输出冒称独立重跑。[最新README：D1/T2段](evidence/repository_sources/demo9_README.md)。

## 5. RCG的精确数学结构

锁定配方先构造

    u=s+0.5[R(g)−R(s)]，
    a_i∝0.1+|2s_i−1|，A=diag(a)>0，
    L=D−W，W=Wᵀ≥0，
    (A+16L)z=Au。

s是FoRIS的minmax分数，g来自纯参考FG的CSLS型guide；W是query mutual20图，经平均degree归一。u不是概率，可能超出[0,1]。目标为

    E(z)=0.5(z−u)ᵀA(z−u)+(16/2)zᵀLz。

`A+16L`正定，连续极小值唯一。逐节点看，它是具有不同restart强度的dense软证据传播：

    z_i = ρ_i u_i+(1−ρ_i)Σ_j P_ij z_j，
    ρ_i=a_i/(a_i+16d_i)，P_ij=W_ij/d_i。

上式P仅对d_i>0定义；孤立节点d_i=0时直接z_i=u_i，无需也不能除以degree。

所以它与PPR存在代数联系；但此前只用16个正负seed再扫conductance的实验，不等于RCG。输入证据密度、graph定义和最终判决均不同。不能仅因“也是图传播”把旧负结果移植过来。[RCG机制核对](evidence/05_06/rcg_diffusion_mechanism_comparison.md)、[锁定源码](evidence/05_06/strong_method_controls_20261005/validation121_locked/locked_methods/rcg_readout.py)。

### 5.1 最大值原理与守恒

对λ≥0，记K=(A+λL)⁻¹A。M-matrix性质给出K逐元素非负；L1=0给出K1=1。K只在图连通分量内混合，因此

    min_C u ≤ z_i ≤ max_C u，i∈C。

同时，对每个分量C有Σ_C a_i z_i=Σ_C a_i u_i。常值unary不变；λ趋于无穷时，分量趋于A加权unary均值。

若max_Cu≤.5，该分量的token对任意λ≥0都不能变成>.5前景。这是严格条件结论。某一错误区域只占大分量的一部分时，结论不能直接套用。图上连通不等于同类，也不等于影响权重足够。

### 5.2 像素插值不能省略

最后renderer是p=Bz，B非负、行和为1，再p>.5。一像素可混合来自不同W分量的相邻token。因此“全负分量token无法翻正”不等于其16×16 footprint每个像素都无法翻正。充分像素证书是其贡献token对应分量极值经B混合后仍≤.5。

对近似解z_hat，若r=(A+λL)z_hat−Au，则||z_hat−z||∞≤||r||∞/min_i a_i，可用作证书数值余量。这不是新预测门控。

### 5.3 实测预算排除了一个看似自然的解释

旧120的native深漏检定义为GT前景、native预测背景且距native前景>48工作像素。共有1,145,941像素：

- 73.8204%位于“固定rank修正规则下，让该位置guide排到最高也仍有u上界≤.5”的token footprint；这是任意guide排序的保守逐点上界，不是当前guide实际u≤.5的数量；
- 仅0.1074%位于实际u全负的图分量；
- 严格bilinear后不可达证书只覆盖56像素，即0.0049%；
- 99.8921%的深漏检所在分量，含至少一个u>.5且全格为GT前景的token。

因此“图完全没有正确正证据”不能解释绝大多数深漏检。正确支持的存在仍不保证足够影响权重或正确身份连接。所有含深漏检分量的A加权均值均≤.5，也不支持通过无限增大lambda获取整个目标。[真实预算与定义](evidence/05_06/rcg_unreachable_budget_result.md)、[机器结果](evidence/05_06/rcg_unreachable_budget/report.json)、[正确正witness核查](evidence/05_06/rcg_unreachable_budget/positive_witness_budget.json)。

## 6. 更精确的优化，不等于更正确的mask

将同一RCG目标限制z∈{0,1}，唯一无向边系数是8W_ij，节点代价为0.5a_i u_i²与0.5a_i(1−u_i)²，可用mincut优化。实际使用固定2²⁰整数量化；所有二值割浮点误差≤δ，则整数最优解距实数二值最优解≤2δ。

旧120中108例二值能量降低超过误差界，但mincut mIoU61.369低于原连续RCG63.723，差−2.354［−5.025，+.274］（RandomState补充），未见改进证据。先把原RCG token二值化再同样渲染的控制为62.932；mincut相对它也未见增益。仅这一渲染差异已损失−.791［−1.012，−.263］（同一RandomState补充）。[完整结果](evidence/05_06/rcg_binary_cut_result.md)、[原始统计](evidence/05_06/rcg_binary_cut120/report.json)。

同理，精确去掉每个节点自身fidelity项得到LOO_i=(z_i−h_i u_i)/(1−h_i)，h_i=a_i(M⁻¹)_ii。该公式要求h_i<1、去掉data项后的问题仍可识别。孤立点h_i=1，删去唯一data项后该分量矩阵奇异，LOO未定义；实际评价明确排除这些节点。它不删除该token对原graph或全图rank的影响，不能称独立身份观测。实际残差增量由简单Wz/d控制解释，new60精确LOO残差AUC.599921对简单邻域控制.600215，差−.000293［−.001086，+.000365］（RandomState补充）。[LOO结果](evidence/05_06/rcg_loo_result.md)。

本节统一采用[共同RandomState(0)统计补充](evidence/cloud_results/completed_0745_statistics.json)，2000次照片组重采样；原始default_rng区间仍保留在各实验原报告中，不倒改历史结果或择优选择区间。LOO是episode-macro条件AUC，binary是class-I/U指标，二者不混排。

## 7. 全局部件覆盖的数学正确性与语义漏洞

当前joint set目标由query→reference模块化项与两类facility coverage组成：FG在所选区域并集中取max，BG在补集中取max。max覆盖是次模函数，补集变换保持次模，故整体仍次模而通常非单调。

确定性double-greedy的端点界为3F(S)≥F(OPT)+F(∅)+F(V)。CPU目标范围[-2,2]，引用非负1/3近似率须使用F+2；不能借用随机版本1/2，更不能将目标近似率写成IoU保证。[短证明与穷举](evidence/05_06/joint_region_set_math.md)、[可执行检查](evidence/05_06/joint_region_set_mathcheck.py)。

这确实不是独立unary：相同参考part被另一region覆盖后，新增region的收益会下降。但它表达的是“替代性覆盖”，不是“这些parts属于同一物体”。合法单位向量反例中，真mask的F=1.75，加入一个背景region后F=1.88333成为全局最优；单纯forward/additive反而正确。精确优化也无法修复错误目标函数。[反例结果](evidence/05_06/joint_region_set_mathcheck.json)。

实测new60原joint为28.725，对同bank unary44.805；修复可见目标floor后45.129，也未接近同bank INSID3聚合55.977或完整缓存适配61.746。数学保证与正确语义之间的缺口有真实对应。[原结果](evidence/05_06/cpu_method/joint_partition120/report.json)、[目标floor控制](evidence/05_06/cpu_method/joint_partition120_floor/report.json)。

## 8. 单参考歧义与partial matching

一个source part可以在query缺失、被遮挡或外观改变。强迫每个source part都找到query对应，会奖励背景冒充。最小partial匹配可引入可见性z_j：

    max_z [z_j h_j(S)+(1−z_j)ν_j] = max(h_j(S),ν_j)。

固定null证据ν保留集合交互。source背景对该part的最大相似度提供一个无额外参数的候选null，但必须假设“可见真部件超过null，非目标不超过null”能跨图成立。现有输入没有证明这种分离。

反例中，同一组q/r/cov允许“部件缺失、b是背景”和“b是困难真部件”的两种标签假设；当前覆盖模型无法区分。这是对该观测构造及模型假设的反例，不证明实际DINO所有输出都歧义，也不证明单参考任务普遍不可解。

免费删掉所有匹配会让reverse项消失；把null直接设为query补集max，则max(selected,complement)=max(all-query)，同样变成常数。要获得可用partial matching，必须提出可检验的null证据，而不是把新的权重藏进“缺失”变量。[详细推导](evidence/05_06/partial_reference_match_identifiability.md)。

## 9. 16像素格点误差不是DINO丢失信息的证明

“二值16×16token读出限制”与“token表示无法包含格内边界”是不同命题。DINO-L/16输出1024维与一个16×16RGB格的768数值，不能通过简单维数比较推出信息已被压缩丢失；后续混合、归一化和可逆性另需论证。

给真实cell coverage后，几何重建可改善二值块边界；给估计coverage后，同一几何构造却变差。这定位了估计误差，而非完整表示上限。按area与normratio联合拟合特征，即使训练重建误差降低，也可能使真coverage误差增大；竖直边相同混合比例与未知端元范数比存在明确歧义。其理论只约束该混合模型与窗口，不应扩张成所有feature-to-boundary解码器失效。[几何结果](evidence/05_06/subcell_coverage.md)、[联合反演](evidence/05_06/joint_boundary_inference.md)。

### 最新DEV241证据对解释的约束

source验证在最高分token为背景的43例中，选到目标多数candidate从10例升到20例；但全241从189变成184，不能写成整体identity改善。进一步用source-IoU选择query cosine cut，未门控结果39.05对FoRIS58.56，所有按其他折选择的gate都选择不切换。候选纯度改善没有交付query范围。

边界端点GT过滤给固定matte+2.4617，而同一个unknown集合内直接给真实coverage为+18.4779，两者都额外使用GT。24例过滤后无纯FG端点，并不意味着格点上不存在纯FG：全部24例别处都有，22例在预测mask内部也有。故既不能把高oracle当可用读出信息，也不能把“没有找到好端点”写成“表示没有好端点”。最新README还明确matte相对native仅+.3691［−.1111,+1.0722］，不能拿preCRF正增益替代最终对照。[最新README相关段](evidence/repository_sources/demo9_README.md)。

## 10. 当前理论能支撑的研究主线

可以严谨主张：冻结特征上的推断动作、目标函数及渲染会分别影响结果；完整目标假设与合法选择证据必须分别测量；RCG的dense软证据平滑有清楚的数学范围，而现有最大值原理并未解释大部分实际深漏检。

不能据此主张：单参考必然不够、DINO没有目标身份、无标签切点永远不可识别、低图能量就是物体、部件覆盖就是同一对象、监督读出只是面积估计。论文的核心方法claim仍须由同资源强对照、同输入简单控制及独立完整流程验证决定。
