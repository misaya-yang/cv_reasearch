# 第4–6项只读并行审查（2026-10-06）

结论：三项都已有明确的缓存输入到全图输出路径，公式与核心实现基本一致；新候选真实例均为 **0**。第4项有确定的多实例误删风险及原图长宽比未恢复的几何限制；第5项现有正例不能证明参考形状优于通用方形；第6项有通过默认完整算法的单位特征存在性见证，证明完整协方差可提供迹、逐点选择性之外的信息，但不证明真实DINO分布、分割收益或原创性。建议保持三项为待真实检验候选，有限筛选优先级为第6项、第4项、第5项，不把控制行或参数组合计作新增方法。

范围：读取当前 `AGENTS.md`、`docs/research/PLAN.md:3-54`、三项算法及限定证据、必要共用入口和用户给定Pro报告。只做短CPU合成计算；没有SSH、服务器查询、GPU、付费资源、技能调用、大实验、历史失败账本扩查；只新增本报告，未修改算法、PLAN、STATUS或Pro文件。本文引用代码范围是现状证据；新增合成检查输出见下文。

## 4. reference_constellation

**完整定义与实现一致。** 全部参考纯前景聚成最多8模式，空间加权中心构成地标；余弦模式选择性取“本模式响应减最强竞争模式”，查询中位数/MAD标准化。每对地标及其最多8×8峰提出正/镜像相似变换；生成地标、留出支持数和平均值分别门控；反向转移完整参考coverage，IoU NMS后取最大先验，最后与clipped MEAN全局融合。证据：`src/ics/methods/reference_constellation.py:20-32,66-106,109-141,159-192`；`constellation_preparation_01a1100b/method.json:9-17`。共用入口对场做双线性1024读出（`src/ics/methods/prepared_cpu_bundle.py:107-113`），独立入口保存base、clipped-base、bag、prior控制（`scripts/run_reference_constellation.py:29-46`）。这是一条缓存到1024掩码路径，尚不是独立RGB编码全流程的实测成绩。

**优先风险：默认融合实际上把已接受姿态的先验当成全图必要条件。** 设权重=.5，若某token先验为0，则新场=.5×clip(base)≤.5，严格阈值后必为背景。只要任何一处姿态被接受，其他未匹配的基础真目标都可被删除，包括base=1的token；并非只削弱置信度。若先验=1，则任何正base都可被提升到前景，也会转移不匹配轮廓中的背景。证据：`reference_constellation.py:188-198`；`method.json:37`已有风险提示。本轮沿原4地标单位特征夹具，在远处加入16格点base=.9连通块，实际输出：retained_poses=1、该块保留0/16、added_tokens=4、deleted_tokens=16。这里的“原正确块”是假定语义的反例条件，绝非真实漏实例读数。

**几何迁移合同比真实类别几何窄。** 坐标直接取64×64网格，没有原图H/W输入（`reference_constellation.py:155-165`；共用入口`:109`）。两图各自非等比缩放到正方形后，原图中的同一个物理相似变换一般变成仿射变换；尤其长宽比不同时，即便原图目标无形变，网格也未必满足单一相似变换。本轮四地标由正方形变为x轴2倍矩形、各角色峰强度同为3：12个真实峰对姿态全部拒绝。第5项已明确恢复原图几何（`reference_shape.py:38-56`），第4项没有这个恢复；本报告不默改其方法合同。另外，全部参考前景共同聚类而非按参考实例建模板：多实例同模式的质心可能落在实例之间，转移的是整张参考掩码并集（`:161-165,188`）。

**合成价值与证书边界。** 现有同强度正确/扭曲布局检查（`check_geometry.py:17-46`；`geometry_check.json:4-10`）证明逐部位峰强度不充分、布局约束能区分特定构造；原始特征完整夹具仅4稀疏点（`check_geometry.py:55-66`）。没有完整真/干扰同图结果胜过bag控制的已有证据。留出地标没有参与该姿态的两点拟合，但同一批query响应被用来筛选成千上万依赖假设；不形成独立统计holdout，也不提供误报率证书。现有1024维时间0.0447秒来自0个通过姿态的场景（`geometry_check.json:130-145`），未覆盖大量warp/NMS代价。

**CPU复杂度。** 投影O(NKD)、参考Lloyd另有O(L N_R KD)；最多C(8,2)×8²×2=3584姿态。留出采样O(PK)，mask转移O(AN)，NMS最坏O(A²N)，无保留实例上限。`:116-140`逐候选整图warp并逐保留mask比较，所以P上限不能证明整队列分钟数。约3584²×4096级像素扫描的粗上界足以说明必须实测高通过率分支，不能从弃权计时外推。

**有限下一动作。** 当前只把“全局先验外误删”和“canonical长宽比限制”写入主审查。恢复真实运行权限后，固定原算法只做一次同producer输入的base/clipped-base/bag/constellation/prior完整比较，预先单列多实例与姿态未匹配部分delete-TP、pose/abstention计数及高通过率成本。若误删抵消身份收益，关闭该v1；局部融合、按实例模板或仿射恢复都改变当前合同，必须明确新版本，不能悄改后归入原v1成绩。

## 5. reference_shape

**实现和公式一致，几何恢复处理较完整。** 对原图坐标中每个矩形网格单元积分至四阶矩，三点张量Gauss对这些多项式精确；中心二/三/四阶量形成三个相似/反射不变量，实际四连通参考与query组件决定代价，树只提供提案。目标为 E=Σ(base−.5)M−Σ_C |C|d(C)，d=D/(1+D)。证据：`reference_shape.py:38-114,117-181`；`shape_preparation_01a1100b/method.json:8-25`。原图H/W缺失时核心弃权，共用入口推断前拒绝缺失（`reference_shape.py:195-199`；`scripts/run_cpu_feature_candidates.py:223-229`）。主输出nearest1024，base-nearest与generic-square使用相同读出，另存bilinear控制（`prepared_cpu_bundle.py:114-127`）。

**参考迁移未获独立价值证据。** 原正例参考是8×8方块：补回弱一列真目标、删除2×32条状干扰（`check_shape.py:26-47`）。因此参考库与generic-square控制本就等价。本轮重新调用默认完整predict：reference保留目标64、干扰0；generic也64、0，二者掩码逐点相等。现有全尺寸计时例两者都没有编辑（`shape_check.json:59-78`）。现有证据只支持形状/紧致性可改变该合成完整掩码；不能支持参考特定形状的新增价值。

**删除与恢复边界。** 一个全正一元组件默认会被整块删除，当d超过平均(base−.5)；例如全base=.9，只需d>.4（D>2/3），即使基础很自信仍会删除。类别姿态、遮挡、触碰多实例或参考粗网格碎片都能改变这三个矩量；两个不同对象也可能具有同样三量。query组件<8格点完全免罚（`:95-114`），碎片化可绕过形状约束。所有base≤.5时直接空掩码（`:203-206`）；弱部分只有与正证据及形状改善共同成块才可能恢复，不能从无正证据全图凭空发现实例。

**算法证书。** 每次对实际掩码重算能量，负能量完整组件删除和严格改善提案确实单调；精确积分/连续形状不变量不保证光栅、透视和类内形变不变。最多4轮、预选48提案、每轮只选一个增删/替换，甚至不保证终止时在该提案池中局部最优，更无全局掩码最优或IoU证书（`:145-181`）。既有删除单像素后形状债仍存在的检查，仅关闭“树分区逃罚”这个局部错误，不排除真正物理碎片逃罚（`check_shape.py:38-44`；`method.json:44-45`）。

**CPU复杂度。** 空间树4邻边内积O(ND)、排序O(NlogN)，moment/tree统计O(N×11)。每一掩码评估O(N×11)加组件×参考库的距离计算；每搜索最多约4×48×3次提案评估，候选和generic各运行一次，精确mask缓存减少重复但不改变上界。原参考库没有数量上限，8格点最小组件在4096网格可有数百模板；不能只写“少量gallery”作保证。现有0.0821秒是合成无编辑场景、含控制而未含I/O/base/评分（`shape_check.json:59-85`）。

**有限下一动作。** 先保留为较低优先级候选，明确现正例被generic完全追平；不扩展shape变体。恢复权限后固定一次参考法对same-nearest MEAN及generic的完整结果比较，单列delete-TP、多实例/遮挡/小组件编辑。若generic追平，采用简单generic结论，撤回参考迁移价值；若参考有增量但不能改善强完整法，只能称局部机制证据。

## 6. reference_covariance

**实现、修订与公式一致。** 模式响应a_i=q_i·c_k，总体S=E[aaᵀ]−E[a]E[a]ᵀ；每参考四连通组件仅以至少8纯前景样本成模板，范数≤1e−6弃权。D=||S−S_R||²/||S_R||²，d=D/(1+D)；组件E=Σu−3P d，P=Σmax(u,0)。证据：`reference_covariance.py:33-92,128-155`；`covariance_preparation_01a1100b/method.json:9-17,19-23`。已保留初始面积惩罚仅补全56/64的反例（`area_cost_counterexample.json:4-7,35-36`），当前正证据预算修订未隐藏。共享实际组件搜索与第5项相同，完整nearest输出及trace/margin同读出控制齐全（`prepared_cpu_bundle.py:128-142`）。

**新信息确有存在性证据，但范围有限。** 第一对照例也被逐点margin解释（`covariance_check.json:107-109`），不能证明矩阵必要。补充单位特征见证的实际cluster学习8模式，真/干扰均值、模式占用、trace及逐点排序余弦相同；默认完整算法保留64指定目标、删掉64指定干扰，trace与margin各保留两区（`reachable_witness.json:9-40`；`check_reachable_witness.py:73-108`）。本轮在禁止文件写入的patch下重跑现有main，数值仍为full_true=.1379310143、full_false=.4387894161、trace_true≈trace_false=.1379310143；源SHA与记录一致（`reachable_witness.json:62`）。它证明当前算法可使用额外矩阵信息；“单位向量可实现”不等于“真实DINO图像可实现”、发生频率或跨实例标签一致。uniform transport额外代价相等只检查该夹具的费用，不是完整occupancy解码器对照（`check_reachable_witness.py:96-103`）。

**风险与证书。** 全正一元组件E=P(1−3d)，默认保留正收益条件d<1/3，即D<.5（`method.json:27`）。零收益不自动删除，严格负能量才删除；小分量或局部提案还能改变其命运。一个很自信但参考协方差不匹配的真实实例可整块删掉；类内视角/遮挡/模式比例、纹理与参考组件接触使S漂移。响应协方差同时对行置换、共同加性响应偏移不变，但这不是整个归一化DINO特征或预测器的环境不变性；组件内任意排列可同S，错误类别/背景混合也能同S。query<8格点免罚，48提案/4轮及全图无正证据空输出的限制仍在。参考S接近范数下限时归一分母很小，8样本没有统计误差/迁移证书。算法只保证每个接受编辑改善能量。

**CPU复杂度。** 参考Lloyd与两图模式投影O(L N_R KD+NKD)，统计特征维1+K+K²，K≤8为73维。空间树仍O(ND+NlogN)。搜索评估O(NK²+C_Q C_R K²)，gallery可至数百，不是恒定1；full和trace各搜索一次，实际mask缓存按packbits精确比较。现有0.2147秒随机全维例把301基础前景全部删除（`covariance_check.json:213-246`），它是运行分支证据，不能称成功或完整数据集速度。所有计时均排除编码/I/O/base/评分。

**有限下一动作。** 三项中优先在下一次获准的真实缓存小机制屏检验第6项，固定full/trace/pointwise-margin/same-nearest MEAN及完整强基线；直接产出完整掩码、四类编辑和配对结果，不以合成矩阵、区域AUC或收敛证明替代。若full被trace或margin追平，撤回矩阵独立价值；若额外删除真目标抵消delete-FP，关闭v1，不展开新强度/新预算扫描。

## 与Pro五法的差异和重叠

Pro来源：`/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md`。本文只用给定报告作为方法定义对照，未外部核实其文献或成绩。

| 对照 | 真正不同的可观测量 | 重叠及不能扩大主张 |
|---|---|---|
| 第4项 vs Pro M3 | 第4项保留参考模式的2D空间质心布局并转移完整轮廓；M3预测的是角色特征Gram关系指纹，不拟合2D姿态（Pro:559-603），可接受更广的空间姿态差异。 | 都是部分角色拟合/留出验证、可重复参考解释。“heldout”名称、RANSAC/匹配不是独立原创证据。第4项全参考池、多峰相似变换和mask fusion；M3按参考组件模板、全Ward树/子区、Hungarian dummy、DP联合输出，不能当同法复现或相互继承成绩（Pro:542-555,607-624,630）。 |
| 第5项 vs Pro M3 | 第5项是无参考外观的原图几何矩量整体组件一致性；M3为外观角色关系验证。 | 都有树区域候选和完整区域证据，但第5项单调局部编辑不是M3树DP；M3的DP证书也仅是固定树解释，不能反向赋予第5项全局证书。第5项现有新增量未超generic。 |
| 第6项 vs Pro M2 | 第6项是FG内单token模式响应的全组件二阶散布；M2以FG/BG参考标签、四类参考边统计四状态角色关系，cross-ratio去掉可分一元项后得到query signed边（Pro:398-437）。 | 都检验参考的非逐点关系，但S的交叉项不是相邻token的标签cross-ratio，也没有M2的BG证据、边符号或严格凸全图固定点（Pro:439-480）。两者可能同样受单参考类别关系不可迁移制约，不能预先把增益相加。 |
| 第6项 vs Pro M3 | 第6项二阶响应汇总不需角色对应，廉价且保留模式坐标信息；M3显式对局部角色做关系预测与跨图heldout检验。 | 两者均试图区别均值相近而组成不同的区域；矩阵见证没有证明胜过完整M3或现代结构对应。不能将区域协方差模板匹配直接命名“heldout组成预测”。 |
| 第6项 vs Pro M4 | 第6项匹配天然前景部位混合的模式响应协方差；M4跨固定canonical内容、四个干预环境估计1024维环境散布并抑制这些方向（Pro:670-709）。 | “covariance”是共享工具，不是同一信息。第6项没有观测同内容跨环境重复量，不能继承M4的环境消除解释；M4正常9次编码及干预输入超出本次CPU缓存审查资源（Pro:681,923）。 |

第4–6项都不改变编码形成过程；Pro M1/M5新增后缀上下文/消息路径的可观测量，属于另一个信息来源，并需明确新的资源授权和实现，而非把现有CPU缓存方法换名。当前最有价值的新增量候选是第6项已通过算法的非逐点矩阵信息，第4项空间布局为另一信息轴；第5项还没有超generic的证据。以上是有限机制判断，未形成质量/原创排序实测。

## 共用入口和交付边界

共用入口复用同一次MEAN前图及base，若绑定现成base会验证maxdiff≤1e−6（`prepared_cpu_bundle.py:52-69`）。推断保存packed1024完整掩码及场，生成封存收据；评分先核对seal/哈希才读取truth（`prepared_cpu_bundle.py:175-191`；`scripts/run_cpu_feature_candidates.py:320-327,347-374`）。三项各runner JSON现有双worker与GT未读检查，全部明确real_episodes=0；这证明合成基础设施路径，不证明真实producer资产绑定或性能。PLAN当前仍缺真实清单/RGB路径/原图H/W绑定（`docs/research/PLAN.md:47-54`）。

三项读出不同：第4项continuous bilinear，第5/6项binary nearest；必须先比各自同读出MEAN控制，不能把读出造成的变化归给机制。Pro统一renderer明确使用bilinear binary/continuous且原尺寸再映射（Pro:235-254,1230-1237）；现有packed1024路径不能直接宣称已执行Pro原尺寸主指标合同。

本轮已完成的有限动作：第4项全局融合与canonical几何反例，第5项原正例generic追平复核，第6项原单位向量见证无写入复核。至此结束本地审查，不启动真实例，也不生成额外方法/实验队列。后续真实资源恢复后，一次有限机制屏才能决定整合、采用简单控制或关闭；任何结构变更都需明确版本，不能用目前合成证明提前称高质量新法。

## 本轮短计算的复核记录

命令环境：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3`，系统Python3、NumPy2.0.2、SciPy1.13.1；运行耗时分别约0.3秒和0.5秒，无源文件/既有证据写入。

```json
{"check":"constellation_unmatched_base_component","retained_poses":1,"original_component_pixels":16,"component_pixels_retained":0,"added_tokens":4,"deleted_tokens":16}
{"check":"canonical_anisotropic_geometry","proposed_poses":12,"accepted_poses":0}
{"check":"shape_existing_positive_fixture","reference_true_kept":64,"reference_wrong_kept":0,"generic_true_kept":64,"generic_wrong_kept":0,"reference_equals_generic":true}
{"controls_matched":true,"complete_outputs":{"covariance_true":64,"covariance_false":0,"trace_true":64,"trace_false":64,"margin_true":64,"margin_false":64},"covariance_debts":{"full_true":0.13793101432104043,"full_false":0.4387894160802329,"trace_true":0.13793101432104043,"trace_false":0.13793101830560187},"real_episodes":0}
```

可复核构造：第4项原始夹具取`check_geometry.py:55-66`，调用前额外设`base[2:6,45:49]=.9`；canonical反例参考地标`[10+10j,20+10j,10+20j,20+20j]`，query地标为`2*ref.real+1j*ref.imag+10+10j`，每角色仅该峰标准化响应3，原默认`propose/evaluate`调用；第5项直接重建`check_shape.py:26-35`，调用默认`predict(...,reference_hw=(32,32),query_hw=(32,32))`并比较`token_mask`与`compactness_control`；第6项importlib加载现有`check_reachable_witness.py`，用`unittest.mock.patch.object(Path,'write_text',return_value=0)`包围`main()`以禁止其覆盖现有JSON。当前第6项source SHA256为`97a449af59e10f01de06f3bb1649789ea9e83bc65d9a9a2b354692d423d747f6`，等于既有见证记录。
