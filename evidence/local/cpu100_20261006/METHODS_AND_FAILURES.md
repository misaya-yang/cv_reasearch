# CPU方法与失败证据：单一整理记录

**当前有真实开发集收益线索，但没有100个稳定有效方法，也没有完成同协议SOTA结论。原生DINO1024的30方法四例探针已完整，其中12方法随后在复用200例开发集完成。Huber参考读出在200例比ridge控制高0.7109 pp；多数高于prototype的收益可由简单控制解释。RGB02从4例正增益转为200例显著负。本文件只整理证据，不安排实验、不创建第二PLAN。**

记录日期2026-10-07；实验批次20261006。只写本文件，未SSH、未跑方法或实验。

## 证据口径与计数

审查修正：首30 + 二批22项新增 + 旧19 = 71项资格；二批实际请求23方法还包含local_002同方法修订，独立增量0，且不含local_004。batch03另新增context四项、local_005、QP09共6项探针资格，因此当前潜在77（新58 + 旧19），仍不是完成77，更不是稳定有效77或100。source审查、注册、完整运行与质量分别计。

旧600的四动作相对旧native，非MEAN；旧M5/gray相对同renderer MEAN；新原生相对direct prototype。对比基线不同，动作数量不跨组相加。

CPU入口平均秒不含冻结DINO提取；是否含共享prepare/controls/render以每例receipt为准，不把并发各例时间之和当墙钟。旧Pro/gray的真实编码、模型加载和后缀重放费用已另列。

新原生指标为原尺寸完整mask的class-summed mIoU：各类先汇总I/U再平均，不是episode均值或全局像素净收益。2000次RandomState(0)按关联参考/查询照片组配对bootstrap。最强声明control在固定controls中描述性选择，不另计方法。200方法是在4例后posthoc选入的12项，74个观测类，复用开发集，不是独立确认。

合同N：冻结DINOv3原生FP32 final-LN 1024 patch，经单位化供匹配，没有FoRIS Part1/旧score；raw不表示未做任何归一化。输入为R/Q、完整MR合法coverage、physical-valid及几何。RGB方法还要实际hash绑定原RGB/完整R二值mask。只有R标签可参与读出；QGT由封存后的独立scorer读取。冻结DINO不等于所有读出无拟合，DR12等临时R参数优化须披露。

旧合同P：多数工作q/r为归一化→条件FoRIS Part1去位置→归一化→FP16保存的DINO，且使用MEAN/native宿主。Pro原生FP32干预状态另列。旧14项600、4项暴露smoke4、1项只有本地构造；不能搬成raw1024测量，也不能拿4个hash扩成600来源逐例证明。

17项current module SHA匹配历史module，不等于当前CLI/依赖/renderer整链验证。gray源码不同，M4缺历史源码hash，M5缺历史runner hash都保留。Pro1/4/5/gray含模型、真实CPU编码或native后缀，不能标NumPy缓存开销。

[review batch02](reviews/batch02.json)

## 从Opus总结吸收的判断工具

从用户Opus总结实际吸收：先从模型原点解释错判，再问机制改变哪个判断；oracle阈值/面积/方向只指出缺口，不给合法取法；排序、区域、范围、读出分开；错误种子传播一致不等于正确；补真/补假/删真/删假分账；同读出、同字典、ridge/logistic等强控制防止挪用收益；数值正确、CPU速度、完整质量是不同证据。

用户研究档案（第一部分为原会话总结）（用户本地来源，未纳入仓库）

原总结现已并入研究档案；其中历史授权、任务与agent指引作为来源材料，不替代当前用户调度。

同一类别同一基线J=I/U：ΔJ=[(aTP−dTP)−J(aFP−dFP)]/[U+aFP−dFP]。仅补目标纯度需>J/(1+J)，仅删背景纯度需>1/(1+J)。旧37.5%/62.5%与LR>22/<0.24只属于旧队列/基线，不能套本轮。下文T统一顺序为补真/补假/删真/删假。新T相对direct prototype；旧T基线另列。全局T只描述动作，逐类IoU账才是主结论。

[native4 summary](server/probe29_native4_v2/methods_summary.json) SHA 4476f579ab4b00dd715de7b3551e4a0e87394ebc791f53e2725522b1f2184b2c
[native200 summary](server/screen12_native200_v3/methods_summary.json) SHA 760893fc13b3d29f4ed589b48a57914be969a803574925c6994567ee4f77a5d7

## 200例当前结果优先

| Method | mIoU | Delta prototype [95%] | Strongest declared control | Delta control [95%] |
|---|---:|---|---|---|
| inv_huber_reference_readout | 52.3734 | +9.7928 [+3.387, +10.996] | inv_huber_ridge (51.6625) | +0.7109 [+0.135, +1.001] |
| DR08 | 49.0343 | +6.4537 [+1.334, +8.501] | DR_control_average_logistic (50.3467) | -1.3124 [-4.199, +1.250] |
| DR03 | 48.9408 | +6.3602 [+1.410, +8.315] | DR_control_average_logistic (50.3467) | -1.4059 [-4.125, +0.874] |
| DR04 | 48.4795 | +5.8988 [+2.558, +7.337] | DR_control_average_logistic (50.3467) | -1.8672 [-2.524, -0.418] |
| cross_image_csls_hubness | 45.2626 | +2.6820 [-0.396, +4.628] | cross_image_cosine_dictionary_control (45.6269) | -0.3644 [-1.919, +1.882] |
| inv_adversarial_channel_support | 43.9545 | +1.3739 [-4.568, +4.144] | inv_adversarial_constant (50.7178) | -6.7634 [-9.111, -4.250] |
| inv_reference_mad_winsor | 42.8000 | +0.2193 [-0.593, +0.723] | dino_prototype.control (42.5806) | +0.2193 [-0.593, +0.723] |
| QP01 | 42.6975 | +0.1168 [-3.110, +3.409] | QP_center_prototype (43.0419) | -0.3444 [-1.934, +3.586] |
| QP02 | 42.4507 | -0.1299 [-3.212, +3.475] | QP_center_prototype (43.0419) | -0.5912 [-1.939, +3.717] |
| RGB02 | 38.0095 | -4.5711 [-6.760, -2.807] | dino_prototype.control (42.5806) | -4.5711 [-6.760, -2.807] |
| ref_ordinal_copula | 36.2029 | -6.3777 [-11.453, -2.467] | dino_prototype.control (42.5806) | -6.3777 [-11.453, -2.467] |
| local_001 | 12.1145 | -30.4661 [-34.013, -27.612] | dino_prototype.control (42.5806) | -30.4661 [-34.013, -27.612] |

200例Huber vs ridge +0.7109 [0.1353,1.0014]是已测真实开发边际；不能抹掉，也不能把相对prototype的+9.7928全部归给Huber。ridge=51.6625，平均logistic=50.3467，常量erosion=50.7178都是强简单controls。

尚无独立确认、同协议完整FoRIS比较、稳定>=2 pp或完整论文方法交付。新52.3734不能与旧processed60.70/62.93跨协议判优劣。负结果不证明信息用尽/DINO上限已定，也不授权新变体。

## 首30个已完整原生方法

预期字段保留试前卡的条件性假设；若已经被观察反驳，以该方法的失败/所得为准，不能将旧假设写成仍成立的质量主张。下列CPU秒数不包含DINO提取，是否含prepare/render/control以原receipt为准。

### ref_ordinal_copula - 逐图通道秩的单调域变换不变量

- **逻辑/条件性预期:** Algebraic: empirical channel ranks are unchanged by any strictly monotone increasing coordinate transform applied separately to R or Q, if no ties or membership/composition change. Hence the decision can differ from raw cosine without estimating a covariance nuisance. Channel-wise monotone transfer with differing target fractions is unverified and may break it.
- **预期收益（假设）:** algebraic/synthetic_witness pending: recover target ordering under the specified monotone domain-warp construction where raw prototype margin fails. Natural DINO segmentation gain unknown; no numerical prediction.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; native4 frozen source（本地证据，未纳入仓库） SHA 467769ef8c40e3f8; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 5607a1edf58c26c96d1a93f449e965f56c36cf3a114bbff5ffb27bf05208c36a; runtime config（本地证据，未纳入仓库）.
- **强control:** Raw weighted FG/BG prototype margin and raw FG cosine with identical outputs; per-image coordinate z-score prototype control separates ordinal nonlinear invariance from mere centering/scaling. Same marginal ranks under an orthogonal feature rotation tests its coordinate-dependent assumption.
- **4例描述性观察:** mIoU 24.9959; delta prototype -3.9883 [-22.965, +14.988]; strongest dino_prototype.control, delta -3.9883; T=0 / 35,347 / 29,744 / 57,554; mean entry 1.289s
- **200开发观察:** mIoU 36.2029; delta prototype -6.3777 [-11.453, -2.467]; dino_prototype.control delta -6.3777 [-11.453, -2.467]; T=30,143 / 1,109,403 / 2,683,477 / 4,089,263; mean entry 1.293s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> ref_ordinal_copula.class_actions_vs_prototype
- **失败解释与所得:** 200明显负且删真多。图内组成改变可改变通道秩，即便语义不变；当前汇总不提供组成/坐标旋转的唯一因果。单调warp不变量前提未支持自然转移。
- **特有反例:** If query target prevalence differs from reference or a new background mode changes marginal ranks, rank positions change although raw semantic features are unchanged. DINO basis rotations also alter the per-channel quantity; it is intentionally coordinate dependent.

### ref_local_support_radius - 参考类内支持半径校准的局部距离判别

- **逻辑/条件性预期:** The absolute nearest-distance decision and the normalized local-support decision need not agree even with the same FG/BG means and query positions. Candidate hypothesis: reference local appearance dispersion is a useful rejection scale, rather than assuming all nearest anchors equally reliable. This is not semantic confidence or a calibrated posterior.
- **预期收益（假设）:** hypothesized: delete false targets that win raw nearest matching only because a compact class has an unusually close isolated anchor; synthetic positive must establish this action. May lose shifted true targets, so no natural gain prediction.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; native4 frozen source（本地证据，未纳入仓库） SHA 467769ef8c40e3f8; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine, raw FG/BG prototype and raw nearest-class distance; single role-global radius distance control; role-balanced fixed-kernel density control under the same selected tokens and same render. The local-radius arm must alter a decision beyond these controls to justify distinctness.
- **4例描述性观察:** mIoU 0.9745; delta prototype -28.0097 [-44.199, -11.821]; strongest ref_support_nearest_control, delta -29.1460; T=0 / 88 / 70,970 / 118,501; mean entry 0.407s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> ref_local_support_radius.class_actions_vs_prototype
- **失败解释与所得:** 4例大量删真；参考类内紧支持半径不能直接校准Q跨图语义位移。宽BG支持、孤立FG和域偏移是明确反例；没有200测量，不否定全support家族。
- **特有反例:** Broad clutter modes can acquire huge radii and become easy BG explanations, deleting genuine shifted target features. A singleton rare FG mode gets no transferable scale. Reference-only local radii need not predict a different image's semantic displacement.

### ref_joint_channel_code - 参考角色的非因子化通道共现编码

- **逻辑/条件性预期:** An eight-cell joint histogram can separate3-bit parity while univariate and pairwise histograms, means and covariance are identical. This is a finite nonfactorized decision, not an oracle label or new input. Need explicit quadratic control and a saturation/overfit negative. Novelty is not claimed.
- **预期收益（假设）:** synthetic_witness pending: resolve a same-low-moment target/distractor distinction that a quadratic decision cannot separate. In real DINO channels the source-selected tuple may be pure overfit; actual segmentation gain unknown.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; native4 frozen source（本地证据，未纳入仓库） SHA 467769ef8c40e3f8; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine/prototype; selected channels univariate naive-Bayes; selected channels pairwise maximum-entropy/degree2 ridge cell features with the same add-one rule, same labels and same renderer. A degree3 polynomial fit is an additional control to expose whether histogram complexity alone matters.
- **4例描述性观察:** mIoU 6.1412; delta prototype -22.8430 [-37.379, -8.307]; strongest dino_prototype.control, delta -22.8430; T=18 / 144,896 / 57,797 / 96,227; mean entry 0.342s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> ref_joint_channel_code.class_actions_vs_prototype
- **失败解释与所得:** 4例加FP、删TP都大。参考选tuple可能源内过拟合，跨图阈值翻bin可改变身份；parity构造不等于自然收益。
- **特有反例:** An unknown query style shift crosses reference coordinate thresholds; bins then change regardless of target identity. Selected channels may each already separate roles, leaving no joint information; 56-tuples on64rows can overfit reference. Orthogonal rotations destroy the fixed-coordinate parity.

### local_001 - Centered local DINO Gram signature classification

- **逻辑/条件性预期:** For a token neighborhood, weighted centering and trace-normalized Gram formation cancel a common additive component, common nonzero scale and orthogonal coordinate change of the vectors supplied to the Gram step. Because the algorithm first unit-normalizes DINO inputs, arbitrary pre-normalization additive shifts do NOT enjoy this invariance. One exact legal unit-vector construction is y_i=s O x_i+b with b orthogonal to every O x_i and s^2+||b||^2=1: the signature is unchanged. The eigenspectrum and sorted center-to-neighbor Gram row can differ when absolute class means and scalar local trace tie. These are algebraic facts under explicit conditions, not an assertion that real DINO scene shifts meet them.
- **预期收益（假设）:** If target local relational variation transfers while the absolute feature direction changes, the classifier can change a raw-DINO tie or misordering. No numerical natural-image gain, stable benefit or originality is established.
- **source/合法输入:** [source](../../../src/ics/cpu100/local_structure.py) SHA 63b1cfe14ecd5a3e; native4 frozen source（本地证据，未纳入仓库） SHA a15b46892a991cd0; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 22ac1e36e634b9bcc12431e42729e0f5719bce661bbba9945cd0a40cae889c65; runtime config（本地证据，未纳入仓库）.
- **强control:** Same input and same renderer: complete weighted foreground/background mean cosine margin.; Same reference kernel classifier using only normalized local trace and sorted per-row diagonal variances; checks whether cheap local variation rather than higher joint structure suffices.; Same signature kernel classifier with Gram replaced by its diagonal; no change to labels, bandwidth rule, reference weights or renderer.; Label-permuted reference weights as a dependency check; not a new method.
- **4例描述性观察:** mIoU 14.3401; delta prototype -14.6441 [-27.266, -2.022]; strongest dino_prototype.control, delta -14.6441; T=21,789 / 301,546 / 17,265 / 43,782; mean entry 1.226s
- **200开发观察:** mIoU 12.1145; delta prototype -30.4661 [-34.013, -27.612]; dino_prototype.control delta -30.4661 [-34.013, -27.612]; T=523,466 / 12,868,426 / 2,493,406 / 2,881,310; mean entry 1.266s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> local_001.class_actions_vs_prototype
- **失败解释与所得:** 200加FP12,868,426、删TP2,493,406。弱化绝对方向后局部结构不是可靠身份判据。比trace/乱标control强仍不抵消对prototype的大损失；不否定所有空间结构。
- **特有反例:** A distractor with the same local Gram signatures as the target receives the same evidence, even when its absolute DINO direction would distinguish it.; A truly uniform target and uniform background have zero descriptors and are not identifiable by this method.; Changed articulation, patch sampling or local material pattern may change the target signature and cause deletion; this is a local structure transfer assumption, not a universal invariant.; A reference that labels only one visible local pattern cannot cover unseen patterns or very small target interiors; multiple query instances are all classified independently, without identity propagation.; No wrong-seed escape claim: the method has no query seed, but a mislabeled/corrupted reference pattern can confidently misclassify repeated query distractors.

### local_002 - Dense reference-index correspondence with bounded local deformation

- **逻辑/条件性预期:** A joint per-token reference coordinate assignment carries compatibility information discarded by independent cosine matching and by a bag of mode-pair counts. Bounded spatial changes in the inferred correspondence field allow articulation; no single accepted pose transfers a reference mask. This information helps only if appearance correspondences are sufficiently correct and compatible, and a coherent false match remains possible.
- **预期收益（假设）:** When a true object has locally smooth but globally nonrigid reference correspondences and an equally similar distractor requires incompatible matches, dense coherence can improve sorting and preserve separately matching instances. No real numeric gain is established.
- **source/合法输入:** [source](../../../src/ics/cpu100/local_structure.py) SHA 63b1cfe14ecd5a3e; native4 frozen source（本地证据，未纳入仓库） SHA a15b46892a991cd0; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Same pooled anchors and binary/fractional-label readout with per-token best cosine match.; Same candidates and ICM optimizer with only FG/BG Potts compatibility rather than correspondence-coordinate compatibility.; Same graph coherence with all reference coordinates permuted; a mechanism dependency control, not a method.
- **4例描述性观察:** mIoU 29.0748; delta prototype +0.0906 [-5.977, +6.158]; strongest local_002.unary_control, delta -0.4036; T=54 / 14,497 / 19,107 / 47,925; mean entry 3.469s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> local_002.class_actions_vs_prototype
- **失败解释与所得:** 4例不胜同候选unary，局部坐标相容性未立额外价值。pool-aware tau后续改变目标，属于同机制修订计0，不能继承v0分数。
- **特有反例:** A wrong initial unary correspondence can be a local optimum; three starts do not guarantee global MAP.; A copied patchwork with coherent coordinates is indistinguishable; coherent matching is not semantic identity.; Large viewpoint, scale change or articulation gradient can exceed the bounded strain and be penalized.; Similar adjacent true instances can be coupled by the graph and lose one correspondence field.; Reference pooling may remove a tiny target; this loss must be matched in the anchor control and reported.

### QP01 - 查询字典上的完整参考标签反向计数

- **逻辑/条件性预期:** 查询token的几何用于决定参考监督在哪些查询原型上累积。所有参考token独立投票给最近查询原型；FG/BG总票分别归一化，避免MR面积成为类别先验。它和用query原型到reference均值做点积不是代数同一算子：参考token落在哪个Voronoi cell改变计数，即使参考FG/BG均值相同，完整参考分布也可改变query组的标签。但这只说明有额外可用决策，不证明DINO真实分布中的身份迁移成立。
- **预期收益（假设）:** 当参考标签在查询字典上可区分，而查询弱部分仍落在同一纯组时，可一次补回弱部分；不依赖同分数估面积。真实mIoU未测、不预测分数。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; native4 frozen source（本地证据，未纳入仓库） SHA 6767bb1841a99c1d; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40; runtime config（本地证据，未纳入仓库）.
- **强control:** 同输入/同读出的参考FG-BG均值margin。; 同输入/同读出的类别平衡cosine KDE log-density ratio；固定温度0.07。; 相同query字典，直接按query组内KDE log-density ratio均值作共同标签；用于隔离反向标签计数是否只是池化已有分数。; 逐reference最近query-token投票，再按相同query字典累积；隔离中心归纳是否是新差异。
- **4例描述性观察:** mIoU 35.7922; delta prototype +6.8080 [+0.889, +12.727]; strongest QP_center_prototype, delta +1.5881; T=1,408 / 134,971 / 850 / 24,022; mean entry 0.577s
- **200开发观察:** mIoU 42.6975; delta prototype +0.1168 [-3.110, +3.409]; QP_center_prototype delta -0.3444 [-1.934, +3.586]; T=367,765 / 3,581,735 / 533,138 / 2,915,838; mean entry 0.527s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> QP01.class_actions_vs_prototype
- **失败解释与所得:** 4例+6.808缩为200 +0.117且区间跨0，中心prototype control更高。查询分组可一致地标错身份，小目标也可被字典压缩吞并。
- **特有反例:** query目标和干扰共享同一外观字典项时，所有两区必须获得相同标签；小目标对应中心未被K字典保留时会被大背景吞并；跨图偏移使参考FG落在错误query中心时整组误选。错种子负例中本方法没有输入seed，仍需证明支持错落时不会自称识别成功。

### QP02 - 查询字典的全参考成对风险联合命名

- **逻辑/条件性预期:** 固定query字典后，为所有query原型联合选择FG/BG二元标签，使它们作为分类原型时对完整参考标签的类别平衡logistic风险最小。前景角色命名由参考监督共同决定，不先指定某个query峰为FG；与QP01的独立Voronoi票符号不同，候选标签之间通过各类最大响应非线性交互。对这个固定候选域，最小风险不大于任何简单control诱导字典标签的风险是代数保证；这只保证reference拟合，不保证query迁移或分割收益。
- **预期收益（假设）:** 可命名没有获得FG最近票、但作为FG类别原型时比反向赋值更能解释完整参考FG/BG间隔的query组；没有真实收益分数。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; native4 frozen source（本地证据，未纳入仓库） SHA 6767bb1841a99c1d; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40; runtime config（本地证据，未纳入仓库）.
- **强control:** 同query字典QP01反向平衡hard票。; 同query字典按中心FG-BG均值margin命名。; 同query字典按中心KDE log-density ratio命名；同读出。; QP01控制中的pointwise KDE。
- **4例描述性观察:** mIoU 35.9029; delta prototype +6.9186 [+1.389, +12.448]; strongest QP_center_prototype, delta +1.6987; T=2,835 / 79,761 / 333 / 32,953; mean entry 0.641s
- **200开发观察:** mIoU 42.4507; delta prototype -0.1299 [-3.212, +3.475]; QP_center_prototype delta -0.5912 [-1.939, +3.717]; T=451,907 / 3,347,807 / 481,950 / 2,709,926; mean entry 0.564s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> QP02.class_actions_vs_prototype
- **失败解释与所得:** 4例+6.919变为200 −0.130。完整R风险低不保证Q身份；200加FP3,347,807，联合命名未成为可靠补目标机制。
- **特有反例:** reference中局部BG偶然像query目标、域偏移逆转距离时，最低reference风险可选错完整query身份。已知query有多个同类实例时不限制只选一组，但未被reference解释的外观模式仍可能全删。

### cross_image_csls_hubness - 双侧局部密度校正的参考角色匹配

- **逻辑/条件性预期:** 令Sij=q_i dot r_j。对每个r_j，h_j=它到query top-k的平均相似度；若某参考原子因通用外观成为高密度hub，h_j高。匹配分数2Sij-h_j降低它的优势；query侧密度项在FG/BG差值中严格抵消，必须报告此代数事实而非假称双侧都贡献。校正能改变参考原子排序，非单调重标定。hub确实来自非目标而非同类重复是未验证假设。
- **预期收益（假设）:** 降低通用FG原子对背景hub的误匹配，同时可能补回原本被hub遮挡的特异FG匹配
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching.py) SHA 1469acd537b3c7e1; native4 frozen source（本地证据，未纳入仓库） SHA 1469acd537b3c7e1; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 1469acd537b3c7e1fa17b9423c677e078182e3b3a6ba2667e019e08021b1411a; runtime config（本地证据，未纳入仓库）.
- **强control:** 同压缩、同top3支持、同render的未校正cosine；另把h_j替换为常数，恒等于该控制；重复同类Q原子/重复背景Q原子压力测试。
- **4例描述性观察:** mIoU 34.6458; delta prototype +5.6616 [-7.356, +18.679]; strongest dino_prototype.control, delta +5.6616; T=121 / 39,745 / 16,568 / 33,959; mean entry 1.121s
- **200开发观察:** mIoU 45.2626; delta prototype +2.6820 [-0.396, +4.628]; cross_image_cosine_dictionary_control delta -0.3644 [-1.919, +1.882]; T=238,044 / 1,479,502 / 475,948 / 2,556,437; mean entry 0.841s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> cross_image_csls_hubness.class_actions_vs_prototype
- **失败解释与所得:** 200对prototype +2.682跨0，低于同字典cosine0.364；字典本身可解释收益，hub校正必要性未立。真实重复目标也可被罚。Q侧行常数抵消不能当双侧增益。
- **特有反例:** 真实目标在Q内数量大、外观同质，而背景罕见；FG原子h_j高反被惩罚，稀有错误BG原子得优势。若所有h_j相同，输出必须退化为未校正控制。

### cross_image_background_anchor_shift - 共享背景对应约束的参考平移

- **逻辑/条件性预期:** 若共享背景对应满足q_j=r_i+b+epsilon，则这些匹配差的稳健中心估计b，不受R/Q物体占比差直接影响。不改变任何协方差/语义方向；只平移参考FG/BG原子，再与完整Q比较。锚点可错，b可能是背景类别差而非style，必须负例检测；无可信一致锚则fallback。
- **预期收益（假设）:** 锚点识别正确且style近似加性时，纠正FG/BG对应偏移，不假定整个R/Q分布同组成
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching.py) SHA 1469acd537b3c7e1; native4 frozen source（本地证据，未纳入仓库） SHA 1469acd537b3c7e1; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 直接prototype；全图center/mean-shift；同一背景锚集合但无平移。共享背景真对应正例，R/Q不同背景却互NN高margin反例，以及重复FG改变整图mean但不改变真实锚残差的压力测试。
- **4例描述性观察:** mIoU 28.9843; delta prototype +0.0000 [+0.000, +0.000]; strongest cross_image_whole_mean_shift_control, delta -8.0269; T=0 / 0 / 0 / 0; mean entry 1.080s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> cross_image_background_anchor_shift.class_actions_vs_prototype
- **失败解释与所得:** 4例全部fallback、T全0。锚不足是实际活动限制，不能将prototype回退分数当anchor有效。
- **特有反例:** 两图共享的背景类别不是真正同一外观，互NN残差一致但误差是语义差；平移会破坏正确FG方向。锚稀疏/全同质会fallback，本法可能真实数据几乎从不活动。

### cross_image_free_column_gram_matching - 无参考列容量的跨图关系对应

- **逻辑/条件性预期:** Gram在共同正交基变换下不变；若Q类别部位关系保留而绝对向量旋转，距离关系可区分同样cross-cosine的候选。此假设可能随视角/背景组成失效。P每Q原子行和=1而参考列无上限，因此可多Q实例映射同一R模式，不暗设目标占比。
- **预期收益（假设）:** 在相对关系保留但跨图向量方向改变的情形改善角色对应；允许多个查询实例重用reference modes
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching.py) SHA 1469acd537b3c7e1; native4 frozen source（本地证据，未纳入仓库） SHA 1469acd537b3c7e1; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 相同mode压缩和softmax外观匹配；每mode的sorted within-image distance profile nearest match（廉价结构控制）；小图穷举1-to-many hard assignments给目标界，不作为部署；P梯度finite-difference验证。
- **4例描述性观察:** mIoU 26.6055; delta prototype -2.3788 [-9.380, +4.623]; strongest dino_prototype.control, delta -2.3788; T=28 / 21,695 / 14,055 / 25,753; mean entry 0.713s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> cross_image_free_column_gram_matching.class_actions_vs_prototype
- **失败解释与所得:** 指定旋转构造被简单sorted-profile解开且目标偏爱错解，固定收益主张关闭；不能都推给optimizer或继续调参。4例负仅是固定版范围。
- **特有反例:** FG与干扰的内部距离结构同构，或Q背景类别组成不同导致全图关系无法对应；同类多实例带不同姿态亦可能改变距离。非凸P优化还可能困在外观错误初始化。

### RGB01 - 参考背景锚定的RGB光照对齐

- **逻辑/条件性预期:** Use reference BG and DINO-confident query BG only to estimate per-channel median location/scale nuisance; unlike DINO whitening, RGB nuisance fit is explicit diagonal affine and does not remove DINO semantic variance. The correction is useful only if BG-anchor mixture remains comparable. Synthetic first check found the aligned nearest-median control solved the witness equally; primary was simplified to that control rather than attributing the alignment gain to complex density.
- **预期收益（假设）:** synthetic hypothesis: correct a color-distinct distractor rejection despite an affine RGB illumination shift, when BG anchor composition is genuinely stable.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement.py) SHA 027897bcc1013725; native4 frozen source（本地证据，未纳入仓库） SHA 027897bcc1013725; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Unaligned/ aligned two RGB medians with the identical DINO cap; BG-anchor oracle is forbidden.
- **4例描述性观察:** mIoU 17.5579; delta prototype -11.4263 [-21.111, -1.742]; strongest dino_prototype.control, delta -11.4263; T=6,230 / 134,976 / 28,170 / 34,095; mean entry 0.378s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> RGB01.class_actions_vs_prototype
- **失败解释与所得:** aligned和unaligned均低于DINO，不能只怪alignment。加真6230却加假134976；颜色跨实例或背景锚组成假设可能失效，未测唯一因果。复杂Laplace还更弱。
- **特有反例:** Q background is a different scene color or DINO confident BG contains target: nuisance map is not illumination and can invert identity; target changes color across instances.

### RGB02 - 参考监督的局部序纹理直方图

- **逻辑/条件性预期:** For each 16x16 true RGB gray patch, 8-neighbor rank comparisons encode local ordinal structure. Positive affine intensity transformations preserve comparisons. This differs from radial power shares; whether it adds beyond power must be tested.
- **预期收益（假设）:** synthetic hypothesis: local-order identity evidence when patch colors and radial power are insufficient, without assuming target location/size.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement.py) SHA 027897bcc1013725; native4 frozen source（本地证据，未纳入仓库） SHA 027897bcc1013725; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **native200 frozen binding:** module SHA 027897bcc10137258fd1020207bfc1624a31a9bf6da13fd451a245e38d11fb1b; runtime config（本地证据，未纳入仓库）.
- **强control:** Same-window dominant frequency/energy and radial power, not only an intentionally weak global color histogram.
- **4例描述性观察:** mIoU 32.8750; delta prototype +3.8907 [+0.000, +7.781]; strongest RGB02.full_power, delta +0.1962; T=1,423 / 12,908 / 2,104 / 36,050; mean entry 0.440s
- **200开发观察:** mIoU 38.0095; delta prototype -4.5711 [-6.760, -2.807]; dino_prototype.control delta -4.5711 [-6.760, -2.807]; T=221,397 / 2,144,417 / 454,775 / 1,039,609; mean entry 0.432s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> RGB02.class_actions_vs_prototype
- **失败解释与所得:** 4例优势未守住：200加真221397/加假2144417、删真454775/删假1039609。像目标背景可共享纹理，MR外观和视域/尺度未必跨Q稳定；并非已证唯一原因。胜颜色/方差不等于胜DINO，full-power增量仍跨0。不改冻结版、不否定所有RGB。
- **特有反例:** DINO-target and non-target have same local ordinal texture, or texture scale is changed/blurred in Q; repeated ordinal texture on background causes coherent FP.

### RGB03 - 参考监督的三频相位耦合

- **逻辑/条件性预期:** Normalized bispectral products X(k)X(l)conj(X(k+l)) preserve translation phase cancellation and observe phase relationships erased by power. This is a precise new observable; translation on cyclic patch is the exact invariance, not arbitrary natural-image scale invariance.
- **预期收益（假设）:** algebraic observer separation: equal-amplitude different-phase textures may become distinguishable. Natural RGB gains remain unknown.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement.py) SHA 027897bcc1013725; native4 frozen source（本地证据，未纳入仓库） SHA 027897bcc1013725; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Full 2D power spectrum with same reference labels and Euclidean classifier; radial-only control is not enough.
- **4例描述性观察:** mIoU 28.7049; delta prototype -0.2793 [-0.698, +0.139]; strongest RGB03.ordinal, delta -4.1700; T=472 / 34,399 / 951 / 13,595; mean entry 0.529s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> RGB03.class_actions_vs_prototype
- **失败解释与所得:** 同power/ordinal构造可区别phase，4例自然却输ordinal。相位组织跨实例稳定性未立，构造成功不保证身份。
- **特有反例:** Texture has unstable local phase or isotropic noise; patch translation with nonperiodic crop changes observations; identical bispectra/class textures give no identity.

### RGB04 - 参考多模态 RGB patch 字典

- **逻辑/条件性预期:** A nonparametric reference dictionary retains modes and spatial pixel arrangements rather than their mean. The evidence is reference RGB patch similarity, not transfer of the reference silhouette.
- **预期收益（假设）:** synthetic hypothesis: preserve two known target appearance modes when their average coincides with background; benefit requires their actual local arrangement to repeat in Q.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement.py) SHA 027897bcc1013725; native4 frozen source（本地证据，未纳入仓库） SHA 027897bcc1013725; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Nearest RGB patch dictionary using all allowed reference samples; class-mean same descriptor tests whether nonparametric mode retention supplies the benefit.
- **4例描述性观察:** mIoU 29.3625; delta prototype +0.3783 [-5.324, +6.080]; strongest RGB04.ordinal, delta -2.6225; T=1,774 / 54,184 / 2,536 / 13,860; mean entry 0.478s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> RGB04.class_actions_vs_prototype
- **失败解释与所得:** 多模态保留构造可达，4例微正但输同窗ordinal。像素排列视角/尺度脆弱是前提风险，非已测唯一原因；全R字典control源码已补，未见真实评分。
- **特有反例:** Viewpoint/rotation makes pixel arrangements incomparable, nearest dictionary mistakes accidental background pattern, or reference FG support is too small.

### RGB05 - 参考监督 DINO–RGB 邻域关系耦合

- **逻辑/条件性预期:** Local pairwise DINO distances and local pairwise RGB distances encode their coupling. Correlation of distance matrices distinguishes matched vs mismatched co-variation, even if each modality separately has the same marginal histogram. This coupling can be measured from the given pair, not inferred from Q area.
- **预期收益（假设）:** algebraic hypothesis: identity can differ through cross-modal arrangement while separate marginal appearance summaries tie; real transfer unknown.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement.py) SHA 027897bcc1013725; native4 frozen source（本地证据，未纳入仓库） SHA 027897bcc1013725; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Same pairwise distances with uncoupled concatenated histograms and pairing-permutation control. Raw RGB alone is insufficient attribution.
- **4例描述性观察:** mIoU 22.9314; delta prototype -6.0529 [-11.591, -0.514]; strongest dino_prototype.control, delta -6.0529; T=3,685 / 124,787 / 6,203 / 28,523; mean entry 0.706s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> RGB05.class_actions_vs_prototype
- **失败解释与所得:** 4例耦合负且permutation近似，未立自然配对因果增量。场景/影子共同变化可与类身份无关；胜更弱uncoupled control不构成完整收益。
- **特有反例:** FG itself has weak/no RGB–DINO coupling, DINO context couples shadows instead of class, or class and distractor have identical joint relation.

### DR01 - 已知参考覆盖率的仿射趋势外推签名

- **逻辑/条件性预期:** weighted二列线性回归对合法known R coverage提取条件方向；只有在未unit的潜在向量严格r_i=c_i*f+(1-c_i)*b时才有满秩精确恢复性质。common要求所有r unit，因此非平凡多c的unit凸混合通常不可能，不能把该理想代数直接当DINO真值。本算法在unit r上做趋势拟合并外推coverage0/1，属于待证启发式；实际第一构造将最终r单位化并承认不再精确恢复。
- **预期收益（假设）:** unit参考向量仍随known coverage具有稳定趋势时，外推可能减少prototype污染。没有DINO单位向量精确解混定理，真实mIoU未测。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同输入同render的weighted FG/BG prototype；仅pure coverage>=.75/<=.25 reference prototype；这些控制不计方法。
- **4例描述性观察:** mIoU 30.4456; delta prototype +1.4613 [-0.013, +2.936]; strongest DR_control_average_logistic, delta -8.4682; T=0 / 0 / 63 / 4,927; mean entry 0.366s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> DR01.class_actions_vs_prototype
- **失败解释与所得:** unit特征不是精确线性混合；4例小正但远不胜简单logistic，coverage外推未立独立价值。
- **特有反例:** DINO单位化后的r_i不满足线性混合；foreground与background因上下文非线性改变、c无跨度或真实referenceFG内部多模态时可能失败，解析解也可能放大噪声。

### DR02 - 参考边界局部配对的场景抵消判别

- **逻辑/条件性预期:** 若边界相邻r_f=t+n_local和r_b=b+n_local，则差r_f-r_b抵消共享局部项。全图背景均值不共享n_local时不会抵消。该假设须用局部与随机配对同成本控制区分，不能把位置本身说成语义。 common单位rf/rb下mid dot unit(rf-rb)=0，故midpoint项恒消失，实际算法完全等价q dot(weighted mean of unit pair differences)。局部信息仅来自pair选择及逐pair归一化，不声称midpoint增加信息。
- **预期收益（假设）:** 局部共同场景项很强时可从reference中提取更接近目标vs邻背景的方向，降低远背景造成的身份偏差。真实质量未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** weighted prototype；相同FG/BG token数的deterministic远端配对；所有cross-class pair平均。所有pair数量与render匹配，不把更少reference tokens偷作方法优势。
- **4例描述性观察:** mIoU 18.9347; delta prototype -10.0495 [-20.498, +0.399]; strongest DR_control_average_logistic, delta -19.9791; T=3,349 / 199,418 / 102 / 8,415; mean entry 0.296s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> DR02.class_actions_vs_prototype
- **失败解释与所得:** unit pair midpoint项恒0已修。4例加FP199418/加TP3349；边界方向可偏重材质或轮廓，没有额外midpoint身份信息。
- **特有反例:** 边界tokens已经混合；局部BG恰好也是目标材质、目标边缘只剩轮廓而非类别，或Q局部上下文偏移不共享R midpoint，均可导致负迁移。

### DR03 - 参考空间组的最坏组判别风险

- **逻辑/条件性预期:** average loss可在少数reference空间组付出很大分类误差；min max_g balanced reference loss为每个有证据的空间组设置竞争。新增量是已知mask下的空间条件判别错误，不是重新命名全图score。
- **预期收益（假设）:** reference少数合法部位被majority平均牺牲且这些部位出现在query时，有望恢复它们；无真实mIoU增益证据，DRO并非跨域保证。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA aef1516b39f2e7e1278c211bc71fc7953e50db9526ab619f81c74e3615ba5db9; runtime config（本地证据，未纳入仓库）.
- **强control:** 相同模型、200步、正则与数据的class-balanced average logistic；随机打散空间组DRO；weighted prototype。只loss组结构不同，不以更大模型取胜。
- **4例描述性观察:** mIoU 36.6139; delta prototype +7.6296 [-3.683, +18.942]; strongest DR_control_average_logistic, delta -2.2999; T=0 / 27 / 12,929 / 49,962; mean entry 0.328s
- **200开发观察:** mIoU 48.9408; delta prototype +6.3602 [+1.410, +8.315]; DR_control_average_logistic delta -1.4059 [-4.125, +0.874]; T=39,528 / 180,085 / 1,117,228 / 3,717,422; mean entry 0.307s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> DR03.class_actions_vs_prototype
- **失败解释与所得:** 200 +6.360对prototype但低于平均logistic1.406；最坏组额外项未立。source最坏组可为污染或Q不出现的部位，不能归因全部收益于DRO。
- **特有反例:** 最坏组本身包含boundary混合、错标或不共享query的背景；max risk会过拟合那个组并损害所有干净组。空间相关性意味着无IID一般化保证。

### DR04 - 参考标注风险的类内修剪判别

- **逻辑/条件性预期:** 在明确的每类最多25% reference risk污染假设下，least-trimmed logistic可用多数干净reference约束方向；普通平均loss的高损失outlier可能主导梯度。25%是假设不是观测事实，也不是query target area prior。
- **预期收益（假设）:** reference gross-risk异常确为不转移的污染时，可减少跟随污染的误改；不保证真实mIoU，rare但合法目标可能被误丢。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA aef1516b39f2e7e1278c211bc71fc7953e50db9526ab619f81c74e3615ba5db9; runtime config（本地证据，未纳入仓库）.
- **强control:** 相同200 gradient steps、正则、数据、render的average balanced logistic；uniform deterministic 75% reference removal；weighted prototype。
- **4例描述性观察:** mIoU 37.4078; delta prototype +8.4236 [-0.754, +17.601]; strongest DR_control_average_logistic, delta -1.5060; T=18 / 307 / 4,041 / 35,178; mean entry 0.344s
- **200开发观察:** mIoU 48.4795; delta prototype +5.8988 [+2.558, +7.337]; DR_control_average_logistic delta -1.8672 [-2.524, -0.418]; T=5,210 / 428,732 / 617,606 / 2,230,176; mean entry 0.313s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> DR04.class_actions_vs_prototype
- **失败解释与所得:** 200对logistic −1.867且区间全负；trim必要性未立，可能丢合法难部位。有效删FP和删TP代价同时保留，不扫trim比例。
- **特有反例:** 少数高loss reference token恰是query中的合法目标部位，则trim消灭最需要的信息；随机label噪声超过25%或起始方向错误也可能锁进错误局部解。

### DR05 - 参考空间jackknife不确定性选择性改判

- **逻辑/条件性预期:** 对每个Q token，LOO参考方向的变化可揭示该token判别依赖reference某一位置。jackknife偏差修正和方差是不同Q token的量，通常不是prototype单分数的单调函数；空间相关reference下它不是严格概率置信证书。
- **预期收益（假设）:** prototype受到小块非转移影响且jackknife偏差反映它时，可作少量有根据的身份改判；interval不是跨图泛化保证，真实质量未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** full prototype；LOO mean仅不用uncertainty guard；空间块随机token替代的jackknife；同uncertainty但不bias-corrected控制(通常不改变base符号)。
- **4例描述性观察:** mIoU 29.1483; delta prototype +0.1640 [+0.006, +0.322]; strongest DR_control_average_logistic, delta -9.7655; T=3 / 29 / 23 / 687; mean entry 0.487s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> DR05.class_actions_vs_prototype
- **失败解释与所得:** 4例改动极少且不胜logistic；参考块可共同错。空间jackknife非跨域confidence，1.96不具名义95%保证。
- **特有反例:** 所有reference块共享错误身份，jackknife稳定地错；合法特异部位只在一块时删块变化大，guard可能保留base错误。LOO三块非IID使1.96不具名义95%覆盖。

### DR06 - 参考正负分布的全配对秩竞争

- **逻辑/条件性预期:** p(q)=sum_if sum_jb wf_i*wb_j*[q dot r_i>q dot r_j]/(sum wf sum wb)，ties贡献1/2。它衡量FG相似值击败BG相似值的加权频率，可与均值差有相反符号，故不是原score重标定；也不是把AUC直接当校准概率。
- **预期收益（假设）:** 当reference多数pair可靠而少数极端相似token错误主导mean时，可纠正身份；p不是query目标后验，不承诺mIoU最优。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同输入weighted prototype、weighted median FG-BG similarity、最近reference FG/BG竞争；同render。
- **4例描述性观察:** mIoU 28.5806; delta prototype -0.4037 [-0.927, +0.119]; strongest DR_control_average_logistic, delta -10.3332; T=3,070 / 91,757 / 0 / 68; mean entry 0.385s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> DR06.class_actions_vs_prototype
- **失败解释与所得:** 4例加FP91757/加TP3070。pair-rank不是Q后验，目标只对应少量R部位可被多数pair误判；pair数量不是独立样本量。
- **特有反例:** query真目标只对应reference极少数部位，多数pair不支持目标时会删真；跨图全FG相似度下降会被稳定误拒；token相关性使pair数不能当独立样本数。

### DR07 - 完整参考掩码连通成分的多实例判别

- **逻辑/条件性预期:** known class mask中每个disconnected FG component本来就是合法正证据，其appearance均值不应因面积小而消失。max(FG component similarity)-max(BG component similarity)保留少数reference实例，同时用每个已知BG成分竞争。不是证明组件等于真实实例。
- **预期收益（假设）:** reference minor connected object外观重要且query与其相似时，可能恢复wholeclass成员；真实mIoU未测。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** global weighted prototype；同K feature-space kmeans prototypes(同信息成本)；只最大FG component；all native FG/BG nearest-token classifier。
- **4例描述性观察:** mIoU 29.1053; delta prototype +0.1211 [+0.001, +0.241]; strongest DR_control_average_logistic, delta -9.8085; T=0 / 574 / 30 / 929; mean entry 0.340s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> DR07.class_actions_vs_prototype
- **失败解释与所得:** 4例微正不胜logistic。R连通组未必覆盖Q外观，max原型也可放大污染小组件；200 unknown。
- **特有反例:** 单物体mask被native lowresolution断裂，背景切成许多任意块；tiny reference component为错标/上下文时max会高敏感地产生FP。MR threshold .5损失fractional对象时只能退化。

### DR08 - 参考两类凸集的最大间隔判别方向

- **逻辑/条件性预期:** 当FG/BG两个convex hull分离，最短连接f*-b*及其中点是最大间隔hyperplane。相对于mean，它保障reference最困难几何点；这是reference geometry性质，不是跨图语义保证。
- **预期收益（假设）:** reference异类最近几何边界恰能跨图保持时，修正mean方向忽视的hard reference negatives；真实mIoU未知，不能由reference separability推出query成功。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk.py) SHA aef1516b39f2e7e1; native4 frozen source（本地证据，未纳入仓库） SHA 0a82ba2dafe4be10; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA aef1516b39f2e7e1278c211bc71fc7953e50db9526ab619f81c74e3615ba5db9; runtime config（本地证据，未纳入仓库）.
- **强control:** 相同pure-ish tokens weighted prototypes；balanced average logistic；closest-query-hull distance classification(作为已失败族同信息控制)。
- **4例描述性观察:** mIoU 37.5062; delta prototype +8.5220 [-6.509, +23.553]; strongest DR_control_average_logistic, delta -1.4076; T=81 / 3,929 / 18,863 / 56,697; mean entry 0.304s
- **200开发观察:** mIoU 49.0343; delta prototype +6.4537 [+1.334, +8.501]; DR_control_average_logistic delta -1.3124 [-4.199, +1.250]; T=97,926 / 642,470 / 1,165,096 / 3,619,936; mean entry 0.282s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> DR08.class_actions_vs_prototype
- **失败解释与所得:** 200对prototype +6.454但低于logistic1.312；源hull间隔证明R可分，不保证Q身份。边界极值和域变化是反例。
- **特有反例:** nearest FG/BG pointpair由boundary混合决定，方向过拟合该pair；highdim reference容易分离但Q改变appearance后仍错；两hull重叠没有margin方向。

### inv_adversarial_channel_support - Adversarial deletion of concentrated positive feature evidence

- **逻辑/条件性预期:** For c_d=q_d(vFG_d−vBG_d), the exact worst margin under deletion of at most k positive coordinate contributions is sum(c)−sum(top_k(max(c,0))). This is q-dependent and can reverse concentrated false support; it is not a global threshold or normalization-cancelled scalar gate.
- **预期收益（假设）:** False-positive deletion when non-target positive support is concentrated, while target support is spread across more than k coordinates.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 63918543831c4c8a1db98033805a895559fcf77df5d07c2662d558bc3f22773a; runtime config（本地证据，未纳入仓库）.
- **强control:** Exact direct-prototype margin; same decoder; worst k-coordinate deletion versus matched constant erosion to distinguish q-dependent support from merely stricter threshold.
- **4例描述性观察:** mIoU 36.5016; delta prototype +7.5173 [-8.948, +23.983]; strongest inv_adversarial_constant, delta +0.5372; T=0 / 0 / 34,250 / 97,253; mean entry 0.415s
- **200开发观察:** mIoU 43.9545; delta prototype +1.3739 [-4.568, +4.144]; inv_adversarial_constant delta -6.7634 [-9.111, -4.250]; T=0 / 0 / 2,260,698 / 4,952,742; mean entry 0.364s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> inv_adversarial_channel_support.class_actions_vs_prototype
- **失败解释与所得:** 200只能删，低于常量erosion6.763；删TP2,260,698不可由鲁棒证书掩盖。坐标特异最坏删除未优于简单拒绝。
- **特有反例:** A real small part identified by only k useful coordinates is necessarily rejected; a distractor with broad correlated support survives. Feature axes are not rotation invariant.

### inv_reference_mad_winsor - Reference-conditioned coordinate winsorization

- **逻辑/条件性预期:** Clipping q and r by the same reference robust median±3 MAD box is nonlinear before prototype construction; vectors outside observed support can change direction. It preserves central amplitudes instead of inverse-variance rescaling all coordinates.
- **预期收益（假设）:** Distractor/query corruption lying outside reference-observed coordinate support while FG retains stable central responses.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 63918543831c4c8a1db98033805a895559fcf77df5d07c2662d558bc3f22773a; runtime config（本地证据，未纳入仓库）.
- **强control:** Direct cosine prototype and reference-centering-only encoding with exactly the same decoder; central data witness where clipping is the identity.
- **4例描述性观察:** mIoU 29.7572; delta prototype +0.7730 [-0.001, +1.547]; strongest dino_prototype.control, delta +0.7730; T=1 / 17 / 47 / 3,715; mean entry 1.515s
- **200开发观察:** mIoU 42.8000; delta prototype +0.2193 [-0.593, +0.723]; dino_prototype.control delta +0.2193 [-0.593, +0.723]; T=5,149 / 145,943 / 129,034 / 420,373; mean entry 1.451s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> inv_reference_mad_winsor.class_actions_vs_prototype
- **失败解释与所得:** 200 +0.219区间跨0；合法Q语义尾部可被剪。不同于白化不意味着有效。
- **特有反例:** A target viewpoint creates legitimate coordinate responses outside reference support, so clipping removes the discriminating direction; rotation can change behavior.

### inv_multiscale_feature_consensus - Spatial scale consensus after feature pooling

- **逻辑/条件性预期:** unit(mean(features)) matched to pooled reference prototypes is generally not equal to mean(direct token margins); different scale information comes from observable local coherent direction before matching.
- **预期收益（假设）:** A true region with coherent semantic support and token noise; scale-fragile isolated peaks are suppressed.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Pool/upsample scalar direct margins at identical scales then median; no-pooling direct prototypes. Both share spatial scales and decoder.
- **4例描述性观察:** mIoU 29.9770; delta prototype +0.9927 [-1.168, +3.153]; strongest inv_multiscale_score_pool, delta -0.6622; T=522 / 4,887 / 2,330 / 11,854; mean entry 0.702s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> inv_multiscale_feature_consensus.class_actions_vs_prototype
- **失败解释与所得:** 4例不胜同尺度score pooling；不可把平滑增益归给feature consensus，小目标可能被稀释。
- **特有反例:** A small true target occupies less than one pooling cell, so consensus deletes it; a large coherent non-target matching reference remains selected.

### inv_spatial_geomedian - Geometric-median filtering of cached semantic vectors

- **逻辑/条件性预期:** The exact geometric median is orthogonally equivariant and can differ from feature means and scalar score medians because vector distances, rather than scalar ordering, set influence. Eight fixed Weiszfeld iterations approximate this estimator; no exact breakdown guarantee is claimed for the truncated solver. A tightly clustered negative-margin minority can dominate over a dispersed positive-margin majority. If a fixed prototype gives ALL neighbors the same positive scalar margin, convexity forbids sign reversal; this is explicitly not a witness.
- **预期收益（假设）:** Isolated vector contamination occupying a minority of a locally coherent semantic region.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 3×3 feature mean followed by unit matching; 3×3 scalar median of direct score. Same neighborhood and full decoder.
- **4例描述性观察:** mIoU 29.1834; delta prototype +0.1991 [-0.420, +0.818]; strongest inv_spatial_score_median, delta -0.3976; T=428 / 3,737 / 1,096 / 6,011; mean entry 2.324s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> inv_spatial_geomedian.class_actions_vs_prototype
- **失败解释与所得:** 4例不胜scalar median；同正邻居在fixed-v凸包能反号的错误理论已撤回。复杂向量robust读出必要性未立。
- **特有反例:** Thin target objects or true boundaries are a minority of a 3×3 neighborhood and are erased; robust filtering also reinforces a coherent wrong-class region.

### inv_local_affine_reconstruction - Leave-center-out local affine semantic reconstruction

- **逻辑/条件性预期:** A least-squares fit in local (x,y) reconstructs affine coordinates at a missing center, including one-sided borders. A nonconstant whole unit-vector field is generally not strictly affine, so the model on actual unit DINO inputs is a local smooth approximation. A reachable unit witness can have an affine discriminant coordinate plus a nonlinear norm-completing coordinate; this is enough to test signed classification without claiming full-vector exactness.
- **预期收益（假设）:** Sparse center contamination on locally smooth directional gradients, especially one-sided image neighborhoods.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Leave-center-out uniform vector average; affine fit including center; same native grid and decoder.
- **4例描述性观察:** mIoU 30.2325; delta prototype +1.2482 [-2.406, +4.903]; strongest inv_affine_mean, delta -0.0718; T=618 / 8,462 / 5,001 / 18,855; mean entry 1.094s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> inv_local_affine_reconstruction.class_actions_vs_prototype
- **失败解释与所得:** 4例不胜局部mean；精确affine再现定理不能套任意unit场。平滑保真不保证身份。
- **特有反例:** A real feature discontinuity or texture center contains class information not predictable from neighbors; reconstruction deletes rare parts. It can hallucinate across FG/BG boundaries.

### inv_local_lowrank_reconstruction - Leave-center-out local low-rank semantic reconstruction

- **逻辑/条件性预期:** Fit a local affine feature subspace from center-excluded neighbor vectors and project the center onto its leading directions; removing only orthogonal residual preserves local semantic variance rather than flattening covariance.
- **预期收益（假设）:** Feature corruption perpendicular to a low-dimensional local semantic manifold while genuine local variation lies in its principal subspace.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Same neighborhood mean reconstruction (rank zero) and original token; same-information two-direction reconstruction using center included checks contamination leakage.
- **4例描述性观察:** mIoU 29.2787; delta prototype +0.2945 [-0.016, +0.605]; strongest inv_lowrank_mean, delta -0.1148; T=178 / 1,371 / 256 / 3,187; mean entry 1.436s
- **200:** 未在该12项固定screen内完成；unknown，不借其他方法分数填入。
- **证据:** [4 per-class actions / exact gain](server/probe29_native4_v2/methods_summary.json) -> inv_local_lowrank_reconstruction.class_actions_vs_prototype
- **失败解释与所得:** 4例不胜mean；罕见语义残差可被丢。保高方差不同于白化但不保证收益。
- **特有反例:** A true target’s distinguishing part is the rare orthogonal residual and is deleted; a wrong coherent region remains a well-fit manifold. Rank-two assumption is fixed and unverified.

### inv_huber_reference_readout - Huber robust episode-local linear reference readout

- **逻辑/条件性预期:** Huber residual loss gives bounded influence to reference label-prediction errors rather than squared-loss leverage. An explicit fixed residual model can alter the learned direction without external training; whether atypical parts are nuisance is falsifiable.
- **预期收益（假设）:** A minority atypical reference-part residuals misdirecting squared-loss fit; no predicted population gain.
- **source/合法输入:** [source](../../../src/ics/cpu100/invariance_support.py) SHA 63918543831c4c8a; native4 frozen source（本地证据，未纳入仓库） SHA 63918543831c4c8a; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **native200 frozen binding:** module SHA 63918543831c4c8a1db98033805a895559fcf77df5d07c2662d558bc3f22773a; runtime config（本地证据，未纳入仓库）.
- **强control:** Role-balanced ridge with exact same fit subset, lambda, full query input and decoder; direct weighted prototypes.
- **4例描述性观察:** mIoU 40.4669; delta prototype +11.4826 [-2.755, +25.720]; strongest inv_huber_ridge, delta -0.0967; T=4 / 120 / 11,949 / 59,043; mean entry 0.648s
- **200开发观察:** mIoU 52.3734; delta prototype +9.7928 [+3.387, +10.996]; inv_huber_ridge delta +0.7109 [+0.135, +1.001]; T=25,300 / 264,781 / 1,137,142 / 3,753,092; mean entry 0.606s
- **证据:** [200 per-class actions / I-U / exact gain](server/screen12_native200_v3/methods_summary.json) -> inv_huber_reference_readout.class_actions_vs_prototype
- **失败解释与所得:** 200对ridge实测+0.711且区间正，开发边际保留；对prototype的大头由ridge解释。加TP25300但删TP1137142，主要是拒绝/删错区，不是补目标已解决。独立确认与同协议FoRIS未知。
- **特有反例:** Rare but genuine foreground parts cause high residuals; Huber downweights precisely the information required for viewpoint change, making missed-target recovery worse.

## 旧19个CPU方法：保留原合同和原失败

Pro原文也需保留：3.1要求FP32原生特征；3.2允许M1/M2/M3/M5复用FoRIS连续证据，M1/M5使用固定Pi，M2明确末层去位置特征。因此“缓存经过处理”不能单独解释五项全部失败；旧FP16适配也不等于原文完整FP32合同。原文见 Pro五方法（用户本地来源，未纳入仓库）。

existing19 source/CLI/assets/version index（本地证据，未纳入仓库）

### old adjacency

- **逻辑/条件性预期:** Reference mode adjacency / region tree; transfer local role relation. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_adjacency.py); processed DINO q/r, reference coverage, complete MEAN/base field; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=18.8774; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; adjacency_bilinear.control=18.6693; mean_nearest.control=61.7712
- **四类动作T:** 10,517,512 / 262,759,656 / 513,881 / 734,117; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module c4add33980992ea4aeefca3d714fce86d32f40cfcb51039c1a822043c63e989b; current module match=True, runner match=False
- **失败解释与所得:** Reference mode-pair/region-tree structure does not validate query identity; severe negative complete output retained. Do not infer all structural algorithms impossible.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old huber

- **逻辑/条件性预期:** Bounded Huber edge influence on the matched MEAN graph/fidelity. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/huber_graph.py); MEAN target/fidelity and DINO graph; preprocessing uses CPU torch, optimizer NumPy/SciPy; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=62.5274; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; boxed_quadratic.control=62.9318
- **四类动作T:** 1,888,692 / 2,206,602 / 1,311,829 / 3,535,436; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module d9c0a3298cab8990c6ebb550989cf5f0f491f50239865c7fe900fe443122d7c9; current module match=True, runner match=False
- **失败解释与所得:** Bounded edge influence loses to the matched quadratic/MEAN control; native gain belongs to the full host pipeline, not incremental Huber benefit.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old color_bottleneck

- **逻辑/条件性预期:** DINO-seeded RGB minimum-bottleneck basins; no new identity proof. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/color_bottleneck.py); query RGB and complete frozen DINO-derived base field; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=62.8348; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** color_same_resize.control=62.9739; mean.control=62.9315
- **四类动作T:** 1,811,117 / 2,028,749 / 2,809,998 / 5,453,763; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 7f5b644ffd8a0a3df36438c217eded2b40b47d866ccd3182a12a812f86a196ad; current module match=True, runner match=False
- **失败解释与所得:** Color-basin propagation did not beat same-resize base; topology/coherence cannot by itself certify wrong seeds. Separate resize from color effect.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old constellation

- **逻辑/条件性预期:** Reference part layout / pose consistency / transferred support. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_constellation.py); processed DINO q/r, reference coverage, MEAN/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=61.7126; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; constellation_bag.control=39.1860
- **四类动作T:** 1,939,620 / 3,296,862 / 2,243,189 / 4,177,979; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 93e3fe70bad2dcfb80542822626cb94484d653a2097561645f015b4ec5edeb36; current module match=True, runner match=False
- **失败解释与所得:** Rigid reference part layout and warped support harm complete output versus MEAN; local v2 is a repair of this same method, not another delivery count.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_shape

- **逻辑/条件性预期:** Reference silhouette-compatible region proposals and bounded edits. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_shape.py); processed DINO q/r, reference coverage, MEAN/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=57.1310; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; shape_bilinear.control=57.4909
- **四类动作T:** 1,876,777 / 2,560,584 / 9,044,701 / 7,229,346; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 9789ac14e24f319a6465f98fa062239ae77681da8e00c9529373d1c20ff0544d; current module match=True, runner match=False
- **失败解释与所得:** Single-reference shape compatibility over query proposals did not transfer reliably; generic-square/bilinear controls are retained, not independent methods.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_covariance

- **逻辑/条件性预期:** Joint reference-mode response covariance of query regions. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_covariance.py); processed DINO q/r, reference coverage, MEAN/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=12.3432; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; covariance_trace.control=45.9960
- **四类动作T:** 351,741 / 482,420 / 48,011,183 / 19,517,136; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 97a449af59e10f01de06f3bb1649789ea9e83bc65d9a9a2b354692d423d747f6; current module match=True, runner match=False
- **失败解释与所得:** Full role-response covariance strongly harms complete identity/extent selection; trace control is far stronger. Algebraic reachable witnesses do not establish natural transfer.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old query_recurrence

- **逻辑/条件性预期:** Repeated selected query appearance produces new feature prototypes. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/query_recurrence.py); processed DINO q/r and MEAN/base seeds; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=62.8940; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; recurrence_all_seed.control=62.0456
- **四类动作T:** 2,248,658 / 2,431,977 / 1,369,477 / 3,913,358; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module d9951558395346a476022d2926541e999f3477f7e0630a5832062957e8c929d8; current module match=True, runner match=False
- **失败解释与所得:** Repeated appearance around selected seeds did not yield an established increment over MEAN; shared wrong identity can recur too.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_prior_shift

- **逻辑/条件性预期:** Reference class-score density mixture estimates query prior/posterior. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_prior_shift.py); processed DINO q/r, reference coverage and MEAN/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=37.8833; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; prior_balanced.control=47.6945; prior_margin.control=43.0823
- **四类动作T:** 2,589,088 / 12,914,504 / 20,857,763 / 12,860,378; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 1edbe760124f26b8f790b2991f2bd3896f14ef0f99573dd981171cfd2a567fd2; current module match=True, runner match=False
- **失败解释与所得:** Reference-conditioned mixture/prior adaptation is negative; unlabeled query composition is not target size, and model assumptions must not be confused with extra identity evidence.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_quadratic

- **逻辑/条件性预期:** Role-balanced directional nonhomogeneous quadratic kernel regression. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_quadratic.py); processed DINO q/r, reference coverage and MEAN/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=51.6070; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; quadratic_linear.control=49.1334; quadratic_subspace.control=45.6550
- **四类动作T:** 3,944,014 / 15,647,423 / 7,855,492 / 7,693,177; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json) SHA 506acfaa3e88a220; scored module 88ef47cb8950eb55ed16024c5addb5991f464885d9a9e703c3c629546d282361; current module match=True, runner match=False
- **失败解释与所得:** A directional nonlinear kernel can change decisions yet fail real transfer. Gaussian v2 and polynomial/subspace arms are family revisions/controls, not additional counts.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_hull

- **逻辑/条件性预期:** Reference class convex-support distances and residual readout. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_hull.py); processed DINO q/r, reference coverage and MEAN/base; certified class-hull distances; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=48.9153; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; hull_affine.control=47.9552; hull_subspace.control=45.6553
- **四类动作T:** 2,999,616 / 14,521,094 / 10,596,085 / 9,616,858; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/extensions_public600_v2/score/report.json) SHA 2743a7a0c0154fea; scored module 77fd572766a6caa4f0d92df89a9dd5ef91da23374fc3702ffba54dc0e3e9c0bb; current module match=True, runner match=False
- **失败解释与所得:** Reference convex-support fitting can be numerically sound while deleting true cross-image targets; affine/span controls already exist and may not be renamed as new methods.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old reference_triplet_relations

- **逻辑/条件性预期:** Third-order role relation on the same pair host. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_triplet_relations.py); processed DINO q/r, reference coverage and same relation-host unary; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=60.7790; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** triplet_zero.control=60.7736; triplet_no_third.control=60.7736; mean.control=62.9315
- **四类动作T:** 1,115,152 / 2,119,762 / 930,455 / 1,201,383; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/extensions_public600_v2/score/report.json) SHA 2743a7a0c0154fea; scored module 122db24b1e03e4e24e3b48a97d9f33f83b58cc1fb2d860e7769ec8196bf88a7b; current module match=True, runner match=False
- **失败解释与所得:** Increment over its own zero/no-third control is only0.005356 with interval crossing zero; do not attribute the full MEAN difference to the triplet term.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

- **Version / attribution:** 对其自己的zero/no-third增量仅+0.005356且区间跨0；对完整MEAN的差不得全部归给三体项。

### old reference_absorption

- **逻辑/条件性预期:** Reference-anchor first-hit / harmonic graph absorption. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_absorption.py); processed DINO q/r, reference labels, MEAN/base and union feature graph; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=37.8215; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** mean.control=62.9315; absorption_nearest.control=51.8913; absorption_one_step.control=61.5380
- **四类动作T:** 2,326,479 / 51,397,993 / 15,628,756 / 10,123,724; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/next_candidates_public600_v1/score/report.json) SHA ea8df24d6ac2934d; scored module 3876f1d3c834aafc00e42373cdeca1e97852d96ba8d6b20da71b700d5475c57e; current module match=True, runner match=True
- **失败解释与所得:** First-hit/harmonic reference absorption plus residual has a severe complete-output failure; relation/graph solve convergence is not semantic success.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old ProM1

- **逻辑/条件性预期:** Native H20/KV common query context and declared FoRIS host/Pi. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/pro_common_context.py); real frozen FP32 native H20/KV suffix states plus declared cached host score/Pi; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=4, mIoU=25.1128; work1024 only; not original-size GT score; reused development/smoke, not independent confirmation.
- **强control:** uniform.control=35.1353; plain.control=32.4101; native=36.3165
- **四类动作T:** 16,348 / 283,309 / 52,493 / 131,898; baseline old native work1024
- **证据:** [historical result](../research_20261006/pro_context_preparation_01a1100b/score4_v1/report.json) SHA 92cf873aa2b407ec; scored module 0150904d669fc613d9ed37f05e583e552149ff1697e2d5858c4cf54ee8fcece4; current module match=True, runner match=False
- **失败解释与所得:** Native self-audit and graph residual pass, but ctx loses to uniform/plain/native. Numerical parity is not context identity benefit; small exposed batch does not disprove all context methods.
- **CPU:** 4 exposed pairs,801.550s including load at4 CPU threads; peak sampled RSS3.244GB; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old ProM2

- **逻辑/条件性预期:** Signed four-state reference role-pair potentials. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/pro_reference_relations.py); processed DINO role-pair four-state potentials and declared cached seed/unary contract; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=60.6325; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** pro_relations_zero.control=60.7736; mean.control=62.9315
- **四类动作T:** 1,014,861 / 1,873,581 / 1,203,375 / 1,435,604; baseline old native work1024
- **证据:** [historical result](../research_20261006/server_prepared_01a1100b/extensions_public600_v2/score/report.json) SHA 2743a7a0c0154fea; scored module 27c867bd814a986e178696b07a6fe1733aa7f479b47ded83b747db5b81be7c74; current module match=True, runner match=False
- **失败解释与所得:** Signed role-pair relations do not beat zero/control or MEAN; optimizing accumulation cannot repair semantic evidence.
- **CPU:** same-output optimization14.889s to2.938s per exposed case is CPU speed only, not quality gain; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old ProM3

- **逻辑/条件性预期:** Reference role prediction with query hierarchy and explicit native fallback. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/pro_role_prediction.py); processed cached DINO, known MR, query hierarchy, declared score/native fallback; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=600, mIoU=59.5635; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** same-tree zero=60.8783; all-role=59.9293; stored MEAN=62.9315
- **四类动作T:** 2,660,917 / 4,886,375 / 3,989,232 / 6,737,408; baseline old native work1024
- **证据:** [historical result](../research_20261006/pro_roles_preparation_01a1100b/public600_result_46466.md) SHA 8929b5d214a86a63; scored module cfdd881339202fa52205138e82afb26c4bd30af23924735d39a182f9ff1e1957; current module match=True, runner match=True
- **失败解释与所得:** Held-out role-prediction cache adapter loses to same-tree zero and MEAN. This is the fixed adapter/version result, not a strict original FP32 full-pipeline universal failure.
- **CPU:** historical full-backend/shared-control costs; see linked report, no new timing measured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

### old ProM4

- **逻辑/条件性预期:** Paired real-RGB environmental interventions and covariance-suppressed direction. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/pro_paired_environment.py); real reference/query RGB, complete MR, frozen CPU encoder and explicitly bound native fallback; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=4, mIoU=22.1424; original size; reused development/smoke, not independent confirmation.
- **强control:** class-LDA=25.7565; ref-canvas=31.2604; native variant=36.3608
- **四类动作T:** 3,837 / 82,091 / 69 / 51,472; baseline cached native original renderer variant
- **证据:** [historical result](../research_20261006/pro_environment_preparation_01a1100b/real_smoke4_46466/report.json) SHA cab83e023113197e; scored module None; current module match=None, runner match=None
- **失败解释与所得:** Paired environment intervention loses to class-LDA/ref-canvas controls; true RGB intervention and exact cut checks do not establish beneficial identity transfer. Four cases are descriptive.
- **CPU:** 17 encoder calls per case,363–405s per case,approximately3.13GB peak; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

- **Version / attribution:** 本地score/seal没有确切历史module/runner源码SHA，只知道remote code_ce及预测seal；不能拿input source-manifest SHA冒充code身份。

### old ProM5

- **逻辑/条件性预期:** Native H20/QKV local-message extrapolation plus beta1 audit. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/pro_message_extrapolation.py); CPU native H20/QKV cache, bound model/position basis, declared processed host features/base; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=4, mIoU=15.8059; original size; reused development/smoke, not independent confirmation.
- **强control:** endpoint=24.8509; native-region=31.0149; MEAN=35.6952
- **四类动作T:** 3,633 / 187,806 / 53 / 41,249; baseline old same-renderer MEAN original
- **证据:** [historical result](../research_20261006/pro_message_preparation_01a1100b/actual_smoke4_v1.json) SHA 7286adf8a3f1ec73; scored module b9da3fd3b42c08a27c3634d86251b36e75269ff5414105d8408566dabe4c1653; current module match=True, runner match=None
- **失败解释与所得:** All69 ROIx4 block beta1 native errors0 after tail repair, yet output markedly negative. Do not blame the fixed numeric bug or claim all attention-message mechanisms impossible.
- **CPU:** three later worker cases group244.30s at3x4CPU,group peak approximately8.50GiB; per-case/full suffix cost in receipt; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

- **Version / attribution:** 四例69 ROI×4 block的beta1数值误差均0后质量仍负，不能再怪已修尾块bug。原尺寸加入3633真像素却187806假像素，主要副作用已明确；不由四例否定所有message机制。

### old grayscaled_dino

- **逻辑/条件性预期:** Physical RGB/gray frozen DINO128 interventions with processed MEAN/guide host. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/grayscaled_dino.py); real RGB128/gray frozen FP32 native forwards plus explicitly processed MEAN base/guide; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=4, mIoU=36.9797; original size; reused development/smoke, not independent confirmation.
- **强control:** RGB=37.2051; brightness=37.5055; H20=37.7852; MEAN=35.6952
- **四类动作T:** 120 / 225 / 42 / 5,246; baseline old same-first4 MEAN original
- **证据:** [historical result](../research_20261006/grayscaled_dino_preparation_01a1100b/actual_first4/report.json) SHA 987f143117bff4fc; scored module 6cf47db241f9bd8a47dcc9075a5f6c27c2a80007c1b10dc13df4e36993981eb4; current module match=False, runner match=True
- **失败解释与所得:** Gray beats that MEAN but loses to matching RGB, brightness and H20 controls; chromatic collapse necessity not established. Source current hash differs from scored first4: deliver frozen version or label changed version unmeasured.
- **CPU:** 16 real128 forwards25.368s at2CPU,approximately3.131GB; H20 extraction separately8.862s; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

- **Version / attribution:** 历史36.979714仅绑定6cf47db241f9bd8a47dcc9075a5f6c27c2a80007c1b10dc13df4e36993981eb4；当前不同gray源码没有继承该质量证据。16个真实128前向25.368s，已有H20提取另8.862s；原授权600和fresh parity未运行。

### old reference_texture

- **逻辑/条件性预期:** Known-R pure RGB windows radial power fractions plus MEAN residual. Conditional expected benefit only, no new numerical prediction.
- **source/合法输入:** [source](../../../src/ics/methods/reference_texture.py); actual hash-bound RGB, complete MR, declared DINO-derived MEAN/base; CPU NumPy/Pillow/torch renderer; CLI/dependencies/asset binding（本地证据，未纳入仓库）
- **真实质量:** historical n=0, mIoU=unknown; work1024 cached comparison, not original-size or new native1024 result; reused development/smoke, not independent confirmation.
- **强control:** same-window color/variance/dominant-band+energy; dominant band also solves synthetic witness; no natural run
- **四类动作T:** unknown, not zero
- **证据:** [historical result](../research_20261006/reference_texture_01a1100b/README.md) SHA d326288554f90bd7; scored module 779425b5faf11576d3401b88cd3d26f567831fab166c865d59d72c713c55d031; current module match=True, runner match=True
- **失败解释与所得:** Radial-power complete algorithm is runnable but real smoke never ran. Dominant-band+energy control solves its synthetic positive too, so multi-band necessity and natural quality remain unestablished.
- **CPU:** bounded local physical-RGB checks; natural end-to-end time unmeasured; current complete version quality remains unmeasured unless its exact historical full contract is reproduced.

## 二批新增审查资格：ready不等于真实收益

以下22项为二批新增资格；local_004已经撤回并移至关闭记录。local_002在首30保留v0观察，后续修订独立计0，能量/τ口径变动不能默默继承v0分数。root的batch02_launch_selection.json确认23请求项=22新增+local_002，未包含local_004。真实二批质量报告尚待完整收回；缺控制或pending结果不能证明稀疏机制收益。

actual selection（本地证据，未纳入仓库） SHA 903d7156b3f7ca5a9cd17713ea12fb86d83a771142477ed982ac8f92fef73d4d

以下22项为二批新增资格；local_004已经撤回并移至关闭记录。local_002在首30保留v0观察，后续修订独立计0，能量/τ口径变动不能默默继承v0分数。root的batch02_launch_selection.json确认23请求项=22新增+local_002，未包含local_004。真实二批质量报告尚待完整收回；缺控制或pending结果不能证明稀疏机制收益。

### ready cross_image_joint_sparse_role_removal - 联合参考稀疏解释的角色去除代价

- **逻辑/条件性预期:** 先用混合字典共同解释q，再去掉FG或BG贡献，两个残差衡量同一个解释对各role的依赖；不是两个class各自随意换解释。混合可以使一个单role最近点胜利变为另一个role解释占主导，但系数竞争是否对应语义尚未知。
- **预期收益（假设）:** 若目标贡献必须与背景联合解释才能显现，恢复目标；若干扰只能借多类字典拼接，可由角色去除改变身份
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 相同atoms下独立FG/BG非负OMP3残差；same OMP coefficient summed-role vote；class cone/full convex hull；prototype和nearest。同signed margin common render。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** small NNLS independent audit; same-budget independent OMP, dense cone/hull and coefficient-vote controls; reconstruction not identity
- **特有反例:** 字典高度相干时alpha不唯一，贪婪选错先手；真实目标需>3原子，或背景原子替代FG导致误删。重构好不保证语义对。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### ready cross_image_exemplar_facility_cover - 全查询共享参考解释的设施选择

- **逻辑/条件性预期:** 联合总代价sum_i w_i min_open(1-q_i dot r_a)+0.05*number_open，使低支持的偶发原子不值得单独打开，而多个相似query能共付解释成本。绝非新语义证据，依赖重复解释比孤立误匹配更可信的条件。total valid query mass归一化，因此数量增加代表面积影响。
- **预期收益（假设）:** 关闭只服务孤立错误query的FG解释，同时允许同类多实例共同打开参考解释
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same dictionary全部开放nearest；相同开放个数的global unary排序和随机/固定ID选择；same source prototype。完整joint objective值只是求解账，不能当质量。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** state complexity/area prior, matched-open-count controls and tiny-target deletion negative; small subset optimum audit
- **特有反例:** 正确目标很小只需一个独特原子，正好被开放费删除；重复错误区域成为便宜FGhub，会被保护；source多个target外观需要很多open atoms而BG简单。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### ready cross_image_nonreturn_path_consensus - 跨图无立即回退路径的角色共识

- **逻辑/条件性预期:** 输出3-hop Q_i→R_a→Q_j→R_b的末端role平均，只保留j≠i、b≠a。这是directed edge-history状态而非token harmonic固定点；禁返回可使单个互为NN不再凭自循环获得共识。但高连边错误背景仍可提供多条独立路径，须负例。
- **预期收益（假设）:** 抑制互为NN单边回声，把身份依据转到不重复的参考原子支持
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same dictionary、sameW、same3hop长度但允许回退的ordinary walk；1-hop source kernel；wrong high-connectivity background桥接负例。不能与成本/输入不同的old base score仅比总分。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same-W ordinary walk/one-hop and direct enumerator; repeated false bridges may still be coherent, paths not independent samples
- **特有反例:** 错误类别在Q内部高连接并匹配多个FGatoms，则禁返回仍强化错误身份；单实例/少Rmode没有independent path会fallback；真实唯一part对应可能被排除。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### ready cross_image_source_opponent_matching - 参考角色全局对手匹配的中位判决

- **逻辑/条件性预期:** 在R内作最小成本FG/BG一对一匹配，再对每pair的unit difference方向取query margins中位数。normalization和非线性中位读出使不同配对改变决策；必须用all-pairs和independent-nearest opponent同中位控制判断global matching增量。R端均匀mode投票是先验，不能称semantic保证。
- **预期收益（假设）:** 避免单个参考BG对手被无限复用支配比较，在多局部决策方向存在时改变身份
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 相同K字典的独立nearest-opponent median/allCartesian median/unit-globalmean；costmatching只在R上，Qrender和信息完全一样。matched randompermutation median为控制，非多方法。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** nearest/allpairs/random opponent same median; K1 activity and source-matching cost distinct from query quality
- **特有反例:** 独特BGatom被迫配给无关FG，产生不代表真实边界的方向；target参考模态数量少会K=1退化；unit小差方向放大source noise；多数局部pair本身错会中位失效。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### ready local_003 - Foreground/background regularized convex patch reconstruction

- **逻辑/条件性预期:** The ordered nine-patch centered DINO tensor is an available continuous joint observation. Independent-token matching or Gram invariants discard its coordinate directions and ordering. A constrained convex mixture of reference tensors can represent intermediate target local patterns without requiring one nearest template or transferring an entire silhouette. This changes the decision, but whether its reference target span is sufficiently distinctive is unknown. The L2 objective does not guarantee sparse coefficients; it is called regularized convex patch reconstruction. Different ordered patch evidence, rather than the penalty, is the proposed counting distinction from central reference_hull.
- **预期收益（假设）:** The normalized mixture of distinct unit target patch tensors lies outside their convex hull in general, but can still have a smaller nonzero regularized reconstruction residual than the background hull while being farther from every individual target tensor. Ordered raw patch observation is the proposed extra information; L2/cap are not claimed sparse, independently new, or a real gain.
- **source/合法输入:** [source](../../../src/ics/cpu100/local_structure.py) SHA 63b1cfe14ecd5a3e; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Same nine-patch tensors, dictionaries and renderer: nearest individual-template reconstruction error per class.; Same constrained convex reconstruction on central DINO token only (old central-feature hull-type control).; Complete local001 Gram classifier on same legal inputs/renderer, keeping its distinct invariance limitation.; Same class tensors with spatial stencil order shuffled independently in reference; bag contents remain but organization is removed.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** no sparse guarantee from L2; unit mixture is not exact convex combination, use nonzero residual superiority; central hull/nearest ordered patch/complete Gram controls
- **特有反例:** If FG and BG patch cones overlap, reconstruction cannot determine identity.; Large dictionaries can reconstruct both classes and erase discriminative residual; the fixed cap is not a proof of separability.; Viewpoint or feature-coordinate changes can move true target patches outside the reference cone.; Convex mixtures may create nonphysical local composites and accept a distractor.; A center near target boundary uses surrounding reference BG; altered query context can cause target deletion.
- **证据:** [card](cards/local_structure.json); [batch02](reviews/batch02.json)

### ready DR09 - 参考原型相似度弱决策的AdaBoost

- **逻辑/条件性预期:** boosting按reference错误重加权组合多个不同anchor的cosine stump，输出sum alpha*h；不是在一个score扫描阈值，也不把source分类精度当query身份保证。
- **预期收益（假设）:** 当合法MR描述非线性多模态而单决策错误可被另一anchor纠正，boosted分界可救回目标/删错背景；仅hypothesized。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同anchor最佳单stump、same-input nearestreference、sameanchor linear ridge、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** single stump/sameanchor ridge/NN; source-error focus can amplify nontransferable rare material
- **特有反例:** 少数nontransferable reference材质成为高权重error，boosting会追逐它；source100%分类正确仍可能query全错。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 源码已核：bank为空/无信息时回退pure-reference nearest-neighbor；不是泛称prototype。该fallback不算boosting生效；真实二批质量仍unknown。

### ready DR10 - 平衡参考重采样的最大间隔委员会

- **逻辑/条件性预期:** 16个class-balanced参考bootstrap分别求两class hull间隔，Q按hyperplane sign majority；它平均支持向量敏感性，不提供独立证据或IID概率保证。
- **预期收益（假设）:** 若support-vector异常出现率低于委员会多数，可能避免单hull误删/误增；稀有合法FG也可能被bootstrap丢失。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 全reference DR08、同样16bootstrap的prototype voting、DR04 trimmed risk、raw NN。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same-bootstrap prototype voting, fullDR08; members are not independent confidence evidence
- **特有反例:** 所有source成员共享错identity时稳定全错；真正稀有的referenceFG多数bootstrap未见，会被抹掉。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### ready DR12 - 参考原型相似坐标的固定小ReLU判别

- **逻辑/条件性预期:** 固定16anchors cosine坐标上的16-hidden ReLU分类器，可表达若干局部线性条件的联合；MR fit只是available computation，不是新外部model或成功承诺。
- **预期收益（假设）:** 多模态reference逻辑延续到Q时，多个局部平面可能减少mean方向冲突；比NN/核方法更好的收益未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同anchor linear logistic/ridge、DR09 boosted stumps、nearestreference、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** disclose200-step temporary classifier parameter optimization; frozenDINO/no external data remains, cannot call it no fitting; sameanchor linear/boosting controls
- **特有反例:** fixed训练预算可未收敛；source可分但Q不在R模式中；网络可以记住MR context而不识别目标类别。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### ready DR13 - 已知参考角色的近邻冲突编辑

- **逻辑/条件性预期:** reference labels已知时，3NN leave-one-out角色冲突可识别孤立class-support；编辑ref exemplars后再完整Q NN，并非在query错误种子上过滤后传播。
- **预期收益（假设）:** 孤立nontransferable FG/BG表征在reference多数邻域中冲突时，删掉它可减少nearest-match错误；hypothesized。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同pure samples unfiltered NN、仅coverage-purity过滤NN、DR06pair rank、prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** unfiltered/purity-only NN; rare legal target may be removed; class disappears fallback explicit
- **特有反例:** 真正稀有合法FG appearance也会被多数BG邻域排除；同类孤立并不等于污染。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### ready DR14 - 覆盖参考分类约束的凝聚近邻支撑集

- **逻辑/条件性预期:** class medoid初始化，按knownR misclassified加入exemplar直到cap训练集被支撑集分类正确；这是reference判别覆盖而不是无label kmeans密度压缩。
- **预期收益（假设）:** 冗余reference细节被压缩时有成本收益，某些Q泛化可能减少偶然极值；mIoU收益待证，source覆盖不是Qextent保证。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** unfiltered NN、same-size fixed/random support NN、class kmeans同K、global prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** show actual query decision distinction; identical-output runtime optimization alone is not another inference mechanism; same-size NN controls
- **特有反例:** label-conflicting相同feature没有可满足覆盖；容易保留所有noise点；training order会改变supports与Q，不保证de-noise。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### ready DR15 - 参考类别风险驱动的学习向量量化

- **逻辑/条件性预期:** GLVQ更新class-owned prototypes最小化参考correct-vs-wrong squared-distance relative margin；不是把无label cluster当identity或改querycut。
- **预期收益（假设）:** reference跨类混淆需移动判别原型而非密度中心，可能改善接近类别分界的Q；真实质量unknown。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same2 kmeans不risk更新、nearestreference、DR08hard margin、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same initialization no-update control and finite-difference gradient; small denominator/rare-part negatives
- **特有反例:** 原型移动追随source-specific appearance，损失Q合法外观；小分母造成优化不稳定，gradient cap使固定数值行为需核查。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 已有固定构造反驳原故事：近边界预期正例对最强控制−0.1383，高端预期负例反而+0.0060。有限差分通过只支持导数，不证明收敛或跨图收益；保留反例，不自动开变体。

### ready DR16 - 合法参考角色的近邻大间隔度量

- **逻辑/条件性预期:** 只用knownR role triplets学习低rankPSD使same-role近、different-role远；先有标签约束再改度量，区别无label方差白化。但reference监督并不保证learned axes跨图语义。
- **预期收益（假设）:** 有稳定class-separating reference directions时，label-aware metric可能保留语义而压制source内部noise；跨图效果未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** sameanchor identity metricNN、rawfeature NN、referenceLDA、blind whitening(已负族控制)、DR08。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** identity metric/rawNN/LDA controls; source metric can suppress query semantics, no cross-image guarantee
- **特有反例:** reference classdistortion随query变化；假nuisance轴实际是Q语义被metric抹去；class仅1token没有triplets。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 归因限制：PSD初始矩阵仅前min(8,K)坐标，而identity control用全部K；任何未来metric改善需先区分rank截断与监督更新，当前未有自然质量结论。

### ready QP04 - 参考校准的区域绝对分布相容性门

- **逻辑/条件性预期:** 绝对kernel MMD(q,F)保留q-q项；两个FG/BG核均值差相同的query区域仍可有不同绝对dF。以完整reference FG合法子集间dF定相容门，同时要求相对FG>BG。这不是MMD差分换名，不保证跨实例组成迁移。
- **预期收益（假设）:** 可删相对密度为FG而整体与FG不相容的纯干扰；也可能删真，不预测涨分。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同窗MMD差分，即kernel分类池化; 同窗KDE池化; 逐token绝对FG相容门再同窗平均，隔离q-q作用; 同窗均值/cov距离，同删预算KDE排序
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same-window relative/KDE and pointwise absolute gate; true target composition change negative; heuristic tolerance not confidence bound
- **特有反例:** 视角和部位比例变了，真目标可被门拒绝；参考不完整或小窗容差大时不能删。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### ready QP06 - 完整查询的四叉树最短描述分割

- **逻辑/条件性预期:** 树DP共同决定所有分区与标签，允许多实例；范围先验惩罚分区节点编码复杂度，区别局部边界惩罚。身份仍由reference unary。
- **预期收益（假设）:** 净参考证据明确的连贯块可补弱token/删碎片，多实例不强迫单连通；真实收益未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** samepointwise KDE; same树最细叶无编码代价; 同窗4×4平均KDE; QP05二元Potts
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** sameunary leaf/window/Potts controls; hierarchy prior and thin-target deletion explicit
- **特有反例:** 细长斜边需要许多叶块，编码代价可吞目标；错unary连贯块会整体错选。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### ready QP07 - 查询中心候选超平面的参考风险分割

- **逻辑/条件性预期:** 所有query字典中心差方向是query内合法分离候选；完整R平衡风险选方向/符号，一般不是reference均值差同一排序。
- **预期收益（假设）:** query可分而reference均值多模式抵消时可能身份改判；无迁移保证、不预测分数。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** FG-BG prototype; QP02离散字典命名; reference-only pair方向风险选择（最多64同预算）; pointwise KDE
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** unit center midpoint offset cancels; candidate source risk not query guarantee; reference-pair/KDE/whole-dictionary controls
- **特有反例:** 目标模式环绕背景不能由一个过原点平面分开；跨图标签距离逆转可选错方向；R风险选择虚高。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### ready QP08 - 查询地标响应上的完整参考条件决策树

- **逻辑/条件性预期:** query球面字典的响应向量s(x)=x·Cq提供<=8维共享坐标；以完整R的平衡FG/BG覆盖在这些轴上贪心最小化Gini可得到非线性条件身份规则。它可在多模式环绕BG构造中改变单平面判决，是真正不同推断；query地标是否比reference地标有益须对照，不默认。
- **预期收益（假设）:** 多模式R/Q支持合法非线性分离时可完整恢复两侧FG；不预测真实涨分。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同Q响应FG/BG均值线性判别; same CART但reference-only同预算球面字典; same Q响应单层stump; pointwise KDE、QP02及QP07同readout
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** cross-slot ref_conditional_tree and sameCART reference-landmark controls; no further landmark/basis variants counted
- **特有反例:** reference少样本轴阈值偶然分开两类而Q关系逆转则整块错；类条件响应强域偏移；query未出现的FG模式会被背景吸收。
- **证据:** [card](cards/query_partition_batch3.json); [batch02](reviews/batch02.json)

### ready ref_conditional_tree - 参考角色的条件阈值树

- **逻辑/条件性预期:** A depth3 binary tree implements conditional, multi-threshold axis regions that a single stump, prototype or fixed three-coordinate median histogram cannot always express. A complete unit-feature conditional-boundary fixture is the first real evidence; no natural transfer benefit is assumed.
- **预期收益（假设）:** hypothesized/synthetic_witness pending: recover FG modes needing conditional source boundaries without forcing an EM query seed. Natural signature transfer unknown; no numerical prediction.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Common raw FG cosine/prototype; same-selected-channel depth1 stump; first-batch three-channel code with same render; reference degree2 readout if available. Source copy fit alone is insufficient: evaluate different query feature values governed by the same fixed partition.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same-selected source/branch budget stump/code and held-out feature values; source fit not domain transfer
- **特有反例:** Cross-image coordinate warp crosses learned cuts. A diagonal or curved decision boundary requiring more than eight rectangles cannot be represented at fixed depth. Rare intra-class modes can be omitted by fixed spatial sampling; stratifying known roles prevents dropping a tiny known FG class entirely but does not guarantee its modes survive.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### ready ref_role_support_box - 角色轴向支持盒的最大违约判别

- **逻辑/条件性预期:** A class coordinate box admits independent combinations and measures worst range violation. It differs from convex mixtures, average prototype distance and a metric density. A box can include a target outside the source convex hull, but can also admit a false impossible combination. Both are decisive constructed cases, not semantics guarantees.
- **预期收益（假设）:** hypothesized: reject a high-mean-similarity distractor with one class-specific impossible coordinate and tolerate independent within-role variation. Worst-coordinate nuisance shift may instead delete true targets. No predicted mIoU.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine/prototype; same intervals with mean violation and squared Euclidean box distance; exact convex-hull distance on the same small witness. Quantile choices and aggregation controls count0.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** same intervals mean/squared-box controls, impossible combinations and broad-BG/one-coordinate-nuisance negatives;5/95 variants0
- **特有反例:** One coordinate carrying image-specific nuisance can dominate all other correct coordinates. Independent boxes include feature combinations never produced by the target. Different rotation changes the support set although pair distances are identical.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### ready ref_distribution_energy - 无带宽的类分布能量评分

- **逻辑/条件性预期:** For a class distribution P define S(P,q)=E||q-X||-.5E||X-X'||. Algebraically its expected advantage over Q when q~P is half the energy distance, nonnegative. This establishes proper distribution scoring under the same distribution assumption, not pointwise classification, cross-image transfer or probability calibration. Squared-distance version collapses exactly to mean distance and is a necessary control.
- **预期收益（假设）:** algebraic under source/query distribution identity: account for multimodal class support rather than one normalized mean or one closest point. Actual identity drift can invalidate the proper-score expectation. Natural segmentation gain unknown.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine/prototype, squared-energy score (must equal unnormalized centroid squared distance to numerical tolerance), unsquared mean distance with self-dispersion term removed, and original0.07 fixed RBF density on exactly same selected rows.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** squared-collapse algebra, sameRBF and no-selfterm controls; proper expected score not pointwise identity or mIoU guarantee
- **特有反例:** P can be far broader than the query's true target subset; subtracting its dispersion may make a broad clutter role too attractive. Proper expectation does not guarantee each individual q is identified. Image style drift changes the class distributions.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### ready RGB06 - 参考监督的跨模态边界似然切分

- **逻辑/条件性预期:** Known R mask labels edge pairs as same-class or cut. Joint bins of relative DINO contrast and relative RGB contrast retain their dependency, unlike generic Q edge smoothing. Only positive same/cut log-likelihood becomes attractive Potts strength; identity still comes from direct DINO unary.
- **预期收益（假设）:** synthetic hypothesis: close a weak DINO hole while retaining true boundary when RGB-only contrasts are identical for texture and object edge but DINO/RGB dependency differs.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Existing reference_boundary DINO-only calibrated cut and RGB Potts with identical unary, graph, readout and pairwise mass; no old pipeline mismatch.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** actually verify source/query bins with tiny-label DINO component, singlemodality same-mass controls and complete masks; no verbal bin-invariance assumption
- **特有反例:** Reference has too few pure cut edges, Q boundary appearance changes, or wrong DINO identity seeds are strengthened; boundary fitting cannot identify an entirely wrong object.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### ready RGB07 - 逐像素阴影商空间颜色判据

- **逻辑/条件性预期:** Algebraic invariant u(s(x)*RGB)=u(RGB) for positive non-clipped scalar s(x), where u(c)=c/||c||. This removes only local brightness magnitude, preserving chromatic direction. Direct DINO identity remains the baseline.
- **预期收益（假设）:** algebraic invariance; hypothesized recovery of chromatically distinct target under spatially varying achromatic shadow, not semantic guarantee.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Raw and globally aligned RGB prototypes with same class support and readout; primary itself is the simple cosine RGB invariant, no elaborate log-color operator.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** invariance only for observed nonclipped scalar shade/constant-color patches, resize/quantization scope explicit; achromatic semantic information loss negative
- **特有反例:** Achromatic FG/BG, colored illumination, clipped channels or same chromaticity distractor; eliminating intensity can remove class evidence.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### ready RGB09 - 参考边界两侧的方向性角色判据

- **逻辑/条件性预期:** Known reference cut edges are ordered FG to BG; their unit RGB difference direction provides a foreground-side role. Query edges can then vote toward one endpoint independently of DINO seeds. This is a learned relational contrast, not an assumed bright-target prior.
- **预期收益（假设）:** hypothesized recovery of a whole visually coherent missed region using a reference-conditioned foreground-side relation, without a correct DINO seed.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** The same learned polarity applied pointwise, plus symmetric boundary segmentation, to isolate role and component propagation; an oracle component identity is forbidden.
- **真实质量:** 未见这些新增项的真实原生4/200质量报告；自然分数和四类增删均unknown。源码/构造不是方法结果。
- **未知前提/审查限制:** pointwise same-role control, reversed context and wrong component negatives; components are not class proof
- **特有反例:** Q surrounding context reverses contrast, internal texture partitions object or merges target/background, one target borders several incompatible colors, same relational contrast distractor.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### closed local_004

local_004固定读出已关闭，独立计0；辅助源码保留但未注册METHODS，不是ready。查询branch pruning只匹配了子集，whole-node奖励与FG读出却按整节点面积，把未匹配Q背景也标FG。完整构造IoU0.428571，低于centroid0.514286及leafbag0.5；rewired同0.428571。它反驳当前读出scope，不证明所有hierarchy不可能。强制全Q覆盖会改alignment合同，只能是同家族修订计0，未自动开始。

[closure receipt](reports/local_structure/local_004_closure.json) ; [batch03](reviews/batch03.json).

## 第三批新增探针资格：6项，自然质量未知

第三批6项已accept_for_probe，取代此前“未见接受”的旧状态。资格不等于真实模型前向、质量或交付完成。context需要有界CPU编码器与同一checkpoint/config/FP32、原RGB和完整MR；共享view缓存/编码费用另列，假callback不计实际DINO前向。root拥有运行调度，本记录不自动扩4/200。

### ready context_remote_reference_response - 未改动查询 patch 的参考条件响应

- **逻辑/条件性预期:** Changing only a distant FG/BG probe can expose conditional attention sensitivity absent from the original final representation. This is only a possible observable: duplicate-instance echo, global colour shifts or positional effects can dominate it, and source response separation does not prove semantic transfer to Q.
- **source/合法输入:** [source](../../../src/ics/cpu100/context_interventions.py) SHA e7d7d652eb657e18d4da84dc7cf5f2885087991ac5c1e20ea86e03e24bf4086c; 合同N加实际原RGB/完整MR及受限CPU冻结encoder callback（同checkpoint/config/实现hash，eval/noGrad/noAutocast/FP32），需要新编码；不能伪称只有缓存NumPy成本。
- **强control:** Same8 views: main feature average prototype; foreground-only main feature prototype; background-only main feature prototype; source probe feature matching readout (ProM4-style absolute inserted evidence) on unchanged Q. Same main-window geometry and decoder. Native raw prototype is an additional resource-separated baseline.
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** Identical reference object in R induces strong duplicate-instance echo, but different-instance same-class Q does not respond. Global ambient delta identical everywhere gives no identity signal. Near-class distractor may respond more strongly. Probe panel can cause attention competition rather than attraction.
- **可得到什么/未决限制:** main-window MR area mapping and byte-identical-condition audit; compare same8view mean/FG/BG/inserted evidence; source identical-instance echo is not same-class Q transfer
- **review:** [batch03](reviews/batch03.json) ; [card](cards/context_interventions.json).

- **Reviewed source / scope correction:** 实际 `context_panel_inserted_probe_control` 为整个16×64底部panel均值（包括neutral canvas）与未改动主图匹配；不是ProM4复现，也不是插入对象专属token控制。卡中ProM4-style旧措辞以此源码审查为准。

### ready context_external_shuffle_sensitivity - 保留局部像素的外部上下文敏感度

- **逻辑/条件性预期:** For each quadrant leave its RGB and position intact while cycling the other three quadrants. At its tokens, ||H_original-H_perturbed|| measures dependence on outside arrangement while local pixels do not change. Whether FG/BG sensitivity measured on R predicts Q is unknown; sensitivity is not inherently background.
- **source/合法输入:** [source](../../../src/ics/cpu100/context_interventions.py) SHA e7d7d652eb657e18d4da84dc7cf5f2885087991ac5c1e20ea86e03e24bf4086c; 合同N加实际原RGB/完整MR及受限CPU冻结encoder callback（同checkpoint/config/实现hash，eval/noGrad/noAutocast/FP32），需要新编码；不能伪称只有缓存NumPy成本。
- **强control:** Same8 views: average original and preserved-patch edited features then known-R prototype; full vector-change prototype; scalar original token norm (constant for unit features). Same geometry/window. No different crop or extra forward-budget advantage.
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** A true object is context-dependent while generic grass is stable; background sensitivity from R flips in Q. Query object spans a quadrant boundary and outside permutation destroys its continuation. Source means may differ merely because boundary proportions differ.
- **可得到什么/未决限制:** original Episode must be same native1024 RGB/checkpoint/geometry as callback; all-patch preserved-quadrant coverage; scalar/vector/mean sameview controls; continuation destruction negative
- **review:** [batch03](reviews/batch03.json) ; [card](cards/context_interventions.json).

### ready context_probe_nonadditive_competition - 双参考刺激的非加性交互响应

- **逻辑/条件性预期:** Encode assignments(F,F),(F,B),(B,F),(B,B). The raw-LN mixed difference C=H_FF-H_FB-H_BF+H_BB is identicallyzero for any additive probe effect. It can carry class-dependent nonlinear interaction even when H_FF-H_BB is identical on classes. It is not monotonic rescaling of firstdifference, but could still be duplicate-instance interference rather than category signal.
- **source/合法输入:** [source](../../../src/ics/cpu100/context_interventions.py) SHA e7d7d652eb657e18d4da84dc7cf5f2885087991ac5c1e20ea86e03e24bf4086c; 合同N加实际原RGB/完整MR及受限CPU冻结encoder callback（同checkpoint/config/实现hash，eval/noGrad/noAutocast/FP32），需要新编码；不能伪称只有缓存NumPy成本。
- **强control:** All above same-view8 controls; additive toy encoder must giveCzero and exactfallback. Nonadditive toy with equal firstdifferences must distinguish different-instance target before acceptance of uniqueobservable. Panel token exclusion/seam geometry fixed. Same-view four-state raw-stack squared-distance prototype control must also be evaluated, not just firstdifferences.
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** Two reference copies compete for attention and suppress identical-instance features while realQ objects have different effect; panel seam effects themselves nonadditive. Signal may be tiny float32 cancellation or universallyzero. Nonzero curvature is not category discrimination.
- **可得到什么/未决限制:** complete cross-scene additive-nuisance witness beyond firstFF-BB AND both single-slot AND average/stack controls; rawLN before unit; additive/instance-only echo negatives, report precision scale; no further order/intensity variants counted
- **review:** [batch03](reviews/batch03.json) ; [card](cards/context_interventions.json).

### ready context_patch_position_orbit - 局部像素不变的位置轨道重编码

- **逻辑/条件性预期:** Inverse-mapped re-encoded features can separate content from position-dependent terms if their semantic component survives a coarse roll and nuisance changes sign/averagesout. This is a newforwardobservable, not shifting oldarrays (which would unroll exactly and add0information). Wholeobjects crossing the seam may be damaged, and RoPE relative invariance could leave nochange.
- **source/合法输入:** [source](../../../src/ics/cpu100/context_interventions.py) SHA e7d7d652eb657e18d4da84dc7cf5f2885087991ac5c1e20ea86e03e24bf4086c; 合同N加实际原RGB/完整MR及受限CPU冻结encoder callback（同checkpoint/config/实现hash，eval/noGrad/noAutocast/FP32），需要新编码；不能伪称只有缓存NumPy成本。
- **强control:** Same4newforwards scalar score averaging; same inversegeometry andreadout. Originalprototype resource-separated; duplicate unchanged forward audit optionaladapterparity (engineering, not method). No finepixel interpolation or layer difference can be credited.
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** Object/scene continuation crossesrollseam andsemanticfeaturechanges; randombackground acquires spuriousstableaverageddirection. True discriminativecontext is discarded. IfrelativeRoPE/encoder is perfectlyroll-equivariant, newforwardoutputs equaloriginal and method noactiveobservable.
- **可得到什么/未决限制:** patch-byte/inverse-map audit, same4forward scalar-average control, actual original1024 producer/view parity; separate seam/context harm from positional claims
- **review:** [batch03](reviews/batch03.json) ; [card](cards/context_interventions.json).

### ready local_005 - Nonparametric joint DINO patch-to-patch label voting

- **逻辑/条件性预期:** The output label vector of a matched reference neighborhood is legal supervised information that is not just its center label. Transferring and reconciling neighboring label predictions changes the full decoder while retaining each source local appearance configuration. This is a local mask-pattern prior, disclosed explicitly; it is not identity evidence when two classes share that appearance and label pattern.
- **source/合法输入:** [source](../../../src/ics/cpu100/local_structure.py) SHA 213d452c7be4491066ec7ae3ded664f0b448f2a3c3b2791267ade99ee7bf5c6f; N; known-MR local labels and physical-valid weights only.
- **强control:** Identical descriptors, anchors, bandwidth and nearest4 matching, but each matched patch votes only its central MR label to the query center.; The same central-only complete margin, followed by generic uniform nine-stencil score averaging; compares label transfer with ordinary smoothing.; Same matched local patches with noncentral MR labels permuted within each patch, keeping center label and foreground-label sum; isolates organized label geometry.; Full local003 convex ordered-patch classifier on same input/renderer; it reads center labels rather than joint label vectors.
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** Changed boundary geometry or scale can make reference neighbor labels wrong despite matching appearance.; Dense BG templates and repeated patterns can outvote a correct thin target; equal anchor counts do not prove unbiased votes.; A coherent false object with the same local label motif is accepted.; Common or zero tensor descriptors can create unsupported identity ties; no entropy or vote consensus claim substitutes for class evidence.; Occluded or missing parts may be filled according to the reference motif and create false positives.
- **可得到什么/未决限制:** same-matcher center-only/generic smoothing/noncentral-label permutation controls; explicit local label-pattern prior; distinguish truly empty MR from missing majority-grid anchors; thin-target and wrong-motif negatives
- **review:** [batch03](reviews/batch03.json) ; [card](cards/local_structure.json).

### ready QP09 - 完整参考类距离的查询字典最短测地线

- **逻辑/条件性预期:** 单位特征的cost=1-cos不是角距离的三角不等式度量，多个小角步可比一个大角捷径便宜。因此完整R类初值在query字典min-plus闭包下可改变最近参考类别排序；这个代数作用只在同类沿连续链、跨类有间隙时有利。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40; N; known-MR local labels and physical-valid weights only.
- **强control:** 相同query字典完整R softmin类初值差（无路径）; samegraph continuous harmonic solve初值差; samegraph min/max瓶颈路径（旧路径原则仅不计控制）; same DINO prototype/pointwise KDE/QP02
- **观测质量/四动作:** 真实二批/第三批质量 unknown；四类增删 unknown，未把未回收/缺控制的结果写成收益。
- **特有反例:** 类别边界可在连续feature链中没有间隙；路径会补入本来正确排BG的近类干扰。dense sampling改变路径成本，远端弱类支持可被错误捷径传播。
- **可得到什么/未决限制:** same-dictionary no-path/harmonic/minmax controls,1-cos small-step chain proof and exact same-features cross-class-bridge negative; density dependence and mistaken initial identity explicit; semiring/k variants0
- **review:** [batch03](reviews/batch03.json) ; [card](cards/query_partition_batch4.json).

### 新完成的静态source审查（不加方法数／不升级quality）

context源码静态审查已完成：当前物理probe由完整MR包围框加16像素自然上下文构成；negative仅替换已知MR像素为最近已知R背景，外MR字节保持相同。旧灰色alpha负刺激保留了FG silhouette，不能解释为物体不存在；旧toy保留。新刺激也不保证DINO感知语义缺席，仍可有轮廓、接缝和纹理伪影。

toy只有规定的pixel-derived假编码器，实际DINO前向0。remote/mixed/orbit完整构造1.0；external sensitivity0.25 vs mean0/vector0.0833；这只证明可作用，不证明DINO具该响应。四象限保留字节与RGB inverse-roll映射审计通过；21个callback尝试在28 cap内。whole-panel-average control包含neutral canvas，不是ProM4复现或插入对象专属token控制。

[context source audit](reviews/context_source_20261007.json) SHA 7e0474cbde148e1cb14e3a3e4e26a71ac05b1db00be91934b7ab8f7650e0d279

注：`context_source_20261007.json` 绑定源 SHA `afb0ab16070208ad82a214d8f43e896c7af44c919cb0ec04e3ac3edcfed53fdf`，与当前 `context_interventions.py` SHA `e7d7d652eb657e18d4da84dc7cf5f2885087991ac5c1e20ea86e03e24bf4086c` 不同；当前版本另含 probe-area pooled control，不能称该审查已覆盖此完整源码。新的 toy receipt 绑定当前源码，但只使用构造假编码器，真实 DINO 前向仍为0。

DR09/10/12/13/14/15/16静态源码审查已完成，只用合法R特征/MR拟合，Q仅最终完整margin推断。DINO冻结但ReLU/GLVQ/PSD确有episode参数优化；source源码资格不升级自然质量。DR09无效bank回退是pure-reference NN，不是卡的泛称prototype；源码多数为vectorized Q×support workspace，不能称统一block128。

DR_control_average_logistic采用每角色最多64条weighted-quantile压缩，不能称完整R逐token未压缩拟合；Frank-Wolfe gap<1e-8是数值容差，不是字面精确几何。DR16初始矩阵仅前min(8,K)坐标，而identity-control用全部K；未来若有提升不能直接归因监督metric更新。

保留构造审查：ReLU梯度误差<1e-10、GLVQ2.04e-11、LMNN2.51e-10只支持实现导数；七项完整shape、padding、空类等合同检查不是自然收益。DR10/14/15存在相对最强所给control的构造见证；DR09/12/13/16没有严格增量见证。DR15原预期近边界正例反而低于control0.1383，预期高端负例却高0.0060；应撤回原故事，不扫变体。

[risk source audit](reviews/decision_risk_batch2_source_20261007.json) SHA f8d10ecf7424b696c83de4fe8aebe8f3e8a121c5bc3f40698debabd3d6b898fe

## 重复、代数关闭和未审核卡：不计数

| ID | 失败解释与所得 |
|---|---|
| ref_affine_tangent_support | 旧hull已有role-specific affine control；rank2/pole修订计0，reviewer撤回漏审放行。 |
| QP03 | 高阶均值tensor距离差展开为pointwise三阶score pooling，未加joint信息，计0。 |
| DR11 | 旧reference_quadratic已有role-balanced非齐次degree2核，压缩/ridge/raw适配非新机制。 |
| QP05 | 已有DINO-neighbor binary cut；只换unary/scale不计新机制。 |
| RGB08 | 凹mixture ML alpha*>.5 iff导数(.5)>0 iff平均equal-prior posterior>.5；fusion幅度是confidence读出，不是新identity。未实现，计0。 |
| ref_class_typicality_rank | R距离scale/tail跨Q转移仍被local-radius反例针对；未放行，不改rank名字绕反证。 |
| local_005 / QP09 | batch03已accept_for_probe；完整真实质量未见，新增资格2，非完成2。 |
| context四项 | batch03已accept_for_probe且源码静态审查完成；只获有界CPU probe资格，fake callback不算DINO，真实质量unknown。 |

## 当前获得的结论与缺口

本文件现记录首30、二批22新增、旧19及第三批6资格，合计潜在77。local_004单列关闭计0；local_002修订计0；source/static/toy/完整真实输出/稳定正收益严格分开。未完成100。

200例Huber vs ridge +0.7109 [0.1353,1.0014]是已测真实开发边际；不能抹掉，也不能把相对prototype的+9.7928全部归给Huber。ridge=51.6625，平均logistic=50.3467，常量erosion=50.7178都是强简单controls。

尚无独立确认、同协议完整FoRIS比较、稳定>=2 pp或完整论文方法交付。新52.3734不能与旧processed60.70/62.93跨协议判优劣。负结果不证明信息用尽/DINO上限已定，也不授权新变体。

更新只跟随新封存结果修订相应方法的观察、control、T和解释范围。本文件不授权SSH/实验，也不是新pending-work清单。

本次只同步必要审查修订，不新增方法、文件或实验；尚未回收的二批真实得分维持unknown。
