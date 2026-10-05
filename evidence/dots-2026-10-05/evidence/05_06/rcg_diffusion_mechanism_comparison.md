# RCG与已失败seeded graph的机制边界

RCG不能被旧pure-seed PPR/conductance的失败直接否定。它保留每个query位置的强宿主软证据，以正定fidelity约束解一个二次平滑问题；旧构造用少量seed产生排序，再依据conductance选区域。两者同属非负图传播家族，但输入的标签证据与最后的决策目标不同。

本次只核机制和数学条件，不重复RCG数值复现，不改变任何参数。用户release来源：`/workspace/scratch/c4693694c197/rcg_result_0636/CVPR2027/extracted/RCG_release/rcg_readout.py`。旧算子来源：`seeded_extent_capability.py`、`seeded_nonlocal_extent60.py`及各自冻结报告。

## 实际RCG做了什么

设s为FoRIS score的minmax归一值。source纯FG经CSLS型度量产生guide g，代码先形成

    y = s + .5 [rank(g) − rank(s)]

rank是平均秩百分位。然后query unit features构造20近邻有向图，只留互为近邻的边，取双向权重几何均值形成对称W≥0，缩放至平均degree=1。令L=D−W，A=diag(a)，其中a_i∝.1+|2s_i−1|，故所有a_i>0。

最终求解

    (A+16L)z = Ay

这对应唯一极小值

    E(z)=.5(z−y)^T A(z−y) + (16/2) z^T Lz。

最后直接双线性放大连续z，再>.5；不重新minmax，没有CRF。y并非概率，理论上可能落在[0,1]之外。以下边界必须对修正后的y成立，不能偷换为原始s或原FoRIS二值mask。

## 和旧seeded PPR/conductance的实质差别

|方面|旧固定seeded构造|RCG|
|---|---|---|
|标签证据|16个FG、16个BG query seed；普通版由source NN选，能力诊断用GT纯seed|全部4096位置的FoRIS软证据，并有source guide秩修正|
|传播|两路非负PPR响应，再取FG响应/(FG+BG响应)|单路dense软unary的置信度加权二次解|
|图|旧local四邻接；追加版为query16-NN的union对称图|query20-NN的mutual图，自适应距离温度|
|输出|强制seed标签，最小conductance的合法PPR排序prefix|连续场放大后固定>.5，没有conductance扫描|
|错误风险|纯seed也不能保证图的低cut区域是指定语义对象；排序和cut都可失败|仍受unary证据范围约束，但无需把目标当作低conductance区域|

RCG不是一种全然不同的传播数学。逐节点改写为

    z_i = ρ_i y_i + (1−ρ_i) Σ_j P_ij z_j
    ρ_i = a_i/(a_i+λd_i),   P_ij=W_ij/d_i

可见它等价于每个位置都有软注入、restart率随节点变化的广义PPR。孤立点直接z_i=y_i。关键区别是dense输入和readout；不能仅凭“也是graph/PPR”跨实验否定，也不能把改写形式称新颖性。

旧实验中，纯seed的非局部PPR最佳GT prefix仍低于同seed直接NN最佳prefix；这限制的是其稀疏seed强制扩散排序。其conductance通常比真实目标更偏爱错误大区域，则限制该无监督cut目标。RCG没有复用这两个完整决定，因此旧结论不能原样转移。

## 严格maximum principle与证据守恒

对任意λ≥0，定义K=(A+λL)^−1A。A正对角、W对称非负，所以A+λL是严格对角占优M-matrix：逆矩阵逐元素非负。又因L1=0，K1=1。K在每个W连通分量内是行随机矩阵，跨分量元素为0。因此

    z_i=Σ_j K_ij y_j，且min_{j∈C}y_j ≤ z_i ≤ max_{j∈C}y_j。

若λ>0且C连通，K在C内严格为正。另有逐分量守恒式

    Σ_{i∈C} a_i z_i = Σ_{i∈C} a_i y_i。

由此得到以下精确条件，而不是泛泛说“图不能发现新对象”：

1. **完整图分量全负。** 若max_C y≤τ，τ=.5，则任意λ都不能使其任何token成为>τ的FG。若min_C y>τ，则任意λ都不能把其中token改为BG。改变正的fidelity权重也不改变此结论。
2. **常值分量。** 若y在C恒等c，则z在C也恒等c，任何λ无效。
3. **强平滑极限。** λ→∞时，C上的z趋于A加权unary均值。若该均值≤τ，整个分量全变FG不可能；这是守恒限制。均值>τ时，足够大的λ可以把整个分量推成FG，但不说明这就是正确语义。
4. **部分区域。** 一块错误语义区域R不必等于完整图分量。若R内y≤τ−δ，外面y≤τ+M，记κ_i=Σ_{j∉R}K_ij，则

       z_i−τ ≤ −δ(1−κ_i)+Mκ_i。

   当κ_i≤δ/(δ+M)时，i不可翻成FG。外部有足够正确证据及足够图影响时，R即使在图像中很远，也可能被修复。物理距离不能替代κ。

因此，release所称距native mask超过48px的deep-FN，仅是图像空间错误分层。它不自动满足“图分量全部y负”，不能由本定理直接解释其回收率，也不能据该回收率反推整片表征无语义。

CG给出数值近似。若残差r=(A+λL)z_hat−Ay，则||z_hat−z||∞≤||r||∞/min(a)。这只是求解容差界，不是额外方法改动。本任务没有重做release的solver或数据统计。

## 最终像素级的额外限定

双线性插值写成B，B≥0且每行和为1；最终连续像素场为p=BKy，仍是原unary的非负平均。

但一个像素的四个贡献token可能属于不同W连通分量。某个全负图分量中的token不能变正，不意味着该分量覆盖的每个1024像素都不能因邻近正token插值而越线。严格像素保证是：其所有贡献token所连接到的unary支持都≤τ，或对应的总外部影响权重仍满足上面的泄漏界。不能忽略renderer后直接从token定理声称整个对象像素永不恢复。

## 什么新可观测量才可能改掉整片错误

在固定W、固定y下，换λ仅改变混合权重；若整个分量已同侧错误，正则化强弱或新的求解器不能创造另一侧证据。要突破上述条件，至少要改变其中一个前提：

- **新的reference-conditioned membership evidence改变y。** 一个完整对象假设H，如果其whole-region reference匹配能在目标与错对象之间提供当前unary没有的区分，可改变该区域的标签证据。候选是object-like本身不够，必须证明它属于reference指定概念。
- **有语义依据的新连接。** 将错误分量连接到另一块有正确反向证据的区域，可以允许信号流入。但仅增大k不等于观察到同类关系；一个错误连接也会扩散错误标签。
- **明确的新边界标签。** 有合法来源的reference条件对象标签或part标签，可以作为新的约束。query GT、oracle seed或oracle hypothesis只能诊断，不是合法证据来源。

一个精确代数例子：若独立观测为整个图分量提供常量证据增量d，则y' = y+d·1_C会导致z'=z+d·1_C，因为K1_C=1_C。这说明真正改变整片判定的是标签证据增量，而不是λ。这里不提出d的拟合、阈值或融合方法；目前尚无已验证的d。

## 对完整object-hypothesis acquisition的具体判据

先由bank容量实验判断：bank里是否存在覆盖深漏检、同时避免错误对象的H。oracle存在只说明动作空间够，不能确认有可用选择信号。

之后真正需要的可观测量，是同一个bank中一个合法reference条件分数，能把正确完整对象排在高分错对象前，并超过同bank原unary均值、reference均值和既有fixed roundtrip等简单控制。该分数需要在whole-object错误、特别是当前同侧错误分量上有条件区分，而不是只在易背景上抬高global AUC。

此前两例已说明这个区别：完整对象bank存在高IoU假设，但fixed roundtrip未都选到，dynamic reverse也未稳定超出简单forward mean。它们是有效反例，不能把“拿到了完整对象proposal”自动当成新的语义标签。下一步应由正在进行的bank容量与同bank身份分离证据决定，而非再调RCG λ。

## 本次结论

保留RCG作为有dense宿主证据的完整graph readout，不能用旧pure-seed/cut负结果替代它的实际比较。其可修复的主要对象，是unary中已有正确支持、但空间或语义一致性尚未利用好的错误。要处理整片同侧错分量，需要可检验的新reference条件标签证据或有依据的跨分量语义连接；现有完整对象假设路线还只提出了这个可能性，尚未兑现。


## 后续已量到的实际预算

固定recipe诊断见`rcg_unreachable_budget_result.md`。全120仅0.107%的deep-FN质量处于实际unary全负图分量，保守最终像素证书仅0.0049%；所以这条严格定理并不能解释当前绝大多数深漏检。99.892%的deep-FN所在分量还含正确纯FG正unary token。此时需要研究的是混合分量内的语义关联与有选择的证据传递，而不是把定理误写成深漏检普遍不可恢复。
