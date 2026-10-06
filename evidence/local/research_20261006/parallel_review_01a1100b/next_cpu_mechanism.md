# 下一 CPU 机制：优先核验 Pro M2，M3 暂保留成本受限合同

日期：2026-10-06。状态：只读比较与数学合同；无实现修改、无真实分割运行、无 SSH/实例查询/GPU/付费资源。本文件是审查产物，不是第二份 PLAN。当前关机边界见 [PLAN](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:3) L3–54；遵守 [AGENTS](/Users/yang/projects/CVPR2027/AGENTS.md:3) L3–29。

**结论：先做 Pro M2 的原样合同核验，比再增加一个前景统计量更有决定价值。** 九项尚未直接使用“参考 FF/FB/BF/BB 配对依赖的交叉比”作为查询标签耦合。Pro M3 的“关系先预测对应、跨图外观后揭晓”也是未覆盖的程序，但已与 constellation 的留出验证明显重叠，且可能代数退化成低维外观匹配；应先核这个退化和成本，再决定是否值得完整实现。下面只有两个合同，均保留 Pro v0，不将其与九项拼成新组合，也不将本报告计作两项已实现方法。

## 1. 九项覆盖了什么，缺失量是什么

“新增量”是相对既有算法读取的统计对象而言；两者仍从同一合法 R/M/Q 提取，不声称新增传感信息，也不声称论文原创性已成立。九项本身全部真实效果未测，见 [PLAN](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:7) L7–50。

| 已有候选 | 已读取的量 | 与下一机制的边界及源行 |
|---|---|---|
| adjacency | 纯参考 FG 模式的空间相邻频率，减模式数量置换零假设；区域树 MAP | 已有 signed **区域奖励**，但不是由四标签状态估计的 signed **查询标签边耦合**。不读取 BG 配对角色。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/method.json:10) L10–20。 |
| huber | 同一参考一元/查询图，改变大对比边上的作用力上界 | 只改求解规则，不估参考标签依赖；卡明确不主张新增语义身份信息。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/method.json:25) L25–41。 |
| color bottleneck | 查询 RGB 边瓶颈与现有强 FG/BG 种子 | 增加查询光度边界，不读取参考角色对关系。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/method.json:10) L10–19。 |
| constellation | 两参考部位确定相似变换，其余部位在预测位置验证；转移整张参考轮廓 | **已经存在 heldout 思想**。M3 若只将距离换成 Gram 后仍做同一姿态验证，不算新机制；Pro M3 的区别是外观关系指纹注入匹配与树上区域并集，既不要求单相似变换，也不转移参考轮廓。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/constellation_preparation_01a1100b/method.json:5) L5–17。 |
| shape | 完整参考轮廓矩与真实查询 FG 连通区域的形状差 | 区域几何一致性，不揭晓被留出的角色对应，也无四标签状态边关系。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/shape_preparation_01a1100b/method.json:5) L5–25。 |
| covariance | 查询 FG 区域对参考 FG 模式的响应协方差；有界正证据预算惩罚 | 区域内二阶响应汇总，不能据此认定已经检验了跨边 FF/FB/BF/BB 交叉比，或先预测后揭晓的局部对应；其可实现合成例证明了自己的附加量，不能移植给 M2/M3。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/covariance_preparation_01a1100b/method.json:9) L9–17、L54–60。 |
| recurrence | 查询高置信分离组件之间的重复外观与参考门控 | 查询伪标签原型适配，不估四状态参考边关系；不强制对被留出角色作预测。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/recurrence_preparation_01a1100b/method.json:5) L5–17。 |
| prior shift | 参考 FG/BG 标量得分密度＋查询全局混合比例 | 已使用 BG，但仅一元密度；不能表达条件端点相同而配对不同的边依赖。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/prior_shift_preparation_01a1100b/method.json:10) L10–16。 |
| quadratic | 参考 FG/BG 锚点上的方向二次核分类，逐点推断 | “二次”是单 token 内特征乘积，不等于两个空间 token 的条件标签依赖。见 [卡](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/quadratic_preparation_01a1100b/method.json:5) L5–15。 |

因此真正值得问的两个问题是：**固定角色边缘和 FG 内部结构后，FB/BF/BB 的配对结构能否改变完整输出并有利迁移？** 以及 **当完整区域允许非刚性变化时，关系指纹能否预测未直接用于对应拟合的角色，比全角色共同拟合与区域均值更可靠？** 前者优先。没有依据预测 mIoU 增量或预期修复例数。

## 2. 合同一：Pro M2 原样——参考四状态标签关系

来源：[Pro 文件](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:388) L388–519；共用输入/renderer 为 L162–250、L1230–1237。这里复述 CPU 可执行数学对象，**不把已有 MEAN 场替代 Pro 的 FoRIS pre-CRF 场**，不将 32 角色改成既有八个 FG 模式，不将 mutual20/空间多尺度改成单一四邻域。

### 从输入到完整 mask

1. 入口是唯一参考 RGB、完整二值 mask、查询 RGB、冻结 DINOv3-L/16；工作尺寸 1024、网格 64²、D=1024。若使用本地缓存，必须绑定与 Pro 合同相同的 producer、位置去偏、FP32/归一化、coverage 与连续宿主 s；现有九项的 MEAN packet 不自动满足这些绑定。缓存后 CPU 新模块无额外编码；独立图像入口仍需两个编码及宿主成本。
2. 合并 R/Q **无标签**末层特征，确定性 spherical k-means：K≤32、farthest-point 初始化、token ID 平手、最多20轮；每 token 在最近2角色上温度 .07 softmax，得稀疏 a_i。按实际非空 K 处理空簇/top1 退化。
3. R/Q 各自建立轴向距离1/2/4和特征 mutual20-NN 四类边。参考 m_i 是面积覆盖率，w_i¹=m_i，w_i⁰=1−m_i。对每类 e、每种标签状态 s,t∈{0,1}：

   \[
   p_s(k)=\frac{\sum_iw_i^s a_i(k)+1/K}{\sum_iw_i^s+1},\quad
   N_{st}^e=\sum_{\{i,j\}\in E_R^e}\left(w_i^sw_j^ta_ia_j^T+w_j^sw_i^ta_ja_i^T\right).
   \]

   令 n_st 为 N_st 全表之和，P_st=.9N_st/n_st+.1p_sp_tᵀ。若任一 n_st=0，整类关系归零。所有状态同 shrinkage，不能以“稳定性修复”为由改成不同系数。
4. 查询边 L_st(i,j)=a_iᵀP_sta_j，计算纯相互作用：

   \[
   J^{raw,e}_{ij}=\tfrac14\log\frac{L_{11}L_{00}}{L_{10}L_{01}}.
   \]

   FP64，数值残差绝对值<1e−10置零，再 clip[−2,2]；重复类型等权平均。用最大绝对行和归一到≤1。它抵消标签状态下可分的一元项，**不是把负 FG 邻接奖励换一个名字**。
5. s 为完整 FoRIS 二值化/CRF 前场按原 minmax 得到；ℓ=s−.5，λ=.9。求严格凸连续对象：

   \[
   E(z)=-\sum_i\ell_i z_i-.9\sum_{\{i,j\}}J_{ij}(2z_i-1)(2z_j-1)
   +\sum_i[z_i\log z_i+(1-z_i)\log(1-z_i)].
   \]

   z⁰=σ(ℓ)，同步 z←σ(ℓ+1.8J(2z−1))，250步上限，相邻变化<1e−6停止。收缩界给出约1e−5解误差；记录异常，不默默换求解器/参数。
6. 输出 **t=s+1.8J(2z−1)**；不 clamp/minmax，进入renderer统一转FP32。bilinear64→1024后严格>.5，再将工作二值 mask bilinear 到原查询尺寸、严格>.5，两次均align_corners=False。J=0 时 t=s 逐值，插值后的完整 mask 也一致；不等于复现 native CRF。
7. 空参考返回空；无 BG/四状态不足关系归零；纯 FG token 不足按 Pro coverage 回退，零向量固定零。保留所有有证据的查询实例，不使用 class/fold/query GT。预测先封存、后评分。

### 最强简单替代和可观测反例

同 s、字典、边、renderer 六行一次固定：signed 主法；zero；positive（raw负数截零后同归一）；absolute（raw绝对值后同归一）；pair-independent（每状态 N_st 的**加权端点边缘外积**替代经验联合分布，再同 .9/.1混合）；block（参考8×8块四奇偶组删除边，只保留四次一致非零符号并取均值）。定义见 Pro L494–513。最先要求胜过 **absolute/positive/pair-independent**；如 block 更好，只归到 block 控制。另与原完整 MEAN、native、已绑定历史强行比较。不得只拿 zero 作为充分对照。

**明确反例：**同类 Q 中参考负边更多切开真目标内部，signed 增加 delete_TP；跨场景背景关系不迁移；角色字典主要描述纹理而非类别；signed 与 absolute/positive 相同或更差；pair-independent 完整结果追平。任何一项都收缩相应主张。旧 joint R/Q harmonic 大负是风险依据，不是本法已经被实测否定；来源 Pro L109–125、L490、L515。

**一个局部数学区分，不是完整分割收益：** 两角色、各状态端点边缘均(.5,.5)，固定 FF 与 BB 表为 [[.4,.1],[.1,.4]]；只把 FB/BF 从同表改成 [[.1,.4],[.4,.1]]。经 Pro .9/.1混合后，前者所有 J=0，后者同角色 J=+.604155603、异角色 J=−.604155603。FG 内部表未变、状态条件端点边缘未变，但标签依赖变了；前景邻接与一元边缘无法表达此差别。本报告已用本地 Python 标量计算核对数字。**这是给定概率表的代数检查，不是已经由真实64²单位特征/空间图生成的全算法可实现例，也不是已通过 M2 完整解码。**

### CPU 复杂度与第一个改变决定的校验

字典 O(T(N_R+N_Q)KD)，T≤20、K≤32；精确 mutual20 构建 O((N_R²+N_Q²)D)，分块可免完整矩阵驻留但不能消除乘算。N=4096：字典上界约5.37 GMAC，两图余弦各约17.18 GMAC；每张完整 FP32 affinity 64MiB，仅作规模估算。四状态表 O(4·4K²) FP64≈128KiB；top2 每 L 只读取四表项；关系与迭代 O(|E|+I(N+|E|))，I≤250，存储 O((N_R+N_Q)D+|E|+K²)。宿主及图重建、I/O、renderer 必须另计。**CPU 后处理低成本不意味着从现有缓存启动就廉价**，R mutual20 与32角色往往需要新增构建；真实秒数未知。（Pro L517。）

当前关机下，先做一次有限 **可实现性排除检查**：在≤16个 token、D≤8的归一化合成 R/Q 上，用原字典/四状态估计/同步解码/renderer 规则寻找或构造“FG内部量及状态边缘相同、配对关系不同”的两例，要求 signed 真正改变完整 mask，且 zero/absolute/positive/pair-independent 无法解释该特定差别，同时保留一个负边损坏正确输出的对照。禁止拿手工注入 J 代替从合法 R/m 估计。只做这个单一检查，不扫常数或变体。若当前固定算法/字典下无法形成有效符号，先报告具体不能辨识的原因，不能仅靠概率表数字登记第10项可用方法。

真正决定迁移价值的下一证据只能是：资源恢复且重新授权、完整缓存绑定后，同批已暴露自然例上的上述固定完整行；报告完整 class-summed mIoU、照片组配对区间与四类增删，重点看 signed 相对 absolute/pair-independent 的净贡献。既有真实错误诊断没有测该量，因此不承诺正增益，也不自动从参考 holdout 进入大实验。

## 3. 合同二：Pro M3 原样——关系预测之后揭晓角色外观

来源：[Pro 文件](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:523) L523–662。与 constellation 的区别成立条件是：测试对象是 **关系所确定的 query 子区身份及后揭晓的跨图比较**，允许非相似变换布局；不能仅把空间距离改成特征距离计作新方法。本合同不与 M2/九项组合。

### 从输入到完整 mask

1. 同一合法 R/M/Q、冻结特征及 FoRIS pre-CRF s；u=logit(clip(s,σ(−4),σ(4)))。参考 coverage≥.5 的四连通分量各作模板；每模板≤8角色、20轮确定性球面聚类，实际非空角色≥4才可用。超过8模板以均值 farthest-point 保留8；BG≤16原型。无可用模板返回**完整 native FoRIS**，非空原 mask 无≥.5 token 使用最大coverage回退；空参考返回空。回退调用及成本显式保留。
2. Q 构建**空间相邻 Ward**完整合并树，合并代价 |A||B|/(|A|+|B|)·||q̄_A−q̄_B||²；8191节点。每节点反复拆最大可拆后代，形成≤16子区，归一化均值作角色描述。按最小 leaf ID 平手。不能偷换成九项已有 cosine single-linkage 树；不按 s 删候选。
3. 每参考模板按初始化序偶数角色 A、奇数 B。D_R(a,a′)=1−p_aᵀp_a′，D_Q(j,j′)=1−q_jᵀq_j′。A 首次 Hungarian 注入匹配用 C_aj=1−p_aᵀq_j，独立 dummy代价1；最多5轮按原对应加平均 Gram平方残差再重匹配，保留原 A 目标最优者，重复/不改善停。固定目标：

   \[
   E_A(\pi)=\sum_a C_{a\pi(a)}+\frac1{|A|}\sum_{a<a',\,real}
   [D_R(a,a')-D_Q(\pi(a),\pi(a'))]^2.
   \]

   这是固定近似法，不宣称全局 FGW 最优。
4. B 只能用与已匹配 A 的**关系指纹**选择未占用子区：

   \[
   E^{pred}_{bj}=\frac1{|A_{real}|}\sum_{a\in A_{real}}
   [D_R(b,a)-D_Q(j,\pi(a))]^2.
   \]

   用此代价 Hungarian、足够独立 dummy代价1；这一步不读取 p_bᵀq_j。若 A真实匹配不足2，该方向验证−2。然后才揭晓

   \[
   h_b=\frac{p_b^Tq_{\pi(b)}-\max_l b_l^Tq_{\pi(b)}}{.07}-\frac{E^{pred}_{b\pi(b)}}{.1}.
   \]

   无 BG 时背景项0，dummy h=−2且不访问下标；交换 A/B 同样一次。两方向角色平均clip[−4,4]，各模板取最大得V(v)；子区不足2时V=0。留出的不是特征统计独立性，也不是未使用 p_b 的所有信息：p_b 与 A 的 Gram 关系已经用于预测。
5. G(v)=∑_{i∈v}u_i+.5|v|V(v)−κ，κ=log(1+N)。F(v)=max{0,G(v),F(v_L)+F(v_R)}，叶比较0/G；平手 background、children、whole-node。回溯互不重叠 FG节点并集，二值64²交与合同一相同的 Pro统一bilinear renderer。可重复解释任意多个查询节点，不限制参考全图容量，不复制参考轮廓。
6. 单token不能独立入选：κ≈8.318>u≤4，叶V=0。不能免除叶子罚、削模板、删候选以“修实现”名义改变 Pro v0。

### 最强简单替代、与既有方法真正不同的条件

同模板、Ward树、子区、dummy、DP、κ、renderer，比较 heldout、all-role（全角色共同跨图＋关系拟合、同5轮并在拟合对应上验证）、mean（节点均值对参考FG均值减 hardestBG的margin/.07并clip）、zero V。all-role具体计分、dummy与不足2真实角色的退化按 Pro L636–646。与完整 native/强历史行同口径；要宣称优于现代对应法，另需绑定固定现代不平衡匹配控制，不能把自制 all-role 视为现代最佳。

相对 covariance，M3读取**候选子区集合的成对 Gram 关系与对应映射**，而 covariance 将区域token响应压成一、二阶矩；并不保证前者永远包含额外信息，也不保证优于完整矩阵模板。相对 constellation，它能在形变下不用空间相似变换而预测外观角色；但要证明这种区别实际有价值，必须在同类非刚性例中 heldout 能胜 all-role/mean，并改善完整 mask。仅“留出角色分数不同”不足。

**关键可观测退化：**当真实拟合锚点 q_{π(a)}=p_a，关系预测就是用 p_b 对已知锚点的投影匹配。若锚点是包含 p_b 与 query角色的 d维子空间的正交单位基，则

\[
E^{pred}_{bj}=\frac1d\|p_b-q_j\|^2=\frac2d(1-p_b^Tq_j).
\]

这与被“留出”的直接外观相似严格同序；完全没有新的选择依据。D=2、p_b=(.6,.8)、锚点e₁/e₂的本地标量检查中，q=(.6,.8)、(.8,.6)、(−.6,−.8) 的 Epred分别0、.04、2，恰等于1−cos。这里在给定描述子上验证代数，未运行聚类/树/Hungarian/完整算法。故“未读 p_bᵀq_j”这一代码边界不能单独证明旧终点外观以外的身份依据。

其他明确反例：遮挡使B缺失/dummy增多；同类特征关系随视角改变比干扰变化大；子区均值混合角色导致预测无意义；小目标因κ丢失；真对象不在Ward树完整节点中；同类别共享部件关系仍不能拒绝错对象；模板取max会选中偶然过拟合者。source负证据见 Pro L624、L632、L648–660。

### CPU 复杂度与第一个改变决定的校验

树维护成本取决于精确邻接更新/heap实现，不能笼统报线性；树区域和特征均值 O(ND)存储，8191×1024 FP32≈32MiB。每节点子区≤L=16，Gram构建 O(L²D)，与模板跨图 O(KLD)，K≤8；每次矩形Hungarian保守O(L³)，另有固定5轮关系更新。最坏调用≤8模板×8191节点×2方向×7=**917,392**；即便每个很小，Python/SciPy调度也可能主导。流式内存不等于高吞吐。条件∑u+2|v|−κ≤0可严格跳过该节点整块验证，但仍递归子节点；只有数学等价缓存/批量化属于同合同。（Pro L652–660。）

第一个有限校验是**关系预测有没有逃出直接匹配的代数退化**：固定上述原样合同，在≤16个单位描述子的明确锚点变换例中比较 Epred排序与直接cos/all-role排序，先重现正交锚点退化，再构造非刚性/跨图旋转条件下 Epred真正改变对应且后揭晓margin仍正的可实现正例，并同时保留“Gram正确、类别错误”的反例。若只能在人工注入角色/对应时成功，不能登记为通过完整M3。如果这些条件全通，再做N=4096、K≤8、L≤16原样调度小基准核调用数/时间，仍不宣称真实分割效果。这个合同当前优先级低于M2；不因正例合成成功而自动启动真实M3大实验。

真正决定价值的自然校验按 Pro先固定8个已暴露例核成本/调用/内存，再同候选全mask比较 heldout与all-role/mean；如果调度成本超过已绑定历史fine真实额外时间，降低优先级。不能删模板/候选或免κ后仍报原版结果。当前仅建议，未执行。

## 4. 唯一优先的有限下一动作

**在关机边界内，先完成合同一 M2 的一个≤16 token、单位特征、全估计到完整mask的可实现性排除检查，固定Pro参数和全部关键对照，成功或失败均记录后结束。** 本次只交合同与已有标量推导，没有实施这个下一动作。当前不新增第11项及更多名字；M3不与M2串联，不开始新参数搜索。

若主代理选择实现M2，必须先明确“保留Pro原宿主”还是另立修订版；将FoRIS pre-CRF换成MEAN、32角色换8角色、图类型减少、renderer换nearest，均是方法修订，理由和代价必须显式写出。现有九项共用入口能省工程工作，但不能替代以上语义绑定。下一次真实资源授权后，只有匹配简单完整控制与自然实例结果才决定是否继续；现在的结论是 **M2缺失量明确、M3缺失程序有退化风险，二者真实收益均未知**。
