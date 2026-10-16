# W39 F：机制排重、可保留的贡献与训练式 C0

2026-10-15。对应 [W39 计划](W39_astra_open_questions_PLAN.md)。本线只研究文献、方法与读法，未运行模型、模拟、审计包脚本、压缩证据解码、GT 诊断或新评分；未启动训练，未恢复暂停任务。本报告中的候选构造不是执行指令。

**本轮新增结论：不能以“冻结骨干、两路互补、参考自验证、内部干预、参考控制粒度、软掩码”中的任意一个概念单独定义原创性。仍可并列研究的贡献是：新的可部署证据究竟改对了哪类共同错误，以及它在相同输入、候选、监督和计算预算的近邻控制后是否仍有增量。** 下面给出逐路线的最近威胁、可证伪贡献，以及完整 C0 训练合同；不在没有比较结果时排序赢家。

## 1 检索、精读和版本身份

本线使用 Exa Search 技能，读取 `searching.md` 与 `patterns-papers.md`。截至日为 2026-10-15；若按“最近一年”理解，范围是 2025-10-15 至 2026-10-15，以下同时保留更早的基础先例，没有用时间窗口排除危险工作。

- `sources_reviewed = 130`：18 次 `web_search_exa` 的 `numResults` 之和，包含重复；本轮实际返回 130 个结果块，精确 URL 去重后 114 个。它不是 130 篇论文精读数。
- 从中筛选 13 项工作进入方法级审查，实际阅读各自与本任务相关的方法段、公式或算法。13 项是下表 13 行，不把其全部实验、补充材料和所有引用计为精读。
- 其中 7 套作者实现核到具体源模块：RSRM、FROST、FSSDINO、PAHNet、SCLIP、SegIC、IconMatting；前三套沿用 W36 已固定的只读快照并本轮再读关键源码，后四套本轮读取公开源代码。CAAP、RefCut、SPHERETRUST 本线未确认可用作者实现，不能写成已代码复现。
- 搜索中的转载、综述、无关 GitHub 仓库和同名结果没有被用作技术结论的依据。作者代码也只是阅读，没有运行。

| 工作 | 本轮固定来源与定位 | 代码身份/核查范围 |
|---|---|---|
| RSRM | [2606.24297v1](https://arxiv.org/html/2606.24297v1)，§3 SAFR/ASE/HPM | `4173c9fd38b4156b9d9200dd6ef941d88eca91bd`，[rsrm.py](https://github.com/Sparkling-Water/RSRM/blob/4173c9fd38b4156b9d9200dd6ef941d88eca91bd/model/rsrm.py#L44)：44–110、118–226、250–283、349–421 |
| FROST | [2606.31136v1](https://arxiv.org/html/2606.31136v1)，密度比、支持 LOO 与传播 | `b9ece69d7495a698c298e7cc3d16efacd4497a43`，[density.py](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/density.py#L63)：63–108、129–247；model.py 翻转合同见 W36 |
| FSSDINO | [2602.07550v1](https://arxiv.org/html/2602.07550v1)，Gram、§VI-D 选择差距 | `24858986ea55e312200079937989ad32360f09b2`，[module.py](https://github.com/hussni0997/fssdino/blob/24858986ea55e312200079937989ad32360f09b2/fssdino/module.py#L67)：67–174；[fss_oracle.py](https://github.com/hussni0997/fssdino/blob/24858986ea55e312200079937989ad32360f09b2/fss_oracle.py#L295)：295–347 |
| PAHNet | [2507.19140v1](https://arxiv.org/html/2507.19140v1)，prototype/affinity 的偏差互补与引导 | `b38d57280d77efef37b56a0a01b7e5c67f345970`，[HDMNetPlus.py](https://github.com/tianyu-zou/PAHNet/blob/b38d57280d77efef37b56a0a01b7e5c67f345970/HDMNet_PAHNet/model/HDMNetPlus.py#L115)：115–126、244–252、415–440、470–488；不是完整重审全部变体 |
| SPHERETRUST | [2609.30080 原文](https://arxiv.org/html/2609.30080)，2026-09-24，§3.1 Eq.1–4、§3.2；本次以该未带版本后缀的公开正文为准 | 只读正文；不把 release promise 当已有实现，不声称已核代码 |
| RefCut | [2503.17820v2](https://arxiv.org/html/2503.17820v2)，§3.1–3.3，尤其真实 part 组合采样 | 只读论文及作者 demo 页；未确认可核实现。作者另有标题 RefPrompt 的页面，本报告不把它当两个独立方法计数 |
| SegGPT | [2304.03284v1](https://arxiv.org/html/2304.03284v1)，任务统一与随机颜色映射 | [作者库](https://github.com/baaivision/Painter/tree/main/SegGPT)仅作发布身份核对，本轮未作源码级重审 |
| SegIC | [2311.14671v3](https://arxiv.org/html/2311.14671v3)，§3、context reversion/negative entity augmentation | `73bbcc340825e89f18e1b607cdbaaa09adc9619a`，[segic.py](https://github.com/MengLcool/SEGIC/blob/73bbcc340825e89f18e1b607cdbaaa09adc9619a/model/segic.py#L458)：458–474 反向 context，733–783 训练损失；不是所有数据管线审计 |
| In-Context Matting / IconMatting | [2403.15789v1](https://arxiv.org/html/2403.15789v1)，§3–4 | `d5197e0e2ae3b1c8ecff37f12e3c93700c9f276b`，[in_context_matting.py](https://github.com/tiny-smart/in-context-matting/blob/d5197e0e2ae3b1c8ecff37f12e3c93700c9f276b/icm/models/in_context_matting.py#L40)：40–55、257–269 |
| SCLIP | [2312.01597v4](https://arxiv.org/html/2312.01597v4)，§3.2–3.3 Eq.2–3 | `3608360267b6130c1ef18090d7289f17c771cb90`，[clip/model.py](https://github.com/wangf3014/SCLIP/blob/3608360267b6130c1ef18090d7289f17c771cb90/clip/model.py#L283)：283–313 |
| CAAP | [2603.13652v2](https://arxiv.org/html/2603.13652v2)，§2.2、§3 Eq.1–3/Algorithm 1 | 未确认作者代码；没有把搜索返回的其他 attribution 仓库误认为作者库 |
| On Identifiability in Transformers | [ICLR2020 原文](https://openreview.net/pdf?id=BJg1f6EFDB)，§3、Appendix A | 读取 accepted paper 的左零空间与概率约束推导；没有运行其分析 |
| Star Shape Prior for Graph-Cut Image Segmentation | [Veksler，ECCV2008 原文](https://cs.uwaterloo.ca/~oveksler/Papers/eccv2008Final.pdf)，§3 Eq.3–4 | 读取有向无限代价约束与 submodularity；不以年代早或任务不同排除该优化先例 |

固定 commit 是阅读身份，不是已复现性能。论文与代码的差别仍以 [W36](W36_method_directions_REPORT.md) §2 为准；本轮没有把任意代码摘要推成官方完整性能。

## 2 四个最直接近邻已经占了什么

### 2.1 RSRM 与 PAHNet：互补融合本身已不是贡献

RSRM 的 SAFR 以参考 mask 的 Fisher 判别与位置敏感程度形成层/通道权重；ASE 根据查询高置信区反向增强参考；HPM 同时匹配 global、regional、pixel 原型。固定代码还按各层标准差缩放。它已经覆盖“冻结 DINO、参考监督内部表示选择、多个压缩尺度的匹配融合”。这里的 regional 是参考外观聚类，不等于已经识别查询实体。[原文](https://arxiv.org/html/2606.24297v1)

PAHNet 明确以 prototype 与 affinity 的保守/激进偏差互补为问题，并用预训练 prototype predictor 引导 affinity learner；代码加载 SSP 模型且冻结其参数，另一部分仍训练。故“首次发现两路互补”不成立，训练合同不同也不能成为忽略它的理由。反过来，PAHNet 的分支偏差方向不能直接赋给本地 U0/M′。[原文](https://arxiv.org/html/2507.19140v1)、[作者代码](https://github.com/tianyu-zou/PAHNet/blob/b38d57280d77efef37b56a0a01b7e5c67f345970/HDMNet_PAHNet/model/HDMNetPlus.py#L115)

本地仍能保留的事实是 W27/W28 的同条件功能互补。可争取的进一步贡献是：用固定场的秩/标度交换和合法任务切换，明确什么机制解释实际收益，并构造能纠正共同错误的读出。若最简单核+原型组合或相同标度替换已兑现全部收益，则应简化方法和主张；不是再给融合起一个名字。

### 2.2 FROST：支持自标定与密度读出已是直接控制

FROST 对 FG/BG 全锚点 KDE 作类内归一化密度比，LOO 选带宽，随后传播、门控。相同硬标签和核下，设核和为 `F,B`：

```text
log(F/Nf) - log(B/Nb) = logit(F/(F+B)) + log(Nb/Nf).
```

因此“改为平衡密度”“利用完整参考分布”“支持选核宽”已有明确先例。此恒等式只解释简化点场，不能称两个完整方法等价。其支持 LOO 也没有自动成为 query 风险证书。[原文](https://arxiv.org/html/2606.31136v1)、[LOO 代码](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/density.py#L76)

若新选择/标定只在弱核控制上有效，而同输入合同的 FROST 消除了增量，相关贡献被直接削弱；还应判断增量来自 whitening、先验、带宽还是新结构。不能在比较中偷偷移除其 reference flip，再称官方完整方法。

### 2.3 FSSDINO：自分割、逆分割、选择差距都有明确实现

其 Gram 能量为 `qᵀGq = mean_j(qᵀr_j)^2`，是参考外观二阶矩，不是查询空间归属。更危险的是选择代码：用完整支持集构原型，预测参考自身算 `sp_self_miou`；用 query 伪掩码重建原型再预测参考算 `rv_miou`。论文已报告这些支持导向指标仍落后 last-layer 的选择差距。[原文§VI-D](https://arxiv.org/html/2602.07550v1)、[代码295–326](https://github.com/hussni0997/fssdino/blob/24858986ea55e312200079937989ad32360f09b2/fss_oracle.py#L295)

B0 的 blocked harmonic completion 不等于该自拟合评分：它隐藏一块真实参考标签，且预测目标是图上的补全风险。这是具体差异，值得检验；但“参考自验证内部选择”“oracle 高、选择器失败”都不能再泛称创新。要兑现贡献，必须分别控制源内泄漏、跨图风险转移以及 harmonic 风险与 SH 求解目标错位。图 head 的数目和名称不是新信息的证明。

## 3 新找到的更危险近邻

### 3.1 SPHERETRUST：冻结特征、固定候选池、无训练范围选择已有方法

该工作冻结 DINOv2 ViT-B/14，将候选看作方向特征的二分：

```text
F(M) = 1 - <mean_direction(M), mean_direction(complement M)>
C(M) = 各推断前景外观模式上，候选覆盖比例的等权平均
E(M) = 图像外边框被候选占据的比例
R(M) = z_within_pool(F) + z_within_pool(C) - z_within_pool(E).
```

前景模式通过候选多数交集形成的 core 和边框背景原型挑出，再作至多 4 个模式；`C` 随加入 patch 单调不减，单独会奖励泄漏。确切同分回退 generator confidence。这里的 coverage 是**外观模式召回**，不是每个 16×16 patch 的真实前景面积。其目标身份主要来自候选池的 prompts；它没有证明单参考完整任务语义或亚 patch 质量可恢复。以上均来自原文 §3.1 Eq.1–4，而非仅摘要。[原文](https://arxiv.org/html/2609.30080)

它直接否定“首次用冻结特征给范围候选无监督打分”这一宽泛主张。对本地原生响应或选择性编辑，最低必要控制应包含同池、同实际可用特征预算的几何分离/模式覆盖选择，适配处明确标明；其原始 SAM 候选、350 输入与本地 SH 九级候选不是相同协议，不能横减发表分数。边框先验若与本地域冲突，要按已固定规则报告，不能只给它一个有利/不利版本。

本地可能剩余的贡献是“完整 reference-defined task 带来超过 query 几何的选择增量”，或“原生 value 响应超过固定方向几何和 cut mass”。若 task-blind 的同池几何控制已解释增量，不能继续称任务结构发现。没有查到已释放代码，限制了当前源码级审查，未冒称已复现。

### 3.2 RefCut、SegGPT、SegIC：参考任务与粒度不是空白领域

RefCut 以 reference 的正/可选负 mask 生成 prompt，输入交互分割网络；它明确处理 part/whole 和对象组合歧义。§3.3 在同类不同对象的 R/Q 中随机选择真实 `1..N` 个 part，分别合并成参考与查询目标，其余 part 作负参考；评价也含单 part、相连两 part 组合和整体。因此“用参考控制粒度”“训练真实 part 合并任务”本身已有先例。它额外使用 query clicks、训练网络，部署合同与本地 training-free 一参考不同，必须明列，不能伪称已经击败或取代本地方法。[RefCut原文§3.1–3.3](https://arxiv.org/html/2503.17820v2)

SegGPT 早已用随机颜色映射迫使 in-context 规则来自示范，覆盖 object、part、instance 等任务；它不是本地冻结 DINO 轻量读出，但排除了“首次由示范决定分割任务”的主张。[原文](https://arxiv.org/html/2304.03284v1)

SegIC 更接近 C0：单 frozen VFM + trained mask decoder，并把 dense correspondence 与 geometric/visual/meta instructions 输入解码。它还训练 context reversion，即用 query 预测反向作示范；代码确有前向/反向 loss 平均。故“冻结 DINO，学一个 structured decoder”不足为新方法。[原文](https://arxiv.org/html/2311.14671v3)、[作者代码](https://github.com/MengLcool/SEGIC/blob/73bbcc340825e89f18e1b607cdbaaa09adc9619a/model/segic.py#L458)

尚可被检验的区别是：**同 RGB、同一合法 R/Q 对，仅切换真实参考目标的配对监督，能否在完全相同四张目标 mask、同结构/更新预算下，超过独立 episodic 监督与同区域加权监督，并泛化到未见任务组合。** 这不是已经确认的首创，下一节把它变成可计算假说。

### 3.3 内部干预：要区分归因、改写预测和可部署选择

SCLIP 在最后层以 `softmax(QQᵀ/τ)+softmax(KKᵀ/τ)` 替代原注意力，改善 CLIP 密集预测；代码该和的行质量为 2，不应随意描述为行随机矩阵。CAAP 在空白目标上下文中注入某 patch 的原图中层 activation，用分类 score 测影响；它不以完整 reference mask 选择查询范围。两者已分别覆盖 frozen 最后层改造和内部 activation 的功能干预，不能将这些操作本身作为新增贡献。[SCLIP原文](https://arxiv.org/html/2312.01597v4)、[代码283–313](https://github.com/wangf3014/SCLIP/blob/3608360267b6130c1ef18090d7289f17c771cb90/clip/model.py#L283)、[CAAP原文§3](https://arxiv.org/html/2603.13652v2)

E 线另核了 FSA 与 Value Zeroing 的原文/实现，具体位置归 [E 线报告](W39_E_native_response_REPORT.md)；本线不把别人完成的代码检查计为自身精读。FSA 的 attention isolation 和输出回馈尤其要求正面对照，不能只与输入遮罩 CLS 的失败比较。

新增最危险的理论先例是 Brunner 等：attention 可在 value-output 左零空间改变而不改变层输出，故 raw attention 并不由输出唯一识别。`ΔA V=0` 不是本轮首次发现。[ICLR2020原文§3](https://openreview.net/pdf?id=BJg1f6EFDB)

E 新构造增加了不同约束：`ΔA=u vᵀ`，`u` 仅两行 ±1，`v` 避开这两行且正交于 `1`、所有固定候选 coverage `a^k` 和该 head 的所有 V 通道。这样可保 row/column/self/prefix、每候选逐行 soft cut mass、native `AV` 和重放输出；隔离后仍可能改变：

```text
Δ(T_a(A)V)_i = -u_i*(1-2a_i)*vᵀ diag(a) V.
```

这是本轮由 E 推导、本线独立代入 `χ_ij=a_i+(1−2a_i)a_j` 核查的严格区别。它可以排除“原始输出或切走质量已变”的解释；不能把任意 post-softmax A 当固定 Q/K 下真实可产生的 attention，也不能由变化断言原始关系是正确语义因果。若约束退化、非负步长极小或右端为零，控制无有效扰动；这不是结构机制阴性。潜在贡献只能是该严格控制支持的**实际任务选择增量**，不是零空间技巧自己。

### 3.4 coverage 与 matting：估计对象先分清

IconMatting 已用 reference RGB+RoI prior，以 diffusion inter/intra similarity 和训练 matting head 预测无 query prompt 的 alpha。它是直接的“参考引导软输出”先例；但物理 alpha 满足 `I=αF+(1−α)B`，与二值 benchmark target 的每 patch 像素占比不同。相同均值 coverage 也不标明 patch 内目标在哪里。[原文§3–4](https://arxiv.org/html/2403.15789v1)、[作者接口](https://github.com/tiny-smart/in-context-matting/blob/d5197e0e2ae3b1c8ecff37f12e3c93700c9f276b/icm/models/in_context_matting.py#L40)

D 线提出参考条件电导、每 patch 沿固定 F 排序的 prefix、跨 patch Potts cut 联立求局部量，和 W38 的颜色类别分类是不同假说。其优化基础有明确先例：Veksler §3 Eq.3 已用有向无限代价禁止 `(f_p=1,f_q=0)`，把先后约束与 Potts/data term 一起作全局 graph cut。因此“加无限有向边后可精确 mincut”不是新颖点；F 排序偏序、参考电导及质量/排序分解的组合究竟产生新收益仍待检验。[原文](https://cs.uwaterloo.ca/~oveksler/Papers/eccv2008Final.pdf)

可争取的区别是：把“补全局部质量”与“改写 patch 内排序”分开，对同一可部署逆问题验证。若去掉 reference 条件、只用通用 RGB 电导即可得到相同收益，便没有 reference-specific 贡献；若去掉 prefix 更好，则“原排序值得保留”未获支持。不能由 W37 的 O2 容量自动证明该逆问题可识别。

## 4 C0：完整、可计算且同监督量的训练式 task-switch 设计

### 4.1 学习的对象与数据合同

这是一条**新训练设定**，用户本轮只授权研究设计，未授权启动训练或读取新训练标签。它不进入当前 training-free 主表，不改 SH、不继承或改写 W30。

一个合法训练单元为

```text
z = (R,Q, r_A,r_B, G_A,G_B, metadata), R != Q.
```

R/Q RGB 固定；`r_t` 是参考上任务 t 的完整真实 mask，`G_t` 是同一语义任务在 Q 上的完整真实 mask。两种任务来自有真实 homologous 标注的 part、whole、对象组合或真实类别；禁止凭任意裁切伪造标签。`r_A != r_B` 且 `G_A != G_B` 是响应训练层的必要条件。若 r 相同却 G 不同，保留为允许输入的歧义层，不强迫确定性模型做到不可辨识区分。

四个完整 mask 均列入标签账本。原始密集 part label 可派生多个合法二值任务，也要报告原始标注照片/像素、派生任务和曝光次数；不能把派生任务说成新免费独立样本。未来数据来源、可用域、类、照片及合法 part 对应表尚未选定；不能默许把现有 T、新照片留出或其他已保护照片用于训练。参考与查询任一角色都按照片/个体/场景的可识别分组隔离；任务组合另作未见组合划分，避免仅记住 A/B 名单。

部署每次仅给 `(R,r_t,Q)`，不给任务名字、另一任务 mask、`G_t`、query clicks 或候选 oracle。分别前向 A/B 只是评价协议，每次仍 one-reference。额外训练是合同改变，不能用“部署一张参考”掩盖它。

### 4.2 同结构基线：保留完整参考 mask 的空间读出

给出一个具体读出，供三臂完全共用；其结构本身不作新颖主张：

1. 冻结同一 DINOv3-L/16，1024 输入，R/Q 各一次，取相同版本 O24 特征 `X_R,X_Q∈R^(4096×1024)`，预处理和位置去偏定义全部固定。
2. 将每个参考 16×16 mask patch 展成 `b_j∈{0,1}^256`，**不用仅一个 coverage 标量替代完整 mask**。参考 token `m_j=W_x X_Rj+W_b b_j+PE(j)`，query token `q_i=W_q X_Qi+PE(i)`，维数 d=256；`PE` 为固定二维 sin/cos 坐标编码。
3. 参考 memory 过一层 pre-LN 4-head self-attention，残差后接 `256→1024→256` GELU FFN。Query 过一层 pre-LN 4-head cross-attention 到 memory，残差与同尺寸 FFN。每个 head 维数 64，attention 使用标准 scaled softmax；无其他 encoder 或预训练 decoder。
4. Query token reshape 为64×64×256，接 padding=1 的3×3卷积256→256、GELU、1×1卷积256→256，最后 `PixelShuffle(16)` 得1024×1024×1 logits；sigmoid 得 p。这个输出保留 patch 内256个位置，并不证明仅凭 frozen tokens 一定学得到真细节。

本例是全空间 memory，不是 W26 的全图 summary selector。完整 mask bit embedding 避免不同细节 mask 被强制压成相同 coverage 输入。fixed PE、attention、pixel shuffle 均为既有部件；参数少不等于运行便宜，reference self/cross attention 仍为 O(N²d)，训练显存、反向和时间必须单列。此处未量化为“轻量/更快”。

未来若资源不允许，须在评分前另定共同架构；不能看成绩后只给 switch 臂改结构。所有臂相同初始化配对、encoder identity、训练 quad、任务曝光、前向数、batch、更新次数、优化器和学习率计划；不单给某臂额外模型、增强、标签或更新。

### 4.3 三臂目标：将额外强调与真正配对耦合分开

每个训练单元分别计算 `p_A=gθ(R,r_A,Q)`、`p_B=gθ(R,r_B,Q)`。定义每任务完整像素 BCE 和 soft Dice：

```text
L_seg(p,G) = mean_pixel BCE(p,G)
             + 1 - (2 sum(pG)+eps)/(sum(p)+sum(G)+eps)
L_base = .5*(L_seg(p_A,G_A)+L_seg(p_B,G_B)).
```

`eps=10^-6` 仅作 Dice 数值保护、所有臂一致；BCE 从 logits 作数值稳定计算。令 `D+=G_B\G_A`、`D−=G_A\G_B`；给非空 D+/D− 等总权重，各集合内部像素等权，归一得到 `w_x`；若一侧空，只用另一侧。令 `e_t=p_t−G_t`：

```text
L_point  = sum_{x in D+ union D−} w_x*(e_A(x)^2+e_B(x)^2)
L_switch = sum_x w_x*[(p_B(x)-p_A(x))-(G_B(x)-G_A(x))]^2.

Episodic: L_base
Weighted: L_base + L_point
Switch:   L_base + L_switch.
```

三个臂对同一 quad 同批次计算两次合法任务前向。Episodic 臂已经看到了**相同四张 mask 和全部 A/B 训练任务**，只按单任务目标监督；Weighted 臂同样使用 GT changed region，因此 Switch 不能靠额外局部标签权重获益。系数 1 是这一个构造的定义，不是调参最优值；若未来改损失尺度，必须对各臂的对称预算和比较目的一起预先冻结。

该构造有一个精确、且限制创新叙事的分解：

```text
L_switch - L_point = -2 sum_x w_x e_A(x)e_B(x).
```

因此它不产生新标签，只加入 A/B 误差的交叉耦合。改变区域上标签相反，概率误差乘积非正，Switch 比独立平方项更强调两次同时满足；但是在共同区域若两次同错 `e_A=e_B`，差分损失为 0。它不能单独修复共同 FN/FP，完整 `L_base` 和共同区域评价必须保留。

若模型忽略 reference、令 `p_A=p_B`，在 changed region 上 Switch 残差始终 ±1；有明确不可降低的目标残差，但这不证明优化一定找到利用 reference 的参数。普通 episodic A/B 真标签本来也会惩罚忽略示范，因此必须由上述同预算比较证明耦合有增量。泛称“迫使模型使用参考”不是贡献证明。

本线没有新增 RGB 合成、伪负类或 label shuffle。随机把无语义对应的 R/Q 标成同一任务，会更改学习目标，不能当公平的普通监督控制。

### 4.4 未来一轮最小区分读法，不执行

正式执行前还需确定并冻结合法训练来源/成员、训练与开发照片隔离、更新预算、优化器、随机性处理和一次主端点；本轮不编造足够样本量、统一相关门槛或新照片访问权。

最低必要比较就是以上三臂。每个合法评价 pair 产生完整2×2：`J_tg=IoU(A_t,G_g)`，报告两个 matched `J_AA,J_BB`、其 min 和

```text
specificity = .5*(J_AA+J_BB-J_AB-J_BA).
direction = sum_{G_A△G_B}(G_B-G_A)*(A_B-A_A) / |G_A△G_B|.
```

另完整报告共同 FG `G_A∩G_B` 与共同 BG `complement(G_A∪G_B)` 的破坏；同一自然基线共有的错处是否被改正，按增真/增假/删真/删假记录。完整 benchmark 分数仍按官方槽池化 I/U，不把 specificity、direction 或逐例 IoU 平均代替主指标；pair 等权只作任务诊断。嵌套 part/whole 可以共享排序再改切点，不把成功切换自动解释成新增排序信息。

| 未来结果 | 能支持什么；对下一步的限制 |
|---|---|
| Switch 同时胜 Episodic、Weighted，且 held-out 组合的 matched 指标/共同区域净收益改善 | 支持该监督耦合有增量；再查是否超出已训练的 composition 与参考外观，不立刻称独立新范式 |
| Switch 胜 Episodic、与 Weighted 相当 | 优先解释为困难改变区域被多强调，不能主张 switch coupling 独有贡献 |
| 三臂都胜 training-free，但互相相当 | 支持训练结构化读出的价值；没有新 task-switch 方法贡献，训练合同必须留在独立表 |
| specificity/direction 上升但 matched IoU 或共同区域变坏 | 学会了反应，未证明更正确，不能升级方法 |
| 只在已见组合/照片近邻正，未见组合消失 | 学到的可能是语义组合记忆或参考外观捷径，收窄泛化主张 |
| 没有同预算增量 | 关闭该配对损失构造，不由此宣布训练方法或 reference granularity 全部无效 |

## 5 多路线并列的贡献与直接反证

下表是研究合同，不是建议先押其中一个。各线更完整算子、条件和拒识规则由分线报告负责。

| 路线 | 已占空间/最危险近邻 | 余下可检验的具体贡献 | 最低必要区分；什么结果直接否定该贡献 |
|---|---|---|---|
| A 互补与任务干预 | RSRM 多粒度融合；PAHNet prototype/affinity；FROST 标度/先验 | 相同 unary 下分辨秩与标度的作用，找到真实任务依赖的共同错误修正 | 固定八臂分位交换+合法同 RGB 2×2。若标度替换/简单融合解释全部收益，不能声称新增语义排序；若 task-blind baseline 同样切对，reference-specific 贡献未立 |
| B0 关系选图 | RSRM reference 选表示；FSSDINO support-self/reverse 选层；经典 harmonic propagation | 隐藏参考标签的补全风险能否选择跨图有效关系，并量化目标错位 | h0、均匀 head、实际 source 选图，固定 unary/λ；oracle只诊断容量。若源内更好但 query 更差，否定该 transport；若 oracle 无余量，关闭此图族，不关闭全部关系信息 |
| B 闭包/森林 | graph propagation、watershed、query 分组均已有；纯度与可选择性不同 | 对多个可能关系保持不确定性，只接受有条件可辩护的相对 IoU 编辑 | 必须有真实可见证据预算/保留项与未覆盖拒识。若纯划分/GT终端有效、实际终端或选择无净益，贡献停留 oracle，不能称可部署成员识别 |
| C 选择/标定 | FROST LOO；FSSDINO 选择差距；SPHERETRUST 同池评分 | 有明确迁移条件与不确定集合的实际净收益，而非仅相关 | 同候选自然、几何评分、实际保守编辑/选择；若只降选择 regret proxy、毁掉正确例或共同错未改善，否定实际编辑贡献；证据预算未识别时不写无条件安全 |
| D 局部质量逆问题 | IconMatting soft output；RGB graph cut；Veksler 有向偏序约束 | reference 电导与 per-patch F-prefix 把正确量变成可部署完整输出 | 与相同 unary 的通用电导、去 prefix 对照；质量误差与位置误差分列。若 generic 电导相同，reference-specific 性无增量；去链更好则保序贡献不立 |
| E 原生响应 | SCLIP 改最后层；FSA isolation；CAAP/Value Zeroing；Brunner value-nullspace | 在固定 F/候选/native 输出/cutmass 后，value 边配对提供可转移的完整范围选择信息 | 同池几何/cutmass、线性/全块、质量替换与有效约束破坏。若只 zero-cut 有效，优先幅度解释；若严格保持控制扰动无效，不能冒称阴性；若实际选择不胜，关闭此选择器 |
| C0 训练式 task-switch | RefCut 真 part 组合；SegGPT task context；SegIC frozen VFM+decoder | 同完整标签预算下，显式正确方向耦合超越困难区域加权，并泛化未见组合 | §4 三臂。若 Weighted 解释收益或只反应更强而不更正确，否定该贡献；不得混进 training-free 主表 |

主张强度必须停在证据处：算法定义清楚不等于经验成立；条件命题不等于迁移条件已验证；与最近邻有差异不等于有增量。严格保持的机制控制可以成为研究贡献的一部分，但不能代替完整输出收益。

## 6 当前真正完成和未完成的边界

本轮完成：13 项近邻的具体机制占位；补入 SPHERETRUST、RefCut、SegIC、IconMatting、attention identifiability 和有向 graph-cut 先例；跨线核查 E 的严格保持控制与 D 的偏序优化；C0 的完整输入/架构/损失、等标签预算控制和可证伪读法。其余线不因这些排重自动失去研究价值。

仍未完成：任何方法的实现身份、运行资源冻结、合法训练/任务对清单、可部署跨图假设及新实证。没有找到与某方案名字一样的论文不是原创性证明；本线也没有穷尽全部优先权。先保留多个可区分构造，再用预写的一轮结果决定哪项贡献成立，不提前将 QK→V、B0、coverage 或 C0 定为主方法。

## 附：Exa 查询账本

`sources_reviewed` 只累计下表 `numResults`，fetch 不加这个数；精确 URL 重复也不扣请求数。

| 调用 | numResults | 原查询 |
|---|---:|---|
| 1 | 5 | category:research paper RSRM Robust Support Reference Matching DINOv3 few shot segmentation 2026 official GitHub |
| 2 | 5 | category:research paper FROST frozen DINOv3 training-free segmentation kernel density 2026 |
| 3 | 5 | category:research paper FSSDINO Semantic Selection Gap few shot segmentation DINOv3 layer selection 2026 |
| 4 | 5 | category:research paper PAHNet prototype affinity learning few shot segmentation ICCV 2025 official code |
| 5 | 12 | category:research paper few shot segmentation reference mask task ambiguity part whole counterfactual support masks same image granularity training |
| 6 | 10 | category:research paper SegGPT Matcher Visual Prompting in context segmentation multiple tasks part whole reference mask training official code |
| 7 | 10 | category:research paper training free segmentation reference self validation graph attention head selection harmonic completion support mask |
| 8 | 5 | category:research paper RefCut RefPrompt reference guidance interactive ambiguity segmentation task semantic granularity official GitHub |
| 9 | 5 | category:research paper SPHERETRUST Can Frozen Hyperspherical Features Guide the Selection of Pseudo Masks 2609.30080 official code |
| 10 | 10 | category:research paper candidate mask attention isolation FSA SCLIP CAAP activation patching visual attribution official code |
| 11 | 10 | category:research paper semantic matting reference based image matting few shot alpha coverage patch mixture pretrained features frozen support mask |
| 12 | 10 | category:research paper SegIC in context segmentation frozen foundation model decoder training synthetic support query masks task prompt granularity |
| 13 | 8 | category:research paper binary segmentation graph cut foreground prefix ordering directed infinite edges star shape geodesic convexity constraints |
| 14 | 10 | category:research paper few shot segmentation counterfactual support mask consistency loss same query part whole training reference switching |
| 15 | 5 | Causal Attribution via Activation Patching CAAP Amirmohammad Izadi official GitHub 2026 |
| 16 | 5 | RefCut reference guidance Zheng Lin official code github partimagenet |
| 17 | 5 | In-Context Matting IconMatting official github code |
| 18 | 5 | category:research paper On Identifiability in Transformers attention non identifiable null space value matrix effective attention Brunner 2020 |
| **合计** | **130** | 18 次；130 返回块；114 个精确 URL；筛选/方法段精读 13 项工作，7 套源模块核查 |
