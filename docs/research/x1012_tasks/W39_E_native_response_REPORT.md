# W39 E：原生混合响应的可辨识条件与范围选择

2026-10-15。对应 [W39 计划](W39_astra_open_questions_PLAN.md) E 线。本报告只读论文、作者代码、本机安装源码和已发布报告；**没有构造模型、运行实验或模拟、解码数据/GT、读取新结果、生成预测或评分**。只编辑本文件。不将本线排在 B0、覆盖率、闭包或选择性编辑之前。

## 1 本轮推进的对象

[W36 §5–§7](W36_method_directions_REPORT.md) 提出了“候选边界切断最后块的哪些原生混合，参考的同类响应能否选对范围”。本轮把三个缺口推进到具体对象：

1. 静态核实实际 timm DINOv3 的完整末块，并给出可流式计算的精确干预公式，区分守恒量与非守恒量。
2. 构造同时保持**原始完整末块输出、全部固定候选的逐行 soft cut-mass、行列和、自环及 prefix**的 attention 反事实；它仍可改变候选隔离响应。这个控制排除了比普通打乱更明确的替代解释。
3. 把“参考响应可迁移”拆成有反例、有间隔条件和合法任务切换读法的假说；不以能量距离的名称替代识别问题。

闭合的是算子、一个受约束控制和条件命题。没有闭合的是：真实冻结权重的响应是否携带可用任务范围信息、单参考是否足以识别它，以及所得收益是否值得计算成本。不能将抽象 attention 反事实误写成两张真实图像具有相同 O24 的证明。

## 2 已有负面结果限制了什么

[旧 context 报告](../archive/TASK_CONTEXT_COMPATIBILITY120_20261012.md) 已经在真实 X23、Q/K/V、原投影、残差、MLP 和 final norm 上完成过参考角色条件化：参考 FG/BG 的跨图 V-context 各与 query 原生 context 混合 1/2，以“哪个角色更少改变原表示”输出标量。它对正确参考 mask 有任务响应，却输给简单 O24 强控制，且 GT 阈值上界亦不足。其非线性相对线性零阈值的主要完整输出增量，已由“线性排序 + 非线性分布”解释。

[mask-trace 报告](../archive/REFERENCE_MASK_TASK_TRACE_20261010.md) 同样区分了“参考 mask 确实进入推理”与“正确分割两个合法任务”。它已有同 R/Q RGB、真实 A/B 类或兄弟部件以及等面积错支持控制；这些是开发证据，不是未使用的独立确认。

所以本线不能再以“完整块重放”“mask 引起变化”“比线性弱控制好”作为贡献。本轮的新对象是**同图候选边界对原生 value 混合的干预响应**：不把参考 V 注入 query，不裁剪 RGB，不比较 CLS，不修改固定 F 或候选池。跨图相容性仍是共享风险，尚未被解决。

## 3 实际最后块：源码合同而非运行验证

静态读取本机 timm **1.0.30**：vit_large_patch16_dinov3 在该安装中创建 **Eva / EvaBlock / EvaAttention**，不是标准 vision_transformer.Block。本项目配置的 architecture 与之相同。源码定义为24块、D=1024、H=16、head dim=64、bias-free fused QKV、4个 register 加 CLS、DINOv3 RoPE、half rotation layout、LayerNorm epsilon=1e-5、GELU MLP。q_norm/k_norm 与 attention 内部 norm 在此默认配置是 Identity；实现必须仍调用实际模块，不能推广到别的变体。

定位：[本机 EvaAttention](/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/lib/python3.13/site-packages/timm/models/eva.py:222)、[EvaBlock](/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/lib/python3.13/site-packages/timm/models/eva.py:433)、[L/16 构造](/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/lib/python3.13/site-packages/timm/models/eva.py:3135)、[RoPE](/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/lib/python3.13/site-packages/timm/layers/pos_embed_sincos.py:1122)、[当前 wrapper](../../../src/ics/data.py:13)。

令 X 是**未经过 model.norm 的第23块输出，包含全部 prefix**。不能用 norm=True 的 O23 替代 X，也不能把 SH 去位置后的 O24 当作末块输入。完整依赖为：

\[
\begin{aligned}
Y&=\operatorname{norm1}(X),\quad
(Q,K,V)=\operatorname{reshape}(\operatorname{qkv}(Y)),\\
\bar Q&=\operatorname{q\_norm}(Q),\quad
\bar K=\operatorname{k\_norm}(K),\\
\widetilde Q_P&=\operatorname{RoPE}(\bar Q_P),\quad
\widetilde K_P=\operatorname{RoPE}(\bar K_P),\\
A_h&=\operatorname{softmax}
       (\widetilde Q_h\widetilde K_h^\top/\sqrt{64}+B),\\
C_i&=\operatorname{concat}_h\sum_j A_{hij}V_{hj},\\
T&=X+\Gamma_1\,\operatorname{proj\_drop}
       [\operatorname{proj}(\operatorname{attn.norm}(C))],\\
Z&=\operatorname{model.norm}
       [T+\Gamma_2\operatorname{MLP}(\operatorname{norm2}(T))].
\end{aligned}
\]

RoPE 只作用 patch，prefix 的 Q/K 原样拼回；V 不旋转。B 是实际 mask/bias，当前全图非因果路径通常为空，但应核对后记录。推理 eval 下 attention/projection dropout 和 drop-path 关闭；不能在 train 模式复用本推导。若存在 attention gate，必须保留它；当前构造没有 gate。Gamma 是加载后逐通道参数，不能用初始化值 \(10^{-5}\) 代替其实际值。

这里保留的是同一 timm 实现的原生数值定义。[timm 模型卡](https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m) 特别说明原仓库与 timm 的 RoPE periods 精度差异及 QKV bias 处理。不能从“相同权重名”推导两个实现逐位相同，也不应在探针里擅自改为另一实现的 periods。

## 4 干预、守恒与可计算形式

### 4.1 唯一主算子

参考合法完整 mask 与每个 query 候选在同一 CLI-1024 画面，以16×16平均池化得 \(a_i\in[0,1]\)。两侧定义相同。对 patch \(i\ne j\)：

\[
\chi_{ij}(a)=a_i(1-a_j)+(1-a_i)a_j
             =a_i+(1-2a_i)a_j,\qquad
r_{hi}(a)=\sum_{j\in P,j\ne i}A_{hij}\chi_{ij}(a).
\]

\[
A^a_{hij}=A_{hij}(1-\chi_{ij}),\quad
A^a_{hii}=A_{hii}+r_{hi}.
\]

所有 prefix 行及 patch→prefix 列不变；self 不先衰减再补。以 \(A^a\) 替代 A，Q/K/V、原投影及块内其他模块固定。数学上非负且每行和不变；**列和、attention 熵、对角质量、context 范数、最终特征范数和语义正确性均不守恒**。

Hard a 时是切掉异角色 patch 边并回填 self；soft a 时是角色不一致概率形式的确定 gate，不能称真实 patch 内像素解混。尤其 \(a_i=.5\) 时该行对其他 patch 的 gate 恒为 .5；即使邻居也有 .5 coverage，仍会切混合。相同 coverage 的不同像素 mask 无法被这个测量区分。Prefix 提供的跨区域通道、早23块的混合仍然存在，不能称整编码器实体隔离。

### 4.2 精确 context 差与幅度解释

\[
\delta C_{hi}(a)=
\sum_{j\in P,j\ne i}A_{hij}\chi_{ij}(a)(V_{hi}-V_{hj}).
\tag{1}
\]

当前 attention 内 norm/gate 为 Identity，故
\(\delta T_i=\Gamma_1 W_O\operatorname{concat}_h\delta C_{hi}\)；
输出投影 bias 在差中消去。不同变体若有非线性内部 norm/gate，应比较实际输出，不能仍用该线性差式。

\[
\|\delta C_{hi}\|
\le r_{hi}\max_{j:A_{hij}\chi_{ij}>0}\|V_{hi}-V_{hj}\|.
\tag{2}
\]

同 cut-mass 可以相消或产生很大响应；少切质量也可能切到高差异 value。反过来，若相关 value 相同，或投影/LayerScale 将差消去，则候选响应全零。完整 MLP/final norm 还可能消去或放大差异，响应大小不是实体概率或 IoU 增益。

式(1)可不显式保存每个候选的 \(A^a\)。令 patch 子矩阵上
\(p=A1,\ n=AV,\ m(a)=Aa,\ \ell(a)=A\operatorname{diag}(a)V\)，则

\[
\delta C_i=
[a_i p_i+(1-2a_i)m_i]V_i
-a_i n_i-(1-2a_i)\ell_i(a).
\tag{3}
\]

可将 self 纳入式(3)，因其 \(V_i-V_i=0\)；但报告 cut-mass 必须扣掉 \(A_{ii}2a_i(1-a_i)\)，仍以排除 self 的 r 为准。p、n 一次缓存，按 head/候选流式计算。它避免候选级 A 复制，不改变 \(O(mHN^2d)\) 主复杂度。

## 5 同时保 native 输出与 soft cut-mass 的控制

### 5.1 普通 circulation 不够

两行、两列的 \((+\epsilon,-\epsilon;-\epsilon,+\epsilon)\) 保行列和。对第 i 行，soft cut-mass 改变为
\(\epsilon(1-2a_i)(a_j-a_k)\)，通常不为零。只宣称“行列边际相同”不能排除主测量其实读到不同切走质量。更强控制必须对**全部事先固定候选**一起约束，不能各候选生成不同伪图后称其他信息固定。

### 5.2 显式构造

对一个 head，固定两个 patch 行 \(p,q\)，令 \(u=e_p-e_q\)。取与这两行不相交的目的集合 J。将 m 个固定 query 候选的 coverage 记为 \(a^{(1)},\ldots,a^{(m)}\)，在 J 上定义

\[
G_J=[1,\ a^{(1)},\ldots,a^{(m)},\
                 V_h[:,1],\ldots,V_h[:,d]]_J .
\]

取 v 支持于 J 且 \(G_J^\top v=0\)。可用固定公开 hash 产生、与 GT 无关的方向 g，经 \(I-G_JG_J^+\) 投影并归一化 \(\|v\|_\infty=1\)。不选择一个“最会毁伤”的方向。定义

\[
\Delta A_h=\epsilon u v^\top,\qquad
0<\epsilon\le
\eta\min_{\Delta A^0_{ij}<0}
  {A_{hij}\over-\Delta A^0_{ij}},\quad
\Delta A^0=uv^\top,\quad 0<\eta<1 .
\tag{4}
\]

η、J 分块、行对和 hash 版本在未来计划中一次固定，不按 GT/效果改变。维数为 \(|J|-\operatorname{rank}(G_J)\)，仅当它正且投影非零才有该方向；\(|J|>1+m+d\) 是充分的代数条件，不是非退化强度保证。

式(4)使 \(A+\Delta A\ge0\)。逐项得到：

\[
\Delta A1=0,\quad \Delta A^\top1=0,\quad
\Delta A\,a^{(k)}=0,\quad \Delta A V_h=0 .
\tag{5}
\]

由于行/目的支持不相交，diagonal 不变；prefix 不变。由 chi 的线性式：

\[
\sum_j\Delta A_{ij}\chi_{ij}(a^{(k)})
=a_i^{(k)}\sum_j\Delta A_{ij}
+ (1-2a_i^{(k)})\sum_j\Delta A_{ij}a_j^{(k)}=0.
\tag{6}
\]

**全部候选每行每 head 的 r 保持**，不只是均值或直方图。又因 \(\Delta AV=0\)，未干预 native context、原 W_O 后更新、完整末块 Z 均保持，故原始 O24、F、U0/M′、候选及像素几何也保持。

但隔离后的 context 差为

\[
C^a(A+\Delta A)-C^a(A)=
-\epsilon u_i(1-2a_i)\,
                   v^\top\operatorname{diag}(a)V_h.
\tag{7}
\]

通常式(7)不为零：该控制保普通混合的 value 均值，却改变“哪些 value 被候选边界分开”。完整块可能进一步消去它，仍须报告实际响应变化。

不运行的手算见证：四个目的 token 的
\(a=(0,0,1,1)\)、一维 \(V=(1,-1,1,-1)\)、
\(v=(1,-1,-1,1)\)。则 \(v^\top1=v^\top a=v^\top V=0\)，
而 \(v^\top\operatorname{diag}(a)V=-2\)。与目的不重合且 \(a_i=0\) 的行，其式(7)为 \(2\epsilon u_i\ne0\)。严格正原 A 及足够小 epsilon 满足非负。这证明响应不只是 native AV/cut-mass 的函数；**不是实际 DINO 图像或可用增益的存在证明**。

### 5.3 退化与解释边界

- \(a_i=.5\) 的行在式(7)下必不变化；若 a 在全部 patch 为常数，\(\operatorname{diag}(a)V\) 落在 V 的空间，控制整体无信息。
- 若 \(\operatorname{diag}(a)V\) 已在 G 列空间，或 v 碰巧与它正交，也不改变隔离响应。多候选约束缩小可行空间。
- FP32 softmax 的零/极小项可能将可行步长压到零/极小。非零 nullspace 不代表足够强的非负扰动，不可看效果后换更有利方向。
- 记录 rank/nullity、投影残差、epsilon、\(\|\Delta A\|/\|A\|\)、保持量误差及候选 context/最终响应变化。没有破坏测量时，性能不降不能否定结构，也不能自动判控制通过。
- 该 A 反事实不保证有同 head dim 的 Q′K′ 实现，更没有另一张真实 RGB。它是模型内受控敏感性分析，不是真实任务随机化因果效应。
- 不保持 attention 熵、logit 内积、边的空间距离或所有未列图统计，单一控制不能排除它们。

参考与 query 必须应用同一预定构造后再互相匹配；只打乱 query 额外引入两侧处理不一致。原 F/候选不由反事实重算。可用固定不重叠行对/目的块顺序作一遍组合；每次满足同一 G 约束，保持量数学上可叠加。未实际实现或验证该组合。

同一构造不等于两侧有相同扰动强度：参考只约束完整 mask，query 同时约束 m 个候选，nullity 和非负步长可能不同。因此还须逐角色披露实际 context/响应改变量；控制下降也可能包含两侧响应分布漂移，不足独立证明正确语义配对。不能用补齐假 mask 或隐含重新调幅，声称已经消除这个限制。

[Brunner 等，ICLR2020 §3](https://arxiv.org/pdf/1908.04211) 已证明 attention 的 value/output 左零空间及概率约束内非唯一性。**\(\Delta AV=0\) 和 nullspace 扰动本身不新**；这里推进的是追加全部候选 soft cut-mass、列和及 self/prefix 约束后，式(7)的隔离响应区别与可计算控制。不据此宣称新 attention 理论。

## 6 从响应到选择：不能省掉任务识别

### 6.1 固定可部署定义

候选池用 [W36 §5.3](W36_method_directions_REPORT.md) 的 W31 修正版九尺度：s=0 严格保自然完整 mask，其余由固定 F 的稳定 top-k、\(k=\operatorname{clip}(\lfloor |A_0|2^s+.5\rfloor,0,1024^2)\) 产生并去重。没有 query GT、候选重生成或 max-256 限制。空自然 mask、195个既定细结构回退若纳入原分母，都保原输出；披露有效测量覆盖率，不从分母剔除。

令 \(b_i(a)=\operatorname{unit}(Z_i^a)-\operatorname{unit}(Z_i)\)。以 a 与1−a的归一化质量，形成 FG/BG 两个响应分布 \(P_{I,F}(a),P_{I,B}(a)\)。参考用完整合法 mask 的同一定义，query 用候选。定义

\[
\mathcal E(P,Q)=2E\|p-q\|-E\|p-p'\|-E\|q-q'\|,\quad
L(a)=\tfrac12\mathcal E(P_{R,F},P_{Q,F}(a))
      +\tfrac12\mathcal E(P_{R,B},P_{Q,B}(a)).
\tag{8}
\]

期望是包含对角的加权有限和；不用无偏 U-statistic，不把相邻 token 当独立样本。选择最小 L；精确平局优先自然，再较小 |s|，再较小 s。能量距离无带宽，却仍选择了距离、角色权重及被忽略的空间排列；“无带宽”不等于无归纳假设。

双角色质量不正、自然响应数值不可定义时整例回退；其他单候选角色为空/数值失败，只将该候选标不可选，不删除任务。参考与 query 总背景面积被归一化移除，但背景内容组成没有被匹配。

### 6.2 严格失效情形

**零响应反例。** 若最后块 patch V 全相同，式(1)对全部候选为零，式(8)全部相同。即使正确候选在池内、与自然 IoU 差很大，规则仍保自然。零 value 投影能构造这种 Transformer；没有 DINO 训练目标排除该逻辑反例。它不声称当前权重实际退化。

**角色补集对称。** \(\chi(a)=\chi(1-a)\)，干预 Z 完全相同。读出只能靠 FG/BG 权重区分 a 与补集；若参考两角色响应分布相同，二者 L 相同。算子本身不判定“哪侧是指定目标”。

**相同分布不同空间。** 两候选若有相同 FG/BG 加权响应多重集，式(8)相同，哪怕布局与 GT 不同。分布距离不恢复左右身份或 tokenization 已消去的细边界。

**跨图不变性失败。** 正确 query 目标可因视角、部件或周围场景而有不同响应；错候选可因相同低纹理背景更接近参考。小 L 不必对应高 IoU。完整背景尤其没有跨图同分布协议保证；FG/BG各1/2不是 Bayes 推导。

### 6.3 一个条件性正结果

令
\[
d(P,Q)=\sqrt{\tfrac12\mathcal E(P_F,Q_F)
                   +\tfrac12\mathcal E(P_B,Q_B)} .
\]
在欧氏空间有限一阶矩的分布上，这是两个能量距离 Hilbert 嵌入组成的乘积距离，最小 d 与最小 L 等价。若存在任务级响应描述 \(P_\star\)，且

\[
d(P_R,P_\star)\le\epsilon_R,\quad
d(P_Q(a^\star),P_\star)\le\epsilon_Q,\quad
\inf_{a\ne a^\star}d(P_Q(a),P_\star)\ge\gamma,
\]

三角不等式给出：当
\[
\gamma>2\epsilon_R+\epsilon_Q
\tag{9}
\]
时该规则选中 \(a^\star\)。这暴露了**类内跨图稳定性加候选间分离**，没有证明 DINO 满足它。只证明“真/错响应不同”而无正确方向或分离大于跨图漂移，仍不足选择。若 \(a^\star\) 只是最佳候选而非完整 GT，只保证选中它，不保证任意大小的收益。

一张参考内数千 token 不提供多张独立参考/任务实例，不能据此估计 \(\epsilon_R,\epsilon_Q,\gamma\) 的跨图置信界。无标签 query 的小 L 也不能验证式(9)，因“哪个是真候选”正是未知量。

### 6.4 可证伪规则

| 必要环节 | 未来固定检验 | 何种结果否定本构造的解释 |
|---|---|---|
| 固定池有余量 | 预测先封存，后以同池 oracle 标容量 | 池容量不足，不能据 selector 阴性推编码器无信息 |
| 超出旧证据的方向 | 主 L 对自然、同池 J、逐 head cut-mass 分布同形式距离 | 仅优于弱线性/零场而不胜完整控制，无实际增量 |
| value 配对而非质量驱动 | §5 保全部候选 r、native Z/F，两侧同法处理 | 确认隔离响应确被改变后却无选择区别，不支持此配对解释 |
| 参考描述合法任务 | 同 RGB 真实 A/B，公共候选并集，完整2×2正确/交叉参考 | 内部变化而正确A/B不分别改善对应完整输出，不支持任务范围识别 |
| 跨图迁移 | 开发内事前固定不同合法参考，每次仍单参考 | 参考变化大于候选分离，或较小 L 系统选错，否定式(9)经验适用 |
| 效用兑现 | 固定槽池化完整 I/U、四编辑、基线正确例毁伤 | 相关/排序变好但完整效用不增，不构成新方法 |

这是独立可失败的环节，不是统一相关门槛；不先筛正相关才计算完整输出。全部预定臂均须给完整结果。

## 7 原生等价性、数值退化与成本合同

### 7.1 未来验收，现未运行

1. 保存源码版本、权重身份、预处理、真实 X23、RoPE 张量/periods、prefix 数、模块类型；原生 O24 与位置去偏 O24 分开。不从 O24 反推 X23。
2. 不接触 query GT 的固定图像样本上，比较 native 块、显式 softmax 无干预 replay、式(1)/(3)零干预及完整 Z。检查 Q/K 前后 RoPE、C、T、MLP 更新、norm 后 patch/prefix；只查最终平均余弦不足。
3. 相同非 fused FP32运算图/顺序的 replay 争取逐位相等。与 fused SDPA 另报最大绝对/相对误差，不把理论等价写成逐位等价。工程草案为 FP32 输出 atol=3e-5、rtol=3e-5；超出终止实现验证，不按 GT 改容差。正式执行仍需冻结设备/后端及容差适用范围。
4. 未来恒等性质验收：a全0/全1无干预；a与1−a的A相同；每行守恒；prefix及self处理正确；constant V零变化；式(1)/(3)相同；式(5)–(7)相同。此处不执行 synthetic fixtures。
5. 干预 A 后不再 softmax，否则成为另一算子。半精度、dropout、错误 prefix、重复 RoPE、norm 后 X23 再输入 norm1均会造伪响应。

若所有 b 的向量数值误差不超过 e、权重固定，则每个 pair距离误差至多2e，式(8)每候选误差至多8e；两候选 L 差在16e内不能称稳定排序，标数值并列并按固定规则处理。e需来自预定精度参考的实际误差上界；观测最大误差只能称经验界，不能冒充全部输入保证。

单位化前出现非有限/极小范数时，不靠 clamp 造任意方向：按§6.1回退/拒选。距离用一致精度和固定分块顺序；能量距离小负值仅在预定舍入界内归零，超界判失败。NaN不能改成最佳分，失败例不能删除。zero-cut、renorm-cut、self-return分别报告；只有zero-cut成功不解释成原生实体边界。

### 7.2 计算账本

N=4096、D=1024、H=16、m≤9；以下为主导乘法量级，**不是测得时延**：

| 项 | 量级与实现后果 |
|---|---|
| 两图原编码 | 2次完整编码并观测X23/QKV；不能把首次编码省略或算为0 |
| QK、native context | 各约 \(N^2D\approx17.2\) G MAC，一次/图，attention可按head/行流式 |
| 每候选加权context | 式(3)仍约 \(N^2D\) MAC，另加Aa；省重复QKV/QK，不省value混合 |
| 每候选投影+MLP | 约 \(ND^2+8ND^2\approx38.7\) G MAC，保原GELU及norm |
| 完整响应距离 | 参考内可复用；每query候选仍有R–Q、Q–Q两个 \(N^2D\) 级距离，约34.4G MAC及开方/归约 |
| 内存 | FP32一head A约64MiB，16heads约1GiB；单N×D响应约16MiB，另有模型/X/QKV/MLP/设备开销 |
| 控制成本 | G投影/分解、非负步长独立计费，不能仅给慢控制证明主法快 |

估计忽略5个prefix。两角色可复用同一pair距离矩阵再加权；仍可能由距离而非replay主导。改随机投影、token子采样或近似能量距离会改变selector，须另定误差合同，不能负结果后挽救。

项目不建磁盘特征缓存；未来逐图对读取/编码，在内存复用后释放。每候选末块、距离、控制、CRF的实际时延/峰值分开计入端到端。此设计没有“更快”的证据。

## 8 最近先例与创新性边界

本线用Exa四个检索请求，numResults=5+5+5+5，故 **sources_reviewed=20**。这是技能要求的请求结果数累计，不表示20篇都精读；随后定向fetch原论文、作者代码、模型卡。没有运行作者程序。

| 检索请求 query | numResults |
|---|---:|
| category:research paper CAAP causal activation patching visual attribution 2603.13652 official paper code | 5 |
| category:research paper FSA attention isolation feedback 2508.20265 ICCV 2025 training free segmentation official implementation | 5 |
| category:research paper Value Zeroing Measuring Context Mixing in Transformers EACL 2023 code | 5 |
| DINOv3 timm vit_large_patch16_dinov3 QKV RoPE LayerScale source attention implementation | 5 |

| 原工作 | 已核对的操作 | 对本线的限制 |
|---|---|---|
| [Value Zeroing，EACL2023](https://aclanthology.org/2023.eacl-main.245/) | Q/K和attention固定，单token所有head value置0，完整层原/反事实输出cosine distance得混合矩阵。作者 [scoring/value_zeroing.py](https://github.com/hmohebbi/ValueZeroing/blob/57d448daf9541e88062883f8374341566e4c06ee/scoring/value_zeroing.py) 逐层逐token调用；[BERT修改](https://github.com/hmohebbi/ValueZeroing/blob/57d448daf9541e88062883f8374341566e4c06ee/modeling/customized_modeling_bert.py)在context matmul前置0；输出行归一化无一般零行保护 | 全块输出、V依赖、冻结/无标签混合测量均非新。self-return与共同边界不等于单token zeroing，但仍须证明任务范围用途 |
| [CAAP v2](https://arxiv.org/html/2603.13652v2) | 原图单patch/邻域内部activation置入blank目标context，跨选定中间层patch，读classifier目标类分；非完整单参考范围选择 | 视觉内部干预、非输入遮罩、下游响应有先例。不以“因果”占新颖；作者代码由 [F线](W39_F_novelty_REPORT.md)交叉核验，不冒称本线精读其全部实现 |
| [FSA，ICCV2025](https://arxiv.org/html/2508.20265v1) | 原/均匀attention分支输出相减后softmax，KL得patch相似，稀疏反馈。作者 [transformer.py](https://github.com/chi-chi-zx/FSA/blob/c274a6bac4608a4851e86141092ed2fee0faa698/open_clip/transformer.py) 的custom_attn/FSA/attend_v_ens用原/均匀logit scale **20/35**、p=.45剪枝、\((A_{\rm new}A_{\rm orig}+A_{\rm orig}A_{\rm new}+A_{\rm new})/3\) ensemble | 输出响应分离、训练自由最后层反馈已有分割先例。该实现非原生完整残差/MLP同算子回放；uniform相减经非线性也非严格因果加法分解 |
| [Brunner等，ICLR2020](https://arxiv.org/pdf/1908.04211) | output/value左零空间、概率单纯形内alternative attention、effective attention投影 | 输出保持依托既有原理；只能争取任务约束控制及实际决策增量 |

FSA/CAAP的分类或文本合同不同，但不免除机制排重。SCLIP等最后层改造由F线统一核对；没有检到完全相同selector不等于已证原创。

可争取的最小主张是：**固定单参考任务、原外观场和候选族下，边界value配对响应提供attention质量及旧selector无法解释的实际范围决策增量**。须完整输出、保持量控制、合法task-switch支持。只剩敏感性图或新距离匹配不足为主线。

## 9 最小可否定实验：只预写，未启动

不重定W36的90例抽样，不从T/holdout凑任务。未来若执行，另立计划冻结开发S身份、固定池、合法A/B成员，不能按oracle余量选例。沿用统一后处理/指标；科学判别不先设统一相关门槛或每域全正。

最少五个**完整可部署输出**：自然A0、同池J、cut-mass分布匹配、原生self-return响应匹配、§5输出/质量保持控制的响应匹配。后三者同两角色距离、同候选/默认。线性context响应、zero-cut、renorm-cut为固定机制诊断，不事后选为主法。

封存后给同池容量与选择regret；oracle只解释。合法A/B用公共候选并集、全部2×2输出，不仅正确配对或独占区域AUC。范围增量进入全图I/U、四编辑、自然正确例毁伤；CRF前为主，CRF后统一各臂。

| 预定结果 | 判断与下一步 |
|---|---|
| 池有容量，主法胜自然/J/cut-mass，非退化保持控制破坏优势，正确参考双向选对任务 | 支持本构造的任务相关value配对；仍为开发证据，另定完整方法验证，不直接读T/照片 |
| 响应有区别但selector不胜完整控制 | 测量未变成可部署决策；关闭此分布匹配，不换距离/层/head搜赢家 |
| 与cut-mass同样有效，或仅zero-cut有效 | 不支持value/全块解释，分别转为质量描述或幅度效应 |
| mIoU正但正确/交叉参考无区别 | 可能通用候选正则化，未证单参考任务识别 |
| task-switch正但完整I/U不增 | 任务响应存在、效用未兑现，重复旧context教训 |
| 保持控制未实际改变响应 | 控制退化，不能否定/支持配对机制，不换方向找下降 |
| 只部分域/参考正 | 报取舍及背景/视角漂移，不按域开关、不宣称统一迁移 |
| replay或数值稳定性失败 | 尚未测到所定义算法，停止实现核查，不称科学阴/阳性 |

核心不确定性是跨图任务分离，不是公式能否计算。第5节使“新结构而非cut-mass/原输出”有可操作检验；它不能替代第6节尚缺条件，更不能据数学可行性定本线为赢家。
