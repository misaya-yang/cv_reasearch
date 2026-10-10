# 联合角色模型的最小修正规格

目标是独立的冻结DINO分割：reference RGB＋完整reference mask＋query RGB，全部合法O24观察共同形成一张mask。FoRIS只作对照，不是必须保留的anchor。本文仅准备一个明确的新模型与可证伪检查；没有修改PLAN / solver、运行真实分割 / query GT、新增十集或向Pro发消息。

已有数学核查确认两种功能限制：原式的同atom塌缩在σ1时仅支付1/256；标准exp(−E)、λ1与incident预算≤1限制角色odds最多改变e倍，初始>.731或<.269不能翻转。修正必须改变这些功能，不能只补齐BP实现细节。来源：[原数学审阅](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/pro_joint_correspondence_review_20261010/math_review.md)、[已确认合成检查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/pro_joint_correspondence_review_20261010/checks/results.json)。

## 决定：先修可辨识的几何，再赋予它有限、明确的角色纠错能力

只提高pair力度会把量化误差、坐标失配和未经确认的对应放大。应先把“合法粗细别名”与“大块塌成同atom”分开，观察真实关系在原有可翻转区间能否提供判别；强role-logit的功能修正先定义清楚，不能因此立即启用。参考纯度只是别名的一种来源，不能成为这些问题的主解释。

下面固定唯一的首个实现候选 **R1-G：带参考位置支持集的相对几何，保留原b0 / λ1 / T1**。有界role-logit仅作为第5节的独立R2能力修正定义，**不进入R1-G，也不自动启动另一真实候选**。它们都不是由已有结果唯一推导出的正确算法；本文件给出确定能量、计算预算和失败条件，而不是扩λ/σ菜单。

### 理论功能风险与已准备真实首例分开

root完成的Deep首例`dg18-0000`合法input-only准备有如下证据，尚无solver / posterior / GT / mask：

- P0 FG>.731为13.568%，FG<.269为11.865%，约74.567%在原模型理论可翻转区间。这些是无标签先验比例，不是错误率；不能据此说原关系无用，也不能据强25.4%猜测其都错误。
- 38/16384节点触minscale，即0.232%；初值scale median2.787，candidate位置方差median189.491(refpatch²)。不能把理论minscale反例称作真实全幅已经collapse，或者盲目提高minscale。
- 99.225%邻边存在共同atom可选，但independent unary同atom pairmass均值只有5.922%。可行overlap不是实际collapse；共同atom也可能是合法fine/coarse别名。
- retained alpha FG中位6.768%、BG2.471%，位置质量重分放大分别14.774 / 40.467倍。**位置不确定性比“把λ立刻调大”更值得先处理。** 条件位置 / 支持集修正不能恢复被候选预算丢掉的其它patch；R1-G仍需记录alpha，不能把它的留存候选当完整确定对应。

[输入收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/joint_reference_preparation_20261010/input_receipt.json)、[candidate位置诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/joint_reference_preparation_20261010/candidate_position_diagnostic.json)。六份raw实际身份已验，CPU2线程观察3.574秒仅match / candidate准备，不是冷完整时延。本轨只读这两份结果，没有再次加载真实raw或做solver / GT。

## 1. Faithful F0必须保留，不能偷换成R1

F0按Pro原式复现：同whole＋四角真实O24、全角色LSE、最多16候选、原patch质心、`J=query→reference`、原绝对σ2→1、λ1。BP模型温度T=1；log消息零初始化、同步四轮＋四轮、概率凸阻尼1/2、第二阶段warm-start，读真实edge belief。原J更新使用`γ=w*edge_belief*1[d_old<1]`，正确γ下其固定信念 / 固定σ的MM推导可成立。

F0如实呈现塌缩与有限纠错，不为了让反例通过就悄悄改坐标、pair温度、候选、初值或unary。它现在已有[21项真实solve合成验收的CPU FP64参照](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/pro_joint_correspondence_review_20261010/SOLVER_REFERENCE.md)，不需要重写另一份faithful solver。R1-G以下的坐标 / 支持 / 几何项和第5节R2各自都是新的模型定义；将来评分必须标明，不能称原方法的数值等价修复。

## 2. Known resize只改几何坐标，不改已有编码

一张合法输入视图X的真实宽高为`W_X,H_X`，图像拉伸1024²的已知矩阵为`A_X=diag(1024/W_X,1024/H_X)`。以同一视图的几何均长`l_X=sqrt(W_X H_X)`定义无量纲原比例坐标：

```text
z_X(u_canvas) = A_X^(-1) u_canvas / l_X
             = (u_x W_X, u_y H_X) / (1024 l_X).
```

所有reference role位置、支持集、query128网格中心都进入这个坐标系。输入原比例不因拉伸被误当相似变换；额外单位scale由后面的K吸收。输入crop就是实际编码视图，使用它的宽高和已知坐标变换，不把未编码的整张照片坐标或query GT父框混进去。

例如合成R=400×200、Q=200×400，一份真正的1.3倍、30°旋转在square canvas中有3.392754的奇异值比；去掉已知resize后重新成为相似变换，数值误差0。known preprocessing各向异性可因此精确处理；透视、反射、真实非刚性形变仍不在各向同性K族中，要靠有限断边代价承担，不预称已覆盖。

## 3. 参考atom保留位置支持集，允许合法fine→同coarse别名

沿用`a=(reference patch r, role y)`、同一个单位O24向量和连续coverage / area角色权重。新增`S_a`：该16×16编码patch内、reference mask属于role y的像素小方格的并集，转换到z_R。它是已知参考mask定义的合法位置集合，不是新视觉特征。

每个query节点i的每个候选a有一个条件位置`xi_i(a)∈S_a`；它在节点i的所有邻边中共用。不同query节点可以选同一atom并在其支持集内有不同位置，也可以重复同一位置。没有一一对应、列容量、前景配额或实例数限制。

仅把v=0全部判为错误会损坏合法别名：四个fine节点位于一个coarse patch内，四个潜在位置可以构成2×2子格，尽管O24 / atom ID相同，几何关系完全成立。不能为每条边独立挑一份有利位置而省略节点共用xi，那会重新允许互相矛盾的邻边解释。

empty FG按任务定义返回空输出；empty BG仍标记缺少本模型负角色信息，不能捏造负向量。单token共享两个角色、相同feature和几何支持对称时，模型仍可能不可辨识；支持集没有制造新的语义细节。

## 4. 相对、逆向几何让点塌缩不随scale变便宜

每节点共用一个inverse similarity `K_i=kappa_i R(phi_i)`，从reference z位移映到query z位移，`kappa∈[1/8,8]`。这是对原相似变换的逆向参数化；在去resize坐标后保持相同正尺度范围。

对每条query边，delta=z_Qj−z_Qi，v=xi_j(b)−xi_i(a)，定义

```text
d_ij(a,b) = [||K_i v-delta||² + ||K_j v-delta||²] / (2 ||delta||²)
c_ij(a,b) = min(d_ij(a,b), 1).
```

没有绝对patch单位σ。共同点塌缩v=0时d恒为1，与kappa无关；只把尺度推到下界不能再把惩罚变成1/256。与此同时，合法fine/coarse别名可以通过共用xi实现v=K^(-1)delta，d=0。这个模型依然不是拓扑 / 不折叠保证。

仅有短邻边也不够：支持集可能让一小片fine区域合法落在同patch内。保留原四邻接，并追加一组**按采样分辨率推导的跨度边**，区分可容纳的量化别名与大块同atom塌缩。

令reference coarse cell在z_R的最大直径为

```text
Delta_R = sqrt((W_R/(64*l_R))² + (H_R/(64*l_R))²).
```

query fine横纵步长为`h_x=W_Q/(128*l_Q)`、`h_y=H_Q/(128*l_Q)`。冻结`kappa_max=8`，选择最小整数

```text
L_x = ceil(2*kappa_max*Delta_R/h_x)
L_y = ceil(2*kappa_max*Delta_R/h_y).
```

每节点连接仍在画布内的±L_x横边和±L_y纵边，去重；若某轴所需跨度超过画布，就没有该轴可分辨跨度，不缩短它冒充有约束。系数2明确对应“参考cell量化直径不超过预测query跨度的一半”的精度定义，不是按mIoU挑选的距离。

若跨度边两端仍对应同一atom，则任意合法xi都有`||v||≤Delta_R`、`||K v||≤8 Delta_R≤||delta||/2`，故**d至少1/4**。真实跨patch非塌缩解释可以取得0；短边允许同patch别名，长边阻止整个大块区域免费借此逃逸。

所有边仍用`w=1/max(deg i,deg j)`和cap1。跨不同实例 / 遮挡的关系可以付有限代价断开，重复reference使用始终合法；多实例最终能否保留需要检验，不能把“未硬禁止”写成已正确识别多个实例。

## 5. 独立R2能力规格：只改role优势，不偷降整个BP温度（当前不启用）

保持全角色LSE与原16候选构造。记完整初始role概率为P0、原候选内条件位置分布为`pi_i(a|y)=b_i0(a)/P0_i(y)`。单独读取

```text
ell_i = log(P0_i(FG)/P0_i(BG)).
S_i_span = sum(w_ij, incident resolvable-span edges).
ell_i_star = clip(ell_i, -S_i_span/2, S_i_span/2)  # 仅S_span>0
```

`S_span=0`时保留ell，不因为没有关系而抹掉appearance。新role概率`P_star(FG)=sigmoid(ell_star)`，新candidate unary为`b_star(a)=P_star(y)*pi_i(a|y)`。模型温度仍T=1、pair λ仍1，原appearance温度τ=.07只承担候选 / 位置响应的定义。

这是明确的新model prior：认为未经校准的role-logit不应独自超过可分辨关系预算的一半，留出反转余量。它不是校准，也不保证关系比appearance正确；半预算不是已有实测给出的最优值，不再扫描它。**真实首例多数先验已可翻转，目前没有证据要求启用这个cap；R1-G先保留原b0。** 只有完整关系能可靠改变决定、却确实因强错role优势受阻的后续证据，才有理由将这个独立能力规格交root选择；不能用无标签confident比例自动授权。

这个变换保留ell符号、角色内位置分布和关闭关系时的二元决定。**无关系的连续role质量已不同于P0**，不能继续称原role-mass字段逐位相同。必要unary对照必须使用这个同b_star；并检查它的硬mask与P0硬mask相同，避免把读出变化偷偷归到关系。

当span的全部关系真正以最大预算反对错误FG，原prior0.9、S_span1会从无关系0.622459降至0.377541；S_span.5则0.562177→0.437823。原模型对应0.768031 / 0.845172，无法翻转。这证明**新模型有这种功能**，不是证明真实关系有足够信息或会朝正确方向改。

不能把不同温度混为一谈：在同一个prior0.9 / pair优势1反例中，原T1结果0.768031；只将pair除.07结果约5.62e−6，已改变相对模型；把整个旧E除.07反而得到0.999999963，强化原MAP，不会救回错误。独立R2定义的是有界role优势，不是这两种降温的隐式替代；R1-G三者均不启用。

## 6. 一个确定的完整候选及固定求解预算

首个R1-G能量为

```text
E_R1G(a,xi,K) = -sum_i log b0_i(a_i)
              + sum_(i,j) w_ij * min(d_ij(a_i,a_j),1).
```

输出流程与编码资产保持Pro的whole＋四角真实O24、query128²统一网格、单一role marginal双线性回原图并严格>0.5。没有FoRIS unary / hardmask / fallback，也不恢复九套局部头。

固定实现顺序：

1. xi_i(a)初始化为role质心到S_a的精确投影，等距时按reference像素行优先索引选择。角色有多个不连通区域时，其平均质心可能落在异角色空隙，不能直接称为合法支持位置。在去resize坐标中重算F0候选期望的forward similarity作为**初始化假设**，取其逆K；保留所有候选 / 方差，不能把期望点当已确认的对应。只有D=0或A=B=0的真正无尺度 / 方向证据情形采用公开identity fallback并记原因；不因真实首例minscale少数或median2.787就全幅重设scale。多峰时可运行这一seed，但必须标明其不等于模式确定，后续gamma来自真实edge belief。
2. 新pair上同步sum-product BP四轮，T1、概率凸阻尼1/2、log初值0，取得实际joint edge belief q_ij。
3. 冻结此q与`active=1[d_old≤1]`做一次geometry MM：先按预先固定的图合法着色顺序更新所有条件位置xi，再更新所有K。包含d=1边界是公开的MM上界选择，避免全collapsed初始化因严格<1完全没有更新依据；不是按GT切阈值。
4. 更新pair后warm-start四轮同BP，读出一次完整mask；不追加收敛长循环。

MM权重必须有`w*q*active/||delta||²`。固定K时，K的相似性使每个xi条件位置的二次Hessian为scalar×I，先求其无约束加权目标点，再**精确投影到S_a的像素方格并集**；相同颜色节点之间不得有边，可并行。该位置更新读取实际候选联合q，不能退回仅用一份候选均值替代多个模式。不能拿原网格红黑两色直接处理新增偶数跨度边，需一次确定性greedy coloring。

固定xi时，inverse Procrustes拟合的是`K v≈delta`：A/B来自v与delta的点积 / 叉积，D来自`||v||²`，都含上述归一权重；D=0保留K，A=B=0旋转无证据。不能套用原forward公式的delta分母。按正确权重及active固定，位置 / K的条件更新降低该固定信念MM上界；不保证八轮loopy BP的全局能量或真实分数单调。

candidate状态仍≤16；每候选新增两维条件位置，不枚举角度 / 尺度。标准square128示例跨度46，短边32512、span20992，总53504条，无向pair表约52.25MiB、最大degree8、确定性greedy着色实测4色。相较原32512边，BP边工作量约1.646倍，绝不是免费修正；额外位置表约2MiB，支持集投影必须分块，不能物化全部query×candidate×256像素的大张量。真实非编码预算和质量验证附带测量，不再只为计时重启真实模型。

## 7. 本轮合成验收与下一次可证伪检查

本轮只在内存运行numpy小数组及无图像的128²几何graph，共约0.024秒，单进程、最多2线程；真实输入、raw读取、GT、模型、分割均0，未写solver或额外脚本。下列R2 / 温度检查只验证能力定义，**不表示R1-G已启用它**。以下证明功能与可实现性，不证明真实涨分：

|检查|本轮结果|它排除的错误修复|
|---|---|---|
|同点collapse，kappa=.125/1/8|新relative d全为1|只靠缩小scale继续消除惩罚。|
|2×2 fine共享一个coarse atom，xi在同cell内部|存在d=0合法解释|把同atom使用一律拒绝 / 硬容量限制。|
|同fixture从全xi相同初始化|一轮合法着色位置MM把cost2降到0|只保留centroid、或在d=1丢弃所有更新关系。|
|span23、cell直径sqrt2、kappa≤8|同cell任何位置的d下界0.258165>.25|用量化别名掩盖大块collapse。|
|已知anisotropic resize|canvas奇异值比3.392754；dewarp后相似误差0|把预处理拉伸当真实非相似形变。|
|独立R2：强错role0.9，有完整反向span支持|S1结果.377541；S.5结果.437823|R2在功能上仍不能翻转；R1-G保留原限制。|
|pair-only与whole-E温度|分别约5.62e−6、.999999963|把两种新模型混称同一实现。|
|短＋span图着色|128²、L46、4色，所有边跨色|MM并行更新同色邻居破坏条件极小化。|

多实例检查在本轮只验证**可行性**：两份分开的2×2别名实例可以重用全部atom / 坐标且局部cost0，跨实例关系至多付有限cap1；这不是已通过多实例标签正确率测试。下一准备必须增加“两个完整实例＋背景桥接”的全候选精确小图，若新的角色cap / 长边压掉其中一个，不能因无全局容量就宣称修好，应停止该R1进入真实验证，不自动换λ。

实施后的最小准入门槛还有：相同role / feature / 支持的不可辨识输入应保留对称信念；两个正确对应模式不能先被位置均值制造虚假第三模式；完整candidate padding / 全角色质量闭合；真正edge belief与node product不同；彩色MM固定q的目标不增；原/去resize坐标的同一几何fixture一致。若失败，先修定义或代码，不用更多BP轮 / 参数grid掩盖。

## 8. 实测预期与明确否决条件

若首个R1-G有效，应在全部输出先封存后出现：可分辨大块collapse的实际模式质量下降（不是仅统计可行atom overlap或scale触底）；原可翻转区间里的错误被**合法关系**改正；fine同coarse别名不因此损伤细目标；多实例重复使用仍保留；相同观察下完整mIoU优于同b0无关系与普通对应，并且真reference绑定优于一次固定坐标置换。它保留strong-prior限制，不能预称已解决那些错误；未来R2需另与同b_star对照。

若只降低collapse率却不增加净TP / 减少净FP，不能当结构成功；若出现伪模板、父物体扩张、跨实例抹除，说明新增关系先验在损害任务；若留存候选中根本没有可解释的真实结构、或大部分正确模式仍在tail，不能调大λ代替补回位置证据。若只赢unary但未赢普通对应，不能归到任务结构；R2同理需胜同b_star。字段、能量和少数图都不能替代最终净分。真实质量和延迟必须同批配对，新增MM / span成本如实计入。

目前不能证明R1-G是唯一或最优解，也不能由一个input-only首例宣称实际collapse / 强错角色已发生。当前唯一下一实现对象是**保留b0 / λ1、修合法位置和相对几何的R1-G**；R2只把强错角色可翻转能力写清并暂存，不启用、不扫描。二者都不靠FoRIS保护。下一具体动作是实现R1-G并过上述无GT门槛，检验位置 / 关系是否真增加判别；本文没有启动该实现后的真实评测。
