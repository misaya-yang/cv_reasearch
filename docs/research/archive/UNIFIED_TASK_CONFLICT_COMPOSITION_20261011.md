# 统一完整组合：参考角色与查询共同前景的相对任务外观

2026-10-11，正式冻结v2混合精度。主臂`anchor_relative_complete`不是裸cue测试：保留完整FoRIS B作默认，使用原MEAN G和原有界参考R0定义争议与查询外观anchor，再让新的合法FG/BG相对证据同时裁决新增和删除。**完整6200、两帧、三个新臂全部封存后评分；不要求cue先独立胜FoRIS。**执行线唯一生产，本设计线不计算候选、不改旧输出。

最新人类要求是推进完整组合，理论服务设计。因此此前“版本空间／普遍信息损失证书”的讨论不再作为开跑前置。本文冻结一条完整可证伪方案，不声称已有效、已创新或能识别所有污染anchor。九窗1300与Pro200保持关闭，不再运行其公式、变体或视图。

唯一机器协议：[protocol.json](../../evidence/local/unified_task_conflict_composition_20261011/protocol.json)。其各步骤、参数、fallback、输入SHA、预算与评分口径均已固定。

## 输入与完整职责

沿原五域6200：COCO200、PACO600、LVIS1400、原生PASCAL2000、原生SUIM2000。CLI／original是旧6200各自保存的完整终点，官方固定槽位分别40／303／920／56／7；不混入Deep或旧P/S40，不按domain挑臂或frame。

B是原完整FoRIS，G是原完整MEAN，R0是旧有界`old_arms.R`。保留它们已实际完成的语义前景、有效背景、query region／面积与边界处理；新组合不以弱KDE重置全图，不再跑CRF或形态学。C/N及旧预算K是完整评分控制，不决定候选删除资格。

已有原whole R/Q O24 float32[4096,1024]全部可用，profile `0a5559…6158`。执行preflight缺失0，12400请求／8016不同raw entry，实际来源及payload／tensor SHA逐例消费时核验。新cue只使用这两个whole缓存，不编码、下载、扩样本或新增native crop。

R Mask沿原reader直接由保存尺寸NN到1024再取16×16平均coverage，不能再次crop已经保存的Mask。保留SUIM35个R、32个Q的Mask／RGB尺寸差异；评分GT按原官方nearest直达实际frame。真实R/Q annotation的instance／parent分组仍只是诊断oracle，不能给推理额外标签。

输入封存：[执行preflight](../../evidence/local/task_conflict_execution6200_20261011/preflight_receipt.json)，SHA `2d44050c1ac5f74fb290fcc2e4f762004144bfd060760861b80413a8b2aa663a`；[6200旧来源](../../../../cv_data/a/consensus_reference_calibration6380_v1_20261011/source_bindings.json)，SHA `1b301d5d44d7416ea11afda414d97e2d7e74f7723c6bf357f493134309fa1ac1`。

## 一个相对FG/BG cue

将R/Q原O24逐token在MPS FP32单位化，不做新APD。原B/G/R0各自实际APD与完整链完全保留；raw不是新tap。令R合法软coverage为p，角色质量`MF=sum(p)`、`MB=sum(1-p)`，全4096个R token的软前景中心为

```text
mu_RF = sum(p_i*r_i)/MF
s_R²  = .5 * [E_RF ||r-mu_RF||² + E_RB ||r-mu_RF||²]
```

这个source-centred二阶尺度对FG／BG与两个cue臂共用，不从Q方差、真实面积或reference calibration拟合尺度。MF／MB任一为0，或尺度非正／非有限，则两cue臂及预算均回该帧完整B。没有纯token／最小面积阈值，也不把混合支持当空bank。

FG/BG分别按其软面积质量作128次确定性row-major midpoint quantile。重复index保留重数权重，不把unique位置改成等权；两role的bank向量均为`(r_i-mu_RF)/s_R`。这样保留分布的多个模式，不压成一个FG／BG均值。R Mask是唯一标签；query anchor只提供外观中心，不作为训练标签或拟合目标。

固定CLI共同FG为`A=B_cli&G_cli&R0_cli`，其真实16×16面积coverage为a。`mu_QA=sum(a_j*q_j)/sum(a)`。主臂比较

```text
reference: v = (r-mu_RF)/s_R
query:     u = (q-mu_QA)/s_R
ell_C(u)   = logmeanexp[-||u-v_Ck||²/(2*.07)], C=FG/BG
h(u)       = tanh[(ell_F-ell_B)/2] ∈ [-1,1]
```

Gaussian距离、role128、temperature .07固定；MPS FP32 logsumexp避免下溢，只将距离计算的负roundoff截到0。h是两个面积归一化角色likelihood的对比，不使用旧reference t、H的零点、GT阈值或查询前景比例估计。数据依赖中心与尺度是被冻结算法的输入统计，不是参数扫描。

任何完整cue／contrast出现非有限值，该臂两帧均回B并记录numeric invalid，不把NaN／Inf作BG。h的frame bilinear出口clip到[-1,1]只确保理论界不被roundoff破坏；不改阈值或核的定义。

直观假设是：共同前景暴露了查询中的任务外观，用它的中心消去相对参考FG的共同偏移，再检验各查询点相对合法FG/BG多模分布的位置。这个假设不要求原角色条件分布相同，也不保证中心差全是共模噪声；它可能包含错误anchor、任务模式变化或真实语义差异，交由完整输出和消融裁定。

## 缺失与污染的固定处理

A为空，主臂回B。计算一次h64后，检查`S_A=sum(a*h64)/sum(a)`；若S_A≤0或非有限，说明这个anchor在相对核下整体仍更像BG／无法支持，主臂回B。其预算控制也回B，两个frame共用同一availability／S_A判断。

这是一个能发现部分不兼容anchor的完整fallback，不是正确性证书。共同误报、只保一个错误模式或很小的假anchor可以通过：尤其所有query都中心对齐不会自动证明“真任务”。111内误报继续保留，污染还可能传播到争议区。记录anchor质量、S_A、中心差及fallback；GT仅在全部mask seal以后计价污染，不能驱动中心、fallback或动作。

旧H=0／无效不被解释成BG。此时若R0=B，只记录它是宿主身份旁路，不能称独立第三票；新cue的合法角色来自真实R Mask。缺的是新bank／anchor时回B，不把旧C的删除或旧reference正号借作新授权。

既有八状态账本明确anchor并不纯、union也未覆盖所有目标。原图anchor111的TP／FP分别为COCO5,007,584／1,156,314，PACO1,806,352／1,187,033，LVIS15,710,476／7,563,363，PASCAL14,222,340／2,697,290，SUIM119,846,272／46,883,423。000外真实漏点分别1,272,739／626,077／4,482,040／4,973,788／35,734,681，本方案也保留其原BG。因此这是争议动作组合，不是对所有一致误报或union外目标的修复保证。GT数量只用于暴露边界，不设污染比例门槛。

## 双向冲突裁决与FoRIS默认

两个frame各自直接读取原B/G/R0，并定义

```text
D_add = (~B) & (G|R0)
D_del = B & ((~G)|(~R0))
```

111／000保留B；G或R0任一提出的新增都可进入，B与任一其他臂有分歧的删除都可重新判断。G不是唯一proposal：PASCAL NECK中G只有7TP，B/R0有129TP。C完全不参与动作，所以其误删的B-only前景仍可留住，或由新任务证据重新删除。

继续用原有界宿主规则。原score64先FP64 minmax（分母floor1e−6），再bilinear直达该frame：`d_B=2*norm_frame-1`，`conf_B=abs(d_B)`，常量span=0时conf=0。`m_B=(2*B-1)*conf_B`使用**完整B的符号**，不能用CRF前score符号偷偷替换。

```text
z      = m_B + .5*bilinear(h64)
add    = D_add & (z>0)
delete = D_del & (z<0)
F      = (B & ~delete) | add
```

.5沿旧有界读出，不调系数；精确tie继承B。h=0不作BG、不改B。由h的界与严格翻符号，所有变化自动限于`conf_B<.5`的冲突像素；强宿主、完全一致区域和完整B其它位置保留，没有另训练confidence门控。

CLI与original分别用保存的该帧B/G/R0、共享h64直接FP64 bilinear到1024／原query尺寸，再同式组合。**原图不是整张新canonical mask回渲染**，与旧6200直接frame读出一致，才能保证缺cue时原图严格等于原B。不能把预算或集合性质从一个frame推到另一个；不以frame挑正结果。

## 两个必要完整控制

`reference_relative_complete`保留同一source bank／尺度、temperature、D、宿主及.5有界输出，仅用`u=(q-mu_RF)/s_R`，完全去掉A、mu_QA与S_A。reference本身退化时仍回B，anchor为空时此消融可继续用合法R cue；因此比较的是完整query-anchor构造，availability不同处不能声称纯粹隔离中心偏移。它不是裸cue成绩。

`same_edit_budget_foris_margin`在**每例每frame**取得主臂的实际新增K_add／删除K_del，两个计数分别精确匹配。可选域与主臂一致：`E_add=D_add&(conf_B<.5)`、`E_del=D_del&(conf_B<.5)`。在E_add按FoRIS原连续d_B从高到低选K_add；在E_del从低到高选K_del；同分row-major。只从主臂取数量，位置不使用h。它检验增益是否只是更少、更保守的编辑或面积变化。

全套保留原B/G/R0/C/N/K及三个新完整臂。原C/N/K不重产，复用相同来源旧I/U，必要直接验证来源；同批完整主臂／消融／预算／强基线才决定结果。

## 已有证据要求公开的收益与反例

|固定原图例|已有真实四编辑／限制|新完整组合必须接受的检验|
|---|---|---|
|PACO1/75|R0删411TP/3022FP，C补371TP/40FP，C→N=0|允许保真与拒假分别判断，不以never-delete B取消必要清理|
|COCO1/13|N相对B删450TP/15566FP|正确清理的收益也要保留；不能只追求覆盖|
|PACO0/29|N相对B删4817TP/1656FP，reference拟合好仍失败|任务anchor相对cue是否减少错删必须看完整输出，旧fit不授权删除|
|SUIM428 HD|N删40097TP仅拒94FP；41853个B-only真像素旧H正计数0|B默认与重开冲突能保留覆盖，但新cue仍可能误删，不能预称已修好|
|LVIS2/105|R0加3TP/1378FP，C拒964FP，N再加6FP|新增也需真实任务关系，不能拿召回／模型同意作真值|
|PASCAL NECK2/63|G只有7TP，B/R0保129TP|不能让G成为唯一候选或唯一query任务解释|
|SUIM1873 RO|N补22TP/7FP，同预算K补28TP/1FP|相对位置贡献须胜必要预算控制，不把面积收益包装为身份迁移|
|现成PAS WINDOWS/SIDE|encoded身份同；111可能含另一任务独有像素；N局部比K多676TP但仍有4865SIDE独有FP|anchor是预测而非GT；相同视觉的任务响应／局部成功不证明整体anchor正确|

这些证据完善同一公式，不触发域名／case／GT规则。真实同父区域、实例组、query组件与GT面积只供解释，不能进入cue。普通6200不是Mask-only干预，历史60对与唯一PAS配对单列。

原图B-only的TP／FP为COCO96,669／193,492，PACO39,982／258,783，LVIS315,378／1,443,116，PASCAL232,441／495,613，SUIM1,680,703／3,317,494；旧C/N/K把两者全删。R0-only真目标也有COCO113,095／PACO3,249／LVIS133,497／PASCAL318,192／SUIM2,033,013，旧C/N/K均拒绝。本公式分别让这些区域保留或进入任务对比，不要求某个孤立模块先胜完整FoRIS。范围与510448项既有整数恢复见[八状态组合账本](../../evidence/local/task_conflict_complementarity6200_20261011/report.json)，无新GT／Mask读取或重scorer。

## 封存、验收与实际成本

执行线用现有缓存逐例消费，稠密cue采用MPS FP32，CPU FP64完成出口标量／两帧读出／排序，参考／query raw SHA、Mask、B/G/R0、score64来源和geometry全部绑定。先做无GT数值／公式检查，检查不改变方案。新mask三臂×两帧×6200=37200全部pack与seal后才打开query GT。运行guard拒绝6200 query Mask／annotation；原B/G/R0与source64是明确授权的推理输入。

仅精度更新：原全CPU FP64的6296e83c版与receipt／文档快照已保留，执行线确认该版无producer PID、cue及候选均0，只有metadata preflight。新唯一契约在MPS FP32计算unit／中心／source variance／Gram／距离／logmeanexp／tanh；完成native64 cue转CPU FP64，MF/MB/MA、S_A、宿主minmax／bilinear／threshold／budget及记录标量用CPU FP64。MPS产生的source尺度只是转成FP64记录，不为生产重算另一尺度。

固定前3行在无GT阶段作CPU FP64同公式数值参照，仅报告误差／标量差／成本；不生成单独mask或评分臂、不设分数gate、不以参照选择输出。仅实际MPS backend／runtime／OOM错误时，记录失败成本并将该例两cue／中心／尺度完整改由CPU FP32重算，同一CPU FP64出口；数学退化或NaN cue仍按原B fallback，不触发试别的精度。MPS阶段计时先同步，记录MPS allocated／driver peak与CPU32 fallback成本。

每例临时特征、bank、距离、likelihood、h及frame score算完释放；不归档dense场，不批量保持6200特征。仅保存final packed masks、小source/scale/anchor/fallback／计数／cutoff与成本收据。独立无GT公式检查验证111/000、B fallback、有界编辑支持与每frame同预算；seal后直接AND/OR I/U、gross四编辑与官方class池化复核。

分别报告五域CLI主端点、原图次端点，强FoRIS与完整消融／预算，所有addTP／addFP／deleteTP／deleteFP，class贡献与全像素量分开。沿既有query-photo5000／connected R/Q2000敏感性实现并保留缺槽／稀疏限制，不再加bootstrap或域开关。GT只能在seal后揭示111污染、失败和反例。

架构只增加共享whole R/Q上的两个128-role核对比，没有九个独立完整链，也无新增编码。**实际低成本与完整泛化仍需执行结果裁定**：记录IO/hash、bank/中心/尺度、两cue、render/组合/排序/write、评分验收和RSS；缓存CPU／MPS时间不冒称冷端到端速度，已有编码superset成本不按比例缩放。成功或失败都保留，不因部分GT结果调参数、加变体或扩样本。

证据背景：[五域真实案例](FIVE_DATASET_CASE_MECHANISMS_20261011.md)、[有效FoRIS路径](FIVE_DATASET_EFFECTIVE_READOUT_20261011.md)、[统一机制与oracle边界](FIVE_DATASET_UNIFIED_MECHANISM_20261011.md)。本文是完整可执行冻结，不是新核函数的创新证明或已成功结果。
