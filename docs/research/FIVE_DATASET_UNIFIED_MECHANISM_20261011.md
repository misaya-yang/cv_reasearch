# 五域统一机制：任务可表达、可识别与完整读出

2026-10-11。已有证据更支持**单参考任务解释向查询迁移时，角色判别与完整读出失配**，不支持把全部失败归为DINO表征缺信息、一个被消去的gate或几何重建不足。几何对齐、参考拟合和正确Mask响应都是真实能力，但它们各自不能保证完整查询判断正确。下一方案应保留FoRIS实际有效的任务与背景处理，处理争议决策的得失；不能再以弱裸头替换完整读出，或把相关预测的共识当真值。

本文只解释已执行证据并规定统一低成本方案需求。**尚未找到已证明有效的合法任务解释选择依据；candidate=null，未命名或执行新算法。**九窗1300和原Pro200均已完成并关闭，不恢复九窗路线、不扩队列、不改Pro的rank64／temperature .07／role128或原图终点。全部源码、原始缓存、最终mask与I/U保留。

## 证据各自回答什么

|材料|范围与终点|能够支持的结论|不能替代的结论|
|---|---|---|---|
|五域旧输出atlas|COCO200、PACO600、LVIS1400、原生PASCAL2000、原生SUIM2000，共6200；CLI／原图各自原封存终点|完整方法得失、官方固定槽位贡献、相同预算控制|不是6200个Mask-only干预，不把缺槽置0开发指标称完整官方测试集成绩|
|三域真实Mask-only|COCO／PACO／SUIM各20对，固定R/Q、两个真实Mask任务，共120个任务|参考Mask通道确实响应；计算图路径与任务排序变化|不是五域全量因果证明；移位等面积Mask只是负控制|
|role-transfer／source-span|同60对；固定原O24、空间fold和线性容量|可表达性、有限源监督的约束、排序与零点迁移缺口|含Q标签的oracle不是部署算法；空间fold共享整图上下文|
|mixture600|Deep100、PACO100、COCO200、LVIS200|参考与查询角色条件均值不同，纯label-shift必要条件失败|不是任何无监督校准都不可能，不将AUC当最终IoU|
|固定20例像素／annotation|6200内10个正负解释例，加最小参考支持、旧无效校准、唯一现成Mask-only两成员；新像素读取前封存成员|逐阶段四类编辑、合法目标／部件／实例边界|不是部署选择器或普遍发生率；CC8不等于实例|
|Pro200六臂|Deep100＋PACO100，原固定顺序，1200张原图mask全部先封存后GT|完整重建与物理控制的实际任务性能、FP/FN成本|不是第二canonical端点，不是冷编码速度，不改裸头再复验原Pro|

6200只用旧B／G／R0／C／N／K。新6380的full-reference和case-deleted校准是另一个已完成控制，不能替换本atlas中的旧H，也不能把经典LOO记作创新。历史源臂的实际R/Q、crop、Mask、frame、APD与读出由atlas逐例绑定，不凭相同arm名称假定历史链完全相同。Deep不计入五域目标。

来源：[6200库存](../../evidence/local/five_dataset_case_mechanisms_20261011/inventory_report.json)、[逐例整数](../../evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl)、[固定20例](../../evidence/local/five_dataset_case_mechanisms_20261011/case_diagnostics.jsonl)、[紧凑统一证据](../../evidence/local/five_dataset_unified_mechanism_20261011/evidence.json)。

## 表征可表达，不等于单参考可识别

同容量原始单位O24线性诊断中，R训练→R留出AUC为COCO .9922／PACO .9926／SUIM .9909；R训练→Q留出降为.9487／.9306／.8806。使用Q训练块标签的Q自身oracle为.9950／.9930／.9892，R+Q共同oracle为.9946／.9916／.9875且R留出仍约.99。这说明该有限假设族能够读出更多任务信息；不能据此说只有一个参考就能选出这条规则。

把全部R训练token密集监督替换原quadrature，仅COCO改善约.00385 AUC，PACO／SUIM几乎不变。由Q标签构造的安全null反事实保留全部4096个R二值判断，Q AUC仍增加.02348／.01457／.01995。它只说明**同样满足当前有限源判据的解释仍可对Q作出不同判断**：null是相对原采样参考矩阵，rowspace增量也有收益，二值R不变不等于每个R logit或原图像素不变。它不是单参考普遍不可能定理，也不是合法候选方向。

固定反向soft核重建并未提供可靠选择依据：完整1024输出COCO45.851→36.371、PACO41.879→42.548、SUIM47.104→46.455。连真实Q标签经该固定核回到R，R前景得到的Q前景质量均值也只有.404／.437／.541。强制正确答案满足一个未经证明保标签的循环条件，会处罚正确解释。真实物理配准和循环闭合应作为证据，不能直接作为任务真值。

来源：[可识别性](TASK_TRANSFER_CORE_20261010.md)、[源约束审计](TASK_TRANSFER_SOURCE_SPAN_20261010.md)。这些oracle与旧focus/null-rank控制保持原结论，不恢复方法路线。

## 不能从某个分支消失宣称完整模型不可恢复

FoRIS的正标量gate在后续逐token单位化中满足`normalize(g*x)=normalize(x)`；无gate时118/120最终mask逐位相同。这定位了一个没有传到输出的局部幅度通道。其后FG多prototype、hard-BG、NN投票和seed prior仍重新使用R Mask或其任务派生量，因而该gate不是覆盖全部任务路径的cut。

若要证明一个具体完整模型的真正不可恢复点，须在固定R/Q与随机状态下，找一个覆盖**所有从参考Mask到最终判断路径**的joint frontier：合法Mask A/B在其全部值上相同，而要求的任务答案不同。确定性后续于是无法同时答对两个任务。现在没有这种跨域joint-frontier碰撞证据；某个score的task-AUC下降、branch归零或有限读出变弱都不满足该条件。本文不新增碰撞实验，也不作熵／数据处理全局瓶颈论断。

同RGB真实Mask-only的60对没有hard-token输出碰撞；全6200只有一个已核编码身份的现成同R/Q不同Mask组。PASCAL3/312（bus WINDOWS）与3/422（bus VEHICLE_SIDE）R/Q RGB、crop、尺寸以及encoded metadata key／input／tensor／payload SHA完全相同，B与N的原图任务间XOR分别54281／54501，说明仍有任务响应。WINDOWS与SIDE的真实GT高度重叠，不能假设两任务互斥；响应不同也不等于两者正确。

来源：[Mask路径与干预等级](REFERENCE_MASK_TASK_TRACE_20261010.md)、[现成Mask-only配对](../../evidence/local/five_dataset_case_mechanisms_20261011/existing_mask_only_pair_diagnostics.json)。

## 角色迁移同时改变排序和绝对读出

原角色诊断中R自身FG/BG对比AUC约.9970／.9884／.9874，到Q降为.9666／.9169／.8802。Q上单FG方向为.9473／.9121／.7623，加入BG平均有效，尤其SUIM；但Q背景方向也有3/40、8/40、5/40例AUC<.5。因此不能全局删BG，也不能认为每一个R背景方向都可以原样迁移。

mixture600把参考角色条件均值坐标定为FG=1、BG=0，实际Q FG均值却为Deep .81386／PACO .54713／COCO .66629／LVIS .59817，Q BG为.14246／.12693／.19654／.13270。大量BG上的小正移足以造成大的FG面积偏差；对应prototype AUC仍有.90643／.91074／.95437／.95920。APD改善排序但恶化这个prior估计，直接区分了排序能力与绝对校准。

误差恒等式为`pi_coordinate-pi_GT = pi_GT*(E_QF[s]-1) + (1-pi_GT)*E_QB[s]`。这否定纯label-shift的必要条件，不能据此再假定源FG比例、reference零点、目标均值或一个新标量offset能自动修复。单纯通过必要条件也不构成充分性证明。

来源：[原role-transfer](REFERENCE_MASK_TASK_TRACE_20261010.md)、[mixture600](autonomous_20261010/REFERENCE_QUERY_MIXTURE600_RESULTS.md)。

## 五域完整输出：主要共识收益与身份选点收益分开

原图固定槽位mIoU百分数如下；各域按其官方fold/class池化，不跨域平均。

|旧完整方法|COCO200|PACO600|LVIS1400|PASCAL2000|SUIM2000|
|---|---:|---:|---:|---:|---:|
|FoRIS B|63.358900|38.460868|43.912229|55.496178|59.660311|
|MEAN G|64.006821|39.352045|43.872534|55.715834|59.549329|
|原有界R0|62.459822|40.774944|42.771402|56.534176|59.169070|
|共识C|64.080288|39.664615|44.519548|56.255810|59.907918|
|支持准入N|64.140667|39.684486|44.438002|56.394769|60.095403|
|同补数自MEAN排序K|64.127715|39.678407|44.435575|56.386686|60.083051|
|N−B pp|+.781767|+1.223618|+.525773|+.898591|+.435092|
|N−K pp|+.012952|+.006079|+.002427|+.008082|+.012352|

N在五域都高于B，但低于PACO／PASCAL的R0及LVIS的C；不能按domain挑赢家。CLI的N−K为+.012069／+.006928／−.001444／+.018619／+.009369。PASCAL CLI小正的选点差到原图区间含0；SUIM两帧的连通区间含0。N总增益不能包装成普遍的参考身份辨识能力，局部有效例也不能抹掉。

旧`C=majority(B,R0,G)`，`N=C | ((~B)&(~R0)&G&(H>0))`，因此严格有`N&B=C&B`。已由实际调用核实：2380v2在每帧分别解包B/R0/C/G及对应shape的H；native4000在每帧取`parents[frame]`后直接Boolean组合。因此恒等式在旧6200的CLI与original均成立，不依赖canonical导出；不能搬到新1300整mask统一导出的链。见[源码路径收据](../../evidence/local/five_dataset_effective_readout_20261011/path_evidence.json)。

N可以补C未接纳的G-only前景，却永远不能恢复C已删掉的FoRIS前景。这是**当前固定融合可达输出中的不可恢复删除**：R、B及其任务信息仍在上游／缓存，并非完整计算图的Mask信息消失。PACO1/75与COCO1/13证明正确删除FP不可省略，PACO0/29与SUIM428证明相同删除资格会误删TP；不能因此简单never-delete B，缺的是可迁移任务／背景证据对删除的授权。

## 实际正负案例约束同一规则

以下均为原图、真实保存mask与GT的gross四类编辑；案例是新像素读取前冻结的解释成员，不能作为推理开关。

|例子|真实完整动作|机制与反例|
|---|---|---|
|COCO1/13|B→N删450 TP、15566 FP；C=N=K|清理误报使IoU提高18.523点，不能归于N的新准入|
|COCO1/36|B→N删4208 TP、9001 FP；C=N=K|同样清理却降低3.746点，删假数量不单独决定收益|
|PACO1/75|B→R0删411 TP／3022 FP；R0→C补371 TP／40 FP；C→N无改|C保回部分正确覆盖，同时保留大部分清理，完整N比B提高36.595点|
|PACO0/29|B→R0删7923 TP／2884 FP；R0→C补3106 TP／1228 FP；C→N无改|仍净丢4817 TP且降低15.947点；reference sep=1.35885、ref IoU=.91296，并不能授权查询删除|
|LVIS8/82|B→N删11 TP／5359 FP；C=N=K|参考面积仅.3386%，FG质量几何purity=.75196仍有成功；不能用小支持直接判失败|
|LVIS2/105|R0加3 TP／1378 FP；C删964 FP；N再加6 FP；相对B最终加3 TP／420 FP|旧sep=.00318482仍为正，但完整IoU下降47.566点；正号与相关模型同意不足以识别真FG|
|PASCAL2/63|B→N删2 TP／315 FP，R0=C=N=K|person NECK的清理成功；真实annotation只有对象bbox及其他part，不能把partial union叫完整parent mask|
|PASCAL1/178|B→R0删315 TP／251 FP；R0→C补144 TP／229 FP；C=N=K|最终删171 TP／22 FP，pottedplant POT的目标覆盖损失；不从“部件域”标签切分规则|
|SUIM1873|C→N补22 TP／7 FP，同预算K补28 TP／1 FP|N相对B正收益14.729点，但新身份选点仍弱简单控制|
|SUIM428|B→N删40097 TP／94 FP，C=N=K|HD已有FoRIS真覆盖被共识擦掉；参考geometry purity=.92829，旧sep=1.68251、ref IoU=.86329，均不能防止失败|

COCO／SUIM的当前输入是semantic binary mask，实例数unknown。LVIS实例数只按真实annotation；PACO的parent由真实`obj_segmentation`逐位复核。PASCAL没有完整父mask，bbox或其他parts的partial union不可冒充parent。**真实R/Q annotation的实例分组也属于诊断oracle**，输入的R binary Mask并不额外提供真实实例ID。WINDOWS Mask-only例中，N相对K多676 TP且少676 FP是局部成功，但仍有4865个SIDE独有FP；正确Mask响应并未消除任务混淆。

6200合法参考支持记录的LVIS FG面积中位数为1.36%，几何FG质量purity中位数.850，191/1400没有≥90%覆盖的token；COCO／PACO均0，PASCAL4/2000，SUIM34/2000。它说明监督常混合或很小，**不等于embedding semantic purity**。合法FG/BG面积退化与非正分离必须记缺证据；H=0不能解释成语义BG。SUIM428的充足bank与LVIS8/82的小bank分别反驳“只靠纯度／reference fit即可决定查询动作”的两个方向。

N−B逐例赢／输／同为COCO98/98/4、PACO373/212/15、LVIS669/656/75、PASCAL1235/726/39、SUIM1004/960/36；它们不能替代官方class池化。LVIS原图N−K官方mIoU虽+.002427点，全像素却净少1432 TP、多1432 FP，两种口径不等价。SUIM35个R、32个Q的源Mask与RGB尺寸不同，atlas保留原尺寸并按已绑定官方nearest直接到预测frame，没有补修或改协议。

完整逐例、annotation、ROI与独立核验见[case atlas报告](FIVE_DATASET_CASE_MECHANISMS_20261011.md)、[reference支持](../../evidence/local/five_dataset_case_mechanisms_20261011/reference_support_report.json)、[原整数机制汇总](../../evidence/local/five_dataset_case_mechanisms_20261011/mechanism_summary.json)。

## Pro200：重建与守恒不能代替任务读出

原固定六臂、rank64、temperature .07、role128；唯一signed128→原图FP64 bilinear、align_corners=False、严格>0终点。Deep为道路总I/U；PACO为本pilot的87观察class/fold槽位，不能与PACO600官方303点数直接比较。

|完整输出／同批强控制|Deep100|PACO100，87槽|
|---|---:|---:|
|FoRIS|29.377542|45.563915|
|MEAN|26.465369|46.687946|
|global_repeat|11.140274|29.617305|
|local_raw|11.825616|26.231223|
|local_registered|12.176989|27.765578|
|scene_reconstructed|11.499408|29.657600|
|shuffled_registration|11.138866|29.621162|
|shuffled_reference|3.983723|12.291078|

|scene相对FoRIS，原图gross编辑|补真|补假|删真|删假|
|---|---:|---:|---:|---:|
|Deep100|1,890,234|27,458,704|32,389|341,508|
|PACO100|66,889|689,029|11,696|38,088|

Deep scene TP／FP／FN=3,929,066／29,659,538／578,947，FoRIS=2,071,221／2,542,342／2,436,792；PACO scene=382,440／912,909／41,223，FoRIS=327,247／261,968／96,416。scene确实覆盖更多真目标，却无法抑制大量非目标。`global_repeat`无需局部配准也已同样弱，因此失败不能全归某个配准步骤；正确参考优于shuffle-reference也只证明任务响应有效，不能证明正确任务身份。

所有200例真实paired特征fit都好于shuffle，均值误差Deep .418665 vs .992684、PACO .261752 vs .992723；raw coarse守恒最大误差1.55e−15。这与差的完整输出同时成立。`Z=PG+alpha(L-PAL)UV^T`的新local纠正只在span(V)，对其正交的线性方向d纠正严格为0，但PG仍保留，不能称全Z任务信息被rank64删掉。逐token单位化与KDE是非线性的，不能从`AZ=G`推出标签保全。

shuffle-registration还改变alpha：paired均值Deep .817425／PACO .897475，shuffle .093466／.091120。它同时改变方向和残差强度，真实优于shuffle并不是纯粹物理身份的因果隔离。现有六臂足以否定“低特征误差／raw守恒即可保证当前完整输出”，但没有单独隔离出FoRIS中哪一个BG算子贡献全部收益；不追加alpha控制或换头。

代表失败如Deep dg18-0043，scene [17834,188691] vs FoRIS [16564,25719]；PACO1/24，scene [3059,20004] vs FoRIS [3010,3226]。真目标TP几乎保住，union却大幅增加。这些封存后极端IoU例仅解释实际误报，不是新选择器。

来源：[Pro完整报告](../../../cv_data/a/scene_reconstruction_20261011/REPORT.md)、[直接验证](../../../cv_data/a/scene_reconstruction_20261011/verification_enrichment.json)、[root配准／bank标量](../../evidence/local/sol_validation_20261010/pro200_registration_metadata_summary.json)。全部2500 I/U与4800编辑closure已通过；没有重新读raw、拟合或bootstrap。PACO稀疏槽位bootstrap继承缺槽置0限制，以点数和直接counts为准。

Pro缓存六臂生产wall241.698943s、CPU276.844017s、peakRSS1.950GB，encoder新增0；这不是每臂成本或冷端到端速度。历史编码是共享superset实测，不按200比例缩放为冷成本。

## 统一低成本方案必须解决的实际问题

当前可以冻结的是需求与反例，不能由这些诊断自动推出一个已有效的新公式。

1. **保留强的任务读出。**FoRIS使用Mask定义FG多prototype、背景对比、NN投票／seed prior及query region组织；proposal是软分项，densify grid避免全局硬拒。实际source背景权重=.55。新方法须保留这些有效职责，不能用当前弱KDE／单均值裸头重置完整判断。60对干预和Pro包级差异尚不支持只归因一个算子；必要职责不等于要求保留每个冗余实现。见[有效读出源码与证据等级](FIVE_DATASET_EFFECTIVE_READOUT_20261011.md)。
2. **使删除与新增都可被任务信息重新解释。**不能把C的全部删除锁成不可恢复，再只补G-only；PACO0/29和SUIM428要求恢复已有独有真覆盖，PACO1/75／LVIS8/82又要求保留有效误报清理。不存在额外“零TP损失”硬门槛，完整输出必须公开补真／补假／删真／删假的代价。
3. **把小／混合／缺失支持定义完整。**reference coverage是监督几何而非语义真值；不能要求纯token才有任务，也不能把缺参考证据、无proposal或H=0当BG。对象／部件annotation只供解释，部署不能读class、parent或query GT。
4. **只有一条共同规则和一个共享完整读出。**R/M/Q为合法输入，不按dataset、例子、暴露GT或历史赢家选pipeline。复用已绑定whole信息，避免九个独立完整前向／head的重复成本；细节观察若要进入，必须说明其新增任务信息与实际冷成本，不能以缓存CPU快冒称廉价。
5. **完整结果检验信息增量。**强FoRIS、必要简单控制和同预算位置控制同输入、同指定frame；先全输出seal后GT。新单模块不必独立胜FoRIS，但最终组合须胜过只靠编辑面积、保守动作或当前强读出的解释。oracle、AUC、feature fit、响应XOR均不能替代完整预测。

唯一尚缺的识别内容是：**怎样仅用合法R Mask和无标签Q，选择可迁移的任务解释／争议动作，而不是另一条同样拟合R却误读Q的解释。**现在reference纯度／fit、正校准、相关mask共识、循环重建及raw守恒都已有明确反例，不能直接充当这个选择依据。已有atlas与Pro已足够作此边界判断，不以新ROI清单、阈值菜单或额外模块轮换拖延。

## 对唯一下一决策原则的具体判断

唯一进一步考察的原则是：**由合法R标签定义所有仍能解释R的判据，只有它们对争议Q的动作一致才改动，不一致则保留FoRIS完整读出。**它直接回应source-span的“R判断不变而Q可变”，并允许重新检查C删除B的资格。新增与删除适用同一原则，不采用“全部保留B”替代身份判断。

首先必须固定“判据集合”，否则一致性没有内容。若允许所有满足有限R标签的函数，未在R监督中出现的Q特征可以任意赋FG或BG；通常只有确切重合且标签一致的点才能被R独自决定。若改为“固定lambda ridge目标的所有最优解”，严格凸目标只有一个解，只是重复旧R fit，不是消除了任务歧义。若用near-optimal risk、norm球、GP方差或epsilon定义集合，就新增了尚无合法依据的尺度／模型假设，不能自动替代证书。

存在一个**无需norm半径的、严格族内**数学表述，但它的语义与成本前提尚未满足。固定原单位O24仿射线性族，令`z_Ri=[x_Ri,1]`，`y_i`为既有官方R hard-role标签的±1值，定义`V={theta: y_i*z_Ri·theta>0 for every observed R role}`。必须先给出V非空的可行见证；空集的“所有成员一致”是空真，不能授权任何修改。由于有限严格margin可统一正缩放，检测可等价写为`y_i*z_Ri·theta>=1`，这个1不限制norm也不改变符号族。

在V非空时，Q增广描述子`z_Q`的FG符号对所有V一致，当且仅当

```text
z_Q = sum_i lambda_i * y_i * z_Ri,   lambda_i >= 0.
```

BG一致则要求`-z_Q`在同一个锥中。前向充分性是把每条合法R正margin非负相加；反向由分离方向得出：若不在锥中，可沿保持全部R margin的方向让该Q符号翻转。这可保存非空见证与非负系数／残差界核对，**不是用一个已拟合theta的正号、方差或reference score自证**。

条件动作表达式只有一条：`FG证书→FG；BG证书→BG；无证书/空V/未可靠求解→原FoRIS`。它目前不是冻结的预测协议，原因具体如下：

- “所有合法R hard-role”不等于全部原图R Mask：16×16混合覆盖、两种token化与centroid回退已有真实反例。现有source-span只保全已有R二值判断，并未证明这些判断逐位等于合法Mask，因此不能直接用它的nullspace／旧系数替代上述V。
- 尚未证明当前任务的正确Q判断属于这个同一线性族。Q／joint oracle较好只证明有限读出能力，不证明族内一致必为语义真值；把更丰富的任务规则排除后产生“一致”可能是假确定。
- 锥判定要求使用完整特征方向与合法FG/BG标签。reference和query原O24、Mask coverage、官方role assignment都是可缓存合法观测，可能给出比scalar H或多数票更强的区别；FG/BG均值、fit score、纯度、角色面积、无标签空间对齐不能替代该见证。实际R/Q annotation的instance／parent不可参与。
- 没有当前低成本见证。逐Q的高维锥／可行性判定不能当一次ridge或方差点积；只测region均值也不能证明该region每个删除像素。现有缓存省编码，但不证明足够动作覆盖或低求解成本。不能先加norm、阈值或抽样把它变成便宜近似，再保留“证书”称呼。

因此**推翻“仅R一致性即可给一般语义修改充分证书”的强版本**，保留上述精确定义的族内条件原则。具体可证伪下一步已经限定：若未来将它作为完整候选，须在固定同族／同R标签下给出可行见证和每次授权的非负锥见证，保留FoRIS回退，再以PACO1/75／COCO1/13的必要删FP和PACO0/29／SUIM428的误删TP、LVIS2/105的错新增检验完整四编辑及同预算收益。V为空、锥见证不存在／动作过少、实际授权错删／错加或求解过贵，任一个都是拒绝依据，不能改family／norm／epsilon后重称同一原则。

**本轮不执行这项候选，不新增LP／GP／方差头、参数或实验。**已有材料足够否定无前提语义证书；当前尚不能把条件锥原则冻结为有效低成本算法。后续完整方案需要明确的合法判据族及其对真实任务的适用依据，这就是单一识别缺口；不是另缺一项reference fit、九窗一致性或新的零点。机读具体判断见[decision_principle.json](../../evidence/local/five_dataset_unified_mechanism_20261011/decision_principle.json)。

机制机读声明：[mechanism_claims.json](../../evidence/local/five_dataset_unified_mechanism_20261011/mechanism_claims.json)；需求：[requirements.json](../../evidence/local/five_dataset_unified_mechanism_20261011/requirements.json)；全部当前小产物SHA：[source_receipt.json](../../evidence/local/five_dataset_unified_mechanism_20261011/source_receipt.json)；文档整数算术核对：[validation.json](../../evidence/local/five_dataset_unified_mechanism_20261011/validation.json)。本设计线只读既有ledger／receipt，新增GT／mask／raw读取、拟合、预测、encoder均0。
