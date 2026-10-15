# 图、细读出与固定残差：实际增量及适用边界

2026-10-10。状态：**完成本线审查、旧4000重聚合、LVIS100完整重放与新增机制诊断**。没有新编码、MPS、raw读取、参数搜索、独立新样本或默认方法修改。所有样本此前已曝光。

结论是：已有正结果应保留，但它们支持三个不同、有限的作用：图可修剪错误unary；真实shifted视图能改善既有场的边界读出；固定残差能将一部分细读出改动移到MEAN。它们没有共同证明新的跨图目标身份能力。最新条件rank图相对匹配削边控制有完整输出正差，但困难区残差排序很弱，仍未胜完整FoRIS，不能直接作为当前Deep/PACO主线的新增组件。

本轮另发现两项历史表述需纠正：`fine16/fine64`中的16/64是**图λ**，不是输出分辨率；12源外层留折是**类别折留出，存在照片共享**，不能称照片隔离。12源并不含sizecut，原报告那句sizecut警告是通用模板；含all600拟合尺寸切点的是13源扩展。

## 1. 本轮实际执行及证据等级

|证据|本轮做了什么|仍不能声称什么|
|---|---|---|
|旧COCO4000 residual `counts.npz`、`episodes.jsonl`|7臂每例I/U逐位互核、80类聚合mIoU重算、4类编辑闭合、逐折重算|没有旧4000场/完整mask的本机重放；历史CI只引用原已审查结果，没有重估|
|12源 `counts.npz`、`selection.json`、`sealed.json`|重算全部表分数；与上述7臂I/U逐位一致；4000 draw key有序一致；核查4个固定配方与照片共享|没有重新搜索50,331,648配方；没有恢复缺失旧mask/库histogram|
|旧五数据集 `rcg_transfer_20261007/audit.json`|从保存的每类base/new IoU重新聚合，核对聚合点值与贡献表|这不是从原始mask/IU重评；旧跨集目录本机只见rows，没有原counts/fields/predictions|
|当前LVIS固定100图字段与预测|从原FP64 H/A/y/s/g重建原MEAN及条件rank图/匹配图；全部图数组、场、CLI1024/原图mask逐位等于封存结果；重评分|这是已有配方的再执行，不是独立新样本或新算法收益|
|同100新增机制量|全100重放封存后才读取query GT；实际像素ROI/GT先相交，再pool到64²做加权AUC；逐例unary冲突账本|AUC是节点场在像素质量上的排序，不是双线性像素AUC，也不是mIoU|

完整执行脚本：[analyze.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/analyze.py)。产物：[历史重算](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/historical_recompute.json)、[重放封存](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/replay_sealed.json)、[新诊断汇总](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/diagnostic_summary.json)、[逐例账本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/diagnostic_episodes.jsonl)、[完成收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/COMPLETE.json)。

100例图/三场与两帧重放计时4.522秒，脚本内总计时9.638秒，单进程/1 CPU线程。含磁盘读取、压缩输出和评分；不包含DINO、源FoRIS或冷端到端模型成本。完整重放mask和小字段保存在同目录的`predictions/`与`fields/`，源文件保持只读。

加权AUC已用含tie的直接正负配对枚举核验，全部100汇总从逐例重新聚合一致；收据：[verification.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/verification.json)。旧Git关键原文按原行号保存在[历史摘录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/historical_git_excerpts.txt)，原blob SHA亦在收据。

## 2. 每项旧成功实际增加了什么

### RCG与MEAN

两者都从已有FoRIS原连续响应的minmax场s出发，加rank校正，再解带fidelity的reciprocal20NN query图，λ=16，双线性读到1024后严格>.5，无新CRF。

- RCG用纯参考FG的去hub max guide，rank系数α=.5。
- MEAN用纯FG均值prototype guide，α=.25。它同时改变guide和校正幅度，并不只是RCG的“平均版本”或较弱平滑。参考纯FG不足时使用最大coverage回退。
- 图系数、rank系数、纯度等均是历史开发选择后锁定；当前推断不读取query GT不等于整个方法没有开发标签暴露。

定义证据：[rcg.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/src/ics/methods/rcg.py)、[mean_control.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/src/ics/methods/mean_control.py)。旧600拆分保存在Git `82df916:evidence/local/RESULTS.md`：pre60.944→仅graph63.582→完整RCG64.238，rank单独60.939；当时主要增益在graph，并有graph条件下rank+.655。该段引用的独立原报告现在本地缺失，不能写成本轮重新验证的600因果结果。

旧COCO4000/1024上RCG62.333671、MEAN62.512972；MEAN相对RCG+.179301，历史配对区间[−.003145,+.375652]，并未稳定区分。当前raw-native LVIS1400则完整FoRIS原图46.105282、MEAN46.056276；不能把旧processed-cache结果与这个新输入/协议混减或声称MEAN已经跨集胜出。[当前1400证据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/EXISTING_EVIDENCE_AUDIT_20261010.md)。

### fine16、fine64与feature-guided readout

`fine16`/`fine64`均从64²粗场读到128²细场；后缀指粗场图λ=16/64。细guide来自四个±4像素reflect-shift的真实query编码，以5×5粗邻域的空间权重与feature cosine权重重读原粗场，再双线性到1024。它有四次额外query编码，不是纯插值的免费高分辨率版本。

旧600 nested读出为64.625380，对RCG64.237696 +.387683，对RGB同类读出64.390177 +.235202，对RCG+CRF64.448578 +.176802。σ/τ从其它折选择；保留报告可读。[600原报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/subtoken_fresh600/report.md)。

更强的信息移除控制在旧Git账本确有记载：相同kernel/邻域/finalizer，将fine features替换为单位双线性coarse features，64.273121；真实fine64.625380，差+.352259，历史区间[+.325705,+.544086]。这支持**该读出中真实shifted观察有用**。对应`subtoken_coarse_control600_v2/report.md`现本机缺失，本轮不能重评分；同时旧80例boundary NN诊断未胜插值coarse margin，不能把完整读出改善直接称新reference-NN语义分离。原实现：[run_rcg2.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/scripts/run_rcg2.py)。

旧4000完整结果fine16 62.709100，fine64 62.651617。增加λ到64未胜16；它不是“加到64分辨率反而坏了”。同4000 fine64相对自身coarse64 +.353680，但相对MEAN仅+.138645、历史区间跨0。128²fine读出对小边界的收益不能自动推广为PACO部件身份识别能力。

### 固定scalar residual transfer

记F=RCG16，G=MEAN16，Y=fine(RCG16)，B为64→128双线性。固定构造是：

`C = B(G) + (Y - B(F))`。

系数1、FP32、不clip、不重新拟合。它把RCG的细读出残差搬到MEAN，**不是**直接`fine(MEAN)`。源细读出σ1.25，τ=.07（fold0/3）或.15（fold1/2），继承此前GT开发折选择。原4000/CLI1024计数本轮全部重算：

|方法|mIoU|相对MEAN|关键强控制|
|---|---:|---:|---|
|完整FoRIS|60.931741|−1.581231|完整主参照|
|RCG|62.333671|−.179301|query图+另一guide|
|MEAN|62.512972|0|同源简单图控制|
|fine16|62.709100|+.196128|同真实shifted观察|
|fine64|62.651617|+.138645|更大λ，非更高分辨率|
|mask-only转移|62.744357|+.231385|只用已有R/M/V二值增删|
|scalar残差|62.844537|+.331565|比mask-only+.100180；比fine16+.135437|
|12源外层留折|62.964699|+.451727|比scalar残差+.120162|

scalar对MEAN历史CI[+.303193,+.372805]，对mask-only[+.079995,+.128147]，但对fine16[−.048326,+.334529]、fine64[−.078619,+.522409]均跨0。四折对MEAN分别+.325940/+.362928/+.323162/+.314230；有稳定的同源小增量，但还没有胜过全部同信息强控制的证据。

本轮新增精确计价：scalar相对MEAN的+.331565由补真+.769458、补假−.561405、删真−.503206、删假+.626719组成；相对fine16的+.135437中，净TP项仅+.002128，净FP项+.133309。相对fine16的点差主要是背景范围调整，不能把相对FoRIS+1.912796都归细残差的新语义。四项使用`ΔIoU = ΔI/U_new − IoU_base*ΔU/U_new`，不是Shapley或因果模块归因。

保留配置明确把直接`P(G)`作为不同强控制，但本机未找到完整4000的直接MEAN-fine结果；不能假定残差胜过它。残差原推理报告有2536.420秒source replay/render、其中2178.503秒paired encoder；10.253秒仅固定render，不能拿它当整个方法的端到端时延。[原报告/配置](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.json)。本轮未找到scalar transfer跨PACO/LVIS/SUIM的完整结果，因此不宣称该配方跨集成功。

### 12源配方的监督与照片依赖

它是12个**完整输出producer**构成的库，并不是推断时把12种信号全相加。每个外层fold用其它3折GT在50,331,648个A/B增删组合中选固定配方，再在该fold读出。选中配方的外层producer数代理为3，上游复合依赖可能更多，也不等于3次实际编码或成本。

四个fold均用fine64提供添加；删除分别依赖fine16或scalar。基准origin为`external_mean_delete_sameK_RCG`、`conservative_delete`或fine16。12源全局开发最优63.064818，只比11源最优+.002513；scalar单独+.331565无法与其它组件收益相加。外层留折62.964699对scalar+.120162，历史CI[−.103035,+.319822]。

**本轮纠正照片隔离表述。** `82df916`旧账本写“with photo exclusion”，但该版本与当前`run_exact_family_crossfold.py` SHA256相同，实际训练索引为`folds!=fold`，收据每fold均3000训练/1000评价。4000有序draw key及7臂I/U逐位对应；按保留query/support身份核查：

|held fold|train/held共有照片|至少一张照片出现在train的held episode|其中query也在train|
|---|---:|---:|---:|
|0|166|174/1000|95|
|1|169|174/1000|96|
|2|147|155/1000|87|
|3|168|181/1000|92|

这是类别折留出、同照片可能跨类；不是同一目标标签必然重复，也不是推断读取query GT，但不是照片隔离验证。connected-photo bootstrap仅处理评分重采样依赖，不会自动修复训练/评价照片共享。核查：[provenance_audit.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/parallel_signal_20261010/03_graph_readout/provenance_audit.json)。

12个实际arm没有sizecut。`report.md`的“all600 sizecut”句子由脚本无条件输出，不能用它给12源安上不存在的尺寸拟合。13源扩展才新增allfresh600拟合的sizecut；它更明确不是整套独立留折确认。两者都承认source参数/方法库开发暴露，CI也未重复整个配方选择过程。[arm清单](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/arms.json)、[配方选择](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/selection.json)。

## 3. 跨数据集：改善的位置并不相同

下表是旧self-sampled pack、**原图每类pooled I/U宏平均**；PACO为264个fold-class、LVIS为367个pack-offset fold-class，不能与当前官方十折LVIS或PACO303槽位混减。本轮只从保存的每类IoU复核聚合，未重读这些旧mask。

|数据集/例数|完整FoRIS|RCG|RCG+fine|再加COCO拟合sizecut|
|---|---:|---:|---:|---:|
|COCO B+C+D /1800|60.517470|61.798393|62.178626|62.380275|
|PASCAL-Part /598|53.868184|55.327753|55.351584|54.831769|
|SUIM /582|60.949710|60.872139|61.250148|60.612608|
|PACO-Part /599|41.445074|41.744694|41.691069|41.497054|
|LVIS-92i /599|44.273080|44.586329|45.173469|46.770297|

源：[跨集audit](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/rcg_transfer_20261007/audit.json)、[冻结RCG2配置](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/rcg2_frozen.json)。冻结配置记录在fresh600全部标签上选择sizecut，再用于其它数据集，不在各迁移集重拟合。

- COCO的RCG +1.280923中，净TP项+.025350、净FP项+1.255574；主要是范围修剪。fine进一步+.380233，TP/FP贡献约各半。
- PASCAL的RCG +1.459568中，净TP项−.335512、净FP项+1.795080。fine仅+.023832，sizecut再−.519815；part数据并非统一无法从图获益，但边界读出的剩余价值小。
- SUIM的RCG −.077572是净TP+.547841与FP代价−.625413抵消；fine+.378010，但sizecut再−.637540。它不能用“小目标削边一定有效”解释：GT<2%的73例RCG27.690866→22.072622。
- PACO的RCG +.299620区间跨0；fine−.053625，sizecut再−.194015。当前证据没有显示这些范围操作能可靠区分同父部件。
- 旧LVIS主要收益在sizecut阶段：相对fine +1.596828，净TP项−1.985246、净FP项+3.582074；是删错收益超过删真代价。不是RCG graph跨集贡献+2.50。LVIS复用COCO照片域，也不能把它叫图像域迁移。

GT面积分组只用于解释。比如旧LVIS<2%有358/599，full34.653对native30.722；但SUIM同范围反而失败。按数据集名或GT面积选图/阈值不是可部署统一规则。

## 4. 最新LVIS图正结果的强控制

以下全部同一已曝光LVIS100、CLI1024，来源是各冻结原`report.json`，条件rank两臂及主baseline在本轮再次完整重放。

|干预|candidate|匹配控制|MEAN|完整FoRIS|判定|
|---|---:|---:|---:|---:|---|
|参考角色密度graph|47.386081|47.226239|47.109370|47.704029|有小差，未胜FoRIS|
|positive-tail local rank|47.187467|同L1幅度47.473022|47.109370|47.704029|所谓local排序收益被幅度控制超过|
|条件rank graph|47.701403|同E总边权47.472579|47.109370|47.704029|较匹配+.228825，仍未胜FoRIS|
|reference-weighted fidelity|46.268881|同总fidelity45.935585|47.109370|47.704029|虽胜较差控制，整体明确下降|

条件rank graph固定`G=s>.5`，u为guide在G的条件CDF；仅至少一端在G的边乘`1−abs(u_i−u_j)`。matched在完全相同E统一衰减η、匹配E总边权；原y/A、λ16、原FP64 H对角残差、CG与读出不变。它没有新reference内容，也不保证相同节点degree或Laplacian谱，+.228825尚不能唯一归因于目标身份信息。

原图两臂47.789883/47.573653，MEAN47.182508，完整FoRIS47.873899；候选仍低FoRIS.084016。图外边不变不等于G外最终预测不变。source来自：[条件rank报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_context_rank_graph100_20261009/report.json)、[tail控制](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_tail_rank100_20261009/report.json)、[fidelity](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_reference_fidelity100_20261009/report.json)。

同100 FG-anchor曾CLI50.047216，但另200完整复核45.533288低FoRIS45.904436；这条已失败的复核也应保留，不能重新以旧100高点支持任意读出组合。[复核原报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_fg_anchor200_20261009/report.json)。参考pair graph只做过首例构图，已搁置，无完整解/mask/分数，不算部分成功。

## 5. 新机制量：图既覆盖正确unary，也修复错误unary

冻结问题为：“条件rank残差是在识别被图误删的目标，还是仅放松图已有删减？”直接从相同读出生成的rank-only mask到MEAN、到candidate/control比较，query GT只用于以下诊断。

原MEAN相对rank-only CLI45.943340→47.109370。graph删掉246,529 TP、补回91,020 TP，同时删486,210 FP、加264,578 FP。精确mIoU收支：补真+.568551、补假−.774426、删真−4.589978、删假+5.961883，总+1.166030。**图的误删真实存在，但错误unary被图清除也有价值；直接关闭图会降低完整输出。** rank-only零交集1、MEAN7，也不能只按零交集数选择方法。

条件rank与matched相对MEAN的冲突账本如下，均为CLI实际像素，不是token计数：

|改动|条件rank|同量控制|
|---|---:|---:|
|恢复原graph删除的正确positive-unary TP|9,418|5,034|
|恢复原graph已正确删除的错误positive-unary FP|19,125|8,924|
|新增TP，原unary也为BG|247|222|
|新增FP，原unary也为BG|934|474|
|新删原unary/MEAN都保住的TP|5,453|1,639|
|新删原graph补回的TP|9,104|4,295|
|新删原unary/MEAN都误保留的FP|9,208|3,296|
|纠正原graph误加的FP|23,511|11,960|

条件rank新增TP中97.44%是恢复原graph误删，但它只追回全部被误删TP的3.82%；同一动作恢复已删对FP的比例为3.93%。它还删14,557 TP，故总TP净变化−4,892。class mIoU仍+.592033不矛盾，类权重和各类union不同；不能把“补真贡献为主”写成总召回提高。

同E统一削边已达到+.363208，具体条件rank再+.228825；这可以作为固定顺序的实际增量，不是相互独立的因果份额。相对matched精确贡献仍是补真+.432754、补假−.207758、删真−.120871、删假+.124700。

实际被graph删除ROI内的FG/BG质量加权AUC，先相交像素再pool到64²，每例有FG/BG时才纳入：

|信号|全部有效90例|query<1%的有效30例|
|---|---:|---:|
|原source s|.518425|.608418|
|原unary y|.519647|.606423|
|原guide|.543752|.607811|
|条件rank的完整final场|.509996|.579181|
|条件rank减MEAN残差|.515095|.533375|
|同量控制减MEAN残差|.496212|.582116|
|条件rank减matched残差|.530716|.525521|

没有出现可用来广泛选择“正确unary应保留”的强排序。该表不否认条件rank已有小幅完整分数增量；它反驳的是“这项正差证明已经找到可靠困难区身份信号”的更强主张。小目标中候选残差排序低于matched，但候选完整mIoU21.439252高于matched21.055882，仍低FoRIS22.139022。排序和最终固定阈值收益不能互相替代。

## 6. 方案、失败条件与当前不执行项

本线交付的是已可重放的图候选/匹配控制与机制测量，不新增缺乏支持的图强度或mask配方。最小可运行用法是：当主线有新的合法unary时，保留其直接输出为一臂，再在**同原图、同unary/A、同读出**上比较原graph与固定条件rank graph/同E质量控制；先验证正确unary与错误unary的冲突分布，再看完整FoRIS、MEAN及新直接输出的净收益。不能只根据旧COCO总+1.4自动叠图，也不能根据本LVIS100+.228825选择新Deep/PACO图参数。

这不是本轮已运行的新候选；本轮完整运行的是旧LVIS100配方的原样再执行。最终读取ROOT_RESEARCH时，主线APD-ridge whole/local已经完成第一项完整输出：有相对FoRIS/MEAN的正差，Deep仍低于快九窗。它的下一问题是如何保住真实局部观察与目标身份各自的收益，不能因本线旧graph正差自动叠图。

可反驳/失败条件：

1. 新graph即使胜matched，若仍低于完整FoRIS或新直接unary，不采纳；reference fidelity已有这种反例。
2. 正确unary被恢复的同时错误unary更大量恢复，且完整类I/U净损失，则“保护unary”机制失败；不能以召回或零交集改善替代mIoU。
3. 加入fine/residual却不能胜过同真实观察的fine16或直接MEAN-fine，则不能称残差特有优势；缺直接P(G)结果必须留空。
4. 任何跨集结论必须同样本、同数据映射与指标重聚合；本轮旧RCG2跨集mask不在本机，不能直接重放或拿它当当前raw-native基线。
5. 配方如再次使用query开发GT挑阈值/源组合，必须保持开发选择标签；三折留出和bootstrap均不能把历史暴露变成未见确认。

没有恢复旧全量/6000/4000编码队列，没有修改全局PLAN/state、原输入或其它研究线。没有把缺失资产补下载或重编码。

## 7. 与其它线的接口

已读01/02/04/05的阶段MD及ROOT_RESEARCH最新完整输出。01确认局部九窗多数擦除是另一个可测融合问题；02确认PACO whole竞争主要靠同父内FP削减且新增区h排序弱；04确认简单面积/纯度/视图一致性未提供可靠选择器，且AUC差不等于anchored完整输出差；05确认旧ridge/Huber合法reference拟合并不等于query标定。这与本线一致：要分开“已有场范围改善”和“新增困难区判别信息”。不能将它们的不同ROI AUC与本线graph删除ROI直接作差。

待主代理整合：保留shifted-readout与scalar的小幅同源正证据；纠正fine后缀/12源照片隔离表述；当前没有来自本线的证据支持将图作为新ridge的默认组件。后续若要运行上节的图对照，应由主线根据已生成完整unary的实际问题选择，不为历史正差自动追加。
