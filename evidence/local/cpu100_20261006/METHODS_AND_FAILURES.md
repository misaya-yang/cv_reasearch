# 方法与失败证据：CPU100历史及Astra300当前映射

**当前任务已改为按给定Astra300原始定义实现并直接评固定600，不先筛200；尚无Astra600质量测量。300个ID定义已固定，已到件实现、待审批次、缺真实观测与待实现分开记录，不把代码或合成检查计成300方法交付。CPU100历史仍有真实开发收益线索：Huber在200例比ridge高0.7109 pp；RGB02从4例正增益转为200例负。本文件只整理证据，不安排实验、不创建第二PLAN。**

记录日期2026-10-07；本次来源/到件状态窗口为05:14–05:25 UTC，活跃源码可能继续变化。只写本文件，未SSH、未跑方法或实验；下文读取的运行收据来自对应执行者，不是本整理员新跑结果。

## 全队公共阅读区与两条当前claim

本文件是全队共用的来源、方法、失败与到件证据阅读区；各owner把源码/原卡映射/实现假设/hash/ready或缺项落到自己的证据目录，本整理仅把实际新来源和收据归入同一MD。批次不互相刷状态，只有必须root处理的故障或完整批次交接进入聊天。协议与行动范围仍见当前AGENTS、[PLAN](/Users/yang/projects/CVPR2027/docs/research/PLAN.md)和用户指令，本文件不另立规则或待办。

两条最新目标来自当前[CLAIM](/Users/yang/projects/CVPR2027/docs/research/CLAIM.md)，尚无新完整方法证明达到任一条：

| 目标 | 对应必须交付的证据 |
|---|---|
| 极简效率：极简、高效率，完整准确率超过同协议完整INSID3 | 同输入/样本/分辨率的完整INSID3与最强简单替代；完整mask质量、端到端/缓存后CPU延迟、内存及实际编码次数。轻量、拟合正确或source准备不等于准确率已胜。 |
| 适度延迟换精度：允许适度推理延迟，超过同协议完整FoRIS及已有强方法 | 完整FoRIS、实际MEAN/RCG/fine和同信息/同预算强控；完整净收益、逐类I/U与四动作、配对区间及精度/延迟。不能只胜弱prototype或跨pipeline挪用历史分。 |

可原创或重构既有方法，可使用FoRIS操作，不预设必须保留四段流程；可读真实内部层/attention/响应/新增view，不受现缓存或末层限制，仍不增加外部类别名、模型、图池、基类标签。旧“稳定至少+2”是历史愿望，当前不是无限搜索或论文价值的硬门。两路研究与给定300项交付并行，不互相替代；本整理员不设计或启动任一路实验。

## Astra300当前来源、合同与到件状态

原始来源为[用户给定300定义](/Users/yang/Downloads/冻结DINOv3单参考分割_300个完整方法候选.md)，仓内封存副本为[astra300.md](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/source/astra300.md)，628,580 bytes，实际SHA256 `8c85ec4e8d1f32b7b98aef902ad94a4bb525b2ebd7023302f7551dbfab356f83`。[methods300.json](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/source/methods300.json)实际SHA256 `5fcd68101c95d687f13ff1e4f9024193a2bf453eca33543a505fbe399f85412e`；本次只读核对300个唯一ID、number 1–300连续、300段definition的逐段SHA全匹配。JSON中的初始`supplied_pending_implementation`不是实时实现状态，`owner`是原六组写作分工，不是当前四组执行职责。

原文六族各50：A001–A050参考身份；B051–B100跨图对应；C101–C150查询组织；D151–D200负证据与参考内辨别；E201–E250真实RGB/混合采样；F251–F300联合决策与有限新观测。每卡的完整原文、原题、缺口、合法输入、终判、强控、反例与成本都由上述JSON保留，不在本文件另造或改写300条定义。本整理核读全局合同、六组公共协议及到件批次说明/收据；逐卡原文覆盖限实际读到的卡，不声称已经语义精读全部300条。

| 当前执行组（各75） | 原始ID范围 | 本窗口到件证据与实际状态 |
|---|---|---|
| 001–075，reference_evidence | A001–A050、B051–B075 | 首批8ID已正式审查：revision1 **7个主方法通过**（A001/002/003/004/005/007/008），A006有缺pure-token组件被漏掉的合同错误、需修订；取代先前“A8 ready”的泛称。A009、A011–A016另有7方法合成检查，待审；A010求解器存在但原卡要求的跨图坐标稀疏预检真实证据缺失，显式不可运行。A017–A020只见活动源码注册调用，未据此计完成。其余本窗口未见完整到件证据。 |
| 076–150，local_structure | B076–B100、C101–C150 | status.json到件C102/C104/C105/C106/C108/C109/C114/C116/C117/C119共10项，`implemented_for_review`、固定600结果null。活动源码已增加注册调用，不能拿旧10项status外推新增项已审/完整运行；其余仍待相应批次收据。 |
| 151–225，decision_risk | D151–D200、E201–E225 | 首批D151/152/153/155/156和二批D160/161/162/163/164共10项有recipes及完整合成probe，两批ready均`review_status=pending`；PROGRESS尚余65。各probe只绑定其记录的source SHA，后续活动源码不能静默继承旧分数。D160真实pre-final-LN缺项另列，不把合成幅值或合法angle-only退出分支叫真实幅值主机制运行。 |
| 226–300，cross_image_matching | E226–E250、F251–F300 | E236/E238/E242/E245/E250共5项实现与数值/完整合成mask检查已到件，root报告E5 review；没有接受/真实600质量凭据。E235/E239/E240/E243/E244只见活动源码导出，待完整收据。F251–F275已有内核；其中23项wrapper及同名control合成协议检查通过，F256/F268显式缺实际观测，当前group export未因此交付全部F；F276–F300只见内核源码，不能当已完成公共协议接入。 |

以上为证据层级快照，不加成“已交付方法数”。未列入具体到件ID的定义仍在300映射中，状态为待完整实现/收据/审查；“未见收据”不等于断言活动源码绝不存在。当前300项的自然mIoU、逐类I/U、四类增删和同协议完整FoRIS/INSID3比较均unknown。原文的30/200诊断顺序被最新固定600指令替换；各卡的参考内留块重建、校准与有限求解仍是算法组成，未被取消。

### 实现合同与合法输入的区别

固定模型、单张R RGB/完整类别MR、Q RGB；不加查询真值、类别名、额外模型、图池或基类标签。末层缓存是加速资产，不限定设计可观察信息。reference小头/字典/阈值允许按原卡求解，但不能叫无需优化。A的U和D的G把连续场一次映回原尺寸再阈值，与CPU100的binary1024→original读出不同；本轮保留源规定的renderer和零模块控制，不把渲染差额当新增类别证据。

各组原点/回退不合并：A为每role至多64锚5NN差；D为完整R coverage≥.5最近role；E通常使用实际MEAN宿主，指定E207/216/218/237/238/242/243等才用U0；B的H/W/Z还要求实际完整FoRIS宿主；C、F也有各自基场及退化规则。项目历史`model.raw_nn`另绑定，不能借用旧42.90给任何新NN/均值头。源内拟合必须先移除未知留块及规定缓冲，再从FG/BG同时重建所有标签相关状态；只把held-out wf置0会漏成BG。共享编码上下文的参考块校准不提供独立跨图保证。

### 旧1200可读与新固定600生产状态

[逐项1200可读性核查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/existing1200_readability.json)实际SHA256 `18bb8c5404f80b1bc1265c5212b6df205a8e94427d301c9a9ae24d5da29a1d45`：1200唯一manifest key，feature、packet、R/Q原RGB、MEANfield及stored host mask各1200可读。此前目录glob只数到546个文件，不表示manifest缺654项，已撤回这种解释。此审计曾打开packet truth成员的shape/dtype，因此`query_GT_read=true`须保留；没有以标签做推断/评分，packet禁止进入方法推断。可读性不自动证明某个卡的producer、renderer或内部状态合同已满足。

[实际MEAN600宿主绑定](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/mean_host600.json)实际SHA256 `93250d921749a7aaab57aefe7f8bff9fdcd1c453a130e5b2f1ca67a24c9eff1f`记录exact_fixed600、连续field和stored mask逐例hash，以及其旧processed链：单位化→条件FoRIS Part1→单位化→FP16。stored M0为连续64→1024阈值→二值1024再映原图阈值；另一个continuous-original zero单独列。这是合法且来源明确的MEAN宿主，不能改称raw-native或完整FoRIS强基线。[首对provider核查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/provider_first_pair_v1.json)确认该对actual host mask identity、工作mask差0、无QGT读取；不外推600全链已运行。

新fixed600 raw提取root本窗口报告已440+，尚未收到全部600完成的封存收据，不写600已全备或300已评分。新native source pack为FP32 final-LN原生patch，无FoRIS Part1；CPU100匹配loader随后单位化。source pack保留的final-LN原始norm可供明确需该量的卡使用，**不能供D160冒充pre-final-LN**：D160原卡明确要求最后block输出、最终LayerNorm之前的实际h；reviewer曾仅看标题误判成final-LN norm，已按原文纠正。D160原卡允许未保存LN前状态时幅值臂退出到同信息angle分类器，这是已写明的合法退化；输出可运行不等于幅值主机制获得真实测量。真实中间层、QKV/attention、特殊token、输入导数及额外view仍须按卡给实际绑定观测，不从unit final或旧processed缓存伪造。

最新PLAN已授权1200扩充，且旧1200资产逐项可读；原生1024FP32目前生产的是同一fixed600，不能把旧1200可读写成raw1200已齐。fixed600独立评分清单600例/80类、四折各150，PLAN记录SHA `913272d7f0d4290141a870e3d5a4922886d6345c64cc1693930096039e07c8b6`；本整理没有远端重读该清单。1200仍是复用开发数据，不称新独立确认；缓存复用/生命周期由root调度，本MD不新起资源计划。

[实际CPU冻结encoder首view parity](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/encoder_first_view_parity_v6.json)实际SHA256 `2215d0ad6a55d6c3761ae9ed989f66b5868c2be038873abd1016293624e72bcc`：有1次真实FP32 CPU前向、同checkpoint/config、eval/frozen、无Part1；输出64×64×1024，单位化与已有native pack最大差0、重复缓存差0。实测该前向wall36.9978s、CPU73.5537s（2线程），首view过程46.8103s；这是单view receipt，不是某方法端到端时间、全部内部观测或全部600成本。不能再把真实encoder适配说成始终假callback，也不能由一个正常view声称输入梯度/尾层重放全部可用。

### 当前检查能支持什么，不能支持什么

A001–A008修订后[冻结manifest](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_001_075/batch01_revision1_manifest.json)绑定[checks](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_001_075/batch01_revision1_checks.json)，后者实际SHA `c5caeca1d7c08a3e7fda2c6ad8bd28d035fd5d16df2cfceb40a7c6f6df1d2b07`。覆盖率阈值已按literal半role加权|prediction−coverage|修正，不能把仅不同类条件归一的旧阈值检查称原卡实现。[正式revision1审查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/reviews/batch_A001_A008_rev1.json)，本窗口实际SHA `94648d9f21d124aa9660dc6f856c4b1236d95bef37fa7410d8c69bd2bbec32f4`：7主方法通过可直接固定600，A006需修订。旧revision0共同校准错误，曾误启动进程已由root按实际服务器hash停止；旧partial receipt不与修订结果合并。

本整理此前误从陈旧READY说明把A004的revision0四角差2.43e−16迁到revision1，已纠正：revision1实际checks记录四角差0.0629708293、16个SVM拟合均无失败，参数由balanced total hinge weight1改为per-sample C1，不能沿用旧强制均值差的代数退化关闭新版本。零four-corner只推出包括intercept的可分离learned potentials，不等于centroid cosine；原文这个欠定义说法已显式修正。独立性不由8个ID直接推定，但不能再把A004 revision1自动计0。此例说明必须对准冻结版本的实际record，不能用较早READY摘要替代。

A006实际像素组件计算存在，但把面积>一patch且无pure-FG token的合法组件丢掉，可能把真实多组件R变成单组件B0，违反原卡“每个合格组件建bank”及共同缺pure-support时coverage均值协议；它当前未通过。review要求保留该组件真实footprint权重并用其weighted FG mean，方法owner负责修复，本整理不改代码。另有控制命名/匹配缺口：A003的same_endpoint_5NN实际为global bank，A004的same_anchor_RBF实际global FPS而非SVM支持点；这些不是7主方法立即运行的附加质量门，但不能用于宣称配对或SVM独有收益。

A009/A011–A016的[batch02工程检查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_001_075/batch02_checks.json)只有16维合成144R/48Q完整输出。A009在该合成例所有48Q到30步仍KKT>1e−4，实际求解后按合同回B0，不能从“输出完成”声称竞争机制已改判断；正交字典反例KKT=0是求解器检查。A014该例8域各仅一个role，均identity、学习迭代0；A012半空间0，也不能当主体活动。A010缺真实跨图预检证据，未完成卡，不把普通近邻fallback填成已交付。

D首批5项[probe](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_151_225/batch01_probe.json)在easy checkerboard全都源内选zero adaptation；direct builders分别运行不等于最终query新增收益。二批D160–D164的[probe](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_151_225/batch02_probe.json)实际SHA `2170e89fd0d5457e85334f71aa247c1eac649d9c173587aa18fe60aaa940c33b`也全选zero adaptation；幅值、robust median、shared map和mode-constraint仅有合成算法/反例活动，不是自然类别收益。[二批ready](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_151_225/batch02_ready.json)绑定group SHA `29574c27869c79dd34784a585de90499a9deb82d29e9aeb97e36aa8d0a19fac2`、helper `0c2a1665fb018a36e565bff05b8b59da239912863bd5ec957c9c19ec91fcd973`、d160_164 `7dafa674072bbd25c9bca1a51b7aef3f77dd7158965da5ef8a37d7b016045e4d`，仍pending review。

E5的[工程/完整合成检查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_226_300/e_batch1_checks.json)对仿射QP、footprint伴随、占有量primal-dual gap有数值证据，actual_encoder_forwards=0、自然quality unknown；shared-colour合成正例以及colour-reversal/same-colour负例均保留，不能用合成1.0 IoU代替自然身份可靠性。

F251–F275的[wrapper协议检查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/group_226_300/f251_275/protocol_checks.json)实际SHA `920661d7bb2f7e1affb96aaf4fbebafadc4ddca16b09d39c8b300f719b02dbb6`支持23 wrapper/控制合成完整输出、12次源IoU标定、移除held标签及一patch缓冲、字典逐折重建、共同二值disagreement读出。它不含真实DINO或自然评分，源例选weight0；F256需实际三view双端观测，F268需真实末两block前状态及重估QKV尾层重放，resources仍executed=false。

[公共API检查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/common_api/checks.json)和[runner检查](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/runner/checks.json)只支持输入封存、缺资源拒绝、GT隔离、原尺寸连续renderer及paired scorer等工程合同。runner的`all_arms_complete=true`属于合成测试工作区，不是Astra600 run结果。[源合同review](/Users/yang/projects/CVPR2027/evidence/local/astra300_20261007/reviews/source_contract.json)本窗口实际SHA `5ebbbee04a5fe63b344421d9e22609741a4d95df2942dbd0d7c05afbde7c24d6`；它是共同合同审查，不是300卡逐项接受或已测收益。

### 既有失败对本轮的具体约束

4例RGB02比prototype高3.8907，200例反为−4.5711；因此本轮合成/小样本正例只证明可达判决，不能给300卡预先写收益。200例Huber比ridge高0.7109的真实开发边际应保留，但相对prototype的9.7928大头已被ridge解释；A/B/D小头不能把拟合相对弱头的收益全部称新身份信息。QP01/02四例约+6.8到200近0，local001大负，提醒参考或query组织正确求解不保证跨图终判。

旧两槽EM补回53.9%漏检却新增只有25.2%为真、只补−2.44；它可提示A新增集合有可用TP，但GT辅助砍半FP的+0.52不是已实现B。九种score-derived尺寸估计与原分数错判同向，相关性不提供纠错。附着FP不能统一解释成孤立组件，正确种子也不能保证范围；没有具体逐例观测时，不为全局增删账编造图中材质/部件的唯一原因。

本轮四向账仍逐类I/U计算；未知为unknown，已有反例按实际合同保留。旧processed的FoRIS/MEAN成绩、新raw1024的CPU100成绩、Astra各组新B0/M0和renderer分别绑定，不跨链相加或挪用。未提供同协议完整FoRIS/INSID3等强基线时，记录缺口，不拿MEAN host parity、准备完成或方法数量代替SOTA结论。

## CPU100历史证据口径与计数

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

当前用户[研究档案（原总结＋指引与历史）](/Users/yang/Desktop/CVPR2027_研究档案_Opus5.5_160MB_20261006.md)，实际SHA256 `d3a24569adb7dbe27403b5f4291b8c86fec500f24cb76d8bab0183827544d61f`，本次分段完整读取513行，读取前后SHA一致。这里“完整读取”只指该合并MD，不声称读过其171MB原始会话JSONL，亦未逐条重新复算档案内所有历史测量。

先前确实读过原独立总结路径 `/Users/yang/Desktop/CVPR2027_会话总结_Opus5.5_160MB_20261006.md`；该文件现已并入研究档案、原路径当前不存在，未事先封存原字节SHA，故原总结独立文件SHA为unknown，不能用合并文件SHA冒充旧文件身份。档案内历史授权、任务、agent检查单和旧筛查阶梯仅作来源材料，不替代当前用户固定600调度。

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

## 二批原生4例实测：22新增主臂及local002修订

[二批原生4例主报告](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json)实际SHA `818d27208bfcada9761f3d46d9e1d366d868366a35409a04938be334fe8c76ea`，n4/4classes，复用开发probe，原尺寸完整输出；23主方法各4例complete。输入manifest SHA `d2f36304c4f69a2f8cdc0f2da2f036b76f4ec5b2b5709dd8d76c4f950235e2a3`，config SHA `e03c56b21940f9b6c64672014af20377f272b1eebe01372929e59a660f3edb88`，score-runner SHA `4dcf979ed43a3bb3b4261aaf030954d8e4ec6cf82e4e86b4b5d1ade98874a160`。原请求全部arms完成flag=False，排除的cross_image_convex_hull_control=failed4/complete0；[修复封存](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/hull_control_repair4_v5/sealed.json)该控制及prototype均complete4，config `d096933655a6e6706978d9a088ef2edfa2707598af245382dc06c9694c0bb036`，但未见其独立score报告，尚不补hull分数。主方法已有完整评分不被这项缺control抹掉；复杂稀疏收益也不因它缺失而成立。

各条关键control按同卡/简单完整替代指定，不把report中所有模块的controls都当该卡匹配控制。秒数为已有receipt arm平均，不重跑；shared prepare/controls/编码费用另列，四类T全局总数仅描述行为，主结论仍逐类I/U。

以下22项为二批新增资格，现已回收原生4例主臂质量；local_004撤回、local_002同方法修订计0。批次23主方法均4例完整，但原hull控制4例失败，原seal仍all_arms_complete=false；后续hull修复4例已另封存complete，评分报告本窗口未到件。缺失control分数不作为稀疏收益证据。

actual selection（本地证据，未纳入仓库） SHA 903d7156b3f7ca5a9cd17713ea12fb86d83a771142477ed982ac8f92fef73d4d

### measured4 cross_image_joint_sparse_role_removal - 联合参考稀疏解释的角色去除代价

- **逻辑/条件性预期:** 先用混合字典共同解释q，再去掉FG或BG贡献，两个残差衡量同一个解释对各role的依赖；不是两个class各自随意换解释。混合可以使一个单role最近点胜利变为另一个role解释占主导，但系数竞争是否对应语义尚未知。
- **预期收益（假设）:** 若目标贡献必须与背景联合解释才能显现，恢复目标；若干扰只能借多类字典拼接，可由角色去除改变身份
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 相同atoms下独立FG/BG非负OMP3残差；same OMP coefficient summed-role vote；class cone/full convex hull；prototype和nearest。同signed margin common render。
- **原生4例v4实测:** mIoU 35.6663; Δprototype +6.6820 [-6.6059, +19.9700]; T=24 / 10,441 / 19,193 / 53,828; receipt arm平均1.857s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** cross_image_dense_class_cone_control=39.4955; Δcontrol -3.8292 [-7.9154, +0.2570]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/cross_image_matching_batch2.py SHA `bfde408146bb24f6ddbcddc03cb2db4788f511680a8d8dfb64b6bbb30659ff5a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 相对prototype+6.6820但CI跨0；dense cone高3.8292、independent sparse高3.3249、coefficient vote高3.1778。后者paired差CI为[−5.6885,−0.6671]。即使暂缺hull评分，也不能写复杂joint/sparse增量成立；当前主要删区，仍删真19,193。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.cross_image_joint_sparse_role_removal。
- **未知前提/审查限制:** small NNLS independent audit; same-budget independent OMP, dense cone/hull and coefficient-vote controls; reconstruction not identity
- **特有反例:** 字典高度相干时alpha不唯一，贪婪选错先手；真实目标需>3原子，或背景原子替代FG导致误删。重构好不保证语义对。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### measured4 cross_image_exemplar_facility_cover - 全查询共享参考解释的设施选择

- **逻辑/条件性预期:** 联合总代价sum_i w_i min_open(1-q_i dot r_a)+0.05*number_open，使低支持的偶发原子不值得单独打开，而多个相似query能共付解释成本。绝非新语义证据，依赖重复解释比孤立误匹配更可信的条件。total valid query mass归一化，因此数量增加代表面积影响。
- **预期收益（假设）:** 关闭只服务孤立错误query的FG解释，同时允许同类多实例共同打开参考解释
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same dictionary全部开放nearest；相同开放个数的global unary排序和随机/固定ID选择；same source prototype。完整joint objective值只是求解账，不能当质量。
- **原生4例v4实测:** mIoU 0.0000; Δprototype -28.9843 [-46.1480, -11.8205]; T=0 / 0 / 73,228 / 118,865; receipt arm平均0.367s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** cross_image_all_exemplars_control=28.2192; Δcontrol -28.2192 [-38.7801, -17.6583]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/cross_image_matching_batch2.py SHA `bfde408146bb24f6ddbcddc03cb2db4788f511680a8d8dfb64b6bbb30659ff5a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 四例最终空输出，开放个数匹配的unary/random控制也为0，全部开放nearest为28.2192。固定开放费/共同解释本版使身份输出整体关闭，matched-count同为0只显示退化，不是结构收益。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.cross_image_exemplar_facility_cover。
- **未知前提/审查限制:** state complexity/area prior, matched-open-count controls and tiny-target deletion negative; small subset optimum audit
- **特有反例:** 正确目标很小只需一个独特原子，正好被开放费删除；重复错误区域成为便宜FGhub，会被保护；source多个target外观需要很多open atoms而BG简单。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### measured4 cross_image_nonreturn_path_consensus - 跨图无立即回退路径的角色共识

- **逻辑/条件性预期:** 输出3-hop Q_i→R_a→Q_j→R_b的末端role平均，只保留j≠i、b≠a。这是directed edge-history状态而非token harmonic固定点；禁返回可使单个互为NN不再凭自循环获得共识。但高连边错误背景仍可提供多条独立路径，须负例。
- **预期收益（假设）:** 抑制互为NN单边回声，把身份依据转到不重复的参考原子支持
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same dictionary、sameW、same3hop长度但允许回退的ordinary walk；1-hop source kernel；wrong high-connectivity background桥接负例。不能与成本/输入不同的old base score仅比总分。
- **原生4例v4实测:** mIoU 25.9988; Δprototype -2.9855 [-10.9217, +4.9508]; T=10 / 23,229 / 15,376 / 39,635; receipt arm平均0.523s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** cross_image_one_hop_path_control=26.2995; Δcontrol -0.3008 [-0.9016, +0.3001]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/cross_image_matching_batch2.py SHA `bfde408146bb24f6ddbcddc03cb2db4788f511680a8d8dfb64b6bbb30659ff5a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 比one-hop低0.3007，新增TP10、FP23,229且删真15,376。排除立即返回没有把路径一致性变成独立身份；现有路径图可共识错误，只是解释，未归因到每个边。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.cross_image_nonreturn_path_consensus。
- **未知前提/审查限制:** same-W ordinary walk/one-hop and direct enumerator; repeated false bridges may still be coherent, paths not independent samples
- **特有反例:** 错误类别在Q内部高连接并匹配多个FGatoms，则禁返回仍强化错误身份；单实例/少Rmode没有independent path会fallback；真实唯一part对应可能被排除。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### measured4 cross_image_source_opponent_matching - 参考角色全局对手匹配的中位判决

- **逻辑/条件性预期:** 在R内作最小成本FG/BG一对一匹配，再对每pair的unit difference方向取query margins中位数。normalization和非线性中位读出使不同配对改变决策；必须用all-pairs和independent-nearest opponent同中位控制判断global matching增量。R端均匀mode投票是先验，不能称semantic保证。
- **预期收益（假设）:** 避免单个参考BG对手被无限复用支配比较，在多局部决策方向存在时改变身份
- **source/合法输入:** [source](../../../src/ics/cpu100/cross_image_matching_batch2.py) SHA bfde408146bb24f6; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 相同K字典的独立nearest-opponent median/allCartesian median/unit-globalmean；costmatching只在R上，Qrender和信息完全一样。matched randompermutation median为控制，非多方法。
- **原生4例v4实测:** mIoU 19.4300; Δprototype -9.5542 [-17.2542, -1.8542]; T=822 / 96,868 / 2,817 / 183; receipt arm平均0.375s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** cross_image_nearest_opponent_median_control=27.9560; Δcontrol -8.5260 [-19.6556, +2.6036]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/cross_image_matching_batch2.py SHA `bfde408146bb24f6ddbcddc03cb2db4788f511680a8d8dfb64b6bbb30659ff5a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 新增TP822、FP96,868，低于nearest-opponent median；在R上优化配对成本没有带来Q终判收益。错对手/参考模态失配符合预列反例，但不能把全局账当逐图原因已确认。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.cross_image_source_opponent_matching。
- **未知前提/审查限制:** nearest/allpairs/random opponent same median; K1 activity and source-matching cost distinct from query quality
- **特有反例:** 独特BGatom被迫配给无关FG，产生不代表真实边界的方向；target参考模态数量少会K=1退化；unit小差方向放大source noise；多数局部pair本身错会中位失效。
- **证据:** [card](cards/cross_image_matching_batch2.json); [batch02](reviews/batch02.json)

### measured4 local_003 - Foreground/background regularized convex patch reconstruction

- **逻辑/条件性预期:** The ordered nine-patch centered DINO tensor is an available continuous joint observation. Independent-token matching or Gram invariants discard its coordinate directions and ordering. A constrained convex mixture of reference tensors can represent intermediate target local patterns without requiring one nearest template or transferring an entire silhouette. This changes the decision, but whether its reference target span is sufficiently distinctive is unknown. The L2 objective does not guarantee sparse coefficients; it is called regularized convex patch reconstruction. Different ordered patch evidence, rather than the penalty, is the proposed counting distinction from central reference_hull.
- **预期收益（假设）:** The normalized mixture of distinct unit target patch tensors lies outside their convex hull in general, but can still have a smaller nonzero regularized reconstruction residual than the background hull while being farther from every individual target tensor. Ordered raw patch observation is the proposed extra information; L2/cap are not claimed sparse, independently new, or a real gain.
- **source/合法输入:** [source](../../../src/ics/cpu100/local_structure.py) SHA 63b1cfe14ecd5a3e; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Same nine-patch tensors, dictionaries and renderer: nearest individual-template reconstruction error per class.; Same constrained convex reconstruction on central DINO token only (old central-feature hull-type control).; Complete local001 Gram classifier on same legal inputs/renderer, keeping its distinct invariance limitation.; Same class tensors with spatial stencil order shuffled independently in reference; bag contents remain but organization is removed.
- **原生4例v4实测:** mIoU 19.9813; Δprototype -9.0030 [-16.2168, -1.7891]; T=14,239 / 268,435 / 16,334 / 33,943; receipt arm平均5.346s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** local_003.central_hull_control=32.9569; Δcontrol -12.9756 [-31.6887, +5.7375]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/local_structure.py SHA `22ac1e36e634b9bcc12431e42729e0f5719bce661bbba9945cd0a40cae889c65`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 有序9patch hull比central-token hull低12.9756；补假268,435远多补真14,239并删真16,334。空间/上下文组合本版损伤迁移，不能把更完整的局部张量当自然身份保证；未因此否定所有局部观测。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.local_003。
- **未知前提/审查限制:** no sparse guarantee from L2; unit mixture is not exact convex combination, use nonzero residual superiority; central hull/nearest ordered patch/complete Gram controls
- **特有反例:** If FG and BG patch cones overlap, reconstruction cannot determine identity.; Large dictionaries can reconstruct both classes and erase discriminative residual; the fixed cap is not a proof of separability.; Viewpoint or feature-coordinate changes can move true target patches outside the reference cone.; Convex mixtures may create nonphysical local composites and accept a distractor.; A center near target boundary uses surrounding reference BG; altered query context can cause target deletion.
- **证据:** [card](cards/local_structure.json); [batch02](reviews/batch02.json)

### measured4 DR09 - 参考原型相似度弱决策的AdaBoost

- **逻辑/条件性预期:** boosting按reference错误重加权组合多个不同anchor的cosine stump，输出sum alpha*h；不是在一个score扫描阈值，也不把source分类精度当query身份保证。
- **预期收益（假设）:** 当合法MR描述非线性多模态而单决策错误可被另一anchor纠正，boosted分界可救回目标/删错背景；仅hypothesized。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同anchor最佳单stump、same-input nearestreference、sameanchor linear ridge、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 19.4702; Δprototype -9.5141 [-30.8099, +11.7818]; T=35 / 328 / 40,738 / 86,989; receipt arm平均0.259s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol -19.4436 [-46.1132, +7.2260]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 删真40,738、删假86,989且完整分低于prototype9.5141。boosting可以拟合R弱分类但本版误删Q目标；多stump/source重加权没有给出跨图身份安全性。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR09。
- **未知前提/审查限制:** single stump/sameanchor ridge/NN; source-error focus can amplify nontransferable rare material
- **特有反例:** 少数nontransferable reference材质成为高权重error，boosting会追逐它；source100%分类正确仍可能query全错。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 源码已核：bank为空/无信息时回退pure-reference nearest-neighbor；不是泛称prototype。该fallback不算boosting生效；真实二批质量仍unknown。

### measured4 DR10 - 平衡参考重采样的最大间隔委员会

- **逻辑/条件性预期:** 16个class-balanced参考bootstrap分别求两class hull间隔，Q按hyperplane sign majority；它平均支持向量敏感性，不提供独立证据或IID概率保证。
- **预期收益（假设）:** 若support-vector异常出现率低于委员会多数，可能避免单hull误删/误增；稀有合法FG也可能被bootstrap丢失。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 全reference DR08、同样16bootstrap的prototype voting、DR04 trimmed risk、raw NN。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 39.3909; Δprototype +10.4066 [-5.6999, +26.5131]; T=6 / 551 / 13,627 / 50,394; receipt arm平均0.793s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol +0.4771 [-4.1343, +5.0884]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 相对prototype+10.4066但宽CI跨0；相对average logistic仅+0.4771[−4.1343,5.0884]。参考重采样hull委员会在本四例确有删错区活动，尚不能把简单logistic已有收益归给委员会；成员共享同一R，不是独立身份证据。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR10。
- **未知前提/审查限制:** same-bootstrap prototype voting, fullDR08; members are not independent confidence evidence
- **特有反例:** 所有source成员共享错identity时稳定全错；真正稀有的referenceFG多数bootstrap未见，会被抹掉。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### measured4 DR12 - 参考原型相似坐标的固定小ReLU判别

- **逻辑/条件性预期:** 固定16anchors cosine坐标上的16-hidden ReLU分类器，可表达若干局部线性条件的联合；MR fit只是available computation，不是新外部model或成功承诺。
- **预期收益（假设）:** 多模态reference逻辑延续到Q时，多个局部平面可能减少mean方向冲突；比NN/核方法更好的收益未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同anchor linear logistic/ridge、DR09 boosted stumps、nearestreference、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 29.6953; Δprototype +0.7111 [-4.6700, +6.0921]; T=52 / 3,121 / 12,632 / 36,109; receipt arm平均0.272s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol -9.2185 [-15.9052, -2.5318]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 比prototype+0.7111区间跨0，却比available average logistic低9.2185。参考小ReLU的表达力不是query收益；公平评价要保留完整简单头，不因比特定压缩anchor控制好而宣称需要非线性。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR12。
- **未知前提/审查限制:** disclose200-step temporary classifier parameter optimization; frozenDINO/no external data remains, cannot call it no fitting; sameanchor linear/boosting controls
- **特有反例:** fixed训练预算可未收敛；source可分但Q不在R模式中；网络可以记住MR context而不识别目标类别。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### measured4 DR13 - 已知参考角色的近邻冲突编辑

- **逻辑/条件性预期:** reference labels已知时，3NN leave-one-out角色冲突可识别孤立class-support；编辑ref exemplars后再完整Q NN，并非在query错误种子上过滤后传播。
- **预期收益（假设）:** 孤立nontransferable FG/BG表征在reference多数邻域中冲突时，删掉它可减少nearest-match错误；hypothesized。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同pure samples unfiltered NN、仅coverage-purity过滤NN、DR06pair rank、prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 23.3740; Δprototype -5.6103 [-13.2901, +2.0695]; T=90 / 25,519 / 15,677 / 17,145; receipt arm平均0.270s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol -15.5398 [-28.5934, -2.4863]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 比prototype低5.6103，补假25,519、删真15,677。已知R标签的3NN冲突编辑在本四例未改善Q；参考邻域多数冲突可同时删除稀有但合法的类别外观，源内孤立不能直接解释成跨图污染。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR13。
- **未知前提/审查限制:** unfiltered/purity-only NN; rare legal target may be removed; class disappears fallback explicit
- **特有反例:** 真正稀有合法FG appearance也会被多数BG邻域排除；同类孤立并不等于污染。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### measured4 DR14 - 覆盖参考分类约束的凝聚近邻支撑集

- **逻辑/条件性预期:** class medoid初始化，按knownR misclassified加入exemplar直到cap训练集被支撑集分类正确；这是reference判别覆盖而不是无label kmeans密度压缩。
- **预期收益（假设）:** 冗余reference细节被压缩时有成本收益，某些Q泛化可能减少偶然极值；mIoU收益待证，source覆盖不是Qextent保证。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** unfiltered NN、same-size fixed/random support NN、class kmeans同K、global prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 19.6278; Δprototype -9.3565 [-20.3093, +1.5963]; T=107 / 97,100 / 16,043 / 26,485; receipt arm平均0.257s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol -19.2861 [-35.6127, -2.9595]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 补真107、补假97,100，完整低于prototype9.3565。按R分类覆盖凝聚的NN支撑集改变了Q分界，源样本被正确分类不保证Q类别范围；此为本版迁移失败，不由小样负判定所有参考压缩不可能。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR14。
- **未知前提/审查限制:** show actual query decision distinction; identical-output runtime optimization alone is not another inference mechanism; same-size NN controls
- **特有反例:** label-conflicting相同feature没有可满足覆盖；容易保留所有noise点；training order会改变supports与Q，不保证de-noise。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

### measured4 DR15 - 参考类别风险驱动的学习向量量化

- **逻辑/条件性预期:** GLVQ更新class-owned prototypes最小化参考correct-vs-wrong squared-distance relative margin；不是把无label cluster当identity或改querycut。
- **预期收益（假设）:** reference跨类混淆需移动判别原型而非密度中心，可能改善接近类别分界的Q；真实质量unknown。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** same2 kmeans不risk更新、nearestreference、DR08hard margin、raw prototype。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 39.5171; Δprototype +10.5328 [+0.2785, +20.7872]; T=299 / 110 / 286 / 32,367; receipt arm平均1.238s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_two_class_kmeans=41.0339; Δcontrol -1.5168 [-4.9236, +1.8900]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** prototype+10.5328的四例CI[0.2785,20.7872]不证明稳定；同两类kmeans仍高1.5168，average logistic仅差+0.6033且CI跨0。真实删错区32,367保留为活动事实；移动原型优化的独有质量收益尚未成立，旧数值反例不抹掉。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR15。
- **未知前提/审查限制:** same initialization no-update control and finite-difference gradient; small denominator/rare-part negatives
- **特有反例:** 原型移动追随source-specific appearance，损失Q合法外观；小分母造成优化不稳定，gradient cap使固定数值行为需核查。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 已有固定构造反驳原故事：近边界预期正例对最强控制−0.1383，高端预期负例反而+0.0060。有限差分通过只支持导数，不证明收敛或跨图收益；保留反例，不自动开变体。

### measured4 DR16 - 合法参考角色的近邻大间隔度量

- **逻辑/条件性预期:** 只用knownR role triplets学习低rankPSD使same-role近、different-role远；先有标签约束再改度量，区别无label方差白化。但reference监督并不保证learned axes跨图语义。
- **预期收益（假设）:** 有稳定class-separating reference directions时，label-aware metric可能保留语义而压制source内部noise；跨图效果未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/decision_risk_batch2.py) SHA 89ce21f6251988db; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** sameanchor identity metricNN、rawfeature NN、referenceLDA、blind whitening(已负族控制)、DR08。 另与same-anchor ridge/logistic及all-legal-MR平均logistic对照(均为control，不增加method count)。
- **原生4例v4实测:** mIoU 29.3949; Δprototype +0.4106 [-6.9832, +7.8045]; T=0 / 2,991 / 21,437 / 59,210; receipt arm平均0.919s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** DR_control_average_logistic=38.9138; Δcontrol -9.5189 [-22.2866, +3.2487]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/decision_risk_batch2.py SHA `89ce21f6251988db499c91c1130d72022ff3a98c80e6c825e34fc58b04ad0182`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** prototype+0.4106区间跨0且低于average logistic9.5189；删真21,437。参考role监督的PSD度量本版不能说明比完整简单头更好，源目标/低维求解检查不能替代Q完整分割。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.DR16。
- **未知前提/审查限制:** identity metric/rawNN/LDA controls; source metric can suppress query semantics, no cross-image guarantee
- **特有反例:** reference classdistortion随query变化；假nuisance轴实际是Q语义被metric抹去；class仅1token没有triplets。
- **证据:** [card](cards/decision_risk.json); [batch02](reviews/batch02.json)

- **Reviewed source / scope correction:** 归因限制：PSD初始矩阵仅前min(8,K)坐标，而identity control用全部K；任何未来metric改善需先区分rank截断与监督更新，当前未有自然质量结论。

### measured4 QP04 - 参考校准的区域绝对分布相容性门

- **逻辑/条件性预期:** 绝对kernel MMD(q,F)保留q-q项；两个FG/BG核均值差相同的query区域仍可有不同绝对dF。以完整reference FG合法子集间dF定相容门，同时要求相对FG>BG。这不是MMD差分换名，不保证跨实例组成迁移。
- **预期收益（假设）:** 可删相对密度为FG而整体与FG不相容的纯干扰；也可能删真，不预测涨分。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同窗MMD差分，即kernel分类池化; 同窗KDE池化; 逐token绝对FG相容门再同窗平均，隔离q-q作用; 同窗均值/cov距离，同删预算KDE排序
- **原生4例v4实测:** mIoU 0.0000; Δprototype -28.9843 [-46.1480, -11.8205]; T=0 / 0 / 73,228 / 118,865; receipt arm平均0.351s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** QP_pooled_KDE=24.2521; Δcontrol -24.2521 [-38.4124, -10.0918]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/query_partition.py SHA `1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 四例最终空输出，所有原点TP/FP均删去，mIoU0。候选absolute foreground explanation本版没有形成合法正决策；空输出是实测大退化，不能拿区域组织/求解完成冒充识别成功。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.QP04。
- **未知前提/审查限制:** same-window relative/KDE and pointwise absolute gate; true target composition change negative; heuristic tolerance not confidence bound
- **特有反例:** 视角和部位比例变了，真目标可被门拒绝；参考不完整或小窗容差大时不能删。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### measured4 QP06 - 完整查询的四叉树最短描述分割

- **逻辑/条件性预期:** 树DP共同决定所有分区与标签，允许多实例；范围先验惩罚分区节点编码复杂度，区别局部边界惩罚。身份仍由reference unary。
- **预期收益（假设）:** 净参考证据明确的连贯块可补弱token/删碎片，多实例不强迫单连通；真实收益未知。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** samepointwise KDE; same树最细叶无编码代价; 同窗4×4平均KDE; QP05二元Potts
- **原生4例v4实测:** mIoU 19.3954; Δprototype -9.5888 [-16.2802, -2.8975]; T=116 / 61,129 / 10,243 / 2,324; receipt arm平均0.900s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** QP_pointwise_KDE=19.6494; Δcontrol -0.2540 [-1.0999, +0.5919]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/query_partition.py SHA `1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 补真116、补假61,129，完整结果19.3954，低于pointwise KDE且大幅低于prototype。四叉树DP的范围/描述长度组织本版没有纠正已有核unary错判；完整分区求解不保证正区域是目标。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.QP06。
- **未知前提/审查限制:** sameunary leaf/window/Potts controls; hierarchy prior and thin-target deletion explicit
- **特有反例:** 细长斜边需要许多叶块，编码代价可吞目标；错unary连贯块会整体错选。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### measured4 QP07 - 查询中心候选超平面的参考风险分割

- **逻辑/条件性预期:** 所有query字典中心差方向是query内合法分离候选；完整R平衡风险选方向/符号，一般不是reference均值差同一排序。
- **预期收益（假设）:** query可分而reference均值多模式抵消时可能身份改判；无迁移保证、不预测分数。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** FG-BG prototype; QP02离散字典命名; reference-only pair方向风险选择（最多64同预算）; pointwise KDE
- **原生4例v4实测:** mIoU 18.4263; Δprototype -10.5579 [-19.9324, -1.1835]; T=2,497 / 207,996 / 15 / 26,482; receipt arm平均0.547s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** QP_fine_inverse_vote=33.3834; Δcontrol -14.9571 [-32.6702, +2.7560]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/query_partition.py SHA `1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 补真2,497、补假207,996，低于fine inverse vote 14.9571。R风险选择的query中心差超平面能够扩张，却未建立新增集合身份纯度；选中某个合法方向与query正确类别分界是不同断言，不能只报召回。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.QP07。
- **未知前提/审查限制:** unit center midpoint offset cancels; candidate source risk not query guarantee; reference-pair/KDE/whole-dictionary controls
- **特有反例:** 目标模式环绕背景不能由一个过原点平面分开；跨图标签距离逆转可选错方向；R风险选择虚高。
- **证据:** [card](cards/query_partition_batch2.json); [batch02](reviews/batch02.json)

### measured4 QP08 - 查询地标响应上的完整参考条件决策树

- **逻辑/条件性预期:** query球面字典的响应向量s(x)=x·Cq提供<=8维共享坐标；以完整R的平衡FG/BG覆盖在这些轴上贪心最小化Gini可得到非线性条件身份规则。它可在多模式环绕BG构造中改变单平面判决，是真正不同推断；query地标是否比reference地标有益须对照，不默认。
- **预期收益（假设）:** 多模式R/Q支持合法非线性分离时可完整恢复两侧FG；不预测真实涨分。
- **source/合法输入:** [source](../../../src/ics/cpu100/query_partition.py) SHA 1528f46486309a68; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** 同Q响应FG/BG均值线性判别; same CART但reference-only同预算球面字典; same Q响应单层stump; pointwise KDE、QP02及QP07同readout
- **原生4例v4实测:** mIoU 15.7317; Δprototype -13.2526 [-35.2268, +8.7216]; T=11,539 / 155,619 / 47,935 / 62,774; receipt arm平均0.566s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** QP08_stump=23.7728; Δcontrol -8.0412 [-20.4517, +4.3694]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/query_partition.py SHA `1528f46486309a68d4707e57fcbfbd0993bc0127f393871492b70ab2e46f5b40`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 补假155,619且删真47,935，低于简单stump 8.0412。源训练的非线性规则不能由源拟合外推Q；当前反例是本定义与本四例，不是所有核树均不可能。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.QP08。
- **未知前提/审查限制:** cross-slot ref_conditional_tree and sameCART reference-landmark controls; no further landmark/basis variants counted
- **特有反例:** reference少样本轴阈值偶然分开两类而Q关系逆转则整块错；类条件响应强域偏移；query未出现的FG模式会被背景吸收。
- **证据:** [card](cards/query_partition_batch3.json); [batch02](reviews/batch02.json)

### measured4 ref_conditional_tree - 参考角色的条件阈值树

- **逻辑/条件性预期:** A depth3 binary tree implements conditional, multi-threshold axis regions that a single stump, prototype or fixed three-coordinate median histogram cannot always express. A complete unit-feature conditional-boundary fixture is the first real evidence; no natural transfer benefit is assumed.
- **预期收益（假设）:** hypothesized/synthetic_witness pending: recover FG modes needing conditional source boundaries without forcing an EM query seed. Natural signature transfer unknown; no numerical prediction.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Common raw FG cosine/prototype; same-selected-channel depth1 stump; first-batch three-channel code with same render; reference degree2 readout if available. Source copy fit alone is insufficient: evaluate different query feature values governed by the same fixed partition.
- **原生4例v4实测:** mIoU 19.6910; Δprototype -9.2933 [-17.4548, -1.1317]; T=1,429 / 81,470 / 7,175 / 54,412; receipt arm平均0.275s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** ref_tree_stump_control=16.9384; Δcontrol +2.7525 [-0.2161, +5.7212]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/reference_evidence.py SHA `5607a1edf58c26c96d1a93f449e965f56c36cf3a114bbff5ffb27bf05208c36a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 本版比prototype低9.2933，补假81,470、删真7,175。树比stump好不能消除相对原点损失；条件分区增加参考表达力未建立Q身份迁移。此为解释范围，未逐例证明唯一因果。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.ref_conditional_tree。
- **未知前提/审查限制:** same-selected source/branch budget stump/code and held-out feature values; source fit not domain transfer
- **特有反例:** Cross-image coordinate warp crosses learned cuts. A diagonal or curved decision boundary requiring more than eight rectangles cannot be represented at fixed depth. Rare intra-class modes can be omitted by fixed spatial sampling; stratifying known roles prevents dropping a tiny known FG class entirely but does not guarantee its modes survive.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### measured4 ref_role_support_box - 角色轴向支持盒的最大违约判别

- **逻辑/条件性预期:** A class coordinate box admits independent combinations and measures worst range violation. It differs from convex mixtures, average prototype distance and a metric density. A box can include a target outside the source convex hull, but can also admit a false impossible combination. Both are decisive constructed cases, not semantics guarantees.
- **预期收益（假设）:** hypothesized: reject a high-mean-similarity distractor with one class-specific impossible coordinate and tolerate independent within-role variation. Worst-coordinate nuisance shift may instead delete true targets. No predicted mIoU.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine/prototype; same intervals with mean violation and squared Euclidean box distance; exact convex-hull distance on the same small witness. Quantile choices and aggregation controls count0.
- **原生4例v4实测:** mIoU 24.4195; Δprototype -4.5647 [-16.0948, +6.9654]; T=28,050 / 353,662 / 46,756 / 73,589; receipt arm平均0.980s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** ref_box_squared_control=4.5943; Δcontrol +19.8252 [+3.5635, +36.0870]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/reference_evidence.py SHA `5607a1edf58c26c96d1a93f449e965f56c36cf3a114bbff5ffb27bf05208c36a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 补真28,050同时补假353,662、删真46,756；参考逐坐标支持盒在本四例不构成可靠跨图范围。比很弱的box控制好也不等于比完整原点好；不能用宽盒回补量宣称收益。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.ref_role_support_box。
- **未知前提/审查限制:** same intervals mean/squared-box controls, impossible combinations and broad-BG/one-coordinate-nuisance negatives;5/95 variants0
- **特有反例:** One coordinate carrying image-specific nuisance can dominate all other correct coordinates. Independent boxes include feature combinations never produced by the target. Different rotation changes the support set although pair distances are identical.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### measured4 ref_distribution_energy - 无带宽的类分布能量评分

- **逻辑/条件性预期:** For a class distribution P define S(P,q)=E||q-X||-.5E||X-X'||. Algebraically its expected advantage over Q when q~P is half the energy distance, nonnegative. This establishes proper distribution scoring under the same distribution assumption, not pointwise classification, cross-image transfer or probability calibration. Squared-distance version collapses exactly to mean distance and is a necessary control.
- **预期收益（假设）:** algebraic under source/query distribution identity: account for multimodal class support rather than one normalized mean or one closest point. Actual identity drift can invalidate the proper-score expectation. Natural segmentation gain unknown.
- **source/合法输入:** [source](../../../src/ics/cpu100/reference_evidence.py) SHA 5607a1edf58c26c9; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N
- **强control:** Raw FG cosine/prototype, squared-energy score (must equal unnormalized centroid squared distance to numerical tolerance), unsquared mean distance with self-dispersion term removed, and original0.07 fixed RBF density on exactly same selected rows.
- **原生4例v4实测:** mIoU 22.9034; Δprototype -6.0809 [-24.1181, +11.9564]; T=0 / 0 / 21,862 / 65,831; receipt arm平均0.401s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** ref_energy_centroid_control=31.3881; Δcontrol -8.4848 [-23.3374, +6.3679]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/reference_evidence.py SHA `5607a1edf58c26c96d1a93f449e965f56c36cf3a114bbff5ffb27bf05208c36a`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 纯删账删真21,862、删假65,831，完整结果低于同输入centroid 8.4847。分布形状项本版没有超过简单均值解释；能量/数值性质不证明类别判决。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.ref_distribution_energy。
- **未知前提/审查限制:** squared-collapse algebra, sameRBF and no-selfterm controls; proper expected score not pointwise identity or mIoU guarantee
- **特有反例:** P can be far broader than the query's true target subset; subtracting its dispersion may make a broad clutter role too attractive. Proper expectation does not guarantee each individual q is identified. Image style drift changes the class distributions.
- **证据:** [card](cards/reference_evidence_batch2.json); [batch02](reviews/batch02.json)

### measured4 RGB06 - 参考监督的跨模态边界似然切分

- **逻辑/条件性预期:** Known R mask labels edge pairs as same-class or cut. Joint bins of relative DINO contrast and relative RGB contrast retain their dependency, unlike generic Q edge smoothing. Only positive same/cut log-likelihood becomes attractive Potts strength; identity still comes from direct DINO unary.
- **预期收益（假设）:** synthetic hypothesis: close a weak DINO hole while retaining true boundary when RGB-only contrasts are identical for texture and object edge but DINO/RGB dependency differs.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Existing reference_boundary DINO-only calibrated cut and RGB Potts with identical unary, graph, readout and pairwise mass; no old pipeline mismatch.
- **原生4例v4实测:** mIoU 29.3693; Δprototype +0.3851 [-0.1027, +0.8729]; T=1,406 / 4,413 / 246 / 5,215; receipt arm平均0.431s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** RGB06.dino=29.4816; Δcontrol -0.1122 [-0.2244, +0.0000]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/rgb_complement_extra.py SHA `947abb93258a5ab7d02202dd0de6640103c91ea600355bab094e88e59e03e3e7`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 相对prototype+0.3851区间跨0，joint比DINO-only cut低0.1122，pairedCI[−0.2244,0]。本四例不能把cut/renderer回收归给联合RGB/DINO依赖；相同mass/zero等控制已完整，保留真正joint构造但不宣称自然有效。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.RGB06。
- **未知前提/审查限制:** actually verify source/query bins with tiny-label DINO component, singlemodality same-mass controls and complete masks; no verbal bin-invariance assumption
- **特有反例:** Reference has too few pure cut edges, Q boundary appearance changes, or wrong DINO identity seeds are strengthened; boundary fitting cannot identify an entirely wrong object.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### measured4 RGB07 - 逐像素阴影商空间颜色判据

- **逻辑/条件性预期:** Algebraic invariant u(s(x)*RGB)=u(RGB) for positive non-clipped scalar s(x), where u(c)=c/||c||. This removes only local brightness magnitude, preserving chromatic direction. Direct DINO identity remains the baseline.
- **预期收益（假设）:** algebraic invariance; hypothesized recovery of chromatically distinct target under spatially varying achromatic shadow, not semantic guarantee.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** Raw and globally aligned RGB prototypes with same class support and readout; primary itself is the simple cosine RGB invariant, no elaborate log-color operator.
- **原生4例v4实测:** mIoU 28.9289; Δprototype -0.0553 [-0.2907, +0.1800]; T=73 / 10,909 / 63 / 495; receipt arm平均0.301s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** RGB07.global_affine=17.5579; Δcontrol +11.3710 [+1.5385, +21.2034]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/rgb_complement_extra.py SHA `947abb93258a5ab7d02202dd0de6640103c91ea600355bab094e88e59e03e3e7`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 比raw/global-affine颜色臂大幅好，完整结果却相对prototype−0.0553；补真73、补假10,909。quotient在本版主要减少坏颜色证据的损伤，不能把相对坏RGB控制的11.371当DINO增益。理论不变性仍限记录encoder-view/nonclipped条件。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.RGB07。
- **未知前提/审查限制:** invariance only for observed nonclipped scalar shade/constant-color patches, resize/quantization scope explicit; achromatic semantic information loss negative
- **特有反例:** Achromatic FG/BG, colored illumination, clipped channels or same chromaticity distractor; eliminating intensity can remove class evidence.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### measured4 RGB09 - 参考边界两侧的方向性角色判据

- **逻辑/条件性预期:** Known reference cut edges are ordered FG to BG; their unit RGB difference direction provides a foreground-side role. Query edges can then vote toward one endpoint independently of DINO seeds. This is a learned relational contrast, not an assumed bright-target prior.
- **预期收益（假设）:** hypothesized recovery of a whole visually coherent missed region using a reference-conditioned foreground-side relation, without a correct DINO seed.
- **source/合法输入:** [source](../../../src/ics/cpu100/rgb_complement_extra.py) SHA 947abb93258a5ab7; 当前源码可能在旧快照后追加；历史分数仅绑定该冻结运行，不静默迁移。; N + actual RGB/complete R mask
- **强control:** The same learned polarity applied pointwise, plus symmetric boundary segmentation, to isolate role and component propagation; an oracle component identity is forbidden.
- **原生4例v4实测:** mIoU 25.9948; Δprototype -2.9895 [-7.5911, +1.6121]; T=4,863 / 108,878 / 9,055 / 34,287; receipt arm平均0.308s（不含DINO生产，不是端到端延迟）。
- **关键已完整control:** RGB09.symmetric_Potts=28.6471; Δcontrol -2.6523 [-8.1572, +2.8525]。这是指定比较，非遍历所有跨模块control后的因果归属。
- **冻结source:** src/ics/cpu100/rgb_complement_extra.py SHA `947abb93258a5ab7d02202dd0de6640103c91ea600355bab094e88e59e03e3e7`；本批输入/config/scorer身份见下述原report，当前文件增补不迁移该分数。
- **200/600:** 本项未见新原生200/600评分；unknown，不能由4例外推。
- **失败/所得（解释与观察分开）:** 比pointwise role低0.6914、比symmetric Potts低2.6523；补真4,863、补假108,878并删真9,055。本例有向边界角色没有稳定纠正DINO，组件/同关系对比仍可能传错身份；无逐图新视觉解释。
- **逐类四动作/I-U:** [v4 report](/Users/yang/projects/CVPR2027/evidence/local/cpu100_20261006/server/batch02_native4_v4/batch02_native4_v4/score/report.json) -> class_actions_vs_dino_prototype.RGB09。
- **未知前提/审查限制:** pointwise same-role control, reversed context and wrong component negatives; components are not class proof
- **特有反例:** Q surrounding context reverses contrast, internal texture partitions object or merges target/background, one target borders several incompatible colors, same relational contrast distractor.
- **证据:** [card](cards/rgb_complement.json); [batch02](reviews/batch02.json)
- **Synthetic only:** [fixed constructive/negative full masks](reports/rgb_complement/checks_extra.json); no natural-quality claim.

### closed local_004

local_004固定读出已关闭，独立计0；辅助源码保留但未注册METHODS，不是ready。查询branch pruning只匹配了子集，whole-node奖励与FG读出却按整节点面积，把未匹配Q背景也标FG。完整构造IoU0.428571，低于centroid0.514286及leafbag0.5；rewired同0.428571。它反驳当前读出scope，不证明所有hierarchy不可能。强制全Q覆盖会改alignment合同，只能是同家族修订计0，未自动开始。

[closure receipt](reports/local_structure/local_004_closure.json) ; [batch03](reviews/batch03.json).


### local_002 v4同方法修订实测（独立计0）

原生4例mIoU 28.7410; Δprototype -0.2432 [-6.0890, +5.6026]; T=55 / 15,847 / 19,416 / 47,519。local_002.unary_control=29.4784; Δcontrol -0.7374 [-1.5232, +0.0484]。冻结module SHA `22ac1e36e634b9bcc12431e42729e0f5719bce661bbba9945cd0a40cae889c65`。本次为首批local002同方法修订，独立计0；prototype−0.2432、低于同unary0.7374，CI跨0。坐标strain本四例未显示增量；v4能量/温度已改，不能静默替换v0数值或拼作新方法。 逐类账与修订输入配置均在本批report；首30的v0仍保留原冻结版本分数。

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

CPU100历史记录首30、二批22新增、旧19及第三批6资格，合计潜在77，已被Astra300当前映射任务接替。local_004单列关闭计0；local_002修订计0；source/static/toy/完整真实输出/稳定正收益严格分开。未完成100。

200例Huber vs ridge +0.7109 [0.1353,1.0014]是已测真实开发边际；不能抹掉，也不能把相对prototype的+9.7928全部归给Huber。ridge=51.6625，平均logistic=50.3467，常量erosion=50.7178都是强简单controls。

尚无独立确认、同协议完整FoRIS比较、稳定>=2 pp或完整论文方法交付。新52.3734不能与旧processed60.70/62.93跨协议判优劣。负结果不证明信息用尽/DINO上限已定，也不授权新变体。

更新只跟随新封存结果修订相应方法的观察、control、T和解释范围。本文件不授权SSH/实验，也不是新pending-work清单。

本次同步Astra300固定源定义/实际到件、旧1200可读性、真实encoder首view以及二批4例主评分。Astra固定600尚无质量结果，未新增方法、文件或实验；当前300交付状态与CPU100历史资格分开。

## 当前A共享回退合同修复（2026-10-07）

原共享helper把所有score减参考t_R，导致原文指定B0回退也被改阈值。root已按实际cmdline暂停A首批与A006修复两个进程组；这不是自然质量负例，也不能把此前pass或toy记录当完整合同已经成立。回退的参考OOF与query两端均应保B0 cut0，校准不能随候选t_R改变fallback点；新版source/checks单独封存，旧source/partial预测保留且未质量评分。E两批继续固定600。C v1小构造probe被owner重跑覆盖，源码快照仍存；等revision2独立报告，不补造旧probe证据。详细依据在../astra300_20261007/reviews/A_shared_B0_threshold_erratum.json及group_076_150/batch01_old_probe_mutation_incident.json。

### 当前交接修正：只通过文件读取批次（2026-10-07）

reference_evidence 是原卡1–75 owner，cross_image_matching 是226–300 owner，沿用名称不代表当前方法分类，也不承担全队调度。进度、完成、READY、普通阻断、接口问题均写各自目录；root/reviewer 主动读取，不索要回执。只有需要立即停止活动实验的实质错误可消息。

- E235/E239/E243 按不可变审查源码直接同一600启动，服务器 `E235_E239_E243_fixed600_v8`，review SHA256 `20b6431545ab06eb56fbe588916ec650d1ba6144807b8c9175b72a6ef793817e`，5workers×2threads；与既有两批合计线程预算30。实际CPU最近约18核，预算不等于实际利用率。新批首例E235主动分支已有完整输出，但整个600未完成，未评分，没有已验证收益。
- A共同B0回退/OOF错误 revision3 已独立审查通过，review SHA256 `6c0c216406e05a046b6f7807c70350aefedb452b530fde53754d300642cb3f3b`。旧9527/11454进程组身份核实后终止，输出保留且未质量评分。新的A001/A003–A008与A002分为两队列，各自完整600等待资源；A002原求解预算保持，慢方法不拖住其他七项交付。修复不计新方法。
- 完整基线来源归档25成员包含8个AppleDouble元数据。实际17个Python源已单独提取并逐文件SHA验证，索引为 `../astra300_20261007/host_source/complete_baseline_python_source_hashes.json`；原归档与全成员哈希保留。这证明代码来源，不是完整基线分割结果。
- 原始基础600已全齐，以上部署均复用同一manifest，不再重提取，也不补1200。

### Pro30优先后的性能事实与工程失败（2026-10-07）

- **完整结果：** Pro M04固定同一原生600，8个单线程worker，600/600成功，封存墙钟149.44秒；分配worker折算1.992秒/例包含批执行开销，arm自身平均1.032秒。来源为 `../pro30_20261007/runtime/M04_fixed600/runtime.json`。这支持分钟级吞吐可达，尚无质量评分或准确率收益结论。
- **为何旧批慢：** E242单例profile为29.137秒，10次相似度计算11.669秒、5次像素拟合16.665秒；同例控制重复准备。C102首例1667.59秒、C1041066.19秒。已停止旧E/C慢批并保留输出；慢实现不是DINO特征缺失或600数据规模必然需要数小时，也不是科学上的方法负结果。精确复用/向量化需真实场与mask等价后才重新调度。
- **root调度错误：** M04对照批遗漏 `--methods none`，实际包含已审核M03/M04，重复运行M04。保留已完成合法输出，后续对照命令明确none。M04主批无control，原self-baseline评分不符合runner合同，已撤掉，不用假对照评分。等待同一600完整对照批作公平比较。
- **监控错误：** OS state Z进程的cmdline为空，旧版先检查cmdline会错误返回identity mismatch，导致封存结果无法进入评分。v3改为先识别Z，只替换监控，未重跑或中断有效实验。
- **未解决：** 完整INSID3 bilinear固定600进程池在250输出后BrokenProcessPool，cgroup没有OOM事件；根因未知。旧输出保留，后续只续缺项、隔离并记录真实故障case与退出信号，不能把软件崩溃解释成方法失败。CPU CRF原扩展出现malloc断言，原版本不可称已验证；不得重复原样放大运行。


### Pro30 M04固定600质量结论与逐类精确账目（2026-10-07）

**M04此固定版关闭。** 同一原生FP32 DINOv3/1024、600个复用开发episode、80类、原尺寸class-summed mIoU：M04 **29.091009**，B_R **46.575235**，差 **−17.484226 pp**，照片关联组描述性配对95%区间 **[−21.401753,−15.104086]**（570组、RandomState0、2000次）。这更新此前“尚无质量评分”的时点状态；此前分钟级吞吐事实保留，速度成立不代表准确率成立。来源：[封存评分](../pro30_20261007/runtime/M04_fixed600/paired_score_report.json)、[质量决定及80类账目](../pro30_20261007/runtime/M04_fixed600/quality_decision.json)。不是新独立确认，不与旧processed或native200绝对分数混比。

按Opus研究档案§3.6的逐类价格、并用原卡公共§2.2的**精确**恒等式，先每类汇总四动作，再算 `ΔJ_c=[(aT−dT)−J_c(aF−dF)]/(U_c+aF−dF)`，最后对80类平均。全部四项使用该类最终union作共同分母；因此下面是同一完整mask差的精确算术拆解，不能解释成各模块的因果收益，也不能把独立“只补/只删”反事实直接相加。

| 四动作 | 总像素数（只说明规模） | 精确逐类平均贡献（pp） |
|---|---:|---:|
| 补真目标aT | 5,617,829 | +8.368856 |
| 补假目标aF | 36,921,503 | −26.057704 |
| 删真目标dT | 27,717 | −0.028745 |
| 删假目标dF | 728,417 | +0.233367 |

合并“补”的共同分母贡献 **−17.688848 pp**，“删”贡献 **+0.204622 pp**，精确合为 **−17.484226 pp**。**65/80类补集合不划算，65类最终IoU下降，15类提高**；删在18类划算、6类不划算、56类完全没有删，规模不足以抵消新增假前景。此版的观测主要是**大量错误扩张**，不能套用“多数规则主要抹掉稀有真部位”的故事。以class0为例，基线J=0.737864，补纯度0.222429低于该类盈亏线0.424581；删背景纯度0.918121高于该类0.575419线，仍净−14.717519 pp。class64的补纯度0.047678远低于其0.406813线，净−52.221156 pp。每类使用自己的J；没有拿总体purity或历史37.5%/62.5%替代这80个判断。

**全局配对有一个小效应，完整方法仍失败。** M04对随机配对中位数 **+0.662335 pp [0.485334,0.917188]**、对allpair中位数 **+0.355572 [0.244581,0.527514]**、对最近对手中位数 **+2.553348 [0.421778,4.237908]**。其中对随机控制的逐类精确净差可拆为intersection项+0.026516、假前景价格项+0.635819 pp；净假前景减少1,500,184像素，净intersection增加20,307。它支持这些固定控制下配对略有排他收益，不能推成跨配置或独立数据的通用保证。对更简单且原始均值不依赖配对π的raw_matched_mean **30.285193**，M04却 **−1.194185 pp [−1.440194,−0.965023]**，74/80类更差；其中intersection项+0.087810、假前景价格项−1.281994 pp，新增净假前景3,095,282像素。故“单位化差向量后中位数”的整套读出没有胜过其简单控制。不能仅凭这些边际计数分离M04与控制之间的四类独有编辑，质量JSON明确只报可识别的净差。

按照[M04原卡](../pro30_20261007/source/M04.md)“输B_R不进入路线①下一轮”的失败条件，**关闭此版，不自动改阈值、扩字典或另开变体**。配对费用低/数值正确不保证跨图类别排他性；四动作还不能定位到底是差向量归一化、中位数、对手错误还是迁移错位造成损失。此结果否定的是该完整读出版本，**不是原始DINO特征失效或DINO信息上限**；同批B_R仍为46.575235。

**M03结论单列，归因尚未完成。** M03=**46.079531**，对B_R差 **−0.495704 pp [−3.405414,1.067063]**，目前没有支持超过B_R；区间仍跨零。其自己的同字典independent OMP3、系数角色投票、dense-class-cone和convex-hull控制在此报告中**未跑/未评分**。M04的配对控制不能充当M03的机制控制，因此不能宣称“M03全部控制也失败”，也不能将当前结果写成共享稀疏角色解释已被完整归因否决。本条不新开任何实验。


### Pro30 M06固定600：有向差分负结果与保留的renderer工程（2026-10-07）

**M06方法此固定版关闭；精确renderer工程保留。** 同一原生DINOv3/1024的600个复用开发episode、80类，M06 **46.057272** vs B_R **46.575235**，差 **-0.517963 pp [-0.805102,-0.263999]**。来源：[封存评分](../pro30_20261007/runtime/M06_fixed600/score_report.json)、[80类精确账和关闭决定](../pro30_20261007/runtime/M06_fixed600/quality_decision.json)。原尺寸class-summed口径、570照片关联组/2000次RandomState0描述性配对bootstrap；不当独立确认。

**保留原逻辑与控制。** [M06原卡](../pro30_20261007/source/M06.md)用R完整软mask的边差分监督16维未白化空间的反对称双线性K，预测跨图有向e，再以B_R锚定Poisson读出；区别在新增散度，不在另换Laplacian。纯环流会被积分投影去掉，数值环流/边拟合不等于类别收益。源中的B_R、零e同图平滑、同容量pointwise quadratic、center-neighbor双线性、divergence unary、gradient_e、反转/打乱K在这次评分中均保留，不能只挑较弱行解释改善。

精确按每类自己的`J_c=I_c/U_c`算`ΔJ_c=[(aT−dT)−J_c(aF−dF)]/(U_c+aF−dF)`再平均80类。四项使用该类最终union作共同分母，算术拆解为：

| 四动作 | 总像素数（仅规模） | 逐类精确平均贡献（pp） |
|---|---:|---:|
| 补真aT | 115,163 | +0.446760 |
| 补假aF | 142,322 | -0.236460 |
| 删真dT | 239,891 | -1.420420 |
| 删假dF | 255,870 | +0.692157 |

补净 **+0.210300 pp**，删净 **-0.728263 pp**，相抵后正好 **−0.517963 pp**。54类的补集合划算、26类不划算；删在59类不划算，仅21类划算；最终52类下降、28类上升。主要观测是**删除真目标的代价超过删除背景收益**；不能由总体纯度或历史37.5%/62.5%线替代这80类判断，也不能直接推成整实例误删（此报告是像素账）。

**新增边方向效应没有给完整构造过线。** 对zero_e **+0.068899 pp [−0.021777,+0.152852]**，对同容量pointwise quadratic **+0.011364 [−0.134220,+0.291527]**，都未建立可靠增量；对更简单的divergence unary反而 **−0.476558 [−0.789610,−0.144901]**。对反转K **+0.180399 [+0.007942,+0.363332]**、对打乱K **+0.103702 [+0.011333,+0.203805]**是这些固定弱化控制下的小方向效应，不能取代胜B_R/同容量简单读出的目标，也不作多比较独立确认。

**gradient_e=Cs_B这行与B_R评测mask恒等，四动作全0、分数完全同为46.575235**。代数上`(gammaI+L)z=(gammaI+L)s_B`只返回原场；它不能被当成新语义信号或Poisson方法已有收益。相反，真正双线性e的完整版本产生负净编辑；其具体原因尚不能从四动作证明是纯环流、阴影材质、16维漏语义还是边转移失配。

按原卡需超同图零e/同容量node监督的条件，**关闭此版，不增gamma退回B_R后宣称有效，不自动新变体**。原field/mask逐位一致的blocked renderer是单独成立的速度工程，继续保留；它不会因方法精度失败而失效。本记录否定M06完整读出，不是DINO上限或所有图/边机制不可能。


### Pro30 M18固定600：绝对集合门的负结果与所得（2026-10-07）

**M18绝对集合相容性此固定版关闭。** 实际完整600/600输出、同一原生DINOv3/1024、80类原尺寸class-summed mIoU：M18 **14.172234** vs B_R **46.575235**，差 **-32.403001 pp [-36.394437,-29.752803]**。来源：[封存评分](../pro30_20261007/runtime/M18_fixed600/score_report.json)、[80类精确账和关闭决定](../pro30_20261007/runtime/M18_fixed600/quality_decision.json)。仅在评分报告实际回收后分析方向，未先由分数猜错因；570照片关联组、2000次RandomState0的描述性配对区间，数据仍是复用开发。

**原方案逻辑保留，绝对项确实不同但未产生收益。** [M18原卡](../pro30_20261007/source/M18.md)在共同Ward128连通叶子/合并树上，以最多64个面积代表点计算完整FG/BG MMD，保留Q-Q/R-R/Q-R三项；`v=min(DB-DF,tF-DF)`由参考四组分布给绝对FG相容门，完整树DP输出整节点/叶子并集。相对MMD仅去掉绝对门，此时Q-Q项在DB-DF中代数抵消；它是必要控制。主法没有剪去不相似部位，不把低能量或低DF当类别概率。

按每类真实基线`J_c=I_c/U_c`和同一最终union，用`ΔJ_c=[(aT−dT)−J_c(aF−dF)]/(U_c+aF−dF)`精确记账后平均80类：

| 四动作 | 总像素数（仅规模） | 逐类精确平均贡献（pp） |
|---|---:|---:|
| 补真aT | 154,907 | +0.581602 |
| 补假aF | 2,455,165 | -2.587550 |
| 删真dT | 8,595,968 | -40.587753 |
| 删假dF | 3,561,593 | +10.190699 |

补净 **-2.005947 pp**，删净 **-30.397054 pp**，合为 **−32.403001 pp**。删在**78/80类不划算、仅2类划算**；补在40类不划算、24类划算、16类没有补；最终77类下降、3类提高。主损失是**删除原先正确的目标覆盖**，不是没有制造足够假前景去涨面积。class0删除484,860真目标、23,463背景；删集合背景比例0.046158远低于该类盈亏线0.575419。class64将B_R已覆盖的62,495真像素全部删掉（最终intersection=0），删背景5,578，对该类净−68.580866 pp。这里不能从class-summed像素账推成每张图/整实例都被删光；也不能拿总体pure或固定62.5%线代替每类预算。

**最关键的同结构控制已否定绝对门的完整价值。** relativeMMD=**28.189687**，主法对它 **−14.017453 pp [−16.859913,−11.678730]**；pointwiseKDE=**28.213257**，差 **−14.041023 [−17.513805,−11.396408]**。相比relativeMMD，主法净intersection减少**4,730,713**，净假前景减少**1,284,444**；以控制每类J和主法最终union定价，真覆盖项 **−16.765396 pp**、假前景项 **+2.747943 pp**，精确仍为−14.017453 pp。这只识别净差，不能由BR边际动作伪造二者之间四类独有编辑。

same_count_prototype=**14.376918**，M18差 **−0.204684 pp [−0.439682,+0.157105]**，没有支持相容性比已有同数量原型排序更好；这个实现匹配的是**有效native-token阳性数**，不是严格原尺寸像素数，应保留该限制。mean_variance仅**0.277167**，M18虽高13.895067 pp，但赢一个几乎崩坏的控制不是成功结论。报告没有完整INSID3/FoRIS/同候选MEAN行，不能声称已测过这些差值；现有BR及相对MMD/KDE控制已足够否决此固定方案的推进。

**所得与关闭范围。** 绝对分布项没有因“保留Q-Q”就自动形成类别证据，实际补/删账表现为大量正确覆盖被拒；与单参考容忍度/跨姿态材料分布失配的风险一致，但尚不能用这些计数确认究竟是容忍度、区域平均还是表征迁移导致。按原卡完整主方案需赢简单控制的条件，**关闭此版，不再扫核/带宽、不改为只取最相似部分后继续冒充完整集合相容性**。源码、完整输出、合法参考信息和负结果全部保留；这不是DINO信息上限，也不否定全部集合或图方法。本条不授权新实验或新变体。
