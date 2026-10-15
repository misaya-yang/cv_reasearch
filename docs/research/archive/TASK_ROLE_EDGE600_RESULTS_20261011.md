# 原LVIS600任务角色边：完整配方淘汰

五个完整臂/两frame/6000最终Mask已先封存后GT，直接IU/四编辑/保护/实际预算验证完成，root另168394项整数/class/源角色条件边际/真实密度核验通过。完整原图 **46.206795 < 最强S52.005138**，也低于FoRIS47.779371和同预算46.255442，按预先S+1pp/CLI/完整控门槛淘汰，不扩6200、不改tau/host/seed/context/采样配方，不借消融赢家。主研究目标未完成。

| 完整方法 | 原图 observed600 | CLI observed600 | 原图 official920 |
|---|---:|---:|---:|
| FoRIS | 47.779371 | 47.619569 | 31.160459 |
| 保留的强S | 52.005138 | 51.702409 | 33.916394 |
| 角色边绑定full | 46.206795 | 46.044548 | 30.134866 |
| 同角色条件边际factorized | 45.859739 | 45.696342 | 29.908525 |
| 去Q邻角色 | 43.963719 | 43.836105 | 28.671990 |
| 保角色边际打乱源邻描述 | 45.796769 | 45.630161 | 29.867458 |
| S同实际编辑预算 | 46.255442 | 46.086779 | 30.166592 |

600为原独立历史LVIS zoom600与旧6200的精确身份join，每fold60独立class槽；official920每fold92槽按原policy，包括未观察槽位，两池不能混用。任务/image/Mask/crop/geometry/S旧路径SHA全部冻结。该曝光开发缓存不是独立验证或五域结果。

新判据保留S终点并消费S全部正负邻域。R中心FG/BG各128面积中点，重复id保留multiplicity nF/nB，每选中中心扩合法8邻，每边w=n_role/(128*真实degree)，每role总1。source相邻两点descriptor及邻点软Mask标签绑定。Q每点与8合法邻点共32004有向边，Q邻点角色s是已有效S CLI 16×16 area fraction。既有O24先各token unit，τ=.07固定：

`K1=exp((cos(q_i,r_center)-1)/.07); K2=exp((cos(q_j,r_neighbor)-1)/.07)`

`compat=s_j*p_neighbor+(1-s_j)*(1-p_neighbor)`

`F/B=sum_source_edge w_F/B*K1*K2*compat; h_i=mean_Qneighbor(logF-logB)`

factorized保持同角色centre/neighbor条件边际，分别求和后乘积；noQneighborrole仅移除compat；shuffle只在exact(nF,nB,degree,p_neighbor_integer)组内reverse邻点descriptor，中心/权重/邻点标签及带标签的角色条件descriptor边际SHA前后相同。源物理邻接信息被打乱，未随机seed挑结果。标准KDE/条件依赖/邻接不声称原创，也不假定整图同构、固定R foreground面积或相同物体姿态；固定grid局部角色分布的尺度/位置依赖仍有反例。

每完整field有任一合法Q边非正/非有限density就整臂回S，不epsilon或借其它field。其它情况下原图/CLI独立CPU64直接bilinear field、strict正号新增/负号删除、0留S；Dadd=~S&(B|G|R0)，Ddel=S&~(B&G&R0)，1111/0000逐像素保护。预算按full实际S增/删数，在同全D域按原FoRIS source margin选点，row-major打平。

| 原图完整臂相对S | 补真 | 补假 | 删真 | 删假 |
|---|---:|---:|---:|---:|
| full | 475,299 | 1,696,870 | 27,779 | 86,845 |
| factorized | 524,227 | 1,970,594 | 16,369 | 71,598 |
| noQrole | 637,285 | 2,727,330 | 87,359 | 314,902 |
| shuffled | 554,103 | 2,148,297 | 10,177 | 47,629 |
| samebudget | 479,463 | 1,692,706 | 26,809 | 87,815 |

full原图相对S补475299真/1696870假、删27779真/86845假；122例改善、343下降、135相同。补假远超补真，不能称准确任务迁移。原图full比factorized+.347056pp、比role-preserving shuffle+.410026pp，这两臂的114回退例均同S，确有配对关系带来的局部增量；仍比同预算低.048647pp，不能证明争议域选点优于原FoRIS位置。像素池不是class等权mIoU。

full比去Qrole高2.243076pp，其中共同有效486例在固定600池贡献+.580642pp，回退114例贡献+1.662435pp。后者是full退回强S而noQrole继续弱输出产生的保护效应，不能全归为Q角色语义收益。共同有效例/回退例的固定600贡献精确相加，未换成另一个486池作赢家。CLI同分解为+.579247/+1.629197pp。

新source支持审计只读合法R Mask和同固定source quadrature，不加载raw或重推field：127/600例中所有4096源点有正的BG→FG邻角色质量，128背景中心采样后变为0。113次整field回退由这项采样丢失与某Q邻域Sfraction=1共同决定；余下一次index299在全部4096源点中本来就没有FG→FG，而S有纯FG邻点，也使F density为0。四fields尝试2400，其中full/factorized/shuffled各486有效+114回退，noQrole600有效，共2058有效场/342回退。全部R都有正负角色，这不是R Mask为空或backend失败。

条件证书：若所有非零BG-edge权重都对应p_neighbor=0，则任意Sneighbor=1都有 `sum w_BG*K1*K2*p_neighbor=0`，对任意有限正kernel都成立。完整source BG→FG质量正而sampler后为0，证明保存role边际总量1不保证保存角色条件支持。这是本抽样/表示/读出族的条件损失机制，不是冻结DINO或五域首个不可恢复定理，也不能据此宣称增加BG/修sampler就能领先。该配方整体仍因169.7万FP失败。

index299实际case dev_s1/lvis/4/89：reference全源FG→FG质量0，同category query GT的native FG→FG质量.006640625（query原图32px、canonical100px、native foreground质量.390625）。这是网格位置/尺度/目标形态下role-support不必跨图不变的一个真实反例。只作seal后额外1次GT诊断，没有将QGT或source支持审计反馈到推理、阈值或字段赢家。

原S完整处理源码已定位：合法R Mask确定Z2参考RGB裁剪后编码；真实固定局部Q窗口运行原完整FoRIS；8连接前景区域pool local descriptor后与wholeQ FoRIS前景/背景bank比较，margin>0才接受整区域；最终按覆盖像素strict多数，ties背景。600 S终点包仅有bit masks，未存连续S field；原区域margin/选择收据在其它legacy目录须再核对，不可用FoRIS confidence代替S拒绝强度。本批新判据只取wholeR/Q及S二值邻域，无法保留这些原区域拒绝理由，下一先按原收据定位S得失由观察/区域门控/最终票决哪个环节产生，不再直接把weakwhole64字段当修复依据。

生产main wall 441.230498s / CPU 266.990271s / peakRSS 4.120804GiB，600全MPS32、0实际runtimefailure或CPUfallback。旧R/Q 1200 raw loads/1166unique，container请求18,669,988,053B；物理磁盘IO未知。字段计算含GPU工作/传输/CPU日志共333.653241s，不能叫纯GPUtime；最终双frame17.545412s。MPS driver每例返回后采样最大1.065399GiB，allocated此时0，不是实际峰值/利用率或能耗。

评分wall 26.016108s / CPU 26.845746s / RSS 0.790543GiB。Source支持审计main1.983155s/600合法R读/0QGT/raw/推理；case299额外1QGT。固定首3例各32最前row-major Q点（top row）的CPU64真公式参照，共384field-node比较，最大logratio差6.673e-6、strictsign差0；不是全600/全图FP64逐像素一致证明。编译/import和独立constant-center factorization/shuffle恒等式及source条件边际fixture通过，已有signals只读静态review无错误。

6000全部finalpacked包SHA/identity/schema/padding先通过后才评分600QGT；trueGT主打分读600，额外case诊断1，0监督/选阈值/挑字段。root168394项整数/类池/源条件边际/实际density checks及所有保护/预算通过。0新encoder/CRF/训练/照片/view/raw cache/dense archive；原数据/权重/raw/有效S继续保留。原producer PID65539/session68233已成功退出，scorer session95446已成功退出，其PID未记录；无当前运行，历史handle不得重启。

冻结protocol `fa29864180394724e8feaf8761613c2120746ba7b4a4d59d94ffc72444173624` / core `f5a950a8972926c0652ceb4017cc3ed265cddcf68b5414d038ab0f7f4c5d2efc`；seal `b12cc44e4db343a97f17c354f54af5cff966c73b8c3fba1aeb1f9813b77e7b0d`。

[core](../../../evidence/local/task_role_edge600_20261011/core.py)、[producer](../../../evidence/local/task_role_edge600_20261011/producer.py)、[postseal scorer](../../../evidence/local/task_role_edge600_20261011/score_and_verify.py)、[root verifier](../../evidence/local/task_role_edge600_20261011/root_verify.py)、[冻结协议](../../evidence/local/task_role_edge600_20261011/protocol.frozen.json)、[逐例IU/gross](../../evidence/local/task_role_edge600_20261011/episodes.jsonl)、[支持审计](../../evidence/local/task_role_edge600_20261011/reference_conditional_support.json)、[回退归因](../../evidence/local/task_role_edge600_20261011/availability_attribution.json)、[原S处理来源](../../evidence/local/task_role_edge600_20261011/S_processing_provenance.json)、[完成/资产SHA](../../evidence/local/task_role_edge600_20261011/completion.json)。600预测包由sealed.json逐文件SHA绑定，600小收据由records_manifest.json绑定；postseal_attribution.py保存已执行inline诊断的复现计算，不再重跑已有完成输出。

本轮两个独立只读诊断：[三域共同误报](TASK_SHARED_ERROR60_RESULTS_20261011.md)否定共同query prior为主错误，[有效S否决被反复推翻](STRONG_VETO_REVERSAL600_RESULTS_20261011.md)定位明确错误新增。主研究继续，原graph/NN/ridge/本角色边配方全部关闭，不扫其参数/补BG/换sampler/模糊投票或扩大。
