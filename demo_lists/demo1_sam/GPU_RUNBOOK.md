# Demo 1：跨提示共享 SAM 完整解码短测

本地目录：`/Users/yang/projects/CVPR2027/demo_lists/demo1_sam`。
远端目录：`/root/autodl-tmp/demo1_sam`。
当前 SSH：`ssh -p 57510 root@connect.westc.seetacloud.com`（模式切换后以控制台实际地址为准）。

有卡连接更新：`ssh -p 57510 root@connect.westc.seetacloud.com`。使用 `bash tools/with_data_disk.sh <命令>` 将临时文件、Inductor/Triton/CUDA 编译缓存等放到数据盘的 `runtime/`。

实验执行原则（用户 2026-10-01 明确要求）：优先寻找显存允许且实测吞吐最好的微批次，减少分批和启动开销，充分利用 GPU。公平对照仍使用相同提示集合与相同批次；1/8/32/64/128 组提示的主实验不能通过填充无关张量提高占用。首轮补微批次 64/128，若后续研究需要更高吞吐，再用额外独立提示作明确标注的容量测试。

## 接管状态（2026-10-01）

Owner：对话 `01a0f783-6584-7e60-b6d3-5bf227e7e7bc` 负责本SAM研究。用户已让原对话 `01a0f6ff-c372-78b3-b8cf-b70b3caf23cd` 开展其他新实验，不再把它视为停止状态，也不接管或终止其作业。用户允许显存容纳时并行实验，正式测速另行避免并发干扰。

- 旧 GPU 证据已保存：eager 96项、大批次36项、compile48项、合并投影36项、相位/隐式26项。最新26项全部完成。
- 同为P128/MB128的最新compiled结果：dense phase最快约35.47 ms、factor phase37.20 ms、implicit factor phase33.48 ms。原先大幅收益受到上采样实现差异影响；合并投影+相位头的强dense组合尚需公平实测。
- 预训练SAM ViT-B、固定官方源码、COCO24图/70实例已在数据盘。两图真实验证原5e-5门槛失败；失败状态保留。
- 本轮新增八臂公平对照、真实输入重放和离线CPU评分。单进程复用同一模型与输入，P128/MB8和128，正逆两轮；真实24图生成包含原四方法，新增八臂在相同已编码输入上重放。
- 所有GPU生成均先导出未压缩NPZ，CPU质量评分在GPU阶段结束后执行，评分不改写原数值状态。没有训练、删提示或放宽阈值。
- 准备阶段复用现有依赖，不重新下载SAM权重或COCO。SAM2/HQ/SAM3属于后续质量竞争对照，当前队列不声称SOTA；其资源缺口保留在research/related_work/ASSET_REQUIREMENTS.md。

## 当前研究判断与外部意见复核（2026-10-01）

### 持续研究与当前队列（2026-10-01，最新）

**最新：用户指定GIC真实匹配验证，覆盖全部旧调度。** Own源码/结果在`research/gic_validation`，remote `/root/autodl-tmp/gic_validation`。已独立实现双端像素误差场（一个ρ、逐点边际不变）、一次小矩阵特征分解的SpectralREML；线性/双端22项与几何9项数值检查通过，不等于任务收益。仅下载DMS作者original-RoMa预计算匹配ScanNet/MegaDepth各1500对，共654940016 bytes，没有图像/权重、没有各向异性precision；本轮是双端等方差legacy-RoMa，不是RoMaV2复现。8对单场景接口完成无fallback，GIC AUC5 38.35低于native43.10且ρ全近.95，不能宣称成功或据此调参。固定g5/ρ≤.95/8次LM/原生初始化与inliers/1px阈值，已启动ScanNet场景隔离test与MegaDepth外部全1500对CPU七臂；日志`scannet_frozen_scene_test.log`/`megadepth_frozen_external.log`。比较完整native PoseLib、native额外迭代、diagonal、GIC、保边际shuffle、逆占用与每格一个点；GT仅最终评分。官方DMS完整后端正在准备，不能拿密度消融冒充DMS；RoMaV2双端各向异性尚未验证。只保存小位姿/指标JSON与必要活跃匹配；旧SAM/info/pixel不恢复、不重建19GB缓存、不干涉他人作业。


**用户最新纠偏：验证器路线已停止并清理（覆盖全部旧实验调度）**：用户明确批评低于强基线还继续补诊断、缓存占满磁盘。已验证PID/cwd/script并停止own info_train154761、CPU followup155746及其子进程；不动其他会话。六模型只完成R0/R1的seed2042：replace51.1819/53.1708，均弱于同300的native54.9469、FoRIS1024+CRF59.5997和旧scalar61.0420；R2/seed2中途停止，不能写成完整否定实验。正确support对照/FG-BG交换/异类support为51.1819/27.3238/31.5552，说明用了support但不证明迁移有效。冻结affinity诊断native full51.7064、native debiased55.9477仍低于强控制，不扩大该构造，不以弱基线增益包装方法。所有info/pixel/SAM旧路线、修补与缓存重抽都禁止恢复。

实际清理完成：删除own scope2549个npz/pt/npy等，20246274082 bytes；data盘实际free6426873856→26679697408（6.0→24.8GiB）。先存2625个JSON/log/source的压缩证据2879702 bytes并本地校验，另存原1200计数149217 bytes；本地`demo8_local_verification/cleanup/cleanup_receipt_20261002.json`为凭据。原DINO/模型/数据、另一会话demo9 stream缓存/源码/作业及冻结未开800manifest保护，未下载/关机/消息其他会话。information_v1状态RETIRED_ASSETS_REMOVED、information_readout状态RETIRED_BY_USER_DIRECTION，禁止自动--resume或重造19GB缓存。

整体任务仍继续：用户截图对应`demo9_transductive_ics/README.md`已只读核查，N16四折平均57.9→64.2、有合法新增无标注同类图像；该项目由另一会话执行，不接管或重复其N32/48、多种子、混负例、FoRIS接续。截图数字是这个协议的初步正信号，不提前叫SOTA或论文成功。后续独立工作须从重要任务、顶尖主文献和新增可用证据出发；当前尚未选定/启动新的own GPU方法实验，不能把盘点或旧路微调当进展。自动任务保持ACTIVE但已整体改写为这一范围，旧调度不再有效。


**第一已冻结R0结果（单seed/单held折，非总判断）**：epoch5由旧内部val选定、训练至50正常早停。train replace72.6598/add65.5828，held fold0 replace51.1819/add55.3840；明显训练可拟合而类别迁移弱，不用训练分数或内部val67.18充当成功，也不据此称最终表征没有信息。原300对照native54.9469、FoRIS1024+CRF59.5997、旧scalar59.9813/61.0420，R0未胜强控制。主管154761已无缝进入R1，健康不重启。为了核查这个reader是否真在使用给定support，新增一次冻结R0的输入干预diagnostic `info_conditioning.py`：相同query/候选/checkpoint，比较正确support、交换FG/BG bank、另一held类support；queryGT仅评分，donor由class元数据预定，无重新训练/新模型/新数据。不把错误support输出当正确任务成绩，也不把敏感性当信息损失证明；先验证实验实际触发support条件。完整同seed R1/R2及第二seed仍未完成，800新图仍未开。

**本轮强对照合同补齐**：`information_readout_v1/matched_controls.json`已CPU复核当前held fold0同300 episodes/原mask及照片类别逐项对齐：native54.9469、FoRIS512+CRF59.2882、FoRIS1024+CRF59.5997、旧scalar两seed59.9813/61.0420（replace）。因此不能拿新fold0输出与全1200的61.3499直接比，也不能省略比FoRIS更强的旧scalar。后者与新reader种子/容量不同，属于实用强参照，不作纯信息归因。已把两旧scalar的replace/add加入自动配对分析；R1/R2对相同R0才是本轮纯信息干预的核心比较。健康训练不重启，当前R0约epoch30，尚无held-fold最终结果。

**最新实际状态**：information_v1 1200/1200提取COMPLETED，1431秒；训练主管154761已自动进入R0/seed2042 GPU实际梯度训练，首epoch val只是未训练充分的过程指标，不作结论。配对分析主管155746等待六模型结果；没有重复启动或打断健康作业。输入接口三臂参数均156035，原final重放差0，FP16舍入诊断保留。现data盘6GB可用（活跃fulltoken约19GB）；未来判断后依授权删除。当前RAM判定时未满足原预留，因此本worker流式读取；不为普通吞吐优化重启健康实验。状态初版直到首模型完成才更新，已据progress和实际CUDA纠正为TRAINING，后续源码修复该显示延迟。候选oracle只证明覆盖，不证明模型语义知道目标；同reader中间层优势也不能区分信息不可逆丢失与非线性可读性改善。

**当前实际接续（同一新实验）**：提取PID152996、CPU等待训练主管154761（原extract完成后自动进入GPU，fold0×R0/R1/R2×seed2042/2043）、配对分析主管155746。三臂实际forward/backward检查通过，均156035参数；`information_v1/training_interface.json`只是接口检查，不是质量结果。旧pixel80模型已全COMPLETED，best global-paired joint max58.7166，仍弱于同1200 FoRIS1024+CRF61.3499；不继续扩这条摘要输入构造。新训练保留原始全向量并尽量在cgroup预算内只加载一次CPU RAM，超预算则stream，不拷贝新磁盘缓存。当前新token提取已1058/1200、约21分钟，data盘剩8.2GB，预计完成约6GB，4GB保护不放宽。完成判断后删除非论文用途大tokencache，不接续旧矩阵。源码实际loss是按训练类频率加权的BCE+Dice+候选expected episode-IoU，最终epoch选择及评分是原尺寸class-mIoU，不能称小batch损失等于全类聚合mIoU。

**2026-10-02 最新用户分析后的单一判别实验（覆盖旧调度）**：用户要求以CVPR oral和当前强方法为竞争标杆，并指出candidate oracle不是可用信息。停止扩展旧SAM/局部MSE/选择规则/foreground-mass/预算证书，不预设semantic compression。摘要readouts64与paired effects已完成，新增dense maps未证明超过旧crop标量的独立收益；FoRIS同1200旧开发episodes完成，512 core59.0291/CRF60.7074，1024 core60.5280/CRF61.3499。源码+timm现有权重adapter/FP32/TF32off/原尺寸合同，是本地强开发控制而非论文精确复现；并发时间不是延迟。旧pixel_verifier_v1 PID147630已60/80，健康收尾后不扩展。

实际新输入提取`demo8_local_verification/info_extract.py` PID152996，remote原目录`/root/autodl-tmp/demo8_local_verification`，看`results/information_v1/status.json/interface.json/error.json`与`information_extract.log`。复用1200旧episodes、同GT-free native+12候选、同1024 RGB及DINOv3-L一次forward block12/block24，保留query全部4096空间tokens/1024通道，不做PCA/摘要；support固定合法128FG+128BG实际tokens，三臂同bank，不能称完整support全token。新final接口对原forward差0。autocast后norm实测FP32，存FP16有舍入，首图middle/final样本cosine最大差.0002855/.00007099，不称无损。最初误设BF16存储的失败与log已保留，修复从完成文件接续。预计约20GB临时cache，4GiB空闲保护，无下载；判断完成后删除无论文用途大features。

接续`info_train.py`：R0=final+final、R1=final+block12、R2=final+原RGB16x16patch（768维零填共同1024输入），同两流support-query交互验证器/精确参数数目、同候选/监督/AdamW/200epoch上限、min50/patience40，BCE+Dice+固定候选expected-IoU，内部原尺寸replace/add class-mIoU选epoch。先fold0两seed2042/2043共6模型，记录训练拟合和held-class、两角色图片隔离表现，不先扩多层/head。800新图cohort仍未开，不能用来选层/规则/epoch。正结果只支持指定reader下额外可用证据，不单独证明不可逆信息损失；负也不证明外部信息必要或整个家族无效。DINOv3附录已有逐层分析，Thinking-Once(arxiv2607.27830)已有HR-VQA中间证据路由，不能仅把中间层命名为创新。整体研究仍未完成。

用户要求及时清理已执行：旧own SAM大结果/照片副本/闲置变体权重/编译cache删除，1755文本源码metrics先压缩18.5MB并复制本地；凭据`cleanup/cleanup_receipt_20261002.json`与`retired_unused_assets_20261002.json`。data盘实际空闲12→25GB；当前DINO/1200依赖用原root盘hardlinks保护，无payload副本，不碰别人资产/作业。demo3 GPU_PAUSED保持。只留小checkpoint、关键结果与必要活跃输入。

**2026-10-02 用户要求当前会话持续执行后的实际接续（最高优先）**：本会话独立`demo8_local_verification`已启动真实GPU提取PID137669，4×300既有开发episodes，复用`/root/demo4_cache/results/l3b_f{0..3}.l3.pt`固定树和既有DINOv3-L，不重新聚类/下载/改他人脚本。当前130/1200健康，GPU100%/约4.9GB，缓存36MB；native全图debiased sim与原缓存最大差.0002444，核查通过。候选是原GT-free F1 top12加native，既评替换候选也评native union候选的遗漏修复；query原尺寸GT只在观测完成后生成训练标签/评分计数。三臂稠密raw/debiased完整支持FG/BG匹配场+候选/base空间mask：global、querycrop对globalsupport、pairedcrop；所有臂同享plain/grey CLS/pool/debiased-pool旧策略标量，区分细粒度信息增量而非只多encode。fixed448 crop/margin.15，global1024，原encoderbf16、matchingFP32TF32off，FP16压缩小maps，无全token大缓存。它不是FoRIS/FROST复现，不称创新/新测试/正式速度。已排CPU等待主管PID138223 `train.py`，提取完成立即加载CUDA并接4fold×4readout×2seed训练：标量MLP与三空间CNN约同容量、同训练标签/机会，任务目标为训练类聚合expectedI/expectedU而非teacherMSE；同数据另有400iter标量boost强控制及六种旧第二眼重排。每fold排除全部test query/support图片，再按图片连通分量分内部train/val，内部val选epoch，outertest旧数据仅作类别+图片隔离开发判断；原尺寸二值掩码共同合同，报告class-mIoU/逐图IoU/补回TP与新增FP，两任务不混用。健康不能重启/重复；ERROR留trace与完成npz，从失败处修复。优先看`/root/autodl-tmp/demo8_local_verification/results/dense_observation_v1/status.json`及`readouts_v1/status.json/report.json`与extract/train日志。未形成局部信息优势前不开发复杂选择/证书；下一步需强基线协议和真正冻结新cohort，不能拿旧数据训练增益直接当发表结论。用户纠偏/询问不构成取消，不因发最终说明而停止当前研究。

**2026-10-02 接收研究决策会话交接（覆盖相关旧推测）**：已读取`research_decision_20261001/HANDOFF.md`并核对selection/verifier/oracle/theory四份JSON。4000例类别+图片隔离的11标量非线性选择器55.75对INSID3 56.06，没有优势；不能继续引用未隔离旧59.9作这套输入的泛化证据。300例crop相似度标量验证器失败：class-mIoU F1 57.22、directIoU49.68、共享前景量38.89，比例MAE.2473；只有标量而无完整crop/token，不排除真正新视觉观测，但停止调这套弱观测器与协调器。1200例四次理想GT前景量观测逐图IoU67.95，对最大面积优先66.98仅+.97；它们是oracle，且目标逐图IoU不同于class-mIoU。四次查询仅1/1200满足ε=.01停止界，不能宣称已有便宜停止或把数学检查当任务成功。区间/可加前景量代码保留作工具，视觉辨识有效之前不扩控制器/证书/理论包装。接收方原授权继续有效；不恢复交接方已停工作、不自动运行其未跑`cpu_relational_probe.py`。大缓存现移到`/root/demo4_cache/results/`，旧路径不可触发重抽；重要JSON已在本地。未发送回信或启动新GPU作业。

**2026-10-02 用户提供方向审查后的判断（最高优先，覆盖下方旧调度）**：停止扩大demo6视频身份和demo5分辨率构造；既有视频控制全部完成，尚无新方法主张。核心问题改为评估“支持条件化的局部信息能否可靠区分遗漏前景与相似背景”，先固定候选与训练/评测合同，检验信息增量，不先开发预算调度。旧`second_look.py`已经同时裁剪支持和查询，因此paired crop本身不能当新机制；区别必须是局部FG/BG及空间证据超过旧CLS/pool重排、同监督/容量/训练充分的简单头。FoRIS作者512 COCO59.5须本协议强对照，FROST是密度比而非KRR，不能只借其名字。已用原seed0 episode生成器CPU核实跨折照片重叠：300/fold测试21/15/19/30个episode至少一角色出现在训练折，1000/fold为174/174/155/181；凭据`exa-results/coco20i-image-overlap-audit-2026-10-02.json`。不推断增益全部由泄漏造成，但旧59.9不能称严格图像独立泛化。方向筛选中小物体收益下界不是排除用上界；单模型bootstrap SD不是paired增益门槛；机制探索不要求事先算出方法实际收益。3–5pp是扩展目标而非首轮探索硬门槛。`p>J/(1+J)`仅在新增区域不与原预测重叠时是IoU改善恒等式，p/J推理时不可用GT，估计误差与最终修复才是检验对象。复用资产、零下载，不恢复SAM，不接管/修改/消息另一会话demo4；先只读核查其当前结果，避免重复实验。当前没有新局部验证器GPU作业，不把核查描述为方法收益。

**2026-10-02 实际跨帧研究接续（最高优先）**：发现现有完整DAVIS约820MB（不是只有首帧），复用现有DINOv3 ViT-L/16/timm权重；未下载/微调模型，SAM研究不恢复，其他项目无改动。独立`demo6_instance_evidence`已完成train40多对象序列的静态九臂、原论文短边480纠正九臂、连续原生特征标签传播两臂（first+7前帧/top5/T.2/radius12；不是作者最佳参数精确复现）。同1249对象—帧/1070可见/176无可见mask/3tiny，连续实际处理2782后续帧（加40首帧共2822 encoder calls）；整体正确point约89.16%，可见性macro AUC .9224。首帧可视核查手工冻结11个同类组，423对象—帧/375可见；连续local/global point88%，同类identity错7.2/7.47%，可见性AUC .9604/.9405、开发ROC90%TPR的FPR .0875/.0964。粗粒度静态长边512只有60.53%同类point；native短边480 ridge76.53%，不能拿弱原型失败立题。仅查询GT评分，首帧mask是给定输入；Ridge是给定首帧的解析拟合控制，不宣称零学习。对象mask消失是不可见/出视野/遮挡，不是物理不存在；不要把parts错误叫同类identity错误。`same_kind_groups.json`与`category_diagnostic_v1.json`保留核查定义，阈值只是描述性ROC，不是部署校准或独立测试。当前没有新方法或论文主张。

当前实际接续PID134353：`results/davis_train_sparse_requests480_v1/status.json/report.json/records.json`、`sparse_requests480.log`。同40train、同固定评测请求/同first mask/同480FP32encoder，仅编码请求帧，三个已知控制request_fixed/global/time（时间半径12sqrt(dt)）。比较连续2822 encoder calls是否必要、真实同类身份/可见性/region IoU代价；请求manifest显式按GT可见性分层，因此不是自然部署请求分布；编码次数下降不是实际延迟证据。健康不重复，失败保留完成阶段。简单global/time若已解决，不包装新颖；若有重要成本—质量缺口，先核查前作与失误可识别性，不能直接调loss/堆更多模板。所有旧v5已停止，两正式报告在本机；不要恢复旧SAM或v5矩阵。当前source/local目录`/Users/yang/projects/CVPR2027/demo_lists/demo6_instance_evidence`，remote `/root/autodl-tmp/demo6_instance_evidence`。只用现有模型和数据，磁盘约12GB；本会话持续执行，heartbeat不替代当前工作的连续接续。

**新机制已被强对照否定，不扩展**：`demo5_resolution_attention/results/dev128_v1`九臂与`low_dev128_v1`十臂均已COMPLETED。same128开发图native512/768/1024 mIoU为53.076/53.136/48.462；1024all-prefix补偿51.505，普通length温度52.078，未形成前缀补偿独立优势。降分辨率native384=50.865/all-prefix384=50.782；native256=48.858/all-prefix256=44.012，配对−4.846pp CI[-7.312,-1.386]；backbone-only256更差41.805。PI-resize32/RGB512强控制48.326，加前缀補偿44.005。PI线性basis恒等残差1.33e-15，原生实现复核差0，故不是未完成/实现失败。只修正attention占比未保证任务改善，停止此构造，保留全部正负报告。GPU阶段共约538+89秒，不继续kernel优化或扩大同矩阵。本机`high_resolution_report.json/low_resolution_report.json`已有副本；其他会话无改动、零下载零训练。下一项仍需重要能力缺口与强控制，正在核查具体对象identity/属性绑定前作；WALDO文中把DINOv3写为ViT-B/14（与官方规格不符）且测试exemplar是未扰动原图crop，不能直接拿其低Success@1当现代DINOv3瓶颈；OH-A-DINO arxivv2只旧DINO，小toy数据不能自动泛化现代真实实例。

**本轮已接续实际新问题检验**：独立目录`demo5_resolution_attention/probe.py`已在`/root/autodl-tmp/demo5_resolution_attention`启动PID124599，128个seed2037 ADE20K旧验证图（仅开发证据），复用现有EoMT-L权重与环境，零下载/零训练，不修改demo2/3/4文件。假设是固定数量query/CLS/register与随分辨率增加的patch混合后注意力质量失衡。九臂包含native512/native768/native1024、1024前缀all/query-only/backbone-only固定log(N/N512)补偿、全局log-length温度、原生512窗口在相同1024RGB上滑窗、多尺度普通均值。GT仅评分；position table统一bicubic插值，原生512与完全原方法qcls/qmask最大差均0。完整原尺寸语义任务评分，保留每图TP/pred/GT和paired image bootstrap；并发时间不作正式速度。ToMe proportional attention及TransNeXt长度缩放是直接前作，log bias不称新颖；DINOv3原文已支持高分辨率，故不把旧head失败当现代backbone普遍缺陷。此128图只决定是否有值得继续的重要任务信号；若原生/滑窗/普通温度追平或收益很小，结束此构造，不延长局部工程。当前无论文级新方法或成功结论。

**用户澄清总体目标继续（最新最高优先）**：结束SAM不是结束研究。整体目标仍是实际任务收益、新颖性和足够支撑发表的证据。本会话自动跟进已恢复ACTIVE并改名“现有资源上的新方法研究”；禁止恢复下方SAM旧路线，禁止下载新模型/权重/数据。新问题尚未选定，没有新的GPU实验在运行，不把盘点当成果。已只读确认demo2区域命名融合也被强像素集成推翻；demo3另有用户GPU暂停授权边界，不接管；demo4是另一会话项目，不修改/重复其研究。下一步在现有资源上选择有重要任务收益空间的可判别机制，先核查强同资源前作再实现受控实验，持续推进，不要求用户代为选题、不用负载填空。

**用户最新资源与方向决定（优先于全部旧条目）**：SAM路线结束；停止SAM3、FluxGraph/TubeletGraph以及旧稀疏head/学习基底/kernel路线的后续实验与资产准备。服务器磁盘约11GB可用，不再下载新模型或数据，也不以CPU/GPU占用代替有价值的研究。`sam`自动任务已通过应用工具改为PAUSED，防止旧heartbeat重新启动；这不关闭服务器、不停止其他会话，也不删除已有重要结果。新问题尚未选定，下一步只能从重要任务失败与现有可用资源出发，不能先下载再找题。此前SAM3登录环境问题撤回，无需提供令牌、账号或权重。已完成的本地FluxGraph/TubeletGraph源码阅读只属前作核查，没有启动其训练、基线推理或权重/视频数据下载。

**研究重置（用户最新明确批评，优先于下方旧队列）**：这一晚的稀疏head、局部MSE/基底、融合kernel迭代没有建立值得发表的主张；停止扩这条主线，不再把局部速度或GPU利用率包装为研究成功。旧结果/唯一小checkpoint/失败证据保留，旧矩阵不重跑。`sam2_fused_head_v1`仅一次编译资源失败（shared memory139776>101376 bytes）；K64修复源码已上传但未启动，用户批评后不继续该工程分支。自动跟进已改为从现代强模型真实任务的重要失败重新选题：先核查强原生控制和主文献，明确能力缺口/机制/任务级可判别效果/否定条件；未形成这些不启动新旧无价值负载，不为旧代数结构保题。已有1–4 GT对象提示的质量与128/1024grid warmdecoder测速尚未证明同真实应用的整体质量—成本收益；所有“发表潜力”判断撤回到未建立。

最新basis新seed2036 256图/706对象和DAVIS首帧验证已全部COMPLETED，未支持整体改善。原boundary同精度phase head损失center/near/box0.0303/0.0817/0.0412点，dynamic0.3245/0.0916/0.0438点，证实上一seed2035 near0.0056只是该cohort表现，不是稳定保持。joint相对fixed dynamic center约下降0.0792点、near下降0.0112点，DAVISnear也更差；不再扩learned basis。下一候选不能随口换模块名：初查[Live Interactive Training for Video Segmentation，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Live_Interactive_Training_for_Video_Segmentation_CVPR_2026_paper.pdf)已直接研究从用户纠错学习、减少未来重复纠错，[SAM2Long，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf)已有不确定性memory tree抑制视频错误积累，不能把这两个概念原样作为新方向。应用scientific-problem-selection判断问题重要性/成功后的改变，并用research-experiment-judgment安排能改变决策的实验。GPU保持开机，不关机/终止/消息其他会话；这不是暂停用户整体研究目标。

当前接续：`sam2_learned_basis_v1`训练/actual compiled旧开发验证均COMPLETED；fixed选35/last135，joint选410/last510，两者都patience100。相同teacher validation objective为0.000334102→0.000318718（约4.6%下降），旧12/35对象10%joint−fixed head中心/near/box为+0.01333/+0.00889/+0.01475点，dynamic为+0.00465/−0.00414/+0.00736，有near退步，仍不是独立质量结论。joint相对上轮原boundary head的near也更差，因此新验证必须保留原boundary控制。已实际启动`sam2_learned_basis_v1/followup/queue_status.json`，新seed2036 COCO256排除全部2584旧图，三冻结模型previous-boundary/fixed/joint同新cohort/同native+phase强控制/actual compiled FP16 decoder与FP32 encoder/0/5/10预算并行质量，随后两basis模型DAVIS30匹配首帧验证。teacher-only epoch在新输出前冻结；新GT不得重选模型epoch/预算。优先监督此队列/报告/log，健康不重启、错误保留已完成worker，从失败阶段接；不要继续重复已完成的objective/formal矩阵。若此表示干预弱，应依据实际gate变化和原政策损失换针对性目标，而非仅继续降局部MSE；若支持，需给新basis真实测速，不从相同张量形状推断速度。

最新：`sam2_boundary_objective_v1/followup`全部COMPLETED。新seed2035 256图/736对象，每模型6624行actual compiled FP16 decoder；10%相对同精度fullphase head中心/near/box损失，原v1为0.06537/0.06374/0.02531点，uniform为0.02704/0.00576/0.01384，boundary为0.02597/0.00560/0.00961。相对v1，boundary head中心提升0.03940点CI[0.01517,0.06572]，near提升0.05814点CI[0.01824,0.09927]，框CI跨0；相对matched coefficient的center/near head CI也正，支持响应目标的归因收益。必须保留原生政策结果：boundary dynamic损失0.05082/0.12434/0.03550点，uniform0.03044/0.23423/0.03722；boundary−v1 dynamic中心−0.01895/near+0.16462/框+0.00803，三CI均跨0，不能仅较小head损失宣布dynamic保持或显著改善。boundary−uniform差异也无正CI证据，暂不把边界权重当已证实增益。DAVIS30首帧/61对象10%boundary相对fullphase head损失0.03346/0.00958/0.01263，dynamic0.03103/0.00992/0.01168点，非无损/视频结果。配对凭据`paired_objective_fresh256.json`保护。

新checkpoint actual排他完整低分辨率decoder计时`checkpoint_perf_formal_v1`COMPLETED，六臂（native/fullphase/v1/coeff/uniform/boundary）、两prompt数/同MB128、正逆24块，每块warmup10/reps30、全4接口。P1024 native两轮251.411/251.831ms，fullphase230.267/230.321，v1 201.397/201.336，uniform201.684/201.815，boundary201.610/201.652：新boundary同精度强phase延迟减少约12.44%，保留原v1速度量级。P128 phase29.136/29.130，uniform25.713/25.530，boundary25.635/27.080，后者两轮下降约12.0%/7.0%有变化，不仅挑最快轮；不能套旧P12811.3%数字。计时排除encoder/promptencoder/static cache/native动态选择/fullres postprocess/model load，不称整流程/全encoderFP16；没有把precision/phase通用收益全归因新头。不重跑这个健康完成矩阵。

已接续`sam2_learned_basis_v1/report.json`PID111832：旧24同前12训练/后12teacher验证，复用456.6MB indexed inputs和33.69MB真实response，两个模型同boundary raw epoch735初始化、seed2036/AdamW .0001 wd.01、min100/max1000/patience100、同boundary objective/sigma。fixed-basis continuation控制对joint learned rank64 basis，center固定、teacher目标选epoch；joint增加32768训练变量（69760→102528），但部署本来已存512x64 basis，所以权重存储/张量形状/操作数不增，尚无新速度或质量结论。无teacher/GT部署访问，仍原0/5/10%原局部phase补回/官方0.98政策/全4输出。主管在formal完成前仅CPU等待，现已实际训练，随后两个actual compiled old12开发质量worker并行。新假设只能用old开发判断；已看过新2035结果不能再称其为该假设独立测试。若支持，应新冻结cohort并给新basis实际测速，不能从尺寸相同推断延迟或用新的GT选epoch。健康不重复启动，失败仅ERROR --resume保留optimizer/RNG与完成模型/阶段。

最新监督：`sam2_local_student_v2_converge/followup`全部COMPLETED。新seed2034 256图/711对象、每模型6399行，matched v2−v1在10%下head中心/near/box为−0.02553/−0.00959/−0.04048百分点，dynamic为−0.02483/−0.02045/−0.02936点。head中心95%image CI[-0.04531,-0.00952]、框[-0.06877,-0.01956]均负；dynamic框[-0.05241,-0.01406]负，中心和near跨0。DAVIS v2−v1 dynamic中心/near/框为−0.01618/+0.00110/−0.01561点。更低teacher coefficient MSE未转化为更好任务；结束纯加长原lr训练的构造，原v1仍是已验证部署参考，v2保留作负归因对照。`paired_v2_vs_v1.json`保留配对凭据，不将两模型各自CI当差异CI。

依据目标偏离任务这一结果，已真实运行`sam2_boundary_objective_v1`PID108494，三臂训练和实际compiled开发验证全部COMPLETED。复用原456.6MB indexed输入/PCA，原seed2027 sampled parent replay147456行动态state全部差0，补充实际native四mask responses和小hyper bank，仅33692275 bytes FLOAT32无损压缩；teacher/GT仅训练/评分，不在部署路径。三臂同69760参数、同收敛v2 raw best初始化、seed2035 AdamW lr.0003/wd.01，min100/max1200/patience100：coefficient MSE控制、四mask logit/hypernorm的uniform MSE、同response MSE权重1+9exp(-|teacher normalized logit|/sigma)，sigma为训练前12图10th percentile0.142734、全局train mean权重归一化。后12各自teacher验证目标选epoch598/746/735，last698/846/835均触发patience；三种objective单位不同，不能直接比较val标量宣称质量。部署仍static768共享/rank64/GELU64/全4mask及0/5/10%原phase补回，无新增teacher访问、GT或参数容量；新checkpoint的真实速度仍待测，不能直接套v1延迟。

同old12开发图35对象、每模型315行actual fullgraph FP16 decoder/同FP32 encoder强native/fullphase对照，10%相对同精度phase head中心/near/框：coefficient −0.08539/−0.09526/−0.07829点；uniform +0.01618/−0.00514/−0.09233；boundary +0.02467/−0.00467/−0.06649。dynamic对应：coefficient −0.13669/−0.08241/+0.06493，uniform −0.02754/−0.00876/+0.05340，boundary −0.02653/−0.01773/+0.08893。响应目标有开发方向信号，但加权对near反而弱于uniform，只有35对象，不能称独立泛化或显著质量提升。保留matched较低LR coefficient控制，避免把optimizer/多训练收益归给响应loss。

已接续`sam2_boundary_objective_v1/followup/queue_status.json`：新seed2035 COCO256排除2328全部历史图，模型及各自teacher验证epoch冻结后，原v1/coefficient/uniform/boundary四模型在同新cohort/同actual compiled decoderFP16/相同native+phase强对照/全部0/5/10预算评估，两个质量worker并行，无任何并发速度主张；随后uniform/boundary两模型DAVIS30首帧外部复核，void排除/无视频memory。普通GPU运行无需请示；健康不重启，错误保留完成worker/日志，从失败阶段续接而非重提取/重训练。下一项先看这一冻结新测试能否支持质量—成本改善；不能用新GT重选epoch/预算。若支持，实测新checkpoint的强基线延迟及完整流程；若弱，检验固定全局PCA是否限制有意义的响应表示，而非机械重跑原teacher coefficient训练。

新颖性补充：[EdgeSAM原文](https://arxiv.org/html/2312.06660v1)已经比较decoder distillation targets并用teacher mask的Dice+BCE、动态提示回路训练，保留原decoder架构。因此response supervision/更重视错误边界本身不构成独立创新；待检验的区别仍是实际替代upsampling的每图static共享、latent rank64收缩和原函数局部补回及其实际质量—成本。[BALR-SAM](https://arxiv.org/html/2509.24204v1)已有medical boundary模块/LoRA和decoder QKV低秩tensor attention；有限检索不证明本方法新颖，也不把该文不同任务/硬件的数字作本机对照。

本轮最新：`pipeline_perf_v1`已COMPLETED_EXCLUSIVE_PIPELINE_TIMING，共96块、两旧开发图/两轮/三臂/四提示量/冷图和缓存图两边界。FP32 actual fullgraph encoder+FP16 decoder，同精度完整phase→student10，P1024 cold-image四个块中位数平均496.734→466.489ms，约6.09%下降；cached-image请求388.383→346.263ms，约10.84%。P128 cold平均161.963→158.695ms，约2.02%，第一图反而约慢0.47%、第二图快4.67%，应报告差异，不能仅两图宣称普遍或显著整流程速度。P1更慢，P32冷图无稳定收益；缩小部署主张到较大批量/编码器已缓存场景。此边界包含原尺寸二值mask和selected IQ下载CPU，排除磁盘/模型加载/编译/CPU点输入解析；仍是FP32 encoder，不是全模型FP16。两图sync隔离static tile+student投影缓存0.12546/0.12454ms，对照phase-only tiles0.05625/0.05690ms。原120–138ms cache读数不能当每图成本，原报告保留。

`policy_priority_dev_v1.json`已完成735行旧开发12图（35对象/3regime/7variants）。同10%预算token0优先x2/x4、coarse距.98<.02 gate没有稳定任务增益：中心略好，但near/box略差，不扩此具体构造。`.02` gate触发中心77.1%/near65.7%/box97.1%，完整token0守卫成本并不稀少；较小`.005`守卫near仍保留原两次选择变化。该机制是actual shared native16 transformer states/hyper+真实廉价头/原局部补回的eager开发对照，参考head仅评测，既非独立测试也非部署速度。保持原政策threshold不改。

为补足原600轮仍未plateau的问题，新`sam2_local_student_v2_converge/report.json`已真实续训完成：PID107118，新目录、同69760参数/相同PCA/训练pairs/seed，恢复epoch600的optimizer/CUDA RNG，max2400/patience100，仅old12 teacher-MSE选epoch；epoch961早停、best861。best validation MSE由v1 epoch598的0.00131552到0.00129815（约1.32%改善），不是误用epoch600的较差读数扩大收益；开发10%head/dynamic任务结果反而略差，不能将MSE下降宣布质量改善。原v1模型/全部正式计时结果保护，pairs/basis软链复用无重提取，best/last/deployment仅小checkpoint。

已接续`sam2_local_student_v2_converge/followup/queue_status.json`主管107317，prepare完成新seed2034 COCO256（排除全部2072旧图）；在teacher-MSE epoch冻结后先v1 actual compiled-decoderFP16子进程107399，再v2同cohort/相同native和fullphase控制、全部0/5/10预算，随后v2 DAVIS30外部首帧复核。新256没有用于epoch/budget选择；这项 matched 归因决定充分优化能否减少实际损失，不把旧384再包装新测试。监督此队列，健康不重跑；ERROR读trace保留完成阶段，训练恢复与新研究续训区分，不能覆盖v1。若新256证实无收益，结束纯延长teacher coefficient MSE构造，转向与掩码边界/选择决策相符的目标或更强表示，不能继续机械加epoch。GPU保持开机，旧正式及prompt_curve均无需重跑。

用户明确要求继续研究效率和质量，直到有足够发表潜力的方法，不能把单轮完成当作研究终止。用户进一步明确“研究不中断”是连续安排有价值的GPU实验。服务器自动队列负责阶段衔接；本对话每5分钟监督检查（automation id `sam`）负责故障修复和在末阶段完成前补充下一项有依据的实验，无状态变化不通知。GPU保持有卡，按剩余显存并行；不终止其他会话实验。不得用空转或机械重跑制造利用率。

最新运行状态：quality队列和三个head oracle检查均完成。真实廉价头训练及不重叠128图验证也已完成；当前`local_student_head_v1/perf_p128/report.json`三臂已成功Inductor编译，等待其他CUDA作业结束后正式排他计时。`takeover_compact_perf_followup.py`接续独立`perf_p128_fourarm`，补入此前最快implicit-factor phase控制；不能只用dense phase当唯一速度对手。编译和未计时检查已在并发阶段准备，正式计时才排他，保持GPU开机并保护其他作业。前一项interior假设是让四个共同probe沿原head候选mask前景射线收缩；固定radius8/8段，所有候选面对同views，不用GT构造。半径1/4/8强控制；box保留原head。原生SAM2动态singlemask（token0 delta .05/threshold .98）对照已核查，不能仅与三候选head比较宣称SOTA。

22:06UTC监督：计时主管PID68063及fourarm接续主管PID72020均健康，前者三臂已编译并等待其他CUDA计算进程，GPU整体100%、约281W。无计时样本，不能拿并发状态给速度结论。未重复启动或终止其他作业。新增`local_student_head_v1/davis30.json`跨域冻结测试完成，30个首帧/61对象，排除void、不含时序记忆：真实10%补回center/near/box IoU分别下降0.1086/0.0242/0.0074百分点，sequence/image CI分别[-0.2337,-0.0251]/[-0.0444,-0.0085]/[-0.0207,+0.0082]。可保留较小质量损失的跨域信号，仍非无损；没有用DAVIS训练或调参。已启动`sam2_patch_check.json`原生SAM2.1-L高res两路skip的局部补回机制验证，旧3开发图，FP32和仅head FP64对照；不是已训练SAM2廉价头，也不是全decoder高精度/速度证据。若局部性成立且计时支持，下一步适配现代模型中每图共享的skip静态贡献。

SAM2局部性检查已完成：旧3开发图、3regime、各提示随机64父格，真实两路highres skip按父格切出2x2/4x4并参加局部计算。9项FP32诊断最大logit误差1.9073e-5；仅head FP64的9项最大误差0，均通过固定1e-10。它支持现代SAM2结构中的局部原函数补回，不能把它写成整个encoder/decoder FP64精确或SAM2廉价模型质量/速度。下一次查看计时队列是否有可判别速度结论；若支持，再训练带image-only static-skip缓存的现代head，输出保持4mask/IQ/objscore接口，避免把SAM1权重直接搬过去。

22:32UTC监督：正式主管68063仍健康但已等待约50分钟，无排他计时样本；GPU实测99%、29.4/32GB被其他三个作业占用。新增`perf_p128_shared_diagnostic/report.json`主管PID78557，四臂dense/implicit/compact5/compact10、8轮循环移位及反序、每臂每块10次的并发交错诊断；仅用于下一步方向筛选，不作为排他速度或论文延迟证据。已真实启动，当前`WAITING_VRAM_FOR_CONCURRENT_COMPILER_PREPARATION`（需至少12GB空闲显存），并不是已经在计算；不重复启动健康作业。正式三臂与fourarm接续保留。后续优先读取此并发诊断和正式结果：若compact无收益，定位真实排序/scatter/局部kernel瓶颈再干预；若有方向信号，可准备SAM2 static-skip廉价头，但正式速度结论仍须排他四臂。磁盘13GB空闲/75%，本轮没有删除唯一或在用资产。

22:55UTC重要进展：上述并发P128四臂已COMPLETED；8轮block中位数的中位数dense/implicit/compact5/compact10为110.36/102.77/98.83/123.58ms。同轮相对implicit的耗时比中位数compact5=0.952（6/8轮更快）、compact10=1.212（仅1/8更快）。仅共享GPU方向筛选，不能当排他延迟、显著速度收益或端到端结果；10%补回尚不能凭质量小损失宣布可部署收益。独立`head_profile_p32`已完成实际Inductor和kernel/eager分阶段trace，定位两次cuDNN局部转置卷积及输出布局开销；并发单次trace不代表稳定成本占比。

按该瓶颈实现`takeover_compact_phase_fallback.py`：同学生、同margin、同5/10%预算，只将选中父格的原head用强dense基线已有相位矩阵+原LN/GELU计算，不再对大量1x1 patch调用转置卷积。`phase_head_profile_p32`已完成六臂/六轮真实Inductor并发头部诊断：dense头/原compact5/原compact10/phase5/phase10 block中位数分别12.91/11.48/13.71/5.59/6.10ms。只支持下一步实现判断，不是P128完整decoder或排他速度。32提示同输入对照5/10%完整廉价头logit差均4.5776e-5、低分辨率pixel flip=0、IQ差=0；2048真实父格仅head FP64原转置卷积对相位patch最大误差5.5511e-16，通过固定1e-10。未改判旧FP32 gate，也未证明整个模型数值等价。

已启动`local_student_head_v1/phase_followup_v1/queue_status.json`主管82926，实际GPU子进程82930正在同128图的冻结质量实现复核（已112图，不能称新独立cohort）：自动接续`phase_perf_p128_shared`六臂P128/MB128完整decoder并发筛选；然后CPU-only等待既有`perf_p128_fourarm`结束，接续`phase_perf_p128_formal`六臂排他正逆两轮，避免多个CUDA等待主管互锁。新完整解码候选仍保留最快implicit控制。已测旧完整decoder自身allocated peak1.95GB/reserved3.17GB，故新增runner的并发准备显存余量设6GB，非机械等整卡空闲。原正式主管和fourarm接续未动，所有旧负结果保留。优先监督此新队列及质量兼容性回执；若新完整成本有方向信号再准备SAM2 static-skip学生，正式速度仍须排他。

实用竞争补充：[PyTorch官方SAM2部署研究](https://pytorch.org/blog/accelerating-generative-ai-segment-anything-2/)已经比较batched prompts/fullgraph/AOTInductor和FP16，并纳入后处理/RLE成本；后续现代模型部署对照需面对这些原生优化路线，不能只胜FP32 eager。本轮固定FP32协议保持，另行设部署精度与质量协议后测，不能借异硬件文章数字当本机速度结果。

最新接续结果：`phase_disjoint128.json`已完成，`phase_quality_compatibility_detail.json`按3204个image/annotation/regime/budget键核对上一版：native reference IoU全部相同；仅2行candidate IoU、BoundaryIoU及pixel flip fraction微变，最大IoU差8.5461e-6（0.0008546百分点）、BoundaryIoU差1.1836e-5，IQ漂移差为0。保留原0/5/10%质量代价，不能把实现一致写成对官方无损。六臂`phase_perf_p128_shared`已完成72块/12轮并发完整低分辨率decoder计时：dense/implicit/compact5/compact10/phase5/phase10 block中位数为115.05/110.70/102.71/123.45/88.69/92.30ms；phase5/10相对同轮implicit耗时比中位数0.804/0.834，均12/12轮更快。是重复并发方向信号，尚非排他发表速度或端到端收益。主管82926现为CPU-only等待既有formal fourarm后接正式phase六臂，不占CUDA，避免互锁。

不能忽略编译敏感性：P128新phase对原cuDNN廉价版本，5%最大logit差6.8665e-5/低分辨率flip0；10%最大差2.04814/flip1.5497e-6/IQ差1.3590e-5。可能由编译浮点漂移及近似同margin排序边界放大，但尚未归因；不声称原5e-5通过或仅用P32小差代表P128。已启动独立`compiled_phase_disjoint128.json`实际GPU PID84626，真实cache transformer+相位补回/Inductor fullgraph dynamicFalse、原native参考，0/5/10%预算，记录固定原IQ选择和candidate自身IQ选择的任务质量/选择变化。它复核同128图的实际编译执行，而非新增独立测试或计时。优先监督该输出/日志，正常编译错误自主修复保留失败；确认任务代价保持后，下一步准备SAM2带每图static-skip贡献缓存的真实学生及现代强部署对照，不再只重复旧head诊断。原生SAM2局部性检查无需重跑；它仍不是已训练SAM2廉价模型。

23:30UTC最新：SAM1实际编译质量验证COMPLETED，3204行，10%补回相对native head中心/near/box IoU下降0.06153/0.05369/0.06994百分点，image CI均负但数值小，所有预算的candidate自身IQ选择变化为0。该实现验证支持任务代价保持；不放宽或改判旧数值门槛。已接续真实SAM2学生，不等待排他队列而停止科学开发。

`sam2_local_student_v1/report.json`已完成真实SAM2.1-L训练/开发诊断：69760参数，旧24按ID排序前12训练/PCA、后12 teacher-MSE选epoch与任务诊断，两个输入分量为最终父格256和高res skip tiles256+512。折叠normalizer后，静态768 affine+bias只在每图投影一次；动态256 contribution每提示求值再GELU/Linear64、直接与原hyper向量收缩。真实局部相位补回加skip1到第一LN前、skip0到最后GELU前；保留4mask/4IQ/masktoken/objscore接口，候选无完整teacher head。新相位kernel的64实际父格仅head FP64检查最大差3.3307e-16；这是新布局验证，不重复此前native locality矩阵或宣称整模型精确。

训练73728/验证73728局部样本，seed2027、AdamW .001、min100/max600/patience80，epoch598由teacher MSE选择；epoch600 train/val MSE=0.0008293/0.0013534，`plateau_before_cap=false`，不能称已收敛。rank64 PCA训练方差保留96.075%，不是质量证据。开发12图10%补回相对原多候选head中心/near/box IoU下降0.07525/0.06179/0.09266百分点；相对官方动态token0政策分别-0.12433/-0.02704/+0.06376百分点（均为开发数据，多项CI跨0，不声称质量提升）。10%时dynamic选择改变1/2/1个对象，0%时中心dynamic下降3.83点，表明必须面对原生政策而非只保留1..3。训练目标无GT质量，benchmark提示仍来源标注，不能称整流程annotation-free。

新`sam2_local_student_v1/followup/queue_status.json`主管88356已自动进入`fresh384`实际GPU子进程88476，后续接DAVIS30。全新seed2033 COCO384排除全部此前1688图，学生/预算在新输出前冻结；DAVIS为既有外部30首帧，排除void、无视频记忆。新测试前的开发调度sanity是10%下各regime/head及dynamic mean损失不超过0.5百分点，已通过；它只控制是否继续此构造，不是发表/等价阈值，没有用新测试作早停。只保存一次静态bank/图+按索引引用的动态256与rank64 targets，FLOAT32无损NPZ为456604016 bytes，base权重复用，小best/last/deployment保留；磁盘13GB空闲/76%。下一步在fresh/DAVIS运行中准备现代native SDPA+原完整phase头强对照/实际student头的完整decoder与部署精度成本，不将SAM1并发速度搬到SAM2，也不把本次训练称已获通用速度。正式SAM1三臂/fourarm/phase六臂CPU接续仍保留，不能启动相互等待的CUDA正式主管。

最新SAM2新测试均已完成：seed2033 COCO384/1092对象/9828行，10%相对原多候选head中心/near/box IoU下降0.04027/0.07487/0.01951百分点；相对官方dynamic token0下降0.17207/0.13159/0.04524点，dynamic选择改变16/13/3次。中心/near的两类CI都负，框CI跨0；不能仅报较小的三候选损失。DAVIS30/61对象/549行，10%原head下降0.04500/0.02288/0.04957点，dynamic下降0.04802/0.02213/0.04835点，dynamic选择改变0；head/dynamic CI均负。支持较小质量代价的跨域实现，不是无损或视频记忆结果。5%和0%的退化更大，保留全部预算；新结果不用于重选epoch/预算。

`sam2_local_student_v1/perf_p128_mb64_shared`七臂已真实Inductor/fullgraph完成98块/14轮，128个真实图像上不同grid正点、共同MB64（两次完整decoder）、原SDPA/完整phase原函数/实际学生，全部4mask/4IQ/masktoken/objscore。FP32 TF32off block中位数native/phase/student5/student10=279.60/214.24/184.13/194.47ms；同轮student5/10对phase耗时比中位数0.855/0.904，均13/14更快。decoder-only FP16的native/phase/student10=100.38/90.37/76.33ms，student10同轮对phase比0.837，14/14更快。这些是并发完整低分辨率decoder方向，不是排他或端到端；precision/通用phase工程的收益不可全部归给学生。所有FP16臂共享FP32 encoder/highres features的半精度副本，绝非整encoder FP16；未计image-static cache成本已单独保存。

实际dispatch trace FP32出现efficient-attention、FP16出现flash-attention forward；只按观测kernel描述。第一microbatch完整phase FP32对native最大logit差3.0518e-5、flip0、IQ/token/obj差0；学生近似max差约39–41、flip约0.48–0.89%，不能宣称原5e-5通过。FP16 native对FP32 max差0.28075、flip4.60e-5、IQ差0.00340、masktoken差0.0731、objscore差0.0103，只有精度诊断，尚不等价于任务代价。新增`fp16_quality_v1/queue_status.json`主管97541、实际GPU97545接续同384及DAVIS的真实编译decoder-only FP16验证；同精度native与完整phase两强任务控制均实际运行，原FP32任务参考也保留，记录相对最强同精度phase的head/dynamic损失。该重复cohort为精度/实现验证，不是新增独立测试；Encoder保持FP32、学生/预算未重拟合，编译fullgraph/dynamicFalse无降级，允许32种shape specialization以覆盖不同对象数/point-box形状，最低空闲5GB依据先前小batch质量worker约2.16GB实际占用。

已启动`sam2_local_student_v1/formal_followup/queue_status.json`CPU-only主管，等待既有SAM1 phase正式接续以及FP16质量queue退出后，才加载`perf_p128_mb128_formal`的CUDA并做七臂统一MB128/128不同提示、正逆两轮warmup10/reps30。每臂前后检查其他CUDA进程，出现并发即标ERROR保留无效样本；不终止其他会话，也不相互等待已加载CUDA的正式主管。下一监督优先FP16任务质量/正式阶段健康；当前还没有SAM2排他速度或完整encoder/postprocess整流程收益。若FP16学生任务损失保持，继续轻encoder/更大提示量及真正冷/热整流程，不重复已完成训练/locality矩阵；native动态政策损失集中在少量选择改变，需要如实保留并在开发数据研究政策敏感性，不在新测试上临时放宽阈值。

最新排他速度与精度任务验证均完成：SAM1 phase正式六臂P128/MB128，implicit两轮33.551/33.529ms；phase5为27.141/27.143，phase10为27.938/27.969，约19.1%/16.7%真实完整低分辨率decoder延迟下降，非encoder/postprocess端到端。原cuDNN compact10仍38.861/38.926ms更慢，改进来自真实局部phase实现，不是更改质量规则。

SAM2 `perf_p128_mb128_formal`七臂COMPLETED_EXCLUSIVE_SAM2_DECODER_TIMING、正逆14块，每臂前后无其他CUDA进程：native FP32 93.208/94.085ms，完整phase FP32 62.268/62.643，student5 FP32 54.277/54.632，student10 FP32 55.714/55.415；native decoder-FP16 31.825/33.008，完整phase FP16 29.837/29.465，student10 FP16 26.488/26.087。同精度最快原函数phase对学生10%，FP32约11.0%、FP16约11.3%延迟减少。通用phase和precision收益分开，不能将93.6→26.3ms全归因本方法。固定128不同真实图像grid点、共同MB128、同FP32 encoder features（FP16臂cast）和完整4mask/IQ/masktoken/objscore；计时不含encoder/promptencoder/cache构建/postprocess/model load，仍没有整流程证据。

`fp16_quality_v1`同384/同DAVIS实际Inductor质量验证已COMPLETED，所有对照与学生无重拟合。10%相对同精度完整phase，COCO384原多候选head损失0.03781/0.07460/0.02028百分点；官方dynamic损失0.17401/0.11372/0.04623点（中心CI负、near跨0、框小负）。DAVIS相对同精度phase head损失0.04469/0.02304/0.05020点，dynamic损失0.04762/0.02210/0.04900点，CI均负。必须保留precision本身的政策敏感性：DAVIS near native FP16对原FP32 dynamic下降0.50221点（CI跨0），学生总下降0.52431点（CI负），大部分来自两次precision诱发选择变化，不应只展示小的同精度附加损失。COCO near native FP16原head变化+0.09936点，学生相对FP32看似+0.02477点，不能将该precision/选择变化归因质量创新。

已启动`sam2_local_student_v1/prompt_curve_v1/queue_status.json`接续有依据的P1/8/32/1024真实不同点提示负载，共同MB=min(P,128)、相同七臂/两轮排他，保留既有P128不重跑。回答静态768投影冷成本何时摊销，以及小提示量是否反而更慢、更大提示量收益能否延续；每图static cache构建时间独立记录，warm decoder不冒充冷/端到端。等待前一formal退出的主管仅CPU；GPU保持开机，不用重复已完成矩阵制造负载。下一监督读取此曲线实际报告/日志及GPU状态；继续实际encoder+postprocess整流程与轻encoder强控制。政策敏感性若要改进，应在旧开发数据设计guard/补回规则，再用新冻结测试，不在已用于提出该假设的新384上重新宣称独立验证。

01:27UTC接续：`prompt_curve_v1`四场景全部完成排他七臂，两轮/真实不同提示/共同MB=min(P,128)。半精度同精度完整phase→student10，P1约2.573→2.870ms（更慢11.6%），P8约2.694→2.911ms（更慢8.1%），P32约6.499→5.396ms（减少17.0%），P1024约230.453→201.211ms（减少12.7%）；原P12829.651→26.287ms保留。FP32 P1024约496.707→440.087ms，减少11.4%。这是warm完整低分辨率decoder曲线；小提示应选原生/完整phase，不能称所有提示量受益。

原curve的FP16 static cache计时约120–138ms，但开始前未同步，紧接CUDA decoder深拷贝/half cast，可能混入模型转换异步工作；不将该数作为每图静态投影冷成本或摊销结论，保留原始报告不改数。已启动`pipeline_perf_v1/report.json`PID105685，共享actual Inductor/fullgraph FP32 encoder、decoder-FP16 native/fullphase/student10三臂、两旧开发图P1/32/128/1024。正式forward/reverse、warmup3/reps10；cold-image含CPU RGB预处理/upload、encoder、promptencoding、feature cast、每图cache、完整decoder/native动态政策、原尺寸插值/threshold及selected binary mask/IQ下载CPU。cached-image请求含promptencoding至回传，排除encoder/cache。模型加载/编译/磁盘读/模型precision转换均排除；预先GPU grid点与标签，不含CPU提示解析。模型-only phase矩阵只缓存一次，两optimized臂公平，fullphase不付student静态投影。isolated image cache前后同步，单独给出真实每图成本。共同encoder编译已完成，实际P1/32整流程计时推进、GPU实测84%/262W；当前未完成矩阵，不提前宣布整流程收益。此实验仍是FP32 encoder+FP16 decoder，不是整个模型FP16。下一监督优先pipeline report/log，健康不重复启动；结果决定是否继续优化或缩小到批量提示部署，轻encoder需独立训练对齐，不能搬L学生冒充通用。

新颖性核查边界（本次主文献）：[PointRend](https://arxiv.org/abs/1912.08193)已用自适应位置选择做精细分割；不能声称首次不确定性驱动空间计算。[MobileSAMv2](https://arxiv.org/abs/2312.09579)已指出SegEvery decoder瓶颈并通过减少无效提示获得收益；本方法保持所有输入提示/4mask接口，差异待证明在固定预训练head的函数蒸馏、按父格复用原精确局部计算及实际成本。[Efficient Track Anything](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Xiong_Efficient_Track_Anything_ICCV_2025_paper.pdf)使用轻量encoder/高效memory，并移除SAM2的高res上采样skip；这说明也要与实用轻模型路线比较，不能将SAM1表示直接宣称现代通用。当前检索未建立发表级新颖性，不用有限搜索声称没有前作。

队列自动衔接train256 → validation128 → 四variants×三种子小head训练 → seed2032全新384图（排除全部历史1304图） → DAVIS30首帧 → 冻结学生评价。模型与规则在新测试前固定；仍需检查学生相对native动态baseline的成对CI，`tools/takeover_compare_native.py`用于对应行与原候选质量逐项核对后计算该对照。新teacher/学生不保证改善，错误和负结果保留，不因GPU空档乱扩展无假设矩阵。新脚本位于本地和执行checkout `runs/takeover_20261001_v1/tools`，结果复用canonical results；本checkout已有后续新增脚本，不再称原始代码快照完全冻结。下一次监督须检查队列实际阶段、日志和进程，提前衔接有依据的后续研究。

最新监督实测：interior_v1的train256/validation128和12个matched ranker均完成，fresh384数据已备好。核查时GPU约31.4/32GB被其他四个实际计算进程占用、利用率100%，本队列以剩余显存至少6GB等待新图质量评测，不终止其他作业。开发128的near：head0.47809、native dynamic0.47313、radius1为0.53658、radius8为0.48966、interior8为0.51050；interior相对半径1尚无优势，必须保留半径控制，不能把所有提升归因前景约束。box原head0.84281、native dynamic0.84406，说明只选1..3的框基线遗漏token0收益。

`tangent_test384.json`已完成固定384图检验：center head0.72040/tangent0.71937（Δ-0.00103，95%CI[-0.01736,0.01465]）；near head0.49914/tangent0.54161（Δ+0.04247，CI[0.02540,0.06068]），同MATH精确四view为0.51108。near响应信号支持继续，但不是精确重放：choice agreement仅74.29%，并无原生快速attention速度证据。下一项服务器队列`tools/takeover_response_distill_queue.py`/`distill_v1/response_distill_v1`接续interior最后阶段：24开发feature-export接口检查、旧256/128训练开发、两方向response教师蒸馏与等架构GT监督/几何消融固定三种子、seed2032 fresh384、DAVIS首帧和冻结学生评估。只覆盖两种点提示；先前384现在是提出此蒸馏假设的开发证据，fresh384测试前不调整规则。新增导出以普通MATH原始forward为候选及feature基础，JVP导数响应也锚定该forward，AD primal漂移单独保存，不虚构数值等价通过。只存compact特征。监督同时检查两个queue status，健康时不重复启动；response队列故障修复后可`--resume`保留失败尝试并跳过完成阶段。

接续实测：interior全队列已COMPLETED，response队列PID44552已自动进入计算，24图接口检查通过，train256导出GPU PID44671。fresh384 head/原生dynamic中心68.709/67.848、near48.176/47.379、box83.349/83.630；interior教师中心71.115、near51.290，radius1 near51.529，未建立前景约束优于简单小半径的证据。三种子latent-distilled中心73.183、near52.366、box83.451；geometry-distilled中心72.308、near53.790、box83.304；latent-supervised中心74.465、near57.913、box83.819。两个distilled在点提示上相对native的每种子image CI均正，但仍弱于GT控制，geometry box反而显著负。DAVIS near latent-distilled68.438，低于largest73.994；不声称通用泛化。新增CPU冻结对照`interior_v1/student_fixed_teacher_test384.json`复用前轮四扰动训练好的head，直接评估同新384特征，检验前景约束教师是否有独立增益，不能从不同测试cohort数字推断该增益。

同图冻结归因对照已完成：`interior_v1/interior_vs_fixed_teacher.json`。新384图上，相对前轮固定四扰动教师训练好的ranker，interior几何near下降1.235百分点，三固定种子平均差的条件image CI[-2.370,-0.074]；latent near下降2.111点，CI[-3.208,-1.019]，三种子均下降。latent中心下降0.680点，CI[-1.246,-0.194]。两个GT监督控制的逐种子逐样本输出差为0，支持该归因对照输入和训练控制一致。CI条件于这三个训练模型，不表示训练种子总体区间。停止扩大此具体前景约束构造，保留实用单次解码收益及全部正负证据，继续已启动的局部响应蒸馏。响应feature exporter补充同encoder/提示的default native SDPA头及dynamic baseline任务质量，避免仅与MATH本身比较；候选/特征仍来自普通MATH原forward，不能偷换成原生优化速度声明。

已启动独立效率机制检查`tools/takeover_head_compression.py`，结果`results/takeover_20261001_v1/head_compress_v1/report.json`：SAM1局部父格子的16个子像素、各32通道合为512维embedding，先用旧24开发图中按image-ID排序的前12拟合全局PCA，后12仅作开发诊断；固定rank16/32/64/128。完整真实head始终计算，再用oracle投影测原head选中mask的GT IoU、BoundaryIoU及全分辨率像素翻转，identity布局重构误差单独记录，不作精确通过或速度声明。只存小basis.pt与metrics，不重复导出full mask/encoded state。此检查回答廉价可训练head是否值得构造；PCA方差高不等于任务质量保留，oracle也不等于部署实现。若低维投影在原任务上已明显退化，收窄该全局线性表示，不能据此否定所有head优化。下一监督同时检查head_compress和两个quality queue，接续有根据的实用干预。

response全队列已COMPLETED。fresh384点提示：head center68.709/near48.176；tangent教师70.660/51.355；三种子geometry-distilled71.401/55.033，latent-distilled73.070/54.112，geometry-supervised72.947/56.085，latent-supervised73.984/57.824。DAVIS near latent-distilled69.908仍弱于largest73.994，center latent-distilled78.508有收益；不宣称新教师已优于固定教师或GT控制。新384 in-flight导出启动早于native字段增补，已用图像/annotation ID/regime及三种已选candidate质量逐项核对的CPU方法补`student_test384_native.json`，无需重跑GPU。

全局head PCA oracle诊断已完成：rank64 center/near IoU分别下降0.955/0.760百分点，flip约0.217%/0.225%；rank128仍有center下降0.507点。这个构造不能直接当作质量保持的廉价头。已启动两个依据新观察的检查：`head_fallback_v1/report.json`按投影mask的归一化logit margin选1/5/10%父格补回原精确4x4 logits，与同预算random强对照，rank32/64，仍为oracle机制检查；`head_residual_v1/report.json`缓存每图一次实际not-a-point padding reference head，只压缩prompt-conditioned head差，而非每次压缩整幅图像内容，rank16/32/64/128。共享reference额外解码成本需后续实测摊销，teacher oracle和像素改善都不代表实际加速。均复用old24开发图、原encoded state和权重，仅保留小basis及metrics。

两个机制检查均完成：共享blank reference后只压缩残差未改善全局PCA的任务损失，rank64 center/near反而下降1.523/0.937点，暂不扩该表示。margin挑5%父格精确补回有实际oracle信号：rank32 center损失1.508点降为0.089点，同预算random仍1.444点；rank64 center损失0.955点降为0.057点。仍不可当作速度结果，因为粗输出用完整teacher head得到。

已开始真实替代头训练`tools/takeover_train_local_head.py`，状态`local_student_head_v1/report.json`：parent最终256维状态经过Linear64/GELU/Linear64预测rank64 coefficients，直接与原hyper向量投影到PCA basis的权重收缩，避免重建512通道feature。训练/epoch选择仅teacher coefficient MSE，旧24的前12训练、后12验证/诊断，min100/max300 epochs、patience50、seed2027；这些都是开发数据。部署把输入normalizer折叠入首affine。0/5/10% margin预算固定；selected parent状态真实gather为256x1x1输入原`output_upscaling`得到局部4x4 feature，再真实计算4mask logits，不使用完整teacher head的patch。完整head仅在质量评测作为reference。局部patch FP32同原full grid的漂移单独记，不改判旧5e-5 gate。保存best/last/deployment小checkpoint及曲线，可`--resume`在ERROR后恢复优化器/RNG并保留失败回执。若实际学生质量与oracle有明显鸿沟，优先查训练曲线/表示/选择器，不能引用oracle数字作为部署性能；实际kernel/完整decoder成本尚需公平测量，正式测速避开并发。

真实头完成300epoch、teacher-MSE选epoch288、20608训练参数。开发12图10%真实局部补回center/near IoU损失0.074/0.030百分点；所计算patch与原full-grid的FP32最大差4.768e-5，仅该诊断，不代表完整近似输出等价。`local_student_head_v1/disjoint128.json`完成128图/356对象、三regime三预算共3204行，图像与全部开发24不重叠；10%补回center/near/box IoU分别下降0.0613/0.0537/0.0696百分点，95%image CI均负；全图flip分别0.0462%/0.1018%/0.1459%，是小幅任务质量代价，不是数值精确保持。5%补回损失0.1898/0.1483/0.1457点，0%损失约2.31/1.87/2.15点。正式完整decoder比较同FP32/TF32off、128实际不同提示、MB128、同dense-associated缓存attention、all4mask/all4IQ；dense phase、当前廉价5/10%和最快implicit phase，真Inductor/fullgraph/dynamicFalse、正逆两轮，warmup10/reps30。测warm低分辨率完整decoder，不含encoder/fullresolution postprocess或冷启动；尚无速度或端到端收益结论。若compact比最快控制有效改善，再扩更现代SAM2/轻encoder、多提示场景和真实整流程，而非只堆SAM1 pilot。若无速度收益，profile真实排序/scatter/局部kernel开销，区分表示收益和实现调度，不复活已败前景约束。新廉价头当前SAM1限定，SAM2 highres skip需独立适配，不能假称已支持。

稳定性蒸馏已做完整匹配控制：原SAM冻结，只训练共享候选scorer；geometry-only为36维/4481参数，latent含mask token、IoU token及原sparse均值共804维/53633参数。四个variants（geometry/latent × response-distilled/GT-supervised）、固定三种子2027/2028/2029，同AdamW/训练上限/早停；蒸馏只用四扰动教师选择作为标签及开发集teacher CE选择epoch，监督控制用GT oracle选择及开发集IoU。基准提示仍来自标注，不能称从提示采样到训练全部annotation-free。旧256图现在是此新方法的训练集、旧128图是开发集，新seed2030的512图/1440对象排除全部旧408图，在模型冻结后评价；DAVIS2017 val的30序列首帧/61对象只作外部测试，不引入视频记忆。

| 新512图，三种子均值 | 中心点IoU | 近边界IoU | 框IoU |
|---|---:|---:|---:|
| 官方头 | 0.52381 | 0.32599 | 0.80895 |
| 最大原始mask | 0.59089 | 0.39993 | 0.81283 |
| 四扰动教师 | 0.62394 | 0.36871 | 0.80420 |
| geometry distilled | 0.62129 | 0.41367 | 0.80893 |
| latent distilled | 0.62633 | 0.40617 | 0.81156 |
| geometry supervised | 0.65137 | 0.44467 | 0.81316 |
| latent supervised | 0.66815 | 0.46313 | 0.81265 |

单次解码蒸馏确有信号，但同容量GT监督控制更强；不能将“新增小head后提升”全归因counterfactual教师。DAVIS首帧near-boundary：最大mask0.5128，蒸馏约0.386～0.388、监督约0.510～0.516，暴露迁移与错误教师问题，不能宣称通用质量改进。开发source重放183/768个regime未过5e-5、最大logit差0.007866；保留诊断，几何特征与质量标签使用原保存输出，重放latent仅作特征，不当FP32等价通过。

强模型实测：真实SAM2.1-L在同新512图，native预处理、同原点击/框、FP32/TF32 off，原head中心/near/box IoU为0.70657/0.48644/0.83512；四扰动选择0.72588/0.50938/0.83534，中心Δ+0.01931，image-cluster95%CI[0.00457,0.03403]；nearΔ+0.02294，CI[0.00408,0.04066]。最大mask中心反而比head差0.06093，说明该模型上的教师收益不能简单归因为偏向最大面积。

`distill_v1/sam2_distill/queue_status.json`八阶段已完成：原生SAM2.1-L latent导出 → 相同12个小head控制 → seed2031新384图（排除全部旧920图）→ DAVIS首帧。新384图原头center/near/box为0.7204/0.4991/0.8341；三种子latent-distilled为0.7437/0.5494/0.8344，geometry-distilled为0.7344/0.5658/0.8350；等架构latent-supervised为0.7531/0.5771/0.8369。教师为0.7269/0.5111/0.8365，新384的教师增量弱于旧512，需使用原逐样本结果计算成对CI，不能只报旧512的较好增量。DAVIS首帧latent-distilled center0.7844对原头0.6733有信号，但near0.7037仍弱于最大mask0.7399，box0.8094弱于原头0.8227。保留实用收益和跨域负例，不宣称已经达到发表水平。

新效率假设 `distill_v1/tangent_pilot24.json`已完成：用两个点击坐标方向JVP重建四个小扰动响应。旧24图开发诊断、同MATH-SDPA精确四视图控制：center原head0.74008、tangent0.72949、exact views0.72301；near原head0.48083、tangent0.54980（Δ+0.06897、imageCI[0.01498,0.12664]）、exact views0.50239；choice agreement center88.57%/near75.71%。logit最大误差在near病例可达49，不能当作高精度教师重放；它作为局部响应选择信号可能有意义。继续同固定参数的`distill_v1/tangent_test384.json`检验，不因一个开发pilot宣布胜负；无native优化延迟证据。若保留质量信号，再研究原生快速attention解析tangent/共享cache和置信度fallback，不能仅凭一阶展开宣称更快。原始一次解码蒸馏头还需真实推理代价、原生token0/dynamic-stability强对照和部件/整实例混合粒度测试；正式风险/coverage需独立校准集，不可把已用于early-stop的128开发图当独立conformal校准。只写小型features/metrics，不重复大张量/base权重；常规错误自主修复。

新颖性边界进一步收窄：近期MICCAI2026 workshop已直接研究缓存embedding后的prompt-perturbation uncertainty，存在跨域失效和高估计噪声（主来源 https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html）。当前可以追问的是原始一次解码能否预测/按需恢复有用的响应信息，以及共享计算的实际质量—成本收益，不能声称首次jitter或uncertainty。后续效率候选可先判断优化后phase head占比与其512维局部embedding的可压缩性，再决定函数蒸馏/置信度控制的精确局部fallback；这些尚未实测，不提前宣称收益。

### 磁盘规则与清理（用户新增要求）

保留重要结果和唯一资产，清除核实的重复数据/权重。服务器曾到99%/剩909MiB；已删除验证过的SAM失败下载残件、测速probe、EfficientSAM已解包重复ZIP、DAVIS已完成组装的分片和与12422个解包文件CRC一致的重复ZIP，释放约1.70GiB；Mac本地已删除与完整checkpoint拼接SHA完全一致的12个权重分片，释放1.845GiB，完整权重保留。

`storage_compaction_20261001.jsonl`已完成对3901个NPZ检查，其中3873个逐数组SHA验证后原子无损压缩，随后仅对逐成员byte一致的文件做物理去重；dtype/shape/数值/路径不变，零校验错误。压缩释放16679085533 bytes，内容相同文件去重释放1530459889 bytes，下载冗余清理释放1828313599 bytes，合计18.662GiB。最后核查df为29G使用/22G空闲/57%（包括其他会话同期变化，不能将所有df变化归因本清理）。保留全部JSON/CSV/log、失败证据、checkpoint与唯一数据；三个原始encoded-input cohort抽读shape=(1,256,64,64)、float32正确。压缩改变NPZ容器SHA，前后SHA及每个npy member SHA均记入回执，不能用旧容器SHA假称文件未变化。回执为`storage_compaction_20261001.summary.json`、`storage_cleanup_20261001.json`和本地`results/local_download_cleanup_20261001.json`。此后新实验使用compact压缩输出，不再写重复大NPZ。不得清理正在运行作业的输入或其他会话唯一文件。

### 质量后续实测（2026-10-01，新256图检验已完成）

用户授权继续实验并同时考虑质量。新方法诊断复用原图embedding，对每个原始点击作固定±8 resized-pixel水平/竖直扰动，用四个反事实提示的候选匹配一致性选择原始tokens1..3；不追加人工点击、不用GT判定扰动点是否前景、不改变原始候选。框作相同位移对照。seed2029新256图/711对象排除全部旧24+128=152图，规则在新输出前固定，主终点为中心点IoU，最大原始mask、原IoU头、logit稳定性是强对照。属于同COCO分布的新图检验，不能称跨数据集/SOTA。

| 选择规则 | 中心点IoU | 近边界点IoU | 框IoU |
|---|---:|---:|---:|
| 原官方IoU头 | 0.51535 | 0.32914 | 0.82418 |
| 最大原始mask | 0.61654 | 0.40197 | 0.81788 |
| 单次左扰动一致性 | 0.58855 | 0.34540 | 0.81984 |
| 两次水平扰动一致性 | 0.59754 | 0.36634 | 0.82021 |
| 四次扰动一致性 | 0.62117 | 0.37726 | 0.81888 |

四扰动比原头中心点Δ+0.10582、image-cluster95%CI[0.08366,0.12690]，近边界Δ+0.04812、CI[0.02527,0.07147]；但相对最大mask中心点只有+0.00463、CI[-0.02087,0.03195]，近边界-0.02471、CI[-0.05047,0.00079]。中心BoundaryIoU相对最大mask+0.02260，CI[-0.000012,0.04837]，仍跨零。因此保留相对官方head的实用收益，不能将其写成已胜过最强对照的主创新；COCO整实例粒度偏好仍是重要解释。

成本瓶颈已实际修复：GPU做完整分辨率binary intersection、FP64 IoU/六排列matching，保留原始四mask输出，辅助view完整计算后留在GPU，避免拷贝辅助mask到CPU逐对比较。新256图三个regime共2133组，GPU选择与既有CPU选择全部相同、consistency差异0。三张固定开发图、每图2或3个原点击、FP32/TF32 off、官方eager，正逆两轮，warmup2/reps5：同轮含encoder/decoder/postprocess/CPU原输出导出/选择，四扰动CPU 184.23ms→GPU 110.89ms，原头95.80ms；cached original embedding时92.73→22.06ms，原头6.66ms。计时不含JPEG读取、预先缩放/H2D/model load；不是最强compiled端到端竞争，也不能外推大批量。GPU匹配属于工程优化，无独立新颖性主张。

其他已完成反证/对照：虚拟feature flip在128旧图中心点+0.06915，但近边界-0.00642、框-0.01327（框CI[-0.02268,-0.00460]），不作为通用替代RGB flip。真实RGB flip等权logit融合在中心/近边界仅再加约0.09/0.10百分点，暂不主攻。标准SAM mask-logit feedback在24开发及128复核图，点提示token0明显退化、反馈后multimask head仍弱于最大mask；128框仅+0.00304对原head，暂不扩展此构造。

下一项有依据的方向：把**目标粒度选择**与**边界内容修复**分开研究，而非继续堆view数。可检验将多提示反事实稳定性蒸馏为单次解码的轻量选择头，保持共享image特征和原多义候选，并用整物体/部件混合标注检查是否仅学COCO粒度偏好；若走内容修复，需与等参数普通refiner及SAM2.1/HQ-SAM2比较。蒸馏/新训练尚未实施，不声称已支持该方法。Stable-SAM与SAMRefiner已经研究提示鲁棒性/多提示细化，扰动或反馈本身不新（主来源：https://arxiv.org/abs/2311.15776、https://arxiv.org/abs/2502.06756）。

本轮回执：`results/takeover_20261001_v1/quality_followup_v1/quality_summary.json`、`holdout_256/{protocol,prompt_stability_256,queue_status}.json`、`gpu_matching_256.json`、`quality_latency_gpu.json`、virtual/fusion/self_refine JSON。模型输出和encoded inputs留在服务器，JSON/log已同步本地。所有本轮作业完成，保持有卡模式。

**纯加速主线的当前增量较小，尚不足以独立支撑顶会主贡献。** 在真实SAM ViT-B权重、固定第一张真实图、128个不同正点、FP32/TF32关闭、完整4mask/4IoU、Inductor同微批128下，两轮最强已测dense phase均值35.876ms、implicit factor phase33.609ms，热缓存延迟下降6.32%；冷图含缓存约36.189→34.273ms。原版eager与通用cache/compile/phase的总收益不能全部归因因子方法。这是实测条件下的增量，不是整个方法族的结构性上限；encoder已缓存、多轮提示、轻量encoder与更大提示数的总成本需分别测量。

数值：原FP32 5e-5失败记录全部保留。第一张真实图三regime×八臂的FP64 decoder检查24/24通过1e-10，max mask error 3.510e-12；它不代表FP32门槛通过。官方同GPU同批重复误差0；128对64的logit max error 6.203e-3、全分辨率121像素不同。旧24图72regime里full batch对batch1有54组超过5e-5、max error 8.942e-3、69像素不同。应把同配置重复、批大小/编译敏感性、候选同协议误差、像素与任务质量分开报告，不能将全局最坏差异乘任意倍数后把旧失败改判通过。

质量：开发24图后冻结原IoU head、匹配平均IoU head、纯匹配一致性三种规则；seed2028的128张新图明确排除全部24开发图，共356对象。仅选原始1..3候选，无训练、额外点击或GT参与运行排序。纯一致性中心点IoU 0.52337→0.56834，Δ+0.04496，image-cluster95%CI[0.02302,0.06784]；近边界点0.32228→0.34606，Δ+0.02379，CI[0.00322,0.04507]；框提示Δ-0.00268，CI[-0.01024,0.00370]。这是COCO整实例协议上的前瞻复核信号，尚未证明跨粒度泛化、新颖性、成本竞争力或SOTA；真实RGB flip需要额外encoder。whole/part歧义是必须验证的解释，不能把oracle gap直接当成可全部回收的排序误差。

外部提出的稀疏头路线：非重叠两层deconv和逐像素LN/GELU的head在低分辨率局部独立，但选中位置实数域精确不意味着跳过位置或全分辨率插值后的完整输出精确；1/16子像素探针无法无条件保证其余子像素符号。边界比例是完整输出后的oracle统计，尚缺低成本可靠selector及scatter/重构开销。按外部69ms transformer+61ms head拆分，头部归零时133/69≈1.93倍，单优化头无法推出整体3倍；应先对最强compiled phase基线测优化后头部占比和oracle跳算上限，再决定实际selector实验。PointRend类先例需纳入创新核查。

所有回执在`results/takeover_20261001_v1/`，包括`real_p128_perf/`、`official_batch_sensitivity.json`、`fp64_real_check.json`与`flip_validation_128/{validation_protocol,flip_rank_score}.json`。用户最新资源指令为保留有卡模式，不能自行切回无卡。

## 接管后的连续队列

准备检查：`tools/takeover_preflight.py`在CPU加载实际SAM权重并检查全部JPEG、GT行序、点坐标、磁盘余量；CPU harness已验证八臂两轮，以及三种提示下的编码输入重放、完整掩码、离线评分与失败状态保留。

代码冻结到数据盘`runs/takeover_20261001_v1`，assets/runtime/results链接到原数据盘，继续复用现有编译缓存。队列路径为`results/takeover_20261001_v1`，拒绝覆盖已有结果。

```bash
cd /root/autodl-tmp/demo1_sam/runs/takeover_20261001_v1
bash tools/with_data_disk.sh /root/miniconda3/bin/python tools/takeover_queue.py \
  --output-dir /root/autodl-tmp/demo1_sam/results/takeover_20261001_v1
```

队列依次执行24图原版/缓存/关联基线/因子输出导出、八臂真实Inductor公平性能对照、24图新增八臂数值/输出重放。每阶段独立日志，`queue_status.json`记录实际PID、状态、命令和wall time；`gpu_telemetry.csv`记录整轮利用率/显存/功耗。错误或超时保留结果并显式停止，不能把对话状态当作作业进度。

以下CPU评分无需切换实例模式；按最新用户指令保持有卡：

```bash
/root/miniconda3/bin/python tools/takeover_score.py --run-dir results/takeover_20261001_v1/real_original --output-dir results/takeover_20261001_v1/score_original
/root/miniconda3/bin/python tools/takeover_score.py --run-dir results/takeover_20261001_v1/real_new_arms --output-dir results/takeover_20261001_v1/score_new_arms
```

下面为导入包和首轮的原协议记录；不要重复启动已完成的矩阵。

## 实验判断

已知随机 CPU 数值检查通过，历史预训练真实图片的所有执行变体仍未通过 5e-5 logit 门槛。首轮要判断：在同权重、FP32、TF32 关闭、同图同 dense prompt、相同完整四 mask + 四 IoU 输出和共同微批次下，共享因子能否胜过最快实测基线，冷启动和缓存复用分别付出多少显存与时间。

候选仍是投影因子加稠密 LN 伴随状态。没有实现 implicit-factor attention，也没有删 token、剪枝或训练。合成已编码输入不代表真实图片端到端性能。历史真实数据失败不因随机测试通过而消失；后续阈值不得放宽。

## 有卡后执行

以下命令在远端目录运行。无卡阶段不要执行本节。

```bash
cd /root/autodl-tmp/demo1_sam
bash tools/setup_gpu.sh
```

该脚本只核验已有环境并采集信息，不安装包。查看 `results/gpu_environment.json` 的 `nvidia-smi` 原文、XML、compute-app 列表、PyTorch 实际 VRAM 和 free VRAM；先确认无其他任务占用，核对驱动与 CUDA 13.0 构建可用性。补充魔改卡实际配置：

```bash
/root/miniconda3/bin/python tools/collect_environment.py \
  --modification-note '填写实际卡型、显存改装信息、功耗限制；无法核验的项写未核验' \
  --output results/gpu_environment_before.json
```

先做极小 FP32 CUDA 四方法检查：

```bash
/root/miniconda3/bin/python tools/run_short_gpu.py --phase smoke \
  --output-dir results/gpu_smoke
```

通过后先 eager，1/8/32/64/128 组独立合成 sparse prompt；总 token 默认 7（5 个 learned output token + 2 个 sparse token）。首轮微批次 8/32；每图 1 组自然使用微批次 1。需要归因时再补微批次 1 或不同 token 数。

```bash
/root/miniconda3/bin/python tools/run_short_gpu.py --phase eager \
  --microbatches 8,32 --warmup 10 --repetitions 30 \
  --output-dir results/gpu_eager
```

每个方法独立进程，保留未过滤的 JSON 和日志；整轮以一秒间隔记录功耗、功耗上限、温度、频率、利用率与显存到 `gpu_telemetry.csv`（仅有卡执行时启动）。同一场景包括未修改官方原版、普通缓存、strong dense_assoc 的 auto/auto、projected/dense、associated/sparse 路线及因子候选；缓存路线均测 explicit/SDPA。另补无缓存官方 SDPA adapter，避免仅让候选享受 SDPA。auto 路线只按运算量选序，**最强基线根据实测选择**，不把 auto 当作最快。

若任一方法 CUDA OOM，该场景所有方法共同减半微批次并全部重跑，原始失败记录保留。汇总拒绝不完整组、CPU 结果、重复组和 dry-run。编译失败与超时不伪装成 OOM，也不静默退回 eager。

首轮 eager 之后，在有判别力的少数场景测真正 inductor。下面只测 8/32/128 组、微批次 8，仍保留全部基线；若 eager 指向微批次 32 的收益，补相应编译场景。默认每个子进程最多 600 秒，编译超时会如实标记，原始日志保留。

```bash
/root/miniconda3/bin/python tools/run_short_gpu.py --phase compile \
  --prompts 8,32,128 --microbatches 8 --warmup 10 --repetitions 30 \
  --output-dir results/gpu_compile
/root/miniconda3/bin/python tools/summarize_gpu.py \
  results/gpu_eager results/gpu_compile --output results/gpu_summary.json
```

`backend=eager/aot_eager` 只能做接口诊断；runner 始终使用 `inductor, fullgraph=True, dynamic=False`，失败不降级。首次完整解码（包含首次编译准备）、固定 PE 建立、每图缓存建立、冷图完整解码、热缓存完整解码分别记录。稳态不包含编译、正确性统计、hooks、profiler、跨微批输出拼接或反复创建输入切片。微批次输出依次产生并释放，完整输出头仍执行；因此计时边界是 complete decoder，包含四 mask 和 IoU，不包含保存全部输出的业务开销。

数值失败默认停止该子运行并保留误差。只有诊断需要才显式加 `--diagnostic-on-failure`（runner 或 benchmark 均支持）；状态固定为 `FAILED_DIAGNOSTIC_TIMING_ONLY`。FP32 不能传大于 5e-5 的容差。汇总不会因为某条基线数值失败就从最快基线选择中剔除它来制造收益。

## 候选变慢时

以实际最有判别力的 P、微批次、attention 和 mode 替换示例参数。分别 profile 最快基线及候选；这条示例同时测四个方法，仅适合短诊断。

```bash
/root/miniconda3/bin/python sam_shared_decoder/execution_baselines/benchmark.py \
  --device cuda:0 --dtype float32 --prompt-batch 32 --microbatch 8 \
  --grid 64 --tokens 7 --attention sdpa --warmup 10 --repetitions 30 \
  --profile-dir results/profiles_p32_mb8 --output results/profile_p32_mb8.json
```

独立 untimed profiler pass 输出 Chrome trace、按 shape 聚合的操作耗时/内存和实际 CUDA events。检查 GEMM 的实际大小、clone/contiguous/broadcast 复制、cat 与内存流量、kernel 启动次数、LN 和两段上采样占比；根据 trace 定位再修实现。候选形成 write factors 的 attention 保持 explicit，不能称为全程 SDPA。

PyTorch SDPA 会按输入和环境选择实现，只有实际 kernel/dispatch 证据才支持 FlashAttention 声明：[PyTorch SDPA 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)。

测量完成再采集 `gpu_environment_after.json`，保存 raw timing samples、功耗/时钟状态和完整命令。小幅度胜负需要对关键场景重复独立轮次、交换测量次序，不能仅凭一次 median 确定稳定收益。只有明确收益才推进预训练真实图像（全部四 mask、IoU、后处理二值变化，原门槛不变）及 SAM2 移植。
