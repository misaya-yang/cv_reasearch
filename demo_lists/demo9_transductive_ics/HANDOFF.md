# demo9：完整参考条件推理的机制与证据交接

最新实测：文档F4七臂已完成241/241，没有新臂过线；001相对完整native−1.374pp，002相对同曝光/槽数全局控制−.048pp且CI跨0，新160未开。SAM3 v3在E69 weste:46597接续，A_DEV241合法视觉62.555对FoRIS59.122，A_CONFIRM600为61.091对59.783；两个paired区间均跨0。特权类别文字72.561/73.376只能诊断。DEV文字收益主要补漏，BBox信息损失已量到但未证明其导致文字差距。B固定17常数仍是监督聚合校准，不是零训练，当前正在同一有限guard执行确认/开发，禁止重复启动或自动扩展。报告、最终B决策和平台STOPPED核实仍待完成；以下是历史快照。

当前执行交接已准备完成：PLAN最新§11的A/B代码与统一有限队列部署在weste:13322的本会话目录`/root/autodl-tmp/demo9_transductive_ics`。入口`bash scripts/run_sam3_handover.sh prepare`默认只做无卡CPU准备；未来获授权后用同一入口`execute`。SAM3官方资产及隔离环境已就绪，A9/9、241/600完整UID和尺寸配对、根8/8、守卫18文件/0待生成依赖、B四个合成任务的冻结/拒绝提前评分/续接/评分链均已通过。GPU仍关闭，真实模型/CRF将在首10例检验，没有新的涨点结果。两支先各10例计时并复用前缀，再读DEV241/CONFIRM600，最后只写决策；超过冻结预算/故障即由唯一外层guard结束，保护他人GPU作业。A是合法语义掩码框适配，不冒称FSS作者实例框协议的精确复现；B的17常数来自跨折监督聚合，不是免训练。以下旧队列、等待循环和训练安排均不自动恢复；另一owner的cache/代码只读。

Claude 决策学习主线（2026-10-03 晚，已有确认集实测，细节与表格在 PLAN.md 对应小节）。方法：匹配不动（完整 FoRIS），学习决策；读出的输入只含相似度、排名和邻域差，输出加在宿主 logit 上且末层零初始化，所以起点就是宿主掩码。推理时把读出的 logit 按 FoRIS 放大自身响应的方式放大后在 0 处切，再走 FoRIS 自己的 CRF；未训练的读出与 FoRIS 逐位相同。依据：误差预算显示损失在决策（真实占比 +9.1、删掉全部误并 +20.0），而 30 多条单对规则与其他层读出都拿不到；同一卷积读出若不从宿主起步、不用留出类别早停，在 180 个 episode 上是 −3.4，加上这两点是 +1.6，1800 个时确认集 +3.9（patch 级）。踩过的坑：一次保持几千个 np.load 句柄会超过 1024 的打开文件上限；小读出的拟合受单核 CPU 限制，要按臂分进程；守卫在 GPU 上有别的进程时返回 DEFER_FOREIGN_GPU（返回码 2 或 4）而不启动，队列必须等待重试；不要修改正在运行的 bash 队列脚本；缓存按折顺序生成，只取前缀做试跑会只有第 0 折；特征主成分在新类别上有害（+1.62 对 +4.12），不要把它当作退路；学习曲线 600 个 episode 后持平，瓶颈是证据而不是基类数据量。

当前用户已关闭付费GPU；只在weste:13322无卡环境准备所给十种不同机制，不开卡、不跑CUDA、不下载、不本地数值测试。公共入口统一、同信息对照和有限八例DEV队列仍须当前版本CPU整合验收，不能把已写代码或合成正确性当涨点。旧联合对应十例156.04秒、完整native10/10 exact，联合−.8161ppCI[−1.5818,−.1067]；后来全FG检索撤销215/262翻转，但完整回放仍未胜native/分数删除，说明检索一致不等于身份判对。R241已完成580.303秒、native241/241 exact：block12删除−16.0593pp，末层删除−2.0107pp，query-core删除+.4560ppCI[+.2815,+.6305]且四折均正；全为已看过DEV，区间是照片组影响函数法。保留小正信号作简单控制，不称发表成果；这只否定固定读出，不是DINO上限。最新SAM3建议仅核了原论文：视觉/文本、实例框/语义框必须分开，负提示崩溃不是可修复收益的证明；17常数若由基类GT拟合就不是免训练推导。PLAN首节为当前状态；以下均为历史快照。小证据本机/服务器保存，共享缓存未动，本会话未运行T1，不接管Claude训练。

最新用户纠正：未经机制论证的响应训练路线已撤回、未执行。当前唯一候选改为参考分数排序＋查询关系定界；读Claude的extent_cut账本，补齐完整公共FoRIS同上采样/CRF及scalar/RGB便宜控制。旧训练入口已移除，小证据压缩归档；不是训练普遍无用或响应科学失败。以下准备快照均为历史，PLAN首节与STATUS优先。

最新真实数据准备已完成（无卡112.49秒、未导入torch）：每折60TRAIN/60DEV/原10评估，held类别及跨所有角色照片隔离；训练2014逻辑ID显式映射已有2017JPEG，尺寸及合法标签通过。TEST query只读PNG尺寸头，不看标签像素。小证据response_data_preparation.json本机/服务器保护；每类1episode只属小试验、40评估仍复用开发，不是充分训练或独立测试。有卡执行入口未完整准备，不租GPU。

当前推进到CPU准备阶段：同一个866实例已通过浏览器开成无卡模式（0GPU、0.5CPU、2GiB）；未开有卡。实际已装timm1.0.30/torch2.12.1的10个小模块案例通过，非预训练DINO/真实质量证据。head补齐全部选定层patch＋prefix QKV与即时响应，2,007,297参数；19项CPU检查、10例端到端小fixture和11项双臂训练恢复检查已过。训练器只按DEV原尺寸class I/U选epoch，TEST不进入训练接口。原始COCO2014训练图目录缺失，但已有train2017照片中10,377张匹配原train2014官方掩码；待显式映射与尺寸检查后复用，不下载或复制大资产。当前仍无新真实涨点、有卡队列未就绪。PLAN首节为当前状态。

截图所指出的FoRIS入口缺失已核实并在causal_v3补齐：旧64.2836只是direct-predict+CRF；同40开发任务的完整公共入口为65.1614，source观察/重放40exact。轴修正相对公共入口仅+.1581ppCI跨0。FUNGI已有gradient特征；参考响应的信息增量与估计器跨图迁移是两个独立前提，参考自拟合稳定/裁剪重编码不能代替整图收益。PLAN首节记录此次纠偏；此次只做源码与已有证据核验，未开机、未启动新训练。

当前直接授权只在关机状态多代理深推完整第五机制，先理解A及四完整改进，再行动。五代理加用户要求的两失败复盘代理已执行；不是七条GPU算法。instance866确认STOPPED，所有v2/v3旧科学队列不可重复。sam automation当前不存在，未重建。整体目标仍active，未完成。

最新实际causal_v3完成40官方COCOseed0复用DEV/四折各10：完整PUBLIC65.1614，Part2only60.8629，SVM56.4638，full publicFROST+timm56.9184（非官方hub复现），sourceaxisfix65.3195仅+.1581CI跨0。完整pipeline与弱逐点后端不能混淆。源码轴问题已证实但非重要任务效应，停止扩展。原生/public/axis2三种重放均40exact；科学报告与所有小trace已增量传回且reportbytesSHA复原exact。199.0877秒实验/2.438秒CPU分析后UI已关机。summary/分析在causal_v3。最早v2合同故障0评分、16.93秒关机证据保留，不称科学失败。

新必须解释的反例：未知真部件可NN更像BG，附着BG也可NN更像FG；完整结构既帮助又伤害，不能简单删。hard64cosineBG-veto固定离线审计65.384→60.313（−5.071CI负），不要复活硬否决。graphcut+matching缺extent与mass定义且已有近邻；perturb-response分组不能把计算依赖称物理同对象，原birth/coverage reward被套利反例否定，256额外编码不进入默认计划。尚未选定可竞争第五算法；按PLAN首节完成推导，禁止CPU绿灯即开机。

最新直接纠偏：主线是比较完整方法如何保留参考证据、排除错误对象解释、决定目标范围，撤回“更好的逐点距离/SVM就是答案”的预设。PLAN首节现有Claude计划逐项验收与Matcher/INSID3/HSNet/DCAMA/FoRIS/FROST完整机制对照。已有七候选只是有限试验，不是十张科学卡全部完成；官方四折paired主表、完整TPA强适配、独立泛化仍未验收。旧失败构造不恢复。

native_membership_v1现已CPU/source/solver/analyzer与199文件preflight通过，但GPU未启动，队列HELD。它仅有完整FoRIS/FROST源码控制及组件/重放/标准分类器，不能冒称六完整方法比较。实例866最新浏览器实见已关机。下一步围绕合法参考背景反证在完整推理哪一步被丢弃或覆盖建立可否定因果测量；若原始对应已混淆，不能据阈值算术反例猜一个分类器就能解决。GPU不开着等研究，下载/旧缓存/提交均不恢复。

最新目标仍是单参考、同DINOv3超越INSID3/FoRIS的具体方法；整体未完成。最新集中批次native-angular40/40完成：原尺寸class-mIoU native64.2836/paired63.9368，差−.3467pp CI[-1.7659,1.2468]；FROST95组件64.6256差+.3420 CI跨0。原生重放及保存mask40/40exact，无错误/回退，无特征缓存。配对算子令全部参考拟合视图变化减小，中位比.1238，但query辨识没有建立改善；这个固定角度协方差构造停止，不扫参数或恢复已结束队列。GPU批次总guard147.46秒后自动关机；结果只在无卡取回，本机results/native_angular_v1/summary.json、mask_diagnosis.json及788KiB archive保护，最终UI已关机。最新用户要求按已有证据组织完整批次再开机，下一步在关机状态区分参考迁移、少数难背景、评分校准和结构解码，不能因全背景AUC高就断言decoder是瓶颈，也不能靠新公式直接再开GPU。PLAN首节为准。

离线新证据：保存40个native1024mask的连通块oracle仅+3.77pp，largest朴素规则−6.72，64%FP连在含真目标的块内，不构建孤岛selector。源码midrange决策存在“新增强负背景使旧背景变前景”的精确CPU反例，但未证明真实任务主因。下一批准备原生逐阶段路径＋完整FROST源码强控制＋reference-only RBF-SVM完整分类标准控制，均非已验证新方法；这是此前准备状态；现有实际服务器source/solver/analysis与199文件preflight已通过，但最新用户纠偏后执行仍HELD，不开GPU。

## Completed pilots: failure attribution and current resource state

Historical shutdown was verified in the platform. The revised user goal permits a new prepared finite experiment cycle; instance866 remains STOPPED while code/data/CPU preflight are incomplete. Old queues stay retired. Seven bounded pilots and reduced native fixtures ran; original E9 all-method concept controls and E10 cross-domain/cost questions remain unanswered. Ten-example development results are neither full-fold scores nor seven independent generalisation proofs.

Three design problems are now concrete:
- Weak component backends: raw fusion baseline42.684 versus same-task INSID3 52.477 / fullFoRIS64.998; final/density baselines49–51 and ungated transportKDE40.261 are different declared pipelines. Their internal comparisons are legitimate, but they do not isolate a one-component replacement inside the strong FoRIS system. FullGL−13.191pp is the clearest negative for its fixed composite algorithm; kernel-pad controls approximately match.
- The deployed adaptive metric scarcely reaches the decision: unit features and orthogonalU give |d_M²−d_I²|<=4epsilon. Each log-density moves<=2epsilon/tau, so H=H0+beta*tau*(L_M−L_I) has |deltaH|<=4beta*epsilon. Selected epoch1 adaptive weights imply maximum .0231216, whereas the interpolated binary host has interior margins ±.5. From saved1024 masks, recovering parent64 masks and reconstructing bilinear H0 reproduced all10 host masks exactly. Only1800/10485760 pixels(.0172%) satisfy |H0|<=the per-task ideal bound. Even allowing every such flip to repair any FN/FP yields at most+.1841pp classmIoU on this cohort; actual change0. This is a conditional ideal-arithmetic reach bound, not a certified fullFP32 bound or proof about direction/generalisable evidence. Norm/QR/logsumexp numerical errors and actual delta fields were not saved. Do not regenerate deleted features to manufacture a stronger attribution.
- Support proxies are not query correction: fitted views lose to same-view uniform by1.6192pp(BG, exploratoryCI[-5.996,2.757]) and.4613pp(lattice,[-.848,.068]); E5 only7/10 tasks have legal region splits, three fall back, fitted prefix norms.00268–.00974. Ten training epochs and first-epoch-selected models are not a sufficiently trained evidence oracle. It is wrong to conclude that training or original DINO information is generally useless.

Evidence:results/native_runtime_v1/metric_reachability_audit.json, metric_reachability_iou_cap.json, completed_failure_attribution.json, completed_runtime_evidence.tgz. Computation:scripts/analyze_metric_reachability.py; no GPU/encoder/new images/training. Specific fixed constructs remain unexpanded. Before any next implementation, the missing premise is a removable error in the full strong pipeline that a legal observable intervention can actually change; candidate GT oracle coverage is insufficient. This is an evidence requirement, not a demand to know the new method's final score in advance.

## 共同输入、输出与公平接口

输入只有参考RGB S、参考掩码 m、查询RGB Q。任务/类别ID只用于采样与评分，不进入模型。DINOv3 ViT-L/16、1024输入、原权重保持不变；使用同一原生精度及声明的batch context。C3另有基类监督学习的字典与全局gain，所有学习型控制给同样标注和训练机会；C5仅用当前合法参考标注拟合32个输入系数。

f_s,f_q为单位化原生去偏特征，位置投影/压缩坐标需全训练/推理统一；不能在不同SVD补空间坐标中复用已训练字典。raw-attention融合空间另行定义，不能移植last-layer U500/APD。所有额外观察计入编码/校准/解码成本。

每个算法使用完整FG/BG参考证据，至少有相同信息的简单控制。FROST式密度为借用组件：L(q)=log mean_FG exp(k(q,s))−log mean_BG exp(k(q,s))，带宽只读参考；各方法同空间处理与原始分辨率评分。一个参考本身并不唯一消除类别/实例/部件歧义，评测任务需明确。

C1/C2独立比较使用原生support-only RSRM SAFR权重w_l与每图原生尺度s_il。SAFR源实现的相关统计是final高空间方差通道在raw层中的方差比例，不是直接坐标相关。借来的层融合、密度与空间处理不是新贡献。

C3的宿主接口H0仅在声明的bilinear/no-original-resize协议定义：FoRIS是实际minmax连续分数上采样后减.5；INSID3是最终patch mask浮点上采样后减.5。`tics/host_signed_field.py`只旁路观察，要求(H0>0)逐像素等于该原生阶段输出，原forward返回原对象。完整FoRIS+CRF仍作为强方法另列；不能声称bilinear小控制代表它。

## C1：原生轨迹上的局部全局内容读出

原生block继续X_(l+1)=Block_l(X_l)。同一原生Q/K/V下，仅patch-query/global-key列取L*_pg=q_p·k_g/sqrt(d)，patch-patch仍为(qr_p·kr_j)/sqrt(d)。U*=softmax_all_keys(L*)V，不能分别softmax再拼接。global query原输出保留。

旁路R*_l=FinalLN(LayerScale(OutputProjection(U*_l)))；不加residual，不回写下一block。φ*_i=normalize(sum_l w_l R*_(il)/s_(il))，同一supportFG/BG density、原生geometry/RGB产生输出。该变换不是纯语义化：Q/K/V包含此前位置交互，全模型不具有由此证明的原点不变性。

精确更新在拥有原row log-normalizer z时可写a_g=exp(L_pg−z)、b_g=exp(L*_pg−z)、r=1−sum a+sum b，u*=(u−sum a_g v_g+sum b_g v_g)/r。没有rowLSE或遇抵消/overflow时不能宣传5token廉价成本。保守实现用d→2d lift或chunk256完整归一化：patch Q=[qr,qraw]；global K=[0,kraw]；patch K=[kr,0]；V=[v,0]；scale仍d^−.5。

为何可能有用：减少当前读取中直接位置相位造成的跨图不稳定，同时保留预训练轨迹；仅当新FG/BG对比净改善才支持。失败意义：差分太小是当前通路弱；query混淆加剧是它承载必要语义/假设不成立，不能调阈值救故事。实现`native_attention_readout.py`；独立命令`native_readout_probe.py --arms native_raw,local_pg,kernel_pad`。CPU数学/原生轨迹验证已过；GPU/真实增益未验。

## C2：整条隐藏轨迹的全局内容交互改写

同一骨干第二个前向，在每层PG与GP混合项均取pre-RoPE内容点积，PP保留相对RoPE，GG保留原内容；全key统一归一化，原V/projection/residual/MLP继续加工修改后的状态。与C1相比，这一算法把改变传播到后续层，两个算法分别测试，不以一个失败证明另一个。

3d lift：patch Q=[qr,qraw,0]，global Q=[0,0,qraw]；patch K=[kr,0,kraw]，global K=[0,kraw,kraw]；V=[v,0,0]，scale=d^−.5。最终表示走同raw融合/density合同；权重和尺度仍来自另一个原生参考run。`--arms native_raw,full_rewrite,full_kernel_pad`保留同3d核的原生logit控制。

为何可能有用：若全局内容需要后续MLP/attention重新加工，局部旁路不足。失败意义：即使消除显式位置项，训练分布偏移可能压倒收益；不无限改层/强度。实现`global_content_attention.py`，回滚hooks/原权重，CPU已验，实际任务未验。

## C3：参考反例驱动的低秩任务距离（需要基类训练）

学习D×32参数A，QR得到U，U^T U=I。每参考只适配32尺度w：

M_S=I+U diag(w−1)U^T，d_S(q,s)=||q−s||²+sum_k(w_k−1)[u_k^T(q−s)]²。

U外特征方向不删除。谱在[.25,4]，禁止变换后再次L2单位化（否则不是这张M）。参考覆盖率<=.1或>=.9的纯patch构造balanced FG/BG三元组(a,p,n)：同label原距离最近2个见证排除空间半径2，异label最近4个反例；各类最多256anchors，总<=4096；邻居冻结。极小/退化参考无合法见证则原宿主回退。

δ+=f_a−f_p，δ−=f_a−f_n，m0=||δ−||²−||δ+||²，设计B_tk=(u_k^Tδ−)²−(u_k^Tδ+)²。B>0表明增强该方向有助正负区分，B<0表明允许同标签在该方向变化可能有利。只在参考拟合不保证跨图。

内层唯一偏好v*=argmin sum_t π_t softplus((κ−m0_t−B_t(v−1))/T)+λ||v−1||²/2。

G=λ(v−1)−B^T(π sigmoid(z))/T，H=λI+B^T diag(π sigmoid(z)(1−sigmoid(z))/T²)B，H>=λI>0。Newton有line search且必须达到stationarity；未收敛不伪装精确implicit backward。给incoming g，解H^T z=g，β_t=π_t σ_t(1−σ_t)/T²：

∂loss/∂B=outer(πσ/T,z)−outer(β*(Bz),v−1)；∂loss/∂m0=−β*(Bz)；∂loss/∂π=σ*(Bz)/T。其它直接outer路径继续普通autograd。

Δ=v−1，w=1+αΔ，α取1、所有谱界与可靠参考G={m0>=κ}的界的最小值。若B_tΔ<0，α<=(1−η)m0_t/(−B_tΔ)，η=.5。由此M谱合法且可靠参考间隔>=ηm0；它仅是该方向上的最大合法步长，不是全受限问题最优。dw=αdv+Δdα，α不能detach；min并列按声明的子梯度，数值gradcheck避开ties。凸性还给J(w)<=J(1)，仍只是参考代理目标保证。

完整anchor的Gaussian密度比L_w，修正e=τ(L_w−L_1)。最终H=H0+β_global*Upsample(e)，β_global=4sigmoid(b)与U一起在基类跨图loss训练。β并非自然同单位或后验校准；固定metric/SCML式控制同样拟合gain。训练loss=balanced BCEWithLogits(H/.1)+softIoU(sigmoid(H/.1))，开发类IoU选checkpoint。标签/所有角色照片及投影坐标严格隔离；测试仅support适配。

距离影响查询证据的局部条件：∂L_w(q)/∂w_k=[E_BG|q (u_k^T(q−b))²−E_FG|q(u_k^T(q−f))²]/(2τ)。同一任务权重因此可使不同位置的分数朝不同方向变化；参考关系在查询中不迁移时就会失败。

关键边界：φ目标与背景相同则任何M都不能区分；参考无对应反例/变化则w无可靠依据；保护α近0表明冲突；support改善但query损失表明本字典/适配迁移失败。不能用增加epoch或去掉保护掩盖这些。完整二元label反转可保持M相同而仅反转anchor标签，不能要求任何mask变化都改变w。

代码`reference_metric.py`、`train_reference_metric.py`、`reference_feature_probe.py`；exact implicit/active-alpha梯度、SPD、chunkdensity、零修正identity及部署回载CPU验证已过。SCML已经有共享rank-one基/三元组凸权重，LMNN/MetaOptNet也相关；新意与实效不能由这个参数化本身承担。

## C4：保留前景、改变参考背景的输入观测

制作原图、背景设为原图全局RGB均值、仅背景RGB确定性循环置换三view；所有合法FG像素逐元素保持。冻结同DINO读取完整表示，query仍原RGB。对每view用同FG/BG dense probability配方得到p_v(q)，输出sum_v a_v p_v(q)，原图geometry不变。

a_v=softmax(−mean_held_regions balancedBrier_v/.1)，只用support固定空间留出。**校准渲染不得使用held标签**：held区域所有RGB保持原图，只改其它train区域BG；最多12校准+3production support编码，均计成本。相同视图uniform融合是必要控制。没有合法留出则显式uniform回退。

为何可能有用：同一真实FG在不同context下得到不同读出，可检验错误是不是被参考背景引入，而非重排原始预测。失败意义：标签指定的对象可能需要上下文，人工背景造成OOD；support内稳定也不保证query更好。不能把保持像素误称保持原特征。`reference_views.py`；CPU检查FG像素、BG直方图、held渲染防漏，实际DINO增益未验。

## C5：参考监督仅改变既有全局token输入

保留CLS/register数量及全部DINO权重，在第一个block前给5个prefix加δ。固定seed4101 QR得到32个正交方向b_k，θ为每任务32个系数：δ=.1||prefix_native||/sqrt(32) *sum tanh(θ_k)b_k；预cast范数不超过原prefix的.1。并未离线学这个字典。

当前support按固定空间region划train/validation，train区域留出FG/BG原型对比BCE+prox .01，20步Adam/lr.05，validation不用作trainanchor/target，含θ=0 checkpoint。其它标签不读取。将选定θ冻结应用到query编码，再走同density/nativegeometry配方。选择零/缺合法regions就native回退。

为何可能有用：让真实参考的判别损失反向改变编码器内的信息汇总，而非固定最后特征的距离；骨干权重仍冻结。梯度必须经过DINO到输入系数，不能用no_grad缓存extractor或只靠prox假优化。20步需1+40support feature calls，不能叫免优化。matched fixedprompt/native controls披露相同拟合诊断调用，但原native实际最低部署成本另列，不能制造GPU负载维持忙。

失败意义：参考专属颜色/位置捷径、32方向/10%幅度无力、真实query分布不迁移。最多当前一次固定预算，不追加prompt层/半径/seed无限搜索。VPT/TPT已建立prompt/tune原则，本法当前只是明确的候选。`reference_prompt.py`；CPU只改prompt、骨干/flags/hooks不变及backprop通过；真实DINO未验。

## C6：跨图完整分布的不平衡运输

C_ij=1−f_qi·f_sj；a_i=1/Nq；参考FG/BG各b总质量.5、类内uniform。求P>=0的凸目标：<P,C>+eps KL(P|a⊗b)+rho_q KL(P1|a)+rho_s KL(P^T1|b)。eps=rho_q=rho_s=.1，max200/tol1e−5；chunk logsumexp不存全coupling。

score_i=sum_FG P_ij/sum_all P_ij，阈值.5。有限rho松弛两侧边际，不强制query前景mass=.5；但参考.5依旧是软先验，不能说完全无面积偏好。rho=0退化同full-anchor KDE，是实现与机制控制；balancedOT硬边际是另一控制。

为何可能有用：区别局部独立最相似与整张query/support分布的竞争，可减少同一参考hub不合理解释所有query，且允许query大小与reference不同。失败意义：softprior仍造成配额/错误hub匹配，或KDE已经解释所有收益；不可加rho/eps sweep。`unbalanced_transport.py`，CPU对偶驻点/分块/极端面积测试过，真实任务未验。

## C7：改变patch格点观察、由参考选择融合

固定pixel offsets(0,0),(8,0),(0,8),(8,8)，reflect pad/crop保持输入尺寸，不改RoPE公式。S与Q读取4fullmaps，按已知originalpixel→shiftedpixel坐标回采原格点，剔除插值读到reflected输入的边界。空间配准是精确坐标关系，不是逆转encoder；globalattention仍可能传播reflect内容。

每view同dense probability配方，support空间留出Brier得到a_v，query probability按valid重归一sum_v a_v p_v。uniform4views是必要控制。mask在view坐标的原像可见区域合法，反射RGB没有凭空复制annotation；不同view/特征平均和概率平均不混在一个算法。

为何可能有用：让原来混在同一16×16patch的FG/BG被不同格点分别观察，检查边界证据是否本来受采样限制。失败意义：只有uniformTTA有效则参考加权无独立价值；若内部语义错误不随格点变，这种新观察不解决它。FeatUp已有多视图特征重建原则，不宣称patch位移新颖。`reference_views.py`，CPU坐标/边界标注/加权归一验证过，无实际DINO结果。

## 七算法共同的失败账本

每类原交并为I/U=J，新追回FN=a、丢TP=b、新FP=c、删FP=d，则J'−J=[a−b+J(d−c)]/[U+c−d]。不能只报召回或代理loss。报告a/b/c/d、supportfit→querytransfer、原来好的episode退化、目标缺席误报与额外编码/训练成本。

有限保存输出bank的GT选择仅作诊断。`finite_bank_oracle.py`按每类ratio用Dinkelbach精确选有限bank中的最大sumI/sumU；不把逐episode最佳IoU误当严格classmIoU上界，更不当可观察信息或任意新算法上界。所有GT评价在prediction冻结后。

图扩散旧ARCHIVE已量56.26 vs56.06，所以仅保留`reference_diffusion.py`作廉价强控制，不把其10个CPU SPD检查重包装成新方法。旧selector/SAM/GIC/info也不恢复。任何候选若只胜弱控制、不迁移或没有新增合法证据收益，保留小失败证据、清重资产；不再无限变体。

## 当前可执行范围和下一状态

这些实现当前完成的是CPU数学/梯度/geometry/hook/接口验收，非7个已证明有效的方法。训练器和feature-probe有独立输入/GT后置/部署合同；RGBview与prompt有frozen-encoder callbacks，真实encoder/host整合仍需现有资产的独立runtime验收。C1/C2可选独立arms，不默认一条十项GPU队列。

未来选任意卡都先CPU完成数据与源preflight，单任务finite guard，失败/无ready next/确认60秒空闲只停own再平台关机，保护foreign jobs。现在用户明确无卡，不申请GPU，不下载资产，不假造利用率。PLAN和`results/prepared_independent_experiments.json`是10独立卡权威入口，旧first10单队列receipt不覆盖改过的源代码，必须重验。

主要主文献：[INSID3](https://arxiv.org/html/2603.28480v1)、[FoRIS](https://arxiv.org/html/2609.03384v1)、[RSRM](https://arxiv.org/html/2606.24297v1)、[FROST](https://arxiv.org/html/2606.31136)、[SCML](https://arxiv.org/html/1404.4105)、[VPT](https://arxiv.org/abs/2203.12119)、[TPT](https://arxiv.org/abs/2209.07511)、[Unbalanced OT](https://arxiv.org/abs/1607.05816)、[FeatUp](https://arxiv.org/abs/2403.10516)。标准原则与当前具体组合分别核对，未宣布新颖性完全确认。

## 历史转导路线证据（仅记录，不恢复队列）

## 一句话

上下文分割都是“一张参考图对一张目标图”地做；实际要分割的是一批图，这批图本身就是不带标注的同类图。
把它们先分割一遍，再把其中可信的图连同预测掩码当作额外参考图，COCO-20i 标准 episode 上 INSID3 从 55.5 涨到 61.7（四折都涨），
套在 FoRIS 上从 59.1 涨到 63.3。池子里的图给真掩码时是 67.5，所以还有约 6 点空间。

## 方法（当前版本）

输入：1 张带标注的参考图，N 张不带标注、含同一概念的图（8–16 张就够；4 张没有收益）。

1. **第 1 轮**：用基础方法（INSID3 或 FoRIS）单样本分割每张无标注图。
2. **估计可靠性**：把每张无标注图 y 连同它当前的掩码当作唯一参考，去分割其他无标注图；预测结果与那些图当前掩码的平均 IoU 就是 y 的可靠性。
   另一种是“往返分数”：用 y 去分割带标注的参考图，与参考图真掩码的 IoU。两者都不需要 y 的标注。
3. **更新**：每张图用“带标注参考图 + 可靠的其他图（目前取可靠性前一半）”重新分割。
4. 重复 2–3 步共 3 轮。

对 INSID3 还改了一处：多参考图时它让每张参考图投一票、过半数才算候选；改成“最近邻最相似的 5 张参考图投票”。
原规则在参考图多时不可靠（16 张真标注参考图：67.7 对 69.6；滑板类参考图越多越差）。

代码：`tics/imageset.py`（缓存特征上的 INSID3，与官方实现逐 episode 一致）、`tics/propagate.py`（可靠性、选择、迭代）。

## 这个想法是怎么来的

1. **先有账本**。demo4 在 INSID3 上量了错误构成：现有簇按真值挑能到 82，实际 56；漏掉的前景 90% 是同一物体里参考图没出现的部分；
   这些部分对参考前景和参考背景的相似度是 0.45 对 0.46，分不开。
2. **十四类单对图规则都失败，说明缺的是证据不是规则**。用基类标注训练的打分器把所有现有证据组合起来也只多 2–4 点。
3. **找“什么确实有用”**。参考图从 1 张加到 5 张涨 6–9 点，这是量到的最大一块。问题变成：不加标注，从哪里拿到等价的证据。
4. **部署时免费的证据来源只有一个**：要分割的那批图本身。它们含同一概念，只是没有掩码。
5. **15 分钟的检验**（官方实现直接跑，150 个 episode）：1 张标注 57.1；加 4 张无标注图不筛选 59.3；用真值筛选 61.6；5 张标注 63.4。
   下限和上限同时量，知道空间在 +2 到 +6 之间才往下做。

## 研究方式（请照做）

- **先量问题，再谈方法**。每张结果表都带三行：基线、最朴素的做法、用真值的上限。没有上限就不知道还剩多少，没有朴素对照就不知道方法本身值多少。
- **每个实验几分钟出信号**。先在几十到一百多个 episode 上看方向，前几组没有信号就停，不排几小时的队列。
- **先搭快循环再试想法**。特征和聚类缓存后，一条规则在四折上跑完不到两分钟；这之后才值得试规则。缓存版必须先和官方实现对齐（见下面的坑）。
- **用真值做“修哪种错误值多少”的干预**，再决定修哪种。伪掩码的两类错误就是这样分出来的，它直接否定了“统一处理”的三条规则。
- **写下预测再跑**。`PLAN.md` 每个实验的四行就是这个用途；不符合预测时，四行里已经写了该改哪里。
- **差异用配对区间说话**。同一批 episode、对 episode 重采样；一折 100 个 episode 的区间约 ±5。
- **在一折上调出来的阈值不算结果**。跨图共识阈值在第 0 折 66.7，换到第 2 折反而比不加还低。
- **换一个基础方法验证**。只在 INSID3 内部成立的东西是 INSID3 的补丁；在 FoRIS 上黑盒也成立，才是独立的一步。

## 已经成立的、还没成立的

实测成立（COCO-20i，重建掩码，池子抽样一个种子）：

- 无标注同类图有用，四折都涨；朴素版本 +3.3，互相一致过滤 +6.1，配对区间 [+4.2, +8.2]。
- 收益需要 8 张以上，16 张以后不再涨。
- 叠加在 FoRIS 上四折为正（每折 60 个 episode，区间未算）。
- 伪掩码错误分两类，各占约一半损失。

由实测推出、未直接验证：

- 互相一致比往返好，是因为带标注的参考图本身可能不典型。只有“前者数字更高”这一条证据。

未验证：

- 其他 8 个数据集；每折全部 1000 个 episode；多个种子。
- 池子里有不含目标概念的图时是否稳健。30 个 episode 的冒烟测试里互相一致规则掉到了单样本以下，需要 E3 确认。
- 可靠性“取前一半”是否优于随机取一半（E1）。这一条不成立的话，方法的说法要改。
- 新颖性：经典少样本分割（PPNet、半监督 FSS）和医学分割（级联式上下文分割）里有“把预测加进参考集”的先例；
  我搜过免训练基础模型 + 一批无标注图的设置，没找到现成工作，但不能当作已核实。

## 踩过的坑

- **编码方式影响 INSID3 的聚类**。编码器在 bfloat16 下运行，同一张图单独编码和与参考图成对编码，特征相差约 0.01，足以改变凝聚聚类的结果：
  单独编码时缓存版与官方实现只有 43% 的 episode 掩码完全相同，成对编码时 97–100%。`scripts/cache_episodes.py` 已改为成对编码并自动核对。
  README 里的四折主表是用单独编码的旧缓存算的（`../demo4_incontext_seg/scripts/episodes_eval.py`），各行之间的差是配对的、成立，但绝对值要用 E1 重新出。
- **每类固定一张参考图的设置方差极大**：同一折换一组参考图，单样本从 62.0 变到 51.3。主结果用标准 episode。
- **FoRIS 和 INSID3 的源码都有顶层包 `models`、`utils`**，同一进程里先导入谁就是谁。`run_blackbox.py` 先导入 FoRIS 再导入 demo4 的工具。
- **FoRIS 每次调用约 0.5 秒**，互相一致过滤要 N² 次调用；池子取 7、只做一轮才跑得动。
- **服务器**：`pgrep -f` 会匹配到自己的 ssh 命令；长命令用 `nohup ... &` 加日志，再用 `until grep ...` 等结果；日志里 `Error importing huggingface_hub` 是无害警告，等结果时不要匹配 `Error`。
- **磁盘**：缓存每折 3.6–8.6 GB，系统盘只有 30 GB。跑完一折删一折。`/root/demo4_cache` 下的环境、权重、COCO 数据被别的会话引用，不要删。

## 文件

| 路径 | 内容 |
|---|---|
| `README.md` | 全部结果和诊断数字 |
| `PLAN.md` | 下一步实验，含命令和停止规则 |
| `tics/` | 方法本体 |
| `scripts/cache_episodes.py` | 建缓存并与官方实现核对 |
| `scripts/run_episodes.py` | 标准 episode 上的全部变体和对照，输出逐 episode 记录 |
| `scripts/run_blackbox.py` | 基础方法当黑盒（FoRIS） |
| `scripts/stats.py` | 分折表和配对区间 |
| `scripts/open_pool_*.py` | 另一个会话在本目录写的混合池子实验，不属于这份交接；改动前先和那个会话确认 |
| `results/` | 已有结果的 JSON（按 `.gitignore` 不入库，只在本地和服务器） |
| `../demo4_incontext_seg/` | 账本、失败的十四类规则、探索阶段的脚本（`scripts/stream_*.py`、`episodes_*.py`、`foris_stream.py`） |
