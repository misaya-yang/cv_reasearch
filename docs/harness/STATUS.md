# 当前状态

## CPU100首批运行记录（2026-10-07）

reviewer首批审查为29张卡 `accept_for_probe`、1张需修订、2项重复关闭；放行不代表有效。随后30个完整方法臂及全部对照在4个复用开发样本上完成native FP32 DINO1024推断，并由独立scorer读取QGT评分，所有请求臂完整。该结果是描述性四例活动探针，不是独立确认或总体结论；最高方法分数`inv_huber_reference_readout=40.4669`，其声明的Huber-ridge对照为`40.5636`。其他臂在这4例上有正负差异，未建立稳定正收益；100方法目标仍未完成。见[四例汇总](../../evidence/local/cpu100_20261006/server/probe29_native4_v2/methods_summary.md)、[完整配对报告](../../evidence/local/cpu100_20261006/server/probe29_native4_v2/methods29_native4_v2/score/report.json)和[审查记录](../../evidence/local/cpu100_20261006/reviews/batch01.md)。

此前本地直接DINO输入记录（2026-10-06）：独立DINO-only入口已完成四例RGB128 prototype输出，不用FoRIS分数/Part1/旧MEAN；该本地活动记录未评分，不代替RGB1024/600目标。之后CPU100四例1024结果见本页上方。详见[本地输入记录](../../evidence/local/research_20261006/direct_dino_input_01a1100b/README.md)。

旧46466端点于2026-10-06关闭；其连接、部署、重试及旧队列停止。之后的CPU100任务使用单独授权的新端点，见[服务器说明](SERVER.md)；不从旧launch恢复工作。灰度first4已完成，随后full600及fresh parity未启动；纹理实际smoke4未运行，600资产导出完成状态未核实。旧19个候选定义不等于1000项或高质量结果。

- 灰度first4报告：[结果与比较](../../evidence/local/research_20261006/grayscaled_dino_preparation_01a1100b/actual_result.md)。
- 纹理候选仅有本地实现和合成RGB检查，未做真实smoke：[README](../../evidence/local/research_20261006/reference_texture_01a1100b/README.md)。
- 表示轨迹观察已关闭，未测查询质量：[结果摘要](../../evidence/local/research_20261006/representation_path_01a1100b/review.md)。

工作缓存仍来自冻结DINOv3，但已过FoRIS Part1条件位置去偏/再归一化/FP16保存，不是未经处理的原始DINO输出。新灰度直接FP32 final-LN特征未Part1，而MEAN基场仍用旧processed cache；不能把完整实验标成已去除FoRIS处理链。

Pro原文不是五个纯DINO-only方案：其3.2明确允许FoRIS宿主连续证据，M1/M5固定位置gate，M2角色字典明确用末层去位置特征；同时3.1要求真实FP32编码。加工DINO缓存的存在不自动解释质量失败，旧FP16适配结果也不证明五个严格原合同均已失败。

以下内容保留旧46466端点关闭前的结果与历史授权；新CPU100状态见页首和PLAN。

2026-10-06：用户要求 **1000种方法、继续实验和多代理并行迭代**，并已明确授权在新46466 CPU服务器上传和运行，覆盖此前关机/本地准备限制。
原九项600例完整实测已完成：native **61.335314**，当前MEAN **62.931546**；九项均未超过同批MEAN。
[完整结果与配对区间](../../evidence/local/research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json)。
这是复用公开开发集，非独立确认；合成区分例没有转化为稳定真实收益。

凸包、ProM2四状态关系、三体关系和布局局部修正版的固定600例已封存并评分：分别为48.915345、60.632462、60.778988、62.308618，均低于同批MEAN 62.931546；布局局部修正版高于native 61.335314。参考锚点吸收、Gaussian密度修订和RGB Potts控制的另一固定600例批次也已完成：分别为37.821542、61.949185、62.903951；RGB Potts对MEAN差−0.027595，配对95%区间[−0.160435, 0.032420]，未显示超越MEAN。两批均复用开发数据，完整配对区间见[扩展结果](../../evidence/local/research_20261006/server_prepared_01a1100b/extensions_public600_v2/score/report.json)和[后续候选结果](../../evidence/local/research_20261006/server_prepared_01a1100b/next_candidates_public600_v1/score/report.json)。
ProM2计算优化的四例六臂全场及两级mask逐值一致，平均单例14.889秒降至2.938秒（5.067倍），详见[记录](../../evidence/local/research_20261006/parallel_review_01a1100b/pro_relations_profile_v1/README.md)。
后续RGB局部proposal v2是已暴露来源上的posthoc读出修订：600例63.007428，对MEAN的差为+0.075882 pp，配对95%区间[−0.006268, 0.135967]，不证明稳定提升，独立方法增量为0，详见[报告](../../evidence/local/research_20261006/mean_rgb_proposal_v2_01a1100b/server_replay600/score/report.json)。
目前19项候选定义（14本轮自主+5所给Pro）：14项完成各自固定600，4项仅完成固定smoke4，1项仅本地检查。灰度DINO主路径first4已封存并评分：原尺寸36.979714高于MEAN35.695213，但低于同RGB37.205113、简单亮度37.505490、H20 37.785206。root此前曾获准唯一固定600，但其后用户关机，该600未启动；当前不续接。1000项和高质量目标尚未完成；控制行/前向次数不计独立候选数，候选计数不认证科学原创性。
Pro真实检查还包括：M1首4例负向活动筛查；M3固定600例低于native/MEAN；M4首4例、17次编码的完整分支低于缓存native；M5已修复实际CPU尾块数值偏差并完成固定四例全审核/评分，但原尺寸15.805910、同renderer MEAN 35.695213（−19.889303 pp），不支持分割收益。来源分别见[M1](../../evidence/local/research_20261006/pro_context_preparation_01a1100b/score4_v1/report.json)、[M3](../../evidence/local/research_20261006/pro_roles_preparation_01a1100b/public600_result_46466.md)、[M4](../../evidence/local/research_20261006/pro_environment_preparation_01a1100b/real_smoke4_46466/report.json)和[M5四例结果](../../evidence/local/research_20261006/pro_message_preparation_01a1100b/actual_smoke4_v1.md)。这些数据均为已暴露public/dev样本或缓存适配结果，不是独立确认。
QK role-consensus v3为既有QK family修订、独立方法增量0；固定暴露smoke4已完成：agreement=33.822374，低于MEAN=35.684576（−1.862202 pp，描述性配对95%区间[−3.045823,−0.678582]），也低于direct H20=34.601188。仅4例/4照片簇，不作总体结论；不自动扩24/600，见[报告](../../evidence/local/research_20261006/qk_role_consensus_v3_preparation_01a1100b/score4_v1/report.md)。
本研究聊天chat01a1100b已完成RCG+RGB固定组合600例：**63.125362**，相对当前RCG +0.036362，配对95%区间[−0.015595,0.135117]；相对native +1.790048，未达+2目标，也低于fine16 63.464527。组合增量0，见[完整结果](../../evidence/local/research_20261006/rcg_rgb_composition_01a1100b/README.md)。Object-crop CLS revision0也已在本聊天授权下完成真实单例：57次新增128 RGB前向，46.09秒/3.239GB；主方法和同crop patch控制的完整IoU均0，无目标恢复，关闭固定修订，见[实际收据](../../evidence/local/research_20261006/object_cls_revision0_01a1100b/actual_result.md)。另一聊天commit-only范围没有执行这些实验，但不撤销本聊天既有CPU授权；其method.json范围文字保留，项目事实以带chat身份的实际收据为准。
同上下文object-removal revision0真实单例已完成并关闭：59次真实128前向、51.078秒、峰值3.239GB。全部六行完整IoU0/新增TP0；CLS差响应弱于同信息patch差控制。相同原背景的Δ表示是不同数值量，但未获得类别恢复证据，计数0，不扩样/调参，见[结果](../../evidence/local/research_20261006/object_removal_revision0_01a1100b/actual_result.md)。局部拓扑核查未形成独立构造也已关闭。最新有限观察均结束，旧队列不恢复。
新机制与改进版本分别记数，对照/参数不记数；首4例不能代替600例结论。

## 上一轮执行与资源口径（历史）

此前执行期间服务器为32CPU/60GiB、无可见GPU；
所有任务合计不超过30线程，使用现存缓存、RGB、标注和权重，不下载、不启动新实例。
目前原九项推断304.025秒、峰值4.60GB、编码0。最新两批600例推断耗时916.4秒/峰值4.76GiB与478.6秒/峰值3.10GiB，均无额外编码；M4四例实测每例17次编码、约363–405秒、峰值约2.91GiB。这些成本只适用于各自报告的执行合同。
各队列使用独立不可变源码快照，预测先封存再评分；历史MEAN producer单独作评价对照。原九项、两个扩展600批次及Pro有限检查均已结束。
唯一后续清单见 [PLAN](../research/PLAN.md)，当前资产及哈希见 [SERVER](SERVER.md)。

## 已完成与未完成

- 本次完成协议、计划、主张、状态、交接及服务器说明的一致化；删除重复导航和旧活动日志入口。
  原始结果、实现、运行快照及未提交的
  [区域联合目标草案](../../evidence/local/research_20261006/standalone_region_objective.md)保留。
- 当前保留的四个角色是参考证据、查询特征图平滑、区域重评分、亚格点读出。
  chat01a1100b 已在 PLAN 固定算子顺序及 [S3/RCG16/fine15定义](../../evidence/local/research_20261006/complete_method_contract_01a1100b.json)，
  并绑定9个本地来源文件及6份样本清单；独立完整入口、数值一致性和完整实测仍未完成。
  查询图不限于空间近邻；MEAN强对照保留同一图与求解，改变的是参考引导及其系数。
- 约 **60.1** 是用户转述的组合预测，不是已测结果或理论保证。已测阶段链为
  51.94 → 56.17 → 58.44 → 59.69；完整 FoRIS 为 59.074825，INSID3 对照为 55.006020。
  精确来源与条件见 CLAIM；旧区域联合目标草案的 59.7/60.1 外推仍保持撤回。
- “打平时不硬挑、两块都留”候选按本次提供的错误事实与推理关闭，无需上卡验证。
  这是对该候选的判断，不是对所有潜在机制的信息论否定。
- 计数预测、边界分支及此前撤回的四组队列保持关闭；层次 oracle 和联合目标草案保留为研究记录，
  不再成为默认下一步。该历史收束记录不替代用户随后明确授权的本轮并行开发。
- 反向重建参考掩码候选关闭。用户新增的600例硬最近邻落点统计中，81个可比较失败例的背景落点
  区分接近随机；最差34例的净前景落点仅9例偏向真目标。它不支持用参考背景排除近类干扰物的
  提议依据。源码与范围见 [账本](../../evidence/local/RESULTS.md)及
  [统计脚本](../../scripts/score_reverse_landing.py)。软互惠矩阵未测，不以这个差别续开变体。
  原“正混合二阶项产生对象间排他竞争”的解释撤回；该项的前景行只上推、背景行只下压，
  交叉耦合不构成对象身份判别的证据。该条是文档整理时的历史操作范围，不是当前CPU授权边界。
- 整理期间另一个聊天新增了[种子不确定性设计](../../evidence/local/research_20261006/seed_uncertainty_01a1100b/design.md)
  及其条件性数学检查记录；已保留并合并其存在事实。它不同于简单“两块都留”，本次未评判其证明，
  也无新分割效果证据；按当前收束要求，它不自动进入验证清单。

## 数据与执行状态

已有 [photo_disjoint_confirm1200_v1](../../evidence/local/research_20261005/pipeline_verified/photo_disjoint_confirm1200_v1/README.md)
元数据记录：1200 对、74 类、2249 张照片，当时与登记暴露照片重叠为 0。
这是待复核的确认清单；不能仅凭旧收据宣称截至现在仍未暴露。其他已有 1200/4000 结果多为复用数据。
跨数据集现固定为已有SUIM582、PASCAL-Part598、PACO-Part599及LVIS599；均为复用数据。
行表及算法常数已绑定，原始照片/类别/掩码映射、独立入口及实际运行成本尚缺，不能称为已可执行。

此前新46466端点已验证并运行实际缓存实验；服务器现已关机。以本页顶部和PLAN为准，不连接端点，也不恢复旧实例或PID。
