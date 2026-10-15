# 合法参考背景约束的完整冲突组合：唯一执行草稿

2026-10-11。完整规则现冻结于 [protocol.json](../../evidence/local/reference_background_conflict6200_20261011/protocol.json)，尚未运行；旧 `4fd6b2cb…` 配方及全部6200结果保持原样。下一生产由主Sol负责。本稿固定一条完整方案、两个必要完整消融和一个同数量位置控制，不以裸cue胜过FoRIS或普遍语义定理作为开跑条件。

## 输入、默认与已有结果给出的约束

仍为COCO200/PACO600/LVIS1400/原生PASCAL2000/原生SUIM2000，同一6200身份、原O24、合法参考Mask及两frame已封存完整B/G/R0/N。B是完整FoRIS，G是MEAN，R0是旧 `old_arms.R`，N是旧完整组合；不使用6380的新校准、不新增编码、样本、训练或大dense归档。

完整FoRIS为默认。旧N参与保护范围和完整对照，既有R0/MEAN仍提供互补proposal。SUIM428旧N删40097TP而仅拒94FP，要求重新裁决B-only；PACO1/75和COCO1/13的正确删假要求保留删除能力。已完成主臂在LVIS原图补595165TP同时补2673109FP，要求对新增作背景反证；原图111共有7563363个FP，不能把共同前景当真标签。去anchor完整臂仍低LVIS B，因此本稿也不假定参考核单独足够。

新增观察是**原坐标中的query多模外观样本**。不将整个Q按预测FG中心平移，不重新训练分类器，不把预测anchor或reference fit称为目标语义证书。

## 模块接口与固定规则

**1. 合法reference双角色核。** 沿用已完成配方的unit O24、合法Mask软coverage (p_i)、128个面积中点quantile FG/BG bank及重复权重。参考中心为 `mu_RF=sum(p*r)/sum(p)`；尺度仍为 `s²=.5*(E_RF||r-mu_RF||²+E_RB||r-mu_RF||²)`。统一 `tau=.07`，不选新带宽、whitening或APD。MF、MB或尺度缺失/非正/非有限时完整回B；0/无效旧H不作为BG。

令所有R/Q描述符均使用**同一个**坐标 `x=(unit_feature-mu_RF)/s`。参考FG/BG bank输出

`ell_c(q)=logmeanexp[-||x_q-v_ci||²/(2*.07)]`，`h_R=tanh((ell_RF-ell_RB)/2)`。

quantile为128个 `(k+.5)*mass/128`，取row-major累积质量严格大于该值的首项；重复项保留count/128。这是旧参考核的原坐标距离，公共减中心在距离中相消；不声称该减法新增信息。

**2. 参考BG先筛query FG样本。** CLI共同预测 `A=B&G&R0`，原16×16面积平均得到native64权重 `a_j`。只保留

`a*_j=a_j * 1{ell_RF(q_j)>ell_RB(q_j)}`。

相等、参考BG更支持、非有限的项不能充当query FG样本；被拒项不自动成为query BG训练标签，也不直接删其最终Mask。没有通过样本时完整回B。这里的正号只是固定模型下的FG/BG比较，允许误判，不称充分语义认证。

**3. 多模query FG与合法reference BG比较。** 对 `a*` 做同一128面积quantile，以原unit Q描述符建多模query FG bank，仍用同一个 `mu_RF/s` 坐标和同一核。输出

`ell_QF(q)=logmeanexp[-||x_q-v_QFi||²/(2*.07)]`，`h_Q=tanh((ell_QF-ell_RB)/2)`。

没有 `mu_QA`、整体Q平移或平均anchor兼容门。query样本之间的距离来自真实Q；其BG反面始终是合法reference BG。self-match可能抬高query得分，因此它只参与有限冲突动作，不独立重置全图。

**4. 完整双向冲突读出。** 每个frame独立读取原B/G/R0/N和原FoRIS连续score，沿旧规则得到 `d_B=2*render(minmax(score64))-1`、`conf=abs(d_B)`、`m_B=(2B-1)*conf`；span=0时conf=0。host minmax floor `1e-6`、FP64 bilinear及完整B符号保持旧口径。

`D_add=(~B)&(G|R0|N)`；`D_del=B&(~N)`。

N在此是完整旧组合保护信号：B与N共同FG不再被本模块删除；B-only旧删除仍可撤销或重新批准。N不锁死B-only删除。G-only及R0-only都可提出新增，不按domain或GT选来源；N已在G/R0并集中，显式列出便于身份核验。

将native64的 `h_R/h_Q` 各自FP64 bilinear到该frame，随后计算方向规则：

`h_add=min(h_R,h_Q)`，`h_del=max(h_R,h_Q)`；

`add=D_add & (m_B+.5*h_add>0)`；

`delete=D_del & (m_B+.5*h_del<0)`；

`F=(B&~delete)|add`。

新增要求两种任务对比都支持FG；删除要求两种对比都支持BG。它们共享reference BG且相关，不称两票独立证据。严格不等式，tie继承B；缺信息完整回B。000及111保持B，B&N共同覆盖保持；只在宿主conf<.5处可能编辑。旧N的新增仍须本规则重新裁决，可能损失旧N收益，完整N对照须公开该代价。

CLI和original依旧**各frame独立组合**：原图用保存的original B/G/R0/N及direct-render cue/host，不以CLI最终Mask回渲染替代。保留SUIM原reader Mask自身NN1024与评分GT直NN到实际frame的尺寸政策。

## 一次生产的完整臂与验收

| 臂 | 唯一差别 | 目的 |
|---|---|---|
| `background_vetted_complete` | 上述完整F | 验证BG筛查与双向冲突的实际整体作用 |
| `unvetted_anchor_complete` | query FG bank使用原a，其余reference双角色核、min/max、N保护、宿主、frame均相同；原A空则B | 完整消融：检验reference BG筛anchor的增量，不能用裸核指标替代 |
| `reference_only_protected_complete` | 同D_add/D_del、N保护和宿主，新增/删除均用h_R；没有anchor构造、query bank或anchor有效性gate | 完整消融：在相同保护域内检验query FG bank的增量 |
| `same_edit_budget_foris_margin` | 完全计算主臂后保留其同frame实际新增/删除数量，在同合法编辑域用原d_B选位置 | 区分更少编辑和位置判断；同成本包含主臂cue计算 |

预算每frame独立：`K_add=count(add)`、`K_del=count(delete)`；可选集合是 `D_add&conf<.5`、`D_del&conf<.5`。新增取d_B降序、删除取d_B升序，同值row-major。不把CLI同数量称为original同数量。

复用旧B/G/R0/C/N/K、完成4fd6的主臂/去anchor完整臂作为强控制；已完成去anchor臂因删除域不同，只是历史完整对照，不能冒充本稿的同保护域reference-only消融。新四臂共49600张双frame最终mask全部先seal后GT，复用原官方40/303/920/56/7槽位、缺槽0，两frame分别报告。直接I/U和四类gross须包含主臂对B/N/旧主臂/未筛anchor/同保护域reference-only/同预算，以及候选编辑相对旧N的补偿和损失；同预算逐位counts/ties独立复核。已曝光队列及条件区间边界保留。

运行沿已验证MPS FP32 dense核、CPU FP64最终cue/interpolation/host/threshold/budget与标量契约；逐例消费、释放，只存最终packed masks、小记录与整数。共享RF/RB核，主臂追加一个query FG bank，完整消融再追加一个未筛query bank；不新建dense档，不以估计核次数宣称实际提速。

## 能被本批推翻的判断

预期作用是：reference BG更支持的共同FG不能传播为query FG样本；合法query FG多模支持可保留B-only，而双向BG支持仍可批准正确删假；新增不能仅凭query self-match放行。若筛查没有改变bank或主要删掉真anchor、完整主臂仍近放行假proposal、或对同预算没有位置收益，完整结果即否定相应作用，不追加tau/alpha扫描。

reference角色迁移失败时错误anchor仍可能过筛，self-support也可能强化source-compatible误报；原111误报作为输出保留，000漏真不恢复。原N保护可能同时保住误报，B-only真目标可能无其它FG样本可救。上述是明确范围与反例，不能声称已解决语义可识别性或全模型信息损失。普通FG/BG density ratio、query self-support与prototype alignment已有先例，参见 [相关先例短表](REFERENCE_TASK_PRIOR_ART_20261011.md)；本稿只是待完整验证的组合，不命名原创算法。
