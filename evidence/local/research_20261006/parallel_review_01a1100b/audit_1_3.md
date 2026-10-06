# 第1–3项 CPU 候选审查

审查日期：2026-10-06。仅本地只读源码、现有合成收据及用户提供 Pro 报告；额外计算仅标准库的20种标签排列与七节点星图代数。没有启动真实分割、SSH、GPU、新资源，也没有改动方法、PLAN、STATUS 或用户报告。遵循 [AGENTS:3–8](/Users/yang/projects/CVPR2027/AGENTS.md:3) 与 [PLAN:5–14](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:5)。本文件是审查结果，不是新增任务清单。

**结论：三个实现基本符合各自公式，真实新方法样本均为0；相邻关系有额外参考结构量，Huber仅改变既有传播，颜色法增加查询光度边界量。三者都不能承诺真实增益或完整分钟预算。** 对应收据明确写明 real_episodes=0：[adjacency:2–6](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/structural_check.json:2)、[Huber:2–7](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/solver_check.json:2)、[color:2–5](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/path_check.json:2)。下面将已有收据、数学推导、合成构造与未验证事项分开。

## 1. 参考内部相邻关系

### 公式与实现

**一致。** 参考纯前景聚成最多8模式，R/Q按最近模式硬分配；参考四邻边的无序模式对计数加入一条置换零分布先验，取裁剪 log(P/P0)。查询区域奖励是平均内部边势减去该区域模式计数的置换期望，再乘0.25×区域大小。实现准确对应 [method:12–18](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/method.json:12)、[RA:39–50](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:39)、[RA:93–118](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:93)、[RA:177–188](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:177)。

**精确解码的是受限树目标。** F强制整棵子树为前景，G至少含一个背景，三项递推排除了同时全前景的子节点组合；每个最大全前景子树奖励只计一次。32个六叶目标各枚举64个掩码，核对MAP能量、完整 traceback 和 max-marginal，误差3.55e-15。因此支持这个指定目标的求解正确性，不能扩展为任意连通区域全局最优或IoU最优。[RA:121–162](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:121)、[check_structure:31–62](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/check_structure.py:31)、[structural:4–7](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/structural_check.json:4)。主输出是token MAP最近邻展开；双线性 max-marginal 是另一控制，不能混用两者成绩。[runner:55–64](/Users/yang/projects/CVPR2027/scripts/run_reference_adjacency.py:55)。

### 合成证据实际覆盖

- **已测合成事实：** 4×4树的31个区域内部边直方图逐区域枚举吻合；等模式计数的条纹与棋盘格有不同奖励；4096×1024随机单位特征的完整predict输出有限；strength=0退化为原基础场及其token阈值。[check_structure:63–110](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/check_structure.py:63)。
- **检查弱点：** `permutation_null_error`原检查比较同一个函数在(1,2)与(2,)形状下的点积，不是独立枚举置换期望。因此该字段的0误差本身不是独立证明。[check_structure:96–99](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/check_structure.py:96)。本轮标准库枚举2×3网格上3/3标签的20种排列，对固定势[1.3,-0.7,0.4]计算平均中心化分数，得到−1.6653e-17；这补充一个有限独立合成例，普遍结论仍来自每条边的无放回抽样概率。[RA:39–50](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:39)。
- **尚未证明参考身份必要性：** 条纹同模式边占比104/112≈0.92857，棋盘格为0。无需学习参考配对关系的“偏好同模式相邻”也能区分该例；现有例只证明相邻统计超出模式计数，不证明参考特有关系或最终完整分割收益。[structural:8–34](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/structural_check.json:8)。该构造直接调用势/奖励，并未使用实际聚类、树与完整decode构成两个最终身份输出。[check_structure:78–95](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/check_structure.py:78)。

### 身份信息与对照

**解释：** 它保留唯一参考内部的模式配对结构，确实比单纯模式占用多一个量；但只统计FF边，没有参考BG与跨标签边。Q的每个token即使与所有FG中心都不相似，也会被argmax强行分到FG模式，没有绝对相似度拒绝条件，因此相邻奖励不能单独保证类别身份。[RA:93–108](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:93)、[RA:180–182](/Users/yang/projects/CVPR2027/src/ics/methods/reference_adjacency.py:180)。形变、同结构干扰、树碎片逃避负奖励都是合同已承认的反例。[method:27–32](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/method.json:27)。

同读出MEAN与双线性控制已真正接入，共用入口目前只输出这些读出控制；“模式比例/置换”写在method卡片却没有接成完整比较行。[bundle:77–86](/Users/yang/projects/CVPR2027/src/ics/methods/prepared_cpu_bundle.py:77)、[method:45](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/method.json:45)。**强简单对照不足：** 应在任何未来资源授权下，保持字典、树、预算及读出一致，比较完整参考势与通用同模式邻接势、破坏参考配对但保留边缘的势；这些是控制，不是新增方法。判断来源是上述通用规则已能解开唯一身份区分例，而非预期它一定追平。

## 2. Huber图求解

### 公式与实现

**一致。** boxed目标的fidelity与同MEAN图保持固定，边项从0.5d²变为Huber h_0.05(d)。对偶共轭为p²/(2λw)加区间限制|p|≤λwδ；代码prox的除法与裁剪准确。primal对[0,1]的投影、保守 incidence 步长、可行dual下界和强凸误差界也匹配合同。[method:14–22](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/method.json:14)、[HG:93–109](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:93)、[HG:127–153](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:127)。certificate保证优化误差；`certified_token_threshold_fraction`只针对粗token符号，不能当作真实分类正确率或完整插值mask准确率。[HG:148–153](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:148)、[render:84–91](/Users/yang/projects/CVPR2027/src/ics/experiment.py:84)。

### 合成证据及简单替代

**已有实测合成证据：** 16个六节点boxed问题与独立L-BFGS-B优化器比较，最大坐标差2.44e-6，且独立能量位于primal/dual界之间；全二次分支、孤立点、λ=0和未收敛拒绝已检查。合成MEAN输入重建与现有producer最大场差2.98e-8；真实producer绑定尚未验证。[check_solver:16–57](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/check_solver.py:16)、[check_solver:71–82](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/check_solver.py:71)、[raw_input_check:7–17](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/raw_input_check.json:7)。

**合成星图支持“饱和可保住高源分数”，无法证明Huber必要。** 中心y=.9、六叶y=.1、每边w=.05：Huber得到[.66,.14×6]，默认λ=16二次解中心降至.31818。[solver_check:8–43](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/solver_check.json:8)。本轮数学小计算发现二次λ=20/13≈1.53846，每边c=1/13也精确得到[.66,.14×6]。这是该合成对象的代数控制，**不是**建议在真实标签上选该参数，更不是新方法；说明当前星图不能区分鲁棒饱和与较弱平滑。未来比较需要预先固定的较弱二次平滑控制，或在同一合成对象同时包含需保留的大对比与需抹平的小对比，再检查二者能否由单一二次强度兼顾。[HG:93–97](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:93)、[HG:168–180](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:168)。

**身份判断：没有新增类别可观测量。** y、a、图完全既有；高置信错误也可能被更好保住，多个坏边仍可积累较大作用力。它应作为传播误差模型候选判断，不能以收敛证书解释“语义选错但内部一致”的修复。[HG:24–66](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:24)、[method:34–41](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/method.json:34)。boxed与unboxed二次、pregraph控制已接入，足以拆分box效应及不平滑；仍不足以排除简单减弱平滑。[bundle:88–96](/Users/yang/projects/CVPR2027/src/ics/methods/prepared_cpu_bundle.py:88)。

## 3. 查询颜色瓶颈路径

### 公式与实现

**一致。** RGB四邻边cost=||ΔRGB||/√3，两次多源min/max Dijkstra产生dF,dB，g=dB/(dF+dB)，零/零沿用base，最终z=.5base+.5g。half-pixel resize、base裁剪、缺任一种高置信种子弃权，与method卡片一致。[method:12–19](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/method.json:12)、[CB:23–55](/Users/yang/projects/CVPR2027/src/ics/methods/color_bottleneck.py:23)、[CB:73–94](/Users/yang/projects/CVPR2027/src/ics/methods/color_bottleneck.py:73)、[CB:101–127](/Users/yang/projects/CVPR2027/src/ics/methods/color_bottleneck.py:101)。

**现有合成覆盖充分用于路径正确性：** 24个七节点图与独立min/max Floyd闭包逐值一致，外加低屏障长路径、两色补全/删除、互补、恒定RGB、中性与缺种子、渐变歧义及resize核对；这不含自然物体光度/种子身份的证据。[check_paths:15–76](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/check_paths.py:15)、[path_check:4–5](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/path_check.json:4)。双worker合成入口只证明重复出现保留、未打开sentinel及1024打包输出等工程性质。[runner_check:2–8](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/runner_check.json:2)。

### 新信息、硬限制与对照

**新增的是光度屏障，并无参考条件身份。** 颜色不能区分同色目标与错误前景种子。数学上每个RGB128前景种子有dF=0：若dB>0则g=1，z≥.875；若dB=0则g=base，z=base≥.75。背景种子对称。故规则不删除该格上的高置信错误，也不恢复被判为高置信背景的格点；允许改动弱证据邻域、补边界，也可能扩散错误种子。这个结论针对128格，不声称最终插值边界逐像素全不变。[CB:111–127](/Users/yang/projects/CVPR2027/src/ics/methods/color_bottleneck.py:111)、[method:33–39](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/method.json:33)。

原base、同resize、裁剪resize及color-only已真正输出，可排除纯读出变化；**强光度对照仍缺完整行**：method声明应比较既有RGB CRF，但共用分支未直接产生该行（入口可从外部封存控制导入）。因此还不能声称瓶颈路径胜过普通光度后处理；同种子的一项预先固定简单边界传播控制也能直接检验min/max路径的必要性。[bundle:97–106](/Users/yang/projects/CVPR2027/src/ics/methods/prepared_cpu_bundle.py:97)、[method:50](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/method.json:50)、[score:375–381](/Users/yang/projects/CVPR2027/scripts/run_cpu_feature_candidates.py:375)。实际RGB路径仍未绑定，来源需同一次canonical query；当前load_rgb提供1024→128两次PIL双线性路径或既有RGB128包，但包的producer一致性不能仅靠shape/hash证明。[bundle:35–49](/Users/yang/projects/CVPR2027/src/ics/methods/prepared_cpu_bundle.py:35)、[PLAN:47–50](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:47)。

## 成本可预测到哪一步

| 方法 | 现有合成局部时间 | 可推导结构 | 不能据此推断 |
|---|---:|---|---|
| 相邻关系 | 0.064609416秒 | 分配O((Nr+Nq)KD)，树含O(ED+E log E)，区域直方图O(NK²)，decode O(N) | 真实类别收益、完整队列分钟数 |
| Huber | 0.017687459秒，50迭代 | solver每步O(N+E)，最多20000迭代；实际迭代依图/权重/fidelity | 所有真实图都50迭代、重建/I/O免费 |
| 颜色 | 0.062948917秒 | N=16384,E=32512，两次O((N+E)log N) heap pass | 图像绑定、读取、完整渲染/评分总成本 |

时间是旧合成收据的观测，不是本轮benchmark：[adjacency:37–42](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/structural_check.json:37)、[Huber:55–70](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/huber_preparation_01a1100b/solver_check.json:55)、[color:77–93](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/color_bottleneck_preparation_01a1100b/path_check.json:77)。本轮仅乘法：600个独立顺序局部调用分别约38.77、10.61、37.77秒；这不是600例完整耗时预测，也不保证并行线性缩放。三项单独约87.15秒尚未含共享宿主、读写和控制。

**成本卡片遗漏需指出：** 相邻树不仅排序，也计算四邻特征距离；其64²网格8064条边、1024维，单个`q[a]`特征gather约31.5MiB，两个gather同时存在时约63MiB，另有特征与直方图，不等于整进程峰值。当前method只将树写为O(E log E)遗漏O(ED)。[occupancy:83–91](/Users/yang/projects/CVPR2027/src/ics/methods/reference_occupancy.py:83)、[adjacency_method:34–43](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/method.json:34)。共享入口每次重建MEAN时还有4096²×1024密集相似度乘法，单个FP32相似度矩阵64MiB；候选纯CPU无新增encoder不代表宿主或完整部署免费。[HG:48–59](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:48)、[bundle:58–71](/Users/yang/projects/CVPR2027/src/ics/methods/prepared_cpu_bundle.py:58)。

## 与Pro五法的关系

用户报告仅作为设计假设来源，不认证原创性或效果。报告自身也要求同信息控制、强完整基线与跨域检验：[Pro:153–158](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:153)。

| Pro方法 | 第1–3项关系 | 精确区别与有限判断 |
|---|---|---|
| M1 同查询上下文续算 | 三者都没有该可观测量 | M1使用原生中间状态共享续算；末层缓存上的统计/优化不能声称实现了它。[Pro:270–272](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:270) |
| M2 参考标签关系 | adjacency最接近 | adjacency只学FG内部模式邻接并对区域置换中心化；M2学FF/FB/BF/BB四状态角色对cross-ratio，抵消可分一元，再全图标签推断。不能把前者算为已准备M2，也不能把signed区域奖励称为负标签边。[Pro:398–437](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:398) |
| M3 留出角色预测 | adjacency共享树区域/组合约束思路 | 现有adjacency无被留出角色及预测验证；精确DP不证明类别组成可预测。[Pro:147](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:147) |
| M4 环境抑制 | color使用Q环境外观，HG处理图响应 | 没有同参考内容的成对移植与环境散布测量，故不是M4简化实现。[Pro:148](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:148) |
| M5 消息外推 | 无对应 | HG在末层图抑制大场差的作用力，M5改变冻结后缀中的区域外→内消息；机制和成本对象不同。[Pro:779–783](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:779) |

## 有限后续建议

1. **当前离线优先补相邻关系的辨识证据。** 若主代理选择继续本地准备，只需一个完整单位特征可实现例，让通用同模式邻接失败而参考特定关系成功，并固定相同树/读出。它直接解决当前条纹例的欠缺；不扩展参数搜索，不计控制为方法。[check_structure:78–110](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/adjacency_preparation_01a1100b/check_structure.py:78)。
2. **Huber保留为误差模型候选，颜色保留为边界补全候选。** Huber需要同一小对象上大小对比的共同约束来排除弱二次平滑；颜色已经能明确预判不修复128格高置信种子身份错误。继续写更多类似正例不会解决这两个缺口。[HG:93–97](/Users/yang/projects/CVPR2027/src/ics/methods/huber_graph.py:93)、[CB:111–127](/Users/yang/projects/CVPR2027/src/ics/methods/color_bottleneck.py:111)。
3. **M2可作为另外一项明确假设讨论；不能自动替换九候选或开跑。** 它比adjacency多了BG/跨标签关系，可直接用zero/positive/absolute/pair-independent/block有限控制检验；报告推荐它优先是研究判断，未有本地实现/真例证据。M1/M4/M5需要当前缓存之外的新计算，当前关机范围内只能审数学/资产缺项。[Pro:494–519](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:494)、[Pro:923–930](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:923)、[PLAN:5–6](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:5)。
4. **未来若获真实缓存运行授权，先绑定输入与producer、一次核基线，再固定有限队列。** 目前baseline重建/双worker只在合成上核对；四类增删计数代码目前相对native，归因三候选的增量还需相对其MEAN/二次直接基础行。当前评分只处理packed1024，不能将结果改称Pro主终点的原尺寸mIoU；若采用Pro合同，应显式补原尺寸绑定与读出，不能悄然换指标。[PLAN:47–50](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:47)、[score:371–392](/Users/yang/projects/CVPR2027/scripts/run_cpu_feature_candidates.py:371)、[unpack:94–97](/Users/yang/projects/CVPR2027/src/ics/experiment.py:94)、[Pro:1230–1241](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:1230)。

以上是有限建议，未修改任何候选合同或计划，未执行新增验证。已有工具正确性与合成区分成立的范围应保留；自然DINO分布、完整收益、同信息必要性、真实成本及原创性仍未验证。
