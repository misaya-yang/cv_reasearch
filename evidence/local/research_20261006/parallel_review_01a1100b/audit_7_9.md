# 第7—9项CPU候选审查（2026-10-06）

结论：三项均有可核对的完整推断定义和合成检查，但真实新例均为0。第7项保留为有条件的查询外观适配候选；第8项更适合作为低成本分布/阈值控制；第9项保留为压缩参考分类器对照，尚无二次核不可替代的证据。没有依据承诺其中任一项涨分、原创性或完整队列分钟数。[docs/research/PLAN.md:32-50；下述三份method.json的checks与novelty]

本次范围：只读本地源代码、候选契约、指定合成检查及Pro报告所引三类负结果的对应账本条目；做一次小型只读CPU复核，写本文件。未使用服务器、SSH、GPU、付费资源、技能；未改算法、PLAN、STATUS或Pro原文。当前关机限制来自docs/research/PLAN.md:3-6,51-54；外部建议不是运行授权，见AGENTS.md:3-8。

## 共同边界

- **实测（合成）**：三个双worker收据均记录主动分支、query GT哨兵未读、1024掩码封存及real_episodes=0；这些不是自然分割实测。证据：recurrence_preparation_01a1100b/runner_check.json:2-13、prior_shift_preparation_01a1100b/runner_check.json:2-13、quadratic_preparation_01a1100b/runner_check.json:2-12（本文件中的候选证据相对根目录均为evidence/local/research_20261006/）。
- **代码观察**：第7项修正MEAN；第8、9项主动时完全替换MEAN，只有弃权时保留MEAN。共用入口仍无条件重建MEAN图，所以主动算法无需图求解不等于当前批量入口省掉该成本。[src/ics/methods/query_recurrence.py:103-120；reference_prior_shift.py:89-130；reference_quadratic.py:63-102；prepared_cpu_bundle.py:58-73]
- **完整输出观察**：三项及控制均以浮点场双线性到1024再严格>0.5；合成核心报告多为64×64 token检查，runner只证明1024输出可封存。不能把token成功数作为完整1024 mIoU，亦不能把token判别不等式当作每个细像素判别式。[src/ics/methods/prepared_cpu_bundle.py:144-169；src/ics/experiment.py:84-91；三份method.json的prediction]
- **尚未知**：真实缓存/清单、版本、照片关联分组、暴露状态、原图尺寸绑定及同协议完整FoRIS比较均未由这些候选检查证明。当前PLAN也明确实际绑定仍缺。[docs/research/PLAN.md:47-50,238-243]

## 7. query_recurrence：支持星的查询外观修正

### 拟合/选择主张核对

**代码与推导一致**：基础场>=0.75的四连通块至少8格，按基础置信均值、大小、标签排序，仅保留32块再构造描述子；参考前景均值余弦>=0.5才有资格。每个根搜集与根余弦>=0.9的合格种子，至少2个且包含根；最大数量优先，参考相似度和基础置信只打破平局。[src/ics/methods/query_recurrence.py:19-25,35-51,79-102]

这是确定性星选择，既非最大团，也非最优类别推断或概率检验。成员可以互相不满足0.9；本次小计算构造根(1,0)、两叶(0.9,±sqrt(0.19))，选中三者，叶间余弦0.62，符合现有契约而非实现bug。“minimum_peers=2”实际要求根外至少一块，不能表述成两次独立支持。[src/ics/methods/query_recurrence.py:36-43；recurrence_preparation_01a1100b/method.json:12-14；本文件末尾只读复核]

完整场为clip(base+0.5*(q·m−q·r),0,1)。等权的是**组件描述子**，面积不同也同票；参考均值不是MEAN完整场的可分解项，因此修正是启发式差分，无法解释成替换某个已知的线性项。代码没有置信自校准或事后联合优化。[src/ics/methods/query_recurrence.py:105-120；recurrence_preparation_01a1100b/method.json:5,14,38]

### 保留价值与实际缺口

**实测（合成）**：当前规则保留192真格，恢复64弱格并删64孤立干扰；所有种子均值仍留干扰、参考最佳单种子漏弱目标。它证明查询重复模式能在这个构造中帮助完整token输出。[recurrence_preparation_01a1100b/recurrence_check.json:4-11；check_recurrence.py:15-47]

**实测（合成反例）**：3个错类种子胜过2个真类，漏目标不恢复、错区不删除；单个强实例、全弱实例弃权。因此“重复=类别正确”“足够实例自行纠错”已被此固定方法反例否定。[recurrence_preparation_01a1100b/recurrence_check.json:12-20；check_recurrence.py:77-89]

**真实泛化风险（待测）**：一物多碎块和多物连块都破坏实例解释；相同场景错误类别也可能重复，参考单均值允许近类通过门控，32块截断及平局标签顺序可改变选择。即便原型中心正确，全图余弦修正仍可能增添附着背景或误删不同姿态目标，没有extent证据。[recurrence_preparation_01a1100b/method.json:31-38；src/ics/methods/query_recurrence.py:86-87,109]

**强简单控制**：保留原MEAN、clipped MEAN、所有合格种子均值和参考最佳单种子四行；后三种适配使用相同差分权重和读出。当前正例支持星筛选优于两项种子控制，但没有真实证据支持。[src/ics/methods/prepared_cpu_bundle.py:72-73,144-150；src/ics/methods/query_recurrence.py:111-115]

**有限下一动作建议**：保留固定v1。若以后用户授权真实缓存评测，只做一次预先固定批次的完整配对比较，按无/单/多合格种子及弃权情况报告四类增删，尤其重复错区和同类异貌；若不胜所有种子均值与同输入MEAN，收束星筛选贡献。现在无需再造合成正例或扫阈值。此为建议，不是新增待执行队列。

## 8. reference_prior_shift：一维混合比例与完整后验场

### 概率/优化主张核对

**推导正确**：固定正密度f_i,b_i，L(pi)=mean log(b_i+pi(f_i−b_i))，二阶导为−mean[(f_i−b_i)^2/(b_i+pi(f_i−b_i))^2]；至少一项不同则严格凹。端点导数确定边界解，内部根二分，与代码一致。严格凹只支持这个给定混合模型的解，不能证明语义正确或真实比例。[src/ics/methods/reference_prior_shift.py:29-64；prior_shift_preparation_01a1100b/method.json:13,21]

**代码观察**：参考纯FG/BG各至少8格；同一参考样本先构造单位均值差，再拟合64 bin、带宽0.1、1%均匀质量的密度。查询只压缩成一个FG/BG均值差分数；估计全图像素比例pi∈[1/N,1−1/N]，不是对象数。posterior=pi*f/(pi*f+(1−pi)*b)仅在标签密度迁移假设下有类别后验意义；参考重代入不是独立校准。[src/ics/methods/reference_prior_shift.py:93-130；prior_shift_preparation_01a1100b/method.json:5,8,10-15]

**更窄的机制判断（推导）**：固定f/b后，pi只移动同一似然比场的阈值f/b>(1−pi)/pi；它不能给相同得分的真目标和干扰不同身份。不应描述成新增的对象级识别或完整范围模型。[prior_shift_preparation_01a1100b/method.json:19,22,38；src/ics/methods/reference_prior_shift.py:117-129]

### 保留价值与实际缺口

**实测（合成/数学检查）**：32例与独立标量优化器最大pi差1.14e−8；精确期望混合恢复0.3。正例保留256目标并删800干扰，平衡先验及直接margin保留两者，参考比例保留零目标。[prior_shift_preparation_01a1100b/prior_shift_check.json:4-24]

**实测（合成反例）**：正例pi=0.08784，真实token比例0.0625；参考FG由两种外观构成而查询目标只用其中一种，所以该正例本身不是纯prior shift。缩小到64格目标后pi=0.00430，输出全空。[prior_shift_preparation_01a1100b/prior_shift_check.json:22-29；check_prior_shift.py:23-40；method.json:30-31]

**真实泛化风险（待测）**：参考/查询背景类别变化、参考均值坍缩、密度尾部与小样本平滑偏差、局部背景异质性均可能被pi误解释为面积变化。空间相关token不能支持独立样本的校准区间。凹优化正确已经通过，核心未证的是密度可迁移。[prior_shift_preparation_01a1100b/method.json:35-42]

**强简单控制**：MEAN/clipped MEAN、同密度平衡pi、同密度参考比例、原始FG−BG margin足以分别比较替换宿主、密度读出及拟合pi；在真实完整输出中应同时保留，不能只挑pi赢的控制。合成正例已显示密度模型+拟合比例具有条件判别价值，但非自然泛化。[src/ics/methods/reference_prior_shift.py:120-122；prepared_cpu_bundle.py:72-73,152-159]

**有限下一动作建议**：优先作为低成本固定分布控制而非原创主线。授权后一次完整比较，保留天然小目标和缺BG弃权，报告拟合比例/边界解、delete TP/add FP及同密度平衡行差值；GT比例仅评分后作诊断，不据其改带宽、先验边界或主方法。若收益仅在大目标而小目标损失抵消，结束v1，不用优化精度掩盖迁移失败。

## 9. reference_quadratic：压缩参考的带方向二次核ridge

### 拟合/核优化主张核对

**推导与代码一致**：纯FG/BG各压到最多16球面模式，各类总权重0.5；k(x,a)=((1+x·a)/2)^2，对应phi(x)=[1,sqrt(2)x,vec(xx^T)]/2，是常数+线性+二次核。最小化Σw_j(g(a_j)−y_j)^2+0.01||g||²_H，系数解(K+0.01diag(1/w))alpha=y；PSD核加正对角得到正定系统。它是核岭回归，不是独立标签校准。[src/ics/methods/reference_quadratic.py:28-41,72-89；quadratic_preparation_01a1100b/method.json:10-13]

**压缩范围需讲清**：优化的是球面簇中心训练问题，而非4096原参考token的二次回归；类内簇质量只作为权重，球面均值会丢失簇内变化，压缩后的非线性损失不能等同于原token损失。簇数/5次Lloyd为固定近似，无全局最优聚类主张。[src/ics/methods/reference_occupancy.py:42-74；src/ics/methods/reference_quadratic.py:72-77；quadratic_preparation_01a1100b/method.json:45]

**概率与范围主张**：clip(0.5+0.5*g,0,1)只是读出；token门槛g>0，输出无对象连通/结构保证，也没有训练query比例。主动分支舍弃正确MEAN的风险是方法本身，而非核求解精度问题。[src/ics/methods/reference_quadratic.py:84-102；quadratic_preparation_01a1100b/method.json:27,43,46-47]

### 保留价值与实际缺口

**实测（合成/数学检查）**：24个小问题显式多项式特征primal与kernel分数最大差3.94e−15。正例保留64目标且超过线性/最近匹配/核均值；但子空间投影也完全解决，故未证明核学习的独有价值。带方向核修复x与−x别名，最近匹配同样解决该极性例。[quadratic_preparation_01a1100b/quadratic_check.json:4-20；check_quadratic.py:71-88]

**实测（合成反例）**：目标外观偏向参考BG后，即使base保留64格真目标，最终二次分类器全删。参考压缩和二次可表达性并未免除跨实例迁移条件。[quadratic_preparation_01a1100b/quadratic_check.json:21-24；check_quadratic.py:89-96]

**强简单控制充分且必要**：同一锚点/权重/ridge的线性核，齐次平方核、核均值、最近FG−BG余弦、两类子空间能量，另有MEAN/clipped MEAN；已有正例已被子空间追平，不应删掉这一控制。各核self-kernel=1使名义ridge尺度一致，但不意味着不同核容量完全相同。[src/ics/methods/reference_quadratic.py:78-97；quadratic_preparation_01a1100b/method.json:14,19；prepared_cpu_bundle.py:161-169]

**有限下一动作建议**：作为固定压缩分类器比较行保留，优先级低于有额外查询证据的第7项。授权后一次完整配对比较全部控制，重点核对罕见部件/跨实例外观误删和add FP；如果线性/子空间追平，则采用更简单行或收束二次核主张。现阶段不用再扫ridge、锚点数，也不把更多参考拟合/更小残差算作证据。

## Pro历史提醒与当前候选：核对后适用的范围

Pro报告路径：`/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md`。其113、119、122行分别提出ridge、中心/extent、原型扩展提醒；125行明确只否定具体已测构造。

| 提醒 | 本地对应事实及协议 | 当前关系与不可直接推断之处 |
|---|---|---|
| 完整向量reference ridge | RESULTS.md:154-167：暴露DEV241、1024无CRF、参考Gram均值特征值罚项；49.72对58.56，−8.84，改对5,835,294像素/改坏10,708,643 | 第9项是最多32模式+固定0.01二次核，训练对象/容量/正则不同，不能标成已测失败。但同一“参考拟合→跨query类别边界”是核心风险，合成外观漂移已显示该风险仍在。第8项也没有额外query语义标签能修复source mismatch。 |
| 正确类别中心不等于extent | RESULTS.md:264-275：query K32候选source验证+source IoU选切分，完整39.05对58.56；source切分面积中位2.14×GT，96/184选对目标例过合并>1.5× | 第7项用星重复筛选并修正MEAN，未重用旧source切分；第8项用后验阈值，第9项用回归sign，也不相同。但“找对中心”不能证明全图范围；必须验证三项完整增删而非描述子/参考拟合。 |
| 原型自训练/角色扩展漂移 | RESULTS.md:1061-1076：角色/reference guard+whole-query增补完整DEV241，−0.815733 [−1.258972,−0.289823]；明确不排除其他条件编辑 | 第7项是当前query自身预测产生外观支持，存在同类漂移风险，但没有迭代refit且以组件星选择，不是旧增补算法重复。其3错种子反胜的合成例是更直接的当前v1负证据。 |

补充：RESULTS.md:1546-1563的“一步ridge自训练69.48/70.68无输入增益”属于fresh600 token最佳切分排序诊断，不可混成上述完整DEV241成绩。Pro报告:939还提醒普通多原型/FG-BG KDE的思想已有近邻；本审查只据本地报告将原创性标为未知，没有重新查外部论文、也没有以报告意见替代实测。

## CPU预算核对

| 项目 | 已有合成算子计时 | 推导成本（N=4096,D=1024） | 当前缺口 |
|---|---|---|---|
| 7 recurrence | 0.01010秒，含两个适配控制 | O(ND+C²D)，C≤32；标记/排序和描述子扫描另计 | 同输入MEAN图、缓存I/O、1024读出、评分及完整峰值未计 |
| 8 prior shift | 0.01734秒，含三个控制 | O(ND+64²+64N)，最多64二分；精度通常34步 | 密度迁移未测，主动算子省图但共用入口仍算图 |
| 9 quadratic | 稀疏构造0.06294秒；密集32锚点0.18547秒，含五控制 | 压缩约5×N×16×D≈3.36e8乘积项；查询32锚点≈1.34e8乘积项，另有SVD/核小解 | 自然特征分布、完整I/O/图/读出、冷启动/峰值/部署编码均未知 |

计时来源：recurrence_check.json:26-27；prior_shift_check.json:64,70；quadratic_check.json:65,67-103。复杂度来源：三份method.json的cpu_cost及reference_occupancy.py:49-68。数字是旧收据中**本地合成算子实测**；乘积项是运算规模推导，不是CPU墙钟/全队列预算。原始两张FP32特征合计约32MiB只是数组大小下界，不是完整进程RSS。

共用入口设置worker×thread≤30且声明内存≤60GB，Linux下检查affinity/cgroup并监视所属worker RSS；设置隐藏CUDA且每worker线程有记录。[scripts/run_cpu_feature_candidates.py:42-46,176-195,290-325] MacOS上的linux_rss因无/proc返回0，不能把本地双worker成功当作60GB运行守卫有效性证明。[同文件:54-61] 当前资源OFF，不推出600例/5分钟或启动权限。

## 本次只读小复核

2026-10-06，`PYTHONDONTWRITEBYTECODE=1`、四个数学库线程变量均设1、`PYTHONPATH=src /usr/bin/python3`，仅导入三个检查脚本的fixture，未执行会覆盖原收据的main；直接调用现有predict/choose_star/fit_prior/solve_kernel。工具命令退出0，墙钟0.284秒。没有读真实GT或启动编码。

| 复核 | 本次输出 | 性质 |
|---|---:|---|
| 三节点支持星的最小两两余弦 | 0.6200000000000001 | 合成选择语义；证实星不是团 |
| 已知期望混合pi=0.3 | 0.29999999997286947 | 推导对象的数值检查 |
| 二次核显式特征与dual得分最大差 | 2.220446049250313e−16 | 四锚点数学检查 |
| recurrence正例输出格数 | 192 | 复核已有合成正例 |
| prior正例/小目标输出格数 | 256 / 0 | 复核已有合成正/反例 |
| quadratic正例/子空间/外观偏移输出格数 | 64 / 64 / 0 | 复核已有合成正例、简单控制和反例 |
| 真实新例 | 0 | 没有分割泛化结论 |

本次没有重跑双worker入口、原32/24全套优化检查或密集计时；这些仍是上面逐行引用的既有收据。保持三个固定v1定义，下一次能改变判断的证据是合法缓存上的完整配对输出，而非更漂亮的source拟合或合成数值残差。
