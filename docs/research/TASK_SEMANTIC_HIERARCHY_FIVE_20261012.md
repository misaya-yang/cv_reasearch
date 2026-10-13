# 任务定义语义粒度：五域现有特征的完整方法检验

2026-10-12，本chat授权的继续实验；当前阶段目标是同一算法在COCO、LVIS、PASCAL-Part、PACO-Part、SUIM五域超过完整FoRIS及已有匹配强控制，最终十域目标保留。主raw目录实测339G。本轮只读已有真实O24，不编码、不生成RGB新视图、不训练主干，不消费FoRIS／MEAN／S预测作为算法输入。

## 为什么选这个问题

当前执行状态：COMPLETE_FIXED_HIERARCHY_REJECTED。原评分前规格的快照已随config封存；本构造已完成1000任务、8000Mask及独立验证，五域完整输出均低FoRIS/density，未通过原标准。[完整结果](TASK_SEMANTIC_HIERARCHY_FIVE_RESULTS_20261012.md)。以下保留冻结设计的研究假设，不再表示待执行队列。

已有效的query组织能展开目标，但固定语义簇会把部件并入父物体，或把目标富集簇整体广播为细结构前景；同Q两真实任务对同一扩张动作有相反结果。参考自身更易拟合又不保证Q正确，旧query层级有好GT候选却选不出来。因此检验的具体假设是：在固定预训练语义度量下，**参考Mask中的正负混合情况可以决定共享区域何时停止、何时继续细分**，比固定粒度更好地迁移任务归属和范围。

这不是“源fit越好越可迁移”的假设。分区由R/Q未标注特征共同形成，Mask只影响分区上的角色判断及停止／分裂混合；完整Q结果、排序、完整阈值诊断会直接检验迁移是否成立。若只改善源编码分数或候选上界而Q不改善，拒绝本构造。

## 固定算法

1. 全部4096个R/Q原生FP32 O24与旧payload/tensor/RGB/变换身份匹配。沿用现有reference-ridge的预训练去位置准备：先单位化，按R的nearest64 Mask与Q均值的semantic score<.8决定是否施加既有native U500投影，再单位化。所有臂共享此准备；没有新的特征学习或按数据集选择。
2. 在联合8192个单位特征上建7层完整二叉球面二均值层级，最大128叶。每个父节点用“离均值最远→离首种子最远”初始化，10次Lloyd，平票归左，空节点保留。不输入坐标、RGB、Mask或旧预测。不扫描树深／步数。
3. Mask nearest到1024后16×16平均，得到每个R patch的完整软质量c。每节点累积F=sum(c)、B=sum(1-c)，不丢弃硬BG里的目标质量，不采样128个BG位置。
4. π=R的FG面积比例，Beta先验质量固定.5、均值π。节点p=(F+.5π)/(F+B+.5)，无R成员的节点继承父估计。节点编码分数为logBeta(F+.5π,B+.5(1-π))-logBeta(.5π,.5(1-π))。从叶向上递归E=logaddexp(local,E_left+E_right)-log2，给出停止／继续细分混合；再按每个Q patch所属路径混合p。
5. 以共同等角色先验读出p_balanced=p(1-π)/(p(1-π)+(1-p)π)，field=p_balanced-.5。原图／CLI1024各自FP64 bilinear，严格>0。连续场及完整Mask均保存，无query minmax、面积预测或数据集分支。原生RGB CRF在当前Python未可直接导入，不把其它CRF替代成官方；本轮输出使用上面的固定完整双帧读出，并同时检查完整level-set诊断，区分任务排序与后处理局限。

停止／分裂的编码混合是经典树模型平均；[CTW分类先例](https://research.tue.nl/en/publications/classification-with-the-ctw-algorithm/)和[Bayesian Context Trees原文](https://academic.oup.com/jrsssb/article/84/4/1287/7073257)均先于本实验。当前二叉语义特征层级及软Mask计数是特定构造，不能把CTW移植本身称原创。相关patch计数不提供独立样本的后验置信保证，源粒度的可迁移性正是待验证问题。

## 完整控制与队列

主臂adaptive；完整控制flat使用同树最细叶、不作停止／分裂混合；density是同预处理、全软面积、固定τ=.07的FG/BG log-mean-exp密度，sigmoid后减.5；null沿相同固定树，把实际1024参考Mask roll512/512后重新编码任务标签，面积相同但可能与目标重叠。null不重新改变APD／树，所以只是任务标签分支控制，不冒充整个原FoRIS的Mask-only干预。

已有6200 atlas中，COCO200全取，其它域分别沿既有顺序取首200，PASCAL/SUIM只取native2000 cohort，共1000。选择不读GT、旧分数或正负案例标签；一次冻结五域，全部8000张完整双帧Mask封存后才评分，不按某域成绩改其它域公式。五域均是已曝光开发，既非完整官方测试池，也非未见确认。

评分：配对完整FoRIS、MEAN、原R0、N；每域分列实际观察class/fold及全部声明槽位mIoU，不跨域平均。CLI为当前阶段主端点，原图另报；全图与FoRIS预测内的GT排序、原图gross四编辑、完整GT level-set诊断、相对flat/density差一并解释。GT最优阈值不变成算法预测。RQ照片关联组的逐例均值差区间只解释开发稳定性，不冒充官方class mIoU区间。

若adaptive未在五域共同改善同粒度／同信息控制，并未显示可兑现的任务排序／完整输出优势，不扩6200、不扫树深／prior／层，也不把按域选择不同臂变成方法。即使1000成功，仍需预先固定规则后跑完整6200、独立确认及与LVIS原S等额外真实观察控制做精确同成员比较。

入口scripts/run_task_semantic_hierarchy.py；算法src/ics/methods/task_semantic_hierarchy.py；独立输出../cv_data/a/task_semantic_hierarchy_five1000_20261012/。M4 MPS执行分区、投影、相似度；CPU读取／编码混合／输出和封存后评价。此规格冻结在评分前，科学结论与原创性均未成立。
