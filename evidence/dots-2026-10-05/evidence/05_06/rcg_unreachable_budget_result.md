# RCG深漏检：固定rank修正上界与图可达预算

实测不支持用“整个图分量无正证据”解释大部分深漏检。全120例深漏检1,145,941像素中，73.82%位于自身unary无论guide多好也不能过.5的token；但是只有0.107%落在实际unary全负的图分量。绝大多数已经与至少一个正确的正证据token连在同一图分量，缺的是有选择的语义关联，不能直接归因为图完全断开。

## 固定公式的精确上界

对N=4096、R(x)=(average_rank−.5)/N，有R(g)_i≤1−1/(2N)。所以

    u_i = s_i + .5(R(g)_i−R(s)_i)
        ≤ s_i + .5(1−1/(2N)−R(s)_i)。

因此，若s_i≤R(s)_i/2+1/(4N)，任何guide都不能让这个位置自身的u_i>.5。特别是s_i=0的token必然满足。这里“任何guide”是固定rank修正规则下的上界，不是调整alpha后的结论，也不是说最终graph/插值场不能救它。每个token同时假设自己guide排第一仅用于保守逐点上界，不意味着这个guide排序能同时实现。

图解z=(A+16L)⁻¹Au在各图分量内是u的非负平均。实际分量max(u)≤.5的token对任意lambda都不能成为FG。最终像素还受bilinear跨分量混合影响；本报告用bilinear(component_max_u)≤.5−数值余量给出保守像素证书。这是充分证书覆盖的预算，不是所有不可恢复像素的完整充要分类。

## 实测预算

深漏检严格沿用用户release：GT前景、native最终mask为背景，且distance_transform_edt(~native)>48。以下是像素质量计数，不是类mIoU，也不是episode等权比例。token footprint均按原16×16格统计，不假装等于最终renderer。

|cohort|deep-FN像素|自身u的guide上界≤.5，token footprint|实际全负图分量，token footprint|含正负unary的图分量|bilinear后严格证书|
|---|---:|---:|---:|---:|---:|
|old20|356,325|312,672 (87.7491%)|0 (0.0000%)|356,325 (100.0000%)|0 (0.0000%)|
|new40|374,730|327,896 (87.5019%)|376 (0.1003%)|374,354 (99.8997%)|26 (0.0069%)|
|new60|414,886|205,370 (49.5003%)|855 (0.2061%)|414,031 (99.7939%)|30 (0.0072%)|
|pooled120|1,145,941|845,938 (73.8204%)|1,231 (0.1074%)|1,144,710 (99.8926%)|56 (0.0049%)|

全120的更强、guide-independent“整个分量即使换任意guide仍上界≤.5”只覆盖251个deep-FN footprint像素；计入bilinear后的保守证书覆盖16像素。

RCG最终恢复13,211个deep-FN。自身unary上界不够高的845,938像素中，仍有739个被图/插值恢复；这是不能把unary约束偷换成最终mask硬上限的直接例子。实际全负分量footprint中的1,231个deep-FN均未恢复；保守像素证书没有与冻结输出产生矛盾。

新60单列：414,886个deep-FN中，guide上界footprint占49.50%，实际全负分量仅855个（0.2061%），最终像素证书30个（0.0072%）。这不是新的独立确认批次，仍是已暴露数据的机制诊断。

## 混合分量并不等于“再加大lambda就好”

只看到分量内存在正unary，可能仍是错误对象在提供正证据。因此追加了仅用冻结u/分量与评价GT的核查：全120有1,144,704个deep-FN像素（99.8921%）所在分量确实含一个u>.5且16×16全部为真实前景的token。没有deep-FN落在“分量有正u但全部正u都是GT背景”的混合分量里。

这表明正确支持在图连通意义上几乎处处存在，但不说明它有足够影响权重、与deep-FN属于同一个实例，或不会同时扩散到背景。所有包含deep-FN的分量，其A加权unary均值都≤.5；lambda→∞的分数趋于该均值。增大lambda不能被解释为自动获取缺失目标，更不能由连通性推出语义身份。

对当前数据更贴切的研究问题是：什么可观测的reference条件对象结构，能够把已有正确支持关联到同一个完整目标，同时隔离同图分量背景？完整object-hypothesis bank如果能做到这种分离，是一种候选观测；但proposal存在只说明动作空间，必须另测合法身份分数。不能从bank的GT oracle直接给它标签。

## 如何保证这是诊断而非新方法run

release只存最终z，没有u或component IDs。本脚本从已经授权的同q/r/cov/score重新构造唯一固定公式的u、A、W与连通分量；没有调用predict或CG，没有生成候选方法mask，没有扫描参数。随后只读取release封存z/mask与GT做预算统计。

120例的边数、分量数、孤立点数全部与release audits一致；冻结z代回重建方程的最大相对残差5.3873e−7，残差给出的解∞范数误差上界小于9.0e−6，已用于像素证书余量。这是相同recipe和冻结输出的一致性证据，不声称仅由边数相同就证明跨PyTorch版本的每条边逐位一致。总耗时140.28秒，CPU单线程，无encoder、无大特征复制。

- 完整脚本：rcg_unreachable_budget.py；补充正确positive witness诊断：rcg_positive_witness_budget.py。
- 逐例数据、汇总：rcg_unreachable_budget/episodes.jsonl、report.json。
- 小型冻结派生字段：rcg_unreachable_budget/fields/，仅保存s、u、上界、a、component IDs与分量极值/均值。
- 补充核查：rcg_unreachable_budget/positive_witness_budget.json。
- 数学机制和旧PPR/cut差异：rcg_diffusion_mechanism_comparison.md。
