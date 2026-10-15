# E3 同一参考的真实观察与双角色信息

当前状态：用户指出猜想与局部试验的反复失败后，本会话撤销E3作为下一主线候选的安排。只保存了未验证源码，未运行prepare、produce或score，也没有E3实验产物。下面保留当时设计以说明为何曾提出它；不代表活动队列或已证明有前景的方法。

目标是实际构造合法单参考独立分割方法。E3承接[本会话推进计划](SINGLE_REFERENCE_SOLVE_PLAN_20261012.md)，使用已有同一参考的真实局部编码；不增加参考图，不读取query标签推理，不消费FoRIS输出，不新增编码。

## 已有证据与新问题

E1的固定13维线性区域读出没有提取出mean_c以外的通用排序增益。[E2固定200例诊断](../x1012s_tasks/E2.md)中，在中心、角色总质量和query分区不变时，增加两侧参考内容同时改善十组自然及严格阈值oracle点估计，单侧添加大多失败。E2全量仍在验证；它不证明FG/BG跨图配对关系，也不保证单图观察能复制多图收益。

旧reference-focus在PACO具有真实观察相对插值的局部增益，但完整方法失败；它采用每角色采样与ridge读出。旧cross-view binding使用多个query观察和乘积核。E3保留原K1全token加和KDE作为精确退化基线，只增加同一参考的真实观察，不重启这两个旧构造。

要回答的问题是：在保持前景和背景总质量、完整原参考证据和query观察不变时，真实参考局部表征是否比原表征与匹配插值更有效。这直接检验E2的双侧信息收益是否能通过合法单参考观察获得。

## 固定方法

复用 `reference_focus_head200_20261010` 的PACO100与DeepGlobe100任务，以及 `reference_focus200_20261010/input_bindings.json` 的真实focus缓存。原whole参考和query、同配对FoRIS/MEAN由既有清单绑定。

以whole参考/whole query的去位置单位特征计算固定中心μ1，所有真实和派生参考观察都在同一规则下处理。温度固定.05，query K64固定，后续为区域均值、1024双线性、minmax及严格>.5。

对于规范的512物理focus，whole的16×16块在focus外保留质量256，在focus内保留128；focus编码中每个8×8物理块保留质量32。总观测质量仍为原1024图面积，每角色质量由真实软Mask覆盖决定。非标准geometry或whole回退必须使用实际绑定的面积/重叠规则，不能硬套512常数。

令a为物理质量，y为合法参考Mask软覆盖，核为k(q,r)=exp(cos(q−μ1,r−μ1)/.05)：

F(q)=Σ a·y·k(q,r)，B(q)=Σ a·(1−y)·k(q,r)，c(q)=F(q)/(F(q)+B(q))。

actual与derived之间只有参考观察的特征内容改变；相对whole，focus内部把一半物理质量从粗观察分给细观察，保留全部whole特征但不保留每个token的原权重。不另拟合头、选择层/温度/阈值或按域分支。

## 比较与控制

三个实质臂为whole原K1、bilinear-derived focus、actual focus。derived与actual的物理位置、Mask覆盖、质量、去位置/单位化顺序和中心规则一致，仅前者的特征来自whole插值。whole与derived还相差细网格Mask和插值几何，不能把两者差值全部称作特征信息增益。nearest父token复制作为退化控制：同一粗块四个细块软质量相加后，必须还原原K1。保留原始完整FoRIS和MEAN同例控制。E3是新的受控观察实验，不宣称加和KDE本身新颖。

全部自然预测先生成后评分；query GT只在评分阶段使用。按域报告完整自然mIoU、逐例IoU、TP/FP增删、严格阈值oracle、相对whole和derived的照片配对差，不把oracle当自然成绩。此200例为已曝光机制开发队列，非全新确认或十域结果。

actual若同时超过whole与derived，且未靠明显损伤另一域换取，才扩大同一规则到其他域并核算缺少的真实参考观察。只胜derived但仍输whole，说明真观察抵消插值损害，不成立新方法。若仅oracle提升或仍为PACO正/道路负，不扫描温度、权重或阈值挽救这一构造。

## 执行范围

新入口 `scripts/x1012s/e3_reference_observations.py`，输出 `../cv_data/a/x1012s_E3/`，报告 `docs/research/x1012s_tasks/E3.md`。旧focus、B1/P0、R1至R5源码与产物保持各自原身份。

E2全量占用一个MPS进程时，E3只用CPU单进程2线程准备与执行；确需MPS则先等E2生产结束。先检查小量自然退化控制及吞吐，再运行预定200例；实现问题只修本入口，不改科学比较。该构造失败也不等于用户目标完成。
