# 跨观察对应身份绑定：固定缓存检验

2026-10-10。独立任务推断候选；DINO冻结，没有跨数据集训练，不使用FoRIS预测／场作为输入。FoRIS仅在预测全部封存之后作为外部完整对照。

## 动机与所检验的信息

已完成研究表明：参考内拟合、反向软重建、全局视图相关先验都未稳定选中正确query任务。全局物理先验仍有较强oracle表达能力，但合法source拟合失败。下一假设不是要求两视图不变，而是保留**哪些观察来自同一物理位置**。

现有不少参考读出先分别汇总whole与focus，再融合分数。这允许query的whole观察对应reference位置i、细观察对应另一个位置j，只要二者各自像同一role。若这是干扰来源，要求两个观察共同由一个reference位置解释，应提高目标身份判别。也可能跨图视图关系不稳定，使该绑定无益；必须用匹配控制检验。

## 输入与固定构造

沿用已有Deep前20、PACO每fold前5的40例，全部曝光开发样本，不是独立确认。每例七个原始O24都已经存在：Rwhole、Qwhole、Q四角、R实际focus。source focus框完全由此前合法R mask固定；不根据query GT或本候选结果移动。

Reference双观察：whole raw64裁出同一512框的32×32网格，再bilinear64，和真实focus raw64逐位置配对。每个位置对应同一个8×8 canonical像素块；其标签使用该块的真实FG面积coverage。Query双观察：whole raw64 bilinear128，与四真实角窗拼成128逐位置配对。所有view先token单位化、固定native黑图位置子空间投影、再单位化；没有按数据集或结果选择APD。

四十个reference focus都同时有FG/BG质量，Deep最少FG49.33个token面积、PACO89.83；未排除空角色案例来改变本队列。全参考单视图臂另保留focus之外的合法R监督。

## 唯一机制与对照

两view核分别为`k_v(q,r)=exp(cos(q_v,r_v)/.07)`，温度固定不搜索。每个role的source权重由coverage或1−coverage归一化。

- **绑定候选**：`p_y(q0,q1)=sum_r w_y(r) k0(q0,r0) k1(q1,r1)`，两个观察必须使用相同reference物理位置r。
- **因子化**：`p_y(q0)p_y(q1)`，同一组view／role边际，但允许不同source位置分别解释两个观察。
- **保边际打乱**：只在reference相同coverage（8×8 binary共65种精确值）的组内置换第二view的位置；FG/BG权重、两侧特征bag、温度、reference数和算量相同。固定seed0，各组按最早物理索引排序，使交换FG/BG时仍是同一置换。
- **全参考单view KDE**：使用所有whole-R与whole-Q插值特征及完整reference coverage，防止只赢过丢掉source范围的弱控制。

输出均为FG/BG的log密度比。绑定−因子化，恰为两role的跨view依赖项之差；是用于定位增量的数学分解，不是自动的新理论或校准概率。真正验证绑定信息的必要比较是绑定胜保边际打乱，而非仅胜因子化。

主完整终点固定为log比>0；同时预声明exact query two-means作为零点诊断，不能按域或结果选择renderer。直接bilinear128→1024后阈值，不追加CRF；original由binary1024按既定双线性>.5映回。全40的连续场、完整mask及两帧一起封存之后才评分。

排序同时报告全图与同一外部FoRIS候选内／外的FG/BG像素质量AUC，ROI仅用于封存后诊断；不把全图高AUC替代困难区身份，也不把局部信号替代完整净收益。Deep合并I/U；PACO观察19个fold/class直接聚合，不能当全官方协议。已曝光小队列的区间仅作开发诊断。

## 执行与创新边界

`src/ics/methods/cross_view_binding.py`和`scripts/run_cross_view_binding.py`；产物`cv_data/a/cross_view_binding40_20261010/`。MPS分块计算全reference核，CPU做合法标签／几何与文件核验，七raw仅只读，无新模型／编码／raw写入。

数学fixture已经验证：FG/BG互换使所有比值翻号；置换严格保持每个view/role边际；第二view没有位置差异时，绑定、因子化、打乱完全没有新增绑定信息。真实边际数值误差逐例记录，超过2e−5即拒绝。

联合KDE、特征融合、条件相关性是经典工具。可研究的是单参考任务迁移中**同一物理source对应身份是否是已有汇总丢掉的有效信息**。在匹配信息、独立完整输出、跨域和最近文献都成立之前，不宣称它是原创完整方法；失败也不通过扫描温度或增加更多融合规则掩盖。
