# 共同oracle高上界的来源：参考监督约束审计

基于当前`TASK_TRANSFER_CORE_20261010.md`及其已封存数据，原始单位O24下只学R的Q AUC为.9487/.9306/.8806，R+Q标签oracle为.9946/.9916/.9875。这不直接证明单参考足以识别共同语义规则：原诊断每角色128次quadrature，高维共同模型也可能在参考监督未约束的方向利用Q标签。

本项固定同60对、同A/B真实mask、同8×8空间fold、同lambda=.01及原始单位O24，不新增照片、编码、全局训练头或FoRIS模块。

先在禁止query标签的阶段，使用训练fold全部reference token的精确连续pixel-mass平衡平方风险，拟合同容量线性读出。原256次quadrature读出为共同控制；全部reference-only字段先封存。它只排除采样近似混淆，不预设密集监督能解决迁移。

之后明确以query训练标签作oracle诊断，重放父共同模型。令训练quadrature的augmented参考矩阵`Z_R=[X_R,1]`，`delta=theta_joint-theta_reference`。以FP64 SVD分解`delta=P_row(Z_R)delta+(I-P_row(Z_R))delta`，分别加回原R模型，检查哪部分产生Q的oracle收益。null分量必须在所有采样参考位置上严格保持logit，另记录其在未采样参考位置和R留出块的变化。

还记录一个只由全部R原预测决定的安全步长：沿null分量移动，取最早可能翻转R token符号的步长一半、上限1。该反事实须对全部4096 R token保持原FP64二值预测，才能比较Q变化。方向本身仍来自Q标签，绝不可用于推理。它不证明所有R标签完全正确，也不证明所有非线性模型不可识别。

判据：密集R若明显补齐缺口，则原证据含采样因素；若null分量保留大部分Q收益而R留出仍近原值，则共同oracle的高分不足以支持任务可识别。若安全反事实几乎不能改变Q，保留该反证，不夸大为基本不可能性定理。空间留出仍共享整张图上下文，当前仍为已曝光开发材料。

入口`scripts/task_transfer_source_span_audit.py`，资产`../cv_data/a/task_transfer_source_span60_20261010/`。MPS仅加速缓存矩阵，CPU FP64求解/SVD；源码、raw SHA、父字段、控制重放误差、逐fold数值、资源实测和结果全部保留。主goal未完成。

## 完成结果与解释边界

60对完成；COCO78、PACO80、SUIM80有效任务×fold。独立核验10,974项，weighted AUC误差3.33e−16，场分解恒等式误差1.55e−15，密集R的CPU FP64复算误差4.47e−7。reference-only密集拟合10.675秒，编码0；未保存全过程GPU峰值，不借用其它队列峰值作为本项测量。

|域|稀疏R→Q AUC|密集R→Q|共同oracle|仅rowspace增量|仅nullspace增量|保全4096 R判断的null增量|
|---|---:|---:|---:|---:|---:|---:|
|COCO|.94867|.95252|.99457|.98766|.99282|.97215|
|PACO|.93058|.93035|.99157|.98664|.98409|.94515|
|SUIM|.88058|.87998|.98750|.97965|.96856|.90053|

密集监督仅在COCO补+.00385，在另两域几乎无变化，不能以quadrature近似解释主缺口。null分量在采样R上logit严格不变，R留出AUC仍约.99；安全缩步后全部4096 R token二值输出不变，Q AUC仍增加+.02348/+.01457/+.01995，三个query-photo配对区间均正。对应Q soft token IoU变化+5.222/+5.928/+5.347，**都是query标签oracle反事实**，不是预测方法或完整mask成绩。

rowspace增量也明显改善Q，不能宣称“问题只在nullspace”。nullspace只针对256次role quadrature，不针对全部R特征；原R判断相同只针对4096 token，不能冒称原分辨率reference像素mask或全部logit相同。不同解释仍可能同样满足有限reference判据，是本假设族内的识别问题，不是普遍不可能性定理。

## 现有物理观察的后续核验

缓存metadata检查显示，本60对COCO/SUIM没有canonical参考四角，PACO有5个完整native4、7个至少一角；metadata不等于payload验收。已有reference-focus200有17个PACO对按RGB＋原maskA精确匹配，已真实验证payload／输入tensor／mask覆盖。无新编码。

`task_transfer_focus_observer_20261010/`在这17对比较对称±oracle-null候选，两者都保留4096 R判断。合法reference held-block平方风险选择后，Q AUC：whole .91639、同几何插值 .89987、真实focus .91146、真实focus位置打乱 .90959。真实focus只相对插值有正差，未稳定超过whole／打乱；不能说真实观察已识别正确任务解释。

`task_transfer_focus_ordinal_observer_20261010/`是看到前项失败后的探索性单项修改：仅用R FG/BG排序，不要求跨view分数大小或零点一致；无角色／排序相同则回到原R模型。真实focus选择后Q AUC .92738，相对插值+.02403 [+ .00120, + .06194]，相对whole+.02213及相对打乱+.01910的区间仍跨零；相对原R模型仅+.00371 [−.00239, +.00985]。68个任务fold中，真实focus选+35、选−21、拒绝12。候选方向仍来自query标签，绝不可包装为合法算法或跨域确认。

由此产生的无query标签独立prototype已经固定并实测，见`TASK_NULL_RANK_PROBE_20261010.md`。理论归Astra，后续具体机制与可证伪规格由其给出；Sol负责同批完整输出及得失分验证。
