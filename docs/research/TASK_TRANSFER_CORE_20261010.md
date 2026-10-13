# 从反复失败转向可识别的跨图任务规则

2026-10-10。用户要求有贡献、有创新的独立算法；FoRIS仅作为外部最终比较。参考细节接入FoRIS的未运行草稿已撤下，不启动该路线。

当前最强解释是**单参考可拟合的判别关系，不等于可跨图识别的任务规则**，但还不能把它认作唯一根因。已有证据分成四个层次：

1. 表示包含部分目标信息。三域同RGB真实Mask切换没有hard-mask碰撞，连续身份AUC较高；不能笼统说DINO不懂mask或没有信息。
2. 参考拟合与跨图迁移分离。方向度量改善reference cluster纯度，却有COCO13/40、PACO15/40、SUIM21/40任务的query完整IoU下降。PACO600固定ridge修正+2.314，COCO200−.899，是直接迁移反例。
3. 多观察改变有效任务响应，不能一律视作应消除的噪声。真实reference focus下，PACO合法FG沿旧判别方向的响应下降，差异含细节、上下文和任务信号；简单强制视图不变会同时抹掉有用信息。
4. 弱身份输入不能靠后续求解补救。已有R1-G大部分损失在几何前已发生；真实邻接copula没有提供足够可迁移cut信息。求解正确或更多迭代不代表新增身份信息。

这些结论指向需要区分的三个问题：**表示是否支持任务、跨图是否存在共同判别规则、单个参考是否足以选中那个规则**。只看reference loss、全图AUC或最终一个分数，无法区分三者。

## 固定诊断：先判定该修哪里

`scripts/task_transfer_identifiability.py`使用现有COCO／PACO／SUIM各20对、每对A/B真实mask和原始O24，零新增编码，MPS计算Gram和读出，小型线性系统CPU FP64求解。原始单位特征为主，固定位置子空间投影为共同控制，不使用FoRIS前端、伪mask、聚类或CRF。

同一固定线性容量、连续像素质量的平衡二元风险、lambda=.01、每角色128次确定性quadrature。空间按8×8 token块的棋盘分成两fold，训练fold与评价fold分开。所有reference-only字段先生成封存，随后单独打开query标签做oracle诊断：

- R→R与R→Q：参考内部可分是否迁移。
- Q→Q与R→Q：query特征本身的同类线性读出能力是否明显高于迁移。
- 共同R+Q规则在R/Q留出上的能力：两个图是否确实需要互相冲突的线性规则。
- 固定R方向、仅从query训练块给oracle阈值：缺口有多少只是零点，有多少需要改变排序。

后两项使用query训练标签，**绝不是可部署算法、独立确认、完整mIoU或性能成果**。空间留出仍共享整张图的DINO上下文，只支持表示／假设族能力诊断。共同规则失败也不证明所有非线性规则不可能。数值先用CPU FP64核对MPS Gram误差，原始数组及源码hash可追溯。

## 创新边界与下一算法条件

[RePRI](https://openaccess.thecvf.com/content/CVPR2021/papers/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.pdf)已有源监督、query熵和面积比例约束；[ABCDFSS](https://arxiv.org/abs/2402.17614)已有冻结特征的测试时视图一致性和任务适配；[SINE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2cc0b08447bf9668db268e6c86364a6e-Abstract-Conference.html)明确研究上下文任务歧义；[GW semantic correspondence](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_721/paper.pdf)已有在匹配中加入局部结构的OT。因此“自适应／一致性／关系／OT”这些名称本身不是贡献。

需要的贡献应是一个可解释并可验证的**跨图任务识别原则**：它要说明哪些现有观察提供了超出单图拟合的信息，如何排除同样解释参考mask却解释错query的竞争规则，以及何时该信息不足。候选必须有独立输出，不借FoRIS预测作前景答案；同RGB换mask应正确切换，部件和整物体同时检验。先在缓存上验证新增身份信息胜简单控制，再进行完整掩码和多数据集验证。未得到以上证据前不命名为新理论或预称原创成功。

## 本轮实测：可以表示，却难以只凭reference识别

全60对／120真实任务完成；COCO78、PACO80、SUIM80个有效任务×空间fold，每种表示各238个。COCO两项因空间半区缺角色而明确排除，没有补成0或.5。下表为原始单位O24的**soft pixel-mass AUC**，每照片内先平均A/B任务和有效fold，再平均照片；不是mIoU。

|域|R训练→R留出|R训练→Q留出|Q标签oracle→Q留出|R+Q标签共同规则→Q留出|
|---|---:|---:|---:|---:|
|COCO|.9922|.9487|.9950|.9946|
|PACO-Part|.9926|.9306|.9930|.9916|
|SUIM|.9909|.8806|.9892|.9875|

共同规则的R留出AUC为.9927/.9908/.9880，接近只学R；固定位置投影重复同一结论，不能把差距全部归结为位置污染。只从query训练块给原R方向选择oracle阈值，soft token IoU可增加12.518/16.350/15.686点，但不改变排序AUC，仍落后Q自身能力。说明**排序转移与零点转移同时存在缺口**；不能再把全部失败解释为单一阈值问题。

这里“共同规则存在”只表示该有限、使用两图标签的线性模型能在两图留出块上读出；高维模型仍可能利用图像特有方向，共享DINO上下文仍相关。它不证明已经找到可从单参考识别的语义规律，也不证明跨新照片泛化。R/Q单独拟合方向余弦均值约.381/.411/.376，只是表示差异，不能单凭角度认定因果。

参考无queryGT生成12.720秒，oracle诊断15.689秒；MPS加速Gram/读出、CPU FP64小系统求解，DINO0。独立验证3332项AUC／soft-IoU，AUC最大误差3.33e−16；三域首例共48个CPU FP64重构场相对GPU混合计算最大误差4.32e−7。完整证据在`../cv_data/a/task_transfer_identifiability60_20261010/`；仓库轻量快照在`evidence/local/task_transfer_core_20261010/identifiability/`。

## 排除一个最直接、已有先例的选择依据

既然共同规则可表达，就先检验最简单的无query标签约束，而不是立即叠复杂网络。固定`K_RQ=softmax(R Q^T/.07)`，以`K_RQ Q`作为reference在query中的重建描述子，原R与重建描述子各占一半监督风险；同样处理R→R且排除自身位置作匹配控制。原始单位O24、相同reference labels／quadrature／lambda；完整独立输出，零FoRIS依赖、零额外编码、零参数搜索。

|完整1024输出，两任务／照片均值IoU|只学R|R→R重建控制|R→Q重建约束|
|---|---:|---:|---:|
|COCO|45.851|44.775|36.371|
|PACO-Part|41.879|41.966|42.548|
|SUIM|47.104|47.490|46.455|

这是同一批的完整输出，仍不是官方class mIoU。COCO相对只学R−9.480，95%照片配对区间[−16.460,−2.390]；PACO+.669与SUIM−.649均未稳定通过。不采用，也不扫温度／权重。此实验是经典循环／转导思想的机制对照，**不主张创新**。全部预测先封存后GT，10.561秒，采样MPS driver峰值1.024GiB。入口`task_transfer_reverse_probe.py`。

进一步把**真实query mask**通过这个固定K送回reference作反证：reference前景收到的query前景质量均值仅.404/.437/.541，对应61.61%/58.43%/42.25%的reference FG质量在该回传中≤.5；同图控制为24.28%/22.00%/15.50%。这些阈值统计只解释固定算子的行为，不能称真实对应错误率。软核混合、query类比例与跨图外观差异都可能造成它，不应擅自只归因“背景不一致”。但它确实说明**正确Q标签并不满足当前强加的R重建假设**：这种自监督目标会同时惩罚正确答案。入口`task_transfer_reverse_audit.py`，全部在预测封存之后执行。

[PANet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.pdf)已有反向prototype alignment，[CyCTR](https://proceedings.neurips.cc/paper_files/paper/2021/file/b8b12f949378552c21f28deff8ba8eb6-Paper.pdf)已有cycle-consistent像素筛选。普通反向重建、视图一致性或OT并不是可直接包装的贡献。本轮Exa五个主题查询及一个定向来源查询返回43条检索结果，按原论文／作者代码核对与本假设最相关的既有机制，不声称穷尽文献或证明新颖性。

## 聚焦后的研究主线

需要研究的是**在单参考允许多种任务解释时，如何用合法观察识别可迁移的解释**。一个可信的新方法要显式处理“这条跨图约束是否保持任务标签”，不把高cosine、低reference loss或循环闭合直接当作正确性。真实多视图的已知物理对应可作为检验材料，但不能一律要求特征相同；只要求它对同一实际mask的解释相容，并保留拒绝无法验证对应的能力。

下一项在提出完整求解器之前，先用现存whole／native-crop物理对应验证：某个仅依赖合法reference标签与无标签query的可匹配性证据，能否区分正确任务解释与同样符合reference的错误解释；必须保留同特征／同边际但打乱物理配对的控制。若不能，就不进入新损失、更多层或大规模实验。该原则是待证伪研究方向，尚不是已成立的新算法。主goal仍未达到，不能将本轮诊断替代最终多数据集完整领先。
