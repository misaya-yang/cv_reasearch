# 参考任务判别的最小先例核对

2026-10-11，直接读作者论文／CVF／arXiv。只约束贡献表述与具体设计，不复跑论文、不新增编码／下载，不把文献或普遍定理作为完整实验前置。当前完整6200失败证据见[TASK_CONFLICT_COMPOSITION6200_RESULTS](TASK_CONFLICT_COMPOSITION6200_RESULTS_20261011.md)。

| 一手来源 | 已有方法 | 对下一完整方案的实际约束 |
|---|---|---|
| [FROST，arXiv2606.31136v1](https://arxiv.org/html/2606.31136v1) §3.2–3.4 | 冻结DINOv3 FG/BG非参数密度比、等先验零阈值；source类内shrinkage whitening、source LOO margin bandwidth与双向gate。作者明确其自然图／object-centric收益小于遥感，并讨论等先验条件。 | 普通FG/BG核、类内whitening、source带宽或reciprocal gate已有先例。论文报告的收益未在本机复现，不作我们的新结果；不能由R角色拟合推出Q任务正确。 |
| [PANet，ICCV2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.html) | FG/BG prototype matching；query预测反向成为support，进行prototype alignment regularization。该正则用于训练阶段。 | 反向重建／prototype alignment并非新概念；也不能把其训练结论直接当冻结特征推理保障。 |
| [SSP，arXiv2207.11549](https://arxiv.org/abs/2207.11549) | 高置信query预测形成self-support query prototypes，并有adaptive self-support BG prototype和self-support loss。 | 单加预测query FG/BG均值不构成原创贡献。新的任务证据须解释怎样拒绝共同误报，而不是只把预测当标签再拟合。 |
| [RePRI，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.html) | 带source监督、query后验熵与预测FG比例KL的transductive线性推断，使用经过base训练的特征。 | query熵／全局前景比例约束已有先例；不能把预测比例当正确GT先验，或把训练设定混同本项目冻结DINOv3。 |

当前应争取的可检验区别：以合法参考任务反证query成员／anchor，区分补真、补假、删真、删假，并在固定完整组合与强对照中体现净收益。当前条件反例已经说明单个预测FG中心的自洽无法证明BG没有被翻成FG；它还不是跨域普遍信息损失定理。原6200已曝光，后续统计仍须说明这一范围。
