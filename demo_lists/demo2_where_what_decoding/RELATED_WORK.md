# 相关工作与新颖性边界

日期：2026-10-02。分两类：**本次联网核对过**的（标题、日期、摘要要点已确认），和**凭记忆列出、尚未核对**的。
执行者在写论文前必须把第二类逐条核对，并做一次系统检索；这份清单不是穷尽的。

## 1. 最接近的工作：必须正面比较

### 1.1 图像级类别存在性用于分割

| 工作 | 状态 | 做了什么 | 与本项目的关系 |
|---|---|---|---|
| [RankSeg / MLSeg](https://arxiv.org/abs/2203.04187)（ECCV 2022） | 已核对 | 把分割拆成"图像级多标签分类 + 只在选中类上做像素分类"，多标签头与分割器联合训练。Mask2Former 上 ADE20K 全景 +0.8 | 最直接的前作。它也报告了真值存在性的上限。差别：它要重训分割器、识别器与分割器共用特征、增益不到 1；本项目不动分割器、用独立的冻结编码器、在区域级而不是图像级融合 |
| [Dense Multi-label Networks](https://arxiv.org/abs/1701.07122)（2017） | 仅见摘要 | 用多层级区域的多标签一致性压制不合理的类，ADE20K 上报告错误类减少 | 与"幻觉类"同一现象的早期处理，属于训练时方法 |
| [LabelBank](https://arxiv.org/abs/1703.09891)（2017） | 已核对 | 用图像级标签预测来过滤分割结果，并报告了"真值 LabelBank"的上限（FCN 上 ADE20K 54.3） | 说明"存在性上限很大"这个观察至少 2017 年就有。本项目不能把这个观察当作新发现，只能说"它在 2022–2026 年的模型上依然成立，而且上限更大" |
| [A Classification Refinement Strategy for Semantic Segmentation](https://arxiv.org/abs/1801.07674)（2018） | 仅见标题 | 分类结果修正分割 | 需读全文 |

**结论**：图像级这条线已经有人做过，而且本项目的先导实验表明图像级接口太粗（+0.3）。
本项目的落点是**区域级**。

### 1.2 冻结基础编码器给掩码定类

| 工作 | 状态 | 做了什么 | 与本项目的关系 |
|---|---|---|---|
| [FC-CLIP](https://arxiv.org/abs/2308.02487)（NeurIPS 2023） | 已核对（10-02） | 三个部件：类无关掩码生成器、词表内分类器、词表外分类器。词表外分类器用冻结 CLIP 骨干特征做掩码池化；两个分类器的分数做几何集成 | **新颖性的最大威胁。** 它本质上就是"分割器的类别判断 × 冻结编码器的区域判断"。差别：它面向开放词表、是训练时设计的一部分；本项目是对任意已训练的闭集分割器做事后解码，并把重点放在诊断上 |
| [Region-Based Representations Revisited](https://arxiv.org/abs/2402.02352)（CVPR 2024） | 已核对摘要（10-02），正文未读 | SAM 等类无关分割器的掩码 + DINOv2 等特征的掩码池化 + 线性解码器做语义分割；结论是区域特征优于 patch 特征 | **第二个直接威胁。** "在区域上池化冻结特征再分类"就是它的做法。差别：它的区域来自类无关分割器，没有与一个有监督分割器的后验融合，也没有"在哪 / 是什么"的误差分解。正文必须读，并要把它的做法（SAM 区域 + 池化 + 线性）作为基线跑一遍 |
| [Uncertainty-Gated Region-Level Retrieval](https://arxiv.org/abs/2512.18082)（2025-12） | 仅见摘要 | 区域级、按不确定度触发的检索，用于域偏移下的鲁棒分割，报告 mIoU +11.3% | 同属"区域级事后修正"。需读正文确认它是否作用在冻结分割器上、用的什么特征 |
| ODISE、OVSeg、MaskCLIP 等两阶段开放词表方法 | 未核对 | 掩码提议 + 冻结图文模型分类 | 同一范式的开放词表版本 |
| MaskFormer / Mask2Former | 未核对 | 掩码分类；论文里分析过收益主要来自识别质量而非分割质量 | "是什么比在哪更关键"这个判断的早期证据，需要引用并说明本项目的量化方式不同 |

**结论**：不能声称"用冻结编码器给区域定类"是新方法。能声称的是：

1. 对已训练的闭集分割器，这一步可以**事后**加上，不需要重训；
2. 它相对"普通像素级集成"有结构性优势（假设 H1，待验证）；
3. 两个上限 + 幻觉像素率构成的**诊断**，以及"骨干变大买到的主要是识别能力"的量化。

如果 H1、H2 都不成立，上面第 2 条就没有了，方法部分的新颖性不足以支撑论文。

### 1.3 为指标解码

| 工作 | 状态 | 做了什么 | 与本项目的关系 |
|---|---|---|---|
| [RankSEG](https://arxiv.org/abs/2206.13086)（JMLR 2023） | 已核对 | Dice / IoU 的贝叶斯最优分割规则：按概率排序取前 τ* 个像素，τ* 自适应。逐图像 | 理论上的直接前作。本项目的规则 M 是它的**数据集级**对应物 |
| [RankSEG-RMA](https://arxiv.org/abs/2510.15362)（NeurIPS 2025） | 已核对 | 用倒数矩近似把复杂度降到线性，并支持多类不重叠。有 `rankseg` 包 | 必须作为基线。先导实测：它提高逐图像 mIoU（+0.34），降低标准的数据集级 mIoU（−0.58） |
| [On the Relaxation of Conditional Independence Assumption for Image Segmentation](https://arxiv.org/abs/2609.38930)（2026-09-30） | 已核对 | 同一作者，放宽条件独立假设 | 这条线仍在活跃，审稿人很可能来自这个组 |
| [Application of Decision Rules for Handling Class Imbalance in Semantic Segmentation](https://arxiv.org/abs/1901.08394)（2019） | 已核对标题与要点 | 贝叶斯规则与最大似然规则（除以先验）的比较 | 对应规则 M 在"平均类准确率"下的特例 |
| [To Each Metric Its Decoding](https://arxiv.org/abs/2506.01552)（2025） | 已核对标题 | 层次分类里按指标做事后最优解码 | 同一思想在另一个任务上 |
| [Rethinking Post-Hoc Calibration in Semantic Segmentation](https://arxiv.org/abs/2607.01902)（2026） | 已核对 | 主张校准器应当保持 argmax 不变 | 立场相反：本项目认为决策应当随指标改变 |
| Narasimhan 等（ICML 2015）、Koyejo 等（NeurIPS 2014） | 未核对 | 混淆矩阵函数型指标的一致插入式分类器；线性分式指标的最优分类器是带阈值的后验 | 定理 1 是这类一般结论在数据集级 mIoU 上的显式形式，不能当作全新的理论 |
| Nowozin（CVPR 2014）、Premachandran 等（CVPR 2014） | 未核对 | 期望 IoU 下的最优决策；经验最小贝叶斯风险 | CRF 时代的同类工作 |
| Lovász-Softmax（CVPR 2018）、Jaccard Metric Losses（NeurIPS 2023） | 未核对 | 训练时直接优化 IoU | 训练侧的对应物。需要实验说明事后解码与它们是否可叠加 |
| Wang 等，Revisiting Evaluation Metrics for Semantic Segmentation（NeurIPS 2023） | 未核对 | 区分数据集级、逐图像、逐类的 mIoU | 支持"两种 mIoU 不是一回事"的论点 |

**结论**：规则 M 的理论贡献是增量的（已有一般理论 + 逐图像版本）。
它能站住的点是：数据集级的闭式、无标签直推、以及"与逐图像规则方向相反"这个实测事实。只能当配角。

### 1.4 其他事后改进分割的路线

| 工作 | 状态 | 关系 |
|---|---|---|
| [SegRAG](https://arxiv.org/abs/2605.17630)（2026） | 仅见标题 | 免训练的检索增强分割。同属"推理时引入额外信息"，需读全文确认是否涉及区域级重分类 |
| [MetaFusion](https://arxiv.org/abs/1912.07420)（2019） | 仅见标题 | 少数类假阴性的受控减少，与决策规则相关 |
| SAM 类掩码 + 分类器的语义分割 | 未核对 | 另一种"在哪"的来源。阶段 A3 的可选项 |

## 2. 写论文前必须回答的三个问题

1. **与 FC-CLIP 式几何集成的区别到底是什么？** 需要一个直接的实验对比：
   把本项目的识别器按 FC-CLIP 的方式接进去，和规则 R_γ 比。
2. **2023–2026 年有没有人对闭集分割器做过"事后区域重分类"？** 本次检索没有找到，但检索不充分。
   关键词建议：post-hoc region reclassification、mask-pooled foundation features closed-set、
   segmentation hallucinated classes、what/where decomposition segmentation。
3. **有没有现成的"分割误差分解工具箱"？** 有（10-02 检索到）：
   [What's Outside the Intersection?](https://openaccess.thecvf.com/content/WACV2024/html/Bernhard_Whats_Outside_the_Intersection_Fine-Grained_Error_Analysis_for_Semantic_Segmentation_WACV_2024_paper.html)
   （WACV 2024，代码 `mxbh/beyond-iou`）。它把每个错误像素归为边界、范围、整段三类之一，在 ADE20K 上分析了当时最好的模型，
   结论之一是 Mask2Former 强在划界、SETR 强在分类，并且**把两个互补的模型组合起来就能提高 mIoU**。
   这与本项目的诊断和方法都直接相关，正文必须读。本项目的"在哪 / 是什么"上限分解要与它的三类错误对齐
   （它的"整段错误"大致对应这里的"是什么"错误），并说明多出来的是什么：用上限给出可恢复的 mIoU，而不只是错误像素的占比。
   另有 [Diagnostics in Semantic Segmentation](https://arxiv.org/abs/1809.10328)（2018），把错误分成定位偏差和三种类别混淆。

## 3. 本次已经证伪、不必再检索的方向

见 `NEGATIVE_RESULTS.md`：SAM 解码器的稀疏化与上下文解耦、模型自估计存在性、全局特征存在性探针。
