# 新方向筛选记录（2026-10-02）

按 `RESEARCH_PROTOCOL.md` 的标准做的筛选。全程没有用 GPU：数据来自 demo2 已保存的逐图统计、已有先导实验和文献。

结论先说：**在闭集语义分割（ADE20K / COCO）里，我没有找到能通过筛选的新方法。** 下面是依据。

## 1. demo2 的失败给出的反馈：误差账本

脚本 `demo2_where_what_decoding/scripts/budget.py`，输出在 `demo2_where_what_decoding/results/ade20k/budget_*.txt`。
“对”指一张图里的一个类。ADE20K 验证集 2000 张，共 16909 个真值对。

### 1.1 损失在大区域，不在小物体

把某个面积档里的真值对全部召回，mIoU 能涨多少（并集不变，是增益的下界）：

| 占图像面积 | 占全部对 | 整个被漏掉的比例（Swin-T / B5） | 全部召回后的 mIoU 增益（Swin-T / B5） |
|---|---:|---:|---:|
| < 0.05% | 1.9% | 70% / 85% | +0.04 / +0.05 |
| 0.05%–0.2% | 6.6% | 44% / 58% | +0.35 / +0.47 |
| 0.2%–1% | 16.8% | 28% / 32% | +1.94 / +2.45 |
| 1%–5% | 26.0% | 17% / 16% | +5.73 / +6.65 |
| 5%–20% | 27.3% | 8.5% / 7.1% | +8.82 / +9.54 |
| > 20% | 21.4% | 4.8% / 3.8% | +10.94 / +9.76 |

Swin-T 指 Mask2Former Swin-T（47.99），B5 指 SegFormer-B5（50.93）。
把“整个被漏掉”的对全部召回：Swin-T +14.6，B5 +11.8。把图里没有却被预测出来的类全部去掉：B5 +5.6。

小物体漏得多，但对 mIoU 几乎没有影响。损失来自占图像 1% 以上的区域被整块叫错。

### 1.2 最差的类是近义类，而且验证集里没几张图

EoMT-L（58.38）IoU 最低的类：ship 0.9、land 4.4、shower 5.2、hovel 7.5、hill 8.3、crt screen 9.9、step 14.3、river 14.8、tower 17.7、bannister 20.2。
它们几乎都有一个更常见的近义类或上位类：boat、earth、house、mountain、monitor、stairs、water、building、railing。

验证集里每个类出现在多少张图里：少于 10 张的有 19 个类，少于 20 张的 35 个，少于 30 张的 52 个，中位数 49 张。
少于 30 张的 52 个类占全部 IoU 缺口的 36%–38%。同一个类在不同模型上能差几十个点（lake 只有 5 张图：Swin-T 6.5，EoMT 56.0；hovel 8 张：EoMT 7.5，B5 30.2，UperNet 56.3）。

### 1.3 这对选题意味着什么

- 实测：损失集中在大区域整块叫错；最差的类是近义类和样本极少的类。
- 推断：靠解码、融合、后处理能拿回的很少，因为模型对这些区域是高置信地叫错，而正确叫法取决于数据集的命名习惯。
- 未验证：这些错误里有多少是标注本身不一致。验证它需要同一批图的第二份标注（COCO 上有 COCONut 重标的验证集可用，ADE20K 没有）。

## 2. 逐个筛掉的候选

| # | 候选 | 依赖的前提 | 否定它的证据 | 来源 |
|---|---|---|---|---|
| 1 | 用 SAM 一类的无类别分组，在组内投票统一名字 | 错误是“同一物体被切成两个名字” | 文献报告在 ADE20K 的监督模型上无显著增益；我测到的 +10.2 用的是真值的类级分组，带了标签信息；主要错误是整块叫错，投票修不了 | Qamar 等 2023；本文 1.1 |
| 2 | 小物体 / 亚 patch 解码 | 损失在小物体和边界 | 面积 < 0.2% 的对全部召回只有 +0.4 到 +0.5；标签在 16 像素网格内取常数的上限是 85.5（硬）/ 91.5（双线性），高于 EoMT 的 78.7，网格不是瓶颈 | 本文 1.1；`demo2_where_what_decoding/pilots/pilot12_grid_ceiling.log` |
| 3 | 用场景 / 全局特征给出“图里有哪些类”的先验 | 全局信息能分清近义类 | 先导实验 7：软融合 +0.30，硬门控 +0.02 | demo2 `NEGATIVE_RESULTS.md` 第 4 节 |
| 4 | 类别头的长尾重加权、为 mIoU 解码 | 少数类被先验压制，后验里还有可分信息 | 先导实验 3：+0.4 到 +1.1，校准后消失 | 同上第 5 节 |
| 5 | 用分割器自己的特征重训命名头 | 类别头没训到位 | Swin-T 上 +1.63 [+0.27, +2.59]，强模型上预计不到 1；撑不起主贡献 | demo2 `RESULTS_TABLES.md` 表 5 |
| 6 | 冻结编码器 + 无类别分组头 + 冻结特征命名 | 分组可以比 EoMT 更纯 | 真值区域命名 75.2，要追平 EoMT 需要纯度不低于 EoMT 的 78.7；冻结编码器加轻量解码器的路线已有 PMT（2026）在做 | demo2 表 1；tue-mps/eomt |
| 7 | 少样本“换命名体系”：分组不动，只学新名字 | 分组能跨数据集迁移，少量标注下胜过微调 | 样本多时预计输给直接微调，样本极少时预计输给零样本开放词表模型，没有明确能赢的区间 | 推断，未实测 |
| 8 | 开放词表分割里去掉图中不存在的类 | — | 已有 FreeCP（ICCV 2025）、CaR（CVPR 2024） | 见文末 |
| 9 | 多模态大模型定位 / 指代分割的一致性解码 | 多次采样里有正确答案，可以靠一致性选出来 | 已有 FORUM（ACCV 2026）及多个智能体式方法；模型大，单卡迭代慢 | 见文末 |

第 6、7 条是由已测到的数推出来的，没有单独做实验。

## 3. 剩下的两条路

### 路 A：把诊断做成分析 / 评测方法的稿件

不是提点方法。内容是三件事：

1. 误差账本（在哪 / 是什么、按面积、按近义类），在 10 个以上公开模型上重复；
2. 评测的统计可靠性：mIoU 的置信区间、配对检验，说明多大的差距才算真的差距；
3. 用 COCO 与 COCONut 两份标注量出“叫错”里有多少其实是标注不一致。

全部是公开权重的推理，每个模型不到 1 GPU 小时。每一步都有产出，不存在“对照追平就归零”的风险。
我估计 CVPR 主会中稿概率 15% 左右，这是判断，不是测出来的。

### 路 B：换赛道找提点方法

闭集语义分割之外，我今天只在纸面上看了两个赛道（上表第 8、9 条），都已经很挤。
要在新赛道里按 `RESEARCH_PROTOCOL.md` 的第 1 步做账本，需要下载新的模型和数据、用 GPU，所以先由用户定赛道。

## 文献

- Qamar 等，Can Segment Anything Model Improve Semantic Segmentation? NeurIPS 2023 workshop：https://nips.cc/virtual/2023/76526
- FreeCP，Training-Free Class Purification for Open-Vocabulary Semantic Segmentation，ICCV 2025：https://arxiv.org/abs/2508.00557
- FORUM，Frozen Outputs Reconciled Using Model Agreement for Visual Grounding，ACCV 2026：https://arxiv.org/abs/2609.37488
- GroundingAgent，Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning：https://arxiv.org/abs/2511.19516
- COCONut，Modernizing COCO Segmentation，CVPR 2024：https://arxiv.org/abs/2404.08639
- EoMT 及后续 PMT：https://github.com/tue-mps/eomt
