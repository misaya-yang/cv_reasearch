# 同图独立提示的质量路线：先诊断，再选机制

日期：2026-10-01。这里只读本地源码和历史结果、执行微型 CPU 推导验证。没有访问远端、GPU、下载资产或训练；未核验 2026-10-01 最新文献，任何新颖性结论需由主代理的相关工作代理补查。CPU 数字不是分割质量或吞吐成绩。

## 当前实际依据

- `../../README.md` / `../../RESULTS.md`：共享图改写在随机权重通过，历史真实图片 FP32 logit 门槛未通过；不能把执行数值等价和质量提升混为一件事。
- `../../sam_shared_decoder/execution_baselines/factor_candidate.py`：图像底座是共享的；提示更新、逐位置缩放、稠密 LayerNorm 辅助状态仍是每提示独立的。质量改法可以复用缓存经验，但改变输出函数后必须单独评估。
- 本地官方 SAM2 固定快照 `.../sam2/modeling/sam/mask_decoder.py:219–225` 已实现高分辨率特征跳连；`247–287` 已实现 token0 的阈值稳定性检查和向最高预测 IoU multimask 的动态回退。不能以“多尺度特征”或“稳定性择优”直接声称新机制。

## 首先做这项最有判别力的预训练诊断

用官方 SAM ViT-B 权重和完整 image encoder / prompt encoder / postprocess；不通过随机 decoder 判断质量。小首轮建议 **24 张不同 COCO val2017 图像，约 100 个非 crowd 实例**，覆盖小物体、细结构、低对比边界、同类相邻实例；不要把这一小集称为 SOTA benchmark。

每个实例预先固定并保存：一个距离实例边界较远的正点、一个距边界较近但在 GT 内的正点、一个 GT tight box 或明确指定扰动幅度的 box。这三个 regime 分开报告。同图的不同对象、不同 prompt row 完全独立；GT 只生成评测提示和评分，不能进入部署排序/细化器。支持多正负点时另建 regime，不能将额外点击与原单点击质量直接比较。

输出所有四个 raw mask token 和四个 IoU 预测；按官方 resize→crop→resize→threshold 存全分辨率布尔 masks，数值实验另存所需 logits/误差。先比较：

1. single token0；multimask 在 1–3 中按官方 IoU 头选择。
2. 1–3 范围的 GT-IoU oracle；全部四个的 GT-IoU oracle，二者分开，避免扩大候选集制造排序收益。
3. 每 mask 的 mask IoU / Boundary IoU；IoU-oracle 所选 mask 的 Boundary IoU 与 boundary-oracle 上界分开。
4. 存每对象、每提示的得失，不只存平均数；多 regime 不汇成一个数字掩盖边界损害。

**决定方式**：如果 multimask oracle 明显优于官方所选（例如开发集中 mean gap >=0.02 且不是少数异常对象），优先机制 B；如果四 mask oracle 仍缺边界/细结构，排序不能创造缺失内容，优先机制 A 的可学习残差版本。这个 0.02 是预先写下的开发分支信号，不是论文成功门槛；保留全部实际数值。两类误差都有时也先选一类作为主干预，别一次叠模块。

### 主代理需要的官方资产

- 完整官方 `facebookresearch/segment-anything` 源码、SAM ViT-B `sam_vit_b_01ec64.pth` 权重。当前包只有 decoder 快照，没有 image/prompt encoders 和权重；下载/版本固定由主代理执行。
- COCO val2017 `instances_val2017.json` 和所选 image_id 对应的 JPEG；不需要整套训练图像。需要解码 RLE/polygon 的 COCO API（先查是否已安装），保留原始大小、annotation_id、iscrowd、面积和提示坐标 JSON。不要按模型表现挑图。
- 若做 SAM2 强基线：官方完整 SAM2 源码、相配 YAML 与 **SAM2.1 Hiera tiny** 权重是低成本首轮；正式比较应补同/相近资源级别官方更大模型。当前最小快照不是完整模型或预训练权重。
- HQ-SAM / SAM-HQ 和合适的边界质量方法作为必须补齐的质量基线，具体当前官方版本/权重与命名由文献代理核验；不能仅打败原 SAM 就声称挑战 SOTA。
- 只在机制 A 需要学习时准备独立训练 split。COCO val 首轮对象不能随后同时充当训练和论文测试；训练版需与基线匹配数据、更新参数和训练预算。

已有可直接评分的脚本：

```bash
python research/quality_mechanisms/diagnose_oracle_gap.py results/real_quality/image_0001.npz \
  --output results/real_quality/image_0001_diagnosis.json
```

NPZ 使用 `allow_pickle=False`：`masks[P,4,H,W]`（官方后处理的 bool）或 `logits[P,4,H,W]`、`iou_prediction[P,4]`、`gt[P,H,W]`；可加 `low_resolution_logits[P,4,h,w]`。逐 image 或小组保存，避免保存所有高分辨率浮点 logits 占用内存。外部 manifest 存 object_id、image_id、prompt regime / seed。

脚本的 SAM2 dynamic rule 只复现本地快照的 0.05/0.98 选择规则；**将该规则应用到 SAM 输出不是 SAM2 pretrained model 成绩**。当前 bootstrap 是 prompt-row 粒度，正式结论应用 image-cluster 配对 bootstrap，避免同图多提示虚增样本量。Boundary IoU 使用 0.02×图像对角线的内侧边界带、零 padding 与 3×3 repeated erosion；正式对照需统一到官方评测器并固定实现，不混合不同半径。

## 机制 A：图像共享的边界残差场 + 每提示约束投影

**目的**：在不增加人类提示、保持独立对象目标的条件下，补 SAM oracle 都没有的细边界；共享的资源是图像几何和残差基底，不是不同对象的 mask 或 token。

具体可学习结构：高分辨率图像特征 `F(I)` 每图计算一次，输出非负边权 `w_ij=softplus(g_theta(F_i,F_j))` 和 `r` 个图像残差基底 `B(I) ∈ R^{N×r}`（先试 r=16/32）。每提示、每 mask 使用 SAM mask token `q_pk`、原 mask 局部 uncertainty / 边界摘要预测系数 `c_pk`，构成 `m'_pk=m_pk+B c_pk`，保留四个候选与其独立排序。只共享 image-only 层；提示条件层和原多义候选完整保留。

随后解凸约束校正：

`min_y 0.5 ||y-m'||² + λ/2 Σ_(ij) w_ij (y_i-y_j)², subject to C_p y=t_p`。

`A(I)=I+λL_w` 是 image-only SPD；`C_p` 只选该提示自己的合法正/负坐标，`t_p=±τ`。不能把框内所有像素强置前景；框提示第一版不加点约束，只使用相同 refiner 和提示 token。硬约束只锁已有标注点，不会恢复未被提示指定的语义。重复点标签相冲突时拒绝该样本，而不是偷偷改标签。

设 `z=A^{-1}m'`，解为：

`y=z + A^{-1} C_p^T (C_p A^{-1} C_p^T)^{-1} (t_p-C_p z)`。

image-only `A` 与其预条件器可共享；每 prompt 的 RHS 与小约束系统独立。硬约束满足 `C_p y=t_p`，且 SPD 保证在约束合法时唯一解。**这两个性质是可验证设计合同，不是新颖性本身。** 当前 CPU 原型用 tiny-grid dense inverse；生产实现必须用稀疏 Laplacian matvec + 固定误差容限的 PCG / image-shared multigrid 预条件器，或训练成可展开少量步数。不可在 256² 上存 dense N² inverse，更不能把 CPU tiny inverse 时间当 GPU 成本。

### 与既有方法的区别和新颖性边界

- GrabCut / random walker / interactive graph segmentation 已有图像图和点约束；DenseCRF 已有图像边缘与 unary 的后处理。**单纯 `A`、约束 solve、Woodbury/KKT 或 bilateral smoothing 都不够构成创新。** 固定色彩图的版本只做低成本归因基线。
- 可值得研究的差异是：用 SAM 实际失败的高分辨率**残差子空间**学习 `B(I)`，以每提示 multimask token 选择残差；结合可复用的 image-only 预条件器与合法提示精确满足，在多独立提示下评估质量—总成本。理论支持的独立性和约束性质只解释设计，顶会价值要来自细结构质量、提示鲁棒性、多个独立提示的成本收益，以及对强模型的实测竞争。
- 与 SAM2 的高分辨率 skip 相比：它直接将高分辨率特征注入每提示上采样，本方案显式学习可重复使用的边界残差基底和像素约束校正；是否互补必须通过**接在 SAM2 上也有提升**、以及普通高分辨率 CNN refiner / 等参数 decoder adapter 对照验证。不能假设 SAM2 缺边界能力。
- 与固定后处理相比：学习边权和残差处理相似颜色不同对象、细线、纹理边缘；这是可检验预测，不能从图模型公式推断已实现。

最小 GPU 干预：先仅固定边权（RGB + encoder feature 相似度）作为 CPU/GPU可复现 refiner，25% 开发对象选固定 λ，75% 留出对象检查 mask IoU 与 Boundary IoU及逐例负增益；同时跑同 λ 的 uniform smoothing / 普通 bilateral-DenseCRF 或 random-walker 式 refiner、SAM2/HQ-SAM。若主要收益等同旧后处理，只保留工程基线，不维护创新故事。只有 oracle-error 诊断指向可补细节、固定图版本有合理边界信号后，才开展 r=16/32 的轻量学习 pilot；原 SAM 冻结、数据与普通 CNN adapter 公平匹配。完整 encoder+decoder+refiner+缓存构建成本都要报。

限制：光滑先验可能抹掉细结构；硬点满足不保证实例整体正确；低秩残差也可能表达不了复杂轮廓；所有这些由真实分层结果决定下一步。不要把改善 Boundary IoU 而破坏主体 IoU 描述为无条件提升。

## 机制 B：保持多义候选的等变视图排序，减少错误选择

**目的**：当四 mask 已覆盖正确对象而 IoU 头选择错误时，利用同一提示的几何一致性证据改善排序；不同原提示永不互投票。

每图生成 identity 与 horizontal flip 两个**语义保持**视图，encoder 只做每图每视图一次并缓存；每原提示按像素坐标精确映射点和框，运行相同四 mask decoder，再 inverse-map 到原坐标。第一版不做可能改变目标含义的点 jitter、不新增点击、不做裁掉物体的 crop。成本是一个额外 encoder 和每提示额外 decoder，必须与完整速度预算比较。

对每提示独立：用 mask IoU 矩阵在四候选之间做一对一最大匹配（K=4 只有24排列），因为 mask token 编号在视图下不保证相同语义。计算每个候选的匹配 IoU、边界 disagreement、原 IoU 预测与视图 IoU 预测差、已有正负点满足度。训练小型 pairwise ranker预测候选间 GT-IoU 顺序，监督只在独立训练 split 使用；不平均互不对应的实例/部件候选，不把“越一致”直接等同“越正确”。排序第一版只在官方 1–3 可用候选中选，确保收益归因不是候选集扩大；token0 动态分支另报。

为什么可能有用：SAM 的 IoU 头以 learned token 表征预测质量，没有直接利用视图间**该候选是否保持对象边界**的外部一致性证据；几何变换不增加人类信息。排序收益上限可由 heldout oracle gap直接度量。为什么未必有用：稳定错误可在两视图完全一致，且系统性歧义可能对所有视图保持一致；CPU 构造已给出“稳定性1、质量0”的反例。学到的排序必须面对这种情况，而不是把稳定性当证书。

强对照：原 IoU head；SAM2 dynamic stability；固定权重的原 IoU+threshold stability+prompt satisfaction ranker；普通 flip TTA 用 highest predicted IoU / matched-logit mean；只用原视图同特征的等参数 ranker（隔离额外 view 的价值）；直接用训练集校准 IoU head；HQ-SAM/SAM2 pretrained整模型。TTA/consistency ranking都是已有广泛范式；**不能宣称首次使用**。可发表的候选是保留 SAM 多义模式的可校准、成本可控排序机制在真实提示鲁棒性任务上证明额外证据与收益，并通过强 TTA/校准对照归因。若简单 IoU recalibration 已获同样收益，应归因校准，不强加视图模块。

最小 GPU 干预：在已完成 oracle诊断同一真实对象集上，只多做 horizontal flip 每图一次 + 每提示四mask，记录匹配一致性与相同 GT 所评分的质量。先统计 score feature是否在**训练外对象**解释 IoU-head错选（不要看测试后手调权重）；有信号后用独立开发集拟合 logistic/pairwise ranker，留出按 image 分割评测。若收益不够覆盖额外 encoder+decoder，则测试有监督的风险 gate：只对高错选风险 prompt追加 flip；gate本身也要与相同预算随机/IoU margin gate比较，不能利用 test GT决定追加。

## 已完成 CPU 证据与可复现命令

```bash
python research/quality_mechanisms/cpu_prototype.py
python research/quality_mechanisms/check_diagnostics.py
```

- `cpu_algebra_results.json`：8×10 grid、四候选，约束解与完整 KKT 解最大误差约2e-15；点约束误差约4.4e-16；独立提示顺序交换误差0；候选仅重排后的 matching误差0；冲突标签拒绝。还构造了threshold stability=1但合成GT IoU=0的反例。
- `diagnostic_contract_results.json`：合成合同检查，验证 oracle gap评分、相同mask的Boundary IoU、低分辨率SAM2 snapshot选择规则，以及full-resolution binary masks与logit接口一致。
- 没有预训练SAM / SAM2真实质量结果；没有GPU成本。新颖性核查未完成，两个机制目前都是有明确对照的研究候选。

建议主代理下一步优先完成真实 oracle-gap / 边界诊断，再根据错误结构选 A 或 B；不要先投入大规模 refiner训练或额外多视图矩阵。这样GPU新增工作直接回答方法选择问题。
