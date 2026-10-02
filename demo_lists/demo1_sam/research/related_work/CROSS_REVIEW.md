# 两条原型路线的交叉审查（2026-10-01）

只读 `quality_mechanisms/PROPOSAL.md`、`compute_structure/prototype.py` 与已有 CPU JSON；未改其他代理代码、未运行 GPU/SSH。下文是可证伪的创新候选，不是新颖性保证或 SOTA 成绩。

## 计算路线：优先公平 phase 实测，其次 implicit；LN 仅诊断

| 原型具体实现 | 已有机制重合 | 可以保留的研究贡献与最小反证 |
|---|---|---|
| `phase_outputs`（prototype.py:206），两个 kernel=stride=2 的转置卷积改为 linear+phase layout，到最终四 masks 才 spatial reorder | 非重叠 deconv→subpixel/PixelShuffle 是经典等价布局；[PyTorch 官方实现文档](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pixelshuffle.py)引用 Shi et al. 2016。普通图优化/融合都可使用，无须 factor/LN。FastSAM 本身是换 CNN 实例分割器，不是这种固定 SAM graph rewrite | 可保留为工程增益；**不能将通用 head 优化归因跨提示共享**。已有 `dense_phase_predict` 和无因子 `DensePhaseCache` 是正确的强对照。若 dense_assoc+phase 与 factor+phase 同样快，主结论应为 head layout 改善，因子价值未成立 |
| `implicit_read` / `research_write`（157/181），直接消耗共享底座和提示因子，不重建稠密逐提示 K/V/Q | attention 关联顺序本身非新；FlashAttention 已覆盖 IO-aware fusion。FastSAM/YOLACT 的共享 mask prototypes 也是低秩组合，但经过训练、换了模型；[SAM3.1 Multiplex](https://arxiv.org/html/2511.16719v2#A8)联合对象记忆与 decoder、重新训练，改变独立对象输出函数 | 有差异的是固定预训练权重、LN 后逐行缩放+低秩状态的**端到端闭包与执行调度**。若在同样 phase head、同样 compiler/attention 最优配置后无净收益，就不能只讲 FLOPs 或 eager 局部 speedup。prototype index>0 实际为 explicit einsum+softmax，不受 `backend='sdpa'` 切换控制；实验标签必须写 mixed/explicit，不能把它称 SDPA implicit kernel |
| `ResearchCache.cross` + `StatisticState`（113）：用各层固定 write out_proj 子空间提前缓存 base-cross，运行时计算 Gram/row variance，删除 dense companion | 充分统计量、Gram 二次型、固定线性字典缓存是既有代数工具。真正可能不同的组合是处理 image state 经不同 LN affine map 的分层固定写子空间，保留 prompt 独立性；SAM3.1 不要求该闭包或原模型输出等价 | **现有数值反例已证伪无条件 FP32 稳定性**：CPU JSON scale100 得 variance=-0.0009765625（真实约9.62e-5），scale10000 得16（真实约9.60e-5）且仍 finite。因此仅检查 NaN/negative variance 不够；不可 clamp 后宣称 exact 或原门槛通过。先隔离 LN 做幅度/抵消 stress 与真实 activations 原门槛验证。只有数值改善后再评速度；FP64 accumulation/保守条件数 gate+dense fallback 的成本必须全部计入 |

已有 CPU 记录共126比较随机全部通过，max mask error=4.1723e-7；这覆盖测试分布，不覆盖 LN 抵消反例、真实权重或 GPU。`diagnostic_wall_s` 不能作性能数字。

**最小因果分解实验（完整 decoder、all four masks/IoU）**：

1. dense_assoc native head；dense_assoc phase head；factor_projected native head；factor_projected phase head。相同 FP32/TF32、相同 prompts，分别 eager/inductor。这四臂先隔离 head gain 和 factor gain。
2. 对胜出的公平 phase 基线加 implicit read-only / write-only / both，LN保持 dense companion。优先 P={8,64,128}、实际最优microbatch，另附匹配microbatch；T={7,9,16}测试rank随tokens增长是否破坏收益。
3. 单独统计 LN native vs statistics 的时间与alloc，记录各stage实际R、`u @ gram`、cross contractions、cat/transpose/clone；`N R²` 项不能因省掉 `N D` storage就默认更快。只有数值gate满足才把statistics完整结果纳入正常候选，其余标 diagnostic failure。

cache construction 用相同功能需要的缓存：phase/implicit 不开 statistics 时 `include_statistics=False`，稠密方法使用 `DensePhaseCache`，避免未用cache抬高某方法冷启动。报告每图cache实际unique storage与峰值allocated，head/layout不同的缓存也算构建成本。einsum源码声称“不复制”不是profiler证据：检查内部 expand+contiguous/clone/bmm，尤其 U 在head维的广播。

## 质量路线：flip selector 是最便宜的机制诊断；shared residual basis 尚不构成新颖性

| 提案 | 直接相邻工作 | 需要证明的增量 |
|---|---|---|
| identity+horizontal flip，inverse mapping、候选一一匹配、learned ranker，仅选官方1–3 | TTA/等变一致性是成熟范式；[PiClick](https://arxiv.org/abs/2304.11609)（2023-04-23首发；2024修订）已生成多候选并训练Target Reasoning module选择目标；[SAMRefiner](https://arxiv.org/abs/2502.06756)（ICLR2025）已有多提示和IoU adaptation；SAM2 dynamic fallback已有稳定性选择 | flip是否给同图original-only校准器无法得到的**错误模式证据**，而不是仅增加ranker容量/训练数据/候选数。按同1–3候选集比较：原IoU；校准IoU；original-only同参数ranker；flip highest-IoU/普通matched-logit TTA；flip matched ranker。必须胜过原视图等参数ranker且质量提升覆盖额外encoder成本。flip不是已有cache复用：它需要第二个image embedding/cache |
| `B(I)c_pk`，image-shared learned residualbasis+prompt gating | [YOLACT](https://arxiv.org/abs/1904.02689)（2019-04-04）已image prototypes+instance coefficients；[CondInst](https://arxiv.org/abs/2003.05664)（2020-03-12）已instance-conditioned compact mask heads；HQ-SAM/HQ-SAM2已用高分辨率图像特征和token改善细节；SAM2已high-res skip | 残差用途+提示独立系数是适配点，但**共享basis或加gating本身不新**。只有冻结base、训练数据/参数/预算匹配的条件下，优于普通CNN residual head与等参数token-conditioned adapter，并在SAM2/HQ-SAM2上仍增益，才支撑独立价值。r=16/32表达限制必须由oracle残差重建曲线验证 |
| 图像Laplacian+每prompt KKT约束、共享预条件器 | random walker/graph methods/CRF与提示约束解是既有方法；[SegRefiner](https://arxiv.org/abs/2312.12425)（NeurIPS2023）已有通用细mask refinement；HQ-SAM2是当前直接SAM质量基线 | “点满足+唯一解”是正确性合同，不是质量证书。需对比uniform/bilateral/feature graph、无约束refiner、普通learned refiner，并隔离B、gating、hard constraints、preconditioner四部分。既不能把旧graph smoothing收益归因低秩残差，也不能只报Boundary-IoU上升掩盖主体IoU下降 |

**最小可证伪质量实验**：

- 先复用官方SAM真实对象输出，只做诊断：每regime的official-selected vs 1–3 oracle gap，分image bootstrap。若oracle无明显空间，暂停flip ranker；若oracle边界仍很差，排序没有创造缺失内容的能力。
- 为同一小对象集每图加一次flip；在训练外image测一致性是否能辨别**原IoU头错选**。GT仅作诊断oracle、独立split监督/评分，不进入runtime selector。没有显著增量信号就结束这一view，不继续blind训练。
- matching CPU合同需补**缺失模态、重复候选、平分最优匹配**，不只测试纯重排。强制4↔4 assignment可把不同对象/部件硬配起来；可允许低IoU unmatched，固定tie处理并验证候选置换不改变selector输出。选择集合仍固定1–3，不能把token0混入制造收益。
- residual basis在训练前先检查真实oracle误差是否集中于可修复边界，并做r={4,16,32,64} 的跨prompt残差低秩拟合上界（这是GT-assisted表达能力诊断，不是部署成绩）。若全image residualbasis需要近满rank、普通高分辨率adapter也修不了，不应先投入basis训练。
- 正式质量竞争需同提示/同候选规则比较SAM2.1、HQ-SAM2、SAM3 visual-prompt/PVS，并匹配数据训练预算；多click交互可加PiClick/SimpleClick的统一click-simulator NoC/AUC，不能拿额外click对单click宣称质量胜出。

## 本轮选择

立即运行 **dense_phase公平四臂 → implicit带dense LN**；同时完成已有真实oracle-gap诊断。LN充分统计量保留为有明确失败反例的数值研究支线。质量若oracle gap支持，先flip特征判别而不是训练basis。顶会论点应来自“独立输出函数闭包/执行收益”或“新证据在相同真实成本下改善强模型错误”，不能来自模块名称、线性代数合同或随机CPU通过。

## 2026 主来源补查：必须更新创新边界

下面核验论文主页面/摘要；SAM-MT另核验作者项目页与官方仓库README。未经本机复现，未核验其模型实测数字或完整训练合同；不将未公开实现冒充现成可跑基线。

- **[SAM-MT: Real-Time Interactive Multi-Target Video Segmentation](https://arxiv.org/abs/2607.08688)**：2026-07-09首发，ECCV2026。明确在SAM2上用独立目标queries+共享global context、decoupled masked attention、sparse memory处理多目标。除SAM3.1外，这也是直接前作；“共享表示+独立对象身份/防串扰”不能当作本项目独有。当前exact路线区别仍是固定模型/输出函数，而不只是对象身份隔离。若转为联合训练decoder，需把SAM-MT纳入视频强基线；它的video FPS不直接比较我们image decoder ms。[官方repo](https://github.com/FudanCVL/SAM-MT)明确已发布checkpoint、inference code/interactive demo，training code尚未发布。
- **[UnfoldCRF: Structured Mask Refinement with Image-Conditioned Latent Regions](https://arxiv.org/abs/2609.33996)**：2026-09-27首发。能量包含corrected unary、learned local pairwise、image-conditioned latent-region consistency，unrolled mean-field；作者明确与同inputs/参数/步数/监督的recurrent black-box refiner对照，并考察未见mask generators。与质量A中的learned边权、image residualbasis/区域结构、共享image信息高度相邻。摘要说code/supporting materials将公开，本次未发现可验证repo，因此先作必须讨论的结构对照而非已经可跑基线。可区分贡献应在**多个独立prompt的可复用求解/真实成本+合法点约束+保持多义候选**，不能只讲structured refinement比CNN更可解释。
- **[U-CFR](https://arxiv.org/abs/2607.20705)**：2026-07-22，ICPR2026；boundary-aware uncertainty结合contour/edge，生成internal pseudo-clicks并cascade refinement。若将quality路线改为自动边界补点/风险gate，这已有直接机制；额外内部查询不是免费信息，应纳入总decoder调用/延迟并比较。
- **[Reason Twice / Rea2Seg](https://arxiv.org/abs/2606.09303)**：2026-06-08；candidate discovery后comparative scoring/reranking已经被作为segmentation主干。它是MLLM复杂text reasoning任务，不是同点/框interactive协议，不应硬拉为首轮GPU数字基线；但“由生成转向候选比较”并非独有叙事。

对LN数值问题的一个可推导修复候选：对固定centered write字典离线构造orthonormal Q，缓存`BcQ`及**显式投影残差**的平方范数，再将方差写成`||s(BcQ)+U(VcQ)||²/D + s²||Bc_perp||²/D`。这是实数等价的非负形式，可避免大正负二次型项最后抵消；不能用`||Bc||²-||BcQ||²`生成残差范数，否则重新引入抵消。需先CPU对比已有scale100/10000反例和真实输出原门槛，再评运行代价。字典rank可能第二层接近D，因此此方案不保证速度或显存优于dense LN，只提供明确可证伪的数值修复路线。
