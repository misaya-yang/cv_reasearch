# SAM 跨提示共享：主来源核验与下一项实验（2026-10-01）

查阅范围：官方论文页面、作者仓库及已实现代码；这是有边界的路线核验，不是穷尽的新颖性检索。只在本地做 NumPy 代数检查，没有 SSH、GPU、模型权重下载或环境安装。网页字节哈希及首发日期见 `source_audit_20261001.json`。原实现边界来自本目录上级 README 与 `factor_candidate.py`：**投影后的因子 + 稠密 LN 伴随状态**；隐式 attention 尚未实现。

## 先给实验决策

1. **不能以 SAM 2023 为当前质量 SOTA。** SAM 3（2025-11-20 首发）已改善 visual prompting；SAM 3.1（2026-03-27 发布，论文 v2 2026-03-28）已实现 Object Multiplex 多对象共享。下一篇工作必须说清楚与它的区别。[SAM 3 论文](https://arxiv.org/abs/2511.16719)、[SAM 3.1 官方说明](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)。
2. 最有判别力的近期改法：**直接用因子收缩计算读 attention logits/output 及写 attention logits，不生成逐提示稠密 K/V/Q**。保留当前稠密 LN 辅助状态与完整上采样头，先做单模块 GPU 对照，再集成完整 decoder。这是补齐当前候选的执行调度，不是凭 FLOPs 宣布新方法胜出。
3. 如纯加速无法胜过 compiled dense_assoc，可转向**同对象多种提示带来的质量预算收益**，先测预测 mask 的 oracle 提升空间与不依赖 GT 的可回收比例。必须对比 ICLR 2025 SAMRefiner 的多提示策略及 SAM 2.1、HQ-SAM2、SAM 3 的 visual-prompt 路线；普通 prompt ensemble 本身不是充分创新。[SAMRefiner](https://arxiv.org/abs/2502.06756)。

## 10 项最相关论文（原始日期与实际机制）

日期为 arXiv v1；发布日期另列，不把后续仓库更新当论文首发。各论文自报速度仅作定位，不外推本机。

| 论文 | 首发 | 已实现的机制 | 与本方法重合与剩余空间 |
|---|---|---|---|
| [SAM 2](https://arxiv.org/abs/2408.00714)；[官方 repo](https://github.com/facebookresearch/sam2) | 2024-08-01；2.1 checkpoints 2024-09-29 | Hiera 图像主干、streaming video memory、SAM 风格双向 decoder；2024-12 官方加入全模型 compile 与独立对象处理 | 图像共享已是标准，不构成创新。固定 SAM2 权重下对不同 sparse prompts 的中间状态共享仍可测；video memory-conditioned 图像状态不同不能无条件共享。必须包含高分辨率特征与 object-score/dynamic fallback |
| [SAM 3 + v2 Appendix H](https://arxiv.org/html/2511.16719v2#A8)；[3.1 release](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) | 2025-11-20；v2 2026-03-28；3.1 release 2026-03-27 | detector/tracker 共用视觉编码器、presence head；Object Multiplex 默认每桶 M=16 个对象，共享 spatial memory + object embeddings；memory encode/attention/mask decoder 在 bucket 中联合工作，每 slot 3 masks，5 tokens；训练时随机 slot | 最近、最直接的“多对象共享”强工作。它重新训练并改变对象间联合表征，不是固定权重独立 decoder 的等价改写；我们的潜在空间是无需重训、逐提示独立语义与精确代数结构。不能宣称首次跨对象共享 |
| [Efficient Track Anything](https://arxiv.org/abs/2411.18933)；[作者 repo](https://github.com/yformer/EfficientTAM) | 2024-11-28 | 轻量 plain ViT、efficient memory cross-attention，SA-1B/SA-V 训练，image/video code 与 checkpoints 已公开 | 同为 memory/attention 优化，但改变模型和训练。适合质量-延迟 Pareto 与 SAM2 移植检验；不作为同权重 exact 对照 |
| [EfficientViT-SAM](https://arxiv.org/abs/2402.05008)；[官方实验说明](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_sam/README.md) | 2024-02-07 | 保留 SAM prompt encoder/decoder，替换图像 encoder，先蒸馏再 SA-1B 端到端训练，FP16 TensorRT 部署 | 本方法可与它组合。既然 encoder 已很快，多提示 decoder 加速价值更突出；其公开 48.9x 是 A100 TensorRT 的模型整体比较，不能当作我们 decoder 基线数字 |
| [EfficientSAM](https://arxiv.org/abs/2312.00863)；[作者 repo](https://github.com/yformer/EfficientSAM) | 2023-12-01 | SAMI masked-image pretraining，轻量 encoder + decoder 后 SA-1B 微调；可用 image/point/box 代码 | 训练式压缩；比较可用质量-延迟而非函数一致性。不要误称未训练 decoder 与原 SAM 逐元素等价 |
| [EdgeSAM](https://arxiv.org/abs/2312.06660)；[作者 repo](https://github.com/chongzhou96/EdgeSAM) | 2023-12-11；IJCV 2025 版本 2025-09-07 | CNN encoder 蒸馏，prompt encoder 和 decoder 进入 distillation loop，点/框、ONNX/CoreML 已实现 | 提示鲁棒性的近邻：prompt-aware distillation 已存在。要证明跨提示结构改善泛化，应在匹配提示类型、密度、质量预算下比较，单纯加入 prompt-loss 不足 |
| [MobileSAMv2](https://arxiv.org/abs/2312.09579)；[作者 repo](https://github.com/ChaoningZhang/MobileSAM) | 2023-12-15 | object-aware valid prompt sampling 代替 segment-everything 网格冗余采样，减少 decoder 次数；可配轻量 encoder | 它优化提示集合。我们的固定提示 exact 路线保留集合，正交；改为质量路线以后应比较相同总时延下对象发现/覆盖率，不能用减少提示数来证明原 exact 方法 |
| [Fast Segment Anything](https://arxiv.org/abs/2306.12156)；[作者 repo](https://github.com/CASIA-IVA-Lab/FastSAM) | 2023-06-21 | CNN 实例分割一次生成 segments，点/框/text 用于筛选；重训模型 | 已将许多查询的重计算改为“共享候选集 + prompting”，是系统任务级反例。不能把任意 independent SAM outputs 等价于一次候选筛选；质量路线要测漏检与细边界 |
| [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567)；[HQ-SAM2](https://github.com/SysCV/sam-hq/tree/main/sam-hq2) | 2023-06-02；HQ-SAM2 2024-11-17 | HQ output token，早期/末期图像特征与 decoder features 融合，44K fine-grained masks 小规模新增参数训练；HQ-SAM2 已有实现与 beta 权重 | 边界质量/新增 token/多尺度融合已被覆盖。不能仅加 refinement token 宣称新颖。若提出 prompt-consistency 增益，需对比 HQ-SAM2 并使用 boundary 指标 |
| [SAMRefiner](https://arxiv.org/abs/2502.06756)；[作者代码](https://github.com/linyq2117/SAMRefiner/blob/main/sam_refiner.py) | 2025-02-10；ICLR 2025 | 从 coarse mask 提取 distance-guided points、context-aware elastic boxes、Gaussian-style mask prompt，split-then-merge；++ 包含无新增标注的 IoU adaptation；公开 code 默认迭代 refinement | 直接覆盖“多提示改善质量、IoU 重排”。重要约束：它使用变化的 dense mask prompt，不能直接复用当前仅支持相同 dense prompt 的缓存。新路线必须证明 sparse-only 同对象 query 多样性与共享预算有独立收益 |

原始 SAM [首发 2023-04-05](https://arxiv.org/abs/2304.02643)，仅作为函数 reference。质量退化路线另有 [RobustSAM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_RobustSAM_Segment_Anything_Robustly_on_Degraded_Images_CVPR_2024_paper.html)，已有新增参数训练与 Robust-Seg；不能把换几个受扰动提示当作超越图像退化鲁棒性 SOTA。

### SAM 3.1 必须保留的比较细节

官方约 7x 数字是在 **H100、128 个 video objects、2025-11 SAM3 为 reference**，同时包含 compile/fusion、减少 CPU-GPU sync、batched postprocess/encoder 等优化。不是只靠 decoder 共享的 7x。Appendix H 对照用同 detector；release SAM 3.1 又用了新训练 detector，应分别报告。质量也非所有指标均增：官方 release 表中 LVVIS/BURST/YTVIS21 有下降。不要把“without sacrificing accuracy”标题替代完整结果。SAM3 checkpoint 需要 Hugging Face access；本次没有请求权限/下载。

### 工程强基线

[FlashAttention-2](https://arxiv.org/abs/2307.08691)（2023-07-17）已以 IO-aware tiling 避免 attention matrix 落盘，SAM decoder 属于短 queries /长 image keys 的非对称形状，泛化大语言模型结论需实测。[官方库](https://github.com/Dao-AILab/flash-attention)明确 CUDA FA2 的 dtype 为 FP16/BF16，支持 Ada；FA3 面向 Hopper，FA4 面向 Hopper/Blackwell，**不能把 RTX 4080 FP32 SDPA 叫作 FA3/FA4 或 FlashAttention**。

首轮 exact 强基线：同权重 official、cached、dense_assoc、factor_projected，各自匹配 explicit/实际适用 SDPA、eager/真实 inductor；包含 all-four masks/IoU/upscale。允许每方法分别搜索最优 microbatch，另提供相同 microbatch 对照。建议增加 CUDA Graph 或 `compile(mode='reduce-overhead')` 后的 dense_assoc；若消除 launch overhead 后候选优势消失，不能只取 eager 优势。TensorRT/ONNX 可作随后部署线，但必须导出 all-four outputs，公开 single-mask export 不能直接与完整四输出候选比较。

## 方法 A：在 attention 收缩中消费因子

对单 head 写成：`K = diag(s) Bk + U Ck + Pk + 1 bk`，`V = diag(s) Bv + U Cv + 1 bv`。共享 Bk/Bv 为图像底座投影，U/C 为提示相关因子，Pk 是独立 PE 项。

- logits：`L = [(Q Bk^T) ⊙ s^T + (Q Ck^T) U^T + Q Pk^T + Q bk^T] / sqrt(dh)`。
- `A = softmax(L)` 后输出：`O = (A ⊙ s^T) Bv + (A U) Cv + bv`，最后 bias 依赖 `sum(A,row)=1`。
- 写 attention 图像 Q：`Lwrite = [s ⊙ (Bq Ktoken^T) + U (Cq Ktoken^T) + Pq Ktoken^T + bq Ktoken^T] / sqrt(dh)`，无需稠密 Q。

每 head 独立 softmax，因子/PE/bias 不可遗漏。优先只做 read，当前 write_factors 已显式计算 probabilities；dense_assoc 有等价聚合重排，必须比较而不是削弱它。GPU 实现分两步：先以普通 matmul 实测直接收缩的净代价；值得融合时再用 Triton tile/online-softmax，将 base-load、row-scale、factor contribution 与归约融合。保留 dense LN companion，**不宣称已经删除全部 N×D 状态**。原子核 launch、R 随 tokens 增大、非线性 upscaling 可吃掉收益，profiling 决定是否继续。

CPU 已验证：`cpu_factor_attention_identity.py`，5 个 FP64 随机形状，rank={0,8,48,98,40}，read/write logits 与 output 最大绝对差 `2.50e-16`，固定 `1e-12` 门槛；结果见 JSON。采用 NumPy einsum 标量收缩，绕开本机 Accelerate BLAS 的无效 FPE 警告。范围仅一 head 的代数恒等式；**不是多层已训练 SAM、FP32 logit 门槛或 GPU 加速通过**。

可复现：

```bash
cd /Users/yang/projects/CVPR2027/demo_lists/demo1_sam
python3 research/related_work/cpu_factor_attention_identity.py
```

最小 GPU 实验：N=4096、T={7,9,16}、R 使用实际层 rank、P={8,32,128}，先 standalone read+write 分别比较 projected、implicit、dense_assoc，统计 allocated 峰值与 CUDA 时间；之后必须在完整 decoder 输出上复验原始 FP32 门槛，并运行全 decoder 的速度检查。单模块优势只作为是否集成的条件，不能作为 paper 主结果。

## 方法 B：同对象提示家族的交叉验证与预算路由（质量假设）

精确共享节省的预算可用于更可靠选择有歧义的 mask，但需要先证明质量信号而不是先训练。对同一个已知目标对象构建 K={1,2,4,8} 个 sparse-only 家族（内点扰动、点/框组合、框轻微扩张；dense prompt 固定为 no-mask）。家族不跨对象合并。

先形成每个 query 的 3 个 multimask candidates，保留 official single-mask 作并列基线。对 candidate m 定义：`S(m)=IoUhat(m)+lambda*mean_family_Dice(m)-mu*prompt_violation(m)`；Dice 只与不同家族的候选 best match 比，避免同 query 三个模态互相重复计数。增加面积/边界一致性作为可选诊断。选择真实 candidate，不用 GT、不平均不同物体 masks。对同对象多模态歧义，简单 majority 可以错误地抹掉正确小目标，因此必做 medoid、mean-logit、official IoU、stability-score 消融。

**新颖性要求**：SAMRefiner 已覆盖多提示和 IoU adaptation。值得投入的剩余假设是“不同 sparse 提示对错误类型提供互补证据，用跨家族一致性估计何时多花 decoder 预算”，并从同权重共享系统获得更好的质量-真实时间曲线；无须变化 dense mask prompt 的家族可利用当前 cache。仅有 mIoU 多一点、但付出 K 倍时间且未胜过 SAM3/HQ-SAM2，不能称 SOTA。

第一项真实短测：固定 100-200 个 COCO/LVIS 对象（附 image/object id、seed、prompt 坐标），各 K 上报告 `oracle-best family IoU` 与 `official-IoU-selected IoU` 的 gap；加入 boundary F、失败对象类别、prompt violation。若 oracle 几乎无空间，停止此家族设计；若 oracle 有空间而可观测 score 回收很少，修改 prompt family/selector，而不是直接训练。小预算路由先用第1个 query 的 multimask ambiguity + predicted-IoU gap / stability 启发式，仅对不确定对象生成第二家族；与同延迟 random allocation、固定 K、SAM2 dynamic fallback 比较。质量收益需 holdout，lambda/mu 不能在测试集挑。

CPU 先验证两件事：读取已保存 logits/IoU/GT 后能计算 oracle gap 与 score 所回收比例；逐对象重排 family 顺序不改变 selector 结果。这里未取得真实图像输出，**没有执行或宣称这条路线已改善质量**。

## 公平的任务边界

| 目标 | 可比任务/基线 | 必须记录 |
|---|---|---|
| exact 执行改善 | 同图固定 P 独立 sparse prompts、相同 dense prompt、同权重的四方法 + compile/SDPA/graph | 全四低分辨率 masks 与 IoU、原门槛状态、最终分辨率 pixels；cache build/cold/hot、decode 与 image E2E 分开 |
| interactive/box 质量-时延 | SAM2.1、HQ-SAM2、SAM3 interactive tracker/PVS、EfficientViT-SAM；同 COCO/LVIS 图/固定点/框与 candidate-selection 规则 | GT-box 和 detector-box 分开；同 detector；IoU/boundary-F、AUC/NoC 需统一 click simulation；端到端 GPU time + memory |
| 多对象 video 共享 | SAM3 November、SAM3 Multiplex 同 detector，另外列 3.1新 detector；SAM2.1、EfficientTAM | DAVIS17/MOSEv2/SA-V 合适 split、J&F、objects counts；完整帧路径，不用静态 decoder 内核速度冒充视频 FPS |
| coarse-mask refinement | SAMRefiner / ++、HQ-SAM2、原模型迭代 | 相同初始 coarse masks/迭代预算；dense prompt 每轮变化后的 cache 重建代价必须算入 |

下一轮建议按顺序：**隐式 read/write 模块短测 → 可胜出的完整执行调度 → 同对象 sparse family 的 oracle/selection gap**。前两步不需要训练；第三步使用少量带标注真实对象后再决定是否投入学习机制。SAM3.1 移植是后续路线，当前原始 SAM 局部速度无论多好都不足以证明跨架构、真实任务或 SOTA。
