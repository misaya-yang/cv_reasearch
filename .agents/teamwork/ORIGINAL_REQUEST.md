# Original User Request

## Initial Request — 2026-10-02T08:34:55Z

深入调研近两年（2025-2026）CV领域变革（如冻结大模型视觉表征DINOv3、In-Context视觉分割及测试期局部验证与推理时缩放Inference-time Scaling），深度结合当前仓库已积累的负结果教训（标量统计选择器失效、无空间上下文的简单裁剪相似度未闭合oracle gap）与已验证理论推导（可加局部前景量重构全局IoU、局部更新判定准则、逐图regret边界），确立一项达到 CVPR 2027 Solid Accept 质量标准的完整学术论文研究计划，包含深层机制创新、自洽闭环的数学理论推导与代码级数值验证、以及无协议漏洞的强基线实验设计方案。

Working directory: /Users/yang/projects/CVPR2027
Integrity mode: demo

## Verification Resources & Local Context
- 历史研究结论与证据边界：/Users/yang/projects/CVPR2027/RESEARCH_STATUS.md
- CPU 检验交接与数学推导记录：/Users/yang/projects/CVPR2027/demo_lists/research_decision_20261001/HANDOFF.md
- 已验证的局部前景量优化与自测代码：/Users/yang/projects/CVPR2027/demo_lists/research_decision_20261001/mass_decision.py
- 独立未观察测试集清单（类别与图像均双隔离）：/Users/yang/projects/CVPR2027/demo_lists/demo8_local_verification/fresh800_seed2040_manifest.json
- 强基线对照实现协议：/Users/yang/projects/CVPR2027/demo_lists/demo8_local_verification/foris_control.py
- 近期22篇代表性文献综述表：/Users/yang/projects/CVPR2027/exa-results/cvpr2027-direction-review-2026-10-02.csv

## Requirements

### R1. CV前沿演变剖析与创新切入点定位 (Frontier Paradigm Analysis & Gap Formulation)
系统分析 2025-2026 年计算机视觉表征学习、密集预测与上下文推断的核心范式迁移，精确标定当前 Oral 级强基线（如 INSID3 种子扩展局限、FoRIS 前景整合瓶颈、FROST 密度比假设）在特征利用上的本质盲区。结合本仓库前期失败实验（标量特征选择器失效、无空间上下文的 crop 相似度未通过）的深层根因，明确论证为何“多粒度跨视图/token级条件化局部观测与全局一致性质量决策”是超越现有强基线的本质突破口。

### R2. 核心论文方法创新设计 (Core Methodological Innovation for CVPR 2027)
提出一套结构自洽、机理新颖且具备顶会 Oral 级竞争力的核心方法框架。该方法必须彻底跳出传统“浅层特征拼接”或“孤立启发式重打分”的窠臼，从信息论与决策论视角，将局部表征的多粒度观测（如高频/细节 patch、跨视图注意力上下文）与全图一致性掩码聚合有机耦合，明确定义输入输出流、特征相互作用机制与自适应推断逻辑。

### R3. 自洽闭环的数学理论推导与可执行代码验证 (Theoretical Derivation & Computational Verification)
为所提方法建立严格的形式化数学理论体系。推导必须包含形式化问题定义、核心引理、定理陈述与完整的数学证明（例如：局部-全局一致性区间收敛性、逐图或类别级 IoU regret 严格上界、动态观测决策的信息增益保证）。同时，提供完全自包含、可脱离 GPU 运行的 Python 数值验证程序，通过模拟生成极端分布、对抗退化案例与大规模抽样，100% 验证所有理论定理的数学成立性与数值稳定性。

### R4. 严格杜绝协议漏洞的端到端实验计划与消融矩阵 (Experimental Protocol & Falsifiable Verification Plan)
制定具备顶级学术严密性的实验验证计划。必须明确基准统一方案（彻底消除 COCO-20i 重建尺度、类别折数、图像重叠污染等历史偏差），设定严格以 FoRIS (512/1024+CRF)、INSID3、FROST 为对照的公平评测矩阵；制定基于 fresh800 未观察集（类别与图像均隔离）的验证流程；设计分阶段的可证伪检查点（包含低算力探针试验判定法则、端到端计算预算对齐、配对置信区间检验以及逐模块消融实验）。

## Acceptance Criteria

### 理论严谨性与数值自检 (Theoretical Rigor & Programmatic Check)
- [ ] 理论推导文档结构完整，包含符号表（Notation）、显式假设前提（Explicit Assumptions）、核心引理与主定理证明，严禁出现隐式无害假设（如隐式假设像素统计独立）或证明逻辑跳步。
- [ ] 编写并执行独立的 Python 数值自测程序（自验套件覆盖至少 1000 组随机及病态极端案例），全部测试通过，输出检验日志并保存至仓库。

### 机制创新性与可证伪性 (Innovation Distinctiveness & Falsifiable Guardrails)
- [ ] 方法方案明确列出与已有 CVPR 2026 SOTA（INSID3, FoRIS, FROST, REBASE）的 3 个以上本质机理差异，杜绝换壳式描述。
- [ ] 设定明确的“关键实验击穿指标”（Kill Criteria）：如果在初始双折配对 200 例探针中，相较强基线在等同算力预算下未实现统计显著的增益，提供备选收敛调整路径，防止无休止自说自话。

### 实验设计严密性与落地性 (Experimental Design & Protocol Integrity)
- [ ] 实验方案包含完整的测评基准对齐表格设计、详细的数据流与张量规格定义、以及配对 Bootstrap 置信区间计算规程。
- [ ] 明确定义全流程算力开销、训练/推理吞吐对照矩阵，确保方法增益不来源于未声明的计算预算不对等。
