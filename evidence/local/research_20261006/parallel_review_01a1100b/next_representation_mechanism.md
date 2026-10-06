# 表示机制独立审查：M1 优先准备，M5 有限备选，M4 暂缓

日期：2026-10-06。仅本地阅读和研究设计；未访问实例、SSH、GPU或付费资源，未加载权重，未测真实分割和墙钟。未修改 Pro、源码、PLAN、STATUS。

结论：三种 Pro 构造均增加了现有九项终点缓存算法未观测的条件响应，但这只说明计算对象不同。建议优先准备 **M1 的只读共同后缀**，只在其接口/代价可行时保留 **M5 的固定消息路径外推** 为第二研究合同。M4 暂缓：九次编码、合成到自然迁移和抑制类别方向三项风险同时存在。没有任何候选拥有可据此预计的正 mIoU 数值。

## 1. 范围和现有九项的准确区别

当前约束以 [AGENTS.md](/Users/yang/projects/CVPR2027/AGENTS.md:1) 和 [PLAN 5–6、46–54 行](/Users/yang/projects/CVPR2027/docs/research/PLAN.md:5) 为准：实例已关机，只做本地准备，历史 CPU/GPU许可不继承。

九项也不只是对 `s,W` 做变换：它们可读取完整末层 q/r、参考 coverage、查询 RGB、原图几何；prior-shift/quadratic 的主动分支还能独立输出完整场。因此不能把它们概括为“都只能平滑原分数”，也不能由 `s,W` 的交换不可辨识性推出九项全都不可能成功。[共用入口 13–24、26–37 行](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/prepared_bundle_01a1100b/entry.json:13)。九项真实实验数仍为 0，合成见证不是 DINO 分布上的收益证明。[同文件 36–37 行](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/prepared_bundle_01a1100b/entry.json:36)。

| 构造 | 新观测量 | 相对九项的实质差异 | 尚不能推出的结论 |
|---|---|---|---|
| M1 | 原始 H20 与 Q 的原生 suffix K/V；同一查询位置上 FG/BG/自身的条件续算差 | 操作表示生成过程；查询条件函数不是终点几何或参考核分类器 | 深层量更语义、后缀去除了所有上下文、一定优于末层 |
| M4 | 同一 canonical 参考点跨位置/尺度/正负移植的响应散布 | 已准备 covariance 是**不同点的模式响应共变**；M4 是**同一点跨条件的变化** | 干预散布纯属环境、移植标签在自然 Q 上校准、消除方向一定保留类别 |
| M5 | 固定 QK 下 β=1/.75 的区域描述子差及其参照类 margin 差 | 测量冻结网络对消息路径的响应；终点树/布局/颜色没有这个响应 | 响应大小等于污染程度、β=0等于无背景语义、局部外推准确 |

来源：Pro [272–317 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:272)、[670–709 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:670)、[783–823 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:783)；现有 covariance [33–63 行](/Users/yang/projects/CVPR2027/src/ics/methods/reference_covariance.py:33)。这些都是同一合法 RGB 输入的新增计算，不是新增语义参考或监督。

**合同边界另需显式区分。** 旧`s3_rcg16_fine15_definition_v1`禁止把FoRIS最终score/mask作为推断输入，而Pro M1/M5定义的是复用完整FoRIS连续宿主的新完整合同；不能给它们沿用旧ID或称同一冻结方法。[旧合同 2–21、53–76 行](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/complete_method_contract_01a1100b.json:2)。宿主在合法RGB内部计算与把外部终点缓存作为入参也要区分。M1/M5不是九项共用CPU缓存入口可直接新增的两行；在当前“CPU高效”目标下仅值得本地评估实现可行性，不能列为已满足成本约束的准备方法。

## 2. 合同一：M1 同查询条件后缀比较（首选）

**所检验的缺口。** 原参考与查询终点可能受不同计算环境影响；共享只读 suffix 是否改善自然查询中的类别比较？这是待检验假设。现有 QK 描述子主臂在旧 DEV40 上为 −6.46 [−9.05,−3.92]，参考 ridge 在已暴露 DEV241、1024无CRF上为 −8.84 [−12.38,−6.44]；不能把内部 attention 存在当作成功先验。[RESULTS 82–94、154–167 行](/Users/yang/projects/CVPR2027/evidence/local/RESULTS.md:82)。两条旧方法不等于 M1，但提高了举证要求。

**冻结完整算法，保留 Pro v0。** 一参考 RGB/完整二值 mask、一查询 RGB、同冻结 DINOv3-L/16；工作分辨率1024，64² token。R/Q原生编码一次，保存原始 block20 residual及Q第21–24块原生K/V。参考FG/BG按coverage门限及确定性最远点选最多各4个真实token，选择空间为已固定投影的末层，初值为对应原始H20。每查询位置分别续算自身与参考探针：使用该位置RoPE，去掉3×3本地键和prefix键，探针不互相attention、不写回原生memory。最终LN→固定Π→单位化，τ=.07的FG/BG log-mean-exp对比得g。没有BG省略负项，空mask输出空，零向量按合同处理。[Pro 166–185、283–330 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:283)。

完整读出为 `y=s+rank(g)-rank(s)`、`(A+16L)z=Ay`，s/A/L由已绑定完整FoRIS连续宿主一次确定；不能因新guide重开gate、换图或anchor。z转FP32，64→1024双线性再阈值，再将该二值工作mask双线性到原尺寸再阈值，均align_corners=False。所有对照同一读出。[Pro 193–252、332–342 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:332)。这是新ρ=1读出，不能给mean-unit填历史ρ=.25的MEAN分数。

**最强简单替代。** 首轮同时跑plain（同参考代表和log-density）、mean-unit（同ρ=1）与uniform（合法V均匀平均但保留相同残差/MLP/四层），从它们中选固定开发集上最强完整结果作为主对照。这样将新增代表数量、融合幅度和通用MLP重读与内容条件attention分开。若确有增量，论文前再补固定memory/Π的一阶JVP；JVP追平只撤回“非线性必要”，并不否定共同条件度量。JVP实际代价未知。[Pro 364–376 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:364)。

**关键限缩。** 原生H20已含参考上下文及位置影响，“同一个Q suffix”不抹除这些历史。不同p的attention权重也不同，所以共享的是条件函数，不是逐值相同的消息；均匀控制尤其重要。正式本地键排除使自身轨迹也被改写，不能要求正式自身probe等于native来证明实现正确；一致性核验必须另设不排除任何原生memory的审核模式。

**数值一致性必要检查。**

1. H20必须来自真实block20输出而非`get_intermediate_layers(..., norm=True)`的归一化输出；当前wrapper固定`norm=True`，没有提供原始suffix合同。[data.py 13–22 行](/Users/yang/projects/CVPR2027/src/ics/data.py:13)。将其直接作为H20是改变算法。
2. 不排除键的自身probe，在每层用对应H^(l−1)产生的K/V、相同LN/QK norm/bias/RoPE/LayerScale及最终LN，需逐层及最终复现native；先定FP32 atol/rtol，报告maxabs而非凭输出看起来接近。前缀顺序、1基/0基层号、已RoPE与pre-RoPE K分别标明。
3. 单位置与64/128位置分块结果一致；缓存哈希运行前后不变；禁止将参考探针拼进memory。空间负控制只能固定槽位后置换内容再按新槽位RoPE，否则共同置换K/V/位置可能没有干预。[Pro 376 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:376)。
4. s/A/L/Π以及rank的并列规则完全同对照；CG报告收敛残差；检查两次插值、FP32边界和严格>0.5，不能仅核对token符号。

当前 `attention_intervention` 是成对前向里单块加SDPA bias，既不是只读suffix，也不能以该模块已可访问QKV证明M1已实现；其终点入口明确拒绝模拟干预。[intervention.py 71–95、162–168 行](/Users/yang/projects/CVPR2027/src/ics/methods/intervention.py:71)。官方DINO与timm的数值等价未在此阅读中建立，需要绑定**真实使用的timm源码**，不能照官方数学草图静默替代。

**推翻条件。** 在固定完整评价上不胜plain/mean-unit，或uniform追平且类别分离无新增；FG/BG被共同Q背景同化、natural true/wrong区域margin未改善；仅AUC提升却完整输出损失明确。CI跨零记未决；不得以小样本“证明没有可能”，也不得扫层位/代表数量/ρ直到名称成立。若JVP追平，改用简单共同度量并收缩贡献。

**资源。** 除已绑定宿主，2次原生编码＋最多9条轨迹的4块suffix。纯patch K/V FP32为 `4×2×4096×1024×4=128MiB`，R/Q H20另32MiB；prefix、模型、FoRIS、Π及分块临时张量另计。禁止物化全部head的9N×N attention。Pro的3.5“图像等效”是算术工作量提示，真实完整墙钟/p95/显存/CPU峰值均未测；不能宣称满足CPU分钟预算。[Pro 382 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:382)。既有终点缓存不足，未来取得H20和suffix缓存本身也要编码；当前不执行。

## 3. 合同二：M5 固定消息路径局部外推（有条件备选）

**检验的量。** 在原R/Q中观测区域外→内消息重分配的响应方向，判断其是否比普通区域表征增加跨图类别信息。旧crop/gray/CLS负例降低预期；响应路径改变并不证明语义改善。[Pro 878–880 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:878)。

**冻结完整算法，保留 Pro v0。** Query固定Ward完整分区K={4,8,16,32}共≤60 ROI，不按宿主峰值筛选；参考coverage≥.5的完整FG与BG≤8分区，全图覆盖。原生R/Q保存原始H20及四层Q/K/V，固定Π。每ROI从H20更新内部状态，只更新V；QK概率始终来自该层native logits，外部V固定。消息为

`m(β)=[p_in+(1−β)p_out]V̄_in(β)+β p_out V̄_out`。

β=1审核native；正式额外β=.75。投影末层LN后先池化、再形成d=4(μ1−μ.75)，η=min(1,.5||μ1||/(||d||+1e−8))，v=unit(μ1−ηd)。新旧参考FG/BG margin差δ_C，四分区平均δ_i，完整场 `t=z_MEAN+tanh(δ/.07)`；固定共享renderer，不重求图、不clamp。空mask/缺BG/小范数按Pro约定处理。[Pro 787–868 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:787)。

**最强简单替代。** 首轮完整比较同ROI/宿主/renderer的endpoint（直接μ.75）、hard-channel（固定QK直接β=0）、native-region（原区域FG/BG margin作修正）以及零修正MEAN；固定开发集最强者为主对照。论文前还要比较同预算自然crop/gray、重新计算QK的masked attention、仅缩放外部而不重分配内部的路径及两条objectness控制。复用ROI和编码可降研究成本，但各部署成本必须单列。[Pro 882–906 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:882)。

**需要纠正的解释边界。** .75到1的局部割线被用于朝0外推，局部曲率小不约束整段[0,1]。若μ(β)=a+βb且η=1，才得到a；η<1时得到 `a+(1−η)b=μ(1−η)`，即使仿射也不是β=0截距。只检查范数限幅不能证明恢复“无污染特征”。路径还增大内部消息质量，不是只删除背景。必须报告限幅率和方向，不能以小响应直接作“对象性”解释。[Pro 823、836–874 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:836)。

**数值一致性必要检查。**

1. β=1逐层复现native，包括prefix作为外部；QK始终native，不能按分支重算。用内部/外部分组logsumexp核对概率和为1，无`1−p_out`消减或分母截断。
2. 验证空外部质量、极小内部质量、单token和全图ROI；禁止保存整图四层全head N² attention，分块消费native logits。
3. 仿射小构造核对η=1截距、η<1的上述退化；恒等描述子δ=0需完整mask逐像素回到同MEAN，而不只核对场符号。
4. 每token四次覆盖、参考FG/BG区域定义、pool/LN/Π/unit顺序及native-region≠零干预一并写入绑定recipe。

**推翻条件。** endpoint/hard/native-region或objectness解释全部完整收益；纯缩放同样好而重分配/截距主张失效；参考FG/BG身份互换仍保持修正；自然正确像素损伤超过纠错；只有候选AUC变好。失败关闭该固定v0，不继续扫β/层/限幅。

**资源。** R/Q原生2次编码；额外R约N、Q四分区4N的四块续算；纯patch四层双图QKV FP32为384MiB，H20另32MiB。每ROI仍访问全图memory，60小分支调度、原生QK重构、Ward构树及完整宿主均计费。Pro约2.833“图像等效”不是比M1更快的证据，真实时间/内存未知；当前已有缓存不能提供该干预量。[Pro 908 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:908)。

## 4. M4为什么暂缓，而不是宣称已被否定

其观测量不是现有模式covariance的重命名；以同canonical点跨条件中心化，确实分开了部件间项和跨条件项。但位置与尺度同时变化，得到的是该移植协议的混合响应，不是纯环境因果效应。参考局部背景、paste形状和尺度变化仍可成为类别方向，抑制C_env也可能抑制真类别。参考ridge及matched-filter既有负结果不能推翻该具体构造，却要求自然Q胜简单方向，不能只看移植点拟合。[Pro 674–683、695–749 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:674)；[RESULTS 1558–1572 行](/Users/yang/projects/CVPR2027/evidence/local/RESULTS.md:1558)。

若未来另获明确资源授权，至少要paired/mean/class-LDA/ref-canvas同Potts完整行，class-LDA复用同λ；论文前balanced线性CE方向及其自身bias读出也要测。点映射、最小尺度纯度/接缝、所有label来自R、未paste的Q从不作负类，是必要数值/标签合同。正常1Q+8 edited Q=9编码，几何失败2、后验统计失败最多10；不能把流式低峰值与快速混为一谈。[Pro 757–775 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:757)。现阶段不推荐为其投入实现或另立第三合同。

## 5. 完整证据合同与一个有限下一动作

二者均不得把诊断oracle余量转换为预期增益，也不能把历史fine增量加到新法。完整结果必须报原尺寸class-summed I/U mIoU、1024附表、四类像素增删、同photo-connected groups配对区间，并保存逐例完整mask。DEV241已暴露，不能叫独立确认；确认只能在另行绑定、照片隔离且暴露状态核对后的清单上，先冻结最多两候选及各自最强简单控制。[Pro 1033–1079 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:1033)。历史强版本与无fold合法强版本分开；未完成组合的成绩留空。[Pro 256–264 行](/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md:256)。

**真正有限的下一动作：只准备一份M1原生suffix一致性接口规格。** 不加载权重、不跑真实图像，限定检查当前本地timm实现能否暴露：原始block20输出、block21–24各层LN前状态/原生K/V、prefix与patch位置映射、RoPE/QK norm/LayerScale调用和最后LN；输出固定字段/shape/dtype/索引表，以及“无排除自身probe=native”的检查入口设计。源码或模型缺失就明确登记缺失，不下载或远程补齐。完成条件是每个字段能指向实际producer或标明缺失，且不存在用norm=True中间层代替H20的情况。未满足就停在接口缺口；满足也不自动开机或执行整套Pro实验。

这项动作解决M1是否可按原合同实施的具体不确定性；它不是方法结果，不增加已验证方法数。M5与M4暂不生成代码或新待办，最终是否验证由主代理按本轮用户范围选择。
