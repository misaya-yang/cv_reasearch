# 研究总览（2026-10-16）

这是研究的唯一入口：当前方法、当前证据、哪些路线已经关闭、下一步做什么。

- 其他文档都是细节或历史，从下面的链接进入。
- 本文件与旧文档冲突时，以本文件为准。
- 旧文档里的“下一步”“正在运行”“唯一候选”等表述，都是当时的快照。

## 1 问题与目标

**设定：** 整个方法免训练；DINOv3-L/16、读出器和选择器均不训练、不微调。输入一张参考图、它的完整前景 mask 和一张查询图，输出查询图的完整 mask。查询 GT 只用于评分和标明的诊断。

**研究问题：** 只有一张参考图和它的 mask，用冻结特征，查询里的目标什么时候能被识别、靠什么机制识别，又在什么情况下确实识别不了。具体分三层（[W45](archive/x1012_tasks/W45_solvability_PLAN.md)）：
1. **表示：** 冻结特征本身能否把目标和其余部分分开。分不开的例子，才是这层表示下的难例。
2. **迁移：** 参考图定义的“目标是什么”，怎样在另一张图里重新找到。
3. **读出：** 找到之后，怎样决定范围（取多少、到哪里为止）。

**研究的做法：**
- 先弄清每一层在哪些例子上失败、为什么失败，再针对失败机制设计方法。
- 判断一个例子“做不出来”，必须给出证据（信息不在给定输入或特征里），不能以别的方法也失败为据。
- 方法的取舍看它解决了哪一层的问题、解决了多少，而不是看在哪个基准上领先谁。

**外部检验：** 在十个公开基准上与已发表方法同协议比较，以此检验方法是否真的解决了问题。比较结果是检验，不是研究对象，也不决定先研究哪个问题。

- **FoRIS 是当前同协议工作基线，不是 SOTA 终点。** [原始论文](https://arxiv.org/abs/2609.03384) v1 提交于2026-09-03，项目原记录的官方代码为 commit 1aa02a1。投稿还需正面比较 INSID3、RSRM、FROST、FSSDINO，不能把单一基线当作穷尽性 SOTA 证明，见 [W35 文献与贡献审计](x1012_tasks/W35_cvpr2027_readiness_REPORT.md)。FoRIS、INSID3与SH的机制差异、已有失败与后续问题见 [W64方法复盘](x1012_tasks/W64_reference_head_graph_REPORT.md)。
- **十个基准和 FoRIS 发表值**（表 1、表 9，one-shot）：

| COCO | LVIS | PASCAL-Part | PACO-Part | SUIM | iSAID | ISIC | Lung | Fundus | DeepGlobe road |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60.9 | 42.8 | 55.8 | 42.3 | 59.1 | 53.6 | 62.9 | 87.6 | 19.7 | 13.3 |

- **Road 的协议无法复现。** 官方代码里没有 road 的数据构建器，抽样方式不明。我们用本地的 custom200，FoRIS 在上面重现为 27.26，与 13.3 不可比，论文中单独标注（见 [protocol_inventory](archive/autonomous_20261010/protocol_inventory.md)）。
- **当前完整官方账本已核实四域**：Fundus 19.13、ISIC 62.73、Lung 87.71，以及不可与发表值直接比较的 Road custom200 27.26。前三域与发表值相差在0.6以内。历史COCO4000曾重现60.93，需找回迁移账本并核验才能用于当前配对；其余域不能写成已完成全量重现，见 [W35 §1.4](x1012_tasks/W35_cvpr2027_readiness_REPORT.md)。

## 2 当前方法：SH（不依赖 FoRIS 的头）

定义和代码见 [W23_simple_pipeline.md](x1012_tasks/W23_simple_pipeline.md)，实现为 `scripts/x1012/w23_simple_pipeline.py::simple_head`。全部常数见 [server/EXPERIMENT_PLAN.md](../../server/EXPERIMENT_PLAN.md) 第 2 节。

1. 位置去偏后的参考和查询 token；参考覆盖率 y；rho = 参考前景中不落在纯前景 token 上的比例。
2. **点证据 U0。** 先做核迁移 c，即温度 .05 的软前景份额。若 rho > .9，直接输出 minmax(up(c)) > .5（细结构回退，如 Fundus）。否则：
   - 用新奇度调制 c，调制强度按无标签的 max-J 逐例选择；
   - 在查询 kNN 图 L 上做 λ=16 的求解。
3. **原型证据 M′。** p = q·μ_fg − .55·q·μ_bg⊥，即一个前景原型对一个难负背景原型。在**同一张图 L** 上做排序引导的 λ=16 求解。
4. **SH+** = 官方 CRF([.5·M′ + .5·U0 > .5])。

**一句话叙事：** 点证据（逐 token 迁移）与原型证据（类级对比）的错误互补；共用一张查询图融合，不需要 FoRIS 的聚类先验、投票和语义修正，就能更准、更快。

## 3 方法是怎么走到这里的

| 阶段 | 做了什么 | 留下什么 |
|---|---|---|
| 10-05～10-11 | 在 FoRIS 上做增强（MEAN、RCG、fine、N、九窗 S） | 能涨分，但都依赖 FoRIS，独有贡献不成立（对同预算控制只多 0.01） |
| 10-12 | 独立基座 IND/C1；R1–R5；token 读出上界分析 | C1 五正五负。关键认识：排序不差，差在范围（切点） |
| 10-13 | 独立候选 U 在 13,613 例上测量（W01–W10）；Astra 的 x1013 T0–T9 | U 在细粒度目标上赢、大目标上输；所有变体落在同一条整合尺度轴上；按大小选尺度失败。E2 在 S 半选出 MFU = CRF(.5·FoRIS 线 M + .5·U)，T 半一次性检验 +3.16，十域全正 |
| 10-14 上午 | MFU 新照片留出（一次性）；官方协议在本机开跑后停止 | 新照片 +2.62 [1.96, 3.24]，七域显著正，零域显著负，iSAID −0.71 不显著 |
| 10-14 下午 | W13–W18 余量解剖；W19–W25 用简单原型替换 FoRIS | SH 与 MFU 打平，但不用 FoRIS、更快；W24 常数在局部最优；W25 max-J 选尺度关闭 |

## 4 当前证据

### 4.0 W30 确认：冻结的 SH+，新照片留出与官方协议（10-15，每部分只评分一次）

**E1 新照片留出（3,211 例，双角色零曝光；FoRIS、MFU 用已封存的逐例预测配对）：**
- SH+ 55.92，FoRIS 52.79，MFU 55.41；**SH+ − FoRIS = +3.13 [2.28, 3.74]**；SH+ − MFU = +0.50 [−0.02, 0.92]（含 0）。
- 逐域：6 域显著高于 FoRIS（LVIS、PASCAL-Part、PACO-Part、Lung、Fundus、Road）；COCO +0.11、SUIM +3.23 区间含 0；**iSAID −0.53、ISIC −0.90 为负但不显著**。按计划读法：无一域显著低于 FoRIS。
- **δ = 1 pp 逐域非劣效不成立**（COCO、iSAID、ISIC 的单侧下界 −1.04、−2.49、−2.61）。
- Fundus 全部走回退路径，SH+ 与 MFU 逐例相同；该域优势不来自融合。去掉 Fundus 的九域点估计 +2.19（事后计算，无区间）。

**E2 官方协议（完成 6 域；iSAID 8,142/10,063 未封存，PASCAL-Part、PACO-Part、LVIS 未跑）：**

| 域 | SH+ [照片区间] | FoRIS 发表值 | FoRIS 重现 | SH+ − 重现 |
|---|---|---:|---:|---|
| COCO | 60.64 [59.48, 62.00] | 60.9 | — | — |
| SUIM | 59.35 [57.68, 61.00] | 59.1 | — | — |
| ISIC | 61.75 [58.33, 64.99] | 62.9 | 62.73 | −0.97 [−2.21, 0.35] |
| Lung | 90.07 [89.51, 90.57] | 87.6 | 87.71 | **+2.35** [1.94, 2.83] |
| Fundus | 34.22 [33.47, 34.93] | 19.7 | 19.13 | **+15.10** [14.20, 16.03] |
| Road custom200 | 29.84 [27.09, 32.37] | — | 27.26 | +2.58 [1.02, 4.12]，图像双向 [−0.10, 5.16] |

- Lung、Fundus 显著高于发表值；SUIM 不低于；COCO、ISIC 略低、不显著。没有一域显著低于发表值。
- ISIC 上 SH+ 显著低于 MFU（−1.77 [−2.72, −0.73]）。
- COCO 官方例曾作开发数据，不是留出。

详见 [W30 报告](x1012_tasks/W30_mac_confirmation_REPORT.md) 第 3–5 节。

### 4.1 其他已有证据

- **S/T 子样本（每域 ≤150 例，CRF 后）：** SH+ − FoRIS 十域均值 S 半 +2.80、T 半 +3.28；SH+ − MFU 为 −0.02 / +0.11。逐域表见[整理前的 OVERVIEW](archive/OVERVIEW_20261016_before_cleanup.md) §4.1。
- **MFU：**
  - 新照片留出 MFU − FoRIS = +2.62 [1.96, 3.24]（[PHOTO_HOLDOUT_RESULT](x1012s_tasks/PHOTO_HOLDOUT_RESULT.md)）；
  - 官方协议四域：Fundus 34.22、ISIC 63.52、Lung 89.57、Road 28.61。

### 4.2 速度（同一套计时协议：CPU 2 线程，特征在内存中，包含 CRF）

| 方法 | 头部时间 | 相对 FoRIS |
|---|---:|---:|
| SH+ | 1.28 s | 0.82× |
| 完整 FoRIS | 1.56 s | 1.00× |
| MFU | 2.16 s | 1.39× |
| SH（不做 CRF） | 0.29 s | 0.19× |

- 双图编码另需约 3 s。
- CRF 约占 1 s。C640 可把它降到 0.46 倍，准确率不变（W21）。

### 4.3 融合优于单线（W27、W28，仅 S 半）

- **原批 1,285 例（十域，CRF 后）：** SH+ − U0+ = **+1.95 [1.31, 2.41]**，SH+ − M′+ = **+3.91 [3.14, 5.24]**。这一优势在 CRF 前已存在；在非回退子集上也成立。
- **另一批 871 例（六域，与原批查询照片不重合）：** 两项分别为 +2.41 [1.77, 3.14]、+3.74 [2.54, 5.08]。
- 不能写逐域全优：SUIM、ISIC 相对 M′ 的点估计为负，LVIS 与 U0 持平。这是开发集结论，不是独立确认。见 [W27](x1012_tasks/W27_sh_single_line_REPORT.md)、[W28](x1012_tasks/W28_sh_second_batch_REPORT.md)。

## 5 论文主张与状态

| # | 主张 | 状态 |
|---|---|---|
| C1 | 跨域平均高于 FoRIS，并报告逐域取舍 | **E1 新照片留出成立：** +3.13 [2.28, 3.74]，6 域显著更高，无一域显著更低；iSAID、ISIC 点估计为负。δ = 1 逐域非劣效不成立，不写“十域都不低于”。Fundus 优势来自与 MFU 共用的回退路径。E2 完成 6 域，无一域显著低于发表值；其余 4 域未跑。与 MFU 相比没有可检出的差别（+0.50，含 0） |
| C2 | 点证据与原型证据的固定融合总体优于两条单线，不使用 FoRIS 的分数、投票和聚类先验 | [W27严格匹配消融](x1012_tasks/W27_sh_single_line_REPORT.md)：原S批十域CRF后 +1.95/+3.91 pp；[W28换批复测](x1012_tasks/W28_sh_second_batch_REPORT.md)：新六域 +2.41/+3.74 pp，各95%区间均为正，CRF前及非回退子集亦支持。限定为开发S总体结论；当前W30只产出SH/SH+，单线独立确认尚缺 |
| C3 | 单线的失败是尺度性的：点线在大目标上漏范围，区域线在小目标上误报 | 成立（W09、W11、W24） |
| C4 | 已测试的按大小/代理选尺度构造未成功，固定融合有效 | 具体构造证据见 W09、T5、S1–S3、R1–R3、W25、W26-0；不推成任意选择器均不可能 |
| C5 | 更简单、头部更快 | 本机CPU缓存特征头部成立（0.82×）；不是整链0.82×，服务器 E3 补 GPU 端到端计时 |

不写的主张：
- 十域点估计全部为正；
- 拓扑恢复；
- 左右身份；
- 按数据集调参。

详见 [PAPER_EVIDENCE_MAP_20261014](PAPER_EVIDENCE_MAP_20261014.md)。

## 6 现状与当前任务（10-16 复盘后）

**已成立（范围见第 4、5 节）：**
- SH+ 在新照片留出上比完整 FoRIS 高 +3.13 [2.28, 3.74]（W30）；与 MFU 没有可检出差别（+0.50，区间含 0）。
- 两线固定融合优于各自单线，两批 S 半都成立（W27、W28）。
- Fundus 回退路径的优势在两批新眼底图上复现（[W54](x1012_tasks/W54_fundus_fresh_PLAN.md)）。[W69](x1012_tasks/W69_fundus_mechanism_REPORT.md) 只重析已有账本，查清了一部分机制：
  - 末层的增益来自误报减少 85%–89%，代价是召回降 25–27 pp；第 19、20 层仍赢，且召回更高；
  - 不只是切点：旧 S 半 108 例中，98 例 SH 默认输出高于 FoRIS 场的逐例最佳阈值；同一场做区域平均会丢信息（34.05 → 28.15）；
  - 软标签、去位置、联合中心化各自的作用尚未隔离。

**未达成：**
- 在 SH 之上，还没有任何新机制带来稳定增益；
- A′ + B > A + B（不用 FoRIS 线而超过 MFU）尚未证明。

**仍然站得住的诊断：**
- **灾难例分组（[W58](x1012_tasks/W58_score_repair_REPORT.md)）：** 261 个灾难例中，69 个现有场上有好切点；107 个现有场达不到 .5，但查询监督探针能达到；85 个两者都未解决。这是操作性分组，不是因果份额。
- **参考确实在起作用：**
  - 同 RGB 换任务时，62/63 对的输出朝正确目标变化，但两个任务都达到 .5 的只有 38/63（[W63](x1012_tasks/W63_task_switch_REPORT.md)）；
  - 同一查询换合法参考，22/89 个灾难例能被救回（[W61](x1012_tasks/W61_reference_stability_REPORT.md)）。
- **范围与归属读不出来：**
  - 每例一个切点或面积的规则，受“80% 的例子不该动、20% 要移约 1.44 倍频”和陡峭的 IoU 限制（[W65](archive/x1012_tasks/W65_extent_headroom_REPORT.md)）；
  - 按区域判断归属时，现有缓存量都不比场值本身更好，G 之外的少量信息偏参考侧；部件域里查询内成组与归属方向相反（[W66](archive/x1012_tasks/W66_region_attribution_REPORT.md)）；
  - 第 19 层参考内外缘对比，在外延带上有 G 和末层之外的少量排序信息（+.063 [.018, .109]），但在真正该补的例子上未确认；按符号直接改，24 个原本正确的例子坏了 22 个（[W70](archive/x1012_tasks/W70_midlayer_attribution_REPORT.md)）。
- **保本门槛高，三次独立量到：**
  - 整片区域加入或去掉（W66，事后，token 级）：只在约 19% 的例子里有益，改对 +16～19、改错 −24～26，保本精度约 .6；
  - W67 对应修正：保本需 70% 改对，实际 50%；
  - W70 零点读法：保本需 60%，实际 24%。
- **读出缺口因域而异（W69 §9）：** Lung、Fundus 的默认输出已接近本场最佳阈值；其他域缺口较大；Road 两种场的最佳阈值都只有约 36。
- **计量注意：**
  - 近纯 token 的 AUC 只覆盖很少的细结构前景，例如 Fundus 正类中位只有 2.2%；
  - SUIM 有已确认的标签解码丢失例（[W63 源核查](x1012_tasks/W63_failure_source_audit_REPORT.md)）。

**卡在哪里：** 用缓存的末层特征对 SH 的场做局部事后修正，这一族已经用尽；信息本身没有用尽（[W72](x1012_tasks/W72_research_review_REPORT.md) 第 2 节）。

**当前任务（复盘见 [W72](x1012_tasks/W72_research_review_REPORT.md)，子问题见 [W73](x1012_tasks/W73_subproblems_PLAN.md)）：**
- **周末离线（最新约束：25GiB内存、CPU/GPU持续负载预算80%，无网络、全自动）：** [W71运行包及命令](x1012_tasks/W71_weekend_offline_REPORT.md)已交付，覆盖W30剩余43,380任务及Fundus三因素对照。9项工程测试通过，科学路径14例实测通过；最终25GiB包由启动链重做完整自检，通过后才计时。用户将让Opus检查，当前未启动。
- **最近邻同协议对比：** FROST、INSID3、RSRM、FSSDINO，先在 S 开发集上跑，再上官方协议。作者代码都在 `../cv_data/third_party/`。
- **子问题，一个数据集一个数据集地做（[W73](x1012_tasks/W73_subproblems_PLAN.md)）：**
  - 顺序：P1 ISIC 单病灶范围读出 → P2 COCO 整物体响应 → P3 iSAID 多个小实例 → P4 部件；
  - 每个数据集内一个方法一个方法地比：SH 各线、FoRIS、MFU、INSID3、FROST、RSRM、FSSDINO；
  - 最终方法仍是一套，可以包含由可观测量触发的子机制（像 rho 回退），不按数据集调参。

## 7 已关闭的路线（失败总结）

每行是一个具体构造的结论，不是方向上限。“缺什么”一列：
- **迁移**：信号与正确修改的关系没有从参考迁移到查询；
- **净账**：改对的收益抵不过改坏的损失；
- **信息**：所用的量本身不含额外判别信息。

旧文档都在 [archive/x1012_tasks/](archive/x1012_tasks/)，一般不需要打开。

| 路线 | 结果 | 缺什么 | 出处 |
|---|---|---|---|
| FoRIS 上的读出增强（MEAN、RCG、fine、残差、N、九窗 S） | 能涨分，但依赖 FoRIS，独有贡献不成立 | — | [RESULTS](archive/RESULTS.md) |
| 角色、几何、判别、任务核、层级、context 等 20 余个固定构造 | 都低于强控制 | 信息 | [RESULTS §2](archive/RESULTS.md) |
| 切点线索：Otsu、MSER、sweep、面积估计、逆预测、监督融合、中间层、金字塔、OT | 不优于常数。多数线索在自然面积之外无信号；J、逆 Simpson 有弱方向信号，但被“该不该动”淹没 | 信息、净账 | [TOKEN_READOUT_CEILING](archive/TOKEN_READOUT_CEILING_20261012.md)、[W65](archive/x1012_tasks/W65_extent_headroom_REPORT.md) |
| 按数据集、参考面积、预测面积、FoRIS 面积、max-J 选尺度；监督选择器 | 未通过。欠分割的大目标本身呈现小预测面积，形成反馈错误 | 信息 | [W09/W10](archive/x1012_tasks/W09_W10_cut.md)、[T0_T5](x1013_tasks/T0_T5_reading.md)、W25、W26 |
| 每例单一切点/面积规则（W65） | 无量过门槛。即使用 GT 完美判断哪些例子该动，也只收回约 1/12 的余量 | 净账 | [W65](archive/x1012_tasks/W65_extent_headroom_REPORT.md) |
| 查询自身分组、线间选择、漏实例、相对峰值去干扰物 | 关闭 | 信息 | [W15/W16](archive/x1012_tasks/W15_W16_extent.md) |
| 聚类先验、CRF 的廉价替代；单一求解；常数调整 | 大目标与小目标/部件之间的域间交换 | 净账 | [W20/W21](archive/x1012_tasks/W20_W21_prior_crf.md) |
| 整体示范匹配：遮罩候选与参考比较冻结 CLS（W31） | −10.17，偏小范围。好候选存在（GT 选 +7.56），但选不中 | 信息 | [W31](archive/x1012_tasks/W31_demonstration_PLAN.md) |
| 末层 key / O24 谱分组（W32） | 用 GT 选片段的 oracle 仍低于切点 oracle | 信息 | [W32](archive/x1012_tasks/W32_key_entity_PLAN.md) |
| 边界带覆盖量：核迁移、解混、颜色切点、参考边界字典（W37/38/40） | −3.05 / −1.73 / −2.91。局部相关不等于 IoU | 净账 | [W37](archive/x1012_tasks/W37_coverage_PLAN.md)、[W38](archive/x1012_tasks/W38_band_amount_PLAN.md)、[W40](archive/x1012_tasks/W40_band_dictionary_PLAN.md) |
| 带内有序链最小割（W42） | 近恒等，能量没有动作空间 | — | [W42](archive/x1012_tasks/W42_ordered_cut_PLAN.md) |
| 参考关系森林（W44） | 约束从不触发，中止 | 迁移 | [W44](archive/x1012_tasks/W44_relation_forest_PLAN.md) |
| 转导纠偏：以 SH 场为带噪标签拟合岭回归（W46） | −1.84 | 信息 | [W46](archive/x1012_tasks/W46_transfer_PLAN.md) |
| 改原型线 M′ 以减少传播误检（W55） | M′ 在非灾难例中同样参与，立项依据不成立 | 净账 | [W55](archive/x1012_tasks/W55_propagation_FP_PLAN.md) |
| 去掉参考先验（W57、W61） | 渗漏大幅下降，但 mIoU −0.95 / −1.07。先验本身是信息 | 迁移 | [W57](archive/x1012_tasks/W57_reference_prior_PLAN.md)、[W61](x1012_tasks/W61_reference_stability_REPORT.md) |
| 去 minmax、用绝对分数（W59） | −0.47 至 −2.1。删 FP 的同时删掉几乎等量的 TP | 净账 | [W59](archive/x1012_tasks/W59_normalization_causal_REPORT.md) |
| 纯背景负核、原型负字典（W60/W62） | 局部 AUC 上升，净 +0.01 至 +0.08，不显著 | 净账 | [W60](archive/x1012_tasks/W60_pure_background_REPORT.md)、[W62](archive/x1012_tasks/W62_background_channels_REPORT.md) |
| 参考选择 key head 图（W64） | −0.18。参考上的风险下降与查询增益无关（ρ = −.04） | 迁移 | [W64](x1012_tasks/W64_reference_head_graph_REPORT.md) |
| 一次局部对应修正（W67，β = 4、半径 4） | −1.07 [−2.20, −0.04]，换参考 −1.62；逐例最佳阈值也无增量。改好 84 例、改坏 85 例，改坏的幅度更大 | 信息、净账、迁移 | [W67](archive/x1012_tasks/W67_correspondence_validation_REPORT.md) |
| 第 19 层参考内外缘对比的零点读法（W70） | 外延带排序 +.063，但不在该补的例子上；按符号改，正确例 22/24 受损；换参考仍会失效 | 净账、迁移 | [W70](archive/x1012_tasks/W70_midlayer_attribution_REPORT.md) |
| 局部区域归属量：参考相似、新颖、反向注意力、边缘类比、近邻、聚类、颜色、边的位置（W66，两条分支） | 无量超过场值本身；G 之外参考侧剩 .58–.64、查询侧 .52–.56 | 信息 | [W66](archive/x1012_tasks/W66_region_attribution_REPORT.md) |
| 训练式读出或选择器（C0）；引入 SAM 等新模型 | 用户否决，不在设定内 | — | 第 1 节 |

**已撤回的解释：**
- “表示或分辨率不够”（W45/W49 的计分错误）；
- Fundus“硬 token 上限 24”；
- “FoRIS 整块判断，我们是软渲染”（两者渲染方式相同）；
- “单参考天然有歧义”（6,200 例审计未见多答案）；
- “排序 AUC 很高，所以只缺范围”（AUC 的覆盖面不足）。

## 8 研究规则

- **数据使用：**
  - S 半是开发数据，可以自由迭代；
  - T 半每个方法只读一次；新照片留出只评分一次（MFU、SH 已各用过一次）；
  - 确认前写定方法和读法，跑完不改。
- **每个候选都看：**
  - 分域 mIoU；
  - 四项编辑账本：新增 TP / FP、删除 TP / FP；
  - W61（同查询换参考）、W63（同 RGB 换任务）上的变化。
- **贡献怎么算：** 研究时任何信息都可以用。写论文时，翻转、位移、提高分辨率、多视图这类通用推理设置不算方法贡献；若使用，对比方法也要同样使用（W29 据此取消）。
- **指标：**
  - 开发：(fold, class) 槽池化后等权，再十域等权；照片成组 bootstrap。
  - 官方：每个 fold 内类 IoU = ΣI/ΣU，对类平均，再对 fold 平均；CLI-1024 画面。
  - 两种口径不能相减。
- **冻结文件：** `r5_candidate.py`、`w23_simple_pipeline.py`、`w27_sh_single_line.py`、`scripts/x1013/*`、`x1013_tasks/REPORT.md`、`scripts/x1012s/photo_holdout.py`、x1012_* 产物。

## 9 文档地图

| 想知道 | 读 |
|---|---|
| 现行方法定义 | [W23_simple_pipeline](x1012_tasks/W23_simple_pipeline.md) |
| 论文主张与证据 | [PAPER_EVIDENCE_MAP_20261014](PAPER_EVIDENCE_MAP_20261014.md)、[W27](x1012_tasks/W27_sh_single_line_REPORT.md)、[W28](x1012_tasks/W28_sh_second_batch_REPORT.md)、[W30](x1012_tasks/W30_mac_confirmation_REPORT.md) |
| 最近邻与贡献边界；FoRIS、INSID3、SH 的机制比较 | [W35](x1012_tasks/W35_cvpr2027_readiness_REPORT.md)、[W64](x1012_tasks/W64_reference_head_graph_REPORT.md) |
| 精确计分与理论裁决 | [W58 修复](x1012_tasks/W58_score_repair_REPORT.md)、[W58 理论审查](x1012_tasks/W58_theory_archive_REPORT.md) |
| 迁移检验用的现成数据 | [W61](x1012_tasks/W61_reference_stability_REPORT.md)、[W63](x1012_tasks/W63_task_switch_REPORT.md) |
| 血管优势的机制；各域读出缺口 | [W69](x1012_tasks/W69_fundus_mechanism_REPORT.md) |
| 复盘与后续方向 | [W72](x1012_tasks/W72_research_review_REPORT.md) |
| 新照片留出；x1013 链（MFU 的来历） | [PHOTO_HOLDOUT_RESULT](x1012s_tasks/PHOTO_HOLDOUT_RESULT.md)、[x1013_tasks/REPORT](x1013_tasks/REPORT.md) |
| 流水记录 | [STATUS](STATUS.md) |
| 服务器 | [server/README.md](../../server/README.md) |
| 已关闭或被取代的全部历史文档 | [archive/](archive/README.md)，含 [archive/x1012_tasks/](archive/x1012_tasks/)；失败总结见第 7 节 |

## 10 术语

| 名字 | 含义 |
|---|---|
| O24 | 冻结 DINOv3-L/16 最后一层 token。1024 输入，64×64 个 token，每个 1024 维 |
| 去位置 | 把特征投影出位置子空间再归一化。位置子空间来自黑图特征（`positional_basis.pt`，500 维通道子空间） |
| y（coverage） | 参考 mask 在每个 16×16 块里的前景比例，保留软值 |
| rho | 参考前景中不落在纯前景 token（y ≥ .9）上的比例。> .9 时走细结构回退：直接输出 minmax(up(c)) > .5，不做 CRF |
| c / T | 核迁移场：每个查询 token 对参考 token 的软前景份额（温度 .05）。T 是直接对 c 取阈值的读出 |
| max-J | 用无标签的 J = P·C/(P + C − P·C) 选择新奇调制的强度 |
| U0 | 点证据线：调制后的 c 在查询 kNN 图 L 上做 λ=16 求解 |
| M′ | 原型证据线：前景原型对难负背景原型的对比，在同一张图 L 上做排序引导求解 |
| SH / SH+ | .5·M′ + .5·U0 > .5；SH+ 再做一次官方 CRF。现行方法 |
| FoRIS | SOTA 基线，指完整方法：前景/背景处理、投票、聚类先验和后处理 |
| MFU | CRF(.5·FoRIS 线 + .5·U)，依赖 FoRIS 的旧主候选 |
| MEAN / fine | 10-10 前在 FoRIS 上的增强：图求解，以及四个位移编码的细读出。已不作为方法 |
| S 半 / T 半 | 按查询照片哈希分成的两半开发数据：S 用于选择，T 用于一次性检验 |
| 新照片留出 | 3,211 例，参考图和查询图都没在开发中出现过。每个方法只评分一次 |
| 官方协议 | FoRIS 代码中的 episode 列表、fold 和指标，共 60,981 例。主表只能用它 |
