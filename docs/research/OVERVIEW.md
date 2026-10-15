# 研究总览（2026-10-14）

这是研究的唯一入口：当前方法、当前证据、哪些路线已经关闭、下一步做什么。

- 其他文档都是细节或历史，从下面的链接进入。
- 本文件与旧文档冲突时，以本文件为准。
- 旧文档里的“下一步”“正在运行”“唯一候选”等表述，都是当时的快照。

## 1 问题与目标

**设定：** 冻结 DINOv3-L/16（不训练、不微调），输入一张参考图、它的完整前景 mask 和一张查询图，输出查询图的完整 mask。查询 GT 只用于评分和标明的诊断。

**目标：** 用同一条规则，在十个基准上超过 FoRIS；同时做到更简单、更快，并有可写成论文的贡献与叙事。

- **FoRIS 就是当前 SOTA。** Astra 已核实：arXiv 2609.03384，2026-09-04 发布，官方代码 commit 1aa02a1。
- **十个基准和 FoRIS 发表值**（表 1、表 9，one-shot）：

| COCO | LVIS | PASCAL-Part | PACO-Part | SUIM | iSAID | ISIC | Lung | Fundus | DeepGlobe road |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60.9 | 42.8 | 55.8 | 42.3 | 59.1 | 53.6 | 62.9 | 87.6 | 19.7 | 13.3 |

- **Road 的协议无法复现。** 官方代码里没有 road 的数据构建器，抽样方式不明。我们用本地的 custom200，FoRIS 在上面重现为 27.26，与 13.3 不可比，论文中单独标注（见 [protocol_inventory](autonomous_20261010/protocol_inventory.md)）。
- **其余九个基准**在官方协议下的 FoRIS 重现，与发表值相差在 0.6 以内：Fundus 19.13、ISIC 62.73、Lung 87.71。

## 2 当前方法：SH（不依赖 FoRIS 的头）

定义和代码见 [W23_simple_pipeline.md](x1012_tasks/W23_simple_pipeline.md)，实现为 `scripts/x1012/w23_simple_pipeline.py::simple_head`。全部常数见 [server/EXPERIMENT_PLAN.md](../../server/EXPERIMENT_PLAN.md) 第 2 节。

1. 位置去偏后的参考和查询 token；参考覆盖率 y；rho = 参考前景中不落在纯前景 token 上的比例。
2. **点证据 U0。** 先做核迁移 c，即温度 .05 的软前景份额。若 rho > .9，直接输出 c > .5（细结构回退，如 Fundus）。否则：
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

### 4.1 SH+ 相对完整 FoRIS（本机，固定种子子样本，每域 ≤150 例，CRF 后）

| 域 | S 半（选择用） | T 半（一次性） |
|---|---:|---:|
| COCO | −0.40 | −0.28 |
| LVIS | +3.58* | +4.42* |
| PASCAL-Part | +2.73* | +3.01* |
| PACO-Part | +3.41* | +3.64* |
| SUIM | −1.01 | +2.14 |
| iSAID | +2.60* | −0.26 |
| ISIC | −0.77 | −0.80 |
| Lung | +2.46* | +2.27* |
| Fundus | +15.35* | +15.01* |
| Road | +0.03 | +3.65* |
| **十域均值** | **+2.80** | **+3.28** |
| SH+ − MFU | −0.02 | +0.11 |

`*` 表示照片 bootstrap 95% 区间不含 0。

### 4.2 其他已有证据（方法都是 MFU，不是 SH）

- **新照片留出**（3,211 例，双角色零曝光，一次性评分）：MFU − FoRIS = +2.62 [1.96, 3.24]；MFU − U = +0.48 [0.11, 0.85]。详见 [PHOTO_HOLDOUT_RESULT](x1012s_tasks/PHOTO_HOLDOUT_RESULT.md)。
- **官方协议**（在本机完成 4 个小域后停止）：

| 域 | FoRIS 重现 | MFU | 发表值 |
|---|---:|---:|---:|
| Fundus | 19.13 | 34.22 | 19.7 |
| ISIC | 62.73 | 63.52 | 62.9 |
| Lung | 87.71 | 89.57 | 87.6 |
| Road custom200 | 27.26 | 28.61 | — |

  其余六域留给服务器。

### 4.3 速度（同一套计时协议：CPU 2 线程，特征在内存中，包含 CRF）

| 方法 | 头部时间 | 相对 FoRIS |
|---|---:|---:|
| SH+ | 1.28 s | 0.82× |
| 完整 FoRIS | 1.56 s | 1.00× |
| MFU | 2.16 s | 1.39× |
| SH（不做 CRF） | 0.29 s | 0.19× |

- 双图编码另需约 3 s。
- CRF 约占 1 s。C640 可把它降到 0.46 倍，准确率不变（W21）。

## 5 论文主张与状态

| # | 主张 | 状态 |
|---|---|---|
| C1 | 十个基准上不低于 FoRIS，均值显著更高 | SH 只有本机子样本证据，需要服务器 E1（新照片）和 E2（官方协议）确认。COCO、ISIC 的点估计略负，措辞预计为“无一域显著更低” |
| C2 | 点证据与原型证据互补，融合优于任一单线，不依赖 FoRIS 组件 | 本机成立：去掉调制 −1.31，单线都不够。服务器补单线消融 |
| C3 | 单线的失败是尺度性的：点线在大目标上漏范围，区域线在小目标上误报 | 成立（W09、W11、W24） |
| C4 | 按目标大小选尺度不可行，融合可行 | 成立（W09、T5、S1–S3、R1–R3、W25、W26-0） |
| C5 | 更简单、更快 | 本机成立（0.82×），服务器 E3 补 GPU 端到端计时 |

不写的主张：
- 十域点估计全部为正；
- 拓扑恢复；
- 左右身份；
- 按数据集调参。

详见 [PAPER_EVIDENCE_MAP_20261014](PAPER_EVIDENCE_MAP_20261014.md)。

## 6 已知缺口与进行中的工作

| 缺口 | 现状 | 去向 |
|---|---|---|
| SH 没有官方协议和新照片成绩 | 计划已写好，代码还没写 | [server/EXPERIMENT_PLAN.md](../../server/EXPERIMENT_PLAN.md)：E0 一致性门槛 → E1 → E2 → E3 |
| COCO、ISIC 略低于 FoRIS | W26-1 三项实现检查通过；S-W26 1,750 例上，A+B+C+D 的 CRF 后留一域选择相对 SH+ 十域均值 −0.045 pp，COCO +0.25、ISIC +0.45 pp，仍低于 FoRIS（−0.20、−0.25 pp）；iSAID 显著下降，单族也不满足目标域条件 | [W26 结果](x1012_tasks/W26_scale_PLAN.md#结果w26-12026-10-14)：按 5.5 关闭路线 C，不进入 W26-2，T 半未读。SH+ 维持主方法，服务器计划不变 |
| iSAID 在新照片上 −0.71（MFU） | 官方 10,063 例会给出更准的估计 | 服务器 E2 |
| CRF 在 iSAID、Lung、Road 上为负 | 方法冻结，不按域开关 | 作为消融如实报告 |
| 约 10 个点的切点余量（W14） | 两条线共享；查询分组、线间选择、漏实例、干扰物四条路都已关闭 | 没有新机制之前不开实验 |

## 7 已关闭的路线（不要重做）

| 路线 | 结论 | 出处 |
|---|---|---|
| FoRIS 上的读出增强（MEAN、RCG、fine、残差、N、S） | 涨分但依赖 FoRIS，独有贡献不成立 | [RESULTS](RESULTS.md)、[RESEARCH_GUIDE](RESEARCH_GUIDE.md) |
| 角色、几何、判别、任务核、层级、context 等 20 余个固定构造 | 都低于强控制 | [RESULTS §2](RESULTS.md) |
| 切点线索：Otsu、MSER、sweep cut、面积估计、逆预测、监督融合、中间层、金字塔、OT | 不优于常数，或低于 FoRIS | [TOKEN_READOUT_CEILING](TOKEN_READOUT_CEILING_20261012.md) §3–4 |
| 按数据集、参考面积、预测面积、FoRIS 面积、max-J 选尺度；监督选择器（现有量） | 全部失败；信息不在面积类量里 | W09/W10、[T0_T5](x1013_tasks/T0_T5_reading.md)、[ADAPTIVE](x1013_tasks/ADAPTIVE_FOLLOWUP.md)、[FIXED_AREA](x1013_tasks/FIXED_AREA_COMPOSITIONS.md)、W25、W26-0 |
| 查询自身分组、线间选择、漏实例、相对峰值去干扰物 | 关闭 | [W15_W16_extent](x1012_tasks/W15_W16_extent.md) |
| 聚类先验的廉价替代、CRF 的廉价替代、单一求解、常数调整 | 是域间交换或未达门槛 | [W20_W21](x1012_tasks/W20_W21_prior_crf.md)、W22、W24 |
| 引入 SAM 等新模型 | 用户否决：先用同一套特征把问题做干净 | — |

## 8 规则

- **数据使用：**
  - S 半用于选择；T 半每个方法只读一次；
  - 新照片留出只评分一次（MFU 已用过一次，SH 的那一次在服务器 E1）；
  - 读法必须在运行前写下。
- **指标：**
  - 开发时：(fold, class) 槽池化后等权，再十域等权；照片成组 bootstrap。
  - 官方：每个 fold 内类 IoU = ΣI/ΣU，对类平均，再对 fold 平均；CLI-1024 画面。
  - 两种口径不能相减。
- **算力：**
  - 本机（M4 Max，14 核）只找方法：每域最多 150 例的固定种子子样本，最多 4 个 worker，留出约 30% CPU，同一时间只跑一个 MPS 编码进程。
  - 全量运行、重新编码和正式计时都在服务器（RTX 4080 Super 32GB）上做。
- **文件：**
  - 生成数据写到 `../cv_data/a/`；
  - 小证据放在 `evidence/research/`；
  - 不要改冻结文件：`r5_candidate.py`、`scripts/x1013/*`、`REPORT.md`、`scripts/x1012s/photo_holdout.py`、x1012_* 产物；
  - 删除和移动先问用户。
- **环境：** `~/miniconda3/envs/aidemo/bin/python`，在仓库根目录运行。

## 9 文档地图

| 想知道 | 读 |
|---|---|
| 术语（O24、coverage、U/M/T、rho、max-J……） | [RESEARCH_GUIDE §2](RESEARCH_GUIDE.md) |
| 现行方法定义 | [W23_simple_pipeline](x1012_tasks/W23_simple_pipeline.md) |
| 论文主张与证据 | [PAPER_EVIDENCE_MAP_20261014](PAPER_EVIDENCE_MAP_20261014.md) |
| 服务器上要做什么、要复制什么 | [server/README.md](../../server/README.md)、[server/EXPERIMENT_PLAN.md](../../server/EXPERIMENT_PLAN.md) |
| Astra 的下一个任务 | [W26_scale_PLAN](x1012_tasks/W26_scale_PLAN.md) |
| 10-13 之后每一步的报告 | `x1012_tasks/` 的 W01–W26：W01_W02、W03、W04_W07、W09_W10、W11、W14、W15_W16（含 W13、W17、W18）、W19、W20_W21、W23（含 W22、W24、W25）、W26 |
| Astra 的 x1013 链（T0–T9、E1/E2） | [x1013_tasks/REPORT](x1013_tasks/REPORT.md)、[T0_T5_reading](x1013_tasks/T0_T5_reading.md)、[E2_T_REPORT](x1013_tasks/E2_T_REPORT.md) |
| 新照片留出 | [PHOTO_HOLDOUT_PLAN](x1012s_tasks/PHOTO_HOLDOUT_PLAN.md)、[PHOTO_HOLDOUT_RESULT](x1012s_tasks/PHOTO_HOLDOUT_RESULT.md) |
| 按日期追加的流水记录 | [STATUS](STATUS.md) |
| 10-12 之前 34 个方法家族的结论 | [RESULTS](RESULTS.md)（10-12 快照） |
| 在别的机器上接续 | [CONTINUING_ELSEWHERE](CONTINUING_ELSEWHERE.md) |

其他按日期命名的顶层文档（`COMPOSITION_*`、`FIVE_DATASET_*`、`TASK_*`、`NEXT_EXPERIMENT_PLAN_*` 等，日期 ≤ 10-13），以及 [PLAN](PLAN.md)、[GAPS](GAPS.md)（10-12 快照），都是历史记录：用来查证据，不是待办。
