# 关键结果：快、强与净纠错

更新：2026-10-07。本页只保留主线证据；完整历史账本可从[清理前版本](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md)读取。
本页的历史原始报告、逐例数据、数组和封存记录没有改写。2026-10-07新增一次固定范围迁移探针，见下节。

## 2026-10-07：参考掩码范围迁移C1关闭

按本轮用户合同，只在已存RCG后增加3×3联合外观/完整掩码模板投票保留门；无新图求解或参数扫描。
PACO f0/f1开发299例：43.8453→39.9900，Δ−3.8554 [−6.0549,−1.8488]；f2/f3单次评测300例：41.5758→40.2521，Δ−1.3237 [−3.2098,−0.1350]。
COCO fresh600：64.5173→60.4391，Δ−4.0782 [−6.3494,−2.8081]。均为位置级class mIoU、1000次episode配对区间，旧数据再用，不是原图完整评测或独立新数据确认。
留出集补真/补假/删真/删假=0/0/31338/69625；逐类精确计价，删真−14.8498点、删假+13.5261点。对同匹配中心标签控制仅+0.2565 [−0.0687,+0.4752]，未建立结构掩码增量。
两道门均失败，固定C1关闭，不做变体；PASCAL-Part/SUIM/LVIS扩展按门槛未运行。只试一个候选；没有填满三个上限或复活旧负路线。
来源：[一页报告](astra_granularity_20261008/REPORT.md)、[固定协议](astra_granularity_20261008/PROTOCOL.md)、[完整统计](astra_granularity_20261008/candidate_C1/summary.json)、[精确增删分解](astra_granularity_20261008/analysis.json)。上一轮跨数据集核算及5例实现烟测在报告末尾单列。

## Strong：已有完整优势，尚非最终SOTA结论

同一公开4000抽样、1024读出、类别汇总I/U再平均；复用了开发数据，不是独立新数据确认。

| 完整构造 | mIoU | 能说明什么 |
|---|---:|---|
| FoRIS | 60.931741 | 完整主参照 |
| RCG / MEAN | 62.333671 / 62.512972 | 简单空间读出已带来明显增益 |
| fine16 / fine64 | 62.709100 / 62.651617 | 更复杂fine64没有胜过fine16 |
| 固定掩码增删转移 | 62.744357 | 相对MEAN +0.231385；不证明胜过fine16 |
| 固定标量残差转移 | 62.844537 | 相对MEAN +0.331565 [0.303193,0.372805] |
| 12源库外层留折配方 | 62.964699 | 相对FoRIS +2.032958 [1.657151,2.387729]；仍为开发复用 |
| 13源扩展库外层留折配方 | 63.102199 | 含sizecut_foldtemp；底层尺寸切点已使用fresh600标签，非整套独立留折确认 |

来源：[完整细读出读数](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md#2026-10-06-complete-frozen-fine4000-and-fixed-ab-transfer-control)、[固定残差](research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.md)、
[12源](research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/report.md)、
[13源](research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v4_extended_crossfold/report.md)。
并非没有全量输出；缺的是按当前主线确定的最终完整方法、同条件强对照及完整成本验证。
这些成绩不能挂给尚未运行的原始DINO新构造，也不能证明整个分割领域SOTA。

## 2026-10-06跨数据集结果（旧账本原文恢复）

来源：`82df9169a23c5f58ecf1ec57de237cddecc45ec4:evidence/local/RESULTS.md`。下列五段从该版本原样摘回，未重算。
例子清单是按各loader自抽的约600例（实际598/599/582等），不是论文官方主表的episode清单；COCO B+C+D合计1800例。
完整版的尺寸切分在COCO基类/开发折上拟合后冻结，没有在这些跨数据集上重拟合；full含读出和尺寸切分，不是当前未执行的L4候选。
指标为原图class mIoU，区间为2000次照片组配对重采样。旧文只给增量或对RCG区间的地方保留原口径，未用加法补造绝对数或缺失区间。

### PASCAL-Part

Pack of INSID3's own loader (150 per benchmark fold, seed 0, 56 dense classes; not INSID3's exact evaluation list; `fold` in
the table is the pack index modulo 4). Same frozen file as on COCO; FoRIS run fresh. Original resolution: FoRIS 53.87; RCG
55.33 = +1.46 [+0.67, +2.40]; RCG + feature-guided readout 55.35, against RCG +0.02 [-0.09, +0.10]; full version 54.83 =
+0.96 [-0.06, +2.50], against RCG -0.50 [-1.24, +0.57]. At 1024: FoRIS 52.03, RCG +1.60 [+0.92, +2.45], readout -0.01, full
-0.13 [-0.82, +0.76] against RCG. RCG transfers; neither auxiliary does. Source: `research_20261005/rcg2_grouppascal_part/`;
server `outputs/claude_rcg2_pascal_part`, pack `outputs/claude_packs/pascal_part`.

### COCO B+C+D

COCO groups B+C+D (1800): FoRIS 60.52; RCG 61.80 = +1.28 [+0.86, +1.77], folds +1.22/+0.78/+1.39/+1.74;
+readout 62.18 = +1.66 [+1.24, +2.19] (vs RCG +0.38 [+0.33, +0.46]); full 62.38 = +1.86 [+1.13, +2.63],
folds +1.48/+1.89/+1.41/+2.67 (vs RCG +0.58 [+0.04, +1.15]).

### SUIM

SUIM (582 episodes, pack `decision_transfer_pack.py --dataset suim`): FoRIS 60.95; RCG -0.08
[-1.01, +0.82], folds -0.45/-1.07/+1.03/+0.47; +readout +0.30 [-0.66, +1.23] (vs RCG +0.38
[+0.30, +0.47]); full -0.34 [-1.80, +1.14]. Unresolved: no gain over FoRIS.

### PACO-Part

PACO-Part (599 episodes = 4 per-fold packs of 150, 264 fold-classes, class ids offset per pack):
FoRIS 41.45; RCG +0.30 [-0.19, +0.75], folds +0.80/+0.05/+0.61/+0.04; +readout +0.25 [-0.29, +0.70]
(vs RCG -0.05 [-0.17, +0.03]); full +0.05 [-0.79, +0.96]. Unresolved: no gain over FoRIS.

### LVIS-92i

LVIS-92i, same frozen RCG2 and protocol (599 episodes = 4 per-fold packs of 150, 367 fold-classes, class ids
offset per pack; original resolution): FoRIS 44.27; RCG +0.31 [-0.65, +0.93], folds +0.39/+0.49/+0.05/+0.78
(unresolved); +readout +0.90 [-0.16, +1.60] (vs RCG +0.59 [+0.28, +0.88]); full 46.77 = +2.50 [+0.91, +3.07],
folds +1.35/+2.34/+2.55/+3.28 (vs RCG +2.18 [+1.30, +2.44]). LVIS uses COCO photographs, so this is a
class transfer, not an image-domain transfer. GT diagnostic, fine field at 1024: fixed 0.5 45.09, best single
level 45.82, best level per episode 54.98. Source: `research_20261005/rcg2_group_lvis{,_f0..3}/`.

## 论文主表与官方episode协议（2026-10-07核查）

以下抄自两篇论文Table 1的1-shot主表，DINOv3-L、1024输入、完整含CRF配置；论文未给这些主表点值的配对区间。
[FoRIS Table 1](https://arxiv.org/html/2609.03384v1) · [INSID3 Table 1](https://arxiv.org/html/2603.28480v1)。

| 数据集 | FoRIS论文 | INSID3论文 |
|---|---:|---:|
| PASCAL-Part | 55.8 | 50.5 |
| LVIS-92i | 42.8 | 41.8 |
| SUIM | 59.1 | 54.9 |
| PACO-Part | 42.3 | 38.7 |
| COCO-20i完整协议 | 60.9 | 57.6 |

COCO B+C+D是本仓库的1800例子集，论文没有这一行；上面的论文点值不能与自抽清单的成绩直接作差来宣布胜出。

**已查明的是官方采样协议，不是已发布的固定query/support配对清单。** 核对FoRIS `1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e` 与INSID3 `0c165a10cf52ab91f335883d06260de86854adbe`：
这五个dataset loader逐字相同；两个仓库树均未包含最终episode配对manifest。COCO的fold pkl是候选图像池，不是已经排好的episode列表。
[INSID3作者确认](https://github.com/visinf/INSID3/issues/18#issuecomment-4491696263)：PASCAL-Part主表平均4折mIoU；多折数据分别为COCO 4、LVIS 10、PACO 4、PASCAL 4。
[作者另确认seed固定为0](https://github.com/visinf/INSID3/issues/19#issuecomment-4526137985)；官方默认`num_workers=0`、`shuffle=False`，由loader在线抽query/support，参见[入口](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/inference_segmentation.py#L24-L65)和[参数](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/opts.py#L102-L115)。

| 数据集 | 官方元数据/类别划分 | 官方loader长度与配对方式 |
|---|---|---|
| COCO-20i | `COCO2014/splits/val/fold{0..3}.pkl`；80类按`fold+4*v`分4折 | 每折1000；随机类别、同类随机query/support且图像不同。[源码](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/datasets/coco.py#L14-L105) |
| LVIS-92i | `LVIS/lvis_val.pkl`；有效类按元数据次序分10折 | 每折2300，按idx轮换类别后随机配对。[源码](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/datasets/lvis.py#L14-L58) |
| PASCAL-Part | `all_obj_part_to_image.json`中的val池；animals/indoor/person/vehicles四组 | 每折`min(len(img_metadata),2500)`；轮换部件类别、随机图像/实例，默认object-box crop。[源码](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/datasets/pascal_part.py#L13-L161) |
| PACO-Part | `paco_part_train.pkl`划分448个候选类为4折，再过滤val中可配对类；图像/标注取`paco_part_val.pkl` | 每折2500；随机类别和图像/实例，默认object-box crop。[源码](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/datasets/paco_part.py#L15-L215) |
| SUIM | 按官方数据准备说明的SUIM合并包；7个前景类别FV/HD/PF/RI/RO/SR/WR | 无fold；长度为各类非空mask文件数之和，轮换类别后随机配对，不等于1635张图像数。[源码](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/datasets/suim.py#L14-L89) |

数据来源与元数据获取位置见[INSID3官方data说明](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/docs/data.md)；本次仅查阅源码，没有下载数据、模型或生成episode。
**公开材料的剩余缺口：** FoRIS论文声明沿用INSID3协议，但其[lvis.sh](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/scripts/lvis.sh)只列fold 0，未发布覆盖10折的主表运行日志或最终manifest；因此不能把该示例脚本视为论文全部评测记录。
PASCAL各折实际长度、SUIM非空mask总数和论文确切query/support身份依赖对应版本数据；本次没有运行loader，未虚报已拿到确切论文清单。

## Fast：方向有线索，完整优势尚未交付

固定600、原图分辨率的[既有读数](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md#2026-10-07-fixed-600-scoreboard-cpu-server-56464-read-only-tally-by-claude)：
FoRIS 61.5627，INSID3双线性无CRF 56.3812，原始最近邻43.0828，位置去偏后的简单D_I图方案48.0981。
这与公开4000/1024、fresh600/64×64不是同一比较，不能相减。

[D汇报](pro_cards_20261008/D_report.md)：fresh600位置级L4直接53.23，相对s2 +1.50；图后55.87，相对s2 +0.48、区间跨零。
L4是简化线索，未证明完整Fast已超过INSID3/FoRIS。卡1无收益；卡2只做零编码控制，完整注意力隔离未执行；B候选复读未运行。
在这批数据上，L4加入前景原型和软最大后的L6直接读出53.23 → 51.73、图后55.87 → 55.39，没有建立相对L4的净收益。
六级证据在统一81张可比较错图上，真目标平均分高于错圈区域的仅28–32张，图后为22–26张。
FoRIS作者已报告512px达到59.5/258ms（RTX3090），完整效率比较须面对这个合理配置；
[来源：FoRIS附录D](https://arxiv.org/html/2609.03384v1#A4)。这不是本仓库的同机测速。

## 错误分析与已关闭路线

来源：[82df916旧账本](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md)，保留原数字及各自比较对象。
- fresh600错图：FoRIS对比分真目标胜过错区32/84；反向匹配净值真目标更高39/81；错区背景命中更多41/81。这些量未可靠区分两者，不将其写成统计上等于抛硬币。
- 旧账本记64%的误并面积与真目标连在同一块；这是该批面积统计，不是所有错误都属纯身份错认。
- public4000的RCG场按真实目标面积占比<2%／2–10%／10–30%／>30%分档，最佳切点0.62／0.57／0.46／0.31。
- 逐图真值选切点+8.91；按真实大小分档+3.98，其中只作用于已找对的图为+1.69。均为标签诊断，非可得收益。
- 已关闭的已测构造：打分图阈值预测、九种大小估计、颜色/边缘估范围、按预测区域放大重跑（−0.05 [−1.40,+0.98]）；不扩张为全部查询特征不可用。

## 增删与组合：哪些认识仍有用

- 12源全局最优63.064818，相对11源只+0.002513，尽管标量残差单独相对MEAN有+0.331565；组件收益不可相加。
- 原始NN作为共同起点已有DEV241记录；相应全4000的原始输出尚缺，不拿FoRIS输出冒充原始起点。
- “17.4%没找对”的操作定义是最佳阈值IoU仍<0.5，可能含严重范围错误，不等于纯身份误识别率；真值oracle不是可得增益。
- 已测参考相似度/阈值估计、复杂图引导的负结果限定那些构造，不证明完整特征已经没有可用信息。
- 数百批量方法未产生主线完整胜出；[既有失败统计](cpu100_20261006/METHODS_AND_FAILURES.md)保留，旧方法卡与执行队列退役。

相对共同base统一报告补回TP、误加FP、误删TP、去掉FP四项最终净变化，逐类计算，再报告完整mIoU与成本。
这是一套分析坐标；只有推理时的可观测依据和完整优势能支持新方法贡献。当前工作仅见[PLAN](../../docs/research/PLAN.md)。

## 从Opus研究档案吸收的判断

来源：用户提供`CVPR2027_研究档案_Opus5.5_160MB_20261006.md`（72,640字节，SHA256 `d3a24569adb7dbe27403b5f4291b8c86fec500f24cb76d8bab0183827544d61f`）；取研究结论，不继承旧Agent指引、阶段或授权。
- 更有价值的判断链是从完整输出的具体错误，走到现有证据的区分能力，再用匹配对照判断最小改动。当前落点就是PLAN唯一的L4/s2完整比较。
- 编辑价值依赖起点：相对FoRIS成品“主要靠删”的观察不能搬到原始NN；仅补的盈亏纯度为J/(1+J)，仅删的背景纯度为1/(1+J)，J是该类当前IoU，不是通用固定常数。
- 大小估计即使与真大小相关，也可能沿着原分数场的同一种错误变化，因此相关性不等于能纠正切点；oracle说明缺口，合法输入能否取回另需结果。
- 档案中的“参考背景和投票是死重”不能覆盖当前L1–L6拆分；“最优切点只由占比决定”还依赖类条件分数分布与目标指标，均不吸收为无条件结论。

## 2026-10-08 为什么 RCG 不通用（Claude，服务器 48002，4080S 约 20 分钟）

代码、日志和三张拼图在 [transfer_probe_20261008](transfer_probe_20261008/)。像素增删用已存的逐例计数（1024），其余为 64×64 位置级。

- RCG 相对 FoRIS 的净改动（占真目标面积）：COCO 删错圈 2.2%、补回 0.0%；PASCAL-Part 删 1.3%、补 0.8%；LVIS 删 7.1%、补 0.1%；
  SUIM 多圈 0.7%、补 1.2%；PACO-Part 多圈 0.1%、补 −0.1%。增益集中在占图不到 10% 的物体上。RCG 只修剪小物体周围的松散错圈，不补漏。
- FoRIS 的错圈面积占真目标：PACO-Part 68%、LVIS 57%、SUIM 32%、COCO 22%、PASCAL-Part 19%。
- PACO-Part 看图（错圈面积最大的 12 张）：参考标的是部件，圈出来的是整个物体。预测面积中位数是真目标的 1.39 倍，34% 的图超过 2 倍，
  85% 的错圈与目标相连。SUIM 的错图是另一类：礁石、沙地等无定形大片区域认错了是哪一片。
- 探针（已关闭）：在 RCG 结果上只保留"最近的参考位置属于前景"的位置，不拟合、不调常数。相对 RCG 的 class mIoU：
  PACO-Part −1.99 [−3.46, −1.00]，PASCAL-Part −2.74 [−4.59, −1.42]，SUIM −4.50 [−6.38, −2.58]，COCO fresh600 −2.75 [−4.78, −1.74]。
  被删位置里错圈的占比：PACO-Part 68%（回本约需 70%），COCO 42%（约需 61%）。参考背景的最近邻有高于随机的信号，但不够回本。
  事先的预期是 PACO-Part 上明显变好，预期错了。校验：同一套特征上 L4 在 COCO fresh600 为 53.17（D 汇报 53.23），RCG 为 64.52。
