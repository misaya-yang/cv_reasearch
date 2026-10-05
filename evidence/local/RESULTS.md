# Experiment evidence ledger

Historical local results, retained through 2026-10-05. This is evidence, not a pending plan or paper claim.
Current interpretation: [CLAIM](../../docs/research/CLAIM.md); external/cloud evidence:
[dots record](../dots-2026-10-05/IMPORT.md). Numbers below retain their original cohorts,
resource settings, rendering stages and intervals. Missing provenance fields are not retroactively invented.
The 2026-10-05 cleanup corrected overbroad interpretations without changing result values.

The opening table is the **historical supervised-readout / early SAM3 configuration**. Later SAM3 results
appear under their dated entries; do not use the early exemplar as the current strongest baseline.
Some reports are server-only or point outside the repository; those runs were not reverified in the cleanup.

## 已确认的数

协议：官方 COCO-20i，种子 0，图片隔离的清单 TRAIN2400／DEV241／CONFIRM600；评哪一折就只用另外三折拟合；
在 DEV241 上选定后才打开 CONFIRM600 的标签。指标是原分辨率、完整流程（含 FoRIS 的 CRF）下的类 mIoU。
增益是同一批 episode 上相对 FoRIS 的配对差，方括号是 95% 区间（2000 次自助抽样）。

| 方法 | CONFIRM600 | 相对 FoRIS | DEV241 相对 FoRIS | 结果文件 |
|---|---:|---|---|---|
| FoRIS | 59.78 | | | `results/decision_v1/infer_1_confirm/report.json` |
| 决策读出 `convctx:layers` | 63.33 | +3.54 [+1.96, +4.96] | +3.44 [+1.56, +5.14] | 同上，`infer_1_dev` |
| 固定公式（17 个常数） | 61.45 | +1.66 [+0.85, +2.51] | +2.09 [+0.80, +3.31] | `results/f4_v1/formula_B_confirm_report.json` |
| SAM3，视觉例子 | 61.09 | +1.31 [−2.45, +3.57] | +3.43 [−2.74, +7.05] | `results/f4_v1/sam3_A_confirm_report.json`、`sam3_A_dev_report.json` |
| SAM3 与 FoRIS 掩码取并 | 62.83 | +3.05 [+1.82, +4.39] | | `results/fixed_host_union_v1/` |
| SAM3，给真实类别名（特权） | 73.38 | | | 同 SAM3 A 报告 |

补充：

- 决策读出四折为 +3.66／+3.94／+2.50／+4.06；249 个 episode 上升、174 个下降，42 个下降超过 10 点。
- 同结构、只看分数的对照是 +2.27 [+1.59, +3.20]（patch 级）；只删不加是 +2.01 [+0.84, +3.14]。
- 学习曲线从每个模型 600 个 episode 起持平。线性档（19 个权重）+2.19，四折系数相近（余弦 0.988），未证明普遍类无关规律。
- 读出按基类掩码拟合，所以不在零训练一栏；63.33 也低于 UINO-FSS 64.5（训练）和 FSS-SAM3 论文报告的 66.1。
- FSS-SAM3 按作者协议在 200 个官方 episode 上是 59.93，论文的 66.1 没有复现（`results/sam3_author200/`）。

误差账（DEV241，patch 级，`results/decision_v1/fit/report.json`）：FoRIS 57.00；同一分数场按真实占比切 66.09
（+9.09 [+6.15, +10.45]）；删掉全部误并的 patch 76.97（+19.97）。这两个数用了标签，只说明分数在哪里，
不说明能拿到。

## 失败账本

按当前问题查相关条目。每行保留构造、对照、样本和教训。旧代码恢复位置见仓库清理记录；不是待执行清单。

共同教训：

1. 这里真实效果是 2 到 4 点。40 个 episode 的配对区间半宽是 3 到 9 点，所以 40 例上的方法探针读不出结论。
2. 已测手写规则在各自协议下约 +1.5，监督读出约 +3.5；这不是手写规则或表示的普遍上限，跨协议差值不能作归因。
3. 在弱后端（42 到 51 分）上测组件，对 FoRIS（65 分）没有说明力。
4. 用标签量出的差距（10 到 20 点）不等于可用信号：先量信号，再造机制。
5. 没有先跑一个真实样例的准备会白费：QK 第一次运行完成 0 例（缺 CRF 路径）；构造样本训练的全部样本类别号是 0。

### 图池路线（2026-10-02，用户于 10-03 关闭：加条件不是贡献）

| 试了什么 | 结果 | 样本 | 教训 |
|---|---|---|---|
| 把无标注同类图当伪参考（INSID3） | 55.0 → 59.1（+4.1）；朴素自训练 57.0；图池给真掩码 67.7 | 第 0 折 400 | 增益来自多用参考图；筛选规则只值约 1 点 |
| 同上，四折 | +6.1 [+4.2, +8.2] | 4×400 | 基线与官方实现只有 43% 逐例一致，绝对值不可用 |
| 无标签的伪掩码可靠度打分 | AUC 0.81 到 0.84；所有变体停在 57 到 59 | 第 0 折 400 | 区分不了对错的伪掩码 |
| 图集上先聚类后贴标签（M1） | 上限 54.2，低于基线 | 第 0 折 | 否定 |

### 七个候选机制的小试验（2026-10-02，第 0 折前 10 例；INSID3 52.48，FoRIS 65.00）

| 试了什么 | 结果 | 教训 |
|---|---|---|
| 局部 patch 到全局注意力读出 | 42.78 对 42.68，+0.10 [−0.37, +0.58] | 后端远弱于 FoRIS |
| 完整全局内容交互改写 | 29.49，−13.19 [−22.94, −0.68] | 明确为负 |
| 跨图方向字典加低秩度量（训练） | 77.85 对 77.85，改变 0 个像素；FoRIS 79.19 | 修正量最多 0.023，只有 0.017% 的像素够得着 |
| 参考背景反事实视图；patch 格点相位视图 | 输给同视图均匀对照 1.62 和 0.46 | 参考加权没有独立价值 |
| 只改 CLS／register 提示；不平衡最优传输 | 无增量 | |
| 亮度 0.75 压力（FoRIS） | −0.31 [−0.85, +0.27] | 不是问题 |

### 40 例上的固定构造（旧 DEV40；FoRIS 65.16）

| 试了什么 | 结果 | 教训 |
|---|---|---|
| 原生角度协方差 | 64.28 → 63.94，−0.35 [−1.77, +1.25] | 参考上拟合变好不等于查询上变好 |
| 源码判别轴修正 | +0.16 [−0.41, +0.71] | 问题存在但不重要 |
| 硬余弦背景否决 | 65.38 → 60.31 | 明确为负 |
| DINO 均值亲和度定界；RGB 边界 | 58.93（−6.23 [−11.52, −0.21]）；36.57 | 明确为负 |
| 连通块选择 | 用标签选最多 +3.77；取最大块 −6.72 | 64% 的误并与真目标连在同一块 |
| RICE 子空间核心 | 负（数字未抄入本表，见 git 历史里的 `results/rice_core_v1`） | |
| QK 描述子，九个臂 | 主臂 58.70，−6.46 [−9.05, −3.92] | 明确为负 |
| 用 FoRIS 的框再提示 SAM3（C2） | 62.49 → 63.56，+1.07 [−7.54, +9.69]；简单取并 67.42 | 40 例读不出；且输给朴素对照 |
| 联合对应零训练构造（10 例） | 62.98 → 62.16，−0.82 [−1.58, −0.11] | 检索一致不等于身份判对 |

### 241 例上的批次（DEV241；FoRIS 59.12）

| 试了什么 | 结果 | 教训 |
|---|---|---|
| 切分水平族：对比度、边界、签名、往返、双图缩放 | +1.53 [−1.29, +2.99]、−0.52、−2.72、−12.66、+0.41；用标签切 +10.52 | 已测切法未建立目标收益；排序和切点都可能有损失，不能据此排除切点问题 |
| 无标签的排序证据（证据审计） | 争议区 AUC：FoRIS 0.664；最好的 0.724，+1.07 [−0.16, +2.39] | 没有一种过 0.75；有标签的探针 0.77 到 0.81 |
| 无标签估计目标占比（四种） | 对数面积误差 0.62 到 0.66；给真实面积 +8.67 | 这四种估计未可靠恢复占比；不构成所有无标签估计不可识别的定理 |
| 匹配尺度后再看一次 | 给真实比例也只有 +0.06 [−1.16, +1.12] | 否定；尺度错配每翻倍掉 9.5 点只是诊断 |
| 零训练的参考读出删除 | 第 12 层 −15.22；末层 −1.39；查询核心删除 +0.46 [+0.28, +0.63] | 只有很小的正信号 |
| 查询自支持修剪 | +1.68 [+0.72, +2.27]（patch 级） | 小的正结果，已被读出覆盖 |
| 读出加 16 个特征主成分 | +1.62，对 +4.12 | 在新类别上有害 |
| 180 个 episode 拟合读出，不从宿主起步 | −3.4 | 过拟合；需要从宿主起步和留出类别早停 |
| 文档 F4 七个臂 | 57.75（−1.37 [−2.70, −0.30]）、58.14、58.19；查询核心 59.58 | 没有臂过线 |
| SAM3 负例挖掘（200 个官方 episode） | 56.22 对 59.93，−3.71 [−5.45, −0.43] | 明确为负 |

### 2026-10-04 开机：读出能否无标注拟合、能否搬到别的数据集

| 试了什么 | 结果 | 教训 |
|---|---|---|
| 读出只用 826 个自造样本对拟合（D1，DEV241，patch 级；标注拟合是 +4.12） | 卷积读出 −7.40、−27.59、−21.18；线性读出 +0.30、+1.00、+1.01，区间都跨零 | 自造样本是同一个实例、目标占比分布也不同，学到的规则搬不到真实 episode |
| COCO 上拟合的读出不重训，搬到 PASCAL-Part（299 例） | 54.66 → 55.63，+0.97 [−1.23, +2.42] | 当前 COCO 拟合读出的跨域收益未建立；校准是一个解释（推断） |
| 同上，PACO-Part（299 例） | 43.41 → 43.50，+0.08 [−1.08, +1.45] | 同上 |

结果文件在服务器 `/root/autodl-tmp/demo9_extent/results/decision_pivot_v1`。LVIS、SUIM、肺部三个数据包没有打完，
没有读数。读出这条线到此为止：它是"FoRIS 加一个用基类标注拟合的校准"，不是独立的方法。

### 2026-10-04 夜至 10-05 凌晨：SAM3 路线与 FoRIS 的逐步定位

协议同上（DEV241 选、CONFIRM600 读一次、原分辨率、配对区间）。结果文件在服务器
`results/sam3_relative_v1/`、`results/sam3_keep_v1/`、`results/zoom_v1/`、`results/foris_confirm_v1/`。

已确认的数（CONFIRM600）：SAM3 视觉示例、按"分数 ≥ 0.7 × 本例最高分"保留 67.12；SAM3 自己从 LVIS 词表给参考物体
读名字、在查询上读语义图 69.55；两者按置信度路由 71.49，相对 FoRIS +11.71 [+9.01, +14.57]；给真实类别名（特权）78.34；
真实类别名加同一个路由 75.43。读名字错掉 8.8 点：600 例里 65 例读错（读出的名字 14.8，真实名字 76.8）。
这一栏是"用基础分割模型"，已发表的 CG-ICS 是 72.3（多模态大模型出名字、SAM3 在参考上验证、再加视觉路线），想法相同。

FoRIS 的错误定位（DEV241，模型分辨率 59.05）：最佳切法 +10.3；把查询图自己当参考 80.17；换同类别的另外三张参考
57 到 61，原本失败的 episode 里换参考只有 17% 到 29% 成功，难的是查询图而不是参考图。失败 83 例：位置对但 IoU 低 49 例
（最佳切法 60 对 36），位置错 34 例（SAM3 视觉示例 41 对 14）。

| 试了什么 | 结果（DEV241） | 教训 |
|---|---|---|
| SAM3 候选的 DINO 相似度过滤、聚类 | −1.73 到 +0.22；最好 +0.88 [−0.10, +1.97] | 全图特征池化认不出对错候选 |
| 按原始分数做 IoU 最优决策；参考负例阈值；拟合校准器 | −11 到 −12.6；−16.9；CONFIRM −0.57 [−2.72, +1.21] | 这些概率解释及校准构造未建立有效性 |
| 第二遍（以锚点为提示）；遮背景、全框、负框、五视图投票 | +0.45 [−0.53, +1.64]；−9.30、+0.02、−2.93、+0.76 | 这些提示变化未建立足够的完整输出增益 |
| 反向检查（查询候选当示例回到参考） | AUC 0.709，低于正向分数 0.776 | 反向路线有同样的外观弱点 |
| 提示反演（连续优化提示复现参考掩码） | −9.3 [−13.9, −2.9]（约 80 例） | 无先验的提示会过拟合参考 |
| 示例路线下读语义图 | −15.51 [−20.64, −12.70] | 语义图只在文本提示下可用 |
| 用 FoRIS 的区域裁决 SAM3 候选（四个臂） | 最好 −0.30 [−0.61, +0.52]（对路由） | 决定结果的错候选也落在 FoRIS 区域里，两个模型错在同一批干扰物 |
| FoRIS 候选区域的区域级证据（池化、双向覆盖、投票） | AUC 0.72 到 0.79，FoRIS 自己的分数 0.83 | 这些区域级统计未超过宿主分数，不否定其他读法 |
| 对候选区域裁剪后重新编码（类别 token、区域 token、灰底） | AUC 0.68、0.77、0.72，低于 0.83 | 物体级 DINO 相似度也认不出目标 |
| FoRIS 每个连通块按自身尺度重跑（至多三轮） | −0.05 [−1.40, +0.98]；用标注挑放大后的块只比挑原块多 0.35 | 裁剪里 FoRIS 得到同一团；过大不是分辨率或占比造成的。相近做法已有 Foveate（ECCV 2026 workshop） |
| SAM3 两条路线在候选级互相印证 | −4.72 [−7.58, −1.76]（对路由） | 留下的候选里印证信号 AUC 0.744，不比分数 0.745 强 |
| 给参考读出的前 8 到 12 个名字都在查询上试，再选名字（查询置信度、与示例路线重合、加权投票、中心名） | DEV241 最好的路由臂 −0.37 [−1.88, +0.35]；915 个新 episode 上最好的 −0.61 [−2.56, +1.12]；用标注选名字 79.1 到 79.9 | 对的名字在候选名单里，但 SAM3 自己的标量信号选不出来。CONFIRM600 上"共识中心名"+3.23 [+0.60, +4.29] 未在 915 例复验中建立收益；不能仅由这两次结果证明其真效应为零 |
| 上述全部信号做跨折线性拟合来选名字（DEV241，留一折） | 名字路线 73.2，低于路由 75.2 | 该固定线性拟合没有获得更好的选择，不能推出这些信号绝无可用信息 |
| 名字在参考上的打分改用语义图或高分实例并集的 IoU | DEV241 +0.35 [−0.31, +1.30]；915 例 +0.04 [−0.53, +0.56]；CONFIRM600 72.62，+1.13 [−0.01, +2.03]（含词表清洗） | CONFIRM600 的增益主要来自词表清洗（mouse、orange 等被展平的 LVIS 限定词），打分方式本身没有作用 |
| 把参考物体单独裁出来（原样、灰底）让 SAM3 给候选名字打分 | 路由臂 −0.46 到 −15.7 | 物体居中的裁剪不改善读名字 |

### 2026-10-05: reference-only discriminant, CPU replay

COCO-20i, exposed DEV241, seed 0, 1024 model resolution without CRF; not original-resolution confirmation.
Reference labels alone fit an episode-local closed-form, role-balanced ridge classifier; no encoder, image pool,
class names, or base-class labels. Fixed penalty is the reference Gram matrix's mean eigenvalue.

| Construction | Paired result | Failed link |
|---|---|---|
| Full-vector reference ridge | 49.72 versus FoRIS pre-CRF 58.56; -8.84 [-12.38, -6.44]; folds -11.99 / -10.95 / -4.01 / -8.38; 80 up, 159 down | A reference-fitted discriminant does not transfer safely: 5,835,294 model pixels corrected, 10,708,643 damaged. This rejects this fixed readout/sign cut, not the full DINO representation. |

The same-information mean-contrast control is 42.86; cached native FoRIS with CRF is 59.07 at model resolution.
Intervals use 2,000 paired connected-photo-group bootstrap draws (239 groups). CPU run took 440.6 seconds.
Result: `results/reference_discriminant_v1/report.json`. The fixed readout is retired; no
regularizer/threshold sweep or new confirmation was run.

### 2026-10-05: joint-reference harmonic inference, CPU replay

COCO-20i, exposed DEV241, seed 0, 1024 model resolution without CRF. All reference labels are fixed graph
constraints; query labels are jointly inferred from the same cached DINOv3 tokens. FG/BG reference edge masses
are equal; the query graph uses symmetrized 32-NN exponential cosine weights, bandwidth .07. No added inputs.

| Construction | Paired result | Failed link |
|---|---|---|
| Joint reference graph | 23.17 versus FoRIS pre-CRF 58.56; -35.39 [-37.70, -30.20]; folds -41.58 / -32.13 / -34.60 / -33.15; 27 up, 212 down | Appearance-based reference transfer remains ambiguous, while reference constraint mass is overwhelmed by query smoothing. 77,297,228 new false-positive model pixels. No evidence justified class-specific correction before this construction was built. |

Same-information controls: reference density without query graph 35.17; FoRIS on the identical graph/operator
31.08. Joint-minus-density is -12.00 [-14.32,-7.15]; joint-minus-FoRIS-graph is -7.91 [-9.88,-3.63].
Intervals use 2,000 paired connected-photo-group bootstrap draws (239 groups). CPU runtime was 1,040.6 seconds.
The median reference-to-query edge-mass ratio is 0.0001612. This is a design/selection failure, not a reason
to declare all joint inference or frozen DINO features exhausted. No kernel-strength, cut, or regularizer sweep follows.
Result: `results/joint_reference_graph_v1/report.json`; the construction and its code are retired.

### 2026-10-05：在缓存特征上逐段重放 FoRIS（本机 CPU，DEV241，CRF 之前，模型分辨率）

重放与原始分数一致（完整 FoRIS 58.52，对原包 58.56）；全部 241 例一遍约 2 分钟，不用 GPU。脚本在本机
`~/cvpr2027_local/{foris_terms,ablate,swap,qbg}.py`，特征缓存 `~/cvpr2027_local/feat`（4 GB）。

| FoRIS 的组成 | FoRIS 自己的切法 | 用标注选最佳切法 | 争议区 AUC |
|---|---:|---:|---:|
| 单个原型（参考物体均值） | 45.59 | 63.88 | 0.661 |
| 多个原型（聚类加 log-sum-exp） | 42.05 | 58.54 | 0.631 |
| 再减 0.55 × 参考图的难背景原型 | 51.94 | 65.73 | 0.666 |
| 再加最近参考投票 | 52.20 | 65.39 | 0.648 |
| 再加查询图聚类的种子簇先验 | 56.31 | 67.70 | 0.651 |
| 再减分歧惩罚 | 57.31 | 67.97 | 0.648 |
| 再加簇重加权（完整 FoRIS） | 58.52 | 68.62 | 0.656 |

从完整 FoRIS 里去掉一项：去掉背景项 +0.13 [−0.47, +0.86]；去掉投票 +0.78 [−0.45, +1.63]；去掉种子簇先验
−2.05 [−3.36, −0.89]；去掉惩罚和重加权 −2.22 [−3.54, −0.87]。FoRIS 各部分把固定切法下的分数从 45.6 提到 58.5，
最佳切法下只从 63.9 到 68.6，争议区排序没有一项改善。

把某一项换成"用查询图自己的真掩码算出来的"（诊断，用了标注；基线是到种子簇先验为止的 56.31）：前景原型
+3.86 [+1.32, +5.49]；背景原型 +9.66 [+7.34, +10.92]；两者 +14.44。投票和先验换成自己的等于直接给标注，不作数。

| 试了什么 | 结果（对完整 FoRIS） | 教训 |
|---|---|---|
| 背景原型改从查询图取：第一遍掩码之外、最像前景的 20%／50%／全部 | +0.31 [−0.58, +0.94]／+0.81 [+0.19, +1.38]／+0.70 [+0.09, +1.39] | 值 9.7 分的难背景在 FoRIS 掩码之内，掩码之外取不到 |
| FoRIS 掩码的连通块按相对分数丢弃（均值／90 分位／最大值 ≥ 0.5 到 0.9 × 最高块） | 最好 +0.43 [−0.42, +1.66]；用标注丢掉"目标占比不到一半"的块 +3.82 [+0.05, +5.54] | 错块与对块的相对分数 AUC 只有 0.675；整块错只值 4 分，大头是与目标连着的多并 |
| 查询图自己的前景原型：取 FoRIS 分数最高的 1%／3% token、掩码内前 25%、整个掩码 | 最好 +0.33 [−0.54, +2.18]（掩码内前 25% 替换前景项）；最佳切法 69.5 对 68.6 | 种子精度中位 0.98，但 36 例低于 0.5；种子是最像参考的那部分，原型与参考原型给出同一排序（真掩码原型的最佳切法是 78.7） |
| 每张图用自己的统计量标准化 token（中心化、去前 1／3 个主方向、白化、联合白化、减参考背景均值） | 单原型 −2.0 到 −35；只有最佳切法在中心化后 63.8 → 66.1／67.4 | 明确为负；图内主方向带着类别信息 |
| FoRIS 的固定切分水平 0.35 到 0.75 | 0.55：+0.51 [−0.22, +1.19]；其余更低 | 0.5 已在最优附近；逐例最佳水平与真实占比的秩相关 −0.47，与参考占比 −0.12 |
| 由查询图自己的特征空间判每个 token：FoRIS 分数最高 1% 当目标种子、分数低于 0.25 当背景种子，其余按最近种子归类（`trimap.py`、`trimap2.py`） | 56.47，−2.06 [−3.89, +0.18]；同样位置的种子换成真标签 75.49，+16.97 [+13.37, +19.82]；只去掉错的目标种子和被背景种子吞掉的目标 62.59，+4.07 [+1.06, +6.27] | 查询图自己的空间在种子对的时候能判对；缺的是种子：目标里被 FoRIS 打到 0.25 以下的那部分（平均占目标 7%，定位错的 34 例里 21%）没有进目标种子。真标签那一行给了 78% 的 token 标签，是诊断不是余量 |
| 查询图按自身特征分段（k-means，K=8 到 64）再判段（`clus.py`、`prefix.py`） | 用标注判段：K=32 是 77.36，K=64 是 80.60；段均分 > 0.5：+0.36 [−0.41, +1.83]；按段均分取前 n 段、用标注选 n：67.84；最大落差定 n：−2.47 | 分段本身够用；按 FoRIS 分数排段的上限与逐例最佳切法相同（68），排序错 9 分、取几段错 10 分 |
| 排错的段与目标段在查询图自己空间里的关系（`segerr.py`，K=32） | 排在最低目标段之上的错段：到最佳目标段的质心余弦 0.46（其余目标段 0.70），AUC 0.85 到 0.89；相邻 54% 对 99%；FoRIS 分数比的 AUC 0.53。最高分段是目标段的占 88% | 条件是"用标注划出的候选"；见下一行，换成无标注的候选后不成立 |
| 两把钥匙：参考只提名（段均分过宽松水平）并指定锚（最高分段），查询图自己的空间裁决（到锚的质心余弦过阈值，可加相邻生长）（`anchor.py`、`anchor2.py`） | 段级 16 格最好 +1.12 [−0.16, +3.17]；只在 FoRIS 掩码上否决远离锚的段 +0.97 [+0.00, +1.86]；锚换成标注给的最佳目标段 +3.56 [+1.12, +4.57]；向低分处扩张全为负 | 锚错（12%）吃掉大半；锚全对也只有 +3.6 |
| 把整张查询图当负样本、参考物体当正样本拟合线性判别（正-未标注，`pu.py`） | −10 到 −46；正则越强越接近单原型 | 判别器学到的是"哪张图"，不是"哪个物体" |
| 种子（最高 1%）的均值在查询图自己空间里当原型，按种子纯度分组（`pure.py`、`meansim.py`） | 纯净的目标 token 集的均值排序很好：整个目标 80.7、随机 5% 79.2、FoRIS 掩码里对的那部分 78.0；掺入等量错 token 73.1；FoRIS 整个掩码 62.6。但种子纯的 149 例里，种子原型的最佳切法 0.816 与 FoRIS 的 0.815 相同 | 排序的余量不在种子纯的例子里；那里缺的是切在哪（0.732 对 0.815） |
| 切在最稳定的水平（面积随水平变化最小），在 FoRIS 分数、种子原型、两者之和上（`mser.py`） | +0.52 [−0.03, +1.73]、−1.83、+1.28 [−0.59, +3.33] | 未分辨，与此前切分族的 +1.5 同量级 |
| 自洽集合：从种子均值出发，反复取"到集合均值余弦 ≥ 绝对水平"的 token（`shift.py`、`spread.py`） | −10 到 −28；真均值加绝对水平 0.70 是 69.95（诊断） | 迭代会漂到整张图；目标内聚度（到自身均值余弦 0.85 ± 0.06）随目标大小变（秩相关 −0.66），参考的内聚度只能预测到 0.44 |
| FoRIS 去掉惰性部分（按保存的各项做算术，`simpl.py`） | 单原型、无投票、无参考背景：59.01，+0.48 [−0.80, +2.26]，最佳切法 69.30 | 该缓存消融中组合简化未显示损失；不是所有输入上的代数等价，不能说各项普遍无作用 |
| 边界：误差在哪（`halo.py` 到 `halo5.py`，用标注） | 把跨真边界的 token（目标像素的 29% 在其中）里的像素修对：58.56 → 66.72（CRF 之后 59.07 → 67.11）；只按 token 多数标签修：60.55；其余全修对、只留这些 token：86.31 | 跨边界 token 值 8.2 分，其中 6.2 分来自所测整块二值读出的限制，2.0 分来自 token 判错；不是软场或 token 表示的信息上限；FoRIS 的 CRF 只拿回 0.5。FoRIS 在这些 token 上的分数随覆盖率单调（0.30／0.44／0.58／0.69），没有系统性偏宽或偏窄 |
| 理想覆盖率场的采样密度（`shiftsim.py`，用标注） | 步长 16：95.53；步长 8：97.76；步长 4：97.95 | 64 × 64 的软场只要数值是覆盖率就够定亚 token 边界；加密网格只多 2 分 |
| 已有的放大重跑结果当边界修正器重读（`zoomband.py`、`zoomread.py`，服务器 `results/zoom_v1/dev`） | 每个连通块保留核心、边界至多按放大结果移动 4／8／16 像素（512 图）：+0.15 [−0.23, +0.62]／+0.17／+0.16；跨边界 token 内的误差只从 0.198 降到 0.158（× 目标面积） | 放大后重跑 FoRIS 并不把边界定得更准 |
| 特征空间里的抠图：边界 token 的混合比 = 在最近的"确定目标"与"确定背景"token 均值连线上的投影（`matte.py`） | 三分图取自真值：混合比 90.14，FoRIS 分数 88.49，真覆盖率 95.53（混合比与覆盖率秩相关 0.67，FoRIS 分数 0.47）；三分图取自 FoRIS 掩码（2 个 token 宽，混合比与分数取均值）：+0.88 [+0.50, +1.71]，四折 +1.2／+1.2／+0.3／+0.7 | 信号存在但小；边界修正的前提是粗判对 |
| 种子原型否决，按纯度分组；无标注判断种子是否纯（`veto.py`） | 种子纯的 149 例里否决最多 0.732 → 0.739；判"定位错"的 AUC：原始分数峰值 0.80、种子到参考均值余弦 0.79、种子互余弦 0.68 | 种子纯的例子里远处多并不能靠查询图自己的原型分开 |
| 用查询图的颜色修边界：引导滤波作用在软分数上（`gf.py`）；带内随机游走（`rw.py`，只跑了 16 例） | 引导滤波最好 59.08，与 FoRIS 的 CRF（59.07）持平；随机游走不如 CRF | 颜色定不准这些边界 |
| 逐图中心化后的特征上跑完整 FoRIS（减参考背景均值／各减自身均值／各减一半） | −2.17 [−3.77, −0.74]／−5.51／−3.27 | 明确为负 |
| 匹配滤波：用查询图自身的均值和协方差当背景模型（`mf.py`） | 不白化 44.75，最佳切法 66.62；白化（收缩 0.5 到 0.01）最佳切法 30 到 16，全 token AUC 从 0.956 掉到 0.72 以下 | 查询图方差最大的方向正是语义方向，白化把语义压掉 |
| 水平集的均值 token 最接近参考物体均值的那个水平（`setmean.py`） | −13.76 [−17.43, −11.20] | 明确为负；与此前"签名"一致 |
| 参考物体均值与查询目标均值之差（`delta.py`，用标注） | 两者余弦 0.665；加上两图均值之差后余弦升到 0.72，排序反而从 63.9 掉到 49 | 更接近目标均值不等于排序更好：排序要的是相对查询背景的方向 |
| 只信 FoRIS 粗判的边界修正器的上限（`bandceil.py`，用标注） | FoRIS 自己边界 ±8 像素内给真值：68.39，+9.83；±16 像素：74.35，+15.78；±1／±2 个 token 的带：76.80／84.66 | 该 GT 修复带也可能删除整块细小错对象，不能归为纯粹无语义边界预算；已试的取法（CRF +0.5、混合比 +0.9、颜色 0）都只拿到很少 |
| 从种子到整个物体：查询图 token 两两亲和（余弦 > τ 记 1，否则 1e-5）做递归归一化割，每层跟着种子（FoRIS 分数最高 1%）多的一侧（`ncut.py`） | 路径上用标注选最好的节点：τ = 0.4／0.5／0.6 为 54.38／60.87／62.90，低于 FoRIS 逐例最佳切法 68.60（种子纯的 149 例平均 IoU 0.680／0.750／0.777 对 0.815）；无标注停法（割代价过阈值、割会分开种子；停在整图时退回 FoRIS）−4.9 到 −26，按折嵌套 −4.87 [−5.44, −1.35] | 失败在分组本身：两两亲和的二分层级里没有“目标”这个节点，停法再好也到不了 FoRIS 的水平集；到此，范围这一环从分数场、种子相似度、k-means 段、归一化割四面都量过，没有一面能定 |

按 FoRIS 最高 1% token 的纯度分组（DEV241，平均 IoU：FoRIS／FoRIS 逐例最佳切法／种子位置给真标签）：纯度 ≥ 0.9 的 149 例
0.732／0.815／0.816；0.5 到 0.9 的 58 例 0.493／0.625／0.681；低于 0.5 的 34 例 0.178／0.250／0.539。

看图（`~/cvpr2027_local/look/`，按 IoU 排序的对照图）：IoU 低于 0.5 的 84 例，目标占图中位 2.5%，多并面积是目标的 1.3 到 3.3 倍，
漏掉的很少。多并的是相近类别的更大物体（手提包对行李箱、烤箱对橱柜、滑雪板对人腿、卡车对飞机），另有标注错误和
参考图本身不可用（救生圈特写标成 boat、绿桌布标成 dining table）的例子。到此为止，末层、两图分开编码的 token 上的读法
今天又试了 8 种，没有一种过 +1；此处是当时的待测记录。下方 finer-token/language 队列已有 joint/layer 结果；不再作为未运行计划。

### 2026-10-05: seed contamination and source-pattern signal (CPU observation)

COCO-20i, exposed DEV241, seed 0; 64x64 patch majority labels. These are diagnostic counts and
episode-level AUCs, not a new mask or original-resolution method score. Among the 34 episodes where
fewer than half of the top 41 FoRIS tokens are target, the highest-scored token is background in 29
(fold counts 8/5/7/9). None has fewer than 21 target tokens: insufficient target capacity cannot explain
these 34 contaminated seeds. Reducing the seed set is not a remedy for the principal wrong-identity cases.

For distinguishing pure seeds (purity >=.9, 149 episodes) from contaminated seeds (<.5, 34), native raw
score peak AUC is .8289 [.7436,.9057]. Ranking reference patches by cosine to the query seed mean and
checking the best union IoU against the allowed reference annotation gives AUC .6390 [.5144,.7537],
paired difference -.1899 [-.3046,-.0864] against raw peak. Source FG/BG ranking AUC as a purity signal is
.7300 [.6304,.8205]; after conditioning on raw-peak deciles its AUC is .5549 [.4026,.7015], unresolved.
Intervals use 2,000 connected-photograph-group bootstrap draws, 239 groups, seed 0.
No source-pattern selector or mask editor follows; this construction has not established incremental
identity evidence. Report: `results/seed_identity_signal_v1/report.json`. No encoder or GPU was run.

### 2026-10-05: query-native candidate validation and failed scope transfer

COCO-20i, exposed DEV241, seed 0. Query-native K=32 candidates are chosen before query labels open.
In the 43 episodes whose highest-scored token is background, source FG/BG rank validation chooses a
target-majority candidate in 20 cases versus 10 for FoRIS segment-mean choice, +23.26 percentage points
[+10.00,+37.14]. Across all 241 it chooses 184 versus 189, -2.07 points [-6.61,+2.08]: conditional
correction is not overall method gain. Report: `results/query_candidate_identity_signal_v1/report.json`.

| Construction | Measured against the host and controls | Failed link |
|---|---|---|
| Source-verified query centroid, source-IoU-selected cosine cut, low-native-peak scalar gate | 1024 model image before CRF: ungated 39.05 versus FoRIS 58.56, -19.51 [-21.54,-14.64], 47 up/193 down. All photograph-disjoint leave-one-fold-out gates select never-switch: 58.56, zero change in every fold and episode. Same source-cut/gate controls using FG-BG cosine and native segment mean also never switch; ungated scores40.68/39.88. | Candidate purity does not deliver query extent. Source-calibrated masks have median2.14 times the true area;96/184 selected-target cases overmerge above1.5 times. Source fit is not query scope calibration. No cut or gate sweep follows. |

Intervals use 2,000 connected-photo-group bootstrap draws, 239 groups, seed 0. Original-resolution CRF
confirmation was not run. Report: `results/query_hypothesis_mask_v1/report.json`; this mask construction is retired.

### 2026-10-05: frozen feature-mixture refinement versus cached FoRIS + CRF

COCO-20i, exposed DEV241, seed 0, 1024 model resolution. Freeze the already-positive two-token unknown
band, six spatial neighbours per endpoint, and mean of alpha with the original score; no parameter is
reselected. Its final mask is59.4439 versus cached FoRIS including CRF59.0748: +.3691 [-.1111,+1.0722],
folds +.5198/+.6256/+.0134/+.3150, 121 up/117 down. The positive increment remains unresolved.
Against FoRIS before CRF58.5620, the same mask gives+.8819 [.4907,1.7283], reproducing the earlier signal.
Intervals use 2,000 connected-photo-group draws (239 groups), seed0. This comparison replaces CRF;
alpha followed by the identical CRF, original-resolution evaluation, and CONFIRM600 are not run.
Report: `results/frozen_matte_vs_native_v1/report.json`. CPU used two threads and15.5 seconds, no encoder/GPU.

The majority-label/nearest-block diagnostic does not establish a16px soft-field representation ceiling;
the ideal stride16 coverage field itself reaches95.53. A prediction-boundary band filled with query
truth can also remove a whole thin wrong object. GT trimap90.14 versus same-trimap score88.49 fixes
foreground/background identities outside the unknown zone, so its gap is not solely a trimap-width problem.

### 2026-10-05: fixed-unknown-set attribution for the matte proposal (GT diagnostics)

COCO-20i, exposed DEV241, seed0, 1024 model resolution, no CRF on the matte output. Keep the two-token
unknown set, original score outside it, nearest-six rule and alpha-score mean fixed. Filtering only
contaminated existing endpoint anchors with query truth gives61.9057 versus frozen matte59.4439,
+2.4617 [1.6520,3.0639], folds +2.8001/+2.4421/+2.3395/+2.2550, 137 up/29 down. Replacing only alpha
inside that same unknown set with true coverage gives77.9218, +18.4779 [16.3707,20.1424] over frozen
matte. Both use query labels and are not deployable methods or limits on other constructions.

Filtering leaves no pure FG anchor in24 cases; the frozen arm is retained in those cases. All24 have pure
target tokens elsewhere on the existing64x64 grid, and22 have them inside the predicted token mask.
Thus these cases do not support a claim that the grid contains no pure foreground token. Endpoint purity
is one contributing error, but the GT-trimap90.14 cannot establish that choosing the trimap is the only gap.
Intervals use 2,000 connected-photo-group draws (239 groups), seed0.
Report: `results/matte_trimap_attribution_v1/report.json`; CPU used two threads and25.4 seconds.

The original CRF CPU replay on real episode0_0_72 aborted in the native extension (`_int_malloc`
assertion). It did not load an encoder or use CUDA, and no library/environment modification or workaround
followed. Alpha plus identical CRF and a fresh-forward confirmation remain unverified. No GPU was launched.

### 2026-10-05: original-resolution frozen matte read

COCO-20i exposed DEV241, seed0, original annotation resolution, cached complete public FoRIS forward and
CRF output. A fixed feature-mixture matte **replaces** the CRF output; it is not followed by the same CRF.
Class mIoU: native59.1218, matte59.5130, paired +0.3912 [-0.0931,+1.0966], folds
+0.5905/+0.5727/+0.0375/+0.3625, 121 episodes up/117 down. The same-information delete-only
control (`matte & native`) is59.5540, +0.4321 [+0.1200,+0.9170] over native, folds
+0.2492/+0.4604/+0.2167/+0.8217, 110 up/128 down. Matte versus delete-only is
-0.0410 [-0.4897,+0.4512], folds +0.3412/+0.1123/-0.1792/-0.4592, 115 up/123 down.
Intervals: 2,000 connected-photo-group bootstrap draws, 239 groups. All241 cached 1024px truth masks
matched the original query annotations exactly before scoring. This fixed construction has no demonstrated
increment over the complete host or the stronger same-information control. The failure is the conversion
from a token-coverage signal into a better final mask; it does not bound other representations or methods.
Fresh-forward equality, matte followed by identical CRF, and CONFIRM600 were not run. CPU four-thread
replay took35.8 seconds, no GPU. Report and per-episode intersections/unions:
`results/frozen_matte_original_v1/dev241.json` and `dev241.episodes.jsonl`.

### 2026-10-05: text-head name-rank overlap with complete FoRIS errors

COCO-20i exposed DEV241, seed0. Join the saved dino.txt reference-crop true-name rank with the same
episodes' complete cached FoRIS+CRF masks at original annotation resolution. When the true name ranks
first (115 episodes), FoRIS mean per-episode IoU is0.645 and37/115 are below0.5. At ranks2–5
(77 episodes) these are0.630 and20/77; below the top five (49 episodes), 0.465 and25/49.
Rank>5 minus rank1 mean IoU is-0.180 [-0.275,-0.089]; the fraction below0.5 differs by+0.188
[+0.030,+0.348]. Intervals resample239 connected-photo groups2,000 times. This grouping uses the
true class name and is **not** a test-time selector or a method gain. It shows the current top-five
name shortlist misses the exact true-name embedding in a group where FoRIS is also weak; a synonym
inside the shortlist may still be useful. Query-token reranking and complete-pipeline confirmation were
not run. Report: `results/lang_name_diagnostic_v1/foRIS_name_rank_correlation.json`.

### 2026-10-05: finer-token boundary and language-channel GPU queue

协议：COCO-20i 1-shot，DEV241（241 exposed episodes），seed 0；CI95 与报告原值一致。除标注诊断和运行计时外，结果均为 JSON 里的相应指标。每个表标题列出结果文件；DEV241 gains 的控制列注明比较对象。

启动接线（首次尝试）：guard 报 PREPARED_PREFLIGHT_MISSING_OR_EMPTY_QUEUE；main 返回 MAIN_RC 3。guard_main 与 guard 分别在 0.5973806772 秒、0.5560286008 秒记录收尾；没有科学阶段启动。原因是有效回执 pre_plan_main.preflight.json / pre_plan.preflight.json 与 state-file 对应的 guard_main.preflight.json / guard.preflight.json 文件名不一致。两份原回执的 plan digest 和 9 / 8 个输入文件元数据均通过核验；复制到 guard 预期路径后，原队列启动成功。记录：guard/guard_main.json、guard/guard.json、session_all.log。

Layer reading，结果文件 cpu/layer_read.json；以下 best_cut_vs_l24 的对照为同类 l24 view：

| view | cut | best_cut | best_cut_vs_l24；CI95 | verdict |
|---|---:|---:|---|---|
| l8_one | 13.524752624022499 | 18.595060675307227 | -44.19344617948791; [-47.781638089262394, -41.18652868945393] | no |
| l12_one | 14.140054073489924 | 19.529703511390952 | -43.25880334340418; [-46.5417154160951, -39.86223204536605] | no |
| l16_one | 15.668389870141894 | 21.761149675973503 | -41.02735717882163; [-44.309359761692896, -37.45644133300031] | no |
| l20_one | 18.628416320830887 | 27.79685940509425 | -34.99164744970088; [-37.958198479214595, -30.62753855746083] | no |
| l22_one | 21.198795241814718 | 34.19535598361698 | -28.593150871178153; [-30.636327063547917, -23.396742104711226] | no |
| l24raw_one | 41.12161288603582 | 59.87686760297966 | -2.9116392518154726; [-4.224024194574973, -1.3017886873887936] | no |
| l24_one | 44.02604203603113 | 62.78850685479513 | 0.0; [0.0, 0.0] | no |
| joint_one | 41.751015837089206 | 60.014400610927964 | -2.774106243867166; [-3.9175558041304304, -1.3314723298183573] | no |
| l8_bg | 14.152925686853147 | 19.014770523198113 | -46.947975815753054; [-50.1059593891752, -43.524484961623514] | no |
| l12_bg | 16.2020642224052 | 22.077600116464847 | -43.88514622248631; [-47.12934102092727, -40.3542329507508] | no |
| l16_bg | 19.011954710941954 | 26.932939548194252 | -39.02980679075691; [-42.221013410946966, -34.9714435944724] | no |
| l20_bg | 26.10603169362411 | 39.60754112509405 | -26.35520521385711; [-29.02996209637863, -22.102060590442616] | no |
| l22_bg | 32.06128225371077 | 46.865859904652346 | -19.096886434298817; [-21.277172217512724, -14.954680061712688] | no |
| l24raw_bg | 47.94525508056607 | 62.90098394739386 | -3.061762391557302; [-4.534274791724804, -1.442932528336834] | no |
| l24_bg | 51.023885356138045 | 65.96274633895116 | 0.0; [0.0, 0.0] | no |
| joint_bg | 49.09361649205348 | 63.989052479886986 | -1.973693859064177; [-2.7459507931448535, -0.688222058217031] | no |

16 个 view 全部 verdict=no，没有 BETTER。低层、raw last layer 与 joint encoding 的 best cut 均未超过对应 l24 view。

Hmatte gate：结果文件 cpu/hmatte_gate.json；控制为 FoRIS after its CRF、model resolution；DEV241，n=241，seed0。

| 项 | gain / 数值 | CI95 | 教训 |
|---|---:|---|---|
| nested；verdict FAIL | mIoU 59.19025592281649；gain +0.11543084720396735；141 up / 98 down / 239 groups | [-0.166544870268344, 0.38443777022125714] | 区间跨 0，未达预设门槛。 |
| frozen={grid:z2, band:1, mix:1} | frozen_gain +0.11543084720396735 | [-0.18472295887174556, 0.3784273864690272] | 冻结臂仍未分辨出正增益。 |
| s8 / band 1 / alpha | -0.11450594386842283 | [-0.6195509983937537, 0.21934902069270562] | 区间跨 0。 |
| s8 / band 1 / mean(alpha, FoRIS) | -0.12236775756482388 | [-0.6087049374445211, 0.18347357839341175] | 区间跨 0。 |
| s8 / band 2 / alpha | -0.6570997330872714 | [-1.4112857274307424, -0.17147328182896884] | 低于对照。 |
| s8 / band 2 / mean(alpha, FoRIS) | -0.42092696155506815 | [-1.0650676045601755, 0.01693093089389286] | 区间跨 0。 |
| s8 / band 3 / alpha | -1.0445319967001794 | [-1.951508711482268, -0.47545805709345723] | 低于对照。 |
| s8 / band 3 / mean(alpha, FoRIS) | -0.5643057462434058 | [-1.305974697043687, -0.07437084364899367] | 低于对照。 |
| z2 / band 1 / alpha | +0.0663721791468248 | [-0.2686067043901236, 0.37783040181204797] | 区间跨 0。 |
| z2 / band 1 / mean(alpha, FoRIS) | +0.11543084720396735 | [-0.18472295887174556, 0.3784273864690272] | 区间跨 0。 |
| z2 / band 2 / alpha | -0.4475324124581519 | [-0.9787834810340333, 0.024874868886476508] | 区间跨 0。 |
| z2 / band 2 / mean(alpha, FoRIS) | -0.17584608777848132 | [-0.6542626735008922, 0.24235661458151123] | 区间跨 0。 |
| z2 / band 3 / alpha | -0.7963684391971881 | [-1.508175616184544, -0.2565908367732313] | 低于对照。 |
| z2 / band 3 / mean(alpha, FoRIS) | -0.2920048102719264 | [-0.8614574524759225, 0.17443805789121372] | 区间跨 0。 |

Hmatte 标注诊断，均为 DEV241、n=241、seed0；使用真值，不是部署结果；JSON 没提供 CI95：

| label diagnostic | 值 |
|---|---:|
| s8 / fine one-vector score / best cut | 66.96497844592602 |
| s8 / truth trimap / alpha | 94.47704714927812 |
| s8 / truth trimap / true coverage | 98.00560969948073 |
| z2 / fine one-vector score / best cut | 67.06259918265468 |
| z2 / truth trimap / alpha | 94.12135738516993 |
| z2 / truth trimap / true coverage | 98.02159316954048 |

Pipe reports：
- results/hires_pipe_smoke/report.json：4 个 smoke episodes；model_resolution 字段实际有 4 个值：foris_before_crf 64.59089142977608、foris 65.11100686328469、refined 66.16371140621958、gate_said 1.0567853485205632。该对象没有 CI95；完整 smoke stage 返回 0。
- results/hires_pipe_dev/report.json：state=COMPLETED，skipped="gate verdict FAIL"，go=false。
- results/hires_pipe_confirm/report.json：state=COMPLETED，skipped="gate verdict FAIL"，go=false；没有 CONFIRM600 查询掩码评分。

Language channel：
- bank/dev/report.json：DEV241，episodes=241、written=237、seconds=200.3、seconds_per_episode=0.845、seed0。
- results/gate_dev.json label-free nested：verdict=NAMING；mIoU 59.65330834210317；gain +1.0913027372649964，CI95 [-0.2668839594184986, 2.335414260243562]；130 up / 109 down / 239 groups。控制为 FoRIS 58.56200560483818 at model size before CRF，DEV241 n=241，seed0。
- results/gate_dev.json true-name nested：mIoU 62.86791017744824；gain +4.305904572610061，CI95 [1.889860994707424, 6.314822518756141]；131 up / 109 down / 239 groups。控制同上。true-name signal passed the NAMING threshold; label-free nested did not.
- results/pipe_dev/report.json：state=COMPLETED，skipped="gate verdict NAMING"，go=false。
- results/pipe_confirm/report.json：state=COMPLETED，skipped="gate verdict NAMING"，go=false；没有 CONFIRM600 text-head pipe 评分。

队列阶段用时由 session_all.log 和 guard elapsed_seconds 的 GPU_RUNNING→STAGE_COMPLETED 差值计算（不是模型分数）：

| 阶段 | 秒 |
|---|---:|
| main / smoke_hires | 27.2295 |
| main / smoke_layers | 15.7647 |
| main / smoke_read_hires | 4.5317 |
| main / smoke_read_layers | 4.5627 |
| main / smoke_pipe | 29.5973 |
| main / hires_dev | 943.1022 |
| main / layers_dev | 306.4852 |
| main / read_layers | 57.1900 |
| main / read_hires | 77.3162 |
| main / pipe_dev | 2.2507 |
| main / pipe_confirm | 2.3023 |
| language / smoke_bank | 131.7686 |
| language / smoke_gate | 4.5462 |
| language / smoke_pipe | 25.0102 |
| language / bank_dev | 213.6145 |
| language / gate_dev | 40.7893 |
| language / pipe_dev | 2.2557 |
| language / pipe_confirm | 2.2056 |
| language / refs_confirm | 306.9375 |
| language / read_window | 51.5386 |

运行/收尾：main guard elapsed_seconds=1476.0639655012637；language guard 到 SHUTDOWN_REQUESTED 为 783.7706240154803；两段 guard 合计 2259.834589516744 秒。session PID 在 read_window 开始时已运行 37m39s，再加 read_window 51.5386 秒及关机请求 0.5179 秒，从 session 进程启动到 guard 关机请求约 38m31s；main 收尾清理 hires bank 约 51 秒不包含在 guard 阶段用时。guard 的 SHUTDOWN_REQUESTED helper=/usr/bin/shutdown，但 billing_stop_confirmed=false。读回时数据盘 17G/50G 可用，overlay 系统盘 19G/30G 可用。用户随后重启为无卡模式以读取结果；额外无卡空闲时长不算入 GPU 队列运行时长。

### 2026-10-05: native's errors as delete and add budgets; sealed arms as edit operators (CPU, GT diagnostics)

DEV241, complete 1024 masks, native 59.07; sealed masks only. Budgets with truth: delete every false pixel +22.84
(whole wrong regions +7.53, attached to the target +11.42); add every missed pixel +13.31 (whole missed objects +3.46,
completing touched objects +9.85). As edit operators: RCG deletions only +1.67 [+0.82, +2.64] (80.1% false), additions
only +0.34 [-0.05, +0.70] (44.6% true); D +0.54 / -0.34; transition only +1.42 / -0.16; concat -0.98 / -2.28. Pooled
break-even: 62.2% false for a deletion, 37.8% true for an addition; all eight signs follow it. Screens: D reads +3.26
on first20, +1.08 on mini50, +0.12 on 241; a random 50 of 241 has sd 1.1 to 1.6. Lesson: every arm so far is a
deleter; nothing adds. With side effects halved and no benefit lost, transition-only additions plus RCG deletions read
+3.14 [+2.20, +4.27] (now +1.32); arms rejected on net gain recover far more missed pixels than RCG (two-slot EM 53.9%
on mini50 against 8.0%). Full tables: [edit_budget.md](research_20261005/edit_budget.md).

### 2026-10-05: the model origin, INSID3 and FoRIS stage by stage on it, and the family containing both (stage241_v1)

DEV241 (79 classes, 239 photo groups), seed 0, 1024, class mIoU, development data; float32 saved final-layer tokens,
no CRF except `native`. Scores: origin `model.raw_nn` 42.90; `model.raw_mean` 39.56; rebuilt INSID3 rule 54.47;
rebuilt FoRIS pre-CRF 58.61; complete FoRIS 59.07. Parity with the cached public stage responses: final response
within 1e-4 on 218 of 241 episodes (s2 240, s3 223); rebuilt pre-CRF mask differs from the cached one in 72,736
pixels over 29 episodes. Complete FoRIS as an edit of the origin: adds 4.14M true / 3.36M false (55% true), deletes
11.74M false / 1.58M true (88% false); additions alone 48.78, deletions alone 52.77. FoRIS chain: s2 51.94, +vote
52.20, +seed prior 56.36, +penalty 57.35, +cluster delta 58.61, +CRF 59.07. Removing one term from pre-CRF (no
selection, in sample): fg -5.16, prior -2.09, delta -1.26, penalty -0.72, bg +0.10, vote +0.75; never projecting -2.97.
Family chosen on three folds and read on the fourth: all terms 57.35, -1.73 [-2.80, +0.05] vs native; threshold only
58.79, -0.29 [-1.24, +0.47]; one-at-a-time lines 58.88, -0.19 [-1.41, +0.66]; best point in sample 59.92.
Value by response level: FoRIS's per-image level turns positive between 0.525 and 0.575 (cut 0.5).
Lesson: re-weighting, re-cutting or mixing the two public pipelines' own terms does not beat complete FoRIS; the cut
and the weights are already near their optimum on this term set, so a gain has to come from evidence outside it.
Source: [report](research_20261005/stage241_v1/report.md); sealed masks on the server, `outputs/stage241_v1`.

### 写了但没有跑就撤回的

部分对应（PCF）、不变概念与可移植提示、响应 pair-ratio 编辑器、响应训练路线曾在未取得足够前提证据时撤回。早期 `pair_bank.py` 准备被撤回，不表示后续 layer/joint 队列没有运行（结果见上）。撤回未运行构造不证明输入缺少信息。

