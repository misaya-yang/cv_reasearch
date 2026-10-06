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

### 2026-10-05: the Astra `external_mean__delete` candidate replayed on DEV241 and read on the 600-episode cohort

Delivered code unchanged (sha256 800ad3dc… of `external_mean_delete600.py`), CPU; seed 0, 1024, class mIoU, 2000 photo-group
draws. DEV241 (development, repeatedly read): native 59.07, RCG 61.02, C 61.84, candidate 62.65 = +3.58 [+1.32, +4.82] vs
native, +0.81 [-0.44, +1.64] vs C, +0.86 [-0.18, +1.85] vs its same-count lowest-RCG control; on the delivered 220 subset
62.67 / 58.83 (reported 62.675 / 58.830). **600 cohort** (`confirm_episodes.json`, 80 classes, 561 photo groups; FoRIS masks
are the stored `foris_confirm_v1`, features re-exported through the same host, 600 of 600 masks bit-identical): native 60.07;
RCG 61.07 = +0.99 [+0.32, +1.58]; C 60.81 = +0.74 [-0.30, +1.61]; candidate 60.67 = +0.60 [-0.63, +1.87] vs native,
-0.39 [-1.38, +0.77] vs RCG, -0.12 vs its same-count control; fold gains of the candidate +1.39 / +0.94 / +1.66 / -1.59;
its last step removes 1.03M true and 1.81M false pixels (64% false; 71% on DEV241). Variants frozen from the DEV241 wide
sweep before the 600 read: `astra + add[far, t0.4]` 61.14 = +1.07 [-0.23, +2.23] vs native, +0.46 [-0.15, +0.88] vs the
candidate; `delete_p + add_far` 61.09; none separates from RCG. My split-pool proposal: 58.75 on DEV241 and 58.20 on 600 (worse).
Operators chosen greedily from existing whole masks (DEV241): two steps reach RCG (61.02), further steps add 0.02.
Lesson: the two deletion steps selected on DEV241 do not transfer; the only gain that holds on the 600 cohort is RCG's.
The 600 cohort has now been read for these arms and for the sweep and is no longer untouched.
Sources: [600 report](research_20261005/recheck600_v1/report_wide.md), [241 report](research_20261005/recheck241_v1/report_wide.md),
[greedy](research_20261005/greedy241_v1/report.md); runs on the server `outputs/{recheck241_v1,recheck600_v1,confirm600_root,greedy241_v1}`.

### 2026-10-06: the same fixed arms on a second 600-episode cohort (fresh600, photo-isolated from DEV241 and the first 600)

First 150 kept draws per fold of the prepared `train_episodes.json` (e 60 to 234), seed 0, class mIoU, 2000 photo-group draws;
FoRIS run fresh through the same host. At 1024: native 61.63; RCG 64.24 = +2.61 [+1.39, +3.23]; C 63.59 = +1.96 [+0.33, +2.79];
Astra candidate 63.50 = +1.87 [+0.15, +2.92] vs native and -0.74 [-1.89, +0.27] vs RCG; folds of the candidate +0.83 / -1.05 /
+4.11 / +3.59. At the original query resolution (FoRIS finish): native 61.42, RCG +2.61 [+1.38, +3.22], C +1.91, candidate +1.83
[+0.09, +2.89]. Frozen pairs: `delete_p + add_far` 64.17 (-0.07 vs RCG); `astra + add_far_strict` 63.43. Both 600 cohorts agree:
RCG is positive (+0.99, +2.61) and the two deletion steps sit below RCG (-0.39, -0.74), unresolved.
Source: [report](research_20261005/recheck_fresh600_v1/report_wide.md), [original resolution](research_20261005/recheck_fresh600_v1/original.md).

### 2026-10-06: fixed arms on the complete public COCO-20i list (4000 episodes, original resolution)

Seed-0 draws 0 to 999 of each fold (six batches of 600 and one of 400), complete FoRIS run fresh; class mIoU at the original
query resolution (sealed 1024 masks resized as FoRIS finishes), 2000 photo-group draws. FoRIS 60.70 (folds 58.5 / 63.5 / 61.0 /
59.8); RCG 62.11 = +1.41 [+1.08, +1.70], folds +1.30 / +1.14 / +1.41 / +1.78; C 61.95 = +1.25 [+0.80, +1.62]; Astra candidate
61.66 = +0.95 [+0.36, +1.46], folds +1.03 / +0.32 / +1.65 / +0.81; its same-count control +1.03 [+0.45, +1.52]. RCG per batch:
+1.77, +2.13, +1.89, +1.53, +1.62, +0.67 [-0.03, +1.48], +0.42 [-0.37, +1.23]; cumulative RCG gain after each batch: +1.77,
+2.06, +1.84, +1.69, +1.67, +1.48, +1.41. At 1024: FoRIS 60.93, RCG +1.40 [+1.09, +1.69], frozen `delete_p` +1.40 [+0.78, +1.97],
frozen addition pairs +0.91 to +1.34: nothing separates from RCG. The list includes the DEV241 draws (in batch 0) and overlaps the
fresh600 draws. Our arms carry no CRF; FoRIS does. Source: [original resolution](research_20261005/official_batches/original.md),
[1024 with frozen pairs](research_20261005/official_batches/cumulative.md); server `outputs/claude_official`.

### 2026-10-06: RCG applied to other input scores (DEV241, 1024)

The delivered RCG readout fed each stored 64 x 64 field in place of the final FoRIS score; features and coverage unchanged (FoRIS
Part-1 debiased features). "Alone" is min-max > 0.5, a crude readout for raw scores. Complete FoRIS 59.07. FoRIS pre-CRF score:
58.61 -> 61.05 (+1.97 [+1.01, +2.94] vs FoRIS); FoRIS stage 3: 56.36 -> 59.69 (+0.62 [-0.39, +2.36]); INSID3 combined: 52.69 ->
53.54; model raw mean similarity: 37.98 -> 42.76 (-16.31 vs FoRIS); model raw nearest neighbour: 20.12 -> 20.89. RCG raises 17 of
18 inputs over themselves (+0.8 to +10.5) but its level follows the input: on the model's own scores it is far below FoRIS.
Source: [report](research_20261005/rcg_hosts241/report.md).

### 2026-10-06: grouping-based adder and deleter on RCG, and where RCG's remaining error lies (fresh600, 1024)

Average-linkage groups of the cached query features (tau 0.5 / 0.6 / 0.7); groups mostly inside R completed, groups mostly
outside trimmed; 72 additions x 45 deletions, nested over folds. RCG 64.24; `add.nested` -0.08 [-0.46, +0.20]; `delete.nested`
-0.04 [-0.40, +0.25]; `both.nested` -0.11 [-0.63, +0.27]; the same-count RCG-score control +0.11 [-0.31, +0.45]. Best addition
alone: 1.23M true / 1.31M false (48% true), 64.29 against 64.40 for its same-count control: no variant beats its control.
GT diagnostic (not inference): RCG leaves 17.13M false and 12.18M missed pixels; 19% / 29% of them lie within 8 px of the true
boundary and 26% / 41% within 16 px; correcting only the pixels within 8 px would read 72.26, within 16 px 75.25 (FoRIS: 68.87, 71.69).
Lesson: similarity to the reference, the query's grouping and the score level are already used up by RCG at the 64 x 64 token
grid; a large share of the remaining price sits below one token from the boundary. Source: [report](research_20261005/cluster_fresh600/report.md).

### 2026-10-06: which part of RCG carries its gain (fresh600, 1024) and what it changes (public 4000, 1024)

Diagnostics; truth is read only to count; nothing was selected. The re-solved RCG field matches the sealed one within 6e-8.
Ablation, fresh600, against the FoRIS pre-CRF mask 60.94 (complete FoRIS 61.63): rank correction alone +0.00 [-0.58, +0.28];
graph smoothing alone +2.64 [+1.79, +3.27]; both (RCG) +3.29 [+2.16, +4.00], and RCG minus smoothing-alone is +0.66 [+0.06, +1.06];
equal anchoring -0.47 [-0.83, -0.18] against RCG; a 5 x 5 position graph in place of the feature graph +1.43 at its best
strength (lambda 1), -11.89 at lambda 16; lambda 1 / 4 / 16 / 64: 62.02 / 63.19 / 64.24 / 64.71 (64 against 16: +0.47 [-0.40,
+1.01]); alpha 1: 46.35. RCG edits on fresh600: adds 1.93M true / 1.91M false, deletes 4.23M false / 1.57M true.
Anatomy, public 4000, against the FoRIS pre-CRF mask 60.41 (complete FoRIS 60.93, RCG 62.33 = +1.92 [+1.61, +2.22] over pre):
each family of RCG's edits applied alone: deleting connected parts almost whole +0.80 [+0.63, +0.98] (10.82M false / 2.31M
true); deleting within 16 px of the edge +0.37; adding within 16 px +0.31; filling enclosed holes +0.26 (70% true); deleting
interior +0.10; adding far +0.03 (37% true). Error RCG leaves (million pixels: object-level / within 8 px of the true
boundary / 8 to 16 px / further): missed 9.4 / 19.9 / 9.2 / 65.4; false 33.6 / 20.0 / 8.1 / 56.8 (FoRIS pre: missed 4.2 / 20.5
/ 9.8 / 69.8; false 43.7 / 21.4 / 8.8 / 61.7). GT ceilings from RCG: object-level alone 68.26, within 8 px 68.55, 8 to 16 px
64.67, further than 16 px 78.43. RCG minus pre by object area (< 2% / 2-10% / 10-30% / > 30% of the image): +2.76 / +2.42 /
+0.97 / +1.09, with scores 33.6 / 58.1 / 70.9 / 70.1. By batch: +2.32, +2.89, +2.41, +2.15, +2.20, +1.33, +0.91 (episodes
better / worse 252 / 134 and 165 / 105 in the last two against about 270 / 120 before); the cause of the weaker last batches
is not identified. Source: [ablation](research_20261005/rcg_ablate_fresh600/report.md), [anatomy](research_20261005/rcg_anatomy4000/report.md).

### 2026-10-06: where RCG's remaining error sits by episode, and four label-free ways to set the cut level

Diagnostics and rules scored from stored per-level counts; truth is read only to count. Public 4000, 1024, RCG 62.33.
Concentration: the worst 10% of episodes hold 60% of the false pixels and 71% of the missed pixels. 511 episodes with precision
< 0.5 and recall > 0.7 hold 52.8M of 118M false pixels; 174 episodes with precision > 0.7 and recall < 0.5 hold 34.9M of 104M
missed pixels. Mask size over object size by object area (< 2% / 2-10% / 10-30% / > 30% of the image): 2.94 / 1.41 / 0.99 / 0.81.
Cut level: best level per episode (GT) 71.24 for the RCG field and 70.18 for the FoRIS score; best single level per true-area
bin 0.625 / 0.575 / 0.463 / 0.312, worth +3.21 / +1.46 / +0.37 / +5.29 inside the bins (GT bins); rank correlation of the best
level with true area -0.36. Label-free rules on the RCG field: Otsu 36.22, isodata 49.71, median midpoint 49.68, Kittler 49.91;
ridge from the whole area-versus-level curve to the best level, nested over folds, -3.44 [-4.14, -2.68]; level chosen per bin
of the observable mask area at 0.5 (six bins, fitted on the other folds without shared photographs, so it uses base-fold
labels): 62.71 = +0.38 [+0.13, +0.64], folds +0.54 / +0.44 / +0.40 / +0.14, picks 0.50 / 0.54 / 0.54 / 0.53 / 0.46 / 0.38 from
small to large masks (FoRIS score: +0.83 [+0.52, +1.15]). fresh600: level maximising the share of reference foreground minus
reference background tokens whose nearest query token is above the level: 57.41 against RCG 64.24 (soft matching 51.03); its
rank correlation with the best level 0.14. Source: `research_20261005/cut_levels4000/`, `research_20261005/reference_cut_fresh600/`;
server `outputs/claude_cut_levels4000`, `outputs/claude_reference_cut_fresh600`.

### 2026-10-06: sub-token readout of the RCG field (fresh600, 1024)

The sealed 64 x 64 RCG field upsampled with a 128 x 128 guide from four shifted encoder passes (+-4 px), nested over folds:
64.63 = +0.39 [+0.33, +0.57] against RCG 64.24 and +0.18 [+0.01, +0.35] against RCG with the FoRIS CRF finish (64.45, itself
+0.21 [+0.08, +0.47]); the colour-guided control +0.15 [+0.02, +0.31]; CRF after the feature-guided readout adds nothing
(64.50). Errors within 8 px of the true boundary move from 3.20M false / 3.57M missed to 3.11M / 3.33M. The GT ceiling for
that band was +8.0. Source: [report](research_20261005/subtoken_fresh600/report.md).

### 2026-10-06: the cut level derived as a decision under an unknown object share (public 4000 and fresh600, 1024)

Derivation: with class-conditional distributions f1, f0 of the field value fixed across episodes and the object's share of the
query pi, the posterior is pi f1 / (pi f1 + (1 - pi) f0) and the set maximising expected IoU is a level set of it; the level of
the field therefore depends on pi alone. Scored from stored per-level counts; f1, f0 pooled from the other folds without
shared photographs (base-fold labels). Public 4000, RCG field (62.33 at 0.5): mean field value on the object by true-area bin
0.69 / 0.73 / 0.72 / 0.64 and on the background 0.158 / 0.156 / 0.156 / 0.170 (the invariance holds approximately); with the
TRUE share given (GT diagnostic) the derived cut reads 67.40 = +5.07 [+4.27, +5.51], folds +6.45 / +4.70 / +4.63 / +4.50, of
the 71.24 per-episode best level; FoRIS score 66.46 = +6.05 [+5.28, +6.55]. Share estimated from the field's own histogram
(mixture EM): 58.25 = -4.08 [-4.79, -3.44] although its rank correlation with the true share is 0.80. Required accuracy
(independent noise on the log-odds of the true share): sd 0.25 / 0.5 / 0.75 / 1.0 / 1.5 -> +4.97 / +4.40 / +3.92 / +2.21 /
-0.92. fresh600 (RCG 64.24; true share given 67.79), estimators that do not read the field: reference share, error sd 1.74,
58.39; reference share times the squared scale from mutual-nearest-neighbour geometry, sd 1.73, 56.75; share of query tokens
whose nearest reference token is on the object, sd 1.46, 59.93; least-squares combination fitted on the other folds, sd 1.21,
62.91 = -1.33 [-2.19, -0.21]; mask area at 0.5, sd 0.75 but dependent on the field's errors, 63.65 = -0.58 [-1.37, +0.08].
No estimator tested reaches the required accuracy. Source: `research_20261005/cut_levels4000/`,
`research_20261005/reference_cut_fresh600/`, `research_20261005/share_fresh600/`; `scripts/run_share_estimators.py`.

### 2026-10-06: RCG with its measured auxiliaries together, and the object share as a mixture proportion (fresh600, 1024)

Combined run, complete FoRIS 61.63, RCG 64.24; the guide temperature and the level per mask-size bin are chosen on the other
folds without shared photographs (the size levels use base-fold labels). Against RCG: strength 64 +0.47 [-0.40, +1.01];
feature-guided sub-token readout +0.39 [+0.33, +0.57]; size-dependent cut +0.49 [-0.33, +1.00]; readout + size cut 65.44 =
+1.20 [+0.33, +1.81], folds +2.61 / +0.48 / +0.44 / +1.29, and +3.81 [+2.23, +4.59] against FoRIS; strength 64 + readout +1.01
[+0.23, +1.64]; strength 64 + size cut +0.27 [-0.73, +1.02]; all three +0.60 [-0.36, +1.45]. The strength-64 field is re-solved
with dense operations on the GPU (largest difference from the sealed field when re-solving strength 16: 0.0018).
Mixture proportion estimate of the object share (reference-object tokens as positives, query tokens as the unlabelled
mixture, ratio of cosine-ball masses, low quantile over reference tokens; no score field, no query truth): at cosine 0.6 and
above the median estimate is 0 (most reference-object tokens have no query token that close); at 0.4 the median estimate over
truth is 1.00 at the 0.25 quantile but the log-odds error sd is 2.92 (rank correlation 0.45); plugged into the derived cut:
-15.9 to -30.6. Failed link: tokens of the same class in two photographs do not share one distribution in this feature
space, which the estimator assumes. Source: `research_20261005/rcg2_fresh600/`, `research_20261005/share_fresh600/pu.npz`;
`scripts/run_rcg2.py`, `scripts/run_share_pu.py`.

### 2026-10-06: frozen combined version on two further groups of 600 (COCO-20i, seed 0, photograph-isolated list)

Settings frozen on fresh600 before these groups were opened ([frozen file](research_20261005/rcg2_frozen.json): RCG unchanged;
feature-guided readout sigma 1.25, tau 0.15; cut level by mask area at 0.5: 0.50 / 0.5625 / 0.5875 / 0.5625 / 0.45 / 0.375).
Groups B and C are draws 150-299 and 300-449 per fold of `train_episodes.json` (fresh600 is 0-149); FoRIS is run fresh,
features are not kept. Class mIoU at the original resolution, 2000 photo-group draws. The re-solved RCG field differs from
the sealed one by up to 0.024 on eight fresh600 episodes (FoRIS rerun float differences).
B: FoRIS 60.22; RCG +1.29 [+0.54, +2.26]; RCG + readout +1.75 [+1.02, +2.78]; full 61.80 = +1.59 [+0.34, +2.98], against RCG
+0.30 [-0.53, +1.22]. C: FoRIS 58.27; RCG +1.81 [+0.87, +2.62]; + readout +2.25 [+1.32, +3.15]; full 61.65 = +3.38 [+1.85,
+4.61], against RCG +1.57 [+0.45, +2.56]. B+C (1200): FoRIS 59.67; RCG 61.19 = +1.52 [+0.91, +2.20]; + readout 61.61 = +1.93
[+1.32, +2.67], against RCG +0.42 [+0.35, +0.51]; full 62.17 = +2.49 [+1.47, +3.47], folds +2.14 / +2.55 / +2.20 / +3.08,
against RCG +0.98 [+0.13, +1.63]. The readout gain repeats in every group (+0.39, +0.46, +0.44); the size cut adds -0.17 in B
and +1.13 in C over the readout. Source: `research_20261005/rcg2_groupB/`, `research_20261005/rcg2_groupC/`;
`scripts/run_rcg2_stream.py`; server `outputs/claude_rcg2_group{B,C}`.

### 2026-10-06: frozen combined version on PASCAL-Part without refitting (598 episodes, 1 shot)

Pack of INSID3's own loader (150 per benchmark fold, seed 0, 56 dense classes; not INSID3's exact evaluation list; `fold` in
the table is the pack index modulo 4). Same frozen file as on COCO; FoRIS run fresh. Original resolution: FoRIS 53.87; RCG
55.33 = +1.46 [+0.67, +2.40]; RCG + feature-guided readout 55.35, against RCG +0.02 [-0.09, +0.10]; full version 54.83 =
+0.96 [-0.06, +2.50], against RCG -0.50 [-1.24, +0.57]. At 1024: FoRIS 52.03, RCG +1.60 [+0.92, +2.45], readout -0.01, full
-0.13 [-0.82, +0.76] against RCG. RCG transfers; neither auxiliary does. Source: `research_20261005/rcg2_grouppascal_part/`;
server `outputs/claude_rcg2_pascal_part`, pack `outputs/claude_packs/pascal_part`.

### 写了但没有跑就撤回的

部分对应（PCF）、不变概念与可移植提示、响应 pair-ratio 编辑器、响应训练路线曾在未取得足够前提证据时撤回。早期 `pair_bank.py` 准备被撤回，不表示后续 layer/joint 队列没有运行（结果见上）。撤回未运行构造不证明输入缺少信息。


### 2026-10-05: locked Astra and all fixed controls on existing600

COCO-20i 1-shot, seed0,1024,600episodes/80classes/561connected-photo groups (largest4).
Previously exposed reevaluation; photograph-disjoint fromDEV241, not fresh confirmation.
All10supplied arms plus completeFoRIS were replayed unchanged and sealed before scoring;
five shared arms match the independent recheck output in every pixel on all600.
FoRIS cache export:600/600bit-identical; response/coverage maximum differences0;889.018s.
Paired2000RandomState(0) photo-connected bootstrap:

| Complete arm | mIoU | Gain vs completeFoRIS [95%] | Consequence |
|---|---:|---|---|
| FoRIS |60.073377|reference|Same-cohort comparator|
| MEAN_CONTROL |61.467653|+1.394276[.824740,1.907805]|Strongest supplied simple control; not the target+2|
| RCG |61.067693|+.994316[.319832,1.583927]|Positive complete difference on this cohort|
| C (RCG_count_matched_delete) |60.813312|+.739935[-.304543,1.611592]|Deletion-budget increment overRCG unestablished|
| Fixed Astra external_mean__delete |60.673832|+.600455[-.631630,1.872440]|DEV241+3.579518 did not persist here|
| Astra same-count RCG ranking |60.793953|+.720576[-.532466,1.832643]|Query-mean selection does not establish an increment|
| Supplied cached INSID3 rule |55.650501|-4.422877[-6.070172,-2.301684]|This is the supplied cached rule, not an officialBF16 reproduction claim|

Astra vsRCG:-.393861[-1.383725,.767139]; vsMEAN:-.793821[-1.876295,.470893].
Astra gain vsFoRIS by fold:1.389016,.941713,1.656552,-1.585461;300up/295down/5tie.
The lastC-to-Astra step is-.139480[-.642106,.729691], with1.025M true pixels and1.810M
false pixels deleted. Same-count RCG ranking removes.935M true/1.901M false. These
point differences locate a possible selection failure but do not establish a negative true effect.
Frozen DEV-selected strict addition61.138748, vsFoRIS+1.065371[-.227695,2.228872],
vsRCG+.071055[-1.008208,1.167585], vs same-count+.079651[-.195803,.441347];fold3
gain-1.737282. No600nested/best-in-sample row was promoted to the fixed primary.
Source:[fixed11-arm report](research_20261005/pipeline_verified/fixed600/report.json),
[per-episode counts](research_20261005/pipeline_verified/fixed600/episodes.jsonl),
[full replay parity](research_20261005/pipeline_verified/fixed600/replay_parity.json),
[frozen comparison](research_20261005/pipeline_verified/recheck600/report_wide.json).

### 2026-10-05: fixed raw-origin witness edits onDEV241

COCO-20i1-shot,seed0,1024,DEV241/79classes/239photo groups. All89rows complete and
scored. Best complete output remainsRCG61.019660,+1.944835[.983141,2.913596]vsFoRIS.
The strongest witness-assisted complete combination is50.866779, vsRCG-10.152880
[-13.031506,-7.764400]. Its RCG edit helper removes50.0%of harmful additions but
loses42.1%of beneficial additions; protects78.4%of harmful deletions while losing59.5%
of beneficial deletions. Selectivity alone does not preserve enough edits to reach a
complete gain from rawNN. This limits the fixed thresholded witness construction.
Source:[report](research_20261005/pipeline_verified/edit_aux241/report.json).

### 2026-10-05: exact bounded two-operation family onDEV241

COCO-20i1-shot,seed0,1024,DEV241/79classes/239photo groups;36sealed source/field masks.
5257fixed<=2-operation recipes from rawNN;three-fold GT fitting excludes connected
photos of the held-out fold. This is DEV-label-based method selection,not per-queryGT inference.
Real2-case GPU byte-popcount check and score passed,then all241final masks sealed/scored.
GPU inference23.145s,peak1,358,301,696bytes,no encoder forward.
One-step53.435379,joint2=61.503062,direct complete-mask selection61.993386,Astra62.654343.
Original scorer:joint2 vsFoRIS+2.428237[.237942,3.899579],vsAstra-1.151281[-1.859609,-.194749].
Supplementary supplied-RandomState comparison:joint2 vsdirect-.490324[-.823979,-.013517],
vsone-step+8.067683[6.107601,10.693817]. Intervals retain scorer identity rather than
being silently substituted; their bootstrap implementations have different draw ordering.
Training-fold joint improvement over direct is only.27654,.00613,.05003,.00083; the extra
selection did not transfer. This limits the testedbounded family/selection,not allA*/B* frameworks.
Sources:[complete report](research_20261005/pipeline_verified/joint241/report.json),
[explicit pairwise comparison](research_20261005/pipeline_verified/joint241/optimizer_pairwise.json),
[sealed masks receipt](research_20261005/pipeline_verified/joint241/sealed.json).

### 2026-10-05: cleanup of completed poor existing600 attempt

User explicitly authorized pruning poor600attempts after recording them. Removed10,086,316,508bytes
from generatedold600featurecache and redundantrecheck prediction/field/count artifacts,after confirming
no active consumer. Fixed11-arm control predictions,code,complete reports,per-episode I/U and hashes
remain; new prediction on theold600cohort requires feature reencoding. Models,datasets,DEV241shared
inputs and the running isolated600queue were preserved. [Receipt](research_20261005/cleanup_existing600_receipt.json).

Correction after the user's clarification: removing10,067,480,435bytes of reusable features exceeded
the authorized cleanup scope. Only18,836,073bytes were disposable exploratory predictions/fields/counts.
The deletion is irreversible from the available records; no cache reconstruction was launched.
The user explicitly redirected priority to A*/B* framework selection. Preserve all remaining reusable features.

### 2026-10-05: matched-depth optimizer and operator-accounting execution

`joint_operator_v3_shared` supervisor15425/start951979446 dispatched at16:28UTC. Adds same-depth
greedy2 and a fixed joint-selection guard requiring improvement on every fitting fold over the selected
complete method; full DEV241 efficacy pending. Same36sealed masks/5257recipes, rawNN origin, no new
encoder, no query-GT routing. One CPU thread, prior measured GPUpeak1.36GB, concurrent with external
encoder14579/start951953583 (measured2.48GB). The waiting-only supervisor14810 was cancelled before
launching any scientific child. No peer process/source change.
Exact four-pixel set accounting passed all65,536truth/origin/two-mask combinations: same-family
overlap and both orders of add/delete conflict, separated into true/false counts. This checks accounting,
not efficacy. [Receipt](research_20261005/operator_pair_accounting_check.json).

### 2026-10-05: matched-depth A*/B* optimization and reference-only inference

COCO-20i1-shot,seed0,1024,DEV241/79classes/239connected-photo groups; rawNN accounting origin.
The matched-depth comparison uses the same36sealed masks and5257fixed recipes. Joint2 vs greedy2
-0.141093[-0.216928,0.020940]; robust fitting-fold selection vs direct +0.000131[-0.000065,0.000422].
It does not establish an optimization advantage. Exact counts/overlaps/conflicts remain reusable.
Source:[matched-depth report](research_20261005/pipeline_verified/joint241_v3/report.json),
[family table](research_20261005/pipeline_verified/joint241_v3/family_table.csv).

All four subsequent inference rules use only the supplied reference coverage, frozen q/r features and
sealed producer masks/fields; no query labels or additional encoder forward. Final masks are sealed
before separate CPU scoring. Configurations were developed sequentially on this exposed DEV cohort.

| Rule | mIoU | vs complete FoRIS [95%] | vs RCG [95%] | vs same-information direct selector [95%] |
|---|---:|---|---|---|
| Reference self-calibration | 37.893435 | -21.181390 [-24.900560,-17.964650] | -23.126235 [-27.092015,-19.776346] | -2.351101 [-2.924906,-1.685347] |
| Symmetric cross-image calibration | 40.726408 | -18.348417 [-21.610803,-14.374340] | -20.293262 [-23.563030,-16.142545] | -3.455120 [-4.345526,-2.351355] |
| Cross-image mass + RCG ordering | 61.042872 | +1.968047 [0.477729,3.497351] | +0.023203 [-1.057945,1.208822] | +0.070994 [-0.194423,0.358380] |
| Two-estimator guard | 61.073647 | +1.998822 [0.567860,3.480652] | +0.053978 [-1.017131,1.207666] | -0.099949 [-0.310364,0.205661] |

The self-calibrated rule overestimates query foreground by400,820pixels on average (Brier0.3434).
Cross-image calibration reduces mean bias to-10,832pixels (Brier0.1395) while complete mIoU remains40.7264.
RCG ordering plus consensus-bounded mass reduces Brier to0.04542 and mean absolute area error to43,445pixels.
Its gain over same-area RCG cutoff is+1.579200[1.071977,3.406243], but gain over RCG is unresolved.
The two-estimator guard improves the unguarded rule only+0.030775[-0.211685,0.290778]; it does not repair
full-mask quality ranking sufficiently. Do not extend this global-calibration grid without a changed failed link.

Guarded rule fold gains vsFoRIS:0.178822,1.939118,1.886174,4.096036;121up/115down/5tie.
Edits vsrawNN:addTP4,253,460/addFP3,416,454/deleteTP1,083,881/deleteFP11,437,357.
Inference runtimes are cached selection only; they exclude complete producer execution. Protocol/seals retain
exact code, inputs and recipe choices. Results and separate readouts:
- [calibrated241_v1](research_20261005/pipeline_verified/calibrated241_v1/report.json).
- [calibrated241_v2](research_20261005/pipeline_verified/calibrated241_v2/report.json).
- [calibrated241_v3](research_20261005/pipeline_verified/calibrated241_v3/report.json).
- [calibrated241_v4](research_20261005/pipeline_verified/calibrated241_v4/report.json).

### 2026-10-05: independent complete controls on historical isolated600

COCO-20i1-shot,seed0,1024,600episodes/80classes/564connected-photo groups; historical training-pool reuse,
not never-seen confirmation. Fixed delete-p was recorded on DEV before this cohort; the wide search stays exploratory.
Independent CPU scoring reproduces all native/RCG/C/Astra per-episode I/U from the sweep and adds MEAN.
FoRIS61.628013,RCG64.237696,MEAN64.264087,Astra63.500125,frozen delete-p64.369663.
Delete-p vsFoRIS+2.741650[0.661121,3.681100],vsRCG+0.131967[-1.315411,1.147701],
vsMEAN+0.105576[-1.414670,1.172892],vsAstra+0.869538[0.051397,1.349022].
Fold gains vsFoRIS:2.077050,0.745346,4.458309,3.685895;298up/294down/8tie.
Frozen delete-p full-mask edit counts are unavailable in the count-only sweep; no zero diagnostics substituted.
The six sealed fixed mask controls retain genuine four-count diagnostics. Strong-control superiority is unresolved.
Source:[complete controls](research_20261005/pipeline_verified/fresh600_complete/report.json),
[per-episode counts](research_20261005/pipeline_verified/fresh600_complete/episodes.jsonl),
[fixed recipe](research_20261005/pipeline_verified/fresh600_complete/recheck_fresh600_v1_frozen.json).

Independent local class-summed reconstruction of every arm in all six reports differs by at most3.56e-14points.
[Receipt](research_20261005/complete_score_reconstruction_receipt.json).

### 2026-10-05: public-labelled4000 manifest boundary

The other queue declares1000draws/fold and4000total, with3999unique episode identities and6722photos.
AllDEV241,old600 andhistorical isolated600 episode pairs recur; batch0 itself contains allDEV241 and305of
historical isolated600. These are benchmark re-evaluations,not4000fresh cases. Do not remove a natural
repeated draw to make the sampled benchmark unique. Official sampling/worker RNG still needs version verification.
[Manifest audit](research_20261005/public_queue_manifest_audit.json).

### 2026-10-05: frozen conditional selector expanded to1200

User explicitly requested a larger sample when an advantage remained uncertain. FrozenDEV241 parameters
were applied unchanged to the first300public-labelled draws/fold:1200episodes,80classes,1046connected
photo groups (largest8),seed0,1024. This is benchmark re-evaluation with prior-exposed cases,not fresh confirmation.
Features were preserved/reused for1146exact episode pairs;54missing pairs were encoded in104.57s.
All54native replays were bit-identical; score/coverage differences0. Temporary identical-encoder overlap
used measured29GBheadroom; the extra model exited after extraction. No peer process/source was changed.

| Arm | mIoU |
|---|---:|
| native | 61.612802 |
| rcg | 63.598020 |
| mean.control | 63.581592 |
| astra.control | 63.162218 |
| conditional.joint | 63.679854 |
| conditional.greedy.control | 63.679293 |
| conditional.same_count.control | 63.676546 |
| conditional.rcg_value.control | 63.761748 |

Primary vs completeFoRIS:+2.067053[1.480652,2.622434];vsRCG:+.081835[.011417,.162600].
VsMEAN:+.098263[-.160051,.386496];vssame-countRCG:+.003309[-.015667,.029584].
VsRCG-value-only:-.081893[-.150924,-.016964]. The extra dual-estimator gate is worse than the
strong simple same-information selector; retire this tested construction rather than promote its control
as an independent winning method. The working objective remains unmet.

Fold gains vsFoRIS:1.920883,1.592062,2.139417,2.615850;656up/529down/15tie.
Edits vsRCG:primaryaddTP487,684/addFP628,614/deleteTP536,332/deleteFP945,243;
RCG-value-onlyaddTP964,720/addFP1,121,141/deleteTP676,000/deleteFP1,108,994.
These aggregate differences motivate a direction-specific test: preserve the simpler addition value
while retaining reference evidence only in deletion eligibility. They are not set-level proof of veto causality.
Its newDEV241construction has separate code/outputs; these1200are now development for any revised rule.

Supplemental frozen delete-p63.958235 vsFoRIS+2.345433[1.135312,3.360488],vsRCG+.360215
[-.703107,1.273603],vsMEAN+.376643[-.648147,1.314328],vssame-countdelete-p+.252550
[-.033335,.615528]. Larger1200does not resolve its strong-control advantage. Missing full-mask edit
diagnostics for this count-only arm remain marked missing,not replaced by zeros.

[Main report](research_20261005/pipeline_verified/conditional1200_v1/report.json),
[per-episode I/U](research_20261005/pipeline_verified/conditional1200_v1/episodes.jsonl),
[local reconstruction/parity](research_20261005/pipeline_verified/conditional1200_v1/verification_receipt.json),
[supplemental summary](research_20261005/conditional1200_extended_summary.json),
[frozen rule](../../launch/confirm1200_conditional_v1/freeze.json).


### 2026-10-05: direction-specific selector on retained1200

The revised construction uses the RCG-ranked mass field for additions and the joint expected-I/U
objective; reference transport also constrains deletion eligibility. Families and5percenteditbudget
remain fixed. It was designed after the previous1200result, so these are development results.
COCO-20i1-shot,seed0,1024;1200draws,300/fold,80classes,1046connected-photo groups.
CompleteFoRIS61.612802,RCG63.598020,MEAN63.581592,Astra63.162218.
Directionaljoint63.760280,greedy63.760375,addonly63.701098,deleteonly63.648647,
same-countRCG63.748883,RCG-value-only63.761748.

PrimaryvsFoRIS+2.147479[1.561716,2.702933];vsRCG+.162261[.083281,.256631];
vsMEAN+.178689[-.072742,.466333];vsRCG-value-only-.001468[-.038359,.037012];
vsgreedy-.000095[-.004253,.003637];vssame-count+.011397[-.000846,.028757].
FoldgainsvsFoRIS:1.963464,1.738175,2.283754,2.604522;664up/521down/15tie.
Batch0gainvsFoRIS+1.975866[1.297560,2.575664],vsRCG+.222238[.103588,.337750];
batch1vsFoRIS+2.237091[1.352302,3.048288],vsRCG+.149927[.052856,.274309].
EditsvsRCG:addTP969,991/addFP1,131,414/deleteTP523,348/deleteFP893,784.

The directional rule recovers the loss of the earlier dual gate but does not establish superiority
to the stronger RCG-value-only orgreedycontrols. StablepositiveFoRISgain is insufficient for the full
framework contribution. Nextcomparison tests unrestricted pixel optimization,p1-onlygreedy andsingle
family with identical information and budget; no control is relabeled as a novel method.

Cached1200selection114.69s,peakCUDA241,310,208bytes; complete producer runtime is excluded.
All predictions were sealed before CPUscoring; no extra encoderforward and no querylabels in inference.
Independentlocal class-summed reconstruction reproduces everyarm within2.85e-14points.
The copiedwrapper's source_code list names the original module; separate verification confirms the
actual directionalmodule and helper against the immutablefreeze bound in the protocol.
OfficialINSID3 replay andrawDINO1200masks remain unavailable; DEVraw-origin accounting is retained.

DEV241directionaljoint61.168023,+2.093198[1.161610,3.060837]vsFoRIS,
+.148354[.017192,.310205]vsRCG,+.023861[-.029879,.102806]vsRCG-value-only.
[DEVreport](research_20261005/pipeline_verified/directional241_v1/report.json),
[1200report](research_20261005/pipeline_verified/directional1200_dev_v1/report.json),
[1200reconstruction](research_20261005/pipeline_verified/directional1200_dev_v1/verification_receipt.json),
[actualsourcebinding](research_20261005/pipeline_verified/directional1200_dev_v1/actual_source_verification.json).


### 2026-10-05: strongest pixel,greedy andsingle-family controls onDEV241

SameCOCO-20iDEV241,seed0,1024,79classes,239photo-connected groups. All previous six directional
output arms reproduced bit-identically. Directionaljoint61.168023;unrestrictedpixel61.108318;
p1familygreedy61.147027;best p1single-family61.166854. Primaryvspixel+.059706[.009841,.154251],
vs p1greedy+.020996[-.033174,.101153],vssingle+.001170[-.067942,.084219]. These additional
controls share the same probability field,complete producers and5percentbudget. The exact pixel
solver weakly dominates every contained structured output on the expected-count objective,checked
in inference. Its true score is lower: family restrictions may regularize this imperfect value field.
Joint superiority to the simplest single-family selector remains unresolved; no unconditional sparsity
or true-IoU optimizer claim follows. All rules remain development on a repeatedly read cohort.

This fixed control expansion is queued/running on retained1200 with automatic subsequent CPUscore,
no extra encoder or parameter fitting. Original predictions remain retained. All arm scores were
independently reconstructed within4.27e-14points. Cached241selection122.94s,complete producer runtime excluded.
[Fullreport](research_20261005/pipeline_verified/directional_controls241_v1/report.json),
[original mask parity](research_20261005/pipeline_verified/directional_controls241_v1/original_arm_parity.json),
[reconstruction](research_20261005/pipeline_verified/directional_controls241_v1/verification_receipt.json),
[1200frozen queue](../../launch/directional_controls1200_dev_v1/plan.json).


### 2026-10-05: complete1200strong controls and larger1800fixed rule

Frozen directional predictor and all previous baseline/control scores reproduce on retained1200,
seed0,1024,80classes,300/fold,1046photo-connected groups. Additional controls:pixel63.718897,
p1greedy63.761862,single-family63.797241. Primary63.760280vspixel+.041384[.003448,.085630],
vsp1greedy-.001581[-.039024,.036369],vssingle-.036960[-.080854,.006758]. Single-familyvsMEAN
+.215649[-.043271,.508077]. Family constraints outperform unrestricted pixel optimization under
this imperfect value field, but neither joint selection nor the strongest single-family output
establishes the full objective. Query GT was opened only after every output was sealed.
[1200report](research_20261005/pipeline_verified/directional_controls1200_dev_v1/report.json),
[reconstruction](research_20261005/pipeline_verified/directional_controls1200_dev_v1/verification_receipt.json).

Predeclared del[p,w0,t0] expanded unchanged to1800,450/fold,80classes,1507photo-connected groups,
largest16;public benchmark reuse,seed0,1024. FoRIS60.484167,RCG62.287662,MEAN62.418279,
Astra61.922673,frozen delete-p62.628810,same-countdeletion62.529434. PrimaryvsFoRIS
+2.144643[1.221813,2.958440],vsRCG+.341148[-.465580,1.085701],vsMEAN+.210531[-.596419,.993175],
vssame-count+.099376[-.092148,.321754]. FoldgainsvsFoRIS:2.634257,1.938934,1.842898,2.162484;
900up/875down/25tie. Advantage over strong controls is still unresolved at this larger sample.
Every previously verified1200I/U is unchanged; fullRCG/MEAN/Astra masks independently match GT
resized by the same nearest-neighbour rule. No encoder or feature recreation/deletion.
Native/delete-p provenance remains count-only in this readout; missing full-mask/raw-DINO edits
are explicitly absent. All1800draws are retained. Independentclass-summed reconstruction error
<=3.56e-14points. [1800report](research_20261005/pipeline_verified/frozen_public1800_v1/report.json),
[source receipts](research_20261005/pipeline_verified/frozen_public1800_v1/receipt.json).

### 2026-10-05: complete released INSID3 logic on commonDEV241

ReleasedINSID3 revision0c165a10cf52ab91f335883d06260de86854adbe is unchanged. The adapter uses the
same local timm DINOv3-L weights as FoRIS;pairedBF16reference/query extraction, nativeFP32black-image
basis with500components,tau.6,merge.2,1024rendering. The optional640CRF anddefaultbilinear outputs
are separately named. Hub numerical parity and identity with the published score are not asserted.
COCO-20iDEV241,seed0,79classes,239connected-photo groups;all predictions sealed beforeCPUscore.

Bilinear54.408042vsFoRIS59.074825:-4.666783[-7.005642,-1.111330];640CRF55.006020vsFoRIS
-4.068805[-6.395323,-.234831]. CRFvsbilinear+.597979[.321421,1.106677]. The earlier cached
approximation54.469550 is retained as a separate arm. No method silently substituted.
CRFfoldgainsvsFoRIS:-4.687947,-6.225261,-.008859,-5.420749;81up/158down/2tie.
Raw-DINOrelativeedits:bilinearaddTP3,619,612/addFP2,629,379/deleteTP2,806,222/deleteFP12,356,987;
CRFaddTP3,672,517/addFP2,441,148/deleteTP3,002,669/deleteFP12,667,482.
FullGPUinference219.86s,including7.03ssetup;peakCUDA1,712,654,848bytes,CPU1. v1failed before
encoding because a minimal snapshot omitted the existing encoder adapter;v2included it unchanged.
No download/install or feature deletion. Independentlocalclass-summed reconstruction andmetadata
seal checks passed. [Report](research_20261005/pipeline_verified/insid3_complete241_v2/report.json),
[protocol](research_20261005/pipeline_verified/insid3_complete241_v2/protocol.json),
[verification](research_20261005/pipeline_verified/insid3_complete241_v2/verification_receipt.json).


### 2026-10-05: complete frozen B plus conditional A composition on1200

Newconstruction after previously inspected1200/1800;development only. Retain exact del[p,w0,t0]
anditsCbase as B. A families can restore deletedRCG pixels or add FoRIS/MEANextent,selected by p1
expected I/U after B. Fixed5percentRCG-area budget andquarter/half/full frontiers. Six domain families;
no extra encoder/querylabels. Compare complete B,blind A+B,fixedunion/restore/extent quotas andthe
unrestricted same-field addition optimizer. FrozenB andsame-count masks reproduce all1200existing
I/U exactly. All new masks sealed before CPUscore;full B masks now supply previously missing edits.

COCO-20i1-shot,seed0,1024;1200/300perfold,80classes,1046connected-photo groups. FoRIS61.612802,
RCG63.598020,MEAN63.581592,B63.958235,conditionalA+B64.119318. Controls:blind64.047304,
unionquota64.087156,restorequota64.004858,extentquota63.935109,pixel64.116684.
PrimaryvsFoRIS+2.506516[1.348066,3.435200],vsRCG+.521298[-.465123,1.318251],
vsMEAN+.537726[-.422437,1.389626],vsB+.161083[.017346,.276713]. Vsunionquota+.032162
[-.017208,.127733],vsblind+.072014[-.040877,.165543],vspixel+.002634[-.001900,.007321].
The complete composition improves its retained deletion component, but superiority to the strongest
simple controls andstable>=2points remain unestablished. Do not promote a tied control as a new method.

FoldgainsvsFoRIS:3.314624,1.261556,2.142025,3.307860;632up/552down/16tie.
FinalRCGrelativeedits:addTP664,326/addFP626,205/deleteTP3,997,175/deleteFP8,349,971.
FullB:addTP0/addFP0/deleteTP5,073,086/deleteFP9,843,565. ConditionalA therefore restores
1,075,911true pixels and1,493,594false pixels previously deleted byB, while adding664,326true
and626,205false pixels outsideRCG. Restoration andextent are explicitly distinct operations.
Blindcomposition:addTP989,716/addFP1,184,757;itsBdeletions are unchanged.
Cached1200selection152.41s,peakCUDA283,252,736bytes;complete producer runtime excluded.
Independentlocalclass-summed reconstruction error<=2.85e-14points;protocol/choices matchseal.
[Report](research_20261005/pipeline_verified/composed1200_dev_v1/report.json),
[per-episode I/U](research_20261005/pipeline_verified/composed1200_dev_v1/episodes.jsonl),
[verification](research_20261005/pipeline_verified/composed1200_dev_v1/verification_receipt.json).

The unchanged predictor is frozen for live publicblocks4/5,chosen for available feature lifetimes,
not scores. Combinedreadout will use blocks0/1/4/5 for2400,600/fold;also report the new1200alone.
Existing1200features are protected;future feature arrays are read fromtheprovider inRAM,with no
feature deletion/recreation. Compact source packets andsealed final masks are retained. Parent
mask/field seals andoriginal deletion/count parity are required before accepting the new result.
This is benchmark reuse with photo overlap,not photograph-isolated confirmation or a full4000SOTAclaim.


### 2026-10-05: complete composition mapped to common raw-DINO DEV241

All241class/support/query identities are matched to the sealed1200prediction set; no encoder rerun,
feature changes or per-query GT routing. Masks assembled andsealed before separateCPUscore. The
original native/RCG andpublic replay rows are separately retained andbit-identical onall241.
COCO-20i,seed0,1024,79classes,239connected-photo groups;repeatedly inspecteddevelopment.
Raw l24 nearest-reference-token42.903888,FoRIS59.074825,RCG61.019669,MEAN60.680117,
fullB63.122337,composition62.936227,blind63.050336,pixel62.940648,unionquota62.769632.

CompositionvsFoRIS+3.861402[1.609918,5.162354],vsRCG+1.916558[.037169,2.840281],
vsMEAN+2.256110[.134353,3.184940],vsfullB-.186110[-.480213,.156544],vsblind-.114108
[-.350990,.084153],vspixel-.004421[-.013464,.000363]. The sign ofaddition benefit differs from
full1200's+.161083overB;it is not evidence ofstable superiority to strong controls.
FoldgainsvsFoRIS:4.633201,2.311919,4.856930,3.632093;136up/102down/3tie.
Commonraw-origin edits:addTP4,104,841/addFP2,219,540/deleteTP1,727,868/deleteFP13,125,956.
Original/replaymask mismatch0pixels. Independentclass-summed reconstructionerror<=3.56e-14.
[Report](research_20261005/pipeline_verified/composed_dev241_v1/report.json),
[provenance](research_20261005/pipeline_verified/composed_dev241_v1/protocol.json),
[verification](research_20261005/pipeline_verified/composed_dev241_v1/verification_receipt.json).

### 2026-10-05: post-seal semantic errors andconditional addition roles

GTdiagnostics only;none ofthese quantities enters the frozen2400prediction rule. Relative torawDINO,
the composition removes4,375,137pixels belonging tootherannotatedclasses,638,563backgroundpixels
within16pixels oftarget GT,and8,112,256far-backgroundpixels. It wrongly removes1,727,868targetpixels
andadds2,219,540falsepixels. True additions:165,570pixels inpreviouslymissedGTregions and3,939,271
inregions withprioroverlap. It recovers>=90percentof10of87previouslymissed8-connectedGTregions.
Relative toRCG:only377trueadditionpixels lie inpreviouslymissedregions,156,222completeexisting
regions;zeroof112missedregionsreach90percentcoverage. This construction primarily changesextent
andfalseforeground; it doesnot yet establish whole missed-target recovery.

Relative tothecompleteBmask,restoration contributes-.348848class-mIoUpoints,extent+.161303,
andthefractional interaction+.001434,yielding-.186110onDEV241. Worstrestorationclassgains:
class12-13.523871,class58-9.049205,class26-4.447630. Do nottreatpooledadditionpurity asclass-macrovalue.

OriginalqueryBGbank:101,926tokens,7,598.168targettoken-equivalents(7.45percent),7,453target-dominated.
Reference-NN BGfilter:97,890tokens,5,266.359targetequivalents(5.38percent),5,112target-dominated.
Unfilteredsame-sizebank:7,551.695targetequivalents(7.71percent),7,408target-dominated. BetterGTpurity
alone doesnotprove useful negative discrimination. ReferenceNNscores are fromdebiased cachedfeatures,
distinctfromtherawaccountingorigin. These diagnostics motivate anewDEV-onlycompletecomparison of
reference-NN-cleanedBplusextent-onlyA,withoriginalB,same-sizebanks,quota andpixelcontrols.
Touchinginstances canmergeinsemantic8-connectedregions; these are region/proximity definitions,
notverifiedinstanceidentities. [Diagnostics](research_20261005/pipeline_verified/composed_diagnostic241_v1/report.json).


### 2026-10-05: frozen composition2400 andseparate new1200readout

COCO-20i,seed0,1024,80classes,600draws/fold;sourceblocks0/1/4/5,2400draws,1888connected-photo
groups. Blocks4/5were chosen for live feature availability andthepredictor was frozen before their
GTreadout. This is benchmark reusewithphoto overlap,notphotograph-isolated confirmation orfull4000.
Parent mask/field seals,all2400original deletion/same-count I/U andRCG/MEAN/Astra counts passparity.
No encoder rerun orfeaturedeletion/recreation. Compact packets/final masks remain retained.

FoRIS61.635100,RCG63.140164,MEAN63.277195,B63.169303,composition63.426455;
blind63.307392,unionquota63.413949,pixel63.423975. PrimaryvsFoRIS+1.791355[1.098103,2.489658],
vsRCG+.286291[-.347554,.878681],vsMEAN+.149259[-.493105,.752387],vsB+.257152[.143756,.330256],
vsblind+.119063[.039823,.185231],vsunionquota+.012506[-.026744,.069725],vspixel+.002480
[.000170,.005456]. The tiny pixel difference doesnot establishthecompletegoal;stable>=2overFoRIS
andstrongRCG/MEANsuperiority remainunmet. Do not keep extending this construction merelybecause
it improved itsretainedBcomponent. Foldgains:1.861547,1.171776,1.970457,2.161640;
1252up/1121down/27tie. RCGrelativeedits:addTP1,483,657/addFP1,251,686/deleteTP8,234,918/deleteFP15,781,100.

New1200alone:FoRIS61.016582,RCG62.011720,MEAN62.302334,B61.699516,composition62.032075.
VsFoRIS+1.015494[.141979,1.965818],vsRCG+.020356[-.784501,.750206],vsMEAN-.270258
[-1.093285,.492537]. The original1200's+2.506516FoRISgain didnot hold onthe newdraws.
[2400report](research_20261005/pipeline_verified/composed2400_v1/report.json),
[per-draw counts](research_20261005/pipeline_verified/composed2400_v1/episodes.jsonl),
[reconstruction](research_20261005/pipeline_verified/composed2400_v1/verification_receipt.json).

### 2026-10-05: complete BG-cleaning andextent-only construction didnot improve

NewDEV241construction,seed0,1024,79classes,239photo groups;no additional encoder. OriginalB63.122337,
NN-BG-cleanB62.908069,same-sizeunfilteredB63.135103. Cleaningversussame-sizeB-.227034
[-.514420,.203817]. CleanerbankGTpurity isnot a sufficient discriminator ofuseful negative evidence.
PrimarycleanB+conditionalextent62.807903;vsB-.314434[-.628340,.187806],vsoriginalcomposition
-.128324[-.402050,.303326],vssame-sizecomposition63.017747:-.209844[-.466653,.194616].
OldB+extent63.006353,cleanBpixel62.809963,cleanBquota62.552528. Primaryvsquota+.255375
[.129356,.444972],butno improvement overthe retainedstrongmethod. No control relabeled as a win.
Allmaskssealed beforeGT;frozen2400method unchanged. [Report](research_20261005/pipeline_verified/background_clean_dev241_v1/report.json),
[verification](research_20261005/pipeline_verified/background_clean_dev241_v1/verification_receipt.json).

Next boundedDEVcomparison retainseasyandhardNN-FG-likebackground pools separately,usesa reference-FG
positiveguard,andallowshigh-margin additionsoutsideprior maskunions. Its confidence marginsare not
claimedascalibrated probabilities. The query-onlysplitB,unsplitguardB,oldB+newA,newB+oldA controls
isolate the newlinks; no queryGToradditionalencoder. Thisisanewconstruction,notanupdated2400rule.


### 2026-10-05: role prototypes andwhole-query addition failed

CompleteDEV241,seed0,1024,79classes,239photo groups;no newencoder. Reference-NN votespartitionthe
originalBGbank intoeasy/hardpools;maxBGscore,andmaxreference/query-seedFGscore guard. Additions can
searchoutsideprior unions,withprototypegap>.1 andreferenceNNpositive,rank minimum margins,cap
5percentRCGarea. Cosineconfidence rules,notcalibratedmembership probabilities. Allmaskssealed
beforeGT;original2400methodunchanged;sourcefeatures/packets andsealedRCGfields verifyunchanged.

Primary62.306604vsoriginalB63.122337:-.815733[-1.258972,-.289823];vsFoRIS+3.231779
[.814910,4.734402],vsRCG+1.286934[-.779090,2.479245]. ComponentB62.400503,query-onlysplitB
62.667190,unsplitreferenceguardB62.883290. OldB+newA63.035282,newB+oldA62.561005.
PrimaryvsnewB+oldA-.254401[-.400551,-.126254],vsoldB+newA-.728678[-1.151337,-.196593].
Neither max-prototype splitting/referenceguard norwhole-queryhigh-margin addition improved this
construction. Stopthis role/margin grid;negativeevidence doesnotruleoutotherconditionaledit mechanisms.
Independentclass-summed reconstructionandmetadata seals passed. [Result](research_20261005/pipeline_verified/role_prototypes_dev241_v1/report.json),
[verification](research_20261005/pipeline_verified/role_prototypes_dev241_v1/verification_receipt.json).

The existingpublic4000chain isfinishing itslast400draws. ACPU-onlyqueue willscore completeRCG/MEAN/Astra
masks andpreviouslyfrozen delete-p/same-count statistics onall4000draws,includingthenaturalrepeat.
Thiscompletesthepairedbaseline/retainedcomponent evidence boundary;itdoesnotpromoteafailedcandidate
orclaimfull4000publicSOTAfor the2400composition. No encoder orfeaturedeletion/recreation.


### 2026-10-05: frozen deletion completed all4000 benchmark draws; target failed

COCO2014 val, four folds1000each, seed0,1024,4000sampled draws with3999unique identities;
benchmark reuse including exposed cohorts,not fresh confirmation. CompleteRCG/MEAN/Astra masks
independently reconstructed against GT; native and fixed deletion use sealed per-draw I/U.
No candidate encoder or feature restoration/deletion. FoRIS60.931741,RCG62.333671,MEAN62.512972,
Astra61.853251,delete-p62.328852,same-count62.351948. Delete-p versusFoRIS+1.397111
[.783637,1.970674],versusRCG-.004819[-.568074,.523499],versusMEAN-.184120[-.751011,.351018],
versussame-count-.023095[-.152283,.117530]. Stable>=2andstrong-control superiority remain unmet.
[Report](research_20261005/pipeline_verified/frozen_public4000_v1/report.json),
[receipt](research_20261005/pipeline_verified/frozen_public4000_v1/receipt.json).
Official serial seed0 sampling replay matches all4000identities with existing fold metadata;
canonical metadata content and official encoder/rendering numerical parity remain unverified.

### 2026-10-05: spatial BG jackknife adds no established advantage

FullDEV241,seed0,1024,79classes,239connected-photo groups. Four-by-four token spatial BGblocks,
retain an original deletion only if every leave-block-out pool agrees. Existing extentA unchanged;
originalBmask replay exact on all241cases; masks sealed before GT. Primary63.248866,Bonly63.088611,
same-save-count63.250375,fixedslack63.256411,originalB+sameextent63.283640.
PrimaryvsFoRIS+4.174040[1.793722,5.642150],vsRCG+2.229196[.185919,3.285708],
vsMEAN+2.568749[.272781,3.633492],butvsoriginalB+sameextent-.034775[-.237678,.183151],
vssamecount-.001509[-.020362,.003954],vsslack-.007546[-.110097,.166609].
This does not establish the new mechanism. Stop this stability branch; do not rename its stronger
control as the proposed method. [Report](research_20261005/pipeline_verified/spatial_jackknife_dev241_v1/report.json).

The user corrected the overly FoRIS-dependent direction. A new complete raw-feature-onlyDEV241
branch uses reference role-balanced similarity and query-neighborhood propagation, with explicit
centering/no-centering and graph/unary controls; FoRIS/RCG/MEAN are evaluation rows only.
It is queued with CPU prefetch/writing and batched CUDA; no result or novelty claim yet.


### 2026-10-05: RCG mechanism attribution replaces speculative component search

User authorized parallel subagents; root owns monitoring/dispatch. Historical600,seed0,1024,
80classes;reused training-pool DEV,nofreshconfirmation. Fixed2x2readout:pre60.944388,
rerank60.939394,smooth63.582277,both64.237696;completeFoRIS61.628013.
Rerankvs pre-.004994[-.584939,.279110];smoothvs pre+2.637889[1.792333,3.268409];
rerankwithgraph+.655420[.056436,1.061844];interaction+.660413[.196835,1.183400].
Bothvsnative+2.609683[1.392159,3.229520]. All600returnedFP32RCGfields/storedmasks exact;
allCGstatus/residual checks pass;fixedthreearms sealedbeforequeryGT. Per-drawI/U retained remotely.
Thegraph accountsformostgaininthiscohort,withconditionalreferencevalue;thisisnot4000stageablation.
[Verified report](research_20261005/pipeline_verified/rcg_ablation_verified600_v1/report.json),
[audit](research_20261005/pipeline_verified/rcg_ablation_audit_v1/audit.md).
MEAN4000=62.512972vsRCG62.333671:+.179301[-.003145,.375652];RCG superioritytoMEANisnotestablished.

All4000publicdraws,80classes,2742connectedphoto groups,seed0,1024;benchmarkreuse.
Native60.931741,pre60.410620,RCG62.333671. RCGvsnative+1.401930[1.094,1.686],
vs pre+1.923051[1.615,2.222];nativevs pre+.521121[.456,.605]. Fullnative/RCGI/Umatchtheprevious
independentannotationledger oneverydraw;16P/N/R/GTstatesexhausteverypixelandexactlyreconstructI/U.
FinaleditsrelativecompleteFoRIS:addTP14,728,544/addFP17,576,193/deleteTP12,697,324/deleteFP28,099,342.
These areGTcounts,notprobabilities;pooledpuritydoesnotdetermineclass-macrogain.
[Native-state report](research_20261005/pipeline_verified/rcg_native_states4000_v1/report.json).

Native-quality>=.9bin:RCGgain-.403[-.691,-.252]. QualityisGTdiagnosticonly,notaninferencegate.
Earlier3000gain+1.661[1.301,2.000];later1000+.618[.142,1.173];late-minus-early-1.043
[-1.597,-.342]. Quality/classcompositionaloneandceilingdonotexplainthisshrinkage;fixedbinand
sharedclass-qualitysupportstandardizationsareobservationalandcannotestablishcausality.
Exactfour-wayclassmacroaccountingusingJ_nativeandU_RCGattributeschangeasaddTP-.101[-.367,.245],
wrongdeleteTP-.478[-.894,-.049],correctdeleteFP-.721[-1.140,-.239],newFP+.256[.015,.461].
GTdistance>16component-.812[-1.326,-.163],about78percentofpointshrinkage. NativeCRFgainchange
only+.053[-.088,.226]. Thisaccountingdoesnotseparateguide/graph/renderercausesonlatebatches.
[Quality report](research_20261005/pipeline_verified/rcg_quality4000_v1/report.md),
[exactterm report](research_20261005/pipeline_verified/rcg_quality4000_v1/state_report.md).

RemainingRCGerrors:122.143Mbody54.98percent,57.096Mwithin16px25.70percent,42.934Muntouched/stray
semanticregions19.32percent. Whole-missedGTregionFNmass9.361Mvsnative5.318M;within8falsemass
19.957Mvsnative17.756M. GTbodyoracle+16.0976,within8+6.2154,8to16+2.3372,objectproxy+5.9258;
twoGTboundarybucketstogether+8.77486,notthesumofseparateratiogains. TheseareGTcapacitydiagnostics,
notmethodresults. ObjectbinsaretotalGTclassareaand8connectedsemanticproxies,notinstancecounts.
Theclaimthaterrorsaredominantlyboundaryisnot supported.
[Anatomy](research_20261005/pipeline_verified/rcg_anatomy4000_v1/report.md),
[independent audit](research_20261005/pipeline_verified/rcg_anatomy_audit_v1/audit.md).

Decision:holdrawgraph/newcomponentvariants;completeexistingboundaryrunwithoutusingitsscoreasnext
methodselection. Fixed80feature-separabilitydiagnosticisqueuedaftercurrentGPUjob,withallfinefeatures
retained;CPUtests existingreferencecuefeasibilityontheobservedremainingerrors. Neitherdiagnostic
substitutestheunmetcomplete-method>=2/strong-controlobjective.


### 2026-10-05: fixed80boundary diagnostic doesnot establish new fine information

Historical600subset,sampledRandomState0beforeGT,20/fold,80episodes. Sourceunaryfeaturesremain
FoRIS-debiasedcachedDINOspace;singleexistingreference,r/cov only. Fourreflected±4queryshifts,
FP16finegridsretained~2.5GiB. All80features/fieldssealedbeforeCPUopenedqueryGT. GPU110.39s,
CPUabout12s;setupincluded,no pricingclaim. Fixedpure8pxcoverage<=.1/>=.9,meansame-sideGTdistance
<=16;80eligibleepisodes,1717eligible16pxparents/327680total,3078FG-BGpairs.
FineNNboundaryAUROC.8762;bilinearcoarsefeatureNN.8568,bilinearcoarseNNmargin.8752,RGB.5564.
Finevsbilinearmargin+.0010[-.0041,.0056];parentrankfine.8588vsbilinearmargin.9179,
-.0591[-.0859,-.0335]. Finevsinterpolatedfeatures+.0194[.0136,.0259],butthatweakercontrol
cannotestablishnewinformationbeyondthestrongerinterpolatedmargin. RCGcorrect-sidevalueexistsin
51215/53989pureboundarycells;availabilityisGTdiagnosticcapacity,notlegalinferenceproof.
[Report](research_20261005/pipeline_verified/boundary_features80_v1/report.md).

Completedpeerfull600subtokenconstruction:nestedfine64.625380vsRCG64.237696,+.387683
[.330002,.569782];vsRGBnested64.390177,+.235202[.143835,.423549];vsRCG+CRF64.448578,
+.176802[.007463,.346140]. ThisisreusedDEVwithGTfitting-foldselection,noseparatefreshconfirmation
orcompletepublicSOTA;sourceprogramopensGTbeforeforwardthoughvisiblepredictiondoesnotuseit,
anddoesnotsealsource/predictions. Mainknownmissingcontrolisthesamereadoutwithinterpolatedcoarse
featuresandzeronewencoderforwards. Neitherthisscorealone norGToracleceilingschooseanextmethod.
[Completedsource result](research_20261005/pipeline_verified/subtoken600_v1/report.json).


### 2026-10-05: actual fine information matters in the matched query readout

Samehistorical600,seed0,1024,80classes/564connectedphotogroups. Retainoriginalper-fold
selectedsigma1.25,tau.07(folds0/3)/.15(folds1/2);no newselection/grid. Replacefour-shiftfinefeatures
withunit-bilinearinterpolatedcoarsefeaturesonly;identical5x5kernel/neighborhoodandCUDAfinalizer.
All600maskssealbeforeGT;allpriornative/RCG/selectedfineorderedI/Uexact. Completecoarseguide64.273121,
fine64.625380;finevscoarse+.352259[.325705,.544086]. CPU6readers/2writers+GPUbatch4:
19.92sinference/4.16sGPUcompute/10.87sCPUscore. Coarseproduceruseszeroencoderforwards/images.
Thisisamatchinginformation-removalcontrol,notanoptimizedcoarsegridorindependentconfirmation;
sourcefinepolicywasGTfitting-foldselected. Thecompletecontrast supportsvalueinthetestedquery
readoutdespitenoimprovedreferenceNNboundarydiscrimination;neitherimplieseffectivehigherresolution.
Legacysealhasnomapofinputhashes,soinitialv1failedbeforeprediction;v2recordsnewinputhashesand
checksalloriginalcountsat score. Failedoutputretained,noassetsdeleted.
[Result](research_20261005/pipeline_verified/subtoken_coarse_control600_v2/report.md).

### 2026-10-05: reference nearest-neighbor margin fails the residual-error feasibility test

All4000publicbenchmarkdraws/seed0/1024,reusedDEV;GTdiagnostic,noencoder. ExistingNNsignedmargin
fg_max-bg_max,FoRISminmaxscoreandRCGfield evaluatedonfixedGTpure16pxtokens. Whole-missedGTsemantic
regionsversusstrayRCGregions:NNmeanepisodeAUROC.253204[.215718,.293471],223eligibleepisodes/209groups.
Deep>16pxFNversusFP:.232535[.215564,.249790],1088episodes/947groups;allfourfoldsbelow.5.
NN>0pooledrecall/FPR23.30/72.44percentand32.60/73.43percent. Wholemiss1720regions,690havepure
representativetokens,421positiveNN;stray3894,3680havepuretokens,2931positiveNN. Absenceofpuretokens
isnotabsenceofsemanticinformation;1030wholemissregionslackpuretokens,including679under256pixels.
RCGfieldpoorAUROCispartlytautologicalbecauseRCGerrorsdefinethecohort. DonotflipNNbasedonGTerror
selection;negativeevidenceonlylimitsthiscue/conditioning,notallfeatures. All4000I/U,regioncountsand
2000photo-bootstrapcomputations independentlyverified. CPU4fullrun170.28s,noGPU/newcomponent.
[Interpretation](research_20261005/pipeline_verified/rcg_remaining_cues4000_v1/interpretation.md),
[validation](research_20261005/pipeline_verified/rcg_remaining_cues4000_v1/validation.json).

Nextfixedcomparisonexpandstheexistingfinequeryreadoutto1200withpreviouslytestedstrongλ64control
andfineλ64composition;thisisnotnewcueinventingorparametersearch. All1200remainDEV,andcomplete
>=2overFoRIS/strong-controlsuperiority remainunmet. Newfinearraysstreamwithoutfullcache;existing
80finegridsandoriginal1200cachedfeaturesprotected. RCGsourcehashes/configpathsareunchangedacross
public7blocks;missingruntime/weight/projectioncontentreceipts meanwecannotfullyexclude numericdrift.


### 2026-10-05: frozen fine/query-graph comparison1200 completed; extend4000

COCO2014val/publicblocks0/1,seed0,300/fold,1200draws,80classes/1046connectedphotogroups,1024;
previouslyexposedDEV,notconfirmation. Singlecompleteclassmaskreference/samefrozenDINO,extra4query
shiftswithfixed600foldsigma/tau;CPUlambda16replayandlambda64preparedwithsamegraph/equations.
All1200ownmaskssealbeforeGT;native/RCG/MEANI/Uexactpriormatch. FoRIS61.612802,RCG63.598020,
MEAN63.581592,RCG64control63.690334,fine16control64.015315,primaryfine64=64.051866.
PrimaryvsFoRIS+2.439064[1.581170,3.237973];vsRCG64+.361532[.312316,.461490];
vsRCG+.453846[-.059653,.905474],MEAN+.470274[-.104917,1.001042],fine16+.036551
[-.520387,.457837]. Strong-control/completegoal remainsunresolved;point>=2aloneisnotsuccess.
Foldprimary-vsFoRIS2.930471,1.471085,1.800627,3.554074;665up/520down/15tie.
CPUprepare179.21s,GPUinfer1199.51s(1145.63compute),peak1.559GB,CPUscore32.22s;
cachedpreparation+4shiftruntime isnotanend-to-endproducerbenchmark.
[Report](research_20261005/pipeline_verified/frozen_subtoken1200_v1/report.md),
[independent reconstruction](research_20261005/pipeline_verified/frozen_subtoken1200_verification_v1/verification.md).

Exactstagegain2.439064=1.985218(RCG-native)+.092315(RCG64-RCG)+.361532(fine64-RCG64).
Fine64relativecoarse64netTP+591286/netFP+95195;classmacro+.365483(TP)-.003951(FP).
Fine64relativenativeclassmacro netTPterm-1.316673/netFPreductionterm+3.755738;
thisisexactaccounting,notindependentcausaleffects. Sourcefour-wayRCGeditcountsverifiedbyI/U
identitiesonly;fullnativecountsrequiremaskrecountinlaterledger. Raworigin1200absent;keepDEV241
rawaccountingseparate,notmanufactured.

Two-pairsourceparitypassedFP16q/r/gate/native/score/covandlambda16field/maskwithoutGT.
Full4000launchedfrozen,inclusiveofprior1200benchmarkreuse;reuseall1200sixarmoutputsandonly
encode2800newpairsinboundedRAM. No newfullfeaturecache,noexistingfeaturedeletion. CPU6CGand
2writersparallelwithGPU,~2.44s/newpairinitialthroughput/GPU100percent. FullsourceForeISfinish
currently retainedfornativeparity;futurePart1-onlyoptimizationmustpassownproofbeforeuse.
All4000ownmaskssealbeforeCPUscore;native/RCG/MEAN all4000andoldsixarms1200mustmatchI/U.
[Two-pair receipt](research_20261005/pipeline_verified/subtoken4000_smoke_v1/parity.json).
Full4000candidateeffectstillunknown;publishedSOTAandindependentconfirmation arenotasserted.

### 2026-10-05: original DEV241 raw-origin accounting and fixed region-mean cue

All original DEV241 draws, seed0,1024,79 classes/239 connected-photo groups; existing frozen1200
outputs mapped by class/reference/query identity without new inference or tuning. Actual raw l24
NN42.903888, canonical complete native59.074825,RCG61.019669,MEAN60.680117,coarse64=60.128680,
fine16=61.500661, primary fine64=60.515912, released complete INSID3 CRF55.006020.
Fine64-native +1.441087[-.754819,3.052193]; fine64-coarse64 +.387231[.095634,.753852];
RCG/MEAN/fine16 superiority unresolved. All241 identities, pixel truth, six-arm I/U and raw/native
four-edit closure verified; original GPU-native25-pixel drift remains a separately named control.
Raw-relative primary edits: addTP4,127,069/addFP2,503,298/deleteTP1,658,440/deleteFP12,917,230.
Native-relative: addTP902,363/addFP837,350/deleteTP994,107/deleteFP2,862,840,netTP-91,744.
Assembly3.536s+CPUscore6.029s; independently reproduced all scores, folds and2000-photo CIs.
[Accounting report](research_20261005/pipeline_verified/frozen_fine_raw_dev241_v1/interpretation.md).

Fixed mean-prototype feasibility diagnostic, same DEV241, CPU2/33.77s, privileged GT regions:
114 eligible whole-missed components and83 eligible stray fine64 components; only18 episodes have
both types. Raw AUROC .3944[.1680,.6072], cached projected prototype .2556[.0733,.4556], stored NN
mean .3272[.1176,.5556]. All four raw fold means are below .5; raw-minus-projected +.1389
[.0278,.2813] does not establish useful separation against a failing control. No reference-FG
fallback;239 caches were projected and2 were not; all18 paired cases were projected. FP32 raw vs
FP16 cache is not a pure projection ablation. All source hashes,241six-arm I/U and independent
AUROC/2000-photo CI reconstructions passed. V1 incorrect all-debiased assumption failed before a
result; v2 records actual source flags, retaining failed assets and reusable features.
Decision: stop this fixed region-mean recovery branch; do not infer that all DINO semantic cues fail.
[Diagnostic](research_20261005/pipeline_verified/region_prototypes_raw241_v2/interpretation.md),
[verification](research_20261005/pipeline_verified/region_prototypes_raw241_v2/verification.json).

### 2026-10-05: fixed object-crop global CLS construction remains unresolved

Same197 eligible regions from exposed DEV241;85 encoded episodes,367 FP32 views/114 batch<=4
encoder calls. Primary18paired episodes/40missed+31stray regions remain unchanged. Actual timm
Eva final-LayerNorm CLS index0, same frozen weights, no autocast/TF32, fixed FG-minus-BG cosine.
Reference FG is cropped and outside-FG RGB erased; negative reference erases known FG. Query RGB
crops use privileged GT/prediction region bboxes, square with10percent context; no query erasure.
This tests the entire crop/CLS construction, not token type alone or legal object proposals.
CLS AUROC .591975[.388889,.787783]; raw region mean .394444[.167967,.607227]; cached projected
mean .255556[.073286,.455556]. CLS-raw +.197531[-.071429,.470370]; CLS-projected +.336420
[.124980,.562500]. All four CLS fold means exceed .5, but the main CI crosses chance; unresolved.
Full197-region pooledCLS AUC .449059 and area-weighted .694408 are secondary descriptions,
not replacements for the primary statistic. No fitted weights, sign reversal or sweep.
Encode153.29s/peak2,080,604,672bytes; compact descriptors retained and sealed before score.
[Report](research_20261005/pipeline_verified/object_cls_dev241_v1/report.md).

Resource sampling: short CLS co-run raised GPU mean71.8->95.8percent but main4000 new-pair time
rose2.448->4.002s; main throughput fell38.8percent. Memory stayed<=6085/32760MiB. Time relation
supports GPU competition, not a controlled same-sample causal timing claim. Fixed CLS run ended,
main resumed~2.39s/newpair. Do not call the utilization rise a free improvement; future long GPU
work is queued serially while CPU analysis runs in parallel.
[Resource audit](research_20261005/pipeline_verified/stream4000_resource_audit_v1/report.md).

CPU-only crop-context audit (1.82s) kept all197 crops and verified original/working GT parity.
Of83 stray crops,14 contain target GT outside the selected component (2 within the original bbox,
12 introduced by square/context expansion); the primary31 negatives include4 such crops.
At the fixed CLS>0 rule, primary false positives are3/4 GT-containing and17/27 GT-empty crops:
85percent of primary false positives occur without any target GT in the crop. Context/ownership
confounding exists but does not explain most false positives. No crops, threshold, subgroup AUC
or method were selected from this diagnostic.
[Context audit](research_20261005/pipeline_verified/object_cls_dev241_v1/crop_context_audit.md).


### 2026-10-06: complete frozen fine4000 and fixed A/B transfer control

COCO2014/public4000 sampled draws, seed0,1000/fold,80classes,2742 connected-photo groups,
1024 working resolution; reused development benchmark, not independent confirmation. Same frozen
DINO and one fully masked reference; extra four query shifts for fine readout. All sampled repeats
retained. Complete native60.931741, RCG62.333671, MEAN62.512972, coarse64=62.297937,
fine16=62.709100, primary fine64=62.651617. Primary-native +1.719876[1.256073,2.113693];
primary-RCG +.317946[.039376,.562082]; primary-MEAN +.138645[-.184598,.422923];
primary-coarse64 +.353680[.321971,.396545]; primary-fine16 -.057483[-.352600,.178003].
Fold native gains1.683,1.546,1.067,2.584; last two batch gains+.430,-.646. All4000 baseline
and prior1200 six-arm I/U, prediction-before-GT seals and independent statistics passed.
This full result does not meet stable>=2 or strong-MEAN superiority.
[Full report](research_20261005/pipeline_verified/frozen_subtoken4000_scored_v1/report.json),
[independent verification](research_20261005/pipeline_verified/frozen_subtoken4000_verification_v1/verification.md).

Fixed cheapest mask-edit transfer uses R=RCG,M=MEAN,V=fine16 and
C=(M | (V & ~R)) & ~(R & ~V); no new encoder, scalar fitting or parameter sweep.
Complete4000 score62.744357: native +1.812616[1.524273,2.098950], MEAN
+.231385[.202166,.266264], fine16 +.035257[-.145128,.218718], fine64
+.092740[-.176077,.421751]. All four folds and seven batches positive relative to MEAN.
MEAN-relative addTP3,323,910/addFP3,739,944/deleteTP2,035,282/deleteFP3,487,708.
CPU4 inference2.95s/scoring28.98s; upstream producer cost excluded. All4000 original six-arm
I/U and independent score/CI checks exact. This control is an actual complete A/B composition,
not the separate continuous residual transfer and not a replacement of the original primary.
[Report](research_20261005/pipeline_verified/mean_fine_bit_transfer4000_v1/report.md).

### 2026-10-06: CLS construction1200 does not establish discrimination

Unchanged fixed diagnostic:332 encoded episodes/736 eligible regions,77 paired episodes/296
regions,1200 reused DEV draws;1400 descriptors(367 reused,1033 new), FP32 final CLS.
Primary AUROC .433959[.340099,.527200], NN .243285[.162874,.332928];
CLS-minus-NN +.190674[.090203,.300785]. The failed control comparison does not establish
useful discrimination. Crop geometry uses query GT and is not legal method inference. All
descriptor hashes and independent statistics passed. Stop this construction, retain features.
Natural-RGB reference-FG intervention retains all other views, cohort and direction:332 newviews,
88.09s, AUROC .494905[.400891,.585473], masked-to-natural +.060946[-.032513,.150309],
unresolved. No further CLS variants are queued.
[CLS1200 report](research_20261005/pipeline_verified/object_cls_dev1200_v1/report.md).

Latest user instruction prioritizes exact optimization over existing complete A*/B* families,
including the supplied fine-readout plus size-cut method. Its reported cross-fold fresh60065.44
and the separate all600-fitted frozen65.86 are distinct versions. No new semantic cue probe
was launched after that correction.


### 2026-10-06: fixed scalar fine residual transfer4000 completed

Same4000 public/reused DEV draws,seed0,1000/fold,80classes/2742 connected-photo groups,1024.
F=coarseRCG16,G=coarseMEAN16,Y=fineRCG16; fixed C128=bilinear(G,128)+Y-bilinear(F,128),
FP32 coefficient1/no clipping, CUDA1024 bilinear and cut>.5. It is not direct fine readout ofG.
Complete62.844537: native+1.912796[1.618404,2.210338], RCG+.510866[.332586,.716025],
MEAN+.331565[.303193,.372805], fine16+.135437[-.048326,.334529], fine64
+.192920[-.078619,.522409]. Fixed mask-level transfer comparison+.100180[.079995,.128147],
2652/1254/94 episode up/down/tie. All four MEAN-relative fold gains positive(.326,.363,.323,.314).
Target>=2 and fine16/fine64 superiority remain unproved. Query GT not read during inference; all
4000 predictions sealed before scoring. Reuse1200 scalar fields; remaining2800 CPU6 exact graph
reconstruction uses verified feature-only encoder producer. Existing reusable features are retained.
Original6baseline4000I/U, per-episode four-edit closure, all stored score/CI/fold/batch statistics
independently reproduced exactly in3.70s. Sampling repeats retained; benchmark reuse not confirmation.
[Report](research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.md),
[Independent reconstruction](research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/verification.json).


### 2026-10-06 UTC: exact complete-mask A*/B* library selection

Public4000 preserved draws,seed0,1024,80classes/2742connected-photo groups; all benchmark
reuse,not independent confirmation. For each allowed complete prior P0 and complete library
mask Si, Ai=Si\P0, Bi=P0\Si. Enumerate every allowed union of Ai and Bi with
C=(P0 union selectedAi) minus selectedBi. Membership-by-class integer histograms and subset/
superset zeta sums compute actual class-summed I/U for each recipe; no pooled purity surrogate.
Single producers are included exactly. Global fitting uses GT; inference uses one frozen global
Boolean recipe without per-query GT routing. Reported global CIs are conditional on selected
DEV recipes and do not adjust for search. Outer complete-mask producer count is a complexity
proxy, not atomic-component count or measured total runtime.

11-source-distinct library:11,534,336 recipes,CPU4 histogram34.54s/search3.404s.
Best63.062305: native+2.130564[1.748400,2.492689], MEAN+.549333[.298681,.767327],
fine16+.353205[.189976,.501315]. Lowest-count highest recipe3 uses
C=(conservative_delete & fine16) | (~conservative_delete & fine64). Outer-count1/2/3
Pareto scores62.744357/62.990095/63.062305. Root independent full enumeration6.025s
returns the same highest recipe/score. All finalist masks/I/U and independent selected-recipe
score/CI/fold reconstruction exact.
[Report](research_20261005/pipeline_verified/exact_family_selection_v1/public4000/report.md),
[Independent verification](research_20261005/pipeline_verified/exact_family_selection_verification_v1/public4000/verification.md).

12-source-distinct strict library adds scalar residual transfer:50,331,648 recipes,11.837s.
Global best63.064818, only+.002513 over the previous optimum despite the scalar component's
standalone MEAN-relative+.331565. New recipe C=(fine16 & scalar_graft) | (~fine16 & fine64).
Global-native+2.133077[1.772388,2.463718]; MEAN+.551846[.394417,.710822].
Three-fold recipe selection/held-fold readout with photo exclusion gives actual complete
62.964699, native+2.032958[1.657151,2.387729], MEAN+.451727[.233044,.659919],
fine16+.255599[.094337,.399785]. This is fold-held recipe evaluation, not new-data confirmation.

13-source-distinct extended library adds fixed size-cut on original fold-temperature fine16:
218,103,808 recipes,94.118s. Best63.238541, C=(conservative_delete & fine16) |
(~conservative_delete & sizecut_foldtemp). Native+2.306800[1.886799,2.768258],
MEAN+.725569[.400378,1.094699], fine16+.529441[.270524,.830262], but best complete
sizecut+.089340[-.086422,.299029] remains unresolved. Native fold gains2.483,2.297,2.064,2.384.
The six size cuts were fitted on all fresh600, which are included in these4000; this is an
explicit label-fitted extended setting, not the supplied uniformtau15 primary. Extended
train3/held1 recipe evaluation63.102199, native+2.170458[1.730749,2.628629],
sizecut-.047002[-.248330,.149315]; producer calibration already includes held-fold labels.
All finalist masks and source13 I/U independently exact, score/CI/fold differences<=7.1e-15.
[Extended report](research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v4_extended/report.md),
[Verification](research_20261005/pipeline_verified/exact_family_selection_verification_v1/public4000_v4_extended/verification.md).

Class-macro four-way edit accounting of global12: addTP+2.345785/deleteTP-2.063222/
deleteFP+3.223777/addFP-1.373264 points; global13:+2.877314/-2.034636/+3.361189/
-1.897067. Each uses native classJ and final classU and closes exactly to the measured gain;
these terms describe pixel changes, not independent causal effects.

OriginalDEV241 common rawNN remains an accounting origin: shared public13 methods plusrawO
are a declared14-method subset, not a full187-library search. FixedrawO global strict/extended
16,777,216/67,108,864 recipes reach62.773896; strictheldfold61.284097, native
+2.209272[-.023515,3.778239], fine16-.216565[-2.276171,.993915], unresolved. Its recipe
requires actual rawO on3953 public draws/34,591,588 potentially changed pixels; rawO4000 is
missing and no alternative origin is substituted.

### 2026-10-06 UTC: frozen size-cut cache replay and continuous GPU successor

Fixed allfresh600 six-level recipe applied to original fold-temperature fine16 gives full4000
63.149201: native+2.217460[1.724180,2.760573], MEAN+.636229[.248187,1.063334],
fine16+.440101[.088318,.831264], fine64+.497584[.213799,.849593]. PureCUDA finalizer
18.687s/peak135,266,816bytes,no encoder; all original>.5 masks pixel-exact and all6baseline
I/U exact. Independent score/2000-photo-bootstrap arithmetic exact. A first launcher failed
with a quoting SyntaxError before any inference; immutable failure log retained and v2 fixed.
Uniformtau15 matching2000(primary on folds1/2 only)=64.304037; not a complete4000 result.
Original crossfold60065.44 is distinct from global60065.86; exact offline fitting policy of
the crossfold report is missing, and it was not silently reconstructed by new fitting.

Original DirectMEAN1200 was automatically started by prepared supervisor56334 at peerB-to-C
GPU release; root's attempt to supersede only a waiting supervisor was rejected by the
identity/state guard once it was active. No signals sent. Preserve healthy child71246 and
original arm/kernel cache. Real28.113s/15-point co-run sample measured100percent every point
and6049..6277MiB/32760MiB; MEAN20-case interval1.518s/draw, no observed write-I/O bottleneck.
This is a measured window, not a claim of always100percent or free parallel speedup.
[Resource audit](research_20261005/pipeline_verified/co_run_resource_audit_v1/report.md).

Validated successor supervisor75010/start955212709,launch/uniform_tau15_sizecut4000_v2:
CPUwait originalMEAN1200 seal ->3case producer parity ->reuse matching2000 plus encode
missing2000 uniformtau15 ->CPU4000 exact6baseline comparison. No duplicateMEAN1200 run;
missing2000 usecached600 plus1400 validatedPart1-only prefix pairs. Keep~3.28GB FP32
cosine affinities. PeerC71178 is explicitly read-only by process identity, not adopted/owned
or signaled. Frozen primary settings and all600 label-fitting scope preserved. No reusable
feature deletion. Complete uniformtau15 primary4000 score remains pending.


### 2026-10-06 UTC: one fixed gain forecaster and new confirmation cohort

Fixed η(target|12 existing binary-mask memberships) from photo-disjoint train3 folds,
Beta(1,1) smoothing and pooled train foreground fallback; no variants/hyperparameter sweep.
Predict E[I]/E[U] and conditional edit value on held unlabeled mask frequencies before heldGT
evaluation,37.448s CPU2. Training rows2676/2674/2750/2633 with zero connected-photo overlap.
PUBLIC4000 pairwise ranking correct96.15/94.87/79.49/82.05percent; best among12 producers
plus globallyDEV-selectedC12 correctly picked3/4 folds, fourth regret.198086point. Absolute
score MAE .585659/3.938149/6.187559/.257006 points. C12-minus-graft predicted+.175811
andactual+.220281 on aggregate; fourfold signs agree, magnitude remains inaccurate.
New DirectMEAN1200 already had scored output before prediction, therefore retrospective:
vs graft predicted-.026030,actual+.003205, fourfold signs all wrong. This tiny effect is not
proof of reliable fine-gain forecasting. Labelled calibration and globallyselected candidate
exposure remain explicit; this research forecast is not a single-reference deployment posterior
or a new segmentation component. Class-macro predicted edit closure error2.69e-14.
[Forecast report](research_20261005/pipeline_verified/membership_gain_predictor_v1/report.md).

New frozen confirmation sampling used existingCOCO metadata only, officialseed0 continuation
after1000draws/fold and photograph exclusion against actual episode/role exposure. A previous
conservative scanner incorrectly treated wholeclass image_pool references as exposure and yielded
n0; it is retained and not interpreted as pool exhaustion. Correct role/episode provenance
combined17741 exposed photographs. Actual1200=300/fold,2249unique photos, overlap0;
12170 post1000draws/10970 rejections preserved in sequential audit, no episode deduplication
or class rebalancing. Official first4000 RNG replay exact. Covers74classes, missing
32,35,68,70,78,79; classes70/78 have no legitimate unexposed pool, other4 are absent naturally
from the first300 accepted/fold despite some legitimate pairs. This is an independent-photo
confirmation cohort, not the official80class1000/fold leaderboard. NoRGB/queryGTpixels read.
Frozen C12 recipe and8 complete strongcontrols configSHA5fec4eca... remain unchanged.
ManifestSHA21475d39973d448a81140b87db875dc70a47c4f60a53d436ccea8bc8432d3ff6;
receiptSHA028e378c2c9eaa519c052395cbcaeccd4a2328b520a5f14276fc572520ec64f9.
[Sampling evidence](research_20261005/pipeline_verified/photo_disjoint_confirm1200_v1/README.md).
No confirmation GPU run has started; immutable complete-method/strongcontrol producer
preparation is assigned to the bounded worker frozen_family_confirm.

Frozen RCG2 (`research_20261005/rcg2_frozen.json`, declared before groups B/C; no refit) on further
episode sets, seed 0, streamed with `scripts/run_rcg2_stream.py`, class mIoU at original resolution,
paired bootstrap 2000 draws over photograph groups. Arms: complete FoRIS (control), RCG, RCG + sub-token
readout, full (readout + size-dependent cut).
COCO-20i group D (draws 450-599 per fold, 600 episodes): FoRIS 62.41; RCG +1.13 [+0.48, +1.81], folds
+1.06/+0.97/+0.67/+1.80; +readout +1.55 [+0.91, +2.29] (vs RCG +0.43 [+0.32, +0.58]); full +1.24
[+0.22, +2.48] (vs RCG +0.12 [-0.63, +1.05]).
COCO groups B+C+D (1800): FoRIS 60.52; RCG 61.80 = +1.28 [+0.86, +1.77], folds +1.22/+0.78/+1.39/+1.74;
+readout 62.18 = +1.66 [+1.24, +2.19] (vs RCG +0.38 [+0.33, +0.46]); full 62.38 = +1.86 [+1.13, +2.63],
folds +1.48/+1.89/+1.41/+2.67 (vs RCG +0.58 [+0.04, +1.15]).
SUIM (582 episodes, pack `decision_transfer_pack.py --dataset suim`): FoRIS 60.95; RCG -0.08
[-1.01, +0.82], folds -0.45/-1.07/+1.03/+0.47; +readout +0.30 [-0.66, +1.23] (vs RCG +0.38
[+0.30, +0.47]); full -0.34 [-1.80, +1.14]. Unresolved: no gain over FoRIS.
PACO-Part (599 episodes = 4 per-fold packs of 150, 264 fold-classes, class ids offset per pack):
FoRIS 41.45; RCG +0.30 [-0.19, +0.75], folds +0.80/+0.05/+0.61/+0.04; +readout +0.25 [-0.29, +0.70]
(vs RCG -0.05 [-0.17, +0.03]); full +0.05 [-0.79, +0.96]. Unresolved: no gain over FoRIS.
Together with PASCAL-Part (RCG +1.46 [+0.67, +2.40]): RCG is positive on COCO and PASCAL-Part and
unresolved on SUIM and PACO-Part; the size-dependent cut fitted on COCO base folds helps only on COCO;
the readout is +0.4 vs RCG on COCO and SUIM and null on both part datasets.
GT diagnostic (not a method), fine field at 1024, fixed 0.5 / best single level / best level per episode:
COCO B 61.98/61.99/69.86; C 60.48/60.48/69.52; D 63.99/64.04/71.13; PASCAL-Part 53.63/53.63/61.95;
SUIM 61.67/62.16/69.80; PACO-Part 42.16/43.38/54.46. The per-episode cut-level pool (+8 to +12) is
present on every dataset tested. Sources: `research_20261005/rcg2_group_{groupD,suim,paco_part,paco_part_f0..3}/`.
LVIS per-fold packs still running.

LVIS-92i, same frozen RCG2 and protocol (599 episodes = 4 per-fold packs of 150, 367 fold-classes, class ids
offset per pack; original resolution): FoRIS 44.27; RCG +0.31 [-0.65, +0.93], folds +0.39/+0.49/+0.05/+0.78
(unresolved); +readout +0.90 [-0.16, +1.60] (vs RCG +0.59 [+0.28, +0.88]); full 46.77 = +2.50 [+0.91, +3.07],
folds +1.35/+2.34/+2.55/+3.28 (vs RCG +2.18 [+1.30, +2.44]). LVIS uses COCO photographs, so this is a
class transfer, not an image-domain transfer. GT diagnostic, fine field at 1024: fixed 0.5 45.09, best single
level 45.82, best level per episode 54.98. Source: `research_20261005/rcg2_group_lvis{,_f0..3}/`.

Role accounting (GT diagnostic, not a method). COCO-20i fresh600 (draws 0-149 per fold, seed 0, DEV), token level
(64 x 64, soft truth), class mIoU; `scripts/run_order_tokens.py` + `scripts/score_order_tokens.py`, run on the server.
Ordering score = mIoU under the best cut of each episode; ordering loss = 100 - that. Nearest-reference vote 55.52;
FoRIS reference-contrast score (s2) 67.97; after its query clustering (s3) 69.48; FoRIS final score 69.88; RCG 72.00.
At the half-range cut the same fields give 21.68/50.92/56.22/59.53/55.34. So with FoRIS about 30 points are lost in
ordering and about 10 in the cut, and FoRIS parts 3-4 add 1.9 and RCG 2.1 to the ordering score.
What the best cut of the FoRIS final score still gets wrong: false area 86.9k tokens vs missed 22.3k; setting all false
right +19.86, all missed right +7.50. False area: 15 % in boundary tokens (+3.33), 22 % in separate regions holding no
object (+3.48), 63 % attached to a detected object (+9.04); 79 % of it has most of its 20 feature neighbours detected
too and 82 % has mostly background neighbours, i.e. whole background feature clusters are scored high. Nearest reference
token of false tokens: object 40 %, background 45 %. 34 of 600 episodes with best-cut IoU < 0.2 hold 40 % of the error
area; 88 below 0.5 hold 54 %. RCG field: same structure (false 71.9k, missed 22.3k).
FoRIS source read (`foris_source/models/foris.py`): part 2 = cosine to clustered reference object prototypes minus one
orthogonalised hard-background direction; part 3 = nearest-reference vote + hard clustering of the query with one seed
cluster; part 4 = disagreement penalty and per-cluster reweighting; fixed additive weights; min-max then 0.5.
Source: [order report](research_20261005/order_fresh600/report.md).

Ordering loss split into resolution, representation and transfer of the object signature (GT diagnostic, not a
method). Same fresh600, token level, ordering score = class mIoU under the best cut of each episode;
`scripts/run_transfer.py` + `scripts/score_transfer.py` on the server. Ladder: cosine to the reference object mean
66.09 -> FoRIS reference-contrast score 67.97 -> FoRIS final score 69.88 -> RCG 72.00 -> cosine to the query's own
object mean (truth) 83.08 -> one linear direction fitted on truth, cross-fitted over 8 x 8 blocks 87.31 -> token
ceiling 91.70. So 17.0 points lie between the reference signature and the query's own signature under the same
cosine readout; all of FoRIS after the prototype recovers 3.8 of them and RCG 2.1 more; 11.1 remain, against 4.2 for
a better readout than cosine, 4.4 for more than one direction and 8.3 for token resolution.
By object share of the query (<0.02 / 0.02-0.1 / 0.1-0.3 / >0.3; n 140/260/143/57): reference cosine
49.94/62.25/74.82/83.94; RCG 55.20/67.52/81.58/87.19; own-signature cosine 71.55/79.94/84.24/90.94.
In the 34 episodes where FoRIS has best-cut IoU < 0.2 the reference object mean is as close to the query background
mean (cos 0.46) as to the query object mean (0.47); own-signature cosine reaches 0.64 mean IoU there, FoRIS 0.12.
Derived detector tested and rejected: the matched filter in the query's metric with the reference signature
(`mf_ref`, shrinkage 0.1/1/10 of the mean eigenvalue) orders at 19.98/28.34/47.84, below plain cosine; with the
query's own signature it reaches 53.44/75.12/85.45. The filter is sound when the signature is right and amplifies
the move of the signature when it is not (8-episode complete-mask smoke agreed; not run at 600).
One self-training step (ridge refit on the FoRIS score or the RCG field) 69.48/70.68: no gain over its input.
Source: [transfer report](research_20261005/transfer_fresh600/report.md).

How the object signature moves between reference and query, and two derived estimators (same fresh600, token level,
ordering score under the best cut per episode; `scripts/run_signature.py`; GT only in the ceilings and the geometry).
Stated before the run: an image-wide offset would be removed by centring each image; an instance difference only by the
query's own structure. Result: centring each image by its token mean 68.07 against 66.09 uncentred (ceiling with the
query's own signature 83.81 centred, 83.08 uncentred), so the image-offset model accounts for 2 of the 17 points;
second-moment alignment (CORAL) 34.65 uncentred / 62.41 centred, worse than plain cosine; matched filter with the
centred reference signature 27.70 / 47.34; mean shift from the reference signature on the query tokens 58.69 (t=0.03),
20.67 (t=0.1), centred 65.93 / 55.35: it leaves the object for the dominant mode. All rejected.
Geometry: cos(reference, query signature) 0.47 / 0.57 / 0.65 / 0.71 for FoRIS best-cut IoU <0.2 / 0.2-0.5 / 0.5-0.8 /
>=0.8; cos(move, offset of image means) 0.41 in every group; 19-27 % of the move lies in the query's ten main
directions. Scale: Spearman of |log linear size ratio| with signature cosine -0.22, with the ordering gap +0.11; the gap
is 0.11 at matched size (235 episodes) and 0.16 / 0.24 when the reference object is 2-4x / >4x larger (78 / 27).
Concentration: replacing only the episodes with RCG best-cut IoU < 0.2 (35) by the own-signature result lifts the
ordering score from 72.00 to 78.49; < 0.5 (83) to 81.26; the full ceiling is 83.08. In those episodes no stored
reference evidence ranks true object tokens above the tokens RCG wrongly keeps: AUC for IoU < 0.2 / 0.2-0.5:
nearest-reference contrast 0.32 / 0.44, 10-NN vote 0.36 / 0.37, reference cosine 0.48 / 0.59, FoRIS s2 0.41 / 0.55,
FoRIS final 0.28 / 0.41; the wrongly kept area is 12.2x / 1.6x the object (medians).
Source: [signature report](research_20261005/signature_fresh600/report.md).

Where evidence against a wrongly chosen region could still be: layer, scale, or nowhere (GT diagnostics; same
fresh600; `scripts/run_region.py`, `scripts/score_region.py`). Ordering score of the cosine to the reference object
mean by encoder layer 11 / 15 / 19 / 23: 18.95 / 20.89 / 27.70 / 61.83 (debiased last layer 66.10). AUC of true object
tokens against the tokens RCG wrongly keeps, episodes with RCG best-cut IoU < 0.2 (n=33): 0.44 / 0.44 / 0.42 / 0.44
by layer, 0.48 debiased; 0.2-0.5 (n=41): 0.53 / 0.55 / 0.59 / 0.62, 0.59. Scale: the window around the true object
enlarged to the reference object's size and re-encoded; inside the same window the reference cosine orders at 68.19
with enlarged tokens against 68.28 with native tokens (600), 40.79 against 42.15 where the window is under 0.75 of the
image and RCG IoU < 0.5 (35); cos(reference signature, object signature) 0.47 -> 0.48 in the worst group. Neither layer
nor scale holds the missing evidence. Contact sheets of the 24 worst episodes
(`research_20261005/region_fresh600/worst_1.jpg`, `worst_2.jpg`; read by eye, not counted): reference masks that are
a frame-filling surface (dining table = red cloth), reference objects of a few tokens, near-class confusions
(handbag/suitcase, cup/bowl), and queries where truth marks one of several same-class instances.
Reliability role (`scripts/score_components.py`, token level, RCG at 0.5 = 63.03, break-even precision 0.39): emptying
the 85 masks below break-even 68.51; deleting every connected component below break-even (truth) 71.85 =
+8.82 [+6.08, +9.56], i.e. as much as the best cut per episode (72.00). 747 of 1811 components are below break-even
and hold 61 % of the false area. Count-level AUC good vs bad component: mean RCG field 0.81, mean FoRIS score 0.77,
share of mask 0.76, mean reference cosine 0.68; episode level at most 0.75. Fold-nested deployable rules (threshold or
logistic fitted on the other folds): best single threshold (component max relative to the image max) +0.71
[-0.38, +1.56]; logistic on field statistics and size +0.83 [+0.15, +1.27], folds +0.85/+0.35/+0.28/+1.84; with
reference-cosine statistics added -0.29 [-1.76, +0.89]; keep-largest control -4.11. The false area sits in large,
confidently scored components, so count-level separability does not turn into area.
Sources: [region report](research_20261005/region_fresh600/report.md),
[component rules](research_20261005/region_fresh600/components.md).
Process note: the background waiter for the return-test groups reached its two-hour limit and was stopped; it was
not restarted. Lane c (PACO-Part, LVIS) of that test was stopped by me to free the GPU; lanes a (COCO group B) and
b (SUIM, then PASCAL-Part) continue.

Two predictions written before their runs, both on fresh600, token level (64 x 64), class mIoU; both failed.
(1) Prior term (`scripts/run_prior.py`): a query-only objecthood should separate true object tokens from the tokens
RCG wrongly keeps where reference evidence does not. AUC for RCG best-cut IoU < 0.2 / 0.2-0.5 (n=33 / 41):
boundary-connectivity 0.58 / 0.57, cosine to the class token 0.54 / 0.50, normalised-cut eigenvector 0.57 / 0.52
(reference cosine 0.48 / 0.59, RCG field 0.14 / 0.30). Fold-nested component deletion with each objecthood added to
the field statistics: +0.73 [+0.02, +1.22], -0.15 [-1.50, +0.84], +0.63 [-0.20, +1.17], all three -0.00
[-1.66, +1.04], against +0.83 [+0.15, +1.27] without objecthood (same-information control). No gain.
(2) Candidates from the query alone (`scripts/run_candidates.py`, `scripts/score_candidates.py`): all nodes of a
complete spatially-connected agglomerative hierarchy of the query tokens. GT ceilings: best single node 73.63 (Ward) /
69.10 (average linkage); greedy union of up to 2 disjoint nodes 79.54 / 76.04; up to 3: 80.95 / 78.45; against the
best level per episode of the FoRIS score 69.88 and of the RCG field 72.00, and the fixed 0.5 level 59.54 / 63.03.
Truth-free selection: the node that best splits the reference cosine 37.68 / 38.14, the RCG field 43.55 / 45.05;
expected IoU with the clipped RCG field 28.92 / 25.36 (the field is not calibrated, background near 0.25); nodes
closest to the RCG mask, up to 1 / 2 / 3 / 8 nodes, Ward 58.22 / 61.10 / 61.31 / 61.36 = -4.81 [-6.22, -3.58] to
-1.67 [-2.31, -0.78] against the RCG mask (control 63.03), average linkage -6.95 to -2.46. The hierarchy holds
better regions than any level set, and no tested rule selects them; the ceilings are selections by truth among 8191
nodes and are not comparable to a method.
Process: to free the GPU I stopped the return test on COCO group B at 400 of 600 (counts are written only at the end,
so nothing was kept) and the role accounting on SUIM / PASCAL-Part / PACO-Part fold 3 / LVIS (PACO-Part folds 0-2
finished, unscored). The return test therefore remains unverified.
Sources: [candidates](research_20261005/candidates_fresh600/report.md), [prior](research_20261005/prior_fresh600/report.md).

Stored-data checks made while the server was in no-card mode (0.5 CPU, no GPU; no new run). (a) Reverse matching,
fresh600 token record: density of nearest-query hits from reference object tokens on true object tokens against the
tokens RCG wrongly keeps, RCG best-cut IoU < 0.2 (n=33): 0.073 vs 0.075, higher on the object in 42 % of episodes;
0.2-0.5 (n=41): 0.128 vs 0.108, 56 %. No information, in agreement with the earlier reverse check (AUC 0.709 against
0.776 forward) already in this ledger. (b) Geometry of the input: per-episode mean IoU of RCG at 0.5 by orientation
of reference / query: landscape-landscape 0.650 (n=327), portrait-portrait 0.674 (43), landscape-portrait 0.650
(90), portrait-landscape 0.658 (76); by reference object share <0.01 / 0.01-0.03 / 0.03-0.1 / 0.1-0.3 / >0.3:
0.640 / 0.624 / 0.656 / 0.664 / 0.643. Neither the square resize nor the reference size is a defect. A class-pooled
reading of the orientation groups showed a 4-point difference that the per-episode means do not; it was class mix.
(c) Break-even law on the public 4000 (`rcg_anatomy4000/counts.npz`, 80 classes, RCG against the FoRIS pre-CRF mask
at 1024). The six edit types reproduce RCG exactly (identity check 0). With the share of true pixels per edit type
taken from the other three folds (disjoint classes) and the edit volumes and FoRIS I/U of the class itself, the
predicted class gain has the sign of the actual gain in 72 of 80 classes, Pearson 0.39, mean absolute error 1.17
points; by fold predicted +1.94/+1.47/+1.62/+1.80 against actual +1.83/+1.78/+1.73/+2.35; overall +1.71 against
+1.92. The prediction uses the class's FoRIS I/U, i.e. labels, so it is a retrospective check of the accounting, not
a deployable forecast. Choosing edit types per class by the break-even at that class's J gives +1.67 and a set that
clears break-even for every J in 0.3-0.75 gives +1.36, both below applying all edits (+1.92); classes with FoRIS
IoU < 0.45 gain +1.79. So type-level purities are not constant enough across classes for type selection to pay, and a
low operating point does not by itself explain the null RCG results on PACO-Part and LVIS.
Process: the earlier ledger sections (failure ledger, 2026-10-02 to 10-05) already hold results I re-derived today
(reverse check, region-level evidence, scale matching, zoomed rerun, 64 % of false area attached to the target) and
one that contradicts my reading of the worst episodes as reference ambiguity: swapping in three other references of
the class rescues only 17-29 % of failed episodes; the difficulty is in the query.

Bound of the cut-level route, prediction written before the result (public 4000, RCG field, stored per-level counts,
server CPU in no-card mode, no GPU; `scripts/score_cut_predictability.py`). Question: the best level per episode is
worth +8.91 with truth and the pool is present on every dataset, while every label-free rule tried so far gives at
most about +0.5 and six fitted area bins +0.38. Is the best level readable from the field at all? Test: boosted trees
fitted on the other folds (no shared photographs, base-fold labels) from label-free descriptors of the level sets
(histogram: area curve, stability, moments; spatial: components, peak component, perimeter, edge gradient) to the value
I - J U of each level; the held fold takes the level of highest predicted value. This bounds label-free rules on these
descriptors from above; it is not a method. Prediction: at most +1.0 over the fixed 0.5 level (expected +0.4 to +0.9).
If it reads +2.5 or more the information is in the field and a rule is worth deriving; if it stays under +1 the cut
level is closed as a source of gain for fitted and label-free rules alike.
Result (4000 episodes, 80 classes, 2000 photograph-group draws; fixed 0.5 = 62.33): six fitted area bins +0.29
[+0.04, +0.53], folds +0.22/+0.42/+0.28/+0.24; boosted trees on histogram descriptors -0.76 [-1.31, -0.29]; on histogram
and spatial descriptors -0.67 [-1.19, -0.20], folds -0.37/-0.64/-0.36/-1.31, by true object share <2% / 2-10% / 10-30% /
>30%: +0.20 / -1.54 / -1.78 / +1.10. Truth rows: best IoU level per episode 71.24 (+8.91); class-optimal level per
episode (each episode maximising I - qU at its class ratio) 77.10 (+14.77). The prediction (at most +1.0) held. A
flexible fit with base-fold labels does not read the best level from the field and its level-set geometry; taking the
level of highest predicted value is itself biased towards noisy extremes, so this is strong evidence against the
route, not a proof of an upper bound. Together with Otsu / isodata / Kittler / stability / area-curve ridge / share
estimators already in this ledger, the cut level of this field is closed as a source of more than about +0.3 to +0.5.
Source: `research_20261005/cut_levels4000/predictability_rcg.md`.

What the failing episodes are, counted (fresh600, token record, no model; `scripts/render_failures.py`, sheets in
`research_20261005/failures_fresh600/`). 89 of 600 episodes have best-cut IoU of the RCG field below 0.5. I read 48 of
them by eye (the 36 worst and numbers 48-59); a single reading, categories are mine: (A) a neighbouring category taken
instead of or together with the object (sofa for chair, suitcase for handbag, plate / pan for bowl, bottle for cup,
other food, jeep with boats) 21; (B) an unrelated region taken while the object covers at most about 1.5 % of the
image 10; (C) reference not usable as a description (dining table = frame-filling cloth, sink mask including the
counter, object barely visible) 6; (D) truth questionable (cake shaped as a fire engine marked on a small part, only
the mirror image of a teddy bear marked, crowd excluded) 4; (E) object found only in part or mask much smaller than
the object 6; unclear 1. 40 of the 48 objects cover less than 6 % of the query.
Per episode, from the stored fields (84 episodes with at least 3 object tokens and 3 wrongly kept tokens): the object
is closer to the reference than the wrongly kept area in 34 by mean cosine to the reference object mean (37 by upper
quartile), 32 by the FoRIS contrast score, 24 by the 10-nearest-reference vote; in category A 9 of 21 (cosine). So in
the failing episodes the wrong region is, more often than not, the better match under every reference comparison
stored, episode by episode and not only on average. Earlier statement corrected: this is not mainly an ambiguous
reference (C is 6 of 48).
