# 真实九窗的统一完整组合：冻结规格

2026-10-11。**已冻结一条完整组合和两个必要控制，交由执行分支唯一生产；设计分支未运行候选，尚无新完整输出成绩。** 已完成保存 whole 场、同头真实九窗和最后一个 whole-only 条件诊断。下一步直接生成全部 1300 例双帧完整 mask，全部封存后评分；不再追加诊断作为执行前提。

目标是保住原九窗区域法在 LVIS600 与 Deep fast100 的完整收益，补回严格多数擦掉的真实少数支持，并识别多视图一致接受的干扰。原九窗区域法、完整 FoRIS 和匹配预算控制始终保留。不能要求某个新增弱读出单独胜过整个 FoRIS 来决定其是否可组合。

## 已证明的作用和必须保留的反例

| 同批原图规定口径 | 完整 FoRIS | 未筛选局部多数 | 原九窗区域法 | 同 K 几何 |
|---|---:|---:|---:|---:|
| LVIS600，600 观察槽位 | 47.779371 | 34.128915 | 52.005138 | 49.744750 |
| Deep fast100，道路总 I/U | 29.377542 | 35.035001 | 37.344028 | 36.537657 |
| PACO fast600，303 固定槽位 | 38.460868 | 29.538809 | 34.971403 | 未保存同条件臂 |

局部真实观察负责产生整图读出缺失的覆盖；组件 query-context 已删除大量容易拒绝的区域；严格多数负责抑制零散支持，却也擦掉真实目标。三个作用需要放在完整输出中共同检验。PACO600 的接受少数支持占 canonical GT 像素 18.92%，同时四覆盖四票一致 FP 达 6,128,630。投票一致本身不证明目标身份。

旧 `M ∪ (B ∩ A)` 已完成并失败：LVIS600/Deep fast100/PACO600 原图分别为 50.681860/36.347595/37.492459；前两组损害原区域法，PACO 仍低于完整 FoRIS，且输给简单 whole union。该规则只作为已封存反例复用，不换阈值重演。

固定 PACO `dev_s1/paco_part/0/14` 有 381,794 个接受少数 TP，其中 guide=FG 部分为 380,618 TP/3,045 FP；`dev_s1/paco_part/0/0` 的同一条件却有 28,180 FP/0 TP，且四覆盖四票接受一致 FP 为 40,518。旧 whole/local4 双负在 PACO 接受少数且 guide=FG 中仍含 11,350,066 TP，M 内含 30,056,136 TP；四票一致的双正仍含 1,428,242 FP。不能把已有正负号或一致票当可靠的添加/删除证书。

单次最终 CRF2380 已完成，但 Deep N=29.685895 仍远低于原快九窗 37.344028；匹配 LVIS600 固定 920 槽位 N=31.925710 仍低于原区域法 33.718963。6380 的 full-reference 和 case-deleted 两种校准、八个 cohort 均已完整完成。case-deleted N 对旧 N 在 LVIS1400 为 +0.443422 pp，却仅比 full-reference N 高约 0.007 pp；native SUIM2000 为 −0.115246 pp。LOO 是经典校准控制，不提供新 query 排序，也不作为创新或局部争议区可辨识性的证据。

信号分支已完成 1400 source-arm 的保存 whole 场审计，独立复核 319,332 项。真实九窗三个目标组的结果如下；这是 canonical 唯一像素 ROI 诊断，不是完整 mask mIoU：

| 组 | 接受少数的 whole ridge 宏 AUC | full-H 正号保留少数 TP | 用 full-H 正号筛 M 会损失 TP | AU4 仍保留 FP |
|---|---:|---:|---:|---:|
| LVIS600 | .7009 | 16.36% | 46.11% | 757,030 |
| Deep fast100 | .6411 | 20.92% | 39.96% | 156,282 |
| PACO600 | .7647 | 21.00% | 50.30% | 2,005,481 |

因此已有 whole 排序有局部关联，却不能直接作删除门控。full-reference 重标定没有提供新 query 排序；PACO 固定 `0/14` 的旧 H 正号仅涵盖 6,453/381,794 个少数 TP，full-H 为 8,566。`0/0` 虽能拒绝大量干扰，旧 H 正号也仅保留 M 内 3,479/23,513 个 TP。这些覆盖代价必须与拒假收益共同保留。[whole 机器报告](../../evidence/local/nine_window_signal_20261011/report.json) SHA 为 `1ff1b2a7bf0bf6060328d4a7037f463d353c71dc0014c559a43229e3e1495c36`。

## 严格绑定的输入与帧

唯一候选执行集是现有 `lvis_original600`、`deep_fast100`、`paco_fast600` 共 1300 个不同 episode；按旧 reconstructed seal 的原顺序选取。不存在新队列、下载、DINO 前向、模型层/头/阈值/系数扫描或全局训练。来源 SHA 和逐例真实 native key 由 [执行输入收据](../../evidence/local/nine_window_execution_20261011/input_binding_receipt.json) 绑定。14065 个唯一 raw 输入的 entry/payload 存在性已核对，missing=0；每个实际消费的 payload 仍须按收据验证 SHA，不能把存在性写成内容验证。

九窗是 `x,y ∈ {0,256,512}` 的真实 512×512 查询视野，每个 canonical1024 像素的覆盖分母是 1、2 或 4。设 `a_v(p)` 为原组件决定所保留的真实窗 mask，`a(p)=Σa_v(p)`、`c(p)` 为原覆盖数，`A={a>0}`、`M={2a>c}`。拒绝组件和无提案都不减少分母。

完整候选先在 canonical1024 组合，再按原 `bilinear(align_corners=False)` 和严格 `>.5` 渲染到原图；两帧 mask 均须封存。LVIS/Deep 旧 FoRIS canonical 字段是原图 mask 的回渲染 guide；PACO 保存真实 CLI1024。原图 FoRIS 三组均为真实保存终点，不把回渲染 guide 冒称 native FoRIS canonical。

LVIS600 同一组 mask 同时报告 600 观察槽位和官方 10 fold×92 class=920 固定槽位：原区域法原图分别为 52.005138 与 33.916394，canonical 为 51.702409 与 33.718963。跨 6380 的比较只能用匹配固定 920 槽位。PACO600 保留 303 固定槽位及缺失槽位零值，264 槽位有观察；不能替换成 pilot100 的 87 观察槽位。Deep100 用道路总 I/U，不能改为逐例 IoU 平均。

组件来源使用额外参考裁剪的 LVIS236/PACO13 单独标注；同参考视野子组为 LVIS364/PACO587，Deep100 全部一致。cohort/裁剪标签仅用于绑定和报告，不能成为推理 pipeline 选择器。

LVIS600 使用历史 local-CRF 读出，Deep/PACO 使用历史无 local-CRF 快读出。本轮固定同一组合公式并限定结论为这些绑定源臂，不把历史版本差异变成部署时的 domain 开关。PACO 同600慢局部 CRF 结果仍缺失，不据快版替它作结论。

## 唯一新增诊断：真实支持窗的任务证据

保存 whole 场的 ROI 审计之后，只新增并完成一项真实窗统计：

`L_A(p) = max { H_v(p) : v 覆盖 p 且 a_v(p)=1 }`。

只在非空支持 ROI 使用该统计；无接受支持处不定义为低分。max 表达“至少一个实际提议该像素的窗有参考任务证据”，其公式在 GT 诊断前确定，不从结果挑选聚合方式。它必须同时保留接受少数 TP、M 内 TP 和一致接受 FP，不能只报告容易清理的未接受区。

`H_v` 使用与该 episode 原 saved whole ridge128 **完全相同的 whole-R 视野、固定 ridge 头、APD 变换和旧 reference calibration**。复原函数为原定量质量采样 128/role、lambda=.01 的同一固定 ridge；不用含其他头的复合 reference-fit，也不新拟合 local-R 校准。先在 whole-Q 哨兵核对保存 ridge128 和 calibration，再仅把 Q 输入换成真实九窗 O24。

每个真实 window-Q 都沿同一 whole-Q 读出流程：`raw64 → bilinear128 → normalize → 原固定 APD normalize → 同 coef/bias → clip((ridge128−t_R)/Δ_R,−1,1)`，之后 `H128 → bilinear512` 放回对应真实窗。原 `Δ_R<=FP64 epsilon` 保持精确零证据。不能用 whole/local4 插值伪造九窗得分，不能同时把 query grid 改成另一条 native64 读出链。原参考裁剪差异仍作为组件来源分层；本统计的身份是 whole-R 指定任务与 native-Q 观察的关系，不称其为组件自身的来源头。

预声明 ROI 为：`A\M`、`M`、`c=4,a=4` 的接受一致区，并在三者中分别保留原 guide FG/BG 分层。后者由合法 mask 成员关系定义，不能以 GT=BG 定义推理 ROI。M 的 guide=BG 子区是原九窗超出整图方法的独有覆盖，必须单列其 TP，不能用 M 总体 AUC 证明它已被保护。各 ROI 保存 `L_A` AUC、旧合法零点 `L_A>0` 的 TP/FP/像素数、whole-H 与 `L_A` 零点四格、零证据例数，并报告同 R/另裁 R 两个子组。AUC 有效例必须同时有 FG/BG，缺失不能填 0 或 1。

固定反例为上面的两个 PACO case、Deep `dg18-0000` 以及本 1300 输入中匹配的旧小参考清零 LVIS case。只保存聚合整数、小型逐例记录、来源/代码 SHA 和成本；真实 raw/临时 dense 得分逐例释放，不保存 dense 派生场。

| 组 | 接受少数 whole-H 宏 AUC | 真实窗 L_A 宏 AUC | L_A 正号 TP/FP | M 超出 B 的 TP 被 L_A 正号保留 |
|---|---:|---:|---:|---:|
| LVIS600 | .691774 | .717504 | 458,590 / 1,438,628 | 194,315 / 719,716 |
| Deep fast100 | .640084 | .754486 | 184,204 / 309,927 | 403,080 / 647,967 |
| PACO600 | .763676 | .695631 | 3,140,143 / 1,997,406 | 551,029 / 3,040,685 |

主 Sol 的唯一配对 R/Q 连通区间支持少数区 L_A−whole-H 在 LVIS 为 +.025730 [.011818,.038851]、Deep 为 +.114402 [.089341,.138674]，PACO 为 −.068046 [−.081575,−.053934]。真实窗确实提供了新排序，但不能全面替换 whole；本方案保留两者的互补正证据，不按数据集名字选择它们。PACO `0/14` 的真实窗正号为 207,831 少数 TP/1,350 FP，whole 仅 6,453 TP/0 FP；`0/0` 的 AM 197,563 FP 和 AU4 40,518 FP 被 L_A 全拒，但它的 M 真覆盖只保留 97/23,513 TP。不能仅展示第一个成功例，也不能把第二个反例改成默认全局删除门控。

真实窗完整报告 SHA 为 `0662354654f1ecfa5702e5041128fd688d2d0167e38b84a543314076d082d960`。全部 1300 个 whole ridge128 的复原最大误差 `5.107e-15`，旧校准误差 0。成功 pass 用 1862.58 wall/4520.92 CPU 秒、单 worker/4 Torch 线程、峰值 RSS 1.70 GB；读取 14,255 次 payload、229.78 GB，复原 1300 个既有参考头，应用 11,655 个真实窗，0 DINO/CRF/新 mask/dense 保存。该成本属于缓存诊断，失败首例的额外工作另保留，不冒称冷端到端速度。

## 最后一个 whole-only 缺口

仅从旧已评分整数 joint 分区可知，B 而零接受支持仍有 LVIS 987,426、Deep 61,807、PACO 9,442,502 个 canonical TP；M 超出 B 也分别有 719,716、647,967、3,040,685 个 TP。两块都不能直接定义为错误。为处理 whole 独有覆盖，只补一次已封存完整旧 `anchor.global`（记 R0）的条件诊断，不重跑其链：

| canonical B 且 A=0 | R0 保留 TP | R0 损失 TP | R0 拒绝 FP | R0 仍保留 FP |
|---|---:|---:|---:|---:|
| LVIS600 | 642,883 | 344,543 | 3,201,322 | 2,764,677 |
| Deep fast100 | 45,806 | 16,001 | 412,070 | 341,315 |
| PACO600 | 6,294,600 | 3,147,902 | 8,541,815 | 5,669,030 |

旧 R0 的作用是保留完整 FoRIS 置信约束和原 `.5H` 的有界参考纠正：它有实在拒假作用，也有不可省略的漏检代价。本组合只在 B 且无接受局部支持处使用它，不要求其孤立完整输出胜过原九窗。原图同 ROI 在 LVIS 保留 173,675 TP、损失 90,620 TP、拒绝 839,157 FP；PACO 保留 101,528 TP、损失 55,445 TP、拒绝 290,851 FP；Deep 两帧相同。

1300 个旧 B 原图与九窗 B 原图逐位一致。旧 B 的实际 CLI 与回渲染 guide 在 LVIS/PACO 有 615,339/3,769,620 个累计边界 XOR，Deep 为 0。五个源零 H case 的旧 R0 原图等于 B，全部保留该 ROI 的 617 TP/23,564 FP；canonical 因 native B/guide 别名少了 84 TP/863 FP，不能称为语义拒绝。因此最终零证据分支直接使用九窗 B guide，避免把这个坐标别名误当 task 证据。

此条件诊断耗时 11.35 wall/10.79 CPU 秒，0 fit/raw/native/CRF/新 mask；报告 SHA `33b9f78d57fdf5c6d34042a061fd503a89b5815c0f894e4994e05d7640995eb5`。它使用各帧已保存 R0，**不是**新组合 canonical mask 导出的原图结果。

## 唯一完整组合与两个必要控制

定义 W 为旧 whole `H128` 按原 bilinear 投到 canonical1024，L 为上述真实支持窗 `L_A`，R0 为已封存 `old_arms.R.cli_key` 完整 mask。`valid` 只取原源校准的有限正分离度 `Δ_R>FP64 epsilon`；它是源数据状态，不是 query selector。设 `D=A∩(M△B)`，只在有合法接受支持的 M/B 争议处使用新任务证据。

当 valid 时冻结：

`F = (M∩B) ∪ (B∩¬A∩R0) ∪ (D∩((W>0)∨(L>0)))`。

当源校准无有效证据时冻结 `F=M∪(B∩¬A)`。五个零 H 参考不是 query 的语义 BG；不以零值删除 M，也不以 native B/guide 的边界差异制造拒绝。此行为和所有控制相同，不按类别、dataset、query GT 或高低 AUC 选择公式。

本组合保留真实局部提案及原组件接受、原多数 M 与整图 B 的一致前景；B-only 依完整旧 FoRIS 置信约束+有界 ridge 判断；接受少数且 B 为前景时，whole 或实际支持窗任一正任务证据可以补回；M 超出 B 时，两个任务视野都非正才撤回。它允许真实窗恢复 `0/14` 的旧 whole 缺失证据，又保留 PACO 已有 whole 独有正证据。没有新增局部组件选择、CRF、源校准或系数。

它有明确边界：M/B 一致的 FP 将保留；两者均 BG 的接受少数支持不会恢复；两个参考场都否定的 M 独有真覆盖仍会丢失；旧 R0 会拒绝上表中的部分 whole 独有 TP。全部删真/删假和补真/补假都公开，不能要求每个 ROI 零 TP 损失，也不能隐去 Deep/LVIS 可能回退的代价。组合操作和旧 ridge 都是已知原语，此次是一个完整可证伪机制假设，尚不构成创新或 Strong 成功声明。

只新增三臂：

1. `composition.native_or`：上述完整 F。
2. `composition.whole`：完全相同的保护块、R0 和缺信息分支，只把 D 中证据改为 `W>0`。它检验真实窗增量是否仅重演已有 whole 作用。
3. `composition.canonical_same_edit_budget`：保留同一保护块和 B-only 块，在 `E_add=A∩B∩¬M` 内按已有未裁剪 whole ridge128 的 canonical bilinear score 降序选与 F **同数量**的添加；在 `E_del=M∩¬B` 内按该 score 升序选与 F 同数量的删除。未裁剪 whole 排序是强简单控制，不能用已裁 H 的大面积饱和 ties 降弱它。所有 ties 用固定 row-major，报告边界 ties。源无效时两种争议域编辑数量为 0，三臂完全相同；相对 M 的全图添加仍可能包含三臂共享的 `B∩¬A` 块，两种数量分开记账。

准确计数为 `K_add_D=|F∩E_add|`、`K_del_D=|E_del∩¬F|`，共同 B-only 块另计。预算等量保证在 canonical1024；所有三臂整体再统一 bilinear/strict>.5 导出原图，原图实际编辑面积单列，不能把 canonical 同 K 冒称原图同 K。R0 原图完整 mask只作为旧控制和已完成条件诊断；执行分支不能直接把它拼进新原图 F，必须整体渲染新 canonical F。

全部 1300×三臂×两帧 mask seal 后评分。旧完整 FoRIS、完整旧 R0、原九窗 M、未筛选多数、已有同 K 几何以及失败旧恢复规则直接复用同输入封存 mask/IU。原本派生的同 K 几何是组件预算控制，新增第三臂是 canonical 像素编辑预算控制，两者保持不同名称和作用。

执行分支复用已完成 native1300 的全部 whole-Q 哨兵证明与旧 calibration；仅在有效源上从原 R O24 复原同一必要 coef，计算真实九窗 H 后立即释放。不得重复 whole-Q1300/ROI/AUC/6380 诊断。实际消费的参考/窗 payload 校验 SHA；现有 whole 字段与旧 R0 mask按绑定 SHA 读取。truth-free 独立 Boolean/预算实现逐例复核完整新 mask后封存，scorer 在全体 seal 后独立读回导出、I/U、编辑与合池。

完整评分保留 TP/FP/FN/I/U、相对 M 的补真/补假/删真/删假、逐 class/fold 贡献和现有 query-photo/RQ-connected 区间。Strong 结论需要同批完整输出胜过 FoRIS、原九窗和必要简单/同预算对照；区域 AUC、像素总数、单 case 或参考侧分离度不能代替这一验收。所有输入是已曝光开发 cohort，不能称未见泛化确认。

主 Sol 的[无 GT 身份库存](../../evidence/local/sol_validation_20261010/nine_window_mask_switch_coverage.json)另证明本 1300 输入在三组中都没有“同 R/Q RGB、crop、尺寸，仅改变 R-mask”的配对组。PACO 的 570 个照片标识对应 580 个实际 Q RGB/crop/尺寸和 595 个实际 R/Q 输入；重复照片不能当 Mask-only 干预。此处的 ROI 结果可支持固定融合规则与参考/查询信号的关系，不能单独证明固定 R/Q 只改 Mask 的因果机制。

机器规格：[protocol.json](../../evidence/local/nine_window_composition_20261011/protocol.json)。原始证据：[真实九窗](parallel_signal_20261010/01_multi_observation/REPORT.md)、[单次 CRF](../../evidence/local/sol_validation_20261010/SINGLE_FINAL_CRF2380_RESULTS.md)、[参考支持](../../evidence/local/sol_validation_20261010/REFERENCE_CALIBRATION_SUPPORT.md)、[6380 两校准/八 cohort](../../evidence/local/sol_cache_experiments_20261011/reference_calibration_loo6380/REPORT.md)。
