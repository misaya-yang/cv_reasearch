# 核心研究路线与当前进展

更新至 2026-10-05 07:49 UTC。仓库版本固定为 `0e1dea19716faec8cc4c623ce54c79f8a9df3173`。本文解释项目要解决的推断问题、已经建立的结果、当前最接近目标的构造，以及尚未完成的验证。仓库保持只读。仓库历史结果与本次缓存上的 CPU 实测分别标注，所有云端独有数字均提供本包证据入口。

## 1 当前结论

项目已经证明：在相同冻结 DINOv3 与 FoRIS 证据上，用基类标注拟合决策读出，可以在原分辨率完整流程的 CONFIRM600 上获得 63.33，相对 FoRIS 59.78 增加 3.54 点，区间 [1.96, 4.96]。但无标注自造样本并未复现这项收益，跨数据集迁移也未建立稳定收益。因此它是一个真实的、使用基类标注的结果，不能改名为零训练方法，也尚不足以支撑当前独立方法主张。[仓库 README 已记录，本文未重新训练或运行该 GPU 流程](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

当前 DINO 缓存研究的主要进展是把“一个分数图怎样切”推进到“若干完整分割假设怎样竞争”，并把部分错误定位到类别身份、参考到查询的范围迁移，以及边界覆盖率三个不同环节。旧 120 例上，RCG 为 63.723，FoRIS native 为 62.018，差 +1.705；同输入的 MEAN_CONTROL 为 63.500。完整假设的 source_contrast_mean 为 62.640，scalar_contrast 为 63.807。它们都还没有在当前样本上达到“相对 native 稳定至少 2 点”的目标，不能因为对 pre-CRF 对照超过 2 点而宣称达标。[本包固定 120 例比较](evidence/01_02/prior120_locked.json)。

新增 100 例的九个固定输出已经冻结，尚未评分；补齐 DEV241 还缺 21 例。此前被称为 new60 的数据属于已反复研究的旧 120 例，不能与此次封存的新增 100 例混淆。[新增数据冻结记录](evidence/01_02/new100_freeze.json)，[字段审计](evidence/01_02/new100_field_audit.json)。

本报告遵循最新判断准则：看相对 native 的效果大小、跨折与新增数据稳定性，以及同信息简单对照。区间跨零表示不确定，不能自动淘汰；区间完全为正也不自动代表达到至少 2 点或证明机制新颖。对已经明显输给简单对照的构造，记录失败环节，保留其他可能解释。

## 2 任务究竟是什么

输入是一张参考图及其目标掩码、一张查询图。目标是在查询图中找出与参考目标同类的所有相关区域。参考掩码既提供“这个对象是什么”的身份条件，也提供哪些外观属于前景、哪些属于参考背景的监督。

冻结主干不等于没有推断选择。至少有四个可独立改变的决策：

1. **身份读出**：查询里哪个相似物体才是目标类，而非外观相近的背景物体。
2. **对象范围**：正确种子应该扩展成哪一整块，哪些不同外观部件属于同一目标。
3. **二值决策**：连续分数怎样变成目标面积和区域选择。
4. **像素边界**：16 像素 token 跨越边界时，怎样恢复真实像素覆盖。

“分数高”“种子纯”“能够匹配参考”“在参考上重建得好”分别针对不同环节。一个信号在其中一项有效，不代表它能完成整个分割。近期全 DEV241 的 source-verified query centroid 实验就是直接反例：在最高分 token 错类的子集里，候选纯度改善，但其完整掩码只有 39.05，对照 58.56；源域选择的范围明显过大。[最新仓库 README 的 query-native candidate validation 与 failed scope transfer 小节](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

### 2.1 输入 表示 关系与oracle的边界

RGB里存在某种形状或类别线索，不等于最终层缓存必然保留了它。最终层、中间层、内部注意力计算也不能互相替代。本文只把实际使用的均值、近邻、核项、图边与候选mask算作算法证据。后处理重用这些关系，不据此宣称给模型新增了跨姿态或跨外观识别能力。

GT oracle额外注入query标签或选择权。高oracle只证明候选中存在好mask，不证明reference/query能辨识该mask，也不证明所需语义规律已可从当前表示读出。反过来，若读出失败，也不能据此证明表示必缺信息。本文所有“空间”“容量”和“可达”均限定于具体输出构造与其额外输入。

## 3 三种信息资源必须分列

| 研究列 | 允许使用的信息 | 当前代表结果与解释 |
|---|---|---|
| 冻结 DINO 零训练列 | 参考图、参考掩码、查询图；使用冻结特征进行推断，不用其他 episode 的标签拟合 | FoRIS、缓存空间的 INSID3、固定完整掩码候选竞争、RCG。当前云端结果在 1024 工作图上，尚不是原分辨率 CONFIRM600 结论 |
| 基类标签拟合列 | 除上述输入，还用训练折的带标注 episodes 学习读出；测试类别留出 | convctx:layers 的 CONFIRM600 63.33、+3.54 是这一列的结果。输入没有类别 ID，不等于训练没有使用标签 |
| 基础分割模型列 | SAM3 的掩码训练先验、视觉提示或文本词表；是否给真实类别名需另列 | 最新 STATUS 的视觉 exemplar 4000 例 68.86；自动命名与 exemplar 路由 73.14。真实类名为特权对照，不是自动方法 |

零训练的进一步细分也必须明确：闭式公式、离散搜索，与每个测试 episode 内优化分类器不是一回事。RePRI 式适配不使用跨 episode 标签，但在测试时优化分类器参数，不能写成“完全无优化”。这不改变冻结主干的事实。

最新 SAM3 数字来自[仓库 STATUS](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/docs/harness/STATUS.md)，不是本文新运行：

| SAM3 设置 | 标准全表 4000 | CONFIRM600 | fresh915 |
|---|---:|---:|---:|
| exemplar，相对最高 proposal 分数 0.7 保留 | 68.86 | 67.12 | 68.76 |
| exemplar 或自动命名，按置信度路由 | 73.14 [71.85, 74.41] | 72.62 | 73.82 |
| 给真实类别名 | 未运行 | 78.34 | 78.92 |

README 开头保留的 61.09 exemplar、62.83 取并，以及后续的 71.49 路由，是较早配置；不能把它们继续当作当前最强 SAM3 对照。也不能把 DEV241 上约 75.25 的某个路由值当作 4000 例成绩。类别名、词表清洗、路由、整表与子集不同，必须保留各自设置。本文不据跨资源、跨样本的数字宣称超过已发表方法。

## 4 数据与指标的解释顺序

### 4.1 仓库正式协议

仓库主协议为 COCO-20i 1-shot、标准四折、seed 0。训练读出使用图片隔离的 TRAIN2400、DEV241、CONFIRM600，评某折时只用另外三折拟合；DEV 用于选择，冻结后再读确认集。正式方法成绩要求原分辨率、完整管线，包含宿主原有后处理。[AGENTS](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/AGENTS.md)，[README 协议](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

因此，下列指标不能直接相减或合并排名：

- 64×64 patch 多数标签 mIoU 与 1024 工作图像素 mIoU；
- 工作图分辨率与原图分辨率；
- pre-CRF 与完整 native；
- episode IoU 的简单均值与按类别汇总 I/U 后的 class-mIoU；
- DEV、确认集、全表、fresh 子集；
- 用 GT 选择阈值、候选或端点得到的诊断值与可部署输出。

### 4.2 本次云端数据

旧 120 例由 old20、new40、new60 组成，共 67 个实际出现的类别，均为已暴露开发数据。早期按上传进度在 20 或 40 例上观察到的数，承担首个真实样例、故障定位和候选诊断作用，不能替代 DEV241 的方法判断。项目 AGENTS 指出，40 例的区间半宽通常达 3 到 9 点，而关心的真实增量是 2 到 4 点。

当前统计以配对 photograph-component bootstrap 为基础，重复抽到的组必须重复计入类内 I/U，最后才对出现的类别平均。报告同时保留效果、区间、折别变化和 episode 上下变动数。对于样本里只有一个 episode 的类别，重复抽中该 episode 的权重在该类 I/U 中约掉，是这一 class-mIoU 统计量的预期行为；不能通过改成 episode 加权均值来声称修复同一指标。只有在同类包含多个 episode 时错误丢掉重复次数，才是实现错误。本包引用所链接固定报告的统计定义；不同指标、历史 RNG 或 episode 排序产生的区间不互相拼接。[统一失败与状态记录](evidence/01_02/running_idea_ledger.csv)，[云端已重汇总结果](evidence/01_02/cloud_families.json)。

## 5 已完成的基类读出路线

### 5.1 算法实际改变的环节

读出在完整 FoRIS 匹配之后工作。输入为宿主相似度、排名、邻域差和多层参考前景/背景边际等关系图，输出一个加到宿主分数上的修正。零初始化使未训练输出等于宿主；早停按留出类别选择，再重拟合。其优点是能够沿用宿主已有的可靠区域，不必从零学习整个 mask。

CONFIRM600 的 +3.54 四折均正，折增量为 +3.66/+3.94/+2.50/+4.06；249 例上升、174 例下降，仍有 42 例下降超过 10 点。这说明均值提升不等于逐例可靠。[README，原始结果路径为 `results/decision_v1/infer_1_confirm/report.json`](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

最强简单对照同样重要：同结构只看宿主分数的读出有 +2.27，但该数为 patch 级。它表明容量和空间上下文可能解释相当部分收益，不能把 3.54−2.27 当作完整管线中“参考关系特征的净贡献”。需要同分辨率、同输出流程的消融才可作这样的归因。

### 5.2 为何不继续把它包装成普适 class-free 决策

D1 只用 826 对自造样本时，卷积读出出现 −7.40、−27.59、−21.18；线性构造约 +0.30 至 +1.01，区间均跨零。构造中同实例对应、拼接痕迹、目标占比与真实跨实例同类任务不同。它没有证明所有自监督都无效，但否定了这次样本生成与训练配方能复现原监督收益的期待。

COCO 拟合后不重训，PASCAL-Part 299 例为 +0.97 [−1.23, 2.42]，PACO-Part 299 例为 +0.08 [−1.08, 1.45]。这些小点数尚未建立跨域通用决策规律，不能用“四折线性系数很像”替代泛化证据。仓库已将该路线描述为基类标注拟合的 FoRIS 校准。[README 2026-10-04 记录，本文转述已记录结果，未重新训练](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

## 6 当前 DINO 缓存路线的实质变化

### 6.1 固定缓存中可使用和缺失的量

主输入是 q/r、参考软 coverage、可用的 FoRIS score。q/r 是提供时已完成相应预处理的最终层特征，绝大多数 episode 做过去位置偏置处理；应直接按提供的向量使用，不能一概视为原始最后层，更不能从缺失 debiased 标记推断应该再做一次变换。FoRIS pre 可以由 score 重放。native 仅用于完整对照，query truth 只用于评分。旧 packet 的 fg_max/bg_max 可以做可选验证，不能成为主方法隐藏必需输入。

这些缓存没有中间层完整 token、decoder entity token 或新的编码结果。原始 query 邻接 aff_r/aff_d 与导出的 debiased q 邻接并非同一空间。只有在实际读取的量上得到证据，才能声称用了那个信息通道。

### 6.2 从单 seed 转到完整假设

缓存空间的 INSID3 先对查询聚类，选一个参考相关 seed，再按 cross affinity、对 seed 的 intra affinity 和候选占比决定完整区域。新构造保留原分群、原候选资格和固定 0.2 阈值，只枚举每个可用 seed 产生的完整 mask。旧 120 例共 960 个候选；原默认 seed 的 1024 输出逐位复现。

主选择分数为：一个完整 mask 内的查询均值，与“参考 FG 均值减一圈参考背景均值”的方向做点积。查询均值不重新单位化。它不使用 FoRIS score，也不增加类别名、图池或新编码器。改变的是搜索单元：先生成整张查询图上的候选 mask，再选择一份输出。这里“完整”指完整预测输出，不保证候选已对应一个语义正确、范围完整的对象；均值本身也不是新特征。

两例已知错误身份可以被修复，但其选择规则正是在这两例上形成，不能把两例当作独立验证。全 120 相对默认只 +0.451；排除这两例后为 −0.597，新 60 去两例后的 58 例为 +0.145，区间都很宽。[固定敏感性结果](evidence/01_02/hypothesis_sensitivity.json)。

更简单的全参考背景均值对照也能修复两例；主方法相对它仅 +0.643，区间跨零。错配 query 外部上下文在两例上也选对，因此不能把修复解释为正确的局部上下文绑定。[同候选全局背景控制](evidence/01_02/context_global_control.json)。

### 6.3 RCG 与同信息对照仍应并列

以下全部是相同旧 120、相同 1024 工作图指标，不是原分辨率确认集：

| 固定输出 | class-mIoU | 相对 FoRIS native | 95% 区间 | episode 升/降/平 |
|---|---:|---:|---|---|
| FoRIS native | 62.0176 | 0 | — | — |
| RCG | 63.7230 | +1.7054 | [0.4534, 3.2503] | 69/50/1 |
| 同输入 MEAN_CONTROL | 63.5001 | +1.4826 | [0.2484, 3.0366] | 68/51/1 |
| 完整假设 source_contrast_mean | 62.6398 | +0.6223 | [−2.8167, 3.2012] | 45/75/0 |
| 预先保留的 scalar_contrast 对照 | 63.8068 | +1.7893 | [−1.9710, 4.2170] | 48/72/0 |
| 缓存空间 INSID3 默认 | 62.1893 | +0.1717 | [−3.3162, 2.7967] | 47/73/0 |

来源为[当前固定比较报告](evidence/01_02/prior120_locked.json)。scalar_contrast 点数较高，不因此在已经看过结果后替换主臂；RCG 高于简单均值的幅度也远小于它相对 native 的总幅度，必须保留这一控制。表中没有任何一行可据当前证据宣称稳定达成至少 2 点。

## 7 边界路线已有真信号 但仍有两个未解决环节

最新仓库 DEV241 的固定 matte 构造为 59.4439，FoRIS native 为 59.0748，差 +0.3691 [−0.1111, 1.0722]；四折为 +0.5198/+0.6256/+0.0134/+0.3150，121 升、117 降。相对 pre-CRF 的 58.5620，它是 +0.8819 [0.4907, 1.7283]。这是同一输出、更强对照后改变了净收益解释，不能继续把 +0.88 写成超过完整 native 的改进。[README frozen feature-mixture refinement，小报告路径 `results/frozen_matte_vs_native_v1/report.json`](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

固定同一个 unknown 集、同一 alpha-score 融合后，用 GT 只移除已有端点中的污染，得到 61.9057，相对固定 matte +2.4617；若只将同带内 alpha 换成真实覆盖率则到 77.9218。这两个诊断说明端点污染和覆盖率估计均有空间，但都使用标签，不是可部署增益。24 例过滤后没有纯 FG 端点，其中全部在现有网格其他位置仍有纯目标 token，22 例在预测 token mask 内也有纯目标。因此不能归因为“网格根本没有纯 token”。[README fixed-unknown-set attribution，`results/matte_trimap_attribution_v1/report.json`](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/README.md)。

理想 stride16 软覆盖率场达到 95.53，也说明硬块多数标签的界不是整个 16 像素软场的表示界。未知带填 GT 还可能擦掉细长的整块错误对象，所以“带内 oracle”不能完全当成纯几何收益。

## 8 最新计划与未验证状态

最新 [STATUS](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/docs/harness/STATUS.md) 把 GPU 生成更细查询 token 与 layer probe 的 `session_all.sh` 列为待办；[PLAN](https://github.com/misaya-yang/cv_reasearch/blob/0e1dea19716faec8cc4c623ce54c79f8a9df3173/demo_lists/demo9_transductive_ics/PLAN.md) 指明 stride8/2048 view、CPU 读出和后续原分辨率门槛。预检查通过与算法已经有效是两回事，准备好的队列也不等于已经获准启动 GPU。

README 同时把旧 `scripts/pair_bank.py` 的联合编码及中间层构造列在“没有跑就撤回”中。它与新 `layer_probe.py`/`hires_bank.py` 的脚本、输出和问题定义须分开，不能由一个条目推断另一个已经运行。文本头 dino.txt 增加了图文训练信息通道，用户此前质疑其新增组件属性，目前是 held；不能静默算入纯 q/r 列。

本次新 100 的冻结不改变上述 GPU 状态。冻结九臂包括 RCG、MEAN_CONTROL、默认 INSID3、完整假设的五个对照和主选择。共 742 个候选；一例原 eligible bank 为空时按事先合同保留原空默认，不补造候选。还缺 part01 和 part12，共 21 例。没有新 100 的精度、增益或区间可以报告。[冻结与字段证据](evidence/01_02/new100_freeze.json)。

## 9 应保留的研究判断

**已测量**：基类读出在原确认集有效；同类零训练手写修正与多个候选身份规则尚未取得目标幅度；边界 alpha 有小的 pre-CRF 信号；在GT选择诊断下，完整分割假设集合中确有部分好mask，但其可由允许输入正确识别仍未建立；当前 RCG 和简单均值都有小的 native 增量。

**可推断但未证明为定理**：当前主要损失不只在全局面积阈值。错误类别、对象部件组合、参考到查询的范围迁移、污染端点和像素覆盖均可能限制结果。一个机制必须说明自己改变哪一项，并击败直接使用同一信息的简单办法。

**仍未验证**：新 100 加缺失 21 例上的固定输出稳定性，原分辨率完整流程及新的 CONFIRM600 验证，更细 token 的边界收益，以及新增语言通道能否满足信息资源定义。

目前应继续推进的是可在真实错误上增加可用辨别力的推断，而不是从旧 oracle 差距直接宣布某个结构有用。MMD 的具体负结果及参考权重的非特异 oracle 诊断见[失败证据](02_failure_evidence.md)。
