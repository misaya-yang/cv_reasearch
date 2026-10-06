# Pro 五法的独立合同审查

结论：M2 是当前最合适的有限 CPU 准备对象，M3 可做 CPU 合同准备但不是廉价算法；M1/M5 需要新的原生中间状态资产，M4 需要八张干预图的冻结编码，均不能用现有末层缓存假装完成。五法有不同计算对象，但 M2 与现有相邻关系、M3 与现有留出部位布局有实质重叠，不能直接把五个名称计成五项已证实创新。九项现有方法与五项 Pro 方法均无本次真实分割结果。

## 阅读与证据边界

- 先读当前 `AGENTS.md`、`docs/research/PLAN.md`，完整分段读了 Pro 报告第 1–1455 行。下文 `Pro:Lx-Ly` 指 `/Users/yang/Downloads/CVPR2027_five_original_segmentation_methods_20261006.md` 的具体行。
- 审查时本地 HEAD 为 `900c3c941b95d936ba5ae368d21231c1070aae25`，由 `git rev-parse HEAD` 读取；不是 Pro 引用的 `baa6aff...`（Pro:L27-L33）。初始 `git status --short --branch` 有其他代理的 `?? src/ics/methods/reference_hull.py`，本审查未改动它。
- 当前只做本地准备：`PLAN.md:3-6,46-54`；旧开机任务明确不执行。本报告不是新计划或开机授权。协议要求外部建议不替代当前请求，且不得静默改 supplied method：`AGENTS.md:3-8,27-29,36-39`。
- 仅查所需当前方法卡与缓存接口，未扫描完整失败账本、未联网检索文献、未调用 SSH/GPU/冻结模型、未运行大实验。Pro 的文献与历史数字本次均未独立认证；公式评价是本地审查解释。
- `prepared_bundle_01a1100b/entry.json:3-5,36-37` 明确九项是准备且真实例数为 0。对 `evidence/local/research_20261006` 执行 `rg --files --hidden --no-ignore ... | rg '\\.(npz|pt|pth|safetensors)$'` 无输出。此有限目录查找不证明整台电脑无真实缓存；只说明本审查没有绑定可用真实队列。

## 共用合同必须先区分的三件事

1. **信息接口可复用，不等于 Pro v0 数值等价。** 当前加载器只读 `q/r/cov/score`，不读 query truth（`src/ics/experiment.py:65-81`）。现有共享后端用这些字段重建固定 MEAN 并校验原 field（`prepared_cpu_bundle.py:58-70`）。Pro 要求 FP32 原生末层、固定位置 gate 和实际 FoRIS 外部 producer（Pro:L166,L189-L199）。若现有特征经过旧投影/FP16量化而缺少原生末层，就只能明确命名缓存适配版，不能称严格 Pro v0；具体缓存 producer 尚未绑定。
2. **renderer 不同。** Pro 是 FP32 场→bilinear64到1024→阈值→工作二值图bilinear到原 H/W→阈值（Pro:L235-L252）。当前 `experiment.render:84-91` 只完成第一段，共享后端的相邻关系/形状/协方差主行还用 nearest（`prepared_cpu_bundle.py:77-86,114-140`）。Pro 的零关系、零干预只能与同一新 renderer 的控制比较；不能静默替换九项的声明读出，也不能跨原尺寸/1024口径相减。
3. **现有工具不能无检查借用。** 当前 `reference_occupancy.spatial_tree:77-117` 是固定四邻边余弦 single-linkage，不是 Pro M3/M5 的相邻 Ward 合并树。当前 `unit:32-39` 对零向量报错，而 Pro:L183 要把零向量固定为零；当前 `cluster:49-66` 的首点是最接近均值且可能提前少于 K，Pro M2:L398 未明确首个 farthest-point 种子，M3:L542 同样需补成确定值。补明确合同即可，不宜扩成参数搜索。

## M1 同查询上下文续算

**新量与可实现性。** 参考 FG/BG 的真实 block20 状态，在每个查询空间槽位通过同一查询只读 K/V 后缀；终点匹配之外多了共享条件函数响应（Pro:L270-L330）。方程可实现，但必须使用真实 block21–24 的 LN、QK norm、bias、RoPE、LayerScale、MLP、final LN，并按层号对齐原生 K/V（Pro:L283-L317）。正式轨迹排除 3×3 和特殊键，所以自身 probe 不应期待等于 native；native 一致性检查必须暂时关闭正式排除并恢复完整 memory（Pro:L1258-L1259）。

**资源。** 需要 R/Q raw H20、Q 后四块原生 K/V、骨干模块与位置基底，末层 q/r 不足以恢复。当前 `layer_extract.py:15-22,51-61` 输出的是 norm=true 的 block16/24，不能代替 raw block20 或 QKV。Pro 估计两次编码加最多 1.5 图像等效后缀，K/V约128MiB（Pro:L382），不是实测时间；JVP会新增自己的计算/内存，不能默认免费。

**强控制。** 同代表、同 log-mean-exp、同单位 rank 系数的 plain、mean-unit、uniform 是必要完整控制（Pro:L332-L342,364-L376）；mean-unit 不是历史 α=.25 的 MEAN。JVP追平只推翻“非线性必要性”，不会自动推翻共享条件度量。

**边界与可推翻条件。** 无 BG 的负项为0；tiny参考需coverage回退；纯前景探针可能被背景同化，类间信息也可能已在block20丢失。局部键删除不消除间接信息。若 ctx 不胜 plain/mean-unit，或 uniform 追平，或只涨诊断排序而损坏完整输出，则不能保持新类别证据主张（Pro:L178-L185,317,378-L380）。

**与九项重叠。** 借用原型、MEAN图和参考掩码是共用资产；九项目前都读取终点缓存，未定义这个内部只读条件函数（`prepared_cpu_bundle.py:14,58-59`）。不能因它比九项多计算就预测成功。

## M2 参考标签关系推断

**新量。** R/Q联合无标签32角色字典；参考完整软coverage分别估计 FF/FB/BF/BB 四状态的角色对联合分布；查询边上提取cross-ratio的纯二阶标签系数（Pro:L396-L437）。它不同于参考仅FG相邻模式富集。背景标签与吸引/排斥符号是最明确的新增对象。

**公式审查。** 在角色权重是概率单纯形、四状态使用相同0.1 product混合且表正值的前提下，纯因子分解确实使 cross-ratio 为0。对无向边的双方向计数使 P11/P00 对称且 P10=P01转置，因此最终J对称。FP64、数值小量归零、绝对行和归一之后，更新的∞范数收缩上界是λ=.9；熵 Hessian ≥4，交互 Hessian=-4λJ，最小特征值下界≥.4（Pro:L411-L476）。这些代数性质成立，不能推出IoU。

最终 `t=s+2λJ(2z-1)` 保留J=0时的逐值退化，包括同renderer边界；直接输出sigmoid概率会破坏该退化（Pro:L464-L472）。需记录迭代变化与约1e-5解误差界，而不是把1e-6停止量冒称解误差。

**注意控制含义。** pair-independent 用的是状态条件边缘外积，再与全局p_s p_t混合（Pro:L502-L511），一般仍可能产生非零cross-ratio；它不是 J=0 控制。不能把它有收益直接解释为配对结构必要。no-BG或四状态任一无样本的edge type必须置零，不应靠伪计数凭空造关系（Pro:L182,L418-L425）。

**CPU范围与成本。** 若绑定的q/r/cov/原宿主s确有对应信息，全部新量可CPU生成，无新前向。字典估计和两张精确20NN有非小的D维矩阵成本；Pro:L517估5.37GMAC字典、每图17.18GMAC相似度，稀疏迭代较便宜。不应因“后处理”就许诺整队列很快。现有共享图只有Q图，M2还要R边和R/Q字典，不能说已经全缓存。

**强控制与重叠。** 保留signed/zero/positive/absolute/pair-independent/block六行（Pro:L492-L515）。现有 `adjacency/method.json:10-19` 同样读取参考模式相邻关系、去边缘占用影响并用signed reward；源码 `reference_adjacency.py:93-105` 只估FG相邻P与排列null，树上整体奖励。M2的新增是四状态可迁移label odds、R/Q联合角色、多边type、连续signed推断；“关系”“signed”“去边缘”不能分别重复计数。必要时与现有adjacency并列完整成绩，但该跨方法比较不能替代M2自身六个同输入控制。

**推翻条件。** signed不胜positive/absolute，pair-independent追平，或负边主要割真前景并净损伤：撤回对应机制。参考holdout有用而自然Q无用：跨图关系假设失败；收敛稳定不是补救证据（Pro:L515-L519）。

## M3 留出角色预测

**新增对象与重叠。** 一半角色以跨图代价和Gram关系拟合，对另一半先仅按关系指纹分配，再读取其跨图margin验证；局部解释可在多个树节点重复（Pro:L523-L603）。现有constellation已经是两部位拟合、留出其他部位验证、转移完整参考轮廓（`constellation/method.json:5,9-17,20-29`）。M3不能把“留出”“多实例重复”本身当全新；具体区别是外观Gram关系预测、dummy遮挡、局部区域树DP，且不使用二维相似变换/轮廓移植。与现有模式响应协方差也共享二阶组成信息，但后者是同一区域内模式响应总体covariance，没有fit/predict分割（`covariance/method.json:5,9-17`）。

**可实现但合同不能偷换。** Ward树必须实现自己的相邻centroid更新，不能借当前single-linkage树称相同。Hungarian迭代是线性化近似，保存原E_A最好者不等于全局FGW；heldout只留出跨图代价，不是统计独立（Pro:L531,L546-L594）。不足两real anchors固定-2，dummy不可访问query索引；模板不足4角色回退完整native，最多8模板会丢参考实例信息（Pro:L542-L544,593-L605）。

**完整输出限制。** κ=log4097≈8.318，而u≤4，单token叶子V=0，所以不能独立选择微小目标；父节点合并覆盖也可能带背景。这是区域证据开销，不是“无面积先验”。上界剪枝只跳当前整块验证，不能删子树（Pro:L607-L624,L658）。

**CPU可准备，成本可能最高。** 无额外编码，但最多917392次Hungarian调用是实质调度风险（Pro:L652-L660）。九项现有相邻树可复用聚合思想，不可直接复用不同树。先用极小人工树检查DP枚举及dummy，不做全4096树×8模板压力跑；之后真实8例成本核验要等资源/队列被绑定。

**强控制与反证。** 同树/模板/dummy/κ/renderer的all-role、mean、zero必需。heldout不胜all-role或mean、姿态/遮挡损失主导、或小目标损失吞没身份收益，即推翻完整v0（Pro:L634-L650）。现有constellation属于有价值的额外比较，不是同合同的替代控制。

## M4 成对移植环境抑制

**真正新增的量。** 相同canonical参考内容，在当前Q的四位置/尺度条件下的特征变化；按同j跨条件散布估C_env，排除不同j部件散布（Pro:L670-L709）。这与现有 `covariance/method.json:5,9-12` 的“末层参考模式响应、查询实际连通组件内协方差”不是同一个统计对象；与现有二次核/先验迁移共享每例参考监督拟合，但并非它们换名称（`quadratic/method.json:5-14`、`prior_shift/method.json:5-16`）。

**公式与限制。** C_env PSD、λ_R>0使线性解唯一；正类/负类均值中点与间距归一化在δ非退化时可实现，但不是自然Q校准。相同j残差跨t和为0，所以T=4,m≤32时C_env秩至多2m(T-1)≤192；Pro的256列Woodbury上界合法但可更紧（Pro:L687-L718,L773）。Potts权重非负，min-cut全局解仅针对给定能量（Pro:L720-L745）。

**资产与额外资源。** 有RGB/参考mask即可CPU准备最大空矩形、四条件正负移植、canonical点与标签映射；不能从现有q/r推出八张edited-Q特征。正常1原Q+8干预Q=9编码，回退懒读R可10（Pro:L674-L683,L773-L775）。ref-canvas强控制另需八张R画布编码，论文CE也有自己成本；“冻结”不等于“不适配”或“不增加编码”。

**边缘情况。** 全BG空参考直接空；无足够背景矩形、边<32、有效FGcanonical点<4走native几何回退；d≤1e-8走统计回退。Q未改像素始终无标签。FG自然矩形含参考背景，接缝距离不排除全局shortcut，也不是纯类别do干预（Pro:L178-L182,L677-L683,L718,L735）。

**强控制与反证。** 同八图mean、同λ的class-LDA、ref-canvas及论文CE是最小机制对照（Pro:L757-L771）。若只分好贴入物、自然Q无益，或mean/LDA/CE追平，或换R画布一样好，环境抑制主张应关闭。廉价后处理不能抵消9编码成本。

## M5 局部消息外推

**新增量。** 四个完整Ward分区对全Q覆盖，末四层固定原生QK下改变outside→inside的value质量分配，测量μ(1)与μ(.75)并限幅外推（Pro:L779-L868）。九项现有方法没有内部value-message轨迹；与颜色路径/Huber的“抑制扩散”只有目标层面的相似，信息与算子不同。

**实现与数学边界。** 固定QK、内部V重算、外部V只读，才能使β=1按层复现native。换成重算QK或logits缩放就改了方法。路径同时削弱外部与增加内部质量，导数不是纯删背景；FG与BG也必须走相同路径。先池化未经token单位化的final-LN投影向量，再差分/限幅/单位化（Pro:L795-L848）。μ沿路径仿射且限幅不触发时截距解释成立，一般有限差分外推不等于无背景语义（Pro:L870-L874）。

**资产/成本。** raw H20及R/Q后四层原生Q/K/V，不能从末层缓存恢复；估缓存384MiB，新增5N×4层约.833图像等效，但每ROI全图读取、60分支调度、attention重构都未计入时延保证（Pro:L908）。没有这些资产时，当前只能准备ROI定义和纯代数合同，不可报告nativeβ=1检查通过。

**强控制。** endpoint/hard-channel/native-region/no-intervention是首轮；普通native-region不是零干预。论文另需无内部重分配缩放、真正重算QK masked-attention、同token预算crop/gray、response/mass和新分支FG/BG身份交换（Pro:L882-L906）。这些部分会新添续算/编码成本，应按实际预算，不能把控件写入表便视为已有资产。

**边界/反证。** ROI全覆盖不是纯目标覆盖；混合区域、语义依赖上下文、参考与Q响应不同，均可严重毁坏正确场。修正系数1且δ=.07时增减约.762，不是细边界微调。若endpoint/hard/native-region/objectness足以解释收益，或正常例净损伤，则收缩或撤回外推必要性，不扫β/层数继续救名称（Pro:L868,L874,L906）。

## 对现有九项的整体映射

| 现有方法 | Pro最接近项 | 实质重叠与区别 |
|---|---|---|
| 参考相邻关系 | M2 | 都用参考模式pair、去一元/占用效应；M2增加四标签状态及排斥关系，解码不同 |
| Huber图 | M2/M5（目的相近） | 都可能限制错误传播，但Huber只改既有图力，没有新增参考关系或神经内部量 |
| 颜色瓶颈 | 无直接重复 | 查询RGB局部min/max路径，五法均没有这个光度量 |
| 参考部位布局 | M3 | 已有fit/heldout、多实例；M3以Gram角色指纹取代二维pose，输出范围来自树而非参考轮廓 |
| 参考整体形状 | M3（区域完整性） | shape用参考轮廓矩与局部搜索；M3用外观角色预测和Ward DP |
| 模式响应协方差 | M3/M4 | 都二阶，但covariance是对象内响应散布，M3是角色关系预测，M4是干预环境散布 |
| 查询重复部件 | M3（可重复解释） | recurrence用高置信query种子star学外观；M3不从伪标签扩展prototype，不限制只有高分节点 |
| 得分分布/查询比例 | M2/M4（FG/BG统计） | 一维label-shift与全局prior不是边label odds，也不是干预环境统计 |
| 二次核分类 | M4（单例闭式拟合） | polynomial source anchors与环境协方差抑制方向不同；“闭式”“参考监督”均非独立创新 |

九项身份出处为 `PLAN.md:7-46` 及 `prepared_bundle_01a1100b/entry.json:7,14-21`；逐项区别另见上述具体卡片引用。这个映射不证明文献原创性，也不改变九项计数。

## 最少的可执行下一动作（建议，不是已授权新任务）

1. 当前如要选一个新增CPU准备项，先把M2六行写成独立合同；仅使用人工可实现unit features核验factorized-P→J=0、对称/行和、收縮、t=s及相同renderer逐值退化。先解决首seed、零向量、缓存producer和renderer身份四个具体缺口即可；不运行真队列，不据此称完成新方法实测。
2. M3仅补相邻Ward/dummy/小树DP合同与调用数估算。若用当前single-linkage替代，应显式命名新适配版，不能报Pro M3 v0；不直接准备近百万调用的全图实现作为CPU高效承诺。
3. M1/M5登记raw-H20、native-QKV、源码/权重/基底缺口；M4登记RGB/canonical点与八干预编码缺口。只留资产定义，不触发新DINO前向，不用CPU仿真称真实native验证。

真实结果获得后的最小决定尺度：M2首先要胜同信息强控制且净增删不反向；M3首先要胜mean/all-role并抵消小目标损失；M1/M4/M5只能在新资产获授权后讨论自然Q完整收益。所有项的数值/CPU准备与分割结论分开。Pro建议E1/E2完整执行（Pro:L949-L1008,L1356-L1360）不自动取得当前资源权限。
