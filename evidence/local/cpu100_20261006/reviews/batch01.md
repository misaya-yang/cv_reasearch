# 第一批方案与接口审查

本记录是专职 reviewer 的阅读审查，不含 reviewer 运行的实验。`accept_for_probe` 不证明自然图像收益、原创性或最终交付质量。所有候选的真实质量仍为 unknown，已实现数由最终源码/完整输出证据另行统计。各控制、阈值、参数和修正版计 0。

## 方法卡结论

| ID | 结论 | 依据与约束 |
|---|---|---|
| cross_image_csls_hubness | accept_for_probe | 参考原子 hub 校正能改变匹配排序；Q 侧密度在角色差中代数抵消已披露。需保留重复真目标被误罚负例、同字典 top3 控制。 |
| cross_image_background_anchor_shift | accept_for_probe | 已知 R 背景对应残差与全图 moments 不同；锚稳定不保证同一 style 适用于 FG。native4 全 fallback，不得声称自然锚有效。 |
| cross_image_free_column_gram_matching | accept_for_probe；固定收益主张已关闭 | 无列容量关系匹配定义完整；其指定旋转正例实际失败，便宜 sorted-profile 控制获胜，目标更偏爱错解，不能称优化误差或继续调参。保留可运行负例与反思。梯度声明/步长合同仍须修清。 |
| RGB01 | accept_for_probe | 明确 RGB 光照 nuisance 而非 DINO 白化；DINO-tied 待判区域与真实 BG anchors 必须分开构造，BG composition/seed 污染是负例。 |
| RGB02 | accept_for_probe | LBP ordinal 观测不是径向功率；同窗口 full/peak/radial power 强控制，不以胜颜色均值就证明序纹理必要。 |
| RGB03 | accept_for_probe | 双谱可观察相同 amplitude 下的相位关系；只声称循环平移和四个 90° 旋转，不虚称任意自然尺度不变。 |
| RGB04 | accept_for_probe | 完整 RGB patch 字典保留多模态/排列，与两个功率均值不同；同 descriptor 类均值与 all-allowed dictionary 控制。 |
| RGB05 | accept_for_probe | DINO/RGB 距离的配对依赖与分离边际不同；边际保持而配对变化见证，影子/场景依赖负例。 |
| RGB06 | needs_revision | 共用 min-cut 不自动重复。必须固定同 signed unary、图、量化、读出和 pairwise mass，并证明联合依赖在两个单模态边分布相同的构造中改变完整输出。仅加 RGB 权重/换 bins 则计 0。 |
| local_001 | accept_for_probe | 局部 centered Gram 结构与 trace/diagonal 有可区分观测；单位可达 equal-diagonal 见证已报告，复制签名错身份失败保留。共同加性不变性仅限保持统一 norm 的特定构造，不能泛化到任意 DINO style。 |
| DR01 | accept_for_probe（代数声明已修） | 已知 R coverage 趋势外推，不等于 DINO 线性解混。精确恢复定理只限未单位潜在模型；首个输入须满足 unit API。 |
| DR02 | accept_for_probe（代数声明已修） | unit pair 下 midpoint 项恒为零；实际只有 R 局部 pair 选取及每 pair 归一化改变线性方向，不得声称新增 midpoint 信息/高阶关系。 |
| DR03 | accept_for_probe | 已知 R 空间最坏组风险与平均风险不同；保护少数部位也可能追逐污染组，固定迭代不等于收敛证书。 |
| DR04 | accept_for_probe | 类内修剪风险与 Huber bounded influence、最坏组保护并非同一固定决策；显式污染比例假设，合法罕见部位被 trim 的负例。 |
| DR05 | accept_for_probe | spatial jackknife 的 Q 特定参考影响可改变偏差修正符号；1.96 不具非 IID 的 95% 保证，all-block 共同错身份负例。 |
| DR06 | accept_for_probe | 全 FG/BG pair weighted rank 非原型 score 单调映射；相似值 pair 不独立，U-statistic 不是 Q 目标概率。 |
| DR07 | accept_for_probe | 已知 MR 连通成分决定 appearance 集合，不迁移 shape；同 K feature-cluster/nearest-token 控制、小组件误标负例。 |
| DR08 | accept_for_probe | 两 R 类 hull 最大间隔方向与旧 Q→hull residual 不同；参考最大间隔不是跨图保证，FW gap 明确、边界污染负例。 |
| inv_adversarial_channel_support | accept_for_probe | exact worst-coordinate removal 是 Q 特定决策，明确只会删；固定坐标与真实稀有方向误删负例，常量 erosion 强控。 |
| inv_reference_mad_winsor | accept_for_probe | 参考箱 clipping 与 inverse-variance whitening 不同；合法 Q style 尾部可被误剪，weighted median/MAD 与坐标依赖披露。 |
| inv_multiscale_feature_consensus | accept_for_probe | 匹配前 unit feature pooling 不一般等于 scalar pooling；同尺度 score-pooling 控制，小物体稀释负例。 |
| inv_spatial_geomedian | accept_for_probe（错误见证已撤回） | robust vector estimator 定义不同；固定 v 且所有邻居同正 margin 时凸包不允许符号反转。修订见证及 Q-only/R-only 归因分开。 |
| inv_local_affine_reconstruction | accept_for_probe（定理范围已修） | 原向量 affine 场的再现定理不能套到任意 unit 球字段；改为可达局部近似/判别坐标见证，独立邻居/partial weights 固定。 |
| inv_local_lowrank_reconstruction | accept_for_probe | 保留 top local variance、不 inverse-scale；空间 affine 与 feature subspace 不同，罕见语义残差被截断负例。 |
| inv_huber_reference_readout | accept_for_probe | episode 内合法 R 标签的 bounded influence 拟合，与旧 scalar Huber graph 不同；同采样/正则 ridge 控制，rare part 被降权负例。 |
| ref_ordinal_copula | accept_for_probe | 坐标经验秩在限定 monotone warp 下不变；不同图组成会改变 CDF，不能称无条件 domain invariance。 |
| ref_local_support_radius | accept_for_probe | local normalized support 与 absolute nearest 不同；同采样 global radius/kernel 控制，纯 R 不足时需明确 inactive，不以 fallback 算实际作用。 |
| ref_affine_tangent_support | reject_duplicate；独立数 0 | 旧 `reference_hull.py:96-102,153-156` 已有 role-specific affine nullspace control；rank2/pole 约束是 family 修订。reviewer 初审漏读控制，已撤销初步放行。 |
| ref_joint_channel_code | accept_for_probe | 三 bit joint code 不同于 degree2/空间三 token；源内 tuple 选择不保证 transfer，parity 只是限定构造。D<8 时通道数取 min，统计支持/未见 bin 记录。 |
| QP01 | accept_for_probe | 完整 query-centered 字典分类，而旧 39/81 是 fine-token hard-landing proxy；不是已证明修复旧反向失败。需 same-dictionary nearest-Q-token vote/pool 和 KDE/pool 控制；K/票规则参数不多计。 |
| QP02 | accept_for_probe | 集合联合命名含 FG/BG prototype 替代竞争，区别旧 row-stochastic Bz 的饱和 vote；最小 R 风险仅保证 R 拟合，不保证 Q 身份。 |
| QP03 | reject_duplicate；独立数 0 | mean 三阶 tensor 到两个均值的距离差严格等于逐 token 三阶线性 score 同窗平均；归一化分母主要改变幅度。窗口没有新增联合信息，与 high-order control 不独立。agent 已关闭且未实现。 |

## 已拦截并修正的错误

1. 未 unit 线性混合/affine 定理冒称 unit DINO 可达；现按精确代数、可达构造、真实未知分层。
2. unit FG/BG pair midpoint 恒消失仍被说成新信息；现简化为真实线性方向。
3. geometric median 被说成能在所有 fixed-v scalar 同正时反转符号；凸包反证已披露。
4. rank2 tangent 重复旧 hull 的 affine 控制；计 0，reviewer 自己的漏审同样留记录。
5. mean 高阶区域 tensor 被说成产生区域联合信息；展开距离差发现它是 pointwise score pooling，计 0。
6. 共用 decoder 被过早当重复；RGB06 改为看新 joint observable 和同合同控制，不用实现名称裁决。

## 共享输入/准备接口审查

`common.py` 已修 dtype、原生 DINO producer、分辨率/通道、checkpoint hash、unit feature、合法 shape、Q physical-valid 一致性及原图/完整 MR hashes。统一 `Result.margin` 为完整原生 signed 网格，`>0` 目标、零背景，不称概率。各方法必须排除 `valid=0` 统计和图边，partial coverage 规则固定。RGB 缺原图/几何时明确 unavailable，不默默 fallback 后声称 RGB 方法已运行。

`prepare_cpu100_native.py` 参考 annotation 路径与 `class+1` 对齐 `data.coco_load`；R mask 由合法参考标注获得，未读 QGT。`run_direct_dino_features.py extract` 两次 CPU forward、FP32、4101×1024、prefix5、模型前后 hash、原图二值 MR 尺寸检查明确；root 确认当前 timm `forward_features` 最后 norm 后，才能把 final-LN 名称视为源代码验证。剩余建议：原图 row hashes 与 producer source-image hashes 交叉绑定；reference geometry/valid 检查；future 异常写 failed receipt，partial manifest 不冒充完成。

这批 32 张卡中，29 项可按固定合同进入 probe，1 项需要修订、2 项重复关闭。**这是方案门数量，不是 29 个已证实有效方法。** 当前 100 项目标仍未完成，后续批次仍须通过同样审查。
