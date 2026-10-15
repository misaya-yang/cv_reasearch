# 单参考分割研究历史复盘

截至 2026 年 10 月 11 日，按 America/New_York 归档。目标是冻结 DINOv3、单张参考 RGB 和完整参考 Mask、单张查询 RGB，形成有实质贡献且在同条件完整输出上稳定胜过 FoRIS 与已有强控制的统一方法。本文回溯早期设置、完整 COCO 结果、批量方法、M4 多域实验和最近强组合修复。

结论需要分三层。已有完整 COCO 增益、LVIS 局部观察与拒识增益、PACO 参考读出增益，都是实测成果；统一跨数据集方法仍未交付；反复失败由研究决策、评价解释和具体推断缺陷共同造成，不能统一归结为最新的负样本遗漏或 DINO 表征无信息。

最反复的研究失误是：已有正结果尚未拆清来源，就用未经验证的代理目标替代任务判别；新方法相对弱控制改好一部分，却没有保住已有强方法承担的功能；失败后更换表示、求解器或融合形式，仍用相近的参考自拟合和预测共识继续裁决同一批争议区域。

## 历史材料覆盖

本轮提取并核对六个本地主聊天和 32 个历史子代理会话的真人要求、结果消息和关键执行转折，共 38 份本地会话文件。重复的等待消息和工具命令不会被计成独立实验。较早结果从清理前提交的 1916 行 RESULTS 和旧 LESSONS 补回，CPU100 与 Astra300 的 1310 行方法账本按实现、真实输出、质量、控制和未测项逐类检查；下列 40 行是研究路线归类，不是 40 个独立方法或独立数据确认。

更早的两个核心原聊天“检查实验运行状态”和“查明五天持续失败的原因”当前读取失败，durable host 不可用；相应历史判断来自保存的 Git 复盘和结果账本。部分旧远端数组及精确源码已退役，所以“历史报告记载”与本轮直接读取的评分 JSON 分开保留。38 会话与本文来源的身份和内容校验见 [coverage.json](../../evidence/local/history_failure_review_20261011/coverage.json)。本轮没有启动科学实验，也没有重算图像指标。

## 四十条研究路线及实际结论

各行保留自己的样本、资源条件、渲染阶段和统计口径。只有同一行明确匹配的比较可以作差。AUC、每任务均值 IoU、按类汇总 mIoU、道路总 I/U 均不互换。


### 早期设置与读出

| 路线 | 已执行及关键读数 | 成立的认识与失败边界 | 来源 |
|---|---|---|---|
| 1 图池与伪参考 | 四折曾报 +6.1；官方基线逐例仅 43% 一致；单折朴素自训练已解释大头。 | 新增图池改变信息条件；基线身份不足，不能作为当前单参考贡献。 | [S01](#来源索引) |
| 2 基类监督决策读出 | CONFIRM600 63.33 vs FoRIS59.78，+3.54；同结构只看分数已有 +2.27。 | 真实正结果；使用基类标签拟合，资源条件与当前免数据集训练主线不同。 | [S01](#来源索引) |
| 3 固定 17 常数公式 | 同 CONFIRM600 61.45，+1.66；另一个 F4 七臂 DEV241 均未过线。 | 固定公式存在正收益，不能把不同 F4 配方混成同一版，也未形成十域优势。 | [S01](#来源索引) |
| 4 SAM3 视觉 文本 路由 | 后期 CONFIRM600 路由71.49 vs FoRIS59.78，+11.71；早期视觉61.09不能代表后期最好配置。 | 额外模型和词表改变条件；命名 错候选和重复提示未被可靠解决，当前纯 DINO 条件不能挪用该成绩。 | [S01](#来源索引) |
| 5 中间层 CLS 与旧 QK | DEV40 QK58.70 vs65.16；CLS1200 主 AUC .434，natural reference 对照有有限正信号。 | 特征类型换了，困难目标与干扰仍未可靠分开；若使用 GT crop，只算诊断。 | [S01](#来源索引) [S03](#来源索引) |
| 6 阈值 面积 混合比例估计 | 4000 真值逐图选阈值 +8.91；描述量 boosted tree −.67，六档面积 +.29；EM 面积估计相关 .80 却退4.08点。 | 原场面积估计与原场错误同向；相关性和 oracle 都不能直接兑换成推理收益。 | [S01](#来源索引) |
| 7 查询候选 树 分组 锚点 | Ward 三节点 GT oracle80.95，合法选择仍低 RCG；源验证在43坏例20/43选对，整体184/241低原189/241。 | 候选能力与选对候选分开；完整目标存在于候选族，并不表示单参考能选出。 | [S01](#来源索引) |
| 8 白化 中心化 域对齐 自训练 | 查询白化可把 AUC 从 .956 压到 .72 以下；query ridge 自训练未改善原排序。 | 统计变化可能压掉类别方向；同图自一致和全图风格修正不足以保证任务迁移。 | [S01](#来源索引) |
| 9 matting 细边界混合读出 | 原图 DEV241 matte59.513 vs FoRIS59.122，+.391区间跨0；delete-only59.554更高。 | 存在混合覆盖信号，完整增量很小，粗身份和合法端点仍是前提。 | [S01](#来源索引) |

### 图读出与批量方法

| 路线 | 已执行及关键读数 | 成立的认识与失败边界 | 来源 |
|---|---|---|---|
| 10 RCG 与 MEAN | COCO4000 CLI：FoRIS60.9317，RCG62.3337，MEAN62.5130；fresh600 图单独+2.638，参考排序在图后额外+.655。 | 真实完整 COCO 增益；强前端 图和排序交互必须保留，不能把全收益归一个新 guide。 | [S01](#来源索引) [S04](#来源索引) |
| 11 四移位细读出与残差 | 4000 fine16 62.7091、fine64 62.6516；固定残差62.8445 vs MEAN +.3316。 | 实际细观察有价值；16/64是图强度，同为128²细网格，残差尚未稳定胜fine16。 | [S01](#来源索引) [S05](#来源索引) [S09](#来源索引) |
| 12 A B 增删优化与条件估价 | 5257配方 joint2低直接完整选择.490；1200 directional joint低single-family.037；2400新1200收益降到+1.015且低MEAN.270。 | 大头是重建已有强输出；复杂优化和估价没有持续胜过同信息简单选择。 | [S01](#来源索引) |
| 13 12 源及 13 源穷举组合 | 12源每折搜索50,331,648配方，留折62.9647 vs60.9317；13源另加标签拟合sizecut。 | 开发增益成立；12源是类别折留出并有照片共享；13源底层切点另有标签曝光，不是统一新算法确认。 | [S04](#来源索引) [S09](#来源索引) [S43](#来源索引) |
| 14 RCG2 冻结跨数据集 | 原图 COCO BCD full+1.86，LVIS599+2.50；PASCAL+.96区间跨0，PACO+.05、SUIM−.34。 | COCO上有效的细读出和切点没有统一迁移；多域不是同一种范围错误。 | [S01](#来源索引) |
| 15 CPU100 原生小批与 200 筛选 | 首30和第二批22新增有4例测量；12项扩200。QP01/02小批+6.8到200近0，RGB02+3.89变−4.57；Huber对ridge+.711保留。 | 合成可达 小批正例与表达容量不能替代跨图效果；多数复杂头输完整简单头。 | [S03](#来源索引) [S11](#来源索引) |
| 16 Astra300 与 Pro30 | 10月7日旧快照 Astra0/300评分，Pro9/30评分且都低B_R46.575；同600 FoRIS61.563、INSID3 56.381。 | 实现数量和资格数量被当成主线进展；弱裸头实验常无法回答完整Strong问题。 | [S02](#来源索引) [S03](#来源索引) [S01](#来源索引) |
| 17 六级证据 L4 与 Pro修复 | L4直接token53.229比s2+1.50，图后+.484区间跨0；mode-return图−3.534；attention隔离仅zero-control，完整臂未测。 | 是简化线索与限定负结果，尚非完整原创分割，也不能把未测假说写成失败。 | [S01](#来源索引) |

### M4 与真实局部观察

| 路线 | 已执行及关键读数 | 成立的认识与失败边界 | 来源 |
|---|---|---|---|
| 18 600探索及6000前提长队列 | 五域各120原图探索，COCO/LVIS/PASCAL/PACO/SUIM的RCGfine差为+1.466/−.722/−.902/+.850/+.290；6000队列停954条（COCO952），官方COCO前缀停533。 | 过早绑定编码 fine CRF和96表示分支，未接缓存复用；规模和准备吞没决策速度。 | [S44](#来源索引) [S48](#来源索引) |
| 19 LVIS1400 原子阶段与 FG锚定 | CLI FoRIS45.948、MEAN45.986；pilot100 FG-anchor+2.343，另一200却低FoRIS.37。 | 早期高绝对分数没有同批基线；小批校准收益未稳定，不能等待扩大单臂来证明胜出。 | [S06](#来源索引) [S20](#来源索引) |
| 20 局部rank 图保真 参考保护 | 同LVIS100 CLI1024：rank47.187低amplitude47.473；条件图47.701仅追平FoRIS47.704；保真46.269，零交集7降1却新增FP反超收益。 | 召回改善仍须付误报价格；同预算简单控制解释了很多增量。 | [S06](#来源索引) [S09](#来源索引) |
| 21 参考擦除 Fisher O12 | 同LVIS100 CLI1024：erase46.560、Fisher46.427、O12 46.616，均低FoRIS47.704；Fisher参考分离好3.38倍而Q前景AUC .774→.739。 | 合法参考拟合及新观察不自动迁移；错误主要是删真未回本，不能笼统否定整层或全部参考信息。 | [S06](#来源索引) |
| 22 Z1 Z2 放大与参考裁剪 | LVIS1400原图FoRIS46.105→Z2 47.994；同200真实query重编码比插值+2.55。 | 真实尺度和上下文观察有独立价值；参考裁剪 同图上下文 查询细节仍须区分。 | [S45](#来源索引) [S46](#来源索引) |
| 23 原九窗 S 与查询上下文门控 | 同LVIS600 S52.005 vsFoRIS47.779；无gate多数34.129；早期774个无目标局部窗全输出前景。 | 真正成功在真实观察与同场景拒识结合；独立局部minmax不能提供可靠缺目标零点。 | [S07](#来源索引) [S29](#来源索引) [S47](#来源索引) |
| 24 九窗效率 共享 三窗 精确早停 | Deep100真实快九窗37.344、共享27.55、三窗32.74；前5例完整耗时约22.49/9.42/10.57秒，精确早停平均8.87窗。 | 省编码会改变有效信息；仅省后处理或1.44%观察量不能解决主要冷推理成本。 | [S34](#来源索引) [S07](#来源索引) |
| 25 PACO快九窗与少数支持恢复 | PACO600 FoRIS38.461→未筛多数29.539→筛后34.971；恢复M加B与接受支持37.492，仍低简单并集37.776。 | 部件父物体混合 真实支持擦除和错误guide否决并存；PACO慢版未测，不能混判。 | [S07](#来源索引) [S10](#来源索引) |

### 独立任务与完整组合

| 路线 | 已执行及关键读数 | 成立的认识与失败边界 | 来源 |
|---|---|---|---|
| 26 整图 anchored角色竞争与 APD ridge | 旧whole PACO39.640、COCO60.896；global ridge PACO40.775 vsFoRIS+2.314、matched-uniform+.841，但COCO低FoRIS.899。 | 背景竞争有真实局部收益；合法reference零点和阈值仍未跨域迁移，不能按域挑head。 | [S08](#来源索引) [S10](#来源索引) [S32](#来源索引) |
| 27 联合几何 R1 G 与 BP | Deep100直接12.631→几何12.597；PACO100 29.237→29.275；几何额外约6.13秒每例。 | 大损失先在基础unary出现；数学修补 塌缩约束与关系收敛没有提供已证有效身份判据。 | [S06](#来源索引) |
| 28 query基底 scene600 与 copula200 | scene+CRF对四域FoRIS全负；copula whole AUC Deep.518/PACO.461，natural NLL改善却balanced NLL恶化。 | 查询组织与关系表完整可计算，但主导背景 改面积与弱排序不能冒充任务判断增量。 | [S13](#来源索引) [S14](#来源索引) |
| 29 reference-focus200 与分布迁移 | PACO真focus比插值+3.079但39.057低FoRIS45.630；Deep真focus−1.359；600诊断两角色均值迁移，APD提高排序却恶化面积估计。 | 新参考细节确有局部价值；不是只修一个全局offset或prior就能可靠校准。 | [S15](#来源索引) [S16](#来源索引) |
| 30 联合编码 APD 与真实Mask提示 | COCO150候选内backAUC .6837→.6831，LVIS150 .6855→.6580；共同APD及20加20 maskcue未稳定胜出。 | 全局上下文联合前向本身不等于任务条件判别；三例ridge收益主要来自一例，扩大后未保持。 | [S12](#来源索引) |
| 31 source-span 反向重建 与 nullspace17 | R留出AUC约.99、R到Q .949/.931/.881；Q标签共同oracle约.99；安全null更新33.190仅比原R33.124高.066，FoRIS42.634。 | 表征可表达与单参考能识别是不同问题；循环条件会惩罚真解，保护旧R判断也可能压没有效更新。 | [S17](#来源索引) [S35](#来源索引) |
| 32 物理先验40 与跨view绑定40 | CCA式物理先验Deep13.471/PACO26.494，低reference-only；绑定16.665/23.759，低FoRIS29.289/44.916。 | 真实绑定对弱空模型有作用，仍不足以可靠迁移任务；共同几何或二阶统计也没有确定Q零点。 | [S18](#来源索引) [S19](#来源索引) |
| 33 末MLP任务梯度核 与末块QK | 120任务核IoU41.172/32.661/46.205 vsFoRIS55.854/47.682/60.239；最佳GT阈值仍低去位置O24；末块QK也未胜O24。 | 确实读取新的内部量，但新增量并未提高有效排序；不是把核 求导 正确hook当成果。 | [S20](#来源索引) [S21](#来源索引) [S41](#来源索引) |
| 34 Mask-only60 与方向度量干预 | 去scalar gate118/120输出相同；task-axis PACO+3.124、SUIM−1.987；PACO原排序加候选分布+3.042，换排序仅+.126。 | 定位了无效通道和读出效应；其它参考路径仍响应，不构成全模型任务信息消失的统一病因。 | [S12](#来源索引) [S25](#来源索引) |
| 35 INSID3 图 Soft cross 类平衡与零票 | 真实图提高任务切换AUC却未提高唯一正确覆盖；类平衡三域41.120/45.811/54.814均低FoRIS；大量零FG票节点仍固定BG。 | 需要支持 选择和绝对读出共同工作；无支持不能当真背景，图可保任务响应但输出仍错。 | [S20](#来源索引) [S21](#来源索引) |
| 36 共识900 2380 及原生两域4000 | N在五域开发原图高FoRIS .53到1.22；N对同补点数K仅.002到.013；PASCAL CLI+ .889、SUIM+.450。 | 完整组合有可重复开发价值；多数新增收益仍可被简单预算排序解释，关键强控和稳定跨域未全面胜出。 | [S25](#来源索引) [S37](#来源索引) [S38](#来源索引) |
| 37 CRF位置2380 与参考校准6380 | 二次CRF伤Deep；单次末CRF缓解但SUIM40低同F MEAN3.104；LOO使LVIS对旧N+.443，仍同full-reference且nativeSUIM−.115。 | 修复质量链和条件支持是必要工程及有限方法改进，但没有建立新增任务判据的独有收益。 | [S20](#来源索引) [S22](#来源索引) |
| 38 九窗1300 与 Pro场重建200 | 完整组合LVIS49.749低S52.005、Deep36.426低37.344、PACO38.349低38.461；Pro200 Deep11.499/PACO29.658。 | 真实观察仍有价值；保粗场 重建一致和新增裸头没保住旧强链，当前配方关闭。 | [S23](#来源索引) [S24](#来源索引) |
| 39 两条完整6200 anchor与background裁决 | 第一条CLI对FoRIS−.257/−.075/−2.554/+.737/+.663；背景版+.335/+.085/−1.749/+.675/+.692。 | 6184可用anchor全过，污染筛查无有效拒绝；多模BG加进去仍无法保住LVIS，不能换控制当赢家。 | [S26](#来源索引) [S27](#来源索引) |
| 40 最近五个600候选与强S证据追踪 | 原图47.685/49.877/47.812/46.207/51.592，均低S52.005；最新查到两个旧head的S负证据过滤。 | 拟合 几何 邻接共识仍会重新放行正确否决；99.70%暴露缺口只限已核判别器，修正收益未验证。 | [S28](#来源索引) [S29](#来源索引) [S30](#来源索引) [S40](#来源索引) [S42](#来源索引) [S49](#来源索引) [S50](#来源索引) |

## 已有成果的可用价值

**完整 COCO 增益应当保留。** 相同公开4000、CLI1024：FoRIS60.931741，RCG62.333671，MEAN62.512972，fine16 62.709100，固定标量残差62.844537，12源留折配方62.964699。它们证明我们有完整输出增益，最高这行对FoRIS+2.032958；并不证明十域通用，也不是无开发曝光的新确认。12源相对fine16只有+.255599，相对固定残差+.120162的历史区间跨零。新增组件的贡献不能把相对FoRIS的大头全部归过去。[S04](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/report.json)[S05](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.json)[S09](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/03_graph_readout/FINDINGS.md)

**真实观察与查询上下文是另一项强成果。** LVIS600原图S52.005138，对同批FoRIS47.779371、Z2 50.234597、同保留区域数量控制49.744750，分别+4.225767、+1.770541、+2.260387。它同时使用参考裁剪、真实局部查询和上下文筛选。未经区域gate的局部多数仅34.128915，gate追回17.876223点；“多看局部图”不能独自解释结果。Deep100真实快九窗37.344028对FoRIS29.377542也成立。[S29](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_STAGE600_RESULTS_20261011.md)[S32](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/protocol_inventory.md)

**PACO的背景竞争和参考方向确有增量。** 原whole竞争39.640268比FoRIS38.460868高1.179399，净FP下降的大头在父物体内部；global ridge40.774944比FoRIS高2.314076、比MEAN高1.422899、比匹配uniform剪除高.840716。它在COCO200却低FoRIS.899078，因此是有价值的局部成果，尚未成为统一选择依据。[S08](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/02_task_identity/FINDINGS.md)[S10](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/04_dataset_differences/FINDINGS.md)[S32](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/protocol_inventory.md)

**五域共识组合已经有完整开发输出。** N在原图五域均高FoRIS，COCO+.781767、PACO+1.223618、LVIS+.525773、PASCAL+.898591、SUIM+.435092。这个正结果不能抹掉；但N相对同补点数K只有+.012952、+.006079、+.002427、+.008082、+.012352。N仍低于PACO/PASCAL的R0和LVIS的C，SUIM的照片关联组区间限制稳定性。主要收益可由覆盖取舍解释，新增参考选点贡献仍小。[S25](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/FIVE_DATASET_UNIFIED_MECHANISM_20261011.md)[S38](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/NATIVE4000_BUDGET_SELECTION.md)

**工程资产已经支持快速科学验证。** 原始O24及真实局部观察、完整终点Mask、源输入身份、缓存复用、双帧评价、四类编辑和有界临时缓存均可复用。原生PASCAL/SUIM各2000例所需4238个唯一输入已齐；不是49,318官方五域draw都已编码。缓存速度是后端增量成本，冷输入仍要支付对应真实观察成本。[S20](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SOL_VALIDATION_20261010.md)[S32](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/protocol_inventory.md)

## 反复失败的研究决策

**一是交付物曾被改写。** 旧核心会话复盘记载：五个独立方法变为一套区域法的五个控制；Pro的masked-query原方案改成自然crop后，原构造被移入控制。这类变化使完成实施计划仍不能回答用户原问题。当前历史复盘采用它们作为旧记录，不能因为原聊天读取失败补造更多细节。[S02](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/docs/research/LESSONS.md)

**二是方法数量和工程资格替代完整质量。** 10月6至7日旧复盘记录60多次提交，Astra300评分0、Pro30评分9且全低自己B_R。CPU100的77项是候选资格，含真实小批、旧方法和未运行探针，未完成100，更不代表77有效方法。数值反例、源代码审查和合成完整Mask解决实现问题，不能替代自然图完整成绩。[S02](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/docs/research/LESSONS.md)[S03](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md)

**三是开跑尺度与决定问题不匹配。** 10月8日先做6000内部表示前提测量，实际小批计时每例约15–16秒，含配对编码、原CRF、四次shift和96分支；预估约26小时以上，原始缓存最初未落盘，接通缓存后旧入口仍未复用。同期又启动49,318 draw完整基线。用户纠正后分别停在954条开发结果（COCO952、PASCAL/PACO各1条）和533条官方COCO前缀。它们已产生保留资产，但此前调度延误了小批可判定的研究问题。与此相反，后期原600完整候选生产只需90至441秒，所以一直失败不能归因于没有足够M4算力。来源为准备会话的实际执行消息、PLAN中的历史停止收据与初筛记录。[S44](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/PLAN.md)[S20](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SOL_VALIDATION_20261010.md)[S28](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS_RESEARCH_STATUS_20261011.md)

**四是新模块的对照常未接近真正目标。** 原生prototype42.58或B_R46.58与完整FoRIS不是同一能力水平。复杂模块胜原型，可能只重复了logistic/ridge已提供的功能；复杂组合胜裸头，可能只重建了原强输出。M04胜随机配对+.662却输简单均值1.194，DR03/04/08胜prototype却输平均logistic，N胜父组合却几乎等于同补数K，是同一模式的不同实例。[S03](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md)[S11](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/05_reference_readouts/FINDINGS.md)[S38](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/NATIVE4000_BUDGET_SELECTION.md)

**五是已有失败多次没有约束下一方案。** 旧local002不胜同候选unary，坐标打乱仍相同；后来的R1-G仍从局部观察有效跳到几何可解决身份，最后主要损失发生在几何之前。reference Fisher已证明源分离改善而Q排序变差，之后仍反复以源fit、局部一致、几何低能量来作新构造动机。相关思路可以重试，但需要不同的、能改变实际错误的依据；当前记录没有证明这些共同前提成立。[S06](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/EXISTING_EVIDENCE_AUDIT_20261010.md)[S09](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/03_graph_readout/FINDINGS.md)[S42](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_CONSTELLATION600_RESULTS_20261011.md)

**六是执行和状态维护反复吞没研究衔接。** 旧600工程有极慢首例、漏C++源导致全批失败、control遗漏参数重跑主方法、zombie判定阻断评分；M4时期有导入冲突、scorer变量覆盖、数值ties修复。开发会话还出现五次响应流失败，续跑会话两次remote compact失败；这些只能确认响应或压缩失败，不能归因具体网络或GPU。多个阶段结束后停在“等Astra”和汇报上，与用户持续研究要求不一致。健康后台计算的等待是合理执行，不能把等待本身都说成无效；问题是错误规模、重复准备和未接续真正决定性的比较。[S02](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/docs/research/LESSONS.md)[S03](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md)[S20](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SOL_VALIDATION_20261010.md)[S12](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS55_STATUS_20261010.md)

旧账本还记载一次误删约10.067GB可复用特征，另有10月10日授权清理O12和擦除专属raw。这些不是同一种操作；前者超出了当时清理范围，后者留有授权和收据。当前旧路径存在于报告不等于仍可重放，不能把所缺资产误说成算法或数据已完全无用。[S01](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md)[S11](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/05_reference_readouts/FINDINGS.md)

## 科学问题的具体卡点

**任务表示能读出，合法单参考还不能选中可靠解释。** 三域reference内AUC约.99，reference到Q降为.9487/.9306/.8806；使用Q标签的共同oracle约.9946/.9916/.9875。oracle说明有限族还存在更好的读法，不能部署，也不是所有单参考算法不可能的证明。高源fit、学习更多源细节、保留源判决和物理一致性，都没有自动选出这条Q规则。[S17](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_TRANSFER_CORE_20261010.md)[S18](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_PHYSICAL_PRIOR_20261010.md)[S35](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_NULL_RANK_PROBE_20261010.md)

**真正困难的是当前要改动的区域。** LVIS1400同一guide全图AUC .954593，FoRIS前景内部 .743454；争议恢复区还可低到.55至.61。全图易背景带来的高AUC，与可靠否决相似误报、恢复被误删目标是不同问题。密度秩、source自信、多个预测同意等量即使总体相关，也不一定在这批争议位置上有足够增量。[S06](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/EXISTING_EVIDENCE_AUDIT_20261010.md)[S08](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/02_task_identity/FINDINGS.md)[S11](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/05_reference_readouts/FINDINGS.md)

**排序、面积零点与完整读出都有失配。** 三域Mask-only的任务轴，PACO+3.780中的主要收益可由原排序加候选分布+3.042解释，而候选排序加原分布仅+.126；tangent即使拿GT最佳阈值，也低于同输入去位置O24的排序上界。两种失效并存，不能统一说只缺阈值。reference两角色分布到Q同时变动，大量BG的小正移足以造成大面积误报；源零点不是Q后验。[S12](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS55_STATUS_20261010.md)[S16](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/REFERENCE_QUERY_MIXTURE600_RESULTS.md)[S41](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/TANGENT_THRESHOLD_CEILING.md)

**局部观察既发现目标，也制造缺目标误报；融合又能擦掉有效支持。** 独立局部minmax让空窗口也得到前景。PACO接受混合组件只有约6.42%的组件数，却占65.77%提议面积；79.19%的接受FP发生量在父物体内。21.27%的目标像素已有局部支持却未过未经gate的多数。与此同时LVIS的gate确实清除大量误报，不能整体取消。需要同一个合法判据区分“局部纠正旧guide”与“局部扩张错身份”；目前还没有这样的规则。[S07](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/01_multi_observation/FINDINGS.md)[S10](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/04_dataset_differences/FINDINGS.md)[S29](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_STAGE600_RESULTS_20261011.md)

**细结构确实有特殊机制，但不能包办所有数据集。** Fundus200 FoRIS完整19.126859；seed prior把22.036686拉到16.211243，伤199/200个任务，并为新增假前景提供大量正boost。半数血管面积被实际参考硬标签分进BG，局部像素信号仍有可迁移性。PACO的像素外观迁移近随机，而部件粗格oracle已较高，不能套用同一分辨率解释。[S31](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/FUNDUS_MECHANISM.md)[S33](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/extended_foris.md)

**最强S的具体证据没有充分进入后续判据。** 两个旧head用FG=S与B与G与R0、BG=四者一致BG，把S正确否决而弱head报FG的争议像素设为无标签。原图2439283正确删B假前景被BG角色定义排除；CLI1024的9305101正确否决像素中，99.7032%所在native格子未被双角色判别器128个BG样本选中。这个范围不覆盖所有后续方法；graph用全area种子，不能套128漏采结论。S也误删675694原图真像素，直接把全S背景都做硬负类可能继续伤害目标。本轮只是确认输入机制，没有修正并测收益。[S30](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_NEGATIVE_EXPOSURE600_20261011.md)

**S误删的最早可见阶段也已查清。** 在S删除FoRIS正确前景的675694原图像素中，无局部支持占36.75%，有支持但原始覆盖多数失败占61.18%，多数成立后gate改变占2.07%。gate全局则删113193真、4475364假，将34.129推到52.005。这三个桶包含渲染和路径重叠，只定位最早可见失败阶段；不能归责encoder，也不证明保留平票就会涨分。已试宽恢复50.682低于52.005，说明找到误删预算仍不等于能安全取回。[S29](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_STAGE600_RESULTS_20261011.md)

**最新交叉核对进一步限制了“补负样本”和“保gate”的解释。** CLI1024中无native背景质量的位置，图法恢复真78.65%、假50.40%，双角色恢复真93.03%、假69.82%；缺负证据与有效恢复同样相关，不能只凭错误恢复关联就判定补BG必有净收益。图法恢复的假前景50.73%无任何局部支持、49.16%有接受支持但原始多数失败，双角色分别57.27%和42.17%；“多数成立后gate改判”只占错误恢复0.10%/.29%。这些是已保存输出的阶段分组，不是因果干预。它们说明问题包含局部观察与全图任务判据的冲突，不能再只用“新head忘了gate拒绝理由”包办。[S49](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/NEGATIVE_RESTORATION600_20261011.md)[S50](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/RESTORED_STAGE600_20261011.md)

这些事实把当前未知问题具体化：在已有强处理与其他预测冲突的位置，如何凭合法R/Mask、Q及实际观察，判断这次否决是正确排除了相似干扰，还是错误删除了目标，并把这种判断变成跨任务的净正完整输出。还缺少这条可迁移决策依据，以及它相对强简单控制的独有收益和完整观察成本证据。

## 评价与论断需要保留的纠正

| 容易造成误判的表述 | 可核对的实际范围 |
|---|---|
| 12源照片隔离留折 | 旧源码仅train=folds!=fold，实际每折3000/1000；当前审计记174/174/155/181个held任务与训练共享至少一张R/Q照片。是类别折留出。 |
| 12源也用了sizecut | 通用report模板无条件写过此警告；实际12源没有sizecut，标签拟合sizecut属于13源扩展。 |
| fine64比fine16分辨率更高 | 两者均128²；64/16是粗图求解lambda。 |
| 4000例是新确认 | 旧public4000包含DEV241、旧600与fresh600；seed2验证另与已评2380重合322个完全相同任务。 |
| 600类均值52代表LVIS官方全量 | 原S600仅每折60个观测类；同预测计全部920槽原图33.916394。对同批FoRIS的正差仍成立，不能与23,000全量直接混比。 |
| PACO44.94到39代表算法退步 | 观察264槽与全部303槽、1024与原图共同变化；需要重池化同一I/U后比较。 |
| 只读COCO等几个像素净改正多就必涨 | 类macro会让不同类收益有不同价格；CVPR突破分析40例ridge净增TP119534却降mIoU1.307是实例。 |
| 去gate不变说明完整模型没读Mask | prototype、投票、APD派生等路径仍使用Mask；770对合法hard任务库存没有D64或联合N64/D64碰撞。 |
| 图传播抹掉全部条件信息 | 已知传播算子时连续场可逆算原场；排序响应可保留而阈值输出仍错。 |
| 86%种子命中，只剩14%涨分 | 是top1%种子纯度判据的例数比例，非mIoU预算或纯身份错误率。 |
| 所有近期负结果都由99.70%负样本遗漏造成 | 暴露审计限两个实际输入定义；角色边和几何匹配不能直接套用，而且没有修正后的因果试验。 |
| Deep论文13.3与本机29.38直接作差 | 精确论文道路协议未发布；当前道路集是自定义探索。 |
| 最新只有几百MB结果说明缓存完整 | 缓存多为O24且对应特定crop/view；新层新观察不能从旧O24反推，旧共享raw缺项与已删支路分开。 |

上述照片隔离和lambda身份纠正由保存执行代码、选择JSON及专门审计共同支持；不是重新改写旧得分。[S09](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/03_graph_readout/FINDINGS.md)[S43](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/scripts/run_exact_family_crossfold.py)[S04](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/report.json)

## 已完成工作与最终交付的距离

| 交付 | 当前事实 |
|---|---|
| 完整COCO开发优势 | 已有；RCG MEAN fine 残差与组合均有封存结果。 |
| 真实局部观察的独立作用 | 在LVIS和自定义Deep已测到；只共享特征的替代未保分。 |
| PACO可用参考判别 | global ridge及背景竞争有局部正证据，COCO迁移未成立。 |
| 五域同队列完整组合 | 有6200逐例账本、原生PASCAL/SUIM各2000；仍未全面胜强控。 |
| 扩展数据 | 九个released-loader输入池准备齐；ISIC600及Fundus200有完整FoRIS，iSAID60是前缀，Chest167保留用户停止前缀；Deep精确官方道路协议仍缺。 |
| 原创贡献 | 新规则的必要性与稳定独有收益尚不足；诊断 新名字 数学重写不能自行完成此项。 |
| 统一十域稳定领先 | 未完成；不能由开发五域、单域大收益或逐域选最佳head替代。 |
| 冷输入成本 | 部分真实同机比较存在；缓存读出速度和全部control批吞吐仍不能代替最终法端到端成本。 |
| 独立确认 | 旧照片独立1200只准备未执行；已曝光库搜索与新seed重合不能代替新的完整确认。 |

当前状态入口记录最近五个600配方均关闭，没有下一冻结候选；保留旧强S、原始缓存、完整负结果及真实局部窗口。继续工作需要的是能同时解释已有正负反例、能被完整对照检验的任务裁决原则，而不是再补一条形式不同却依赖相同自拟合前提的head。[S28](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS_RESEARCH_STATUS_20261011.md)[S30](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_NEGATIVE_EXPOSURE600_20261011.md)

## 来源索引

- S01 [旧完整结果账本](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md)
- S02 [旧核心会话与批量方法复盘](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/docs/research/LESSONS.md)
- S03 [CPU100 与 Astra300 实际执行账本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/cpu100_20261006/METHODS_AND_FAILURES.md)
- S04 [12 源留折实际分数与选择记录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/report.json)
- S05 [4000 例标量残差实际分数](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.json)
- S06 [R1 G 之前本应使用的失败证据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/EXISTING_EVIDENCE_AUDIT_20261010.md)
- S07 [局部观察与完整融合检验](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/01_multi_observation/FINDINGS.md)
- S08 [角色竞争与 source agreement 完整检验](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/02_task_identity/FINDINGS.md)
- S09 [图读出与旧照片隔离纠正](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/03_graph_readout/FINDINGS.md)
- S10 [数据集差异与父部件真实标注](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/04_dataset_differences/FINDINGS.md)
- S11 [参考读出与采样校准语义](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/05_reference_readouts/FINDINGS.md)
- S12 [联合编码 Mask 提示与任务切换](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS55_STATUS_20261010.md)
- S13 [查询基底 600 完整负结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/SCENE600_RESULTS.md)
- S14 [Copula200 真实与打乱信息诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/COPULA200_RESULTS.md)
- S15 [真实参考 focus200 完整对照](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/REFERENCE_FOCUS200_RESULTS.md)
- S16 [参考查询角色分布 600 诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/REFERENCE_QUERY_MIXTURE600_RESULTS.md)
- S17 [任务可表达与合法可识别诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_TRANSFER_CORE_20261010.md)
- S18 [物理对应先验 40 完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_PHYSICAL_PRIOR_20261010.md)
- S19 [跨视图绑定 40 原始评分](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/cross_view_binding40_20261010/report.json)
- S20 [Sol 全部阶段执行与纠正账本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SOL_VALIDATION_20261010.md)
- S21 [图机制与末块 QK 实验](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SOL_CACHE_EXPERIMENTS_20261010.md)
- S22 [6380 例校准完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_cache_experiments_20261011/reference_calibration_loo6380/REPORT.md)
- S23 [九窗 1300 完整组合结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/NINE_WINDOW_EXECUTION_20261011.md)
- S24 [Pro200 场重建原始结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/scene_reconstruction_20261011/REPORT.md)
- S25 [五域 6200 统一机制及完整读出](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/FIVE_DATASET_UNIFIED_MECHANISM_20261011.md)
- S26 [第一条 6200 完整冲突组合](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_CONFLICT_COMPOSITION6200_RESULTS_20261011.md)
- S27 [背景裁决 6200 完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/REFERENCE_BACKGROUND_CONFLICT6200_RESULTS_20261011.md)
- S28 [最近五个完整 600 候选与最新状态](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/OPUS_RESEARCH_STATUS_20261011.md)
- S29 [强 S 局部门控与多数阶段复原](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_STAGE600_RESULTS_20261011.md)
- S30 [强 S 负证据实际输入暴露](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_NEGATIVE_EXPOSURE600_20261011.md)
- S31 [Fundus200 源码阶段与细血管损失](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/FUNDUS_MECHANISM.md)
- S32 [十数据集协议与输入池盘点](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/protocol_inventory.md)
- S33 [扩展数据集完整 FoRIS 执行](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/extended_foris.md)
- S34 [局部观察简化与完整成本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/SHARED_FEATURES_AND_LOCAL_READOUT_20261009.md)
- S35 [17 对参考 nullspace 实际结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_NULL_RANK_PROBE_20261010.md)
- S36 [五域实际四编辑和案例](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/FIVE_DATASET_CASE_MECHANISMS_20261011.md)
- S37 [原生 PASCAL SUIM4000 完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/NATIVE4000_RESULTS.md)
- S38 [两域 4000 同补点数量的选择收益](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/NATIVE4000_BUDGET_SELECTION.md)
- S39 [最近真实任务共同误报审计](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_SHARED_ERROR60_RESULTS_20261011.md)
- S40 [重复推翻强 S 有效否决](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/STRONG_VETO_REVERSAL600_RESULTS_20261011.md)
- S41 [实际任务梯度核与完整上界](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/sol_validation_20261010/TANGENT_THRESHOLD_CEILING.md)
- S42 [最新完整几何匹配结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/TASK_CONSTELLATION600_RESULTS_20261011.md)
- S43 [旧类别折选择的实际源码](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/scripts/run_exact_family_crossfold.py)
- S44 [M4 启停 缓存与历史完整执行状态](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/PLAN.md)
- S45 [Z1 Z2 合并1400同批结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_zoom800_20261009/score/REPORT.md)
- S46 [真实查询编码与同框插值200对照](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_zoom_interpolation200_20261009/score/REPORT.md)
- S47 [九窗口首200及774空窗口完整结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_fixedwindows200_20261009/REPORT.md)
- S48 [M4 五数据集每集120探索原图结果](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/explore600_20261008_v1/SUMMARY.md)
- S49 [负证据暴露与真 假恢复的交叉核对](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/NEGATIVE_RESTORATION600_20261011.md)
- S50 [错误恢复对应局部支持 多数及gate阶段](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/RESTORED_STAGE600_20261011.md)


本地六个主聊天覆盖“准备本地实验所需数据”“检查子分支提交与推送状态”“深入分析当前实验与结果”“Develop DINOv3 segmentation method”“持续跟进实验 目前上一轮session经常阻塞断流 思考为什么 然后完成opus5.的时候实验”“阅读仓库 深入分析我们的已有研究进展 研究相关算法实现 推进核心任务 确保多数据集我们能全面领先foris 合理利用本机gpu实验资源”。另读取归档“分析当前实验与已有结果”以及可访问的“设计参考关系迁移方法”“CVPR突破分析”“CVPR2027 分割研究”；来源间重复的历史文本只计一次。

论文原数字、旧机器人规则和外部模型的执行建议在此仅作历史材料，不产生新的实验授权或新约束。
