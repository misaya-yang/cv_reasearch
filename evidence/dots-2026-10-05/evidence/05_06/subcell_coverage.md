# 从每个16×16格子的面积率重建亚格边界

2026-10-05。结论：这是值得先用已有coverage缓存做一次CPU对照的**现成几何解码基线**，不是已成立的新方法。它在可靠面积率、局部单条边界时能恢复二值格点读出丢掉的几何；面积估计噪声会被逐格守恒放大，多条细结构的格内位置也不能仅凭面积唯一确定。

## 先把用户的新证据说准确

用户最新提供、此次未独立重跑：DEV241约58.6，GT boundary-crossing token内修复到66.7，约8.2诊断差被分为6.2亚patch二值表示限制与2.0 token标签错误；现feature-linear-mixture coverage已取得+.88 [.50,1.71]。真实trimap条件可很高，而自己的trimap只有约+.9，90的具体指标仍需原来源核对。

**6.2是当前二值16×16读出约束的损失，不是DINO token已经丢掉6.2分信息的证明。** ViT-L token为1024维，16×16 RGB为768个数，维数本身甚至不能证明初始patch映射必然压缩信息；后续norm/attention如何改变可恢复性是另一项实证问题。本报告的不可辨识结论只约束标量coverage及其邻域，绝不约束完整1024维特征。两个相同coverage的格子完全可能仍有不同的DINO特征。

## 精确输入与已有方法来源

输入A_ij必须被解释为该16×16格子内前景的**几何面积比例**，不是任意min-max FoRIS分数、类别概率，或自然图像matting的光学不透明度。局部特征线性混合系数是否等于面积率需另行验证；此几何操作不修补这个前提。

Volume-of-fluid的PLIC已有数十年历史。Pilliod与Puckett的LVIRA/ELVIRA根据邻近cell的已知面积率选择局部直线；作者把精确复现直线作为二阶几何重建的条件，并做了对应试验。[作者论文，§2.5及§3](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2004/JEP-EGP-2004.pdf)

自然图像closed-form matting则依赖RGB合成方程与局部颜色假设，求opacity场。本路线不给RGB，重建的是已估计格子面积所对应的binary几何，不能借用matting的高质量示例证明此处可行。[Levin等作者论文](https://www.ee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)

## 单一完整解码操作

1. 保留现有feature-mixture方法给出的alpha和own-trimap；绝不换成GTtrimap，也不重新选择可信种子。
2. alpha=0或1的格子分别全背景/全前景。
3. 对每个0<alpha<1的格子读取3×3面积率。
4. 对三列面积和做前向/后向/中心差分；对三行亦如此，得到六个候选直线法向。
5. 对每个法向n，以二分法求偏置c，使当前格子被半平面n·x≤c覆盖的面积**恰好**为alpha。
6. 将该直线延伸到3×3邻域，计算其预测面积率与输入alpha的平方差，选择最小者。平局按固定候选顺序，不读GT。
7. 在当前格子的16×16输出像素中心判断n·x≤c，输出binary mask；所有格子独立一次完成。
8. 使用既有原分辨率恢复与评分方式。报告pre-CRF以隔离几何，再共用相同CRF；不能只给新方法额外后处理。

这是紧凑ELVIRA-style实现，未声称完整复现流体模拟器的全部边界处理与所有理论保证。连续半平面严格守面积，最终16×16中心采样有离散量化误差，需要单独报告；不能声称输出binary像素数逐格完全等于任意实数alpha×256。

### 可直接检查的面积公式

在以0为中心、边长1的方格中，令a=|n_x|, b=|n_y|, t=c+(a+b)/2。a,b均非零时：

F_n(c)=[t_+²−(t−a)_+²−(t−b)_+²+(t−a−b)_+²]/(2ab)。

a或b为零时退化为一维clip线性函数。F随c单调，所以给定法向与0<alpha<1，c唯一；本脚本用45步二分，未搜索任何任务超参数。

法向为什么可从邻域恢复：一条直线若在某方向的3格高/宽窗口内形成单个graph，其列/行面积和等于直线高度的积分，相邻高度差给出斜率。六种有限差分候选允许选择未被局部裁切破坏的方向。弯曲、角点、多条界面或噪声破坏这一条件，面积约束本身不能恢复条件。

## 已执行首数：全部synthetic，16像素/cell

文件 `subcell_coverage_probe.py` / `.json` / `.log`。NumPy+SciPy CPU，seed11，16×16 cells，每cell16×16 pixels；12条直线随机偏置，评价时去除一圈cell以避免本实现edge-padding边缘假设。

同一**真实面积率**输入：
- 12条直线平均IoU：nearest-half .96060；bilinear-half .99258；PLIC 1.00000。
- 圆：nearest .89779；bilinear .97833；PLIC .99635。
- 圆的连续格子面积误差最大2.13e−14；binary中心采样的平均coverage误差.000259。

这只是在已给可靠面积下检查几何解码，不能报成“DINO +.74/+1.80”，更不能用来预测DEV241。

### 负例一：相同全部面积率，不同细线位置

构造同一行cell中的两个连续水平细条，一个位于每格顶部四分之一、另一个底部四分之一。所有cell alpha逐位相同、所有邻域相同，但两条真mask互不相交。

固定PLIC结果对top真mask IoU=0，对bottom=.35106；面积保留约.984375（差来自binary rasterization）。同输入显然不可能为两个不同真值都输出正确答案。

更一般地，两个互不相交、面积均a且观测完全相同的真mask，对任何也有面积a的预测，至少一个的IoU≤1/3。证明：两者交集x+y≤a，所以min(x,y)≤a/2，对应IoU≤(a/2)/(2a−a/2)=1/3。这个限制属于**这些观测和面积约束**，不是DINO特征的上界。允许输出全图可改变minimax值，但不再守同一面积。

### 负例二：守恒会放大错误coverage

同一条直线，对**所有cells**添加固定seed的独立高斯coverage噪声并clip到[0,1]，不使用oracle trimap挑噪声位置：
- sigma=.01：coverage MAE .00415；bilinear IoU .99416，PLIC .99686。
- sigma=.05：MAE .01813；bilinear .99209，PLIC .96701。
- sigma=.10：MAE .04611；bilinear .98761，PLIC .91308。

每个错误的非零背景alpha都会被几何守恒变成真实FP像素；错误的非1前景alpha会变成孔洞。因此“更精确实现给定alpha”不等于更正确分割，真实首检必须保留same-alpha bilinear，不能只对比硬token输出。

## 能否从完整cached features提取额外法向？

可以作为独立待测问题，不能被上述alpha-only反例排除。一个重要等价关系先排除伪新意：若alpha_i=aᵀq_i+b、a固定且该处没有clipping，那么

∂_x alpha=aᵀ∂_x q，∂_y alpha=aᵀ∂_y q。

把参考FG−BG方向投影到feature梯度上，在此条件下与直接alpha梯度是同一个法向，不能当新增几何信息。球面混合或局部不同端元会改变函数，但仅换链式法则也不是新观测。

最小同cache对照：固定相同alpha和逐格偏置约束，只替换法向来源：
- alpha的3×3几何拟合（本脚本）；
- alpha的中心梯度；
- s_ref(q)=max_{referenceFG}cos(q,r)−max_{referenceBG}cos(q,r)的中心梯度，n∝−∇s_ref；梯度为零沿用预先固定的alpha法向。

第三项使用已有source mask coverage与q/r，或已经缓存的reference_nn_control，不需要新增encoder或像素训练标签。它的法向可能独立于alpha，但仍只是一项功能不同的几何估计，不证明已从token读出亚格信息。若今后想直接由token预测格内方向/phase，必须取得相应合法训练/验证信息；**只有source cov时，不能凭空把source像素mask或格内边缘方向当已知标签**。

当前代码只实现第一项及nearest/bilinear控制，未把未执行的feature-normal对照写成结果，也没有先建立整套normal/levelset求解框架。

## 第一真实cache测试如何选

优先使用用户已经得到+.88的**同一份实际feature-mixture alpha**：冻结其own-trimap、端元、所有超参数，只替换upsampling为本PLIC与same-alpha bilinear，读原分辨率paired I/U、每折、照片连接bootstrap与加删TP/FP。同时读取3×3拟合残差与收益的关系，只作诊断，不在这批上追加gate。

若现20对只有q/r/score/cov及queryGT而没有该alpha，不能将min-max FoRIS score冒充面积率。可用queryGT下采样的alpha做**明确ORACLE几何诊断**，回答真实轮廓是否近似单界面、多少二值格子损失原则上可由area+neighbor恢复；它不能当方法gain或合法trimap。合法估计alpha的实测仍需单列。

脚本可直接执行：

`python subcell_coverage_probe.py --out subcell_coverage_probe.json`

有真实area文件时只做推断、不读取GT：

`python subcell_coverage_probe.py --alpha alpha.npy --scale 16 --save-mask plic_mask.npy --out plic_geometry.json`

复杂度：每mixed cell六个法向×45步标量截面积反演，再做9格残差与256像素raster；内存为alpha grid和输出mask。本轮实际执行的是小型geometry首数，不是完整DINO/CRF integration，没有GPU、模型或数据下载。

## 当前选择

先用同一个现成alpha做PLIC vs bilinear的真实CPU对照，而不搭建新的learned decoder或level-set框架。正例证明值得测，噪声与phase反例给出明确反驳条件：若same-alpha bilinear已相当，或PLIC主要放大wrong-trimap coverage噪声，就把缺口留给alpha/trimap估计，不能继续把它叫upsampling问题。

## 04:03真实20对ORACLE几何诊断

实际cache已到：64×64 token网格、1024×1024 packed queryGT，没有用户原feature-mixture alpha。脚本 `subcell_real20_oracle.py` 只将queryGT下采样为真实cell coverage，再重建同一1024工作栅格；无CRF、不是原始图片分辨率benchmark、不是合法推断方法。耗时15.49秒CPU。

- ORACLE nearest-half：89.98475
- ORACLE same-alpha bilinear-half：94.85715
- ORACLE PLIC：96.27960
- PLIC对bilinear配对 +1.42245 [.90336,2.04577]，20升/0降；四折+1.76522/+1.53365/+1.47509/+.91585。20照片连接组、2000次bootstrap、seed0。

结果 `subcell_real20_oracle.json` 保留完整逐例I/U。这证明这20例真实轮廓在可靠面积条件下有额外可恢复几何，不证明现有token能给出正确alpha，也不能把+1.42加到用户+.88或与6.2直接相乘。

正在接收同一CPU主实验新冻结的query-local own±16px trimap估计alpha，做same-alpha PLIC-vs-bilinear。二者将恢复完全相同的outside-trimap硬mask；先前全域加噪toy只展示一般风险，不能自动外推成带硬trimap约束的实际退步。不加按这20例GT调出的几何residual gate。

## 04:09合法估计alpha首数：PLIC明确输给same-alpha bilinear

已接收CPU主工作者预先冻结的query-local两端元alpha，完全相同的own±16px trimap/eligible与outside硬mask。该alpha是本轮新固定控制，不是用户历史+.88的外部实现。20例全部预测冻结后才评分；每个bilinear对照与CPU主工作者保存的mixture逐位一致，总mismatch=0。

工作1024栅格、pre-CRF：native_pre66.60337，same-alpha bilinear66.26007，PLIC65.57147。PLIC−bilinear=−.68861 [−1.09901,−.33148]，4升16降，四折−.61042/−1.25572/−.52835/−.35993；20照片连接组，2000bootstrap seed0。运行30.91秒。

`subcell_real20_ownalpha/report.json`、`freeze.json`和逐例packed预测都保留。575/10035个active cells的估计alpha与该cell内outside-trimap固定像素所允许的面积区间不相容；只是诊断，没有用它做gate。恢复outside硬mask后，最终binary图并不严格守原whole-cell alpha，报告也记录了该误差。

## 04:12预声明法向对照也为负

同alpha、同trimap与offset约束，仅换预先声明的法向，未拟合参数：
- n=−∇alpha：65.53365，对bilinear −.72642 [−1.16457,−.35343]，4升16降。
- n=−∇referenceNNmargin：65.29955，对bilinear −.96053 [−1.47953,−.50981]，3升17降。
- 两者四折均负，梯度零只沿用已冻结PLIC，不新增quality gate；运行10.31秒。`subcell_real20_normals/report.json`与bitmaps可独立复核。

当前结论收窄为：这套固定面积后几何重建对合法估计alpha有害，不能据GT-area的+1.42往现方法加收益。失败不证明所有高分辨率读出无效，也不证明DINO不含亚格几何；下一机制应改变面积/几何的联合推断，而非按这20例事后调一个PLIC residual gate。另一个重要更正：最新独立工作发现GTtrimap仅unknown=.5、不用feature就约88.94，不能把此前90直接当feature mixture证据。


### 统计区间更正（04:24）

以上真实20的CI已替换为统一核查后的区间，点值/预测完全不变。20例恰为20类各1例，旧class-I/U加权bootstrap使重复抽到的类权重在I/U中抵消；正确做法是对20个配对episode-IoU差有放回抽样后保留重复项求均值。原始report保留作为历史，权威更正为 `strong_method_controls_20261005/corrected_real20_statistics.json`。这些区间仅描述当前便利选择、已暴露DEV20，不代表DEV241/COCO80泛化。后续joint主试验直接使用正确的paired20 bootstrap。
