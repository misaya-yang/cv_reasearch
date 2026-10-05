# 在特征空间联合反演格子面积、边界方向与端元尺度

2026-10-05。此项是新局部推断候选，区别于已失败的“先固定alpha再PLIC”。没有新encoder、训练、GPU或数据下载。全20为用户上传的已暴露DEV便利子集、20类各1例；1024工作分辨率、无CRF，不是确认集或原分辨率benchmark。

## 问题与前一结果

在相同own±16px trimap内，固定alpha后PLIC、alpha梯度法向、reference-NN梯度法向均输给同alpha bilinear。可靠GT面积则允许更好几何复原。因此本候选直接用完整局部特征约束未知面积，而不再从有偏alpha场推法向。

独立前提观察使用了queryGT面积：固定同一center F/B，8邻居真coverage估计共享rho后，中心coverage MAE从.17473降至.12376，19/20改善。这是带特权的假说证据，不能当本算法收益，也不解决无GT时rho与几何能否识别。

## 先例与准确定位

空间–光谱子像素映射早已直接联合解混与细格标签，ELMM早已处理端元尺度变化，LVIRA/ELVIRA早已根据邻格面积拟合直线。当前只能称为**归一化冻结特征上的局部几何约束解混适配**，不是首次联合解混或首次估端元尺度。

- [Xu等，TGRS2018，作者原文](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2018.TGRS.Subpixel.pdf)：细格变量经下采样和端元混合直接解释粗光谱，加入空间约束。
- [Drumetz等ELMM发布页](https://openremotesensing.net/knowledgebase/spectral-variability-and-extended-linear-mixing-model/)及[作者提供实现](https://github.com/ricardoborsoi/Unmixing_with_Deep_Generative_Models/blob/bc7d33f1e13b19ea7871ece5db9a7dc5064f899e/other_methods/ELMM_ADMM.m)：端元缩放与丰度联合估计。
- [Pilliod–Puckett2004作者论文](https://www.math.ucdavis.edu/~egp/PUBLICATIONS/JOURNAL_ARTICLES/APPEARED/2004/JEP-EGP-2004.pdf)：局部面积约束的线性界面重建。

## 模型与两个直接控制

当前cell为中心，邻格中心坐标x_i∈{−1,0,1}²。用单位法向n(θ)=(cosθ,sinθ)与偏置d定义直线；a_i(θ,d)为其半平面在第i格内的精确面积比例。使用同center-selected单位query-local端元F,B：

qhat_i = normalize(rho·a_i·F + (1−a_i)·B)，rho>0。

单位化以后，观测只能给混合方向；若非负锥系数归一成beta，则

beta = rho·a / (rho·a + 1−a)，
logit(beta)=logit(a)+logrho。

因此rho=1是实质假设。普通beta未必等于几何面积。若把多格的共同几何和共同rho同时拟合，斜边的多个不同面积可能补上尺度信息，但并非必然。

固定c=F·B，e1=F，e2=(B−cF)/sqrt(1−c²)。每格q只需两个投影u=q·e1,v=q·e2；拟合归一化混合的二维坐标，与原1024维平方误差只差一个不随参数变化的正交残差。高维向量并未提供1024个独立的rho/几何约束。

所有主臂都使用相同9格输入：
- affine-beta：对9个独立cone beta做含截距的空间仿射LS，中心预测恰为9beta均值。
- rho=1 geometry：同局部面积模型、同特征损失，只固定rho=1。
- free-rho joint：同损失联合估theta,d,logrho。

原query-local alpha+bilinear与native pre共同保留。没有按GT重新选纯端元、trimap宽度或seed。

## 完整推断

1. 固定原FoRIS pre的欧氏±16px unknown；其外硬mask保持原值。
2. 只有完全位于预测pure-FG/pure-BG区域的query tokens作端元候选；每中心在两侧各按最大余弦选同一对F/B，ties沿原flat index。
3. 对每个与eligible band相交且拥有完整3×3窗的中心，固定这对端元给整个窗，不为邻居重选端元。
4. 拟合上述9格二维特征残差。固定rho控制不变；自由rho用profile-logit的2D初始化和同一feature loss求解。
5. 数值界固定为theta初值±pi、d∈[−2.2,2.2]cell、logrho∈[−6,6]，每stage最多60次function evaluation；这些不是按20GT选出的质量阈值。
6. 对候选解求完整参数Jacobian J与中心面积梯度g。数值rank相对容差1e−6；若g在J的nullspace中有非零分量，中心面积不局部可识别，退回原alpha。
7. 参数不满秩不自动拒绝：若中心面积对未识别方向不变，面积仍可局部确定。优化失败或尺度触边仍按固定规则退回原alpha。
8. 只把合格中心的面积写回alpha场；全部窗独立一次，不循环更新q、端元、trimap或相邻窗口结果。
9. 与原控制完全相同的bilinear上采样及eligible mask，outside硬值原样恢复。不在这里再加PLIC，避免混淆面积估计与几何rasterizer。
10. 全20预测冻结后才读queryGT，评分真实coverage与完整same-trimap mask，并比较同有效域控制。

边界格没有完整3×3窗时都沿用原alpha。常数数值门槛不是语义可靠性证书；满秩只给局部数学识别，不能保证数据符合单直线/共同端元/共同rho模型，更不保证全局唯一最优解。

## 9格主部署与8格诊断的区分

原先已开始8邻居、中心不进fit loss的诊断。父级在任何8格真实分数被读取前，确认中心q本来就是合法部署观测，因此9格被锁为唯一主版本。预声明保存在 `joint_boundary_primary9_decision.json`（04:22:47）。8格后来完成，但没有用其成绩选版本、改阈值或改参数。

端元本身由center q选择；所以即使8格不把center放进残差，也只能叫条件留中心诊断，不能叫完全未见中心。9格主版本更不能把中心feature拟合误差当泛化成绩，关键仍是冻结后真实area和mask。

## 已运行的精确正例与歧义反例

`joint_boundary_synthetic_nine.json`：
- 精确斜线theta=.43,d=.11,rho=2.5：真中心面积.621016661；joint恢复到约1e−16，rho=1几何却给.78705。
- 竖直边：中心面积.65/rho3与面积.58208955/rho4给出同一整个3×3单位feature，差1.11e−16。9格joint Jacobian rank2，中心面积梯度含nullspace分量，正确fallback。
- 这一歧义不只属于恰竖直边；少量不同的mixed fraction也可能不足约束三参数。多个mixed cells的数量本身不是识别证书。

Profile初始化解决了最初在rho=1起点的平坦局部盆地问题，但单个初始化仍不提供非凸全局收敛保证。下面的真实检查确实发现了这一实现缺陷。

## 原9格第一读数：负结果保留，但优化器有缺陷

`joint_boundary_real20_nine/report.json`：baseline alpha+bilinear66.26007，native66.60337；affine-beta65.16987，rho=1几何65.44564，free-rho64.21312。free−baseline −2.04696 [−2.96648,−1.22821]，20例全降；free−rho1 −1.23253 [−1.92416,−.62291]。

9812个窗口中4259实际更新（43.4%）；其中706只满足中心面积局部识别、并非全参数满秩。不同更新域的公平检查已单独做：common C=joint.valid∩rho1.valid共4238格，同C下free−rho1 −1.51982 [−2.24629,−.90762]。独立重算mean9与原alpha/mask完全相同。

但是独立审查发现free-rho训练目标在4989/9812窗甚至高于已求出的rho=1可行解，其中2309属于joint有效域。因为free模型包含rho=1，这说明单profile起点落入了更坏的局部盆地。这个负mask结果是真实输出，不能删掉；但不能据它把失败全部归因于模型或可识别性。

## 唯一优化正确性修复：已暴露后，重新冻结一次

修复文件 `joint_boundary_feature_inversion_repaired.py` 保留全部原损失、端元、bounds、rank/area-nullspace标准与trimap。只增加：

- 从已求rho=1解初始化一次同样的free优化；
- 把rho=1解本身列为最终可行floor候选，并重算完整3参数Jacobian；
- 在profile、fixed-start、fixed-feasible-floor三者中，仅按推断时可见训练残差选最小者。

每窗保存三个候选loss、来源与选择，并assert所选目标不高于rho=1。若选择直接fixed feasible候选，convergence沿用该固定优化器的状态，但不声称它是free-rho驻点；完整free Jacobian与面积资格重新求取。此前9格已使用area-nullspace标准，修复没有把“满参数秩”换成另一个gate。

该版本明确标为optimizer-repair-after-exposure，不是新的独立实验集。其完整结果与同C对照完成后追加如下；不会因为更小训练loss就声称coverage改善，也不在这20GT上继续调gate。

## 04:38唯一优化修复后的最终读数

修复确实满足嵌套目标：全部9812个窗口所选free-rho visible loss均不高于rho=1可行floor，最大正违例0。候选来源为fixed初始化7302、profile2510；没有用GT选解。rho=1与affine控制点值未变。

主部署输出（同own-trimap、1024/noCRF）：
- 原alpha+bilinear：66.26007；native pre：66.60337。
- rho=1 geometry：65.44564。
- 修复joint：65.36449，对原alpha −.89558 [−1.38884,−.44772]，3升17降；四折−.82963/−1.04006/−.90049/−.81215。
- 修复joint对rho=1全域：−.08115 [−.25272,.10243]，未决。不能用它证明normratio有价值。

4131/9812窗口实际使用joint更新（42.10%）；其中3321全参数数值满秩，810只满足中心面积局部识别。全体4864个窗口选解未满足优化器convergence条件、798触rho数值界，故回退；这些类别可重叠。没有把全体窗口的数值rank当实际有效覆盖。已接受全参数窗口的condition中位/90分位/99分位约163/2061/441837；数值满秩并不保证在真实feature噪声下稳定。

### 严格相同修改域的公平控制

固定C=joint.valid∩rho1.valid，共4108格。三个臂只在C写入各自已冻结的面积，其他所有格子完全相同原alpha回退，再同样bilinear与hardoutside恢复；没有重跑优化器。原joint有效域另有23格rho1无效，故严格C下joint点值65.36324与全主版本略有差别。

- C内joint−rho1：−.31777 [−.56810,−.08791]，4升16降，四折全负。
- C内joint−mean9：−.34070 [−.71481,.00373]，未决，5升15降。

这排除了“少更新一些区域就赢”的归因问题；本候选并没有靠共享rho优于固定rho。

### 同一组真实mixed cells的面积误差

只为评价而查看GT，在同C的1293个真实mixed中心、全部20个episode上，先episode平均再宏平均：

- 原alpha MAE .21778
- mean9 MAE .22485
- rho=1 MAE .26447
- joint MAE .30267

joint−rho1 MAE +.03820 [.02091,.05761]，越大越坏；joint−原alpha +.08489 [.06606,.10474]。同C全部4108中心上，joint−rho1 MAE +.01678 [.00925,.02409]。

独立审查也确认0个nested-objective违例；在1187个joint训练feature loss严格低于rho=1的共同有效mixed格中，有538个中心area仍更差。额外自由度更好解释feature，并不等于识别了几何面积。对9格主法，center feature本来就在训练损失里，更不能把其拟合改善当额外验证。

所有新区间使用20个配对episode差的直接有放回重抽样（保留重复、2000次seed0、按固定manifest顺序）。面积区间使用相同20episode的区域内平均误差差。独立脚本的bootstrap顺序可造成小的Monte Carlo分位差，本记录引用主manifest顺序。

## 文件与结论

- 主修复记录：`joint_boundary_real20_nine_repaired/report.json`、`freeze.json`、`window_records.json`与`frozen/`。
- 严格同域：`joint_boundary_real20_nine_repaired_matched/report.json`、`freeze.json`、`frozen/`。
- 独立优化与面积复核：`joint_boundary_real20_nine_repaired/independent_repair_audit.json`。
- 原未修复负输出与8格诊断全部保留，未混为新确认结果。

**收束这条具体构造。** 合法9格、相同端元、单条直线、共享自由rho、单次局部拟合即使修好nested优化，仍让真实面积和same-domain mask变差。不再沿这20个已暴露样例添加loss权重、residual gate、rho先验或rank阈值。

这不是所有DINO亚格信息的上界，也不是证明任何共享尺度模型都不可行。已得到的可复用结论是：当前局部normalized-feature重建损失与真正cell面积之间仍存在识别/模型错配，不能只靠更精确地拟合该损失跨过边界缺口。下一机制必须引入或验证对实际格内几何有约束的合法观测，再给出其相对简单控制的实数；不能把此处oracle几何/GT-rho的空间继续当方法增益。

### 独立最终核验已通过

`strong_method_controls_20261005/joint9_repaired_independent_review.md`及对应report已核全部9812个选解的最小visible loss、floor、端元不变、控制alpha/mask不变、资格与fallback；全部20的独立bilinear replay为0像素差，1,764,875个eligible像素外无改动，共同C域与三臂matched masks逐位一致。没有重跑优化器。修复相对旧版本实际+1.15137 [.52253,1.89034]，19升1降，但改进包括资格域改变；仍不改其相对baseline和同域rho1为负的最终结论。
