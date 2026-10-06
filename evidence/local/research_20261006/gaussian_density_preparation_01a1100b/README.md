# 密度路线实质修订 v2：完整特征 Gaussian 密度的有界校正

**可部署的单候选版本，真实效果未测，独立方法增量0。** 不继续要求查询对象的组成匹配参考对象，不做三阶/triplet，也不新建理论候选队列。实现：[reference_gaussian_density.py](/Users/yang/projects/CVPR2027/src/ics/methods/reference_gaussian_density.py:1)；独立入口：[run_reference_gaussian_density.py](/Users/yang/projects/CVPR2027/scripts/run_reference_gaussian_density.py:1)。没有改共享入口/PLAN/STATUS，没有访问服务器。

## 已测依据与修订边界

同已复用public600、同缓存/1024读出中，MEAN为62.931546，原scalar prior-shift为37.883317、quadratic为51.607030，covariance为12.343239。来源：[本轮报告](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.md:5) L5–41；数据暴露标注L47–48。**这些读数否定了旧完整构造有效的预期；总分本身不能证明某一个失败原因，更不能推出本修订必然有效。**

原密度法只读取参考FG/BG质心之差的一维得分，拟合查询全图先验，并以完整后验替换MEAN。原二次法以参考锚点拟合二次函数并同样替换基场。此版本有两个明确变化：

1. 密度从一维质心得分改到整个归一化特征空间，使用参考的多模式质量与Gaussian核响应。因此合法可读取的类条件高阶分布差异不会全部压缩成质心、协方差或有限三阶矩。
2. 放弃查询全局比例估计和整体场替换，只将该类条件证据以固定最大1/8的幅度加到原MEAN场。真实600结果没有支持更强干预，固定cap是事前保守设计，不是根据这600例拟合的最优数值。

**原创性/计数边界：**Gaussian核、KDE、前景/背景原型及LSE有既有先例。对于以下f,b，g=(f−b)/(f+b)=tanh((log f−log b)/2)，本质就是带经验模式质量的Gaussian/加权LSE对比；不把它冒称为与FoRIS类匹配完全独立的新发明。这是`reference_score_density_query_prior_v1`家族的实质修订版本，独立方法增量为0。与Pro或其它新法不预先组合。

## 固定从输入到完整mask合同

输入：同生产路径的冻结q/r特征、唯一参考完整mask的coverage、既有完整MEAN连续base。无query GT/class/fold/类别名/额外编码。q/r通常4096×1024；核心支持小矩形网格。纯参考FG coverage≥.9、BG≤.1；各需≥8token，否则原样保留base；空参考明确返回空。

以既有固定球面压缩（5轮）分别得到≤16个FG锚点a_k、BG锚点b_l及类内经验质量w_k/v_l，各自和为1。相同原始参考标签用于聚类与质量估计；这是重用监督，不是独立校准。原代码及依赖hash被独立runner保存。

对每个查询单位特征x，固定τ=.07：

\[
k(x,a)=\exp((x^Ta-1)/\tau),\qquad
f(x)=\sum_k w_k k(x,a_k),\quad b(x)=\sum_l v_l k(x,b_l).
\]

Gaussian核在单位球面对应欧氏距离Gaussian；核本身有完整分布区分能力，**有限压缩锚点不保证保留所有原始参考分布，更不证明参考到查询可迁移**。

\[
g(x)=\frac{f(x)-b(x)}{f(x)+b(x)},\qquad
\boxed{t_i=base_i+\tfrac18g(q_i)}.
\]

类条件密度按各类质量归一，没有查询目标比例拟合。f,b严格正（数值零和时中性0），无minmax/clamp基场。`g`不是校准类别概率。保存t与全部固定控制；转FP32、bilinear到1024、严格>.5给完整工作mask，不取最大组件。不进行图求解、组件筛选、形状约束或全局参考分布约束。

若有原查询H/W，独立runner另将工作二值mask bilinear到原图再严格>.5并保存；若缺H/W，只声称同历史1024终点。所有预测封存后才由主代理的独立评测器读GT；runner本身不实现GT评分。

### 可证明的修订性质

\[
|t_i-base_i|\le1/8.
\]

相同双线性算子的权重非负且和为1，故连续工作读出仍有L∞误差≤1/8。原连续读出分数>.625或<.375的点不会因本候选翻转。**这不是IoU下界**；接近.5的大区域仍可能整体改坏，转原图二值mask的边界也须实测。

这一修订与原scalar prior有实质差异：低估全图比例不再自动将所有目标后验压为空，与quad/hull也不同：不拟合多项式分类头或凸支持距离，不以它们完整输出覆盖MEAN。

## 最廉价强对照：都保留同一base与cap

六个控制不计作新方法，分别返回：

| 返回键 | 固定定义 | 要排除的简单解释 |
|---|---|---|
| `uniform_control` | 同Gaussian响应，改为FG/BG各锚点均匀质量，再同g与cap | 经验类内质量是否必要；与普通原型LSE的区别 |
| `nearest_control` | base+(1/8)tanh((bestFGcos−bestBGcos)/.07) | 是否只需最近前景/背景匹配 |
| `centroid_control` | base+(1/8)tanh(xᵀ(μFG−μBG)/.07)，原始参考纯类加权均值 | 是否只是均值方向校正 |
| `polynomial_control` | 把k换成((1+cos)/2)²，保留经验质量与g/cap | Gaussian高阶量相对二阶核均值的净贡献 |
| `quadratic_ridge_control` | 同锚点/.01 ridge/类平衡质量的二次核回归，signed回归值clip±1后乘cap加base | 是否只是将旧quadratic完整场改成小残差就够 |
| `standalone_control` | .5+.5g，直接交相同renderer | 本次保留MEAN与约束干预幅度有没有必要 |

最强简单控制是在这些**固定完整行**中，不能只挑一个弱控制。若uniform/nearest/quad追平，撤回相应复杂性必要性；若standalone明显退化而bounded有益，只支持有界组合的价值。Gaussian/LSE数学等价已经说明，不把精确同义算子当额外行或额外方法。

## 合成可实现正负例

收据：[math_check.json](/Users/yang/projects/CVPR2027/evidence/local/research_20261006/gaussian_density_preparation_01a1100b/math_check.json:1)。命令：`python3 evidence/local/research_20261006/gaussian_density_preparation_01a1100b/check_math.py`。

所有特征来自二维单位圆嵌入4维，可直接作为合法归一化输入；默认代码**实际学习**锚点/质量，不注入核表/标签系数。参考FG/BG共享相同8个单位方向，每类32个样本：FG在四个轴方向各3份、四个对角方向各1份，BG交换质量，再重复两次。两类的理想一/二/三阶原始特征矩相同，四阶x矩分别.4375/.3125；共同凸支持也相同。FP32实例有通常舍入，未声称数字逐bit同矩。

64个querytoken前32为轴方向，后32为对角方向，base分别.49/.51。主候选保留32个指定真token并删除32个指定错token，bilinear1024后的完整mask与指定上半图完全一致；同cap的uniform、nearest、centroid、二次kernel-mean、实际二次ridge，以及默认hull控制都保持相反判断。证明的是**这个合法合成输入上，前三阶矩/共同支持不足以表达的质量差异能影响完整输出**。并不证明模式占用、空间adjacency/Pro或别的规则不会解此例，特别不能把它当自然图像收益。

**负例保留，未调参数：**相同R/M/Q/base若自然类别条件改变、GT成为上述正例的反面，原弱base正确，本候选却改坏全部1024²像素。若正确base改为.1/.9，bounded主法保留其完整正确输出，而standalone控制仍把整幅判断反转。它直接说明本版本只能限制接管范围，不能验证参考密度迁移。

另核对共同特征符号翻转、base平移等变、缺BG弃权、空参考；独立入口的两worker重复occurrence运行通过，输入NPZ内故意放不可反序列化的object GT key，预测器不读取它，工作/41×59原尺寸mask和预测封存标记均通过。

## 现实CPU复杂度、接口与有限真实动作

压缩O(5NrKD)，每类K≤16；全部query对≤32锚点投影O(Nq·32·D)，4096×1024约.134GMAC。32维核对照解O(32²D+32³)，Gaussian及所有基本控制复用同一投影。存储O((Nr+Nq)D+Nq·32+32²)，Nq×32 FP64响应约1MiB；不建N²矩阵，无新图/编码/GT访问。这只是规模估算，真实运行时长/峰值RSS未测；输入加载、MEAN缓存重建、renderer与评测要另计。

默认接口：`predict(q,r,cov,base)`返回`field`、六个`*_control`、可JSON序列化`info`。Config/asdict齐备且此版本拒绝参数改变。独立runner读取q/r/cov/base，支持可选`original_hw`，保存packed工作/原尺寸mask、原始field与输入/依赖hash。默认每worker一个数值线程，最多30worker；首次部署应按主代理实际CPU/内存预算设置。

下一动作只有一个：接入同600缓存、参数不变，一次评估主法/MEAN与以上六控制的完整class-summed mIoU、照片组配对区间和四类增删。此队列已暴露，如实标为开发复测。若主法不优于固定强控制，记录并关闭此版本，不继续扫cap/τ。现有证据没有支持任何数值增益预测。
