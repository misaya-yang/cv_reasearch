# Reference RGB texture：接受前固定合同

状态：关机前root接受合同并曾授权一次固定实际smoke4，预算4CPU/8GiB；该授权已随用户关机结束，不授权600。本文保留合同及澄清，完成实现/区别/检查后的候选定义可登记1，不意味着科学原创性或真实收益。此文件不是PLAN。当前任务见 [PLAN](../../../../docs/research/PLAN.md:34) L34–40。

## 定向历史核查

查过当前src/scripts、保留的local报告、INSID3/dots源包、两个demo4/demo9可恢复源码目录，未将关键词缺失当作从未尝试的证明。具体核对：

- 既有 [color bottleneck](../../../../src/ics/methods/color_bottleneck.py:1) L1–5、L16–20、L120–138，用query RGB邻边最大瓶颈与MEAN强种子，不读取参考RGB中FG/BG纹理频谱。
- 既有 [RGB Potts](../../../../src/ics/methods/mean_rgb_potts.py:23) L23–34、L46–73，用query四邻域RGB差形成边权；没有参考纹理类别判据。
- 既有 [RGB proposal v2](../../../../src/ics/methods/mean_rgb_proposal.py:11) L11–18、L41–57，只限制Potts/unary分歧区域的替换；已复用600对MEAN +.075882且区间跨0，见 [报告](../mean_rgb_proposal_v2_01a1100b/server_replay600/score/report.md:19) L19–26。这不能作参考纹理收益预测。
- 旧RGB边界明确负，账本 [RESULTS](../../RESULTS.md:82) L82–94；原始 `native_membership_v1/query_boundary_v1/experiment40/report.json` 的arms为public/scalar/DINO-boundary/RGB-boundary，scope明确40个reusedDEV。它测试query边界切分，不是参考纹理迁移。
- 旧源码曾删除，按 [cleanup](../../../../docs/archive/2026-10-05-cleanup.md:58) L58–64定位到本地backup，无Git恢复。demo4 `icx/common.py`命中的RGB只是图像读入；demo9 `extent_experiment.py` L334–358有“textured target”，实际是四例checkerboard合成fixture供zoom路径使用，未提取纹理统计或做FG/BG频域分类。见 本机备份源码（未纳入仓库）。
- 当前搜到的spectral论述主要是DINO/query图算子的特征值平滑，不是RGB空间频率；保留的texture命中主要是方法卡风险描述。恢复源码与报告的核查仍非穷尽所有已删/远端历史，故结论只是“已核对的最接近构造没有测试下述量”，不说“项目从未做过”或“已有原创性证明”。

## 十行合同（一个候选，不排列参数）

1. 输入唯一参考RGB、其完整二值mask、query RGB、固定MEAN64场和query原H/W；两RGB各依现有canonical1024双线性后再到256，参考mask最近邻到256；不访问query GT、不新编码。
2. 固定64×64全图位置、RGB256上步距4，每点提取16×16与32×32两窗，边界reflect；R/Q用完全相同窗口，不由MEAN/GT筛query ROI。
3. RGB算术平均成灰度，窗内减均值，二维FFT分4径向环带：0<r≤1/16、≤1/8、≤1/4、其余非DC频率；每个量明确为环内SUM功率/总非DC SUM功率，不除频点数、不称平均密度；平窗置0，拼8维份额。
4. 参考patch仅在两窗均完全落入已知FG或BG时供相应监督；各类至少8patch，否则完整保留MEAN；空参考输出空，样本数/弃权原因如实记录。
5. 每类取8维patch描述子的均值μF/μB，Euclidean dF=||h−μF||²、dB=||h−μB||²；若原型无分离则弃权，g=(dB−dF)/(dF+dB)，0/0置0，不引入query伪标签。
6. 全query场t=MEAN+(1/8)g，不clamp/minmax；反映“局部频率组成与R标签可迁移”的假设，不要求整对象形状/部位比例相同，不做参考mask转移。
7. 同RGB/同两窗/同R纯窗样本/同原型分类与cap，固定保存24维8-bin RGB边缘直方图（两窗归一化直方图取均值）、两窗局部灰度方差、每窗最大功率环带的实际编号/3与对应功率份额（平手取最低编号）以及g=0控制；另在真实seal前固定standalone=.5+.5*main g，不增加FFT/不改主法；缺纯窗或主原型不分离时standalone明确回退base，控制不增加方法数。
8. 每场统一FP32 bilinear64→1024严格>.5，再binary-work bilinear→原H/W严格>.5；保存所有完整mask，封存后才评分，零纹理场必须与原MEAN逐值/完整mask等价。
9. 物理uint8 RGB正例使用同两种灰度、相同总量/边界数而run长度不同的16周期条纹（4/4/4/4与2/2/6/6）；原1024 RGB、实际PIL256后同纯窗颜色直方图及方差相同，空间频谱不同；R标签给监督，Q弱base与强边界构成完整图，不向预测器提供GT筛ROI。
10. 同合同保留90°旋转核对、同纹理错类别不可分辨负例与同类纹理频率改变的负例；root接受后一次固定smoke4，若主法输同RGB强控制则关闭不变体，600须root再次明确同意。

“旋转平均”只指在频率平面按半径汇总，离散90°旋转有可核对等价性；不是任意角度插值/rasterization严格不变、尺度不变或语义身份保证。两窗口保留不同的局部支持/分辨率；不把每个尺度/环带记作独立机制。

该单条纹正例已经被强dominant-band控制解开，它只证明“颜色分布之外的空间频率可观测”，**不能证明多环带必要性**。不通过挑一个恰好无能量的弱单频对照制造优势；每个patch由固定最大份额/tie规则选其actual dominant band，使用完全相同参考监督/ROI/cap。如果它在真实同合同结果追平，应收缩多尺度/多环带必要性，不事后改变控制频率或把各频率计成新方法。

## 成本、资产绑定与计数决定

新计算不使用DINO向量。每例R/Q两图、每图4096点、每点两FFT，工作量O(2N(16²log16²+32²log32²))，约1亿FFT尺度运算量；实际秒数必须在本地合成/真实smoke分开测。用256patch分块，最大的complex128 FFT块约4MiB；描述子每图约256KiB，外加RGB/参考mask/完整读出。不构造N²矩阵，不做所有query区域重编码。1CPU/4GB应足够作为首个本地内存合同，不能把该估算当真实时延承诺。

首个阻塞资产是**参考RGB和完整参考二值mask绑定**，不能仅用现有cov64上采样并冒称完整原mask；若root只有cov64，须明确另立缓存覆盖率适配版本，不能偷偷替换合同。query RGB128已有packet也不能直接冒称本合同RGB256频率；应从已存在原RGB确定性重建并记录producer/hash，不下载。

候选读取参考标注RGB中局部空间频率份额，区别于query颜色路径/边界和参考silhouette几何。它可暂作为新信息机制候选，但**现在计0**；只有默认实现、区别与检查齐备后再由root决定是否登记独立1。FFT/纹理原型算子不保证科学原创性，合成正例不保证跨实例或真实mIoU收益。当前仅提交此合同等待root接受，没有任何运行。
