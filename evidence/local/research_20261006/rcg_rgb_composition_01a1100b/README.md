# 固定RCG + RGB局部分歧读出组合：唯一600例验证已完成

**完整class-summed mIoU：63.125362。** 相对当前RCG为+0.036362 [−0.015595,0.135117]，稳定增量未确立；相对fine16仍低−0.339165 [−0.504828,−0.283429]。正增量不等于已建立优势，完成本次指定组合后不自动扩展变体或参数搜索。

## 唯一固定合同

`rcg.predict(q,r,cov,rawscore,device='cpu')`重算当前64×64场及原renderer的1024mask。将该RCG场送入已经固定的RGB128 Potts：ell=2*bilinear(RCG64,128)−1，全部四邻域、RGB平方差median/floor1e−4、平均加权度归一、Potts .25。没有改变tau、lambda、容量、decoder或图。

同源RGBcut与same128unary各自按原Pro二级renderer得到1024mask。`D=cut XOR unary`，`final=where(D,cut,currentRCG)`；D外精确保留当前RCG。最终1024field为二值完整mask，不声称是原128 Potts目标的全局最优解。cut=unary时直接退化为当前RCG。它是同信息的固定组合，independent increment=0，不声称新信息或新原创机制。

保留当前RCG64连续场/currentMEAN64连续场、fullcut128、sameunary128、finalbinary1024，并保存所有五个完整1024mask及主方法原尺寸mask。currentMEAN用当前huber pregraph + original-dtype MEAN CG producer重算；既有historicalnative/RCG/fine16/fine64通过同绑定评测接口保留。

## 4例数值与producer检查

独立新namespace的smoke4只有1worker×2threads（4GiB预算）。重复直接RCG调用：四例field最大误差0、1024mask XOR0；RCG/MEAN field64、fullcut/sameunary128、finalfield1024，均未压缩错误网格。历史RCG四例XOR也恰为0，**只记录，不以此强制改当前producer**。四例未读取query GT，全部mask已封存，不以smoke IoU选配方。smoke总16.656935秒，包含重复RCG数值audit，保守parent+worker峰值1,432,076,288bytes。

之后按同冻结源码/配方运行唯一600例；主队列没有重复数值audit。整个过程没有修改共享prepared入口的base=mean保护、原方法、M4、旧outputs、PLAN或Git。

## 完整实测与边界

同reused public600：600 occurrences、80类、570原照片连通组。全600新预测/field/receipt先封存，单独score才读取query GT。配对95%区间沿用项目2000次RandomState(0)原照片连通组bootstrap；这是exploratory composition，不是独立确认。

| 完整行 | class mIoU | 主组合减该行，95%区间 |
|---|---:|---|
| current recomputed RCG | 63.089000 | +0.036362 [−0.015595,0.135117] |
| historical RCG | 63.088942 | +0.036421 [−0.015541,0.135186] |
| current MEAN | 62.931546 | +0.193816 [−0.246505,0.567217] |
| RCG + RGB fullcut | 63.038936 | +0.086426 [0.055232,0.157114] |
| RCG same128 unary | 62.949403 | +0.175960 [0.131594,0.308443] |
| full native historical | 61.335314 | +1.790048 [1.119833,2.439521] |
| fine16 | 63.464527 | −0.339165 [−0.504828,−0.283429] |
| fine64 | 63.681184 | −0.555822 [−1.279984,0.736232] |

这些都来自新完整mask实际计数；没有将RCG−MEAN与旧MEAN+RGB增益相加。当前RCG与historicalRCG在全600仅有133像素XOR；独立重算currentRCG/主组合的完整计数均精确匹配score。没有因此替换baseline或更改producer。

## 改对/改坏像素（相对当前RCG）

| 动作 | 像素 |
|---|---:|
| 恢复真目标 added TP | 162,934 |
| 删除假目标 deleted FP | 308,236 |
| 错删真目标 deleted TP | 159,775 |
| 新增假目标 added FP | 265,843 |

改对471,170、改坏425,618、净改对45,552。D区域总1,194,578像素；全600 D外RCG XOR=0。像素净改善不替代class-summed mIoU与配对区间判断。

## 实际资源与成本

600推断6workers×2threads=12CPU，墙钟288.573464秒；保守汇总所有worker lifetime peak加parent peak为6,077,370,368bytes（5.66GiB，低于12GiB）。单例wall均值2.856453秒、p95 3.895603秒；包括currentMEAN作为对照的成本。

分阶段单例均值：RCG重算0.963524秒（p95 1.055332）、MEAN控制0.501325秒（p95 0.538456）、RGB组件1.327068秒（p95 2.332307）。RGB和RCG阶段是主方法成本；MEAN仅为同行控制，分别记账。没有新增DINO编码；首次cache producer的DINO成本未重新测量，所以这不是新RGB→frozenDINO端到端冷启动计时。

## 文件、入口与封存

实现：`src/ics/methods/rcg_rgb_proposal.py`。独立入口：`scripts/run_rcg_rgb_proposal.py infer/score`。后封存像素核验：`scripts/audit_rcg_rgb_result.py --out ...`，只在seal且score完成后打开GT，核对新规则、currentRCG计数及历史mask差。无权重/阈值搜索入口。

远端完整新结果：`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/rcg_rgb_composition_v1/public600`；smoke与冻结代码在同namespace的`smoke4/`和`code/`。

本地保留smoke metadata/四receipt、smoke_check、code_hashes、完整public600 config/seal/两个manifest、score/counts/bootstrap及post-score composition_pixel_audit。summary.json包含精确值及新seal、算法hash；600完整预测/field/receipt和原尺寸输出留在服务器。验证已完成，无需重复运行或扩大搜索。

本次Git提交包含实现、代码hash、smoke receipts、public600 config/seal/inference manifest及完整聚合报告；较大的evaluation manifest、逐例metrics、counts/bootstrap数组、压缩代码快照和完整预测保留在本地/服务器。
