# Task constellation同原600完整结果：淘汰，不扩批或调配方

六完整臂/CLI与original共7200 Mask全部先seal后GT，直接I/U、S保护、实际编辑预算、class/fold与真实solver核验完成；root另69330项核验通过。**完整方法51.591958低于强S52.005138，CLI也下降，按预先门槛淘汰。Strong主目标未完成。** 不扩6200，不改pose、scale、iteration、assignment、采样、warp或新增本配方变体；不拿初始pose等消融替换主方法。

| 同原600完整输出，observed class/fold mIoU | 原图 | CLI1024 |
|---|---:|---:|
| 完整FoRIS B | 47.779371 | 47.619569 |
| 现有强S | **52.005138** | **51.702409** |
| 完整task constellation | 51.591958 | 51.300500 |
| 初始语义匹配pose，无联合迭代 | 51.917269 | 51.616307 |
| FG descriptor/坐标绑定打乱 | 51.655964 | 51.365081 |
| 无参考BG项的完整pose | 51.590281 | 51.299072 |
| 无模板约束、宽接受支持恢复 | 50.681671 | 50.409562 |
| S同实际新增数量，原FoRIS排序 | 51.624104 | 51.338130 |

完整方法对FoRIS原图+3.812587并不满足目标，因为主要强处理S已经52.005138。对宽恢复+0.910288只表明几何模板减少部分错误新增；仍低同预算位置控制0.032146，并低初始pose0.325311、打乱绑定0.064005。BG完整对照只差+0.001677，不能当作有实用收益的语义否决。

## 保留有效处理，只修明确冲突的实际兑现

本候选沿用原S所有前景、原reference Mask及冻结whole R/Q O24。原窗口Mask、margin和region selections均不重跑。恢复域只含`~S & B & old_FP32_render(acceptedAny)`，额外要求合法RMask经粒子匹配得到的同一正scale/正orientation相似变换的模板支持。特征-位置绑定是真正不同于仅改平票的约束，但注册/Procrustes此前R1-G已用过，不能称注册本身原创。

RFG512、Q强S128、RBG128等质量粒子保留重复native ID。一对一指粒子槽，512→128仅匹配R子集，不是全部R被解释；未匹配RFG仍随完整Mask warp。已知宽高resize失真先解除，pose在physical坐标拟合。固定初始语义assignment/pose后5轮联合`feature+.5geometry²`匹配与pose优化，每一步核同目标单调；各frame独立CPU64 inverse-warp Mask、strict>.5，原S/B与accepted支持域遵守原frame口径。

候选只新增，不删除。原图相对S补**239210真/360267假**，0删除；186例改善、302例下降、112例相同。相对宽恢复删掉172189真/503903假，尽管减少大量FP，同时损害本来可恢复的目标。

同新增数量控制补239566真/359911假；主方法总体多356假、少356真。原图472例、CLI499例预算pool严格大于非零新增数量，是实际非退化位置控制；原图48例、CLI39例主方法等于整个宽恢复，另有零新增情况，均按实际记录，不能把预算名义当作全例非退化。

## 必要对照定位新增错误

与“初始语义匹配pose”相比，完整联合迭代原图：

| 同一读出下的四编辑 | 像素 |
|---|---:|
| 新增正确FG | 120202 |
| 新增错误FG | 230452 |
| 删除正确FG | 11299 |
| 删除错误FG | 15998 |

这改变的是assignment/pose优化步骤，Mask warp、恢复域和最终读出固定。真实solver能量单调下降，但完整mIoU下降0.325311。**源/伪角色自一致、坐标优化自一致都不能替代query任务判断正确性。** 本对照不能进一步分离assignment、pose更新与warp误差，也不能说这些步骤完全丢失任务信息。

新增相对S均在完整B内。原图236658真/356839假属于raw多数已失败处的恢复；2552真/3428假属于raw多数本成立但gate改变了决定处的恢复。没有恢复渲染后无任意局部支持的目标。99%左右新增来自前一类，仍不是只修正确的coverage损失：该类也存在大量应当保留的S否决。

原图full/shuffle共同有效587例贡献固定600差−0.037831pp，其余13例贡献−0.026175pp。full/noBG共同有效588例贡献+0.014105pp，其余12例贡献−0.012427pp。不能忽略fallback构成，把总差全归语义绑定/BG信息。

## 条件退化与信息边界

full和初始pose各588有效/12回S；shuffle594/6；noBG596/4。full回退为3个单native支持点normal frame、1个缺粒子质量、8个初始匹配全部落在同一个reference坐标。后三者的一对一槽匹配依然可对应同一物理点，因为粒子保留重复ID。

若全部matched reference坐标相同，中心化X为零，相似变换的scale/rotation不能由这组correspondence识别，程序精确检查后回S。这仅是**这个配准表示/匹配结果**的条件证书；合法reference完整Mask和其它FG坐标仍可能存在，不能称整个DINO、全任务或五域不可恢复，更不能以去重/改采样继续本配方。

596个可计算cost的例中192个BG penalty完全是reference列常量，404个有>1e−12非可分interaction，没有纯query行常量例。列常量可影响R子集选择，但不能改变固定子集内对应；混合项也不能自证语义。完整无BG对照已否定把这个项当作明确净收益。

单pose还假设对应具有相似几何；多实例、非刚性和视角变化可能不满足。已有S错误前景和S全空的任务本方案不处理。仅LVIS600，没有新增跨域成功或同query换reference Mask的因果结果。

## 实际资源与复现资产

生产**196.673s/CPU196.481s/RSS1295564800B（1.207GiB）**；600例均使用MPS backend，596例实际计算MPS32 feature cost，4例早期退化不计算cost；CPU64匹配/pose/warp，0MPS失败/CPU重试。评分9.858s；保存Mask逐步归因6.603s/CPU6.751s/RSS0.531GiB。生产中11274次assignment、11274次pose尝试、11256次成功pose拟合；这些是每episode临时对应/几何参数，不更新DINO或global模型权重。

1200旧R/Q请求、1166 unique、18.67GB逻辑container请求，IO约79.85s，**不是物理磁盘IO**。特征cost/配准30.06s包含GPU/搬运/CPU，并非纯GPU利用率；dual-frame Mask warp/组合/预算68.97s。MPS driver返回后采样最大1101807616B（1.026GiB），allocated返回时0，不是真峰值/利用率。0encoder、FoRIS、CRF、RGB读取、新编码view、sample、global训练、raw写入或densearchive。首3中1例退化无cost，另2例MPS/FP64 cost最大误差1.10e−6/1.54e−6，只记误差，不改precision/生产选择。

新增packed最终Mask12.12MB、逐例生产收据10.67MB，原大缓存不复制。评分600GT，后seal逐步归因另600GT；均无监督/阈值选择用途。当前磁盘139GiB空闲。producer PID75258/session81188、scorer session86158、postseal归因session88853均成功退出；不重启历史handles。

冻结[完整方案](TASK_CONSTELLATION600_PROTOCOL_20261011.md)，protocol `f73da7f6…`、core `7eaaab93…`、seal `5b1307ff…`。原数据/权重/raw/有效S保留；主要资产：

- [protocol](../../evidence/local/task_constellation600_20261011/protocol.json)、[源/core](../../../evidence/local/task_constellation600_20261011/core.py)、[producer](../../../evidence/local/task_constellation600_20261011/producer.py)、[静态审查](../../evidence/local/task_constellation600_20261011/static_review.json)、[代数检查](../../evidence/local/task_constellation600_20261011/numeric_sanity.json)。
- [全7200先seal收据](../../evidence/local/task_constellation600_20261011/sealed.json)、[实际资源](../../evidence/local/task_constellation600_20261011/production_receipt.json)、[完整scores/四编辑/配对与cost](../../evidence/local/task_constellation600_20261011/report.json)、[600逐例I/U](../../evidence/local/task_constellation600_20261011/episodes.jsonl)、[screening decision](../../evidence/local/task_constellation600_20261011/screening_decision.json)。
- [69330项root核验](../../evidence/local/task_constellation600_20261011/root_verification.json)、[逐步四编辑/共同有效分解](../../evidence/local/task_constellation600_20261011/postseal_attribution.json)、[归因源码](../../evidence/local/task_constellation600_20261011/postseal_attribution.py)、[completion/hash入口](../../evidence/local/task_constellation600_20261011/completion.json)。原最终Mask在同目录`predictions/000000.npz`至`000599.npz`，实际匹配/pose/退化收据在`records/`。

前一轮[原S阶段复原](STRONG_STAGE600_RESULTS_20261011.md)仍是有效证据，不能因新配准失败降低Strong目标。下一方向需要提供任务身份的额外可否定证据，不能继续本配准/九窗融合或用优化更一致的whole字段任意扩大恢复。
