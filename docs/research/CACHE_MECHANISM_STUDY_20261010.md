# 使用已有原始特征定位角色判别失效

用户明确要求深入利用约320GB特征，继续并行研究。当前R1-G同Deep100/PACO100大幅低于同批FoRIS，基础unary即已误扩张；几何实际只改约.6%/.8%原图像素。合成正确性、模型能量或理论纠错上限不能替代最终净分；当前版本不采用、不扩跑参数或样本。

本轮不先堆新模块。使用现有输入身份与缓存，分清判别信息、观察尺度和读出规则的失效点。目录为`cv_data/a/cache_mechanism_study_20261010/`，四轨并行：

1. 缓存资产地图：按实际profile/层/视图/数据集列metadata覆盖和已read范围，复用现成清单/收据，不SHA整个320GB，不把总容量当任何新视图都已有。
2. 有效前端差异：同批FoRIS和新法在归一化/APD、参考角色、面积权重、合法参考crop、全局/局部匹配、阈值上的精确差异。实现差异先记录，原因需对照，不能先叫根因。
3. 查询读出能力：只读已封存200概率/GT，计算排序AUROC、固定.5的前景面积与GT面积、同排序GT oracle阈值的能力上限。oracle只诊断排序与读出的区别，不成为生产阈值、真实新方法mIoU或涨分。
4. 参考角色自一致性：实际读取同200合法reference原O24，检查完整role LSE在已知reference纯BG/FG上是否已混淆，附leave-self-patch-out的重新归一化及max聚合对照。所有规则固定；self-match的容易正确不作query泛化证明。仅参考feature/label诊断，不新增query分割。

除第4轨实际所需reference原特征读取，其余优先使用已保存字段。统一aidemo、codex_m4；每个数值进程最多2CPU线程，不构造DINO、不重新编码、不补缺视图、不下载、不改默认方法或其他聊天的冻结运行。允许封存GT评价，禁止用query GT选择生产公式或参数。源配置、输入及结果身份单独保存，旧6000/官方全量/已失败队列不恢复。

结果要回答可操作的问题：大幅误检是否来自基本排序无效、参考身份本身混淆、跨视图证据不一致，或.5读出与分数不相容。结论以实际缓存证据区分；在明确规则前，不把另一套完整求解器作为下一步。最终研究目标仍是同条件完整mIoU涨分及可用成本，本轮这些诊断不是目标已达成。

## 四轨完成后的具体对照

四轨均已完成，汇总见`cv_data/a/cache_mechanism_study_20261010/SYNTHESIS.md`。原unary的逐例全图AUC为Deep .9155/PACO .9036，固定.5预测面积是真实目标6.938/3.094倍；几何只改约.6%/.8%像素。全图AUC不替代困难区域区分能力，oracle阈值不作为部署参数。实际reference自回放LOO发现原角色密度BG误报7.786%，直接max又因mixed双角色tie大量漏掉道路。共享逆重复度固定对照84.61秒完成，在Deep连续FN和PACO纯BG FPR各有退步，停止该候选于reference诊断，不进行其query试验。

以这些证据落实一个最简单的读出检验：仅原reference LOO分数与合法reference mask确定连续coverage的balanced-risk最小阈值；使用完整score组与strict>规则。首次冻结/输出前经独立审阅将最优风险tie明确为实际参考概率FP64阈值theta距离.5最小，.5已最优即保留，不强制区间中点；完整概率ties整组，区间端点在概率FP64下nextafter，不先在logit nextafter再sigmoid。同距离取较高theta。原200已保存query P0先按原策略bilinear至原帧，再用FP64 theta直接strict>输出；代码、阈值及全部200预测封存后才读query GT评分。不重新编码、不新样本、不接density或BP、不改变默认方法。它检验参考阈值能否迁移，允许失败；不是预设采用，也不把参考自拟合当成query有效。产物`reference_calibrated_readout/`。

独立dense_role_unary只准备去掉候选/BP的同unary接口，五组合成输入与原字段逐字节一致；尚未测实际加速。新density模块列序FG/BG，而head为BG/FG，将来接入必须显式重排并记录身份；目前未接入。

## 本轮完成结果

唯一参考校准查询对照200/200完成，Deep18.5684/PACO38.1495，虽比原P0增加5.9377/8.9121，仍比同批FoRIS低10.8091/7.4144。真实读取既存概率→参考阈值→200封存输出→GT评分共3.78575秒，新增DINO/BP/raw读取0；不是冷端到端。全部来源/阈值/预测和I/U收据保留，不采用、不扩跑。计划四轨与该单一实际对照已结束，目标未达成；未排队下一批。最新实际状态以`reference_calibrated_readout/COMPLETE.json`、`result.json`及`SYNTHESIS.md`为准。
