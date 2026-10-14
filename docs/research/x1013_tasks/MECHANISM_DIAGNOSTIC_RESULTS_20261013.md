# 细结构与像素修正诊断结果

日期：2026-10-13。复用 R2 已封存输出（Fundus 150、DeepGlobe Road 100）和 x1013 S 半（Fundus 108、Road 62），共 420 例。使用 8 连通分量、Zhang–Suen 骨架及 clDice-style skeleton precision/recall harmonic mean；按 query photo 做配对 bootstrap，R2 1,000 次，x1013 S 1,000 次。所有结果都只读已缓存预测与 GT，无新模型构造、DINOv3 推理或 T 评分。输入哈希在 `../../../../cv_data/a/x1013_mechanism_diagnostics/summary.json`；逐例记录见[压缩证据](../../../evidence/research/x1013_mechanism_diagnostics_iu.jsonl.gz)。

## R2：Fundus 的 IoU 增益不是拓扑恢复

Fundus 150 例上，真实像素修正相对 `VP_identity` 的 IoU 提高 **+2.62 pp** [2.41, 2.85]；相对块内像素置换提高 **+2.49 pp** [2.27, 2.70]。但相对 identity 的 clDice-style 变化为 **−0.15 pp** [−1.19, +0.81]，区间跨 0；骨架 precision **−11.38 pp** [−13.38, −9.41]，recall **+3.15 pp** [+2.91, +3.39]。假阳性预测连通分量增加 **104.53 个/图** [+92.36, +117.11]，被拆分的 GT 分量增加 **0.28 个/图** [+0.16, +0.42]。

真实像素修正没有胜过置换控制的 clDice-style 指标：pixel − pixel_permuted 为 +0.51 pp [−0.41, +1.36]。因此现有 Fundus IoU 收益符合“提高覆盖率、牺牲精度并引入碎片”的解释，没有证据说它恢复了血管拓扑。

## R2：Road 上像素修正伤害中心线质量

Road 100 例上，pixel 相对 `VP_identity` 的 IoU 变化为 −0.01 pp [−0.27, +0.24]，而 clDice-style 下降 **4.56 pp** [−5.94, −3.29]，骨架 precision 下降 **8.62 pp** [−10.71, −6.71]；假阳性分量增加 **32.41 个/图** [+23.44, +42.55]。它没有提供可迁移到 Road 的细结构修正。

## x1013 S：C1 在 Fundus 不改变掩码，在 Road 损害中心线

Fundus 108 例上 A4 与 C1 掩码及所有结构指标完全相同；其平均 clDice-style 为 0.467。Road 62 例上 C1 相对 A4 的 IoU 仅 −0.06 pp [−0.46, +0.36]，但 clDice-style **−2.05 pp** [−3.25, −0.96]，骨架 precision −4.30 pp [−6.84, −2.04]、recall −1.34 pp [−2.42, −0.46]，假阳性分量增加 **21.84 个/图** [+9.70, +34.04]。该 CRF 输出没有改善 Road 的连通结构。

候选连续场在 GT 骨架上的平均分数为 Fundus **0.383**（P10 0.081，66.2% 骨架分数 `<=0.5`），Road **0.518**（P10 0.218，45.3% `<=0.5`）。这说明不少真骨架位置落在当前硬切点以下；它是范围/支持的诊断信号，不能单独证明仅改阈值就能得到正确边界。

## 决策

当前数据不支持“像素修正或 CRF 恢复了 Fundus/Road 拓扑”的主张。Fundus 的像素干预有 IoU 增益，但 clDice-style 没有可信提升，且预测碎片显著增加；Road 的像素干预与 CRF 都降低中心线质量。后续应把分数支持/范围问题与拓扑连接机制分开设计，不以组件数量变少单独判胜。相对几何方向仍需先完成数据标注资格审计。
