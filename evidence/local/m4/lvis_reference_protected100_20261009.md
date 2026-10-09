# 同100：参考倾向参与校准的完整反证

当前参考倾向没有可靠区分被找回的目标与干扰，具体空间校准机制失败。继续使用旧100开发，未新增照片/编码。

| 方法 | CLI1024 | 原图 | CLI完全漏检例数 |
|---|---:|---:|---:|
| foris.crf | 47.704029 | 47.873899 | 1 |
| mean | 47.109370 | 47.182508 | 7 |
| foris.fg_anchor.crf | 50.047216 | 50.182680 | 9 |
| calibration.reference.crf | 48.284207 | 48.460004 | 5 |
| calibration.uniform.crf | 49.834391 | 49.962722 | 8 |

先按原双线性将native、anchor和封存参考FG/BG倾向p读到1024。low=min(native,anchor)、gap=abs(native−anchor)，reference=low+p*gap；uniform=low+γgap，γ=Σpgap/Σgap。两臂完整原CRF和原图回映射相同。无queryGT进入推断，不扫常数。
独立检查全部100字段、mask、I/U、编辑量与两帧一致；L1总校正匹配最大相对误差5.44e−8。p全1/0只保证CRF前两初始mask并/交，并不与非线性CRF交换。
相对uniform，reference补真141,996、补假336,558，删真0、删假4,405；精确点数为补真+1.180、补假−2.730、删真0、删假+0.00002，总差−1.550185。37例增57降6平、仅3折增7折降。原图差−1.502718。
相对anchor，reference完全漏检9→5，但完整分数下降1.763009；不能因为挽回几个零交集案例就当研究成立。找回TP/FP的p均值分别约.833/.829，现有p不够区分两者；未经看图标记的FP不全部称父物体。
前提准备目录`lvis_reference_protected100_20261009_preparation/checks.json`含独立标量、常数p、100原native初始mask/并交端点检查。八进程两臂100完整推断183.855秒，DINO0，原终点全部复用。
下一步检验源头BG统计是否被大量无关参考BG稀释：用参考FG相似度选择匹配的负例cloud，保留整个分布而非正交均值；先只生成同100源字段及独立标签诊断，不将AUC或机制说明当成新mIoU。该统计属于经典非参数hard negatives，不宣称新关系信息。
完整产物在`cv_data/a/lvis_reference_protected100_20261009/`；默认入口不变，无候选自动扩到1400、官方全量或DeepGlobe。
