# 关键结果：快、强与净纠错

更新：2026-10-07。本页只保留主线证据；完整历史账本可从[清理前版本](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md)读取。
本轮没有新实验；原始报告、逐例数据、数组和封存记录没有改写。

## Strong：已有完整优势，尚非最终SOTA结论

同一公开4000抽样、1024读出、类别汇总I/U再平均；复用了开发数据，不是独立新数据确认。

| 完整构造 | mIoU | 能说明什么 |
|---|---:|---|
| FoRIS | 60.931741 | 完整主参照 |
| RCG / MEAN | 62.333671 / 62.512972 | 简单空间读出已带来明显增益 |
| fine16 / fine64 | 62.709100 / 62.651617 | 更复杂fine64没有胜过fine16 |
| 固定掩码增删转移 | 62.744357 | 相对MEAN +0.231385；不证明胜过fine16 |
| 固定标量残差转移 | 62.844537 | 相对MEAN +0.331565 [0.303193,0.372805] |
| 12源库外层留折配方 | 62.964699 | 相对FoRIS +2.032958 [1.657151,2.387729]；仍为开发复用 |
| 13源扩展库外层留折配方 | 63.102199 | 含sizecut_foldtemp；底层尺寸切点已使用fresh600标签，非整套独立留折确认 |

来源：[完整细读出读数](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md#2026-10-06-complete-frozen-fine4000-and-fixed-ab-transfer-control)、[固定残差](research_20261005/pipeline_verified/mean_fine_residual_transfer4000_v1/report.md)、
[12源](research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v3_crossfold/report.md)、
[13源](research_20261005/pipeline_verified/exact_family_selection_v1/public4000_v4_extended_crossfold/report.md)。
并非没有全量输出；缺的是按当前主线确定的最终完整方法、同条件强对照及完整成本验证。
这些成绩不能挂给尚未运行的原始DINO新构造，也不能证明整个分割领域SOTA。

## Fast：方向有线索，完整优势尚未交付

固定600、原图分辨率的[既有读数](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/evidence/local/RESULTS.md#2026-10-07-fixed-600-scoreboard-cpu-server-56464-read-only-tally-by-claude)：
FoRIS 61.5627，INSID3双线性无CRF 56.3812，原始最近邻43.0828，位置去偏后的简单D_I图方案48.0981。
这与公开4000/1024、fresh600/64×64不是同一比较，不能相减。

[D汇报](pro_cards_20261008/D_report.md)：fresh600位置级L4直接53.23，相对s2 +1.50；图后55.87，相对s2 +0.48、区间跨零。
L4是简化线索，未证明完整Fast已超过INSID3/FoRIS。卡1无收益；卡2只做零编码控制，完整注意力隔离未执行；B候选复读未运行。
FoRIS作者已报告512px达到59.5/258ms（RTX3090），完整效率比较须面对这个合理配置；
[来源：FoRIS附录D](https://arxiv.org/html/2609.03384v1#A4)。这不是本仓库的同机测速。

## 增删与组合：哪些认识仍有用

- 12源全局最优63.064818，相对11源只+0.002513，尽管标量残差单独相对MEAN有+0.331565；组件收益不可相加。
- 原始NN作为共同起点已有DEV241记录；相应全4000的原始输出尚缺，不拿FoRIS输出冒充原始起点。
- “17.4%没找对”的操作定义是最佳阈值IoU仍<0.5，可能含严重范围错误，不等于纯身份误识别率；真值oracle不是可得增益。
- 已测参考相似度/阈值估计、复杂图引导的负结果限定那些构造，不证明完整特征已经没有可用信息。
- 数百批量方法未产生主线完整胜出；[既有失败统计](cpu100_20261006/METHODS_AND_FAILURES.md)保留，旧方法卡与执行队列退役。

相对共同base统一报告补回TP、误加FP、误删TP、去掉FP四项最终净变化，逐类计算，再报告完整mIoU与成本。
这是一套分析坐标；只有推理时的可观测依据和完整优势能支持新方法贡献。当前工作仅见[PLAN](../../docs/research/PLAN.md)。
