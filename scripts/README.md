# 主线代码

[目标与状态](../docs/research/PLAN.md) · [结果](../evidence/local/RESULTS.md)。共27个脚本；下一项固定实验见PLAN。

| 要做什么 | 入口 |
|---|---|
| 完整基线 | [run_foris.py](run_foris.py)、[run_insid3_complete.py](run_insid3_complete.py) |
| M4 本机完整 FoRIS/RCG/MEAN | [run_m4_baselines.py](run_m4_baselines.py)；MPS 编码、CPU 原流程与适配 CRF，二值 mask pack，推理/评分分离 |
| M4 SAFR 组合候选 | [run_m4_safr.py](run_m4_safr.py)；基于封存基线比较 SAFR 正负原型 guide 与同图读出，ViT-L 适配，不冒充作者默认 ViT-B 结果 |
| Mac CRF 适配检查 | [check_m4_crf.py](check_m4_crf.py)；原 CPU lattice 的 FP32/NHWC 绑定与滤波/求解检查 |
| 固定L4/s2证据与错图记录 | [bench_evidence.py](bench_evidence.py)、[score_reverse_landing.py](score_reverse_landing.py)；原CLI仍为64×64筛选台，不是尚未实现的原图L4+投票完整runner |
| 原始起点、两套基线的阶段增删 | [run_stage_bank.py](run_stage_bank.py)；原始NN独立实现位于`ics.methods.raw_reference_origin` |
| 既有强方法的细读出与评分 | [run_frozen_subtoken4000.py](run_frozen_subtoken4000.py)、[score_frozen_subtoken4000.py](score_frozen_subtoken4000.py) |
| 固定增删/残差组合 | [run_mean_fine_bit_transfer4000.py](run_mean_fine_bit_transfer4000.py)、[run_mean_fine_residual_transfer4000.py](run_mean_fine_residual_transfer4000.py) |
| 强简单对照 | [run_direct_mean_fine4000.py](run_direct_mean_fine4000.py) |
| 已有尺寸切分配方 | [run_uniform_tau15_sizecut4000.py](run_uniform_tau15_sizecut4000.py)、[replay_size_cut_composition_v1.py](replay_size_cut_composition_v1.py)；其标签暴露见RESULTS |
| 已有完整掩码库与固定配方比较 | [build_complete_mask_library_v1.py](build_complete_mask_library_v1.py)、[build_dev241_shared4000_library_v1.py](build_dev241_shared4000_library_v1.py)、[run_exact_family_selection.py](run_exact_family_selection.py)、[run_exact_family_crossfold.py](run_exact_family_crossfold.py) |
| 必要输入与错误输出 | [export_confirm_cache.py](export_confirm_cache.py)、[score_error_budget.py](score_error_budget.py)、[render_failures.py](render_failures.py) |

依赖脚本：[run_frozen_subtoken1200.py](run_frozen_subtoken1200.py)、[run_fine_mean1200.py](run_fine_mean1200.py)、
[benchmark_part1_producer.py](benchmark_part1_producer.py)、[run_fused_tau15_sizecut4000.py](run_fused_tau15_sizecut4000.py)。
这些名称中的1200/fused沿用既有来源身份；本轮保留代码字节，不重命名或改算法。
bench_evidence封存源码还读取 [run_order_tokens.py](run_order_tokens.py)、[run_rcg2.py](run_rcg2.py)、[score_order_tokens.py](score_order_tokens.py)、[score_size_threshold.py](score_size_threshold.py)，四者随入口恢复。

共享代码仅9个文件：数据/完整FoRIS适配/位置基底、统一统计与读出、原始NN、阶段拆分和RCG。
`launch/part1_producer_two_pair_v1/snapshot`仅保留10个不可变运行来源，供残差转移的prefix哈希绑定；已无启动plan。
direct mean中的uniform `state.json`是可选历史失败信号，本机此前就缺失；旧启动包已退役，输出seal检查保留。
旧runner与其他快照从[历史记录](../docs/archive/2026-10-07-cleanup.md)恢复。完整运行仍需[外部资产](../docs/harness/SERVER.md)。
