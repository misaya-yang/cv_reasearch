# 主线代码

[目标与状态](../docs/research/PLAN.md) · [结果](../evidence/local/RESULTS.md)。共21个脚本；没有自动队列。

| 要做什么 | 入口 |
|---|---|
| 完整基线 | [run_foris.py](run_foris.py)、[run_insid3_complete.py](run_insid3_complete.py) |
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

共享代码仅9个文件：数据/完整FoRIS适配/位置基底、统一统计与读出、原始NN、阶段拆分和RCG。
`launch/part1_producer_two_pair_v1/snapshot`仅保留10个不可变运行来源，供残差转移的prefix哈希绑定；已无启动plan。
direct mean中的uniform `state.json`是可选历史失败信号，本机此前就缺失；旧启动包已退役，输出seal检查保留。
旧runner与其他快照从[历史记录](../docs/archive/2026-10-07-cleanup.md)恢复。完整运行仍需[外部资产](../docs/harness/SERVER.md)。
