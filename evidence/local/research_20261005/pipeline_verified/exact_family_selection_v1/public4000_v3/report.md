# Exact existing A/B family: global DEV optimum

All 50,331,648 recipes were evaluated. Maximum class-summed mIoU **63.064818**, outer complete-mask producer count proxy **3**.

GT selected this single global recipe; inference uses no query GT. All actual finalist masks sealed before final GT recount; every per-draw I/U matches membership-histogram predictions exactly. Paired confidence intervals are conditional on the selected recipe, not selection-adjusted. This is benchmark DEV reuse, not untouched-test/SOTA.

Selected recipe3: P0=`frozen_subtoken4000_v1::fine.rcg16.control`; additions=['frozen_subtoken4000_v1::fine.rcg64']; deletions=['mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1'].

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::native | +2.133077 [+1.772388, +2.463718] |
| frozen_subtoken4000_v1::rcg | +0.731146 [+0.566883, +0.901895] |
| frozen_subtoken4000_v1::mean.control | +0.551846 [+0.394417, +0.710822] |
| frozen_subtoken4000_v1::rcg64.control | +0.766881 [+0.552813, +1.047521] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.355718 [+0.189038, +0.513917] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.413200 [+0.192796, +0.687461] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.320460 [+0.163093, +0.466249] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +0.890930 [+0.582943, +1.269915] |
| claude_official_4000_existing_masks::conservative_delete | +2.185540 [+1.503942, +2.887448] |
| claude_official_4000_existing_masks::external_mean__delete | +1.211567 [+0.718418, +1.737787] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +1.124312 [+0.678142, +1.630908] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.220281 [+0.057350, +0.366154] |

## Full exact gain-cost Pareto

| Outer producer-count proxy | mIoU |
|---|---:|
| 1 | 62.844537 |
| 2 | 62.990095 |
| 3 | 63.064818 |
