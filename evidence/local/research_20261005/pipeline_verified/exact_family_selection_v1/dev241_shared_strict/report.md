# Exact existing A/B family: global DEV optimum

All 16,777,216 recipes were evaluated. Maximum class-summed mIoU **62.773896**, outer complete-mask producer count proxy **4**.

GT selected this single global recipe; inference uses no query GT. All actual finalist masks sealed before final GT recount; every per-draw I/U matches membership-histogram predictions exactly. Paired confidence intervals are conditional on the selected recipe, not selection-adjusted. This is benchmark DEV reuse, not untouched-test/SOTA.

Selected recipe3: P0=`frozen_fine_raw_dev241_v1::model.raw_nn`; additions=['claude_official_4000_existing_masks::conservative_delete']; deletions=['frozen_subtoken4000_v1::fine.rcg16.control', 'claude_official_4000_existing_masks::external_mean__delete'].

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::native | +3.699071 [+1.313321, +5.115718] |
| frozen_subtoken4000_v1::rcg | +1.754227 [-0.276500, +2.859814] |
| frozen_subtoken4000_v1::mean.control | +2.093780 [-0.202797, +3.226823] |
| frozen_subtoken4000_v1::rcg64.control | +2.645216 [+0.393990, +4.632383] |
| frozen_subtoken4000_v1::fine.rcg16.control | +1.273235 [-0.867504, +2.318765] |
| frozen_subtoken4000_v1::fine.rcg64 | +2.257985 [-0.080335, +4.190677] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +1.822679 [-0.494383, +2.911577] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +0.932115 [-0.476382, +1.987969] |
| claude_official_4000_existing_masks::conservative_delete | +0.158063 [-0.665691, +1.442439] |
| claude_official_4000_existing_masks::external_mean__delete | +0.119553 [-0.340452, +0.747647] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +0.982040 [-0.228306, +2.162249] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +1.723028 [-0.685493, +2.849106] |
| frozen_fine_raw_dev241_v1::model.raw_nn | +19.870009 [+16.577543, +23.236017] |

## Full exact gain-cost Pareto

| Outer producer-count proxy | mIoU |
|---|---:|
| 1 | 42.903888 |
| 2 | 62.654343 |
| 3 | 62.750191 |
| 4 | 62.773896 |
