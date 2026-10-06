# Exact existing A/B family: global DEV optimum

All 11,534,336 recipes were evaluated. Maximum class-summed mIoU **63.062305**, outer complete-mask producer count proxy **3**.

GT selected this single global recipe; inference uses no query GT. All actual finalist masks sealed before final GT recount; every per-draw I/U matches membership-histogram predictions exactly. Paired confidence intervals are conditional on the selected recipe, not selection-adjusted. This is benchmark DEV reuse, not untouched-test/SOTA.

Selected recipe3: P0=`claude_official_4000_existing_masks::conservative_delete`; additions=['frozen_subtoken4000_v1::fine.rcg64']; deletions=['frozen_subtoken4000_v1::fine.rcg16.control'].

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::native | +2.130564 [+1.748400, +2.492689] |
| frozen_subtoken4000_v1::rcg | +0.728633 [+0.575339, +0.890289] |
| frozen_subtoken4000_v1::mean.control | +0.549333 [+0.298681, +0.767327] |
| frozen_subtoken4000_v1::rcg64.control | +0.764367 [+0.601426, +0.981339] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.353205 [+0.189976, +0.501315] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.410687 [+0.242400, +0.620334] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.317947 [+0.071690, +0.527497] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +0.888417 [+0.563798, +1.256587] |
| claude_official_4000_existing_masks::conservative_delete | +2.183027 [+1.501792, +2.921292] |
| claude_official_4000_existing_masks::external_mean__delete | +1.209054 [+0.706521, +1.739236] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +1.121799 [+0.662174, +1.630371] |

## Full exact gain-cost Pareto

| Outer producer-count proxy | mIoU |
|---|---:|
| 1 | 62.744357 |
| 2 | 62.990095 |
| 3 | 63.062305 |
