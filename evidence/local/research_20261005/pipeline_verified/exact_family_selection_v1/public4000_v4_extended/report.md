# Exact existing A/B family: global DEV optimum

All 218,103,808 recipes were evaluated. Maximum class-summed mIoU **63.238541**, outer complete-mask producer count proxy **3**.

GT selected this single global recipe; inference uses no query GT. All actual finalist masks sealed before final GT recount; every per-draw I/U matches membership-histogram predictions exactly. Paired confidence intervals are conditional on the selected recipe, not selection-adjusted. This is benchmark DEV reuse, not untouched-test/SOTA.

Selected recipe3: P0=`claude_official_4000_existing_masks::conservative_delete`; additions=['size_cut_composition_cached4000_v2::fixed_size_cut_on_foldtemp_fine16.control']; deletions=['frozen_subtoken4000_v1::fine.rcg16.control'].

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::native | +2.306800 [+1.886799, +2.768258] |
| frozen_subtoken4000_v1::rcg | +0.904869 [+0.655739, +1.217794] |
| frozen_subtoken4000_v1::mean.control | +0.725569 [+0.400378, +1.094699] |
| frozen_subtoken4000_v1::rcg64.control | +0.940604 [+0.664307, +1.287852] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.529441 [+0.270524, +0.830262] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.586923 [+0.306381, +0.931534] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.494183 [+0.175501, +0.847018] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +1.064653 [+0.676643, +1.538586] |
| claude_official_4000_existing_masks::conservative_delete | +2.359263 [+1.604288, +3.199380] |
| claude_official_4000_existing_masks::external_mean__delete | +1.385290 [+0.818555, +2.038049] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +1.298035 [+0.767240, +1.918233] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.394004 [+0.063224, +0.749075] |
| size_cut_composition_cached4000_v2::fixed_size_cut_on_foldtemp_fine16.control | +0.089340 [-0.086422, +0.299029] |

## Full exact gain-cost Pareto

| Outer producer-count proxy | mIoU |
|---|---:|
| 1 | 63.149201 |
| 2 | 63.163237 |
| 3 | 63.238541 |
