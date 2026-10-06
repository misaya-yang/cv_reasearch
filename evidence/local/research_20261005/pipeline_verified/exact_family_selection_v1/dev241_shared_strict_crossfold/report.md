# Fold-held existing A/B family selection

Class-summed mIoU: 61.284097.

Each fold uses one fixed recipe chosen from the other three folds. Source-producer GT exposure is unchanged; supplied all600-fitted sizecut remains label-fitted across held folds. Conditional CIs are not selection-adjusted and this benchmark is development reuse.

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::mean.control | +0.603980 [-1.452451, +1.788880] |
| frozen_subtoken4000_v1::rcg64.control | +1.155417 [-0.827827, +3.088186] |
| frozen_subtoken4000_v1::fine.rcg16.control | -0.216565 [-2.276171, +0.993915] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.332879 [-1.777091, +1.485284] |
| native | +2.209272 [-0.023515, +3.778239] |
| frozen_subtoken4000_v1::native | +2.209272 [-0.023515, +3.778239] |
| frozen_subtoken4000_v1::rcg | +0.264427 [-1.695256, +1.561891] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.768185 [-1.328812, +2.712827] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | -0.557685 [-1.676121, +0.433567] |
| claude_official_4000_existing_masks::conservative_delete | -1.331737 [-2.360143, +0.446079] |
| claude_official_4000_existing_masks::external_mean__delete | -1.370246 [-2.281056, -0.067740] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | -0.507760 [-1.309529, +0.494694] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.233228 [-1.927722, +1.417502] |
| frozen_fine_raw_dev241_v1::model.raw_nn | +18.380209 [+15.131886, +21.999212] |
