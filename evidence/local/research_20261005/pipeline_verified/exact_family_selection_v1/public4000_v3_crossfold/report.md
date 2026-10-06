# Fold-held existing A/B family selection

Class-summed mIoU: 62.964699.

Each fold uses one fixed recipe chosen from the other three folds. Source-producer GT exposure is unchanged; supplied all600-fitted sizecut remains label-fitted across held folds. Conditional CIs are not selection-adjusted and this benchmark is development reuse.

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::mean.control | +0.451727 [+0.233044, +0.659919] |
| frozen_subtoken4000_v1::rcg64.control | +0.666762 [+0.474467, +0.896735] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.255599 [+0.094337, +0.399785] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.220341 [+0.009010, +0.415947] |
| native | +2.032958 [+1.657151, +2.387729] |
| frozen_subtoken4000_v1::native | +2.032958 [+1.657151, +2.387729] |
| frozen_subtoken4000_v1::rcg | +0.631027 [+0.475966, +0.790570] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.313081 [+0.117710, +0.538295] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +0.790811 [+0.457892, +1.158168] |
| claude_official_4000_existing_masks::conservative_delete | +2.085421 [+1.378319, +2.794837] |
| claude_official_4000_existing_masks::external_mean__delete | +1.111448 [+0.600677, +1.632109] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +1.024193 [+0.564547, +1.524333] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.120162 [-0.103035, +0.319822] |
