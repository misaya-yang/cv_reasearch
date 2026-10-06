# Fold-held existing A/B family selection

Class-summed mIoU: 63.102199.

Each fold uses one fixed recipe chosen from the other three folds. Source-producer GT exposure is unchanged; supplied all600-fitted sizecut remains label-fitted across held folds. Conditional CIs are not selection-adjusted and this benchmark is development reuse.

| Baseline | Gain [95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::mean.control | +0.589227 [+0.271782, +0.951518] |
| frozen_subtoken4000_v1::rcg64.control | +0.804262 [+0.531680, +1.142899] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.393099 [+0.123564, +0.690722] |
| mean_fine_bit_transfer4000_v1::mean_fine_bit_edit_transfer.control | +0.357842 [+0.036634, +0.708050] |
| size_cut_composition_cached4000_v2::fixed_size_cut_on_foldtemp_fine16.control | -0.047002 [-0.248330, +0.149315] |
| native | +2.170458 [+1.730749, +2.628629] |
| frozen_subtoken4000_v1::native | +2.170458 [+1.730749, +2.628629] |
| frozen_subtoken4000_v1::rcg | +0.768527 [+0.502919, +1.077590] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.450582 [+0.177470, +0.777880] |
| claude_official_4000_existing_masks::RCG_count_matched_delete | +0.928311 [+0.529785, +1.398129] |
| claude_official_4000_existing_masks::conservative_delete | +2.222921 [+1.464297, +3.070193] |
| claude_official_4000_existing_masks::external_mean__delete | +1.248948 [+0.687329, +1.908368] |
| claude_official_4000_existing_masks::external_mean_delete_sameK_RCG | +1.161693 [+0.629569, +1.787046] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.257662 [-0.071327, +0.605832] |
