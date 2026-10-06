# Independent selected-recipe check: public4000

Main recipe3 class-summed mIoU: **63.062305**. Whole4000 complete masks were independently reconstructed and matched the sealed main masks bit-for-bit.

`C=(conservative_delete & fine16) | (~conservative_delete & fine64)`; producer origin8/add32/delete16. This is the already selected global recipe; no new search or selector was run.

| Original producer | Gain [paired95% CI], pp |
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

All11 original single-producer I/U, main membership and class I/U, finalized main count exports, scores, all CIs/up/down/tie and four fold results match exactly; largest numerical error0. CPU check22.77s, no GPU/encoder/new method or worker source mutation.

Cost3 is an outer complete-mask producer-count proxy. It does not measure atomic upstream producer cost or elapsed inference time.

GT selected the global DEV recipe. These paired intervals condition on that selected recipe and are not selection-adjusted; exposed public benchmark data are not fresh confirmation. No full4000 raw-origin result is inferred from the native control.
