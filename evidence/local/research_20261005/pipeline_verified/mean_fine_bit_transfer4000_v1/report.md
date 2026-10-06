# Fixed MEAN/fine16 bit-edit transfer control: all4000 draws

`mean_fine_bit_edit_transfer.control` = `(M | (V & ~R)) & ~(R & ~V)`; use V at bit-level M/R agreement, otherwise keep M. No parameter sweep, encoder, new scalar field or primary replacement.

Complete class-summed macro mIoU: **62.744357**.

| Fixed comparison | Gain [95% CI], pp | Up / down / tie |
|---|---:|---:|
| native | +1.812616 [+1.524273, +2.098950] | 2398 / 1560 / 42 |
| mean.control | +0.231385 [+0.202166, +0.266264] | 2685 / 1222 / 93 |
| fine.rcg16.control | +0.035257 [-0.145128, +0.218718] | 1528 / 2379 / 93 |
| fine.rcg64 | +0.092740 [-0.176077, +0.421751] | 1898 / 2007 / 95 |

All4000 original six-arm I/U match exactly. All masks sealed before GT. All sampled draws and natural repeats retained. Paired2000 RandomState(0) connected-photo intervals; fold/batch contrasts and native-relative four-way edits are in JSON.
This is the fixed cheapest mask-level combination control for the separate scalar graft. It does not claim equivalence to that scalar construction or replace fine64. Cached CPU runtime excludes upstream inference; benchmark reuse is not fresh confirmation.
