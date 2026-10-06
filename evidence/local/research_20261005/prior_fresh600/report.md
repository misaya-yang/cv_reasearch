## reliability role: delete connected components of the RCG mask, 600 episodes, 1811 components, token-level class mIoU
break-even precision 0.39; components below it: 747 holding 61% of the false area
| rule | mIoU | vs RCG at 0.5 | folds | fitted |
|---|---:|---|---|---|
| RCG at 0.5 (control) | 63.03 | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 / +0.00 |  |
| ceiling: delete below break-even (truth) | 71.85 | +8.82 [+6.08, +9.56] | +9.09 / +8.71 / +6.27 / +11.21 |  |
| control: keep the largest component only | 58.92 | -4.11 [-5.92, -3.08] | -5.33 / -5.72 / -1.79 / -3.62 |  |
| delete if mean_field below threshold | 62.87 | -0.16 [-1.22, +0.89] | -2.61 / +0.31 / +0.29 / +1.35 | [0.652, 0.563, 0.561, 0.568] |
| delete if max_field below threshold | 63.39 | +0.36 [-1.13, +1.23] | -0.44 / -0.38 / +0.87 / +1.39 | [0.754, 0.732, 0.763, 0.741] |
| delete if max_field_rel below threshold | 63.74 | +0.71 [-0.38, +1.56] | -0.72 / -0.18 / +2.38 / +1.37 | [0.895, 0.769, 0.773, 0.778] |
| delete if share_of_mask below threshold | 63.08 | +0.05 [-0.09, +0.13] | +0.03 / +0.19 / -0.29 / +0.26 | [0.042, 0.041, 0.07, 0.045] |
| delete if log_size below threshold | 62.95 | -0.08 [-0.41, +0.06] | +0.06 / +0.06 / -0.46 / +0.03 | [1.386, 1.386, 2.565, 1.099] |
| delete if mean_foris below threshold | 63.30 | +0.27 [-0.64, +0.81] | +0.11 / +0.43 / +0.01 / +0.53 | [0.612, 0.578, 0.572, 0.586] |
| delete if mean_cos below threshold | 62.98 | -0.05 [-1.12, +0.78] | +0.70 / +0.00 / -1.63 / +0.73 | [0.315, 0.103, 0.295, 0.319] |
| delete if max_cos below threshold | 61.57 | -1.46 [-2.62, -0.37] | -4.44 / +0.00 / -1.43 / +0.02 | [0.455, 0.268, 0.388, 0.277] |
| delete if mean_obj_border below threshold | 63.03 | +0.00 [+0.00, +0.01] | +0.01 / +0.00 / +0.00 / +0.00 | [-0.926, -0.961, -0.961, -0.961] |
| delete if mean_obj_cls below threshold | 63.03 | -0.00 [-0.00, +0.00] | -0.00 / +0.00 / +0.00 / +0.00 | [0.006, -0.006, -0.006, -0.006] |
| delete if mean_obj_cut below threshold | 63.03 | -0.00 [-0.00, +0.00] | +0.00 / +0.00 / +0.00 / -0.00 | [-0.959, -0.959, -0.959, -0.946] |
| logistic on field statistics and size | 63.86 | +0.83 [+0.15, +1.27] | +0.85 / +0.35 / +0.28 / +1.84 |  |
| logistic on all statistics | 62.74 | -0.29 [-1.76, +0.89] | -2.68 / +0.30 / -0.53 / +1.74 |  |
| logistic on field statistics, size and obj_border | 63.76 | +0.73 [+0.02, +1.22] | +0.79 / +0.32 / +0.02 / +1.81 |  |
| logistic on field statistics, size and obj_cls | 62.88 | -0.15 [-1.50, +0.84] | -1.62 / +0.39 / -0.25 / +0.90 |  |
| logistic on field statistics, size and obj_cut | 63.66 | +0.63 [-0.20, +1.17] | +0.97 / +0.54 / -0.35 / +1.37 |  |
| logistic on field statistics, size and all objecthood | 63.03 | -0.00 [-1.66, +1.04] | +0.01 / -0.27 / -0.52 / +0.76 |  |

## AUC of true object tokens against the tokens RCG wrongly keeps under its best cut
| field | RCG best-cut IoU < 0.2 | 0.2-0.5 | 0.5-0.8 | >= 0.8 |
|---|---:|---:|---:|---:|
| cos | 0.48 (n=33) | 0.59 (n=41) | 0.58 (n=122) | 0.68 (n=147) |
| rcg | 0.14 (n=33) | 0.30 (n=41) | 0.65 (n=122) | 0.80 (n=147) |
| obj_border | 0.58 (n=33) | 0.57 (n=41) | 0.63 (n=122) | 0.63 (n=147) |
| obj_cls | 0.54 (n=33) | 0.50 (n=41) | 0.54 (n=122) | 0.60 (n=147) |
| obj_cut | 0.57 (n=33) | 0.52 (n=41) | 0.51 (n=122) | 0.57 (n=147) |
