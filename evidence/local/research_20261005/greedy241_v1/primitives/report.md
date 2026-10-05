# Operators chosen greedily: 241 episodes, 25 masks, add or delete

## Start: model.raw_nn (42.90)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | delete rcg | 53.88 | 0 | 0 | 12,386,602 | 1,413,136 | 0.898 | 0.692 |
| 2 | add rcg | 61.02 | 4,255,369 | 2,874,408 | 0 | 0 | 0.597 | 0.360 |
| 3 | delete foris_score>0.3 | 61.05 | 0 | 0 | 22,520 | 11,196 | 0.668 | 0.612 |
| 4 | add rcg_field>0.6 | 61.05 | 565 | 91 | 0 | 0 | 0.861 | 0.388 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 53.88 | -5.19 [-8.12, -3.39] | -8.77 [-11.33, -6.35] | +10.98 [+8.20, +13.51] |
| 2 | 61.02 | +1.94 [+0.98, +2.91] | -1.63 [-2.52, +0.35] | +18.12 [+15.38, +21.59] |
| 3 | 61.03 | +1.95 [+1.01, +2.93] | -1.63 [-2.49, +0.37] | +18.12 [+15.38, +21.63] |
| 4 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +18.13 [+15.39, +21.64] |
| 5 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +18.13 [+15.39, +21.64] |
| 6 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +18.13 [+15.39, +21.64] |
| 7 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +18.13 [+15.39, +21.64] |
| 8 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +18.13 [+15.39, +21.64] |

Fold paths: {"0": ["delete rcg", "add rcg", "delete foris_score>0.4", "add rcg_field>0.6"], "1": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"], "2": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"], "3": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"]}

## Start: native (59.07)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | delete rcg | 60.74 | 0 | 0 | 2,136,159 | 531,735 | 0.801 | 0.622 |
| 2 | add rcg | 61.02 | 813,595 | 1,012,407 | 0 | 0 | 0.446 | 0.386 |
| 3 | delete foris_score>0.3 | 61.05 | 0 | 0 | 22,520 | 11,196 | 0.668 | 0.612 |
| 4 | add rcg_field>0.6 | 61.05 | 565 | 91 | 0 | 0 | 0.861 | 0.388 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 60.74 | +1.67 [+0.82, +2.64] | -1.91 [-2.83, +0.12] | +1.67 [+0.82, +2.64] |
| 2 | 61.02 | +1.94 [+0.98, +2.91] | -1.63 [-2.52, +0.35] | +1.94 [+0.98, +2.91] |
| 3 | 61.03 | +1.95 [+1.01, +2.93] | -1.63 [-2.49, +0.37] | +1.95 [+1.01, +2.93] |
| 4 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +1.96 [+1.02, +2.94] |
| 5 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +1.96 [+1.02, +2.94] |
| 6 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +1.96 [+1.02, +2.94] |
| 7 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +1.96 [+1.02, +2.94] |
| 8 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +1.96 [+1.02, +2.94] |

Fold paths: {"0": ["delete rcg", "add rcg", "delete foris_score>0.4", "add rcg_field>0.6"], "1": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"], "2": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"], "3": ["delete rcg", "add rcg", "delete foris_score>0.3", "add rcg_field>0.6"]}

## Start: insid3.final (54.47)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | add rcg | 58.22 | 3,468,387 | 2,953,586 | 0 | 0 | 0.540 | 0.367 |
| 2 | delete rcg | 61.02 | 0 | 0 | 2,786,342 | 1,305,594 | 0.681 | 0.616 |
| 3 | delete foris_score>0.3 | 61.05 | 0 | 0 | 22,520 | 11,196 | 0.668 | 0.612 |
| 4 | add rcg_field>0.6 | 61.05 | 565 | 91 | 0 | 0 | 0.861 | 0.388 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 57.05 | -2.03 [-3.62, +0.24] | -5.61 [-7.26, -2.45] | +2.58 [+0.55, +4.56] |
| 2 | 60.79 | +1.72 [+0.64, +2.85] | -1.86 [-2.83, +0.20] | +6.32 [+3.32, +8.62] |
| 3 | 60.79 | +1.71 [+0.65, +2.86] | -1.87 [-2.82, +0.23] | +6.32 [+3.33, +8.63] |
| 4 | 60.79 | +1.72 [+0.66, +2.87] | -1.86 [-2.82, +0.23] | +6.32 [+3.33, +8.65] |
| 5 | 60.79 | +1.72 [+0.66, +2.87] | -1.86 [-2.82, +0.23] | +6.32 [+3.33, +8.65] |
| 6 | 60.79 | +1.72 [+0.66, +2.87] | -1.86 [-2.82, +0.23] | +6.32 [+3.33, +8.65] |
| 7 | 60.79 | +1.72 [+0.66, +2.87] | -1.86 [-2.82, +0.23] | +6.32 [+3.33, +8.65] |
| 8 | 60.79 | +1.72 [+0.66, +2.87] | -1.86 [-2.82, +0.23] | +6.32 [+3.33, +8.65] |

Fold paths: {"0": ["add rcg", "delete rcg", "delete foris_score>0.4", "add rcg_field>0.6"], "1": ["delete rcg", "add rcg_field>0.6", "delete foris_score>0.3", "add rcg_field>0.6"], "2": ["add rcg", "delete rcg", "delete foris_score>0.3", "add rcg_field>0.6"], "3": ["add rcg", "delete rcg", "delete foris_score>0.3", "add rcg_field>0.6"]}

## Start: rcg (61.02)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | delete foris_score>0.3 | 61.05 | 0 | 0 | 22,520 | 11,196 | 0.668 | 0.612 |
| 2 | add rcg_field>0.6 | 61.05 | 565 | 91 | 0 | 0 | 0.861 | 0.388 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 61.03 | +1.95 [+1.01, +2.93] | -1.63 [-2.49, +0.37] | +0.01 [-0.05, +0.10] |
| 2 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 3 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 4 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 5 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 6 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 7 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |
| 8 | 61.04 | +1.96 [+1.02, +2.94] | -1.62 [-2.48, +0.37] | +0.02 [-0.03, +0.11] |

Fold paths: {"0": ["delete foris_score>0.4", "add rcg_field>0.6"], "1": ["delete foris_score>0.3", "add rcg_field>0.6"], "2": ["delete foris_score>0.3", "add rcg_field>0.6"], "3": ["delete foris_score>0.3", "add rcg_field>0.6"]}

## Every mask alone

| mask | mIoU |
|---|---:|
| astra | 62.65 |
| rcg&hypotheses | 62.62 |
| c | 61.84 |
| rcg | 61.02 |
| mean_control | 60.68 |
| rcg_field>0.6 | 60.44 |
| native | 59.07 |
| foris_score>0.6 | 58.80 |
| foris.pre | 58.61 |
| foris.s4_penalty | 57.35 |
| rcg_field>0.4 | 57.05 |
| foris.s3 | 56.36 |
| foris_score>0.7 | 55.42 |
| foris_score>0.4 | 54.89 |
| insid3.final | 54.47 |
| rcg_field>0.7 | 53.74 |
| foris.s3_vote | 52.20 |
| foris.s2 | 51.94 |
| foris_score>0.3 | 48.09 |
| rcg_field>0.3 | 45.00 |
| insid3.candidates | 44.29 |
| insid3.vote | 44.29 |
| model.raw_nn | 42.90 |
| insid3.seed | 42.51 |
| model.raw_mean | 39.56 |
