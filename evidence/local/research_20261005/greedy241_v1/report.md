# Operators chosen greedily: 241 episodes, 25 masks, add or delete

## Start: model.raw_nn (42.90)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | delete astra | 54.54 | 0 | 0 | 13,175,345 | 1,675,700 | 0.887 | 0.692 |
| 2 | add rcg&hypotheses | 62.69 | 3,873,562 | 1,904,737 | 0 | 0 | 0.670 | 0.363 |
| 3 | delete astra | 62.71 | 0 | 0 | 135,264 | 101,511 | 0.571 | 0.607 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 54.07 | -5.01 [-8.66, -3.01] | -8.59 [-11.46, -6.42] | +11.16 [+8.01, +13.75] |
| 2 | 61.99 | +2.92 [+0.57, +4.40] | -0.66 [-1.37, -0.01] | +19.09 [+15.78, +22.48] |
| 3 | 61.99 | +2.91 [+0.57, +4.38] | -0.66 [-1.38, -0.02] | +19.09 [+15.78, +22.47] |
| 4 | 61.98 | +2.91 [+0.56, +4.37] | -0.67 [-1.39, -0.03] | +19.08 [+15.78, +22.46] |
| 5 | 61.99 | +2.91 [+0.57, +4.38] | -0.67 [-1.38, -0.02] | +19.09 [+15.78, +22.47] |
| 6 | 61.99 | +2.91 [+0.57, +4.38] | -0.67 [-1.38, -0.02] | +19.09 [+15.78, +22.47] |
| 7 | 61.99 | +2.91 [+0.57, +4.38] | -0.67 [-1.38, -0.02] | +19.09 [+15.78, +22.47] |
| 8 | 61.99 | +2.91 [+0.57, +4.38] | -0.67 [-1.38, -0.02] | +19.09 [+15.78, +22.47] |

Fold paths: {"0": ["delete astra", "add rcg&hypotheses", "delete astra"], "1": ["delete astra", "add astra", "delete foris_score>0.3"], "2": ["delete astra", "add astra", "delete foris_score>0.3", "add rcg&hypotheses", "delete astra"], "3": ["delete rcg&hypotheses", "add rcg&hypotheses"]}

## Start: native (59.07)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | delete rcg&hypotheses | 62.26 | 0 | 0 | 3,978,046 | 1,613,867 | 0.711 | 0.622 |
| 2 | add astra | 62.66 | 1,420,358 | 1,336,215 | 0 | 0 | 0.515 | 0.387 |
| 3 | delete foris_score>0.3 | 62.66 | 0 | 0 | 11,527 | 5,873 | 0.662 | 0.607 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 61.62 | +2.55 [+0.21, +4.18] | -1.03 [-1.86, +0.02] | +2.55 [+0.21, +4.18] |
| 2 | 62.02 | +2.94 [+0.64, +4.46] | -0.63 [-1.29, +0.06] | +2.94 [+0.64, +4.46] |
| 3 | 61.99 | +2.92 [+0.62, +4.46] | -0.66 [-1.33, +0.01] | +2.92 [+0.62, +4.46] |
| 4 | 61.98 | +2.91 [+0.61, +4.43] | -0.67 [-1.33, -0.01] | +2.91 [+0.61, +4.43] |
| 5 | 61.99 | +2.92 [+0.62, +4.45] | -0.66 [-1.33, +0.01] | +2.92 [+0.62, +4.45] |
| 6 | 61.99 | +2.92 [+0.62, +4.45] | -0.66 [-1.33, +0.01] | +2.92 [+0.62, +4.45] |
| 7 | 61.99 | +2.92 [+0.62, +4.45] | -0.66 [-1.33, +0.01] | +2.92 [+0.62, +4.45] |
| 8 | 61.99 | +2.92 [+0.62, +4.45] | -0.66 [-1.33, +0.01] | +2.92 [+0.62, +4.45] |

Fold paths: {"0": ["delete astra", "add astra"], "1": ["delete rcg&hypotheses", "add astra", "delete astra", "delete foris_score>0.3"], "2": ["delete astra", "add astra", "delete foris_score>0.3", "add rcg&hypotheses", "delete astra"], "3": ["delete rcg&hypotheses", "add rcg&hypotheses"]}

## Start: insid3.final (54.47)

In sample (optimistic):

| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | add rcg&hypotheses | 58.60 | 2,473,023 | 1,329,026 | 0 | 0 | 0.650 | 0.367 |
| 2 | delete astra | 62.76 | 0 | 0 | 3,252,881 | 1,466,630 | 0.689 | 0.616 |

Path chosen on three folds, read on the fourth, after k steps:

| k | mIoU | vs native | vs astra candidate | vs start |
|---:|---:|---|---|---|
| 1 | 58.12 | -0.95 [-3.21, +1.19] | -4.53 [-6.36, -1.84] | +3.65 [+1.69, +4.86] |
| 2 | 61.99 | +2.92 [+0.31, +4.56] | -0.66 [-1.62, +0.29] | +7.52 [+3.88, +9.37] |
| 3 | 61.99 | +2.92 [+0.31, +4.56] | -0.66 [-1.63, +0.28] | +7.52 [+3.88, +9.37] |
| 4 | 61.98 | +2.91 [+0.30, +4.55] | -0.67 [-1.65, +0.27] | +7.51 [+3.86, +9.38] |
| 5 | 61.99 | +2.91 [+0.31, +4.56] | -0.66 [-1.63, +0.28] | +7.52 [+3.88, +9.37] |
| 6 | 61.99 | +2.91 [+0.31, +4.56] | -0.66 [-1.63, +0.28] | +7.52 [+3.88, +9.37] |
| 7 | 61.99 | +2.91 [+0.31, +4.56] | -0.66 [-1.63, +0.28] | +7.52 [+3.88, +9.37] |
| 8 | 61.99 | +2.91 [+0.31, +4.56] | -0.66 [-1.63, +0.28] | +7.52 [+3.88, +9.37] |

Fold paths: {"0": ["add rcg&hypotheses", "delete astra"], "1": ["add rcg&hypotheses", "delete astra", "delete foris_score>0.3"], "2": ["add astra", "delete astra", "delete foris_score>0.3", "add rcg&hypotheses", "delete astra"], "3": ["add rcg&hypotheses", "delete rcg&hypotheses"]}

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
