## how much of the best level per episode the field itself predicts: rcg field, 4000 episodes, class mIoU from stored per-level counts
| rule | uses | mIoU | vs fixed 0.5 | folds |
|---|---|---:|---|---|
| fixed 0.5 (control) | nothing | 62.33 | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 / +0.00 |
| level per bin of the mask area at 0.5 | base-fold labels, 6 numbers | 62.62 | +0.29 [+0.04, +0.53] | +0.22 / +0.42 / +0.28 / +0.24 |
| boosted trees on histogram descriptors | base-fold labels | 61.57 | -0.76 [-1.31, -0.29] | -1.11 / -0.68 / -0.23 / -1.02 |
| boosted trees on histogram and spatial descriptors | base-fold labels | 61.66 | -0.67 [-1.19, -0.20] | -0.37 / -0.64 / -0.36 / -1.31 |
| best IoU level per episode | query truth | 71.24 | +8.91 [+7.95, +9.94] | +10.19 / +7.29 / +9.79 / +8.35 |
| class-optimal level per episode | query truth | 77.10 | +14.77 [+13.62, +15.60] | +15.39 / +13.90 / +14.84 / +14.94 |

boosted trees on all descriptors, by true object share: 0-0.02: n=1074, mean level 0.45, gain +0.20; 0.02-0.1: n=1543, mean level 0.44, gain -1.54; 0.1-0.3: n=936, mean level 0.41, gain -1.78; 0.3-1.01: n=447, mean level 0.39, gain +1.10
