## candidates from the query alone against level sets of a reference score: token-level class mIoU, 600 episodes
| candidate set | chosen how | mIoU |
|---|---|---:|
| level sets of the FoRIS score | fixed 0.5 (FoRIS before its CRF) | 59.54 |
| level sets of the RCG field | fixed 0.5 | 63.03 |
| level sets of the FoRIS score | best level per episode (truth) | 69.88 |
| level sets of the RCG field | best level per episode (truth) | 72.00 |
| nodes of the ward hierarchy | best single node (truth) | 73.63 |
| nodes of the ward hierarchy | best union of up to 2 disjoint nodes (truth, greedy) | 79.54 |
| nodes of the ward hierarchy | best union of up to 3 disjoint nodes (truth, greedy) | 80.95 |
| nodes of the ward hierarchy | node that best splits the reference cosine (no truth) | 37.68 |
| nodes of the ward hierarchy | node that best splits the RCG field (no truth) | 43.55 |
| nodes of the average hierarchy | best single node (truth) | 69.10 |
| nodes of the average hierarchy | best union of up to 2 disjoint nodes (truth, greedy) | 76.04 |
| nodes of the average hierarchy | best union of up to 3 disjoint nodes (truth, greedy) | 78.45 |
| nodes of the average hierarchy | node that best splits the reference cosine (no truth) | 38.14 |
| nodes of the average hierarchy | node that best splits the RCG field (no truth) | 45.05 |

## truth-free: snap the RCG field to the query's own regions (expected IoU with the field), against the level set at 0.5
| candidates | mIoU | vs RCG at 0.5 | folds |
|---|---:|---|---|
| level set of the RCG field at 0.5 (control) | 63.03 | +0.00 [+0.00, +0.00] | +0.00 / +0.00 / +0.00 / +0.00 |
| ward hierarchy, up to 1 node | 58.22 | -4.81 [-6.22, -3.58] | -5.87 / -4.83 / -5.17 / -3.37 |
| ward hierarchy, up to 2 nodes | 61.10 | -1.93 [-2.69, -0.99] | -3.13 / -1.51 / -2.77 / -0.30 |
| ward hierarchy, up to 3 nodes | 61.31 | -1.72 [-2.42, -0.84] | -1.97 / -1.36 / -2.83 / -0.74 |
| ward hierarchy, up to 5 nodes | 61.38 | -1.65 [-2.31, -0.76] | -1.96 / -1.05 / -2.73 / -0.85 |
| ward hierarchy, up to 8 nodes | 61.36 | -1.67 [-2.31, -0.78] | -1.95 / -1.12 / -2.68 / -0.95 |
| average hierarchy, up to 1 node | 56.08 | -6.95 [-9.05, -5.64] | -8.96 / -7.91 / -6.83 / -4.10 |
| average hierarchy, up to 2 nodes | 59.22 | -3.81 [-4.83, -2.62] | -5.81 / -3.48 / -5.17 / -0.78 |
| average hierarchy, up to 3 nodes | 60.18 | -2.85 [-3.77, -1.78] | -3.37 / -2.75 / -4.67 / -0.62 |
| average hierarchy, up to 5 nodes | 60.43 | -2.60 [-3.44, -1.53] | -2.83 / -2.37 / -4.44 / -0.76 |
| average hierarchy, up to 8 nodes | 60.57 | -2.46 [-3.29, -1.40] | -2.72 / -2.23 / -4.09 / -0.82 |
