# Boundary feature diagnostic: 80 reused episodes, GT scoring only

| arm | boundary AUROC (95% CI) | parent pair ranking (95% CI) |
|---|---:|---:|
| fine_nn | 0.8762 [0.8518, 0.8993], n=80 | 0.8588 [0.8201, 0.8934], n=80 |
| bilinear_feature_nn.control | 0.8568 [0.8319, 0.8810], n=80 | 0.8301 [0.7954, 0.8635], n=80 |
| bilinear_margin.control | 0.8752 [0.8514, 0.8979], n=80 | 0.9179 [0.8919, 0.9416], n=80 |
| rgb_nn.control | 0.5564 [0.5099, 0.6011], n=80 | 0.5584 [0.5096, 0.6053], n=80 |

Paired differences (fine minus control):

| control | boundary AUROC difference (95% CI) | parent ranking difference (95% CI) |
|---|---:|---:|
| bilinear_feature_nn.control | 0.0194 [0.0136, 0.0259], n=80 | 0.0287 [0.0008, 0.0576], n=80 |
| bilinear_margin.control | 0.0010 [-0.0041, 0.0056], n=80 | -0.0591 [-0.0859, -0.0335], n=80 |
| rgb_nn.control | 0.3198 [0.2720, 0.3676], n=80 | 0.3004 [0.2481, 0.3527], n=80 |

Eligible parents: 1717/327680 total; 21711 with pure boundary cells. FG/BG pairs: 3078.
True-side RCG value available: 51215/53989 pure boundary cells; unavailable: 2774.

GT defines diagnostic cells and labels only after all 80 feature grids and fields are sealed.
This cohort is reused development evidence; no complete-mask or SOTA result is asserted.
