# Grouping-based adder and deleter on RCG: 600 episodes, RCG 64.24, FoRIS 61.63

| arm | mIoU | vs native | vs rcg |
|---|---:|---|---|
| native | 61.63 |  | -2.61 [-3.23, -1.39] |
| rcg | 64.24 | +2.61 [+1.39, +3.23] |  |
| add.nested | 64.16 | +2.53 [+1.22, +3.18] | -0.08 [-0.46, +0.20] |
| delete.nested | 64.20 | +2.57 [+1.25, +3.29] | -0.04 [-0.40, +0.25] |
| both.nested | 64.13 | +2.50 [+1.09, +3.23] | -0.11 [-0.63, +0.27] |
| both.same_count.control | 64.35 | +2.72 [+1.38, +3.40] | +0.11 [-0.31, +0.45] |
| best_in_sample | 64.48 | +2.85 [+1.45, +3.56] | +0.24 [-0.35, +0.71] |

Chosen: {"add.nested": {"0": ["none", "add[tau0.5,cover>=0.6,field>0.4]"], "1": ["none", "add[tau0.5,cover>=0.6,field>0.4]"], "2": ["none", "add[tau0.5,cover>=0.6,field>0.4]"], "3": ["none", "none"]}, "delete.nested": {"0": ["del[tau0.7,cover<=0.2,field<0.6]", "none"], "1": ["del[tau0.5,cover<=0.4,field<0.6]", "none"], "2": ["del[tau0.6,cover<=0.3,field<0.6]", "none"], "3": ["del[tau0.5,cover<=0.4,field<0.6]", "none"]}, "both.nested": {"0": ["del[tau0.7,cover<=0.2,field<0.6]", "add[tau0.5,cover>=0.6,field>0.4]"], "1": ["del[tau0.5,cover<=0.4,field<0.6]", "add[tau0.5,cover>=0.6,field>0.4]"], "2": ["del[tau0.6,cover<=0.3,field<0.6]", "add[tau0.5,cover>=0.6,field>0.4]"], "3": ["del[tau0.5,cover<=0.4,field<0.6]", "none"]}, "both.same_count.control": {"0": ["del[tau0.6,cover<=0.2,field<2]", "add[tau0.5,cover>=0.6,field>0.4]"], "1": ["del[tau0.6,cover<=0.4,field<0.6]", "add[tau0.5,cover>=0.6,field>0.4]"], "2": ["del[tau0.6,cover<=0.4,field<0.6]", "add[tau0.5,cover>=0.6,field>0.4]"], "3": ["del[tau0.6,cover<=0.2,field<2]", "add[tau0.5,cover>=0.7,field>0.4]"]}}

Best in sample: ['del[tau0.6,cover<=0.3,field<0.6]', 'add[tau0.5,cover>=0.6,field>0.4]', 64.48067382960491]

## Additions alone (no selection)

| variant | mIoU | same-count control | added true | added false | purity |
|---|---:|---:|---:|---:|---:|
| add[tau0.5,cover>=0.6,field>0.4] | 64.29 | 64.40 | 1,232,065 | 1,313,310 | 0.48 |
| none | 64.24 | 64.24 | 0 | 0 | 0.00 |
| add[tau0.5,cover>=0.7,field>0.4] | 64.15 | 64.28 | 769,823 | 990,655 | 0.44 |
| add[tau0.5,cover>=0.8,field>0.4] | 64.12 | 64.28 | 505,016 | 751,101 | 0.40 |
| add[tau0.7,cover>=0.8,field>0.4] | 64.12 | 64.22 | 200,844 | 367,544 | 0.35 |
| add[tau0.7,cover>=0.7,field>0.4] | 64.10 | 64.24 | 351,502 | 638,922 | 0.35 |
| add[tau0.7,cover>=0.8,field>0.3] | 64.05 | 64.16 | 241,681 | 471,030 | 0.34 |
| add[tau0.7,cover>=0.8,field>0.2] | 64.05 | 64.16 | 242,345 | 474,087 | 0.34 |
| add[tau0.7,cover>=0.8,field>-1] | 64.05 | 64.16 | 242,384 | 474,519 | 0.34 |
| add[tau0.6,cover>=0.8,field>0.4] | 64.04 | 64.20 | 295,172 | 655,280 | 0.31 |
| add[tau0.5,cover>=0.5,field>0.4] | 64.01 | 64.12 | 1,468,639 | 2,033,952 | 0.42 |
| add[tau0.7,cover>=0.6,field>0.4] | 64.00 | 64.15 | 487,027 | 982,907 | 0.33 |
| add[tau0.5,cover>=0.8,field>0.2] | 63.99 | 64.19 | 714,117 | 1,188,650 | 0.38 |
| add[tau0.5,cover>=0.8,field>-1] | 63.99 | 64.19 | 714,127 | 1,188,947 | 0.38 |
| add[tau0.5,cover>=0.8,field>0.3] | 63.98 | 64.22 | 668,939 | 1,128,598 | 0.37 |
| add[tau0.7,cover>=0.7,field>0.3] | 63.98 | 64.16 | 397,779 | 800,080 | 0.33 |
| add[tau0.7,cover>=0.7,field>0.2] | 63.98 | 64.15 | 401,382 | 817,868 | 0.33 |
| add[tau0.7,cover>=0.7,field>-1] | 63.98 | 64.15 | 401,421 | 818,300 | 0.33 |
| add[tau0.6,cover>=0.7,field>0.4] | 63.95 | 64.14 | 440,083 | 950,444 | 0.32 |
| add[tau0.6,cover>=0.6,field>0.4] | 63.94 | 64.08 | 698,360 | 1,209,567 | 0.37 |
| add[tau0.6,cover>=0.8,field>0.3] | 63.90 | 64.11 | 379,406 | 874,047 | 0.30 |
| add[tau0.7,cover>=0.5,field>0.4] | 63.90 | 64.03 | 654,053 | 1,379,047 | 0.32 |
| add[tau0.6,cover>=0.8,field>0.2] | 63.89 | 64.10 | 381,396 | 891,099 | 0.30 |
| add[tau0.6,cover>=0.8,field>-1] | 63.89 | 64.10 | 381,435 | 891,531 | 0.30 |
| add[tau0.5,cover>=0.4,field>0.4] | 63.88 | 63.98 | 1,509,795 | 2,330,297 | 0.39 |

## Deletions alone (no selection)

| variant | mIoU | same-count control | removed true | removed false | share false |
|---|---:|---:|---:|---:|---:|
| del[tau0.6,cover<=0.3,field<0.6] | 64.43 | 64.45 | 343,250 | 910,133 | 0.73 |
| del[tau0.5,cover<=0.4,field<0.6] | 64.42 | 64.46 | 543,506 | 1,339,141 | 0.71 |
| del[tau0.6,cover<=0.4,field<0.6] | 64.39 | 64.49 | 507,458 | 1,123,025 | 0.69 |
| del[tau0.6,cover<=0.2,field<0.6] | 64.37 | 64.40 | 227,810 | 645,146 | 0.74 |
| del[tau0.7,cover<=0.3,field<0.6] | 64.30 | 64.37 | 360,073 | 671,841 | 0.65 |
| del[tau0.7,cover<=0.2,field<0.6] | 64.29 | 64.34 | 247,360 | 504,263 | 0.67 |
| del[tau0.7,cover<=0.2,field<0.7] | 64.29 | 64.34 | 281,367 | 543,747 | 0.66 |
| del[tau0.5,cover<=0.3,field<0.6] | 64.28 | 64.34 | 479,105 | 984,558 | 0.67 |
| del[tau0.7,cover<=0.2,field<2] | 64.28 | 64.34 | 283,298 | 544,304 | 0.66 |
| del[tau0.6,cover<=0.2,field<2] | 64.26 | 64.46 | 337,608 | 776,609 | 0.70 |
| del[tau0.7,cover<=0.3,field<2] | 64.26 | 64.34 | 466,887 | 749,658 | 0.62 |
| del[tau0.7,cover<=0.3,field<0.7] | 64.24 | 64.33 | 464,199 | 740,553 | 0.61 |
| none | 64.24 | 64.24 | 0 | 0 | 0.00 |
| del[tau0.5,cover<=0.2,field<0.6] | 64.22 | 64.26 | 324,040 | 627,367 | 0.66 |
| del[tau0.5,cover<=0.05,field<0.6] | 64.21 | 64.27 | 150,052 | 175,406 | 0.54 |
| del[tau0.5,cover<=0.1,field<0.6] | 64.20 | 64.27 | 187,110 | 274,101 | 0.59 |
| del[tau0.6,cover<=0.1,field<0.6] | 64.20 | 64.29 | 149,206 | 245,806 | 0.62 |
| del[tau0.7,cover<=0.1,field<0.7] | 64.20 | 64.27 | 132,188 | 195,226 | 0.60 |
| del[tau0.7,cover<=0.1,field<2] | 64.19 | 64.27 | 133,972 | 195,783 | 0.59 |
| del[tau0.7,cover<=0.1,field<0.6] | 64.19 | 64.26 | 120,439 | 166,801 | 0.58 |
| del[tau0.6,cover<=0.2,field<0.7] | 64.19 | 64.40 | 334,251 | 733,977 | 0.69 |
| del[tau0.7,cover<=0.05,field<0.7] | 64.19 | 64.23 | 64,712 | 70,934 | 0.52 |
| del[tau0.7,cover<=0.05,field<2] | 64.19 | 64.23 | 65,369 | 71,309 | 0.52 |
| del[tau0.7,cover<=0.05,field<0.6] | 64.18 | 64.23 | 60,346 | 60,451 | 0.50 |
| del[tau0.6,cover<=0.05,field<0.6] | 64.18 | 64.26 | 105,369 | 129,106 | 0.55 |
