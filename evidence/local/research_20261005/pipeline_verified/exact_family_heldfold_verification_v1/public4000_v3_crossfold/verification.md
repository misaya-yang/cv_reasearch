# Independent strict12 held-fold main check

Full4000 macro mIoU **62.964699**. Every row used its own fold-specific fixed recipe and matched the producer sealed mask bit-for-bit.

`recipe0` is a storage key; it is not one global recipe applied to all folds.

| Held fold | Origin / addition subset / deletion subset |
|---|---|
| 0 | 10 / 32 / 16 |
| 1 | 10 / 32 / 1024 |
| 2 | 8 / 32 / 16 |
| 3 | 4 / 16 / 1024 |

All4000 main membership I/U,80 class I/U,12 original baseline I/U, saved scores/paired2000 RandomState0 photo CIs/sign counts/fold scores reproduce exactly; maximum numerical error0. Actual train3000/held1000 class lists are disjoint and match the rows for each fold. CPU24.301s; no repeated search, encoder or GPU.

| Comparator | Gain [paired95% CI], pp |
|---|---:|
| frozen_subtoken4000_v1::native | +2.032958 [+1.657151, +2.387729] |
| frozen_subtoken4000_v1::mean.control | +0.451727 [+0.233044, +0.659919] |
| frozen_subtoken4000_v1::fine.rcg16.control | +0.255599 [+0.094337, +0.399785] |
| frozen_subtoken4000_v1::fine.rcg64 | +0.313081 [+0.117710, +0.538295] |
| mean_fine_residual_transfer4000_v1::mean_fine_residual_transfer_v1 | +0.120162 [-0.103035, +0.319822] |

Selection uses the other three folds, while original source-producer exposure remains unchanged. These conditional paired intervals do not rerun recipe selection; reused development data are not fresh confirmation. Outer producer-count proxy3 is not atomic upstream cost or measured seconds.
