# Fixed arms against complete FoRIS at the original query resolution (class mIoU, gain [95% interval])

## Cumulative

| cohort | n | FoRIS | rcg | c.control | astra.control | astra_sameK.control | FoRIS by fold |
|---|---:|---:|---|---|---|---|---|
| first 1 batches | 600 | 61.42 | 64.03 +2.61 [+1.38, +3.22] | 63.33 +1.91 [+0.29, +2.72] | 63.25 +1.83 [+0.09, +2.89] | 63.14 +1.72 [-0.03, +2.72] | 58.3 / 64.8 / 64.3 / 58.2 |

## Each batch alone

| cohort | n | FoRIS | rcg | c.control | astra.control | astra_sameK.control | FoRIS by fold |
|---|---:|---:|---|---|---|---|---|
| recheck_fresh600_v1 | 600 | 61.42 | 64.03 +2.61 [+1.38, +3.22] | 63.33 +1.91 [+0.29, +2.72] | 63.25 +1.83 [+0.09, +2.89] | 63.14 +1.72 [-0.03, +2.72] | 58.3 / 64.8 / 64.3 / 58.2 |

