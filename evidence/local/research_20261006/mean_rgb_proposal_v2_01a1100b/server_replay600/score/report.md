# Cached-feature candidate results

| Arm | Class-summed mIoU |
|---|---:|
| native | 61.335314 |
| mean_rgb_proposal_v2 | 63.007428 |
| rgb_full_cut_v1.control | 62.903951 |
| rgb_same128_unary.control | 62.769442 |
| mean.control | 62.931546 |
| stored_mean.control | 62.931530 |
| rcg.control | 63.088942 |
| fine16.control | 63.464527 |
| fine64.control | 63.681184 |

Primary: mean_rgb_proposal_v2
Exposure: posthoc exploratory readout revision on reused public600; source quality was already exposed; not independent confirmation

| Primary versus | Gain, pp | Paired 95% CI |
|---|---:|---|
| native | +1.672114 | [1.0767438196267813, 2.287241606866292] |
| rgb_full_cut_v1.control | +0.103477 | [0.08092409938909544, 0.17796637567787402] |
| rgb_same128_unary.control | +0.237987 | [0.17312650343471656, 0.35221819877674954] |
| mean.control | +0.075882 | [-0.006267983744460537, 0.13596724425168424] |
| stored_mean.control | +0.075898 | [-0.006238472949304885, 0.13601030244983575] |
| rcg.control | -0.081513 | [-0.4281543093626047, 0.35545497504877693] |
| fine16.control | -0.457099 | [-0.8954276774673016, -0.07540590248281696] |
| fine64.control | -0.673756 | [-1.4797069581939577, 0.7515826179262913] |
