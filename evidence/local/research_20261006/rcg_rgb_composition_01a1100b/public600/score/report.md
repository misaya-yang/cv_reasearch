# Cached-feature candidate results

| Arm | Class-summed mIoU |
|---|---:|
| native | 61.335314 |
| rcg_current.control | 63.089000 |
| mean.control | 62.931546 |
| rcg_rgb_proposal_v2 | 63.125362 |
| rcg_rgb_full_cut.control | 63.038936 |
| rcg_rgb_same_unary.control | 62.949403 |
| stored_mean.control | 62.931530 |
| rcg.control | 63.088942 |
| fine16.control | 63.464527 |
| fine64.control | 63.681184 |

Primary: rcg_rgb_proposal_v2
Exposure: reused public600 exploratory composition; no independent confirmation

| Primary versus | Gain, pp | Paired 95% CI |
|---|---:|---|
| native | +1.790048 | [1.1198326141896158, 2.4395209918483123] |
| rcg_current.control | +0.036362 | [-0.015594884917782003, 0.1351165443525339] |
| mean.control | +0.193816 | [-0.24650523514545597, 0.5672170110288994] |
| rcg_rgb_full_cut.control | +0.086426 | [0.055231558604914535, 0.15711374335913728] |
| rcg_rgb_same_unary.control | +0.175960 | [0.1315940056837226, 0.30844322499350174] |
| stored_mean.control | +0.193832 | [-0.24650403737540233, 0.5672219530769804] |
| rcg.control | +0.036421 | [-0.015541314790095129, 0.1351858681806737] |
| fine16.control | -0.339165 | [-0.5048281543662576, -0.2834291830507992] |
| fine64.control | -0.555822 | [-1.279984444030021, 0.7362321530179038] |
