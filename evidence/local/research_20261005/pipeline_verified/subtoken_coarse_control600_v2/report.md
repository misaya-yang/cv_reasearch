# Matched subtoken coarse-feature control: reused DEV600

| arm | class mIoU at 1024 |
|---|---:|
| native | 61.628013 |
| rcg | 64.237696 |
| fine.selected | 64.625380 |
| coarse_guide.control | 64.273121 |

Fine minus coarse-guide control: +0.352259 [+0.325705, +0.544086] percentage points.
All600 source native, RCG and selected fine I/U exactly match the original counts.
Recorded per-fold parameters are unchanged; no encoder, image, refit or parameter grid.
Inference/compute/CPU-score seconds: 19.92 / 4.16 / 10.87.
