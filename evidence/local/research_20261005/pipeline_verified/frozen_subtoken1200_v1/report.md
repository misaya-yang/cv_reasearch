# Frozen subtoken readout: complete reused DEV1200

| arm | class mIoU at1024 |
|---|---:|
| native | 61.612802 |
| rcg | 63.598020 |
| mean.control | 63.581592 |
| rcg64.control | 63.690334 |
| fine.rcg16.control | 64.015315 |
| fine.rcg64 | 64.051866 |

Fine lambda64 versus mandatory controls (paired95% photo-connected intervals):

- native: +2.439064 [+1.581170, +3.237973] percentage points.
- rcg: +0.453846 [-0.059653, +0.905474] percentage points.
- mean.control: +0.470274 [-0.104917, +1.001042] percentage points.
- rcg64.control: +0.361532 [+0.312316, +0.461490] percentage points.
- fine.rcg16.control: +0.036551 [-0.520387, +0.457837] percentage points.

All1200 prior native/RCG/MEAN I/U match exactly. Frozen600 readout parameters are unchanged.
Raw DINO origin masks are absent: raw-origin edit accounting is unavailable. RCG edits are supplemental.
This is reused development evidence, not fresh confirmation or a SOTA result.
