# E2 report-half result

E2 selection was frozen from the 6,752 S rows before reading T. The selected arm is **MFU**.
Selection criterion: maximize the minimum of ten domain deltas vs complete FoRIS; tie-break by the ten-domain mean. T6 S-self was ineligible (its condition failed).
T was scored once for the selected arm and the prespecified UFraw equal-area comparison. Intervals use 500 paired query-photo bootstrap replicates; differences are class mIoU percentage points.

## Selected arm vs complete FoRIS

|Domain|Selected|FoRIS|Difference [95% CI]|
|---|---:|---:|---:|
|coco|64.84|63.71|+1.13 [-0.06,+2.14]|
|lvis|46.99|43.79|+3.20 [+2.59,+3.86]|
|pascal_part|55.38|53.25|+2.13 [+1.46,+2.99]|
|paco_part|46.05|42.32|+3.73 [+2.89,+4.70]|
|suim|61.99|60.96|+1.02 [-0.33,+2.41]|
|isaid|51.96|50.56|+1.41 [+0.02,+2.62]|
|isic|64.56|63.36|+1.20 [+0.32,+2.15]|
|lung|89.82|88.08|+1.75 [+0.69,+3.32]|
|fundus|34.21|19.20|+15.01 [+13.53,+16.40]|
|deepglobe_road|29.90|28.93|+0.97 [-4.61,+4.64]|

Ten-domain equal-weight mean: +3.16 pp; nonnegative domains: 10/10.

## Area bands vs complete FoRIS

|GT area|Mean episode-IoU difference (pp)|
|---|---:|
|<2%|+4.10|
|2-20%|+2.45|
|>20%|+1.28|

## Equal-area ranking: UFraw vs FoRIS-noCRF

|Domain|UFraw − FoRIS-noCRF [95% CI]|
|---|---:|
|coco|+0.28 [-0.06,+0.69]|
|lvis|+0.60 [+0.31,+1.09]|
|pascal_part|+0.66 [+0.37,+1.04]|
|paco_part|+0.24 [+0.01,+0.73]|
|suim|+0.35 [-0.32,+1.02]|
|isaid|-0.02 [-0.55,+0.30]|
|isic|+0.36 [-0.08,+0.70]|
|lung|+1.41 [+1.03,+1.84]|
|fundus|+14.67 [+13.22,+16.07]|
|deepglobe_road|+1.23 [-1.00,+3.43]|

T was not used to change the selected arm, formula, or comparison set.
