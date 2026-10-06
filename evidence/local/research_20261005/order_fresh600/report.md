## ordering role: token-level class mIoU, 600 episodes
| field | cut at half of its range | best cut per episode | ordering loss |
|---|---:|---:|---:|
| nn | 21.68 | 55.52 | 44.48 |
| knn_fg | 47.75 | 52.16 | 47.84 |
| s2 | 50.92 | 67.97 | 32.03 |
| s3 | 56.22 | 69.48 | 30.52 |
| score | 59.53 | 69.88 | 30.12 |
| rcg | 55.34 | 72.00 | 28.00 |

## what the best cut of `score` still gets wrong (best-cut mIoU 69.88); points recovered if that part alone were right
missed object area 22287 tokens, false area 86902 tokens; in mixed (boundary) tokens: missed 42%, false 15%
| part | share of that error | points |
|---|---:|---:|
| all missed | 100% | +7.50 |
| all false | 100% | +19.86 |
| missed, boundary tokens | 42% | +3.83 |
| false, boundary tokens | 15% | +3.33 |
| missed: whole object region undetected | 3% | +0.29 |
| missed: part of a detected region | 54% | +3.20 |
| false: separate region with no object | 22% | +3.48 |
| false: attached to a detected object | 63% | +9.04 |
| missed: nearest reference token is object | 14% | +0.98 |
| missed: nearest reference token is background | 42% | +2.51 |
| false: nearest reference token is object | 40% | +6.00 |
| false: nearest reference token is background | 45% | +5.93 |
| missed: most feature neighbours detected | 9% | +0.60 |
| missed: most feature neighbours undetected | 47% | +2.89 |
| false: most feature neighbours detected | 79% | +12.64 |
| false: most feature neighbours undetected | 6% | +1.01 |
| missed: most feature neighbours are object (GT) | 49% | +2.95 |
| missed: most feature neighbours are background (GT) | 7% | +0.54 |
| false: most feature neighbours are background (GT) | 82% | +14.03 |
| false: most feature neighbours are object (GT) | 3% | +0.47 |
episodes with best-cut IoU < 0.2: 34 (40% of all error area); < 0.5: 88 (54%)

## what the best cut of `rcg` still gets wrong (best-cut mIoU 72.00); points recovered if that part alone were right
missed object area 22325 tokens, false area 71948 tokens; in mixed (boundary) tokens: missed 44%, false 17%
| part | share of that error | points |
|---|---:|---:|
| all missed | 100% | +7.42 |
| all false | 100% | +17.92 |
| missed, boundary tokens | 44% | +4.03 |
| false, boundary tokens | 17% | +3.30 |
| missed: whole object region undetected | 5% | +0.51 |
| missed: part of a detected region | 49% | +2.70 |
| false: separate region with no object | 20% | +3.20 |
| false: attached to a detected object | 63% | +8.18 |
| missed: nearest reference token is object | 14% | +0.95 |
| missed: nearest reference token is background | 40% | +2.25 |
| false: nearest reference token is object | 44% | +5.62 |
| false: nearest reference token is background | 38% | +5.08 |
| missed: most feature neighbours detected | 4% | +0.27 |
| missed: most feature neighbours undetected | 51% | +2.94 |
| false: most feature neighbours detected | 79% | +12.09 |
| false: most feature neighbours undetected | 3% | +0.43 |
| missed: most feature neighbours are object (GT) | 46% | +2.65 |
| missed: most feature neighbours are background (GT) | 8% | +0.56 |
| false: most feature neighbours are background (GT) | 80% | +12.39 |
| false: most feature neighbours are object (GT) | 3% | +0.44 |
episodes with best-cut IoU < 0.2: 35 (37% of all error area); < 0.5: 83 (50%)
