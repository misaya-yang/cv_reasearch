## ordering score: token-level class mIoU under the best cut of each episode, 600 episodes
| field | all | object share 0-0.02 (n=140) | object share 0.02-0.1 (n=260) | object share 0.1-0.3 (n=143) | object share 0.3-1.01 (n=57) | episodes with IoU < 0.2 |
|---|---:|---:|---:|---:|---:|---:|
| mf_ref[0.1] | 19.98 | 16.18 | 10.87 | 17.88 | 48.14 | 446 |
| mf_ref[1] | 28.34 | 36.99 | 24.14 | 22.36 | 48.79 | 191 |
| mf_ref[10] | 47.84 | 52.63 | 52.18 | 43.55 | 55.35 | 51 |
| mf_oracle[0.1] | 53.44 | 77.20 | 66.15 | 36.13 | 49.52 | 1 |
| cos | 66.09 | 49.94 | 62.25 | 74.82 | 83.94 | 43 |
| s2 | 67.97 | 52.84 | 62.87 | 77.08 | 84.61 | 37 |
| lin_pseudo:score | 69.48 | 51.62 | 65.87 | 80.85 | 87.34 | 35 |
| score | 69.88 | 52.81 | 66.02 | 80.52 | 87.17 | 34 |
| lin_pseudo:rcg | 70.68 | 52.14 | 66.17 | 81.17 | 86.92 | 37 |
| rcg | 72.00 | 55.20 | 67.52 | 81.58 | 87.19 | 35 |
| mf_oracle[1] | 75.12 | 75.98 | 84.29 | 73.93 | 62.92 | 0 |
| cos_oracle | 83.08 | 71.55 | 79.94 | 84.24 | 90.94 | 1 |
| mf_oracle[10] | 85.45 | 73.72 | 83.54 | 87.77 | 89.46 | 0 |
| lin_oracle | 87.31 | 71.96 | 84.48 | 90.45 | 95.24 | 2 |
| token ceiling | 91.70 | 80.15 | 89.21 | 94.09 | 97.11 | 0 |

## per episode: FoRIS final score against the ceilings
FoRIS best-cut IoU 0-0.2: n=34; mean IoU cos 0.15, cos_oracle 0.64, lin_oracle 0.69, rcg 0.12; cos(ref object mean, query object mean) 0.47, cos(ref object mean, query background mean) 0.46, cos(query object mean, query background mean) 0.62, object share 0.023
FoRIS best-cut IoU 0.2-0.5: n=54; mean IoU cos 0.36, cos_oracle 0.66, lin_oracle 0.73, rcg 0.42; cos(ref object mean, query object mean) 0.57, cos(ref object mean, query background mean) 0.46, cos(query object mean, query background mean) 0.60, object share 0.042
FoRIS best-cut IoU 0.5-0.8: n=210; mean IoU cos 0.62, cos_oracle 0.75, lin_oracle 0.80, rcg 0.69; cos(ref object mean, query object mean) 0.65, cos(ref object mean, query background mean) 0.43, cos(query object mean, query background mean) 0.56, object share 0.081
FoRIS best-cut IoU 0.8-1.01: n=302; mean IoU cos 0.82, cos_oracle 0.88, lin_oracle 0.90, rcg 0.88; cos(ref object mean, query object mean) 0.71, cos(ref object mean, query background mean) 0.42, cos(query object mean, query background mean) 0.51, object share 0.158
