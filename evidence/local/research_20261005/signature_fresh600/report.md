## ordering score: token-level class mIoU under the best cut of each episode, 600 episodes
| field | all | object share 0-0.02 (n=140) | object share 0.02-0.1 (n=260) | object share 0.1-0.3 (n=143) | object share 0.3-1.01 (n=57) | episodes with IoU < 0.2 |
|---|---:|---:|---:|---:|---:|---:|
| ms[0.1] | 20.67 | 3.86 | 15.74 | 53.90 | 82.53 | 338 |
| mf_c[1] | 27.70 | 35.36 | 22.08 | 21.54 | 49.36 | 205 |
| coral | 34.65 | 10.87 | 30.48 | 67.06 | 85.83 | 199 |
| mf_c[10] | 47.34 | 52.22 | 50.57 | 41.99 | 56.94 | 53 |
| ms_c[0.1] | 55.35 | 27.42 | 56.37 | 79.16 | 87.08 | 106 |
| ms[0.03] | 58.69 | 45.03 | 58.66 | 76.27 | 82.98 | 56 |
| coral_c | 62.41 | 33.86 | 60.72 | 79.73 | 88.94 | 66 |
| ms_c[0.03] | 65.93 | 51.73 | 64.54 | 79.19 | 85.44 | 40 |
| cos | 66.09 | 49.94 | 62.25 | 74.82 | 83.94 | 43 |
| cos_c | 68.07 | 53.04 | 63.96 | 76.26 | 86.16 | 36 |
| score | 69.88 | 52.81 | 66.02 | 80.52 | 87.17 | 34 |
| rcg | 72.00 | 55.20 | 67.52 | 81.58 | 87.19 | 35 |
| cos_oracle | 83.08 | 71.55 | 79.94 | 84.24 | 90.94 | 1 |
| cos_c_oracle | 83.81 | 71.66 | 80.41 | 84.90 | 92.18 | 1 |
| token ceiling | 91.70 | 80.15 | 89.21 | 94.09 | 97.11 | 0 |

## geometry of the move of the object signature, by FoRIS best-cut IoU
FoRIS 0-0.2: n=34; cos(ref, query signature) 0.47; after centring each image 0.21; cos(move, offset of image means) 0.41; length of move 0.87, of image-mean offset 0.54; share of the move inside the query's 10 main directions 0.27; best-cut IoU cos 0.15, cos_c 0.15, ms[0.1] 0.05, cos_oracle 0.64
FoRIS 0.2-0.5: n=54; cos(ref, query signature) 0.57; after centring each image 0.37; cos(move, offset of image means) 0.41; length of move 0.79, of image-mean offset 0.53; share of the move inside the query's 10 main directions 0.24; best-cut IoU cos 0.36, cos_c 0.38, ms[0.1] 0.13, cos_oracle 0.66
FoRIS 0.5-0.8: n=210; cos(ref, query signature) 0.65; after centring each image 0.48; cos(move, offset of image means) 0.41; length of move 0.70, of image-mean offset 0.54; share of the move inside the query's 10 main directions 0.20; best-cut IoU cos 0.62, cos_c 0.64, ms[0.1] 0.23, cos_oracle 0.75
FoRIS 0.8-1.01: n=302; cos(ref, query signature) 0.71; after centring each image 0.54; cos(move, offset of image means) 0.40; length of move 0.63, of image-mean offset 0.53; share of the move inside the query's 10 main directions 0.19; best-cut IoU cos 0.82, cos_c 0.85, ms[0.1] 0.53, cos_oracle 0.88
