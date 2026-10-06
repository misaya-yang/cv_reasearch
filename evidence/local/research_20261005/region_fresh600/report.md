## ordering score by field, 600 episodes; and AUC of true object tokens against the tokens RCG wrongly keeps
| field | ordering score | AUC, RCG best-cut IoU < 0.2 | AUC, 0.2-0.5 | AUC, 0.5-0.8 | AUC, >= 0.8 |
|---|---:|---:|---:|---:|---:|
| cos@11 | 18.95 | 0.44 (n=33) | 0.53 (n=41) | 0.52 (n=122) | 0.57 (n=147) |
| cos@15 | 20.89 | 0.44 (n=33) | 0.55 (n=41) | 0.53 (n=122) | 0.60 (n=147) |
| cos@19 | 27.70 | 0.42 (n=33) | 0.59 (n=41) | 0.56 (n=122) | 0.64 (n=147) |
| cos@23 | 61.83 | 0.44 (n=33) | 0.62 (n=41) | 0.59 (n=122) | 0.69 (n=147) |
| cos | 66.10 | 0.48 (n=33) | 0.59 (n=41) | 0.58 (n=122) | 0.68 (n=147) |
| score | 69.88 | 0.28 (n=33) | 0.41 (n=41) | 0.62 (n=122) | 0.75 (n=147) |
| rcg | 72.00 | 0.14 (n=33) | 0.30 (n=41) | 0.65 (n=122) | 0.80 (n=147) |

## inside the window around the true object: native tokens against tokens of the enlarged window
| episodes | n | window side / image | cos(ref signature, object signature) native | enlarged | window mIoU native tokens | enlarged tokens | mean episode IoU native | enlarged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 600 | 0.84 | 0.66 | 0.67 | 68.28 | 68.19 | 0.69 | 0.69 |
| RCG best-cut IoU < 0.2 | 35 | 0.70 | 0.47 | 0.48 | 23.82 | 22.74 | 0.23 | 0.23 |
| 0.2-0.5 | 48 | 0.79 | 0.56 | 0.56 | 43.31 | 43.31 | 0.44 | 0.44 |
| 0.5-0.8 | 191 | 0.85 | 0.65 | 0.65 | 62.20 | 62.78 | 0.62 | 0.62 |
| >= 0.8 | 326 | 0.86 | 0.71 | 0.71 | 82.35 | 82.24 | 0.82 | 0.82 |
| window < 0.75 of the image | 154 | 0.50 | 0.62 | 0.64 | 68.85 | 69.01 | 0.69 | 0.69 |
| window < 0.75 and RCG IoU < 0.5 | 35 | 0.50 | 0.47 | 0.48 | 42.15 | 40.79 | 0.42 | 0.41 |
| window >= 0.75 | 446 | 0.96 | 0.67 | 0.68 | 69.13 | 69.00 | 0.70 | 0.69 |
