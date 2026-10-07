| level | direct positions mIoU; Δ vs matched s2 [95% CI] | RCG positions mIoU; Δ vs matched s2 [95% CI] | target>wrong, direct/RCG |
|---|---|---|---|
| L1 FG mean cosine | 46.8470; -4.8819 [-7.0307, -3.5664] | 49.7460; -5.6382 [-8.1072, -4.2112] | 32/81 ; 22/81 |
| L2 minus.55 all-BG mean | 49.8131; -1.9159 [-3.5979, -0.4355] | 53.0594; -2.3248 [-4.1989, -0.6866] | 31/81 ; 26/81 |
| L3 hardest20% BG token mean | 52.7628; +1.0339 [-0.1965, +2.1362] | 55.3702; -0.0140 [-1.2285, +1.2822] | 32/81 ; 23/81 |
| L4 orthogonalized hard-BG mean | 53.2289; +1.5000 [+0.1790, +2.5144] | 55.8685; +0.4843 [-0.7096, +1.7279] | 31/81 ; 25/81 |
| L5 FG prototype maximum | 50.6542; -1.0747 [-1.4294, -0.6191] | 55.2327; -0.1515 [-1.0028, +0.3172] | 28/81 ; 24/81 |
| L6 FG prototype LSE T.07: native s2 | 51.7304; +0.0015 [-0.0109, +0.0092] | 55.3870; +0.0028 [-0.0058, +0.0088] | 28/81 ; 24/81 |
| no APD | unavailable in this processed cache | unavailable | — |

Position-level binary-token class IoU; reused development fresh600. Baseline: stored_s2. Native s2 replay drift: 0.0014587903061027419 pp. Gates were chosen after acceptance1.
