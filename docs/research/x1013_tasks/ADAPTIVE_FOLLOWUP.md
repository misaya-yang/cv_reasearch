# Area-gated S-half composition check (2026-10-13)

Offline follow-up to the T0–T5 scale diagnosis. It tests the three explicitly proposed S-half arms against the existing A4 candidate. The inputs are exposed development caches, not a confirmation set. Scope is the selection half only: 6,752 tasks. T was not read or scored. No new DINOv3 inference or model construction occurred.

Motivation: T2 showed that removing V as A3 helped COCO and ISIC as a fixed arm but hurt PACO and Road; T5's predicted-area-gated F1b helped ISIC/SUIM and collapsed Lung. The question here was whether the area gate and a conservative cluster rule retain the local improvements without damaging Lung. Positive results would have justified considering a frozen formula for later evaluation; a negative result rules out these exact S-half compositions.

## Arms

All three use the A4 predicted-mask area to select the large-target branch (`candidate[2] / 1024² > .25`); otherwise they return A4.

| Arm | Large-target branch |
|---|---|
| S1 | T2 A3 (no V) |
| S2 | F1b-safe: average cluster scores only when cluster SD of `VP_UN` is `< .22` and cluster size is `< .35 × 4096`; unsafe clusters keep their pointwise scores. Read out `0.5 × VP_UN + 0.5 × safe_field` with the frozen min-max `>.5` cut. For `rho > .9`, retain A4. |
| S3 | Cached E1 `C1` CRF mask OR normalized A3 score `>.65`. |

S3 follows the requested `T7 OR [A3 > .65]` check. It does not test the earlier continuous-alpha `U*` proposal. A3 was replayed from saved fields for the 1,436 non-fallback large-target cases; its `>.5` mask matched cached T2 A3 exactly. The remaining large-target fallback case kept A4. CRF masks came from the already-produced E1 C1 cache.

## Results

Class mIoU deltas versus complete FoRIS, in percentage points. The metric pools I/U within each observed `(fold, class)` slot, then averages slots equally. Intervals use 300 paired query-photo bootstrap replicates (seed 20261013); full intervals are in the JSON summary.

| Domain | A4 candidate | S1 no-V | S2 F1b-safe | S3 C1 OR A3-core |
|---|---:|---:|---:|---:|
| COCO | −1.490 | −1.625 | −1.302 | −1.423 |
| LVIS | +0.173 | +0.133 | +0.119 | +0.200 |
| PASCAL-Part | +1.162 | +1.116 | +1.177 | +1.218 |
| PACO-Part | +3.151 | +2.995 | +3.077 | +3.203 |
| SUIM | +0.069 | +0.192 | +0.249 | +0.119 |
| iSAID | +3.134 | +3.334 | +2.982 | +3.111 |
| ISIC | −1.467 | −1.120 | −1.504 | −1.514 |
| Lung | +2.183 | +1.915 | +2.116 | +2.136 |
| Fundus | +15.355 | +15.355 | +15.355 | +15.355 |
| DeepGlobe road | +0.203 | +0.203 | +0.203 | +0.203 |
| Ten-domain mean | +2.247 | +2.250 | +2.247 | +2.261 |

Each arm is at or above FoRIS on 8/10 domains. None closes both COCO and ISIC: all remain below FoRIS in those two domains.

### Key paired changes versus A4

| Domain | S1 | S2 | S3 |
|---|---:|---:|---:|
| COCO | −0.135 [−0.275, +0.115] | +0.188 [−0.057, +0.358] | +0.067 [+0.020, +0.095] |
| ISIC | +0.346 [−0.014, +0.685] | −0.037 [−0.252, +0.183] | −0.048 [−0.111, +0.015] |
| Lung | −0.268 [−0.354, −0.185] | −0.066 [−0.146, +0.014] | −0.047 [−0.163, +0.093] |

S1's ISIC recovery is uncertain and COCO moves in the wrong direction; its Lung loss is clear. S2 avoids the F1b-gated Lung collapse (Lung is +2.116 versus FoRIS, compared with −14.97 for unfiltered gated F1b), but it also removes the prior ISIC gain (+1.09 to −1.504) and does not improve SUIM beyond +0.249. In ISIC the median largest cluster is 63.3% of the 64×64 grid, so the size gate rejects its global mean. S3 produces small changes: COCO improves by 0.067 over A4 but remains −1.423 versus FoRIS; ISIC stays −1.514; Lung is statistically unchanged from A4.

Decision: none meets the proposed criterion of flattening COCO and ISIC while preserving Lung. Do not freeze any of these three formulas for T based on this S-half result.

## Reproduction and evidence

Run from the repository root with the `aidemo` Python environment:

```bash
/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/bin/python -B scripts/x1013/adaptive_followup.py
```

The script reads T0/T2/T5 per-example I/U, cached T0 fields, T5 clusters, and E1 C1 packed masks. Sparse replay and score readout operate only on those saved arrays. It writes the full S-half rows to `../cv_data/a/x1013_adaptive_followup/iu.jsonl.gz` and the compact reproducible evidence to [x1013_adaptive_followup_iu.jsonl.gz](../../../evidence/research/x1013_adaptive_followup_iu.jsonl.gz). Aggregate scores and input hashes are in `../cv_data/a/x1013_adaptive_followup/summary.json`.

Frozen input hashes and execution manifest:

- T0 I/U: `d38ce453599a1a7ddd55da7ac647d9cf8b6926da28ef714c68221b3dea12b321`
- T2 I/U: `ba9df38940510eb192b4bb9994e775346ce55396e6175fc2011d01ef3881bb7a`
- T5 I/U: `c22e2626a6275060481e58d02a44d262b8eeee8d124adebd0bdbb7d9bd8be3de`
- S index order: `0856c84098c1898a552a8f75a21541cf7ae988bdcdffc585fb5c9b6617e73f78`
- Large-gate replayed fields: `e6bf042a1c8be40e4ce607f2872f1ac0242ba20e0242caa8afbb1efce889175b`
- Large-gate T5 cluster labels: `a920b2e8ed26a2240a21d2347fd9712bb87dbe7a2345663612327598d78bc217`
- Large-gate E1 C1 packed masks: `5ae4425bc698061f827954b46ba71f0348878a036d30251751d04915ff62cba8`
- Source SHA256: `c19dc85762e182f9af89dd6ee559e6f34f0fb3b0a1ccf51f41a9505150f2c9e2`
