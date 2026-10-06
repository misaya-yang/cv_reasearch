# Independent full4000 scalar-graft statistics verification

Candidate macro mIoU: **62.844537**.

| Comparator | Candidate gain [paired95% CI], pp | Up/down/tie |
|---|---:|---:|
| native | +1.912796 [+1.618404, +2.210338] | 2509/1448/43 |
| rcg | +0.510866 [+0.332586, +0.716025] | 2642/1269/89 |
| mean.control | +0.331565 [+0.303193, +0.372805] | 2841/1062/97 |
| rcg64.control | +0.546600 [+0.278019, +0.890335] | 2521/1378/101 |
| fine.rcg16.control | +0.135437 [-0.048326, +0.334529] | 1978/1929/93 |
| fine.rcg64 | +0.192920 [-0.078619, +0.522409] | 2088/1811/101 |
| mean_fine_bit_edit_transfer.control | +0.100180 [+0.079995, +0.128147] | 2652/1254/94 |

All4000 original six-arm I/U, per-episode exports and four-edit identities match exactly. Saved scores, CIs, folds and batches reproduce with zero error. The fixed bit control shares the same original4000 source counts and manifest.
One CPU count/statistics verification only: no masks changed, encoder run, query GT indexed, parameter sweep or inference snapshot edit. Source replay remains1200 shared/2800 new, with original RCG/MEAN and all fine16 renderer mismatches0.
C is not directP(G); the historical fine64 primary is unchanged. Native edits keep their complete-native origin; full4000 raw-origin edits remain unavailable. CI crossing0 is unresolved; this reused benchmark is not independent confirmation.
