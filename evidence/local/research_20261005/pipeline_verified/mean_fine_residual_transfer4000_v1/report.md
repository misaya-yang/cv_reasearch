# Fixed MEAN/fine residual transfer on all4000 draws

New candidate; original fine64 primary is unchanged. Coefficient1, no clip/refit; all source/mask parities required before GT.

| Arm | Class-summed macro mIoU |
|---|---:|
| native | 60.931741 |
| rcg | 62.333671 |
| mean.control | 62.512972 |
| rcg64.control | 62.297937 |
| fine.rcg16.control | 62.709100 |
| fine.rcg64 | 62.651617 |
| mean_fine_residual_transfer_v1 | 62.844537 |

Candidate minus fixed controls (paired2000 RandomState0 connected-photo95% CI):

- native: +1.912796 [+1.618404, +2.210338] pp; up/down/tie 2509/1448/43.
- rcg: +0.510866 [+0.332586, +0.716025] pp; up/down/tie 2642/1269/89.
- mean.control: +0.331565 [+0.303193, +0.372805] pp; up/down/tie 2841/1062/97.
- rcg64.control: +0.546600 [+0.278019, +0.890335] pp; up/down/tie 2521/1378/101.
- fine.rcg16.control: +0.135437 [-0.048326, +0.334529] pp; up/down/tie 1978/1929/93.
- fine.rcg64: +0.192920 [-0.078619, +0.522409] pp; up/down/tie 2088/1811/101.

All4000 original six-arm I/U match exactly. Natural repeated draws remain retained. Four-edit counts are relative to actual complete native and named controls; full4000 raw-origin edits are unavailable.
C is not directP(G). A CI crossing zero is unresolved; this exposed benchmark is not independent confirmation.
