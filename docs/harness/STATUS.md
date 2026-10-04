# Status

Updated 2026-10-04. Replace, never append. At most 40 lines; history stays in the direction README.

## Now

- Live direction: `demo_lists/demo9_transductive_ics`, COCO-20i in-context segmentation.
- The confirmed +3.54 read-out is trained with base-class masks and modifies FoRIS output; it is not an independent method.
- User-reported D1 loses 7–28 points only on its tested constructed-pair setup; T2 PASCAL/PACO gains +0.97/+0.08 have CIs crossing zero (unresolved).
- Raw D1/T2 JSON is not in this checkout; their scope/provenance is in the direction README.
- SAM3 relative_0.7: DEV241 70.84 (+8.22 vs .5); CONFIRM600 67.12 (+6.27 vs .5, +3.49 vs DEV-selected .3).
- A locked-rule read on 915 image-disjoint rest episodes is 68.76; +5.53 vs .5 [+3.18, +8.58] and +2.23 vs DEV-selected .3 [+0.70, +4.81].
- Backward smoke and DEV are complete (10/10 episodes, 40 passes; DEV 241 episodes, 1,129 passes; query labels unopened).
- At 19:19 +08, backward CONFIRM was RUNNING (0/600); no AUC JSON existed. Read its existing completion only; do not relaunch/sweep.
- No GPU run was started or interrupted here. The script wrapper owns the current process and its shutdown command.

## Measured

COCO-20i 1-shot, seed 0, CONFIRM600, original resolution, complete pipeline; paired 95% intervals.

| Method | mIoU | Gain over FoRIS | Column |
|---|---:|---|---|
| FoRIS (public code) | 59.78 | | training-free |
| SAM3, visual exemplar | 61.09 | +1.31 [-2.45, +3.57] | foundation segmenter |
| Union of SAM3 and FoRIS masks | 62.83 | +3.05 [+1.82, +4.39] | two-host control |
| FoRIS + read-out fitted on base classes | 63.33 | +3.54 [+1.96, +4.96] | fitted on base classes |
| SAM3 with true class name | 73.38 | | privileged diagnostic |

- SAM3 relative proposal filter: CONFIRM600 67.12; paired +6.27 [+4.37, +10.07] over fixed .5 and +3.49 [+1.29, +6.13] over fixed .3; per-fold +4.78/+8.82/+2.70/+8.78.
- Published comparators: INSID3 57.6; FoRIS 60.9; UINO-FSS 64.5 (trained); FSS-SAM3 main table 66.6 (body 66.1); CG-ICS 72.3.
- The D1/T2 values above are not independently re-derived here; see the README's explicit user-report note.

## Unverified

- Raw D1/T2 reports, backward DEV/CONFIRM AUC, LVIS/SUIM/lung transfer packs, and exact TRAIN2400 manifest selection.
- Strict class-held-out interpretation of published SegIC 76.1 and UNICL-SAM 77.8 is not supported; see `docs/reference/ladder.md`.

## Needed from the user

- No input needed for this read-only documentation pass. Any GPU execution would require separate explicit authorization.
