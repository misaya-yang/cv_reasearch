# First complete GPU comparison: first20

COCO-20i, one reference, frozen DINOv3, seed0, 20 exposed DEV episodes, 18 classes, 20 photograph groups,
full 1024 native masks. Selection used source order stratified by four folds and recorded batches, not GT.
Paired class-summed mIoU and 2,000 connected-photo RandomState(0) resamples. No independent confirmation.

Native 55.096; RCG 55.628 (+0.532 [-1.032, +1.505]). GPU native replay and all first-episode SDPA/layer
identity audits passed. Four-case smoke took 61.06 s; first20 complete pipeline succeeded in about 213 s.

| Complete arm | mIoU | Paired gain vs native [95% CI] | Up/down/tie |
|---|---:|---:|---:|
| A transport | 41.963 | -13.134 [-20.453, -5.191] | 3/16/1 |
| B structure | 40.255 | -14.841 [-25.492, -6.660] | 3/16/1 |
| C intervention | 55.174 | +0.078 [-0.027, +0.141] | 9/10/1 |
| D multilayer | 58.356 | +3.260 [-0.142, +5.707] | 12/7/1 |
| E latent (raw-score interface bug) | 48.716 | -6.380 [-11.986, -2.170] | 7/12/1 |
| F reconstruction | 25.966 | -29.130 [-39.332, -22.452] | 2/18/0 |

D's stronger same-layer concat control is 53.701; D-control +4.655 [-1.592, +9.545]. D folds are
-0.154/-0.292/+12.568/+2.097. D batches: old20 +0.383, new40 -1.142, new60 +14.670,
arrived100_exposed +0.092. Its +3.260 pooled effect does not establish stable +2. Complete DEV241 with
unchanged D plus a frozen delta-only complete FoRIS control is the next comparison. Fixed add-only and
delete-only action ablations are diagnostics, not per-episode method selection.

A/B/F loss is dominated by false-positive expansion after replacing the complete host's scope inference.
Their next revisions change the specifically measured witness-selection, joint-correspondence prior and
inverse-label operation, respectively; they do not rerun the original formula with new parameters.
E's raw response clipping was an implementation mismatch: native requires per-image minmax. It is repaired
in a separate unchanged-parameter wrapper, retaining all old results. C's score field barely changes;
its near-zero gain equals its key-only control, so no layer/strength sweep is scheduled.

Exact results, per-episode four-way counts and runtime receipts: [cache](runs/outputs/gpu_first20_v1/cache/report.json),
[forward](runs/outputs/gpu_first20_v1/forward/report.json). All predictions and actual layer outputs remain
on the owned server workspace; query truth was read only after complete prediction sealing.

Execution note: unstable SSH left one dispatch outcome unknown. A subsequent smoke dispatch was rejected
by the existing-output check; the scientific smoke executed once and its complete artifacts were not
replaced. Later dispatches use fresh exclusive launch directories. Do not interpret the second guard's
FileExistsError as failure of the first completed scientific run.
