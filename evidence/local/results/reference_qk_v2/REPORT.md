# Fixed single-reference QK method: completed development experiment

Completed 2026-10-04. COCO-20i, native val2014 images and official semantic masks,
seed 0, four folds with 10 previously exposed development episodes each (40 total).
All arms use the same reference mask, query, frozen DINOv3 ViT-L/16, 1024 input,
and complete public FoRIS pipeline, including RGB/position grouping and CRF.
Each pair is encoded once. All 40 predictions froze before query labels were opened.
Metric: original-resolution class-mIoU, averaged over the four folds. Intervals use
2,000 paired bootstrap draws of connected support/query photograph groups; here
there are 40 groups, each containing one episode. These are development results.

## Measured results

| Fixed arm | Class-mIoU | Difference from native, pp | Paired 95% interval |
|---|---:|---:|---:|
| Complete native FoRIS |65.1614|0|[0, 0]|
| Duplicate-z contract |65.1615|+0.0001|[0, +0.0002]|
| Constant context |59.0895|−6.0719|[−9.0044, −2.8538]|
| Final-z conditional centroid |55.2145|−9.9468|[−13.4546, −6.5196]|
| Q-only |58.7462|−6.4151|[−8.9454, −3.6649]|
| Pre-Part1 query-only grouping |65.2185|+0.0571|[−0.8443, +0.9629]|
| Main conditional native QK |58.7041|−6.4573|[−9.0505, −3.9193]|
| Matched 3072-dimensional local bank |57.9780|−7.1834|[−10.0259, −4.3514]|
| Matched 3072-dimensional shared bank |54.1045|−11.0568|[−15.0264, −6.8608]|
| GT-assisted fixed-family pixel oracle |76.1986|+11.0373|[+7.3801, +14.2424]|

The oracle chooses individual pixels from the union/intersection of these fixed
predictions using query truth. It is a diagnostic of this output family, not an
achievable selector or an upper bound on all frozen-model information.

Main minus Q-only: −0.0421pp [−2.3966, +2.2448]. Main minus the strongest cheap
control, pre-Part1 query-only: −6.5144pp [−8.9359, −4.1467]. Local minus shared,
with width/self-exclusion matched: +3.8735pp [+0.4739, +7.3670]; both still lose
to native. Main fold differences are −5.3893 / −7.2432 / −7.4747 / −5.7219pp.
Per-episode main outcomes: 6 improve, 27 worsen, 7 remain unchanged; all 7 legal
fallback cases stay in the denominator. The positive +0.0571pp query-only result
is reported with its uncertainty, not erased because its interval crosses zero.

## Error accounting from saved predictions

The main arm recovers 19,040 native false-negative original pixels but loses
89,686 previously correct foreground pixels. It deletes 63,990 native false
positives but introduces 233,913 new false positives. These are pooled pixel
counts, not class-mIoU contributions.

On the 33 non-fallback cases, the saved source-template diagnostic gives mean
context margins −0.0325 for 1,205 missed-target patches with no assignment,
−0.0373 for 649 assigned missed-target patches, and +0.1616 for 4,093 native
false-positive patches. Of those false positives, 3,400 have a positive context
margin. The measured direction does not support the hoped-for simultaneous
promotion of missed target parts and suppression of confusers. These template
margins use nearest-sampled patch labels; they are not the complete host score
and do not establish a causal explanation by themselves.

## Inference and limits

The fixed equal-weight augmentation is ineffective on this cohort. Its failure
is an observed aggregate loss, not a rule based on an interval crossing zero.
QK does not establish a task advantage over the cheap controls. Constant context
also loses about six points, so host geometry changes remain a substantial
confound. The local/shared difference supports only a relative bank effect in
this unsuccessful construction. It does not establish a useful method.

This result does not prove that every QK readout or the complete frozen
representation lacks relevant information. It does not justify a layer/head,
weight, temperature or threshold search on these exposed episodes. No expansion
or new GPU queue is scheduled for this fixed construction.

## Execution and provenance

All 40 native masks match their saved complete-public counterparts exactly; all
cached NoOp masks and score fields are exact. Duplicate-z differs by one original
pixel across the cohort, recorded rather than called bit-exact. The actual
post-QK-normalization/post-RoPE SDPA inputs are captured without changing the
encoder output. Part2–4 are recomputed per arm, preserving the published
candidate normalization behavior.

The first invocation failed before any prediction because the existing CRF
runtime was missing from PYTHONPATH. Its error and shutdown receipt remain in
`../reference_qk_v1/`. Version 2 only adds the already-installed CRF and extension
paths; source, weights, masks, parameters and cohort are unchanged. There was
no download, recompilation or base-environment modification.

Version 2 prediction loop:327.3019s; complete inference child:331.7915s;
CPU report:2.2775s. Guard requested shutdown at337.2900s. AutoDL independently
showed E58 stopped. A no-card restart retrieved the compact evidence; final
no-card shutdown is recorded in `resource_receipt.json`. No full-token or
attention-matrix cache was written. Observed GPU allocation during inference
was approximately3.4GiB.

Scripts: `scripts/reference_qk_experiment.py`, `tics/reference_qk_descriptor.py`.
Evidence: `summary.json`, `scored_records.json`, `execution.json`, `diagnosis.json`,
packed masks, frozen per-case records, runtime/preflight proofs and guard log.
Input specification SHA256:
`6d480556bd2ff9080421d9ccbdb66b17caff1c58567d6cedb6c59669678da4d9`.
Runner SHA256:
`eee0ed270b3091bf482e0c722a67fc418d3797bba16ed14afabf5784eb6aa5e7`.
Descriptor SHA256:
`b752eeba1fabdfcb20cb5ef8f7c14e912159680d50e56dbb4a3cae2ffd5bc0e5`.
