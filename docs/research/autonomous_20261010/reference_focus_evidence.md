# Independent reference-focus200 evidence audit

2026-10-10. **Actual newly encoded reference focus improves the fixed PACO
equal-view head over its derived-feature control, but harms Deep and does not win
the complete strong controls on either panel.** New source visual observation is
real; its usefulness as a reliable query identity/readout mechanism remains unproven.
No view, threshold or selector was changed after the result.

## Complete output and identity verification

All200 predictions were sealed before this audit accessed query GT. The audit source
and helper hashes were recorded before GT access in
`evidence/local/autonomous_20261010/reference_focus_evidence/ANALYSIS_FROZEN.json`.
`audit.py` reproduces the checks, `verification.json` preserves all class I/U and
paired intervals, and the episode/field diagnostics are in `scored_episodes.jsonl`
and `mechanism_episodes.jsonl`.

Independent checks passed in21.921 seconds:

- 4,800 candidate I/U recounts: six fields×direct/CRF×two frames×200 episodes;
- 1,200 complete-control I/U recounts for FoRIS, MEAN and region.fast;
- 9,600 pixel-edit identities against FoRIS and derived.equal.crf;
- 6,000 packed-mask geometries,1,200 signed FP32 field shapes/finiteness checks;
- 200 reference-mask and200 query-mask identities;
- 200 independently recreated physical-mass/role-quadrature constructions;
- 2,263 file SHA checks and1,400 raw-request entry/profile/tensor identity bindings;
- 400 parent episode/frame ledgers reproduced exactly;
- 24 independently recreated default paired95% intervals matched within1e-10.

The six registered fields are actual/derived×global/local4/equal. Every field is
FP32128². The executed renderer is **FP32 bilinear128→1024, strict>0**, followed
by the original boundary CRF for CRF arms and binary bilinear original-size
rendering>0.5. The audit retains the saved masks and this frozen source identity;
it does not substitute a FP64 renderer or another CRF.

Raw feature payloads were already read/hash-verified by producer and head runner.
This audit independently checks their entry SHA and recorded payload/tensor
identities, without rereading O24 arrays or replaying an encoder/head/CRF. Thus the
raw readback assertion remains producer/runner evidence; it is not misrepresented
as a second independent tensor computation here.

## Final segmentation results

|CLI1024, same complete outputs|Deep100 road pooled I/U|PACO100 observed87 class/fold slots|
|---|---:|---:|
|Full FoRIS CRF|29.377542|45.629834|
|Full MEAN|26.465369|46.824561|
|Existing region.fast|37.344028|40.914271|
|Fixed actual.equal CRF|25.619896|39.056805|
|Matched derived.equal CRF|26.979236|35.977374|
|Actual global CRF|23.079959|41.163913|
|Derived global CRF|23.865591|39.782990|
|Actual local4 CRF|26.884548|36.003119|
|Derived local4 CRF|28.339876|31.911821|

The default actual.equal CRF minus derived.equal is **−1.359340 Deep**, conditional
paired95% interval[−2.366162,−.393699], and **+3.079431 PACO**,
[+2.801999,+3.321582]. Against FoRIS it is−3.757646 Deep,
[−7.732533,+.416378], and−6.573029 PACO,[−7.335213,−5.453593]. Against region.fast
the deficits are−11.724132/−1.857467. These are conditional development-panel
intervals, not uncertainty over untouched images or model selection.

Direct equal outputs are25.089428 actual /26.268256 derived on Deep and38.677500 /
35.839013 on PACO. CRF does not change the default conclusion. Choosing local4 for
Deep and global for PACO after these scores would be a new dataset-specific
selection rule; neither selected actual view beats FoRIS anyway. PACO global's
small advantage over region.fast does not win the stronger FoRIS/MEAN controls.

Original equal-CRF scores are25.619896/26.979236 Deep and39.008701/36.060131 PACO,
with original FoRIS29.377542/45.563915. The original-frame PACO gain over derived is
2.948570. The fixed303-slot PACO CLI appendix gives actual11.115208,
derived10.221001,FoRIS13.055938; original gives11.102049/10.243804/13.038155.
The87-slot and303-slot absolute scores must not be mixed.

## What changes with actual source focus

Node rankings use each native128 field with exact canonical1024 foreground/
background pixel mass per8×8 cell. Whole/FoRIS-FG/BG ROIs are actual saved pixel
masks. Added/deleted ROIs are where the fixed actual.equal CRF changes FoRIS;
all six fields are ranked on those same regions. They are output-conditioned
diagnostics, not a new segmentation, pixel-resolution ranker or threshold oracle.

|Equal field, macro AUC; valid n|Actual|Derived|
|---|---:|---:|
|Deep whole;100|.947390|.948316|
|Deep FoRIS-FG;100|.740592|.734403|
|Deep FoRIS-BG;100|.929263|.930491|
|Deep actual-added ROI;97|.740345|.737876|
|Deep actual-deleted ROI;77|.712168|.704589|
|PACO whole;100|.903303|.905327|
|PACO FoRIS-FG;95|.744748|.749004|
|PACO FoRIS-BG;96|.872473|.879841|
|PACO actual-added ROI;60|.638506|.652477|
|PACO actual-deleted ROI;94|.670602|.672351|

Deep actual reference fit improves slightly: full physical binary-role loss
.324028 versus.330153. Its existing-FG and hard-region ordering improves slightly,
yet the complete output loses score because the readout adds too much false
foreground. PACO source fit worsens (.142125 versus.136170), and equal-field
ordering is slightly worse across the same whole/FG/BG/hard ROIs despite its final
mIoU gain. This separates new visual information, source fitting, query ranking
and absolute zero readout; none guarantees the others.

|CLI1024 equal-CRF pixel totals|TP|FP|FN|
|---|---:|---:|---:|
|Deep actual|3,266,702|8,242,632|1,241,311|
|Deep derived|3,123,388|7,068,994|1,384,625|
|Deep FoRIS|2,071,221|2,542,342|2,436,792|
|PACO actual|7,512,192|4,236,018|8,579,518|
|PACO derived|6,362,223|3,082,031|9,729,487|
|PACO FoRIS|11,307,596|9,339,473|4,784,114|

Actual adds143,314 net TP but1,173,638 FP versus derived on Deep; the precision
cost exceeds the recovery gain. On PACO it recovers1,149,969 TP at1,153,987 more FP,
improving class-pooled mIoU by reducing the derived head's under-recovery. It still
misses3,795,404 more target pixels than FoRIS. This is a recall/allocation gain,
not evidence that the new crop consistently improves query identity ranking.

The query-view controls retain the earlier domain difference: actual local4
FoRIS-FG AUC.744492 exceeds global.681433 on Deep; on PACO global.761059 exceeds
local4.707250. Global and native fields share the source fit and annotation;
query observation affects the discrimination available in the region. This
diagnosis does not select a deployment view or claim a universal task adapter.

## Physical supervision and encoding cost

Independently reconstructing every lawful reference mask confirms the exact
inverse-overlap accounting: whole16×16 footprints have area256 outside the focus
and128 inside; focus8×8 footprints have area32. Total area is1,048,576 and both
FG/BG pixel masses reproduce the full reference exactly. The saved focus coverage,
all128 foreground and128 background physical-mass quantile occurrences, collapsed
sample IDs and occurrence counts match each frozen receipt. Actual and derived
fits share all those labels, weights, sampled IDs and binary-risk terms.

The registered comparison isolates actual encoded focus from derived focus while
both fits use whole+focus supervision. A matched whole-only arm with this exact
physical quadrature was not registered. Historical heads use different quadrature,
resolution or risk definitions, so comparing them cannot uniquely identify a
causal benefit of adding focus versus using no focus at all.

This excludes an accidental foreground-area prior or label duplication advantage.
It does **not** establish visual purity of an encoder token: contextual receptive
fields and the same correlated reference annotation remain. The added actual
crop has new encoded visual evidence; the derived crop is raw whole32² bilinear
to64² with no encoder. Geometry alone cannot prove the new evidence transfers.

Producer receipt:200 requested episode views,196 unique inputs,192 newly encoded,
four cache-hit unique inputs and four duplicate episode views;96 batch-two MPS
encoder calls;3,221,275,776 new payload bytes. Actual encoding took334.109s and
producer wall time365.476s, with query RGB/GT/head/scoring calls0 during production.

The subsequent paired six-field cached experiment took366.676s wall with four CPU
workers×two threads. Per-episode means: raw IO.5777s, source gate.0316s, paired head
.3317s, rendering.0187s, serialization.0449s and six CRFs6.1789s; total7.1845s.
Encoder forward and raw-cache writes are0 in this head phase. These measured
costs are kept separate. They are not the latency of the single actual.equal
method or a cold end-to-end run, and dividing the paired cost by six would not
establish either quantity. A cold actual method uses seven encoded observations;
the six query/whole-reference observations were already cached here.

The supported conclusion is a genuine PACO gain from an actual added source
observation under matched physical supervision, alongside a Deep regression and
large strong-control deficits. The fundamental source-to-query identity and signed
readout problem remains; this version is not a completed cross-dataset method.
