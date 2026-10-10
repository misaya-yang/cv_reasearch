# Independent reference-query mixture600 evidence audit

2026-10-10. **The saved prototype scalar preserves good whole-query ranking,
but its reference foreground/background coordinates do not transfer as a stable
query class calibration.** Source APD improves average AUC on all four exposed
panels while increasing the clipped prior-coordinate error on all four. This
falsifies the necessary scalar-moment premise for this simple pure-label-shift
coordinate on many examples; it is not a new prior estimator or segmenter.

## Scope and verification

The canonical diagnostic is
`cv_data/a/reference_query_mixture600_20261010`, with frozen source
`30f844ee60226e365d58c96a183d2cfcfbcaaa7bb0c18b1643f1636eac44ce62`.
The independent implementation and evidence are under
`evidence/local/autonomous_20261010/reference_query_mixture_evidence/`:
`audit.py`, `ANALYSIS_FROZEN.json`, `verification.json` and `scored_episodes.jsonl`.
Auditor source was frozen before it opened pixel labels.

All600 saved-query-field audits passed in17.777 seconds:

- 600 query-mask SHA and exact canonical1024/64-cell coverage checks;
- 600 reference-mask SHA/coverage checks, plus two repeated spot references;
- 1,200 rawunit/source-APD scalar AUC, conditional-moment, prior-error and ledger
  comparisons;
- eight dataset×representation pooled mean/median/p10/p90 dictionaries and counts;
- 1,200 parent/raw-request metadata bindings and1,829 file SHA checks;
- source affine-centroid geometry, separation and clipped-segment residual checks;
- maximum ledger numeric difference and raw-error decomposition residual
  **2.220446049250313e−16**.

Only the two predeclared indices0 and100 were replayed from actual cached O24:
Deep `dg18-0000` with APD off and PACO `dev_s1/paco_part/0/0` with APD on. Four
reference/query FP32 arrays passed payload/tensor SHA; both representations'
recomputed query coordinates and three centroids were **bitwise identical** to the
saved fields. Encoder calls0. This is a two-episode raw spot check, **not** a second
full600 raw-feature computation. The full cache read/transform verification remains
root-generator evidence. Original field-generation time was133.111 seconds.

Every diagnostic field was sealed before its original scorer read query GT, and
records retain query-GT/baseline reads0 during generation. This audit recomputes
post-seal statistics; it does not fit a threshold, modify features, write a mask or
choose an estimator using those labels.

## Exact coordinate and error identity

Using the lawful continuous reference coverage, define the conditional centroids
`μ_RF, μ_RB`, `d=μ_RF−μ_RB`, and the saved scalar

```text
s(q)=(q−μ_RB)·d / ||d||²
π_coord=mean_Q s(q)
π_clip=clip(π_coord,0,1).
```

Nonzero reference separation makes `E_RF[s]=1` and `E_RB[s]=0` **by construction**.
Those source equalities validate algebra and normalization, not transfer. With
the exact query coverage priorπ_GT, the following identity holds:

```text
π_coord−π_GT
  = π_GT (E_QF[s]−1) + (1−π_GT) E_QB[s].
```

The two right-hand terms are the foreground/background shift biases. The identity
applies to the **unclipped** coordinate. The clipped errorπ_clip−π_GT is reported
separately and is not substituted into the decomposition. All1200 representation
instances had nondegenerate source separation and two positive-area query roles.

Coverage assigns the same encoder-cell scalar to all its canonical pixels; AUC
weights foreground/background by the exact16×16 pixel masses. This is global
64-cell ranking, not hard-region accuracy, an original-resolution probability or
segmentation mIoU.

## Source APD: good ranking, poor absolute transfer

|Panel; valid n|Mean π_GT|Mean π_coord|Mean π_clip|Mean E_QF[s]|Mean E_QB[s]|Macro AUC|Clipped-prior MAE|
|---|---:|---:|---:|---:|---:|---:|---:|
|Deep100|.042992|.163813|.208035|.813858|.142463|.906429|.175280|
|PACO100|.153463|.181214|.192579|.547127|.126928|.910743|.125627|
|COCO200|.128649|.251122|.252052|.666289|.196536|.954366|.154707|
|LVIS200|.070946|.160307|.164740|.598175|.132700|.959203|.124843|

Query conditional coordinates are systematically unlike the reference1/0 pair.
For example Deep's mean target coverage is.0430, yet mean clipped coordinate is
.2080. High AUC therefore does not establish that the reference affine zero/one
scale can supply a reliable query class prior.

|Source APD per-episode bias decomposition, macro mean|FG shift bias|BG shift bias|Unclipped signed error|
|---|---:|---:|---:|
|Deep100|−.010389|+.131209|+.120821|
|PACO100|−.073683|+.101434|+.027751|
|COCO200|−.045649|+.168122|+.122473|
|LVIS200|−.029692|+.119053|+.089360|

These are means of the individually weighted terms, not products of the separately
reported meanπ and meanconditional coordinates. PACO's small average signed error
is cancellation, not accurate individual estimates: its clipped-coordinate MAE
is.125627 and its median absolute error.097862. Background shift dominates the
mean positive error on the other panels.

Source APD has nonpositive query foreground-minus-background coordinate gap in
1/100 Deep,3/100 PACO,2/200 COCO and1/200 LVIS. Unclipped coordinate lies outside
[0,1] in23/100,12/100,4/200 and17/200 respectively. Clipping removes that visible
range failure but does not create transferable class-conditional distributions.

## Raw-unit control and geometry

|Panel|Raw-unit AUC|APD AUC|Raw-unit clipped-prior MAE|APD MAE|
|---|---:|---:|---:|---:|
|Deep100|.891362|.906429|.136405|.175280|
|PACO100|.904746|.910743|.106785|.125627|
|COCO200|.932010|.954366|.130541|.154707|
|LVIS200|.943698|.959203|.100585|.124843|

The representation intervention improves ranking while worsening this absolute
coordinate estimator. Thus an APD accuracy/ranking result cannot be recycled as
evidence for source-only probability calibration. This comparison does not imply
APD should be removed from an otherwise different complete method.

The source-APD mean residual distance of μ_Q from the clipped reference-centroid
segment, divided by source FG/BG separation, is1.129645 Deep,.898558 PACO,.736902
COCO and.713487 LVIS. Raw-unit values are1.009042/.815102/.655959/.635396.
This is a descriptive vector-geometry mismatch. It uses the clipped least-squares
coordinate and does not measure only the foreground component or establish an
independent probabilistic model.

## Necessary-condition limits

The pure-label-shift premise would preserve both full class-conditional feature
distributions; matching their vector first moments is already weaker. Matching
only their scalar projection to1/0 is weaker still. An accurate π_coord can also
arise from canceling foreground/background shifts while both projected conditional
means are wrong. A low segment residual can hide distribution changes orthogonal
to this probe or canceling between the classes.

Therefore none of the following is certified by this diagnostic: full conditional
mean equality, full distribution transfer, a calibrated posterior, identifiable
class proportion under arbitrary shifts, high difficult-region discrimination, or
an improved segmentation method. Separate synthetic counterexamples in the scene
review address these sufficiency failures; this audit does not merge those toy
examples into an actual benchmark result.

All statistics describe the same historically exposed Deep custom100/PACO100/
COCO twofold200/LVIS first200 panels. No inferential significance, untouched
confirmation or ten-dataset generalization is claimed. The supported mechanism
finding is that the reference/query roles change both location and scale along
this useful discriminative direction, defeating simple source-only label-shift
calibration even when global query ordering remains good.
