# Independent evidence validation — 2026-10-10

The saved PACO ridge-anchor gain is real final-output improvement on the exposed
development panel. It is largely selective foreground pruning, gives only about
0.84–0.85 points beyond the matched uniform-shrink controls, and does not transfer
to a COCO win. The repaired joint-role model fails before geometry: its direct
unary already scores far below FoRIS, and the geometry changes final mIoU by less
than 0.04 points. None of these results establishes a complete method outperforming
FoRIS on all ten benchmarks.

This audit did not run an encoder, read raw features, create candidate predictions,
modify another agent's code, or rerun a solver. Its numerical evidence is under
`evidence/local/autonomous_20261010/evidence_validation/`:

- `audit_saved_outputs.py`: independent scoring and paired-resampling implementation;
- `verified_saved_outputs.json`: per-class pooled I/U, fold means, independently
  reproduced paired intervals, overlap lists and saved-file SHA identities;
- `edit_attribution.json`: independent 24-order edit Shapley valuations.

## 1. What was independently verified

The final audit run passed **8,900 original-frame mask I/U checks, 7,400 original
pixel-edit checks, 1,100 ground-truth array/geometry checks and 3,623 saved-file SHA
checks** in 2.924 seconds. These counts include overlapping panels and repeated
controls; they do not represent 8,900 independent episodes. The unique episode
pool is Deep100 + PACO600 + COCO200. The original joint pilot200 is a subset.

Candidate masks, complete FoRIS masks, and MEAN masks were unpacked directly and
recounted against their saved original-frame scoring masks. All point estimates
matched the existing reports to less than 1e-10. The independent implementation
also reproduced the 10,000 paired photo-within-fold/class bootstrap intervals.
Original and CLI1024 results remain separate. No token-level AUROC is treated as
an attained segmentation score.

|Same panel, original frame|Metric|FoRIS|MEAN|Strong saved control|Ridge global|Ridge local4|
|---|---|---:|---:|---:|---:|---:|
|Deep100|Road total I / total U|29.377542|26.465369|region.fast 37.344028|32.014976|34.288729|
|PACO600|303 official fold/class slots, absent classes retain zero|38.460868|39.352045|uniform.global 39.934228|40.774944|40.629884|
|COCO200, folds0/1|40 official slots, one missing retains zero|63.358900|64.006821|MEAN 64.006821|62.459822|Not run|

Deep ridge global/local4 beat FoRIS but lose to the existing fast nine-window
control. PACO global/local4 beat FoRIS, MEAN and matched uniform on this panel.
COCO global loses to both FoRIS and MEAN. Choosing different winning arms after
seeing these scores would be dataset-specific development selection.

## 2. What PACO's gain actually does

|PACO600, original frame|mIoU|Delta vs FoRIS|Delta vs matched uniform|Conditional paired 95% interval for last delta|
|---|---:|---:|---:|---|
|Global ridge anchor|40.774944|+2.314076|+0.840716|[+0.633059,+1.298533]|
|Local4 ridge anchor|40.629884|+2.169015|+0.851277|[+0.676737,+1.282090]|
|Matched uniform global|39.934228|+1.473360|—|—|
|Matched uniform local4|39.778607|+1.317739|—|—|

Uniform's arithmetic delta is 63.7% of the global candidate's delta and 60.8% of
the local candidate's delta. This is a comparison of two complete algorithms,
not a causal percentage decomposition: equal per-image original-pixel L1 evidence
does not imply equal edited area, confidence-weighted budget or error distribution.
PACO global improves 160 observed classes and worsens 98 versus matched uniform,
with six ties; the advantage is not universal per class.

The final edit ledger makes the limitation more concrete:

|PACO600 relative to FoRIS|Added TP|Added FP|Deleted TP|Deleted FP|
|---|---:|---:|---:|---:|
|Global ridge anchor|5,304|12,016|171,147|748,411|
|Local4 ridge anchor|5,983|11,739|205,187|825,056|
|Uniform matched global|0|0|333,882|1,036,677|
|Uniform matched local4|0|0|347,570|1,056,201|

Global's four edit Shapley contributions are **+0.110273, −0.085121,
−3.745364, +6.034288** points. Its additions contribute only +0.025152 net points;
almost all net benefit is pruning existing foreground while preserving more true
target than uniform. Local4 similarly has +0.019522 net points from additions.
These are exact valuations over the 24 orders of fixed already-produced edits,
not four deployable submethods or proof of a unique causal mechanism.

For comparison, Deep local4's added-TP contribution is +3.039365 points, and its
deleted-FP contribution +4.024336. Deep's improvement involves meaningful recovery
as well as rejection. Uniform global/local4 instead lose 2.476825/2.405317 points
to FoRIS. A generic shrink interpretation does not explain the Deep gain.

## 3. COCO transfer is a genuine negative result

The fixed global recipe scores 62.459822: −0.899078 vs FoRIS, paired interval
[−2.464037,+0.355781], and −1.546999 vs MEAN, [−3.203799,−0.234389]. It improves
over old whole competition by +1.564006, but that weaker comparator does not turn
the result into a strong-baseline win.

The independent original-frame edit totals are added TP150,909 / FP511,506 and
deleted TP301,062 / FP346,452. The four Shapley contributions are
**+1.011833, −1.648223, −3.132615, +2.869926**, reproducing the −0.899078 net delta.
Both addition and deletion directions lose net score. This rules out explaining
the failure solely as excessive foreground expansion.

The saved post-seal rank diagnosis has a narrower positive finding: new ridge's
mutable-BG macro AUC .713156 exceeds old whole h .644093, whereas existing-FG
macro AUC .749689 does not improve over old h .749789. Reference calibration gives
negative h on 42.68% of existing true positives and positive h on only 20.87% of
existing false negatives. These are canonical1024/128-node, area-weighted
diagnostics from the existing verification artifact; they are not recomputed
original-frame scores or untouched validation. They support a cross-image sign/
readout problem alongside incomplete identity discrimination, not a claim that
one new threshold would solve transfer.

## 4. Joint-role repair: substantial model failure, small wrapper drift

|Original frame, exact same pilot|FoRIS|MEAN|Direct role unary|R1-G joint|Joint minus unary|
|---|---:|---:|---:|---:|---:|
|Deep100, road total I/U|29.377542|26.465369|12.630740|12.596665|−0.034075|
|PACO100, 87 observed class/fold strata|45.563915|46.687946|29.237437|29.275000|+0.037562|

The point totals and original masks were independently recounted. The small
geometry changes are real, but they cannot account for the unary's deficit of
about 16 points. Saved independent mechanism diagnoses show only .6162%/.8072%
original-pixel XOR and mean absolute token probability change .004926/.007684.
The original prior-bound locked false positives are only 14.14%/26.64% of all
unary false positives. Therefore increasing role-correction strength is not
supported as an explanation or sufficient repair of the whole failure.

Static checks of `scripts/joint_role_pilot.py` and its frozen dependencies found
no query-mask reads in `infer`, no encoder fallback, correctly separated original/
CLI1024 renderers, proper class-pooled metric aggregation, and the intended
relative inverse geometry and legal-role position support. The geometry source,
BP source and observation source exactly match their executed frozen copies.
The existing 14 synthetic checks establish implementation functionality; they
do not establish successful segmentation.

One actionable provenance defect remains in the older pilot harness: live
`scripts/joint_role_pilot.py:172–175` uses normalized FP64 b0 for unary, whereas
the executed frozen script uses saved FP32 P0. `run` at lines238–240 validates
the copied frozen files but still executes the live wrapper functions. An edited
wrapper could therefore resume an incomplete panel with mixed rendering semantics
under the same frozen config. The already-sealed run is unaffected, and the saved
verification discloses only four original-frame near-tie pixels; aggregate mIoU
does not change. The correction is to execute the archived wrapper or assert the
executing wrapper SHA at run/infer/score, then preserve the archived P0 result as
the historical comparator. This audit did not edit the harness.

## 5. Exposure and metric limitations

PACO pilot100 and remaining500 overlap in **nine query photographs and 32 reference
photographs**; cropped RGB hashes overlap in five query inputs and 23 reference
inputs. Remaining500 had historical scores/analyses exposed before this recipe,
as its frozen config explicitly states. COCO200 was also historically exposed.
These are useful fixed-recipe extensions, not untouched holdouts.

PACO600 has 264 observed class/fold strata, 39 missing official slots retained at
zero, and 98 observed strata with only one query photo. Bootstrap resampling
within those singleton strata adds no uncertainty. The conditional intervals do
not cover unseen photos, unseen classes, future reference draws, model/recipe
selection or all ten benchmark datasets. Photo grouping is within fold/class;
the same photo appearing in multiple strata is not resampled as one global unit.

PACO100's headline 45.56/49.23 FoRIS/global scores use 87 observed strata; the same
subset with all303 official slots is 13.038155/14.085753. The remaining500 and
all600 reports use fixed303. The subset's absolute score must not be compared
with the full panel's absolute score as learning progress or regression.

All efficiency figures here concern existing features or even existing fields.
The old joint candidate's mean cached per-episode time is10.2357s, including
3.7493s observation and6.1322s geometry. The fixed COCO ridge generation is about
.2467s per episode after whole features and FoRIS outputs exist. Neither is a
cold RGB-to-final-mask inference measurement.

## 6. Strongest untested mechanistic prediction

**The modest PACO gain over uniform should depend on the correct binding between
reference features and the supplied reference mask.** The exact edit ledger says
the learned field mainly preserves true target during pruning. It remains
unproven whether this requires the actual target identity or can arise from a
reference-independent spatial/confidence correction.

A single predeclared binding-destruction control is more informative than another
threshold, blend or lambda search: keep query features, original FoRIS anchor,
reference coverage, sampled fit IDs and rendering fixed; apply one deterministic
permutation to reference feature rows relative to that coverage; reconstruct the
same field. Give both real and permuted fields their own already-defined matched
uniform-L1 controls rather than tune strengths to query labels. If the real-bound
field consistently preserves more true target and retains the +.84/.85 point
advantage over its uniform control while the permuted field falls to uniform or
below, the gain has task-specific evidence. If both perform similarly, the current
claim should stay at selective confidence correction. This is a falsifiable
diagnostic prediction, not a proposed adoptable method, and this audit did not run
it. Equal L1 alone is still an incomplete edit-budget control, which must be stated
in interpreting the result.

## 7. Parent's new scene-reconstruction600 runner review

The parent requested review before execution of
`scripts/autonomous_scene_reconstruction.py`. On the inspected version:

- The head is loaded explicitly from the frozen module path; executing-wrapper
  SHA is checked in `infer` and configuration verification. This avoids the old
  joint harness's wrapper-drift issue.
- `infer` uses only R/Q RGB, legal reference mask and raw O24. Archived mask reads
  are absent. All arms share the released FoRIS input-only APD semantic expression
  and its .8 branch. Direct signed64 → bilinear1024 → strict0 and subsequent
  binary original-size rendering follow the stated evaluation contract.
- All200 LVIS FoRIS/MEAN bindings match RGB hashes, both mask hashes, crops, sizes
  and class IDs. All200 archived COCO FoRIS masks match `matched_raw` bitwise in
  both original and CLI1024 frames; using those archived masks is valid.
- **Freezing correction relayed and now verified in the live source:** the first
  inspected version at lines291–293 called `imports()` with the live REPO default
  before importing `ics.metrics`. The parent changed it to `imports(out/'frozen')`
  and added the metrics module `__file__` assertion at lines294–295. This correction
  must be included in the subsequently prepared frozen experiment.
- The metric pooling is appropriate to declared panels: COCO has fixed40 slots;
  PACO primary observed87 plus fixed303 appendix; LVIS first200 has ten folds with
  20 observed classes each. These remain partial exposed panels.

The parent completed the no-GT runtime smoke on the four predefined first dataset
indices0/100/200/400. `scene_model_validation/real_smoke4.json` records frozen
dependency paths, actual query-GT/archived-mask file-open guards, finite fields,
all12 mask geometries, absent encoder/calls0, CRF source hashes, and bitwise
prepared/unprepared CRF parity on one generated LVIS direct mask. These four
records are retained in the600 experiment. No smoke score was read.

`evidence_validation/audit_scene600.py` independently checks all600
after the full seal. It verifies every prediction/field/config/source SHA before
opening any query scoring mask, then recounts all six candidates in both frames
and checks class/fold aggregation, edits and parent ledgers. Separate mechanism
outputs use exact1024 foreground/background pixel mass per64 node: whole ROI,
FoRIS-predicted FG, and actual added/deleted ROIs for each direct/CRF output. All
three fields are compared on the same fixed ROI with exact-score ties. Reference
fit training losses are separate from query ranking and final mIoU. Five
no-image weighted-AUC/AP fixtures passed. The actual completed600 results follow.

## 8. Completed independent scene600 validation

The complete600 seal was verified before any new query annotation opened. The
independent audit then finished in **73.088 seconds**, with600 GT array identities,
1,800 field shape/finiteness checks,9,600 packed-mask geometry checks,2,400 baseline
I/U recounts,7,200 candidate I/U/edit identities,1,200 parent episode/frame ledger
identities and3,619 SHA checks. All ten reported aggregate dictionaries (eight
dataset/frame combinations plus two PACO303 appendices) match the parent within
7.1e-15. No inference, raw features, CRF replay or prediction-mask writing occurred.

The files are `scene600_verified.json`, `scene600_recomputed_episodes.jsonl`,
`scene600_mechanism_summary.json`, `scene600_mechanism_episodes.jsonl` and
`scene600_cli_confusion.json` in the independent evidence directory.

|CLI1024 primary, same inputs|FoRIS|MEAN|Scene direct|Scene CRF|Source kernel CRF|Support ridge CRF|
|---|---:|---:|---:|---:|---:|---:|
|Deep100|29.377542|26.465369|15.975180|15.986349|19.910111|31.545417|
|PACO100, observed87 strata|45.629834|46.824561|43.516621|44.100479|36.231942|39.036596|
|COCO200, official40 slots|63.170230|63.827315|40.898896|41.629344|46.446268|51.097512|
|LVIS200, observed20 classes in each of10 folds|46.679984|46.262735|27.123929|28.064457|31.451061|36.476455|

Original-frame scene CRF is15.986349/43.940819/41.194006/28.128532; original FoRIS
is29.377542/45.563915/63.358900/46.844567 in that order. The scene method fails
to outperform FoRIS on every tested panel. Its PACO gain against source kernel is
real (+7.868537 CLI points), but it loses that same matched control on the other
three. Support ridge improves Deep vs FoRIS by+2.167875 CLI points, while remaining
below the already-saved region.fast37.344028. None of the three heads is an
accepted cross-dataset improvement.

### Where scene reconstruction fails

All rankings below use saved signed64 fields and exact canonical1024 foreground/
background pixel mass in each encoder cell. They are post-seal diagnostics, not
the bilinear final masks, oracle outputs or independent holdout results. Macro
means average valid episodes; paired differences use the same ROI/population.

|Whole-query macro AUC|Scene|Source kernel|Support ridge|
|---|---:|---:|---:|
|Deep100|.877172|.897194|.924887|
|PACO100|.869047|.905883|.911053|
|COCO200|.874249|.938203|.952665|
|LVIS200|.888444|.939469|.962180|

|FoRIS-predicted FG macro AUC; valid episodes|Scene|Source kernel|Support ridge|
|---|---:|---:|---:|
|Deep;100|.619185|.660120|.684069|
|PACO;95|.740623|.738934|.766019|
|COCO;196|.596529|.708760|.713430|
|LVIS;195|.630336|.735800|.769826|

Scene loses whole-query ranking to both controls on all four panels. Its tiny
PACO existing-FG AUC advantage over source kernel (+.001689) is much smaller than
its output-score gain and does not exceed support ridge. Inside **the exact
regions added by scene CRF**, scene AUC is.677572/.547428/.571167/.578764, while
source-kernel scores rank those identical ROIs at.710120/.665133/.712991/.706390.
These added-region valid counts are100/87/182/174. The field is weaker exactly
where its new output incurs false positives. Different candidates' own added
ROIs must not be directly mixed as if they were the same ranking test.

|Mean reference binary-role training loss|Scene|Source kernel|Support ridge|
|---|---:|---:|---:|
|Deep100|.543396|.486976|.528700|
|PACO100|.195937|.117444|.176836|
|COCO200|.170067|.120634|.148666|
|LVIS200|.170340|.125004|.146591|

The scene basis has worse reference reconstruction as well as worse query
ordering. These are fit-subset training losses, with the same continuous-role
irreducible constant within an episode; low source-kernel training loss does not
certify transfer. This evidence rejects an explanation consisting only of a bad
zero threshold: nonlinear scene compression loses useful task discrimination.
It does not isolate which component, landmark, occupancy or normalization causes
that loss. No GT-guided change to these mechanisms was made.

### Decision extent explains the PACO control gain, with a clear limit

|CLI1024 predicted area / GT area|FoRIS|Scene CRF|Source kernel CRF|Support ridge CRF|
|---|---:|---:|---:|---:|
|Deep|1.0234|3.3845|1.1670|1.3299|
|PACO|1.2831|1.2757|.7777|.6439|
|COCO|1.0065|1.5405|.9856|1.0296|
|LVIS|1.1361|2.3226|.8610|.7778|

On PACO, scene retains11.168M true positives versus8.062M for source kernel, while
also producing9.361M versus4.452M false positives. Its larger decision extent
buys recall and better class-pooled mIoU despite weaker whole-query ordering.
It still has less TP and more FP than FoRIS in this pooled pixel diagnostic.
On Deep/COCO/LVIS, scene false positives are12.533M/21.578M/23.897M, compared with
FoRIS2.542M/6.233M/6.073M. Thus the apparent PACO benefit is a real but specific
allocation tradeoff; it is not robust generalization or evidence that query-basis
clustering supplies uniformly better target identity.

Support ridge preserves the best whole-query and difficult-region ordering, yet
its PACO/LVIS area ratios show substantial under-recovery. Those facts distinguish
representation quality from the absolute signed readout; they do not authorize
choosing new thresholds from this query GT. The next research step should explain
the source-to-query mismatch and recover task identity evidence that survives the
actual difficult ROI, rather than assume another renderer or CRF iteration will
repair a weak field. Current direct-to-CRF changes are small relative to these
method deficits.
