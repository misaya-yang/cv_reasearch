# D: layer-transition correspondence through complete FoRIS

Status: the unchanged 20-episode complete-mask run has finished with an observed
positive but unresolved effect. Full DEV241 comparison code, with one new
delta-only simple control, is prepared. Earlier preparation
notes below retain their chronology; the measured update at the end supersedes
their former no-run status.
Owner: mechanism_multilayer. This is one independent complete mechanism, not
one component of another agent's method. Parameters below were fixed before
any query outcome was read. No model/data downloads or remote jobs were started.

## Evidence and gap

The imported demo9 README/HANDOFF and current failure ledger were read. Native
FoRIS has both missed targets and connected over-inclusion; its seed and region
scores can retain the same identity mistakes. This is a reason to inspect a
different representation, not evidence that the proposed representation works.

The existing layer-reading experiment already rejected raw single-layer
prototype and background-contrast constructions. In particular layer16's
single-prototype GT-best-cut was 41.03 points below its layer24 counterpart;
layer22 background contrast was 19.10 points below its layer24 counterpart
(`evidence/local/RESULTS.md`, layer-reading table). The layer12 reference
deletion head, reference ridge, QK and unbalanced-transport attempts must not
be rerun under this name. Supervised convctx:layers is a separate resource
setting and does not establish an unsupervised layer mechanism.

No observation yet establishes that layer transitions improve identity
ranking. The first real complete-mask comparison is necessary to decide that
premise. This proposal does not use those prior lower-layer diagnostics as a
positive result and does not claim that a difference of normalized activations
is a causal transformer residual.

## Fixed mechanism in ten lines

1. Use one frozen DINOv3 ViT-L/16 and the native transformed reference/query pair.
2. Tap blocks 16 and 24 during the same paired forward; both use encoder norm=True.
3. Normalize each token vector independently: early=e, late=l.
4. Main descriptor is concat(l, normalize(l-e))/sqrt(2).
5. The proposed new evidence is how an appearance changes across depth.
6. It can change token correspondence and query clustering, hence both additions and deletions.
7. Obtain this descriptor's rank-500 positional basis from the normalized black image.
8. Run unchanged source FoRIS parts 1, 2, 3 and 4 on these descriptors.
9. Run the same native min-max binarization and CRF to a full 1024 mask.
10. Use no query GT, fitted head, class name, extra natural image, candidate voting or RCG gate.

The hypothesized error is a false object that resembles the reference at the
last layer but follows a different depth transition, or a true part whose final
appearance differs while its transition agrees. These are hypotheses, not
observed categories of corrected errors.

## Controls and attribution

- Same-input simple control: concat(l,e)/sqrt(2), its own normalized-black
  rank-500 basis, exactly the same FoRIS parts and CRF. This tests whether the
  transition operation adds anything beyond access to two layers.
- Native control: layer24 from the identical paired extraction, original native
  positional basis and all source FoRIS stages. The first real pair also reruns
  the ordinary n=1 extractor and requires exact token equality.
- The shared runner must compare the adapter's native 1024 mask with the public
  native entry. Scoring must verify cached native identity before importing old
  RCG as a same-protocol control. A mismatch is not silently accepted.
- Existing RCG and same-cohort native are required in the final analysis. Their
  absence does not become a zero or an assumed result.

Main/control each have 2048 descriptor channels, one identical forward and one
SVD. Both share the same two real layer tensors and black forward. Thus a main
gain over native alone cannot identify a mechanism contribution; main must also
beat the concat control. The native representation remains 1024-dimensional,
which is an explicit readout-compute difference, not free improvement.

## Implementation and integration

- `src/ics/methods/multilayer.py`: predict/control/native, exact stage adapter,
  source CRF finalizer and run_episode(host, episode, manifest, bases).
- `src/ics/methods/layer_extract.py`: extract_pair and prepare_bases. Neither
  constructs an encoder nor starts a job or downloads assets.
- C owns the shared forward runner integration. A single host is used
  sequentially; C's attention intervention must be outside D's extraction.

The shared runner calls prepare_bases once and retains its receipt. It then
calls run_episode for each manifest row, setting verify_native=True only for
the first real episode. Output mask keys are native, multilayer and
multilayer.control. It saves packed prediction files, a clean manifest,
per-episode metadata and all-cohort SHA sealing. Query annotations are only
opened later by the shared independent scorer. D's run_episode only opens the
reference annotation.

Expected feature cost is one paired reference/query forward per episode; one
additional paired forward on the first episode checks native extraction
identity. Preparation uses one global normalized-black forward and two SVDs.
No extra natural image is used. Timings include device-to-CPU synchronization;
feature extraction, each readout, finalization and SVD costs are reported
separately. Actual timing, peak memory and GPU/CPU split remain unmeasured.

All 220 exposed episodes remain development data. Use the actual shared
manifest after duplicate/shared-photo checks; do not select episodes from their
scores. Parameters remain fixed. The scorer must report class-summed mIoU,
paired gain, 2000 connected-photo bootstrap draws with RandomState(0), folds,
batches, up/down counts and all four pixel actions. A small real smoke checks
execution only. No gain, confidence interval or efficacy claim exists yet.

## Cross-agent challenge and failure localization

B's full graph correspondence works on final-layer relationships; D does not
use its Aq T Ar operator. C modifies attention before later model blocks; D
does not modify the forward computation. D owns the depth-transition evidence
and the resource-matched concat comparison.

The key challenge is that l-e may amplify nonsemantic depth changes or geometry
instead of separating identity. A source reconstruction score cannot resolve
that. If native replay fails, fix provenance/adapter before evaluating D. If
concat improves while transition worsens, the failed link is the transition
operator; do not rename it or sweep its weight. If both worsen, inspect which
FoRIS stage destroys discrimination and the four final pixel actions before
concluding anything about all intermediate information. Query-GT diagnostics
must remain separately marked and cannot select an episode-specific arm.

## Verification receipt and limits

Before the user's later instruction to stop CPU testing, synthetic descriptor
shape/unit-norm checks passed for 64x32 and 4096x1024 inputs, and the two modules
compiled. These checks are not real-data or FoRIS validation. An attempted
actual FoRIS import stopped at missing einops; no real backend was executed.
After the stop instruction, only code preparation/static inspection continued.

The discovered local DEV241 sample explicitly contains only q/r/debiased final
features, without images, weights or intermediate layers. Those were not
substituted for real layer inputs. No multilayer predictions, scores, timing or
new scientific result has been produced; GPU availability and real integration
remain unverified for this arm.

Static follow-up found and repaired a native-replay state dependency:
extract_pair clears host._tgt_image, but FoRIS's seed-cluster prior reads it to
include RGB and xy features. The readout now requires the saved query_tensor,
restores its native 3xHxW form during all four stages, and restores the prior
_tgt_image, positional_basis and should_debiass in a finally scope. It validates
shape/device instead of silently omitting RGB. This repair was inspected
statically only; no test was run after the CPU-stop instruction. Native replay
parity remains a required runtime check, not a substitute for restoring state.

## First real complete-mask result: four-episode execution smoke

Source: `runs/outputs/gpu_smoke_v1/forward/report.json`, `episode_metrics.json`
and `audits.json`. COCO-20i one reference, seed-0 source episodes, frozen DINOv3,
complete 1024 masks after native CRF. Four episodes/classes/photo groups, one
episode from each fold and each historical batch. These are exposed development
examples, not confirmation. Intervals use 2000 connected-photo bootstrap draws,
RandomState(0). Query labels were opened by the separate scorer after sealing.

| Arm/comparison | Class-summed mIoU or paired gain | Paired 95% CI | Up/down/tie |
|---|---:|---|---|
| Native | 66.0181 | — | — |
| RCG | 67.0743 | — | — |
| D transition | 76.8955 | — | — |
| Same-input concat | 54.8518 | — | — |
| D minus native | +10.8774 | [-0.6173, 22.3721] | 3/1/0 |
| D minus RCG | +9.8212 | [-0.8522, 20.4946] | 3/1/0 |
| D minus concat | +22.0437 | [1.1200, 42.9673] | 4/0/0 |

Native replay mismatched pixels are zero against the cached masks. The shared
runner's per-episode D-native/public-native checks completed, and the first
pair's tapped layer24 versus ordinary n=1 extraction has maximum absolute
difference 0.0. This establishes actual complete-pipeline execution fidelity
on these four cases. It does not establish efficacy or general replay identity.

### Final-mask changes, including the failure

| Key; fold; batch | Native → D IoU | Gain, points | Add TP | Delete FP | Delete TP | Add FP |
|---|---|---:|---:|---:|---:|---:|
| 0_11_0; 0; old20 | 88.7375 → 89.3189 | +0.5813 | 8,082 | 560 | 1,350 | 4,628 |
| 1_38_5; 1; new40 | 89.4957 → 89.6997 | +0.2041 | 1,127 | 556 | 187 | 976 |
| 2_21_78; 2; new60 | 34.9361 → 79.0990 | +44.1628 | 652 | 29,922 | 172 | 711 |
| 3_0_75; 3; arrived100_exposed | 50.9031 → 49.4644 | -1.4387 | 750 | 9,050 | 967 | 20,706 |
| All four, pixel counts only | — | — | 10,611 | 40,088 | 2,676 | 27,021 |

Observed: the largest improvement preserves the native true-positive core
(intersection 18,658 → 19,138) while removing substantial false-positive mass
(union 53,406 → 24,195). The other two positive episodes mainly add true target
pixels with some false-positive expansion. The fourth episode is a real failure:
it adds 20,706 false pixels while removing 9,050, and loses slightly more true
pixels than it adds. Its union increases from 415,825 to 427,481. Do not hide
this batch regression behind the pooled score.

The +44.1628 episode supplies more than the total pooled four-episode gain;
the other three episodes average approximately -0.218 points. About 75% of all
deleted FP pixels come from that single improved episode, and about 77% of new
FP pixels come from the failed episode. This is a concentrated correction
signal with a visible expansion failure, not stable +2 evidence. With one
episode per batch, fold and batch behavior are completely confounded here.

Interpretation: D can produce useful bidirectional corrections through a full
pipeline, rather than merely exposing a better GT oracle. The final outputs
support a successful false-positive deletion in one case and modest target
completion in two; they do not yet show whether the successful deletion is an
identity correction, a connected extent repair, or a changed global threshold.
The current artifacts do not isolate which FoRIS stage causes it. No claim
that layer trajectories contain a newly identified semantic variable follows.

### Is concat a fair and sufficient control?

Concat is a fair matched-resource representation control: identical real
layers, images, one paired forward, 2048 output channels, rank-500 black-image
basis construction, fixed FoRIS stages and CRF. Its poor result is not explained
by withholding a layer or giving it less computation. The main's four-case
advantage over it is real for the saved outputs.

It is not established as the strongest simple use of these inputs. Equal
weight on the weak early representation is a deliberately simple fixed rule,
and the historical layer-reading failure makes its fragility unsurprising.
An immediate same-information action control is native intersection D
(delete-only), with native union D as its add-only companion. They can be
scored from the already frozen complete 1024 masks with no new forward, and
must be applied globally rather than selected per episode. The observed
four-case actions suggest that deleting D's additions could repair much of
the fourth case while preserving the large third-case gain; the full method
therefore has not yet beaten this strong simple alternative. No exact score
for this control has been computed here. It is a post-hoc action ablation,
not an implicitly promoted gated replacement for D.

The most useful next simple control, if D merits larger evaluation, is complete
FoRIS on **normalize(l24-l16) alone**, with the same two extracted layers and
its corresponding native-style black basis. If that wins equally, the added
late-layer branch is unnecessary. This is a proposed subsequent attribution
control, not a modification of the currently running 20-episode arms, and it
has no measured score. A small reference-only-selected mixture is another
possible stronger use of the inputs, but should not become an unbounded sweep.

The difference and concat descriptors are deterministic reparameterizations
of the same two layers. D's present claim, at most, is a useful depth-contrast
metric for matching and clustering, not the creation of new information.
Moreover, the raw descriptor change also changes the black-image positional
subspace. Matching the basis-building procedure is proper for the complete
method comparison, but it does not isolate trajectory semantics from altered
positional suppression. A successful mechanism claim will need to distinguish
those explanations instead of attributing all native gain to semantics.

### Measured cost and remaining resource limits

One shared black forward and two SVDs took 1.7145 s total; transition SVD 0.8576 s,
concat SVD 0.5539 s. The three ordinary shared pair extractions took 0.5034–0.5108 s.
The first extraction took 0.9771 s because it also ran the extra native-tap
identity forward. D's complete readout plus finalizer took 0.9822–1.0429 s per
episode; its observed ordinary combined extraction/readout cost was roughly
1.486–1.549 s, versus warmed public-native totals 1.428–1.513 s in this smoke.
These are four-case wall timings in the actual sequential runner, not a
throughput benchmark or an amortized claim for a different execution layout.

The D extraction peak allocation was about 1.83 GB (decimal). The runner did
not record a separately reset per-D-readout peak, so full D peak memory is
unverified. The complete comparison suite incurs extra native and control
passes; do not report a candidate's one-forward cost as the whole suite cost.
No extra natural images or model instances were used.

### Decision before the unchanged 20-episode readout

Continue the already-started fixed 20-episode test. Do not change layer choice,
weights, positional rank, thresholds, FoRIS settings or episode selection in
response to these four outcomes. No scientific success verdict is supported.

After the 20 results, inspect paired native/RCG/concat gains together with
all four actions and each batch/fold. If the four smoke episodes are included,
report the remaining 16 separately as an **incremental DEV sensitivity**, not
fresh confirmation. Also report the fixed-cohort result without 2_21_78 as an
explicit post-hoc concentration diagnostic; it cannot replace the primary
20-episode estimate or be used to choose the method per episode.

Expansion to the available full DEV cohort is worthwhile if the fixed method
retains a practically relevant positive complete-mask effect, or a positive
but uncertain effect with multiple real corrected cases, and is not carried
by one extreme case while a batch shows repeated substantial damage. A CI
crossing zero alone is not rejection. Conversely, even a positive pooled score
does not justify calling the method stable if an exposed batch repeatedly
shows the expansion failure above. In that case, first locate whether score
ordering, clustering or native finalization causes the damage; changing that
mechanism would create a new declared arm, not a renamed parameter sweep.

The eventual target remains at least +2 over complete native with uncertainty,
strong same-input controls and no hidden batch regression, evaluated on all
available exposed DEV before any genuinely untouched confirmation is named.

## Unchanged first20 result and complete-DEV preparation

Source: `runs/outputs/gpu_first20_v1/forward/report.json`. Twenty exposed DEV
episodes, 18 classes, 20 connected-photo groups, five episodes per fold;
1024 complete masks, same frozen recipes, 2000 RandomState(0) paired draws.

| Comparison | mIoU / paired gain | 95% CI | Up/down/tie |
|---|---:|---|---|
| Native | 55.0962 | — | — |
| RCG | 55.6279 | — | — |
| D | 58.3560 | — | — |
| Concat | 53.7011 | — | — |
| D minus native | +3.2599 | [-0.1415, 5.7074] | 12/7/1 |
| D minus RCG | +2.7281 | [-0.5122, 5.5139] | 12/7/1 |
| D minus concat | +4.6550 | [-1.5925, 9.5449] | 11/8/1 |

Fold gains: -0.1543 / -0.2920 / +12.5680 / +2.0966. Batch gains:
old20 +0.3834 (n=4), new40 -1.1421 (n=4), new60 +14.6700 (n=4),
arrived100_exposed +0.0919 (n=8). Pixel actions: add TP 30,923; delete FP
153,223; delete TP 26,726; add FP 91,018. Net TP rises by 4,197 and net FP
falls by 62,205, but those aggregate pixel counts are not the class-mIoU
estimand and do not establish uniform behavior across classes.

Decision: retain the primary recipe unchanged for complete DEV. The effect is
large enough to merit a better-powered comparison, and 12 improved episodes
support broader checking beyond the original one-case smoke gain. The strong
concentration in new60 and the new40 regression remain unresolved; none of
the three paired intervals establishes superiority. This is not independent
confirmation or a claim that the +2 stability target is achieved.

New prepared files, without changes to the v1 mechanism:

- `src/ics/methods/multilayer_controls.py` adds only delta-only: normalize(
  normalize(layer24) - normalize(layer16)), with rank-500 normalized-black
  basis and the existing complete FoRIS readout/finalizer. V1 basis preparation
  is reused; its black forward is observed without changing returned values,
  so the three bases share exactly one black-image extraction and three SVDs.
  The saved prediction key is `multilayer.delta.control` so the shared scorer
  automatically includes it as a paired strong control. Its semantic name
  remains delta-only.
- `scripts/run_multilayer_complete.py` runs one shared encoder sequentially.
  Each ordinary episode has exactly one paired extraction for native, D,
  concat and delta-only. The first episode alone has two additional paired
  audit forwards: independent public-native and ordinary n=1 layer24. Public
  native must match tapped-native mask exactly and score to the recorded
  tolerance. All episodes must subsequently match cached native in the
  independent scorer before any same-protocol comparison is reported.
- The runner also saves globally fixed native-intersection-D and native-union-D
  action diagnostics. These never select or replace the primary output.
- `--save-layers` optionally saves q16/q24/r16/r24 as uncompressed raw float32
  arrays, about 64 MiB per episode or 15.1 GiB for 241. Available disk must
  cover the whole archive plus the output reserve before it starts. Layer
  files and the three CPU bases are hashed and included in the final receipt.
- Protocol/source hashes, clean manifest, all per-arm masks, raw score packets
  and audits are preserved. The runner never reads query annotations. Only
  after all predictions are sealed does the existing score_forward_run.py
  verify cached-native identity and compare the sealed RCG run.

The new delta-only arm and dedicated runner have been statically prepared but
not executed by this agent. No result for the delta-only or action controls is
claimed here. The main agent owns the resource guard and remote launch.
