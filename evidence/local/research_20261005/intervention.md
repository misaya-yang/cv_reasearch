# C: bounded reference-conditioned attention routing

Status: implementation and local tensor-contract checks complete; **no real-image
DINOv3/FoRIS run, native-mask result, effect estimate or measured GPU cost yet**.
SSH was initially unavailable; the parent subsequently recovered access and reported no GPU, a 0.5-CPU quota and 2 GiB RAM. The user then stopped CPU tests and asked for code preparation before GPU activation. No further execution tests were run after that correction.
No remote computation, download, rental, model instantiation or scientific result
is claimed by this note.

## Existing gap and non-repetition

The imported demo9 README records 83 failed DEV241 episodes: 49 were correctly
located but had poor extent, and 34 had wrong localization. It also records that
query-native prototypes and source-fitted reference classifiers did not reliably
repair identity ordering. These are historical diagnostic observations, not new
measurements. The former global-content interaction rewrite was negative
(29.49 versus 42.68, -13.19 points), and nine QK descriptor arms were negative
(58.70 versus 65.16, -6.46 points, old40). Those failures directly challenge this
construction; they cannot be dismissed as implementation details.

This construction tests a narrower, previously unmeasured link: can a bounded
reference-conditioned perturbation of actual **intermediate attention routing**
change the final representation usefully after the remaining frozen nonlinear
blocks? It does not replace attention with reference features, read QK as a final
descriptor, crop an image, mask image background, optimize a prompt, filter a
candidate bank, merge masks, or gate RCG. It leaves original Q/K/V, rotary
positions, residual streams, image pixels and pretrained weights intact.

## Complete mechanism in ten lines

1. Encode the actual reference and query together, in their original FoRIS order.
2. At block `L-6` (index 18 for a 24-block encoder), observe attention input tokens.
3. Area-resize the supplied complete reference mask to the current patch grid.
4. Compute role-balanced reference FG and BG cosine-kernel log densities at query tokens.
5. Use temperature 0.1 and signed soft role `s = tanh((log p_F - log p_B)/2)`.
6. Add `log(2) * s_i * s_j` to existing query-patch attention logits only.
7. Do not bias reference attention or prefix-token rows/columns directly.
8. Run the untouched attention values, residual/MLP and remaining frozen blocks.
9. Run the complete original FoRIS score/binarization/CRF using those features.
10. Return one final 1024 mask; no query GT enters any inference decision.

`L-6`, temperature 0.1 and maximum bias `log(2)` are fixed before real-image
results. They are not selected per episode or swept on exposed labels. Positive
role compatibility can preserve interaction among missed parts, while negative
compatibility can reduce foreground/background content mixing. **Neither claim
is yet observed.** If role assignments already confuse a coherent distractor,
this routing can strengthen that error. A bounded update limits perturbation;
it does not establish correct roles.

## Strong simple control and fair resource accounting

The key-only control reuses the identical intermediate reference/query tokens,
coverage, KDE role calculation, layer, one paired model forward and final FoRIS
pipeline. Its bias is `log(2) * rms(s) * s_j`, repeated over query rows. It favors
reference-like keys without conditional token-pair routing. Its matrix RMS is
exactly the candidate matrix RMS, and both remain bounded by `log(2)`. If this
simple prior matches or beats C, there is no demonstrated value in conditional
pair routing. Both controls are part of the first comparison, not late ablations.

Native FoRIS is rerun on the identical manifest. The first episode additionally
runs both an unmasked identity interception and a `masked_identity` forward
with an all-zero additive attention mask of exactly the candidate shape. Both
require an identical final native mask, score agreement within `atol=rtol=1e-5`,
and raw reference feature agreement at that tolerance. Raw reference exactness
and maximum deviation are also recorded: the reference receives no direct bias,
but the additive-mask SDPA kernel may change its numerical output. Candidate
and control reference features are compared against masked identity too. Any
failed required check writes `audit_failure.json` with
`SDPA_KERNEL_OR_REPLAY_CONFOUND` and prevents a completed prediction seal. This
audit is extra validation cost, not part of a deployed method. The first episode
uses two extra C audit forwards, plus one additional D tap-verification forward
when D is enabled; ordinary C-only runs use three paired forwards per episode
(native/control/candidate), and C+D uses four. The first episode uses five or
seven respectively. D also has one setup black-image forward. These planned
counts are explicit in `protocol.json`; actual arm timing is separately logged. Each deployable arm requires exactly one
paired encoder forward. The candidate and control add cosine-KDE GEMMs and one
additive attention mask, but no model forward or external input. Masking can
change SDPA kernel selection; wall time and peak GPU allocation are recorded
rather than assumed equal to native. The maximum additional additive-mask tensor
is roughly 128 MiB at FP32 for a two-image, 4101-token batch (one head broadcast),
plus query-reference chunk products and candidate bias. This is an allocation
estimate, not peak runtime memory. Actual configuration and one-episode timing
must precede cohort scheduling.

## Independent-agent challenge

Agent D noted that erroneous same-role labels would reinforce the same false
object; boundedness alone cannot explain correction. C therefore needs scoring
of additions and deletions, and any later role-quality analysis must be explicitly
GT diagnostic. D reads intermediate/final feature transitions without modifying
attention; the mechanisms do not overlap. Agent B works on global second-order
soft correspondence using final caches; C neither uses nor adds that graph.

The decisive test is complete native/control/C masks, with RCG joined on exactly
the same eligible episode IDs by the common evaluator. A first real episode can
validate execution and time; it cannot establish efficacy. The exposed old120 and
new100 are development data. Native stability and per-batch regressions remain
required even if a pooled score increases.

## Reproduction and observed validation

Implementation: `src/ics/methods/intervention.py`.
Complete RGB entrypoint: `scripts/run_intervention.py`.

```
python scripts/run_intervention.py --manifest EXISTING_MANIFEST.json \
  --foris-root EXISTING_FORIS_CHECKOUT --out FRESH_OUTPUT_DIRECTORY --limit 1
```

Remove `--limit 1` only after the real smoke/timing passes and compute is
available. The runner uses the existing `build_host` configuration, one shared
`TimmDINOv3` instance, public `run_foris`, existing RGB/masks/weights/native basis,
and no downloads. Add `--include-multilayer` to prepare D's black-image bases once and run D sequentially with the same host. Its extracted native masks must match the public native mask on every episode. It verifies actual paired batch order against live host pixels
before patch embedding, requires fused SDPA and a square grid, and fails loudly
for incompatible implementations. It does not silently switch to separate-image
encoding. All patched methods and SDPA functions are restored after exceptions.
The scoped SDPA adapter is single-threaded; independent model forwards must not
run concurrently in its process. CPU evaluation may run in another process.

Outputs: complete packed 1024 masks in `predictions/` under canonical keys
`native`, `intervention`, `intervention.control`, and optional `multilayer` and
`multilayer.control`; float score fields and audit masks in `packets/`; exact
configuration, per-arm times and GPU allocation in `protocol.json`, `episodes.jsonl`
and `audits.json`. The **entire cohort** is frozen with `manifest.json` and a
SHA-256 `sealed.json` before this inference process exits. It never opens query
annotations. A separate process runs `scripts/score_forward_run.py --out RUN
--cache-root ROOT` (optionally joining the parent's RCG run) after model release.
That shared evaluator owns class-summed mIoU, paired differences, photo-connected
RandomState(0) bootstrap with 2000 draws, all folds/batches, up/down counts and the
four correction actions. A cached-native mismatch must prevent a same-protocol
claim; missing RCG must be reported rather than silently omitted.

Observed local validation on 2026-10-05, Python 3.9 / torch 2.8.0:

- Both files pass Python bytecode compilation.
- A deterministic tiny SDPA fixture gives exact identity replay.
- Reference outputs remain exact; query outputs change for candidate routing.
- Candidate and key-only bias RMS agree to numerical precision.
- Unpaired inputs fail and original methods/global SDPA are restored.

The subsequently added masked-identity GPU-kernel audit has **not been run**;
the earlier CPU identity fixture does not validate the GPU additive-mask path.
These are code contracts on synthetic tensors, **not** a real DINOv3 forward,
FoRIS reproduction, semantic role-quality check, or complete segmentation result.
Timm is absent locally. The parent is checking installed server code statically; actual real model execution remains unverified. Therefore this arm has no numeric research result and no eligibility
for a +2-point claim.

## Failure ledger and next decision

| Stage | Observation | Decision |
|---|---|---|
| Python/tensor contract | Passed the checks above | Adapter is ready for the first actual paired call. |
| Real assets and server API | Not verified by this agent; remote access owned by parent | Do not call preparation an executed experiment. |
| Real attention/complete mask | Not run | First execute native/identity/control/C on one real episode and measure cost. |
| Cohort efficacy and uncertainty | Not run | No selection, rejection, claim or interval may be inferred. |

If identity fails, repair the contract before studying masks. If the complete
candidate loses to equal-RMS key-only, conditional routing has not paid for its
interaction structure. If both damage masks, locate whether roles were wrong or
whether correct-role routing altered final features harmfully; that observation
must precede any different intervention. Do not respond by sweeping the same
bias strength/layer and renaming the construction.

## Actual GPU smoke4 readout: limited action, not a kernel failure

This update supersedes the earlier execution-status statements while preserving
the preparation history. The parent ran the frozen configuration on four real
COCO-20i development episodes, one per fold and one per recorded batch, using the
same frozen DINOv3, complete FoRIS/CRF and 1024 output. Source files:
`runs/outputs/gpu_smoke_v1/forward/{report,audits,episode_metrics}.json`.
Seed is inherited from the existing seed-0 episode manifest; the bootstrap uses
RandomState(0), 2000 connected-photo-group draws. There are four classes and four
photo groups. These are exposed development episodes, not independent evidence.

Observed complete-mask class-summed mIoU: native **66.0181**, C **66.2617**,
key-only control **66.2666**, RCG **67.0743**. C-native is **+0.2436
[-0.0119, +0.4991]**; C-control is **-0.0049 [-0.0225, +0.0128]**;
C-RCG is **-0.8126 [-3.4652, +1.8400]**. C has two increases and two decreases
against native and against its simple control. Four episodes cannot estimate
stable efficacy or justify rejecting intermediate intervention in general.

Actual timm/Eva SDPA interception succeeded. On the first real episode, both
identity and masked identity reproduced native scores and reference features
exactly (all recorded maximum absolute errors zero), and final masks were exact.
C/control reference features also exactly matched masked identity. The complete
native replay matched cached native with zero changed pixels. Thus the present
small C-control difference is not explained by the checked mask-kernel confound.
The complete C forward+FoRIS+CRF took 1.44--1.61 seconds per episode and about
1.98--2.01 billion allocated GPU bytes. These are the recorded four-episode
measurements, not a throughput guarantee or device-total memory use.

| Episode / fold / batch | Native IoU | C gain | Control gain | Role mean / RMS | C bias RMS | C changed pixels |
|---|---:|---:|---:|---|---:|---:|
| `0_11_0` / 0 / old20 | 88.7375 | -0.0192 | -0.0353 | -0.00488 / 0.20838 | 0.03010 | 690 |
| `1_38_5` / 1 / new40 | 89.4957 | +0.0088 | -0.0007 | -0.16827 / 0.38659 | 0.10359 | 86 |
| `2_21_78` / 2 / new60 | 34.9361 | -0.0046 | +0.0079 | -0.17094 / 0.26580 | 0.04897 | 7 |
| `3_0_75` / 3 / arrived100_exposed | 50.9031 | +0.9894 | +1.0220 | -0.02114 / 0.33501 | 0.07779 | 9018 |

The action counts localize what happened more precisely than the pooled score:

| Complete arm | Recover true | Remove false | Delete true | Add false |
|---|---:|---:|---:|---:|
| C | 639 | 8068 | 469 | 625 |
| Equal-RMS key-only control | 444 | 8070 | 446 | 241 |

The common improvement is almost entirely removal of approximately 8040 false
pixels in `3_0_75`. C recovers 195 more true pixels in aggregate than control,
but adds 384 more false pixels and deletes 23 more true pixels; these are
**differences of arm-versus-native action counts**, not direct C-versus-control
pixel transitions. In the difficult `2_21_78` episode neither arm recovers any
true pixels; C adds seven false pixels and control removes twelve false pixels.
This is evidence of inadequate corrective reach on that actual hard case,
not evidence that no useful frozen-model mechanism exists.

### What is established about the failure link

1. **The input roles did not collapse to a constant.** The observed within-image
   role standard deviations, derived from the recorded means and RMS values,
   are 0.2083, 0.3481, 0.2035 and 0.3343. Candidate and control bias RMS match on
   each episode, while candidate maximum absolute biases range 0.205--0.359.
2. **The candidate and control attention biases are not numerically the same.**
   Writing `B = gamma s s^T` and `K = gamma rms(s) 1 s^T`, their Frobenius cosine
   equals `mean(s)/rms(s)`: -0.0234, -0.4353, -0.6431 and -0.0631 on these four
   images. A near-equal final score is therefore not sufficient evidence that
   the intervention failed to execute or that the two input operators coincide.
3. **The mechanism has only one key-ranking direction.** For every positive
   query role, C ranks keys in the same order as the key-only control; it changes
   only the row amplitude. Negative query roles reverse that same ordering, and
   uncertain roles approach no intervention. This is an algebraic limitation,
   independent of the four labels. A missed true region assigned near-zero or
   negative reference role cannot receive a new part-specific positive message
   from this bias alone. Whether the observed missed pixels have those roles
   has **not** been measured: summary role moments do not expose their locations.
4. **Where the distinct operator becomes a nearly identical mask is unresolved.**
   The locally available exports currently contain JSON reports and audits,
   not stored score-field packets, per-token roles or intermediate attention
   outputs. They establish tiny final actions, but cannot distinguish small
   attention-value changes, attenuation in remaining blocks, FoRIS score
   normalization/clustering, or CRF erasure. Calling any one of those the proved
   cause would exceed the evidence.

### Locked next decision after the already running 20 episodes

Do not alter the running code, layer, temperature, bias strength or episode
selection. The first20 result must first show whether the same action pattern
persists across batches, and whether C-control differences concentrate on hard
cases or remain almost entirely the same deletion action. No significance-only
accept/reject rule follows from a small interval.

First use **already saved packets**, without new model inference, to compare
C/control/native score fields, their rank changes, native-binarizer outputs and
final CRF masks. This distinguishes a score-to-mask bottleneck from an upstream
lack of meaningful change. Keep four-action accounting at both stages; a useful
pre-CRF recovery erased by CRF is a different failed link from no recovery ever
appearing. Score min/max or affine changes alone are not semantic reordering.
If raw fields differ mostly affinely, normalized fields must be compared before
claiming attenuation.

Only if first20 plus those stage readings still indicate an upstream binary-role
bottleneck, obtain one fixed diagnostic capture on that same exposed cohort:
actual signed roles and attention-weighted value-message changes at the existing
intervention block. Query GT is used **after capture** to report TP/FP/FN/TN role
bins and whether missed true regions have distinct reference correspondence
patterns despite near-zero/negative binary roles. This adds diagnostic forward
cost and must be scheduled by the parent; it has not been launched here. Global
bias RMS is not an adequate substitute for attention-weighted message change.

A substantive subsequent mechanism change would replace the single binary
key-ranking direction with **reference-part relation messages inside the frozen
attention block**, while retaining native residual content and the complete
FoRIS/CRF output. It is warranted only if the diagnostic demonstrates usable
reference correspondence structure on actual missed regions that the scalar
role discards. The simplest same-input controls must include the existing
key-only prior and a query-appearance relation update of equal RMS/rank/compute;
otherwise a higher-rank or larger perturbation could be the entire gain. This
would be a changed relational inference, not a layer/strength sweep. Agent B
already studies correspondence relations in cached feature space, so the parent
must explicitly resolve that overlap before creating another graph variant.
If the diagnostic does not support such additional structure, do not implement
this proposal merely to keep C alive: use the resolved bottleneck to choose a
different intervention, while retaining native as the complete baseline.

No code was changed and no remote computation was started for this analysis.

## First20: the corrective signal is already sparse before CRF

Source: `runs/outputs/gpu_first20_v1/forward/{report,audits,episode_metrics}.json`
and all 20 `packets/*.npz`. This is the existing exposed seed-0 development
cohort: 20 episodes, 18 classes, 20 connected-photo groups, five episodes per
fold. Complete 1024 native is **55.0962**, C **55.1741**, key-only **55.1765**.
C-native is **+0.0779 [-0.0267, +0.1409]**, with 9 up / 10 down / 1 tie;
C-control is **-0.0024 [-0.0287, +0.0255]**, with 8 up / 10 down / 2 ties.
Intervals are paired 2000-draw RandomState(0) photo-group bootstrap. Fold gains
are +0.0280 / +0.0171 / -0.0069 / +0.2464. Batch gains are old20 -0.0182,
new40 -0.1251, new60 +0.0767, arrived100_exposed +0.1859. There is no substantive
mechanism increment in this tested construction; this does not exhaust all
frozen-model intervention operations.

### Measured score-to-mask chain, with no new forward or query-label access

For each stored arm score `s`, compute `u=(s-min(s))/max(max(s)-min(s),1e-6)`;
render `u` with the shared `ics.experiment.render` (bilinear 64→1024,
`align_corners=False`, cut strictly above 0.5). Compare those reconstructed
pre-CRF masks with the stored complete packed masks. This is the common
experiment renderer, not a new threshold. CPU analysis used only saved fields
and masks; no query GT was available or opened.

| Comparison | Median raw-score RMS difference | Median normalized-field RMS | Median Spearman | Total pre-mask differences | Total final-mask differences |
|---|---:|---:|---:|---:|---:|
| Key-only minus native | 0.006672 | 0.005389 | 0.999460 | 22,827 | 18,260 |
| C minus native | 0.006977 | 0.007893 | 0.999457 | 29,046 | 21,470 |
| C minus key-only | 0.007805 | 0.009235 | 0.999028 | 17,153 | 11,180 |

C changes only **0.1385%** of all 20 full masks before CRF and **0.1024%** after
CRF. Its median episode changes 639 pre pixels and 438.5 final pixels. Against
native, the median within-episode absolute normalized-score change is 0.000283;
the median 90th percentile is 0.001279. Only a median 0.708% of tokens change by
more than 0.01, and 0.134% by more than 0.05. Hence the RMS above includes sparse
larger changes; it does not imply widespread class-relevant reordering.

This is **not primarily min-max normalization canceling a large affine change**:
fit each raw score pair by least squares `s_C ≈ a*s_base+b`; the fitted affine
component explains only a median 0.341% of raw C-native squared difference,
and 0.405% of C-control difference. Most raw difference is non-affine, but sparse.
The diagnostic does not establish that those few changed tokens have correct
semantic identity.

CRF reduces total C-native differences by 26.1% and C-control differences by
34.8%; it also relocates boundaries. For C-control, only 1456 pre-difference
pixels survive in the same direction at the same location, three overlap in the
opposite direction, 15,694 are pre-only and 9721 are final-only. Therefore low
pixelwise survival is **not** proof that CRF erased useful segmentation gains:
there are many newly located final differences, and no pre-CRF GT measurement.
Two individual C-control cases expand after CRF (`1_0_73`: 51→365;
`1_21_65`: 1717→2276), so an unconditional 'CRF suppresses everything' account
is false. The main measured limitation is already tiny corrective reach at the
complete FoRIS score/pre-mask stage; CRF adds a smaller aggregate contraction.
The available fields do not separate attention output, remaining encoder blocks
and FoRIS parts 1–4, so the exact upstream attenuation step remains unobserved.

The difficult `2_21_78` confirms the upstream limitation directly: C-native
normalized RMS is 0.000130, maximum difference 0.000787, and only 17 pre pixels
change, becoming seven final pixels. This is not a large new mask destroyed by
CRF. The stored GT-based final action report says all seven are new false pixels;
there is no basis for inventing a pre-CRF TP/FN breakdown.

Across all 20 final masks, C recovers 3518 true pixels, removes 9502 false pixels,
deletes 2332 true pixels and adds 6118 false pixels; key-only gives
1506 / 11,089 / 2275 / 3390. The C/control masks share 14,275 changes relative to
native; their final direct difference is 11,180 pixels. Additional C actions have
not yielded a net mIoU improvement over the simple prior. These final action
counts come from the existing scoring report; they are not new label analysis.

### A different next operation, not a stronger version of the same rank-one bias

Retire this fixed **binary-role query-key rerouting construction** from further
layer/strength grids. The algebra and first20 evidence together favor changing
what message can enter a query token, rather than multiplying the same key order
more strongly. One candidate is **contrastive reference-value residual injection**:
keep the native attention output and residual path intact; at the same fixed
block, retrieve query-specific means of the existing reference foreground and
background *value vectors*, subtract them, and add that bounded contrastive
message as a separate residual before the remaining frozen blocks. Final
segmentation still uses every native FoRIS stage and CRF. Zero residual must
reproduce native exactly. This preserves the complete native computation path;
it does not promise to preserve its accuracy under nonzero intervention.

Unlike `s_i*s_j`, the added vector need not disappear when a missed token's scalar
FG/BG density difference is near zero, and different tokens can retrieve different
foreground-part value directions. The first same-input, same-forward control is
uniform FG-minus-BG mean-value injection with identical residual norm budget;
query-specific retrieval must beat it, not merely beat untouched native. A simple
foreground-only retrieval arm is needed if the claimed increment is attributed
to explicit negative evidence. All pair forwards, retrieval GEMMs, memory and
elapsed time must be reported; one loaded frozen model is sufficient.

This remains a **conditional proposal**, not an implemented or selected method.
Before implementation, recover the old negative global-content-rewrite source
and verify that it did not already test this same residual operation; preserving
a native branch is a concrete difference only if the old construction replaced
or otherwise altered that branch. Also check actual attended-message deltas on
the exposed 20 to distinguish negligible value movement from later attenuation.
The current packets do not contain those values. If the old operation is the
same, do not rename it and rerun; if reference-conditioned value directions have
no discriminative transfer, do not justify another grid by the existence of a
high GT oracle. No running method file was changed and no model forward or remote
process was started in this diagnostic step.
