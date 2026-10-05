# E: reference-constrained latent appearance slots

Status: implementation and one real no-label timing completed; efficacy evaluation pending.
After that timing and synthetic contract checks, the user stopped CPU testing and requested
code preparation only until the GPU is enabled. No 20-example run or label evaluation followed.
The fixed construction is a candidate, not a demonstrated contribution or useful identity signal.

## Mechanism and exact scope

Inputs are cached final-layer frozen DINOv3 query/reference tokens, the complete reference
coverage mask, and the complete native FoRIS response field. There are no added images,
class names, encoder forward passes, intermediate layers, or query labels.

Sixteen appearance slots are initialized by deterministic farthest-point query-token selection
and three spherical Lloyd updates. No native-mask foreground or background seeds are used.
Each slot has a latent foreground identity probability. Reference coverage supplies known
soft labels; foreground and background have equal effective total reference mass, jointly
equal to the query token count. This is a source-loss weighting, not a query-area constraint.

For query token i, slot k and binary identity c, the E step uses

    p(k,c | q_i,s_i) proportional to
      exp(cos(q_i,mu_k)/0.07) theta_k^c (1-theta_k)^(1-c)
      [s_i^c (1-s_i)^(1-c)]^0.25.

For each reference token, its observed foreground/background coverage weights a separate
conditional p(k | r_j,c). The M step updates slot centers and theta from the joint query
responsibilities and labeled reference responsibilities. Five fixed updates produce the full
query foreground probability field. Native scores are clipped into [0.0001,0.9999] only for
logarithms; they are not assumed to be calibrated probabilities. Slight out-of-range cached
native values were observed in the first real packet. Clip frequency is recorded.

The intended change to inference is that an appearance mode inside the native foreground can
become a background competitor, while a weak-native target part can acquire foreground identity
through shared labeled-reference/target slots. No target area is estimated. Whether this actually
corrects those errors is unknown until full-mask scoring. Spatial adjacency is not used.

## Prior failures and non-equivalence

Read the current AGENTS/STATUS/CLAIM, imported demo9 README/HANDOFF, dots failure evidence,
and RCG leave-one-out result. Query foreground/background prototype substitution previously
produced small gains; reference QDA and harmonic inference failed badly. Feature-mixture matte
was a trimap-local coverage interpolation with fixed endpoints. This construction instead
jointly estimates multi-slot identity and membership over the whole query without trimaps,
mask endpoint extraction, covariance fitting, foreground-prior estimation, or RCG gating.

This computational distinction does not establish scientific novelty or additional information.
The main risks remain reference-to-query identity transfer and self-reinforcement of distractors.
A large coherent distractor could recruit both query and source mass. Equal reference class mass
prevents raw reference-area dominance but does not prove protection against that failure.

Agent challenges were incorporated: F identified possible RePRI/prototype reparameterization;
B required separating actual identity correction from expanded initial contamination. The output
records initial/final slot identity, slot sign flips, initial/final foreground fraction and token
flips. The shared evaluator must measure added TP, removed FP, removed TP and added FP from saved
full masks. Neither source reconstruction nor a higher EM objective will count as method evidence.

## Controls and evaluation

- `control`: identical 16-slot initialization, reference mass, temperature and native observation;
  one readout without joint EM updates. This isolates the effect of the updates.
- `two_slot_control`: simple two-slot appearance model with identical five EM updates and terms.
- Complete native FoRIS, RCG and the shared same-input baseline remain required external controls.

No parameter selection or query-GT access occurred while implementing or timing this version.
The fixed renderer is the shared bilinear 64-to-1024 interpolation with align_corners=False,
then threshold >0.5. The complete 1024 masks, not token IoU, determine the result.
Twenty exposed DEV examples can provide an execution/effect-size first look, not independent
confirmation or sufficient evidence for a stable +2 point claim.

## Real input timing (before labels)

Input cache: `/tmp/DEV241_DINO20_export_xknuc38h`, episode `0_11_0`, q/r 4096x1024,
cov/score 64x64. Local CPU, torch one thread, no model loaded.

| Arm | Seconds | Foreground token fraction | Initial-to-final token flips |
|---|---:|---:|---:|
| 16 slots, five EM updates | 0.1065 | 0.498535 | 78 |
| Same slots, fixed initial readout | 0.0647 | 0.505859 | 0 |
| Two slots, five EM updates | 0.0766 | 0.509766 | 77 |

Main field is finite float32, range [0.000520961,0.994371057]. SHA256 of the float32 field bytes:
`de82ace805948eec4401b9a4d53d797cd0564d332df1deb482a847c72aeaebf4`.
Timing excludes loading compressed packets; CUDA timing and final 1024 rendering are unmeasured.
No query truth, native mask or pre mask was read for this timing.

## Reproduction entry points

`src/ics/methods/latent.py`: `CONFIG`, `predict`, `control`, and `two_slot_control`.
All implement `(q,r,cov,score,*,device='cpu',extras=None) -> (HxW float32 field,info)`.
`extras` is unused; no hidden class, GT, filename or episode-index input is consumed.

## GPU smoke4 readout and failure localization

The parent completed the fixed smoke4 GPU run and supplied sealed evaluation reports. This
section reads those results only; no new inference, remote launch, or runtime code change was
performed by E. Sources are `runs/outputs/gpu_smoke_v1/cache/report.json`, `audits.json`, and
`episode_metrics.json`. Prediction seal SHA256:
`1e435f9b612b9b4ed89518731f1624adc7bd785743c23f9b8a8da582e215738a`.

Four exposed DEV episodes, one per fold and imported batch, have four classes and four connected
photo groups. Full 1024 masks, class-summed mIoU, paired 2,000 RandomState(0) connected-photo
bootstrap. These are execution-scale results, not independent confirmation.

| Arm | mIoU | Gain vs native, 95% CI | Up/down |
|---|---:|---|---:|
| Complete native | 66.0181 | -- | -- |
| RCG | 67.0743 | +1.0562 | 2/2 |
| E, 16 slots with EM | 61.3143 | -4.7038 [-11.0515, 1.6440] | 1/3 |
| Same 16 slots without EM | 59.9595 | -6.0586 [-13.0370, 0.9197] | 2/2 |
| Simple two-slot EM | 74.6257 | +8.6076 [-5.9020, 23.1171] | 2/2 |

The 16-slot EM improves over its fixed-slot control by +1.3548 [-0.0527,2.7624],
3 up/1 down, but remains worse than native. This isolates a useful effect in this tiny sample,
not a complete method gain. The simpler two-slot arm is +13.3113 over 16 slots in the pooled
score, but actually loses to 16 slots in three of the four individual episodes.

| Episode / fold / batch | Native | E 16-slot | Fixed 16-slot | Two-slot EM |
|---|---:|---:|---:|---:|
| 0_11_0 / 0 / old20 | 88.7375 | 92.4474 | 90.2725 | 90.8269 |
| 1_38_5 / 1 / new40 | 89.4957 | 89.0738 | 89.8002 | 88.5703 |
| 2_21_78 / 2 / new60 | 34.9361 | 22.4641 | 19.1142 | 79.0810 |
| 3_0_75 / 3 / arrived100_exposed | 50.9031 | 41.2721 | 40.6511 | 40.0246 |

The entire large two-slot pooled gain is driven by `2_21_78` (+44.1448 vs native).
Its gain over the other three episodes averages -3.2382. This leave-one-example calculation is
an explicitly post-result influence diagnostic, not a new selection rule or replacement metric.
The batch/fold singletons cannot establish stability or select two slots as the complete method.

### Four actions and observed failure location

All counts below are full-resolution pixels, relative to the same native masks.

| Arm | Added TP | Deleted FP | Deleted TP | Added FP |
|---|---:|---:|---:|---:|
| E 16-slot | 39,400 | 57,484 | 23,316 | 151,859 |
| Fixed 16-slot | 40,139 | 17,171 | 4,759 | 227,422 |
| Two-slot EM | 45,308 | 36,400 | 4,737 | 183,571 |

1. `2_21_78` distinguishes deletion quality, not simply deletion quantity. Sixteen-slot EM deletes
   32,871 FP but also 14,045 TP, and adds no TP/FP. Two-slot EM deletes virtually the same FP
   (32,841), but deletes only 2,449 TP and adds 54 TP without adding FP. Thus the two-slot advantage
   in this episode is chiefly preserving 11,596 more native true-positive pixels while achieving
   nearly identical false-positive removal. The source of that preserved identity remains unproven.
2. In the same episode, 16-slot foreground fraction collapses from 0.102539 to 0.005859 during EM
   (396 token decisions change). Two initially borderline-FG slot probabilities 0.534405 and
   0.532667 become 0.000858 and 0.003688. The two-slot foreground fraction changes from 0.050049
   to 0.017090, with no slot identity sign flip. The observed failure is excessive target deletion
   during joint updating. Audit fields do not save token-to-slot responsibilities, so the two slot
   flips are associated evidence, not an established causal assignment of the lost target pixels.
3. `3_0_75` already fails before iteration: fixed-slot adds 153,555 FP while deleting only 1,369 FP.
   Sixteen-slot EM improves this to 139,581 added FP and 10,423 deleted FP, but cannot repair the
   initial scope error. Two slots add 165,200 FP with only 382 deleted FP. This concrete batch
   regression must remain visible beside the attractive pooled two-slot score.
4. `0_11_0` shows a useful joint update: 16-slot adds 19,623 TP, deletes 9,831 FP, deletes 817 TP
   and adds 8,784 FP. Here no slot identity sign changes; benefits do not require identity flips.

### Newly discovered native-score interface mismatch

The first timing packet had only 0.244% raw response values outside [0,1]. The actual smoke audits
show clipping fractions of 66.284%, 73.022%, and 39.697% for the other three episodes. The earlier
phrase "slight out-of-range cached native values" therefore applies only to that first packet.

`src/ics/experiment.py` verifies native pre-mask parity using `(score-min)/(max-min)` followed by
the fixed renderer. E instead feeds `clip(raw_score,0.0001,0.9999)` to the noisy-native likelihood.
These are different unaries. Calling the latter the native FoRIS observation without qualification
was an interface-semantics error. Even though E does not assert score calibration, it must preserve
the actual native normalization before applying its fixed likelihood transformation.

The ongoing 20-example run was not changed, stopped, or silently replaced. Its original results
must remain attached to the raw-clipped-score configuration. The clean next check, after it finishes,
is a separately labeled normalization bugfix replay with the identical three arms and all other
parameters frozen. This is not evidence that normalization will rescue E, nor authorization to
sweep temperatures, slot counts, mixture weights, or per-example gates. Two-slot selection waits for
that coherent comparison and broader evidence; if the simpler complete arm then wins, retain it
as the method rather than defending extra slots.

### Approved isolated interface repair (prepared, not executed by E)

The parent approved `src/ics/methods/latent_native.py` as a thin wrapper. It passes
`(raw-min)/max(max-min,1e-6)` into the unchanged functions in `latent.py`, exposing
the same `predict`, `control`, `two_slot_control`, `CONFIG`, and `additional_controls`
API. The base module remains untouched while the original run is active. The wrapper
records raw min/max, denominator, original out-of-range clipping fraction, and an explicit
interface-repair marker. All mechanism parameters are unchanged. No test or inference
of this wrapper was run by E; the parent will seal the original first20 before running
the corrected three arms through the common GPU pipeline.

## Corrected native-normalization GPU run: exposed DEV20

This section supersedes the raw-score smoke result for judging E's current behavior. The parent
ran the three unchanged-mechanism arms through `latent_native.py` on GPU, then evaluated sealed
1024 masks. E read the completed reports and did not launch inference or change any running code.
Sources: `runs/outputs/latent_native20/{report,audits,episode_metrics}.json`.
Seal SHA256: `05ab9424cbda4134d5c471cfcd77294ea895906f34c361234686447f1a46f631`.

Protocol: COCO-20i 1-shot exposed development sample, 20 episodes, 18 classes, 20 connected photo
groups, largest group 1, five episodes per fold. Cached final frozen DINOv3 features, original
reference coverage, correctly normalized native response; no extra forward passes or inputs.
Shared full-1024 rendering and class-summed mIoU; 2,000 paired connected-photo bootstrap draws,
RandomState(0). This run is neither independent confirmation nor evidence about the full 220/241.

| Arm | mIoU | Paired gain vs native, 95% CI | Up/down/tie |
|---|---:|---|---:|
| Complete native | 55.0962 | -- | -- |
| RCG | 55.6279 | +0.5318 [-1.0317, 1.5055] | 10/9/1 |
| Corrected E, 16-slot EM | 54.7861 | -0.3101 [-8.9904, 3.0746] | 7/12/1 |
| Corrected fixed 16-slot | 49.4572 | -5.6390 [-9.8006, -2.4383] | 5/14/1 |
| Corrected two-slot EM | 54.7121 | -0.3841 [-10.7819, 7.5046] | 7/12/1 |

The joint updates retain +5.3289 [-2.8993,10.6045] versus fixed-slot readout, 11 up/8 down/1 tie.
E is -0.8418 [-9.2753,2.2605] versus RCG. The 16-slot versus two-slot difference is only +0.0740
[-9.8597,8.8614], 11 up/8 down/1 tie. Neither slot count demonstrates the complete-method target.
The corrected two-slot control is not adopted: its pooled near-tie hides severe fold/batch losses.
The intervals' crossing zero is not the rejection criterion; observed effect size and instability
fail to support stable +2 superiority, while the intervals leave the underlying effect unresolved.

### Fold and batch stability (gain against complete native)

| Scope | Episodes | E 16-slot | Fixed 16-slot | Two-slot EM |
|---|---:|---:|---:|---:|
| Fold 0 | 5 | +1.1679 | +0.0018 | -12.8947 |
| Fold 1 | 5 | -1.2569 | -11.3222 | -12.8757 |
| Fold 2 | 5 | +3.9447 | -11.1304 | +30.6325 |
| Fold 3 | 5 | -3.9494 | -0.0753 | -2.6971 |
| old20 | 4 | -0.1193 | -2.5123 | -0.6919 |
| new40 | 4 | -4.6316 | +1.4939 | +13.3702 |
| new60 | 4 | +6.1567 | -9.1060 | +10.5755 |
| arrived100_exposed | 8 | -2.7797 | -7.7354 | -15.6841 |

These small strata describe this sample; they do not estimate each original batch's full result.
In particular, two-slot fold-2 +30.6325 cannot cover its approximately -12.9 losses in folds 0 and 1
or arrived100_exposed -15.6841. E's positive new60 also cannot cover new40 -4.6316.

### Four actions: useful rejection and remaining damage

Pixel counts below compare each completed 1024 mask with its same-episode native mask. They
are aggregate action counts, not a substitute for class-summed IoU.

| Arm | Added TP | Deleted FP | Deleted TP | Added FP |
|---|---:|---:|---:|---:|
| E 16-slot | 71,072 | 349,860 | 92,962 | 226,775 |
| Fixed 16-slot | 112,380 | 49,370 | 26,756 | 796,701 |
| Two-slot EM | 522,442 | 116,434 | 99,168 | 1,085,684 |
| RCG | 53,971 | 144,354 | 42,217 | 131,176 |

Compared with fixed slots, E ends with 870,416 fewer false-positive pixels but also 107,514 fewer
true-positive pixels. The joint loop meaningfully repairs the fixed readout's overexpansion while
eroding target extent. Relative to native, E still deletes more TP than it adds (92,962 vs 71,072).
Two slots can recover much more target, but add over one million FP and damage other batches.
The simple control is informative, not presently sufficient.

| E batch | Added TP | Deleted FP | Deleted TP | Added FP |
|---|---:|---:|---:|---:|
| old20 | 38,728 | 19,118 | 5,184 | 31,040 |
| new40 | 4,393 | 9,475 | 48,159 | 6,603 |
| new60 | 5,093 | 84,009 | 20,435 | 11,663 |
| arrived100_exposed | 22,858 | 237,258 | 19,184 | 177,469 |

Examples locate distinct failures rather than attributing everything to the old normalization bug:

- **Retained useful rejection:** corrected `2_21_78` improves native 34.9361 to 74.427 (fixed slots
  12.210). It deletes 32,767 FP and 3,313 TP, adding 16 TP and no FP. Two slots reach 87.145 with
  32,232 deleted FP and only 614 deleted TP. The old raw-score 16-slot collapse is not the result
  of the repaired method and must not be reused to describe it.
- **Improved baseline before falling short:** `1_5_1` improves fixed-slot 35.299 to 54.711, near
  native 55.521; `1_21_65` improves 69.667 to 88.358 but remains below native 90.794. The loop
  preserves meaningful control-relative repair without establishing native superiority.
- **Iteration destroys an initially usable small target:** `2_6_74` is native 90.369, fixed-slot
  45.291, then E 13.541. E deletes 1,508 FP and 12,705 TP, with no added pixels. Its token FG
  fraction contracts from 0.032959 to 0.003418; one slot identity moves from 0.622 to 0.302.
  Two slots are also inferior (59.500), so simply replacing 16 by 2 does not solve this case.
- **Overfragmented identity misses a large target:** `2_38_50` is native 26.997, fixed 25.009 and
  E 19.817. E deletes 39,328 TP while recovering only 414 TP. Two slots reach 95.057 by adding
  375,404 TP and 10,244 FP, deleting only 86 TP. In 16 slots, only one identity starts above 0.5
  (0.940) and it sharpens to approximately 1; the other initial identities are at most 0.264.
  This is observed insufficient target extent under the fine appearance partition, not proof that
  any particular additional target region can be recognized without labels.
- **Loss without identity sign changes:** `3_21_39` changes native 17.204 and fixed 25.314 to
  E 5.777. It deletes 44,946 FP but also 11,219 TP, and adds no TP. Foreground fraction contracts
  from 0.112061 to 0.002197 without any theta sign flip. Therefore counting slot identity flips
  is not an adequate diagnostic for target preservation; responsibilities and center motion matter.
- **Expansion remains wrong before and after EM:** `3_0_75` is native 50.903, fixed 40.093 and E
  40.722. E adds 151,245 FP and 18,831 TP, deleting only 3,380 FP. Two slots worsen this to 39.685
  and 171,719 added FP. Correct normalization does not repair this target/distractor ambiguity.
- **Large deletion can have zero IoU value:** `0_0_72` has zero intersection under all three E arms
  and native. E deletes 176,439 FP, over half of its total deleted-FP count, without recovering a
  target pixel. This helps explain why aggregate pixel cleanup cannot stand in for class mIoU.

### What the iteration audit actually supports

Nineteen of 20 episodes shrink their foreground token fraction during EM. Initial center motion
is substantial: first-step mean old/new-center cosine ranges 0.835650--0.943432. By step five it is
0.993792--0.999457. Identity probabilities become sharp: across episodes, the median minimum theta
is 0.001446 and median maximum is 0.999093. Final-step mean identity change ranges 0.002514--0.028904.

The implementation therefore settles into a sharpened, mostly contractive readout; convergence
of centers does not imply correct identity or extent. Both `2_21_78` (useful rejection) and
`2_6_74`/`3_21_39` (target destruction) contract strongly. These values cannot justify an early-stop
selector without new validation, nor a threshold on shrinkage that treats all contraction as failure.

The audit lacks per-step complete masks, per-token slot responsibilities, and separately saved
reference/query contributions to center updates. It cannot causally decide whether target erosion
comes from the source pulling a query component off its native appearance, from identity sharpening,
or from a valid-but-incomplete source appearance partition. No claim that a specific slot "caused"
a lost GT region is made. Query truth was used only in these explicit scoring/error diagnostics.

### Next mechanism boundary, not an active sweep

No slot-count, temperature, iteration-count or likelihood-weight search is proposed. One concrete
structural hypothesis suggested by the implementation and measured center motion is that using
reference reconstruction mass to update the query's appearance centers allows source-specific
foreground modes to retain high theta while losing their query membership. This could explain
vanishing query target support even with unchanged slot identity signs; it is not yet established.

A distinct complete construction, if pursued after current work, would keep query appearance
geometry query-owned: update centers from query responsibilities only, while reference labels
update identity probabilities but never move those centers. The unchanged initial-slot readout
and the current joint-source-center EM are immediate same-input controls; a fixed-query-center
identity-only update would isolate whether any query-center adaptation is needed. This changes
where source supervision acts, rather than increasing slots or gating two completed masks. It is
separate from B's conditioned pair-correspondence and F's inverse-source-coverage evidence.

No such construction has been coded or run, and no effect is predicted. Before assigning cause,
a run would need compact per-step responsibilities or domain-separated effective masses, while
still delivering complete masks and the four error actions. The present completed result remains
corrected E's unresolved -0.3101 against native, with actual stratum regressions and no selected method.
