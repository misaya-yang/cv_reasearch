# Structure mechanism B: full-image correspondence compatibility

Status: implemented; one real episode executed without query truth. Accuracy remains unmeasured until the shared frozen-prediction evaluation. This is exposed development data, not independent confirmation.

## Gap, inference change and complete output

Complete FoRIS can include appearance-similar background while missing target parts. Earlier local structural templates did not fix this: D4 local matching scored 40.060 versus native-pre 60.174 on exposed old60. The earlier fixed sampled-star correspondence method scored 62.160 versus native 62.976 on ten development episodes and lost to same-count score deletion (63.081). Source-part coverage also failed. These are reasons not to rename those constructions.

B tests a different scope of inference: a joint soft correspondence matrix over **every query and reference token**, not a finite local-template library, candidate-mask bank, or a gate on RCG. The frozen final-layer features are the only appearance input. Reference coverage supplies labels; query score supplies the output grid shape only.

1. Normalize the existing query/reference features.
2. Form all-pairs cross-image cosine likelihoods with fixed temperature 0.05.
3. Equalize reference foreground/background role priors using soft coverage; impose no query foreground proportion.
4. Build a 12-neighbor feature graph separately in each image, excluding self edges and normalizing every row.
5. Initialize every query's soft correspondence row from the cross-image likelihood and reference role prior.
6. Compute correspondence compatibility `Aq @ T @ Ar.T`, involving both image relationships.
7. Divide by the same operation on the static reference prior to remove its graph-induced source mass bias.
8. Recombine the message with the original appearance likelihood using fixed strength 0.5; repeat four times.
9. Transfer the reference labels through the resulting correspondence matrix.
10. Bilinearly render the probability field to 1024 using `align_corners=False`, then threshold at 0.5 in the shared runner.

The testable prediction is that the joint assignment distinguishes an appearance-similar distractor with inconsistent neighborhood correspondences, while retaining missed target tokens with compatible correspondences. This is a hypothesis, not an established object-binding guarantee. A coherent wrong object may satisfy the same relations.

## Controls and independent challenges

- `control`: independent class-balanced kernel label transfer on exactly the same features and reference mask.
- `diffusion_control`: identical graphs and iteration count, but repeatedly averages the initial correspondence matrix with its separable graph diffusion. This tests whether graph smoothing alone explains the gain.
- `entropy_control`: independent transfer whose single global inverse temperature matches the primary's mean assignment entropy, chosen without query labels. This tests whether generic sharpening explains the gain. Its cost includes computing the primary's entropy.
- Shared-runner comparisons must additionally include the same-batch complete native and exact RCG where executable.

Agent C challenged that `Aq T Ar.T` may be mere separable spectral smoothing and amplify high-density background. Agent D challenged foreground-mass collapse and an entropy confound. B added the two controls and per-iteration entropy/foreground-mass logging in response. B does not claim the matrix expression alone establishes a higher-order semantic mechanism. Agent E owns latent appearance slots; D owns cross-layer trajectory evidence; C owns frozen attention interventions. B uses none of those additions.

## Resource and reproducibility contract

Implementation: `src/ics/methods/structure.py`. `CONFIG` freezes the configuration before accuracy readout. There are zero encoder forwards, no model loads, no image downloads, no class names, no base-class fitting, and no query labels in the method API. Heavy matrix products honor the requested `device`; CPU inference is single-threaded under the shared runner.

Real timing on cache key `0_11_0`, 4096 query and 4096 reference tokens of dimension 1024, CPU one PyTorch thread:

| Arm | Seconds | Correspondence matrix |
|---|---:|---:|
| Primary | 0.864 | 64 MiB |
| Independent control | 0.102 | 64 MiB |
| Diffusion control | 0.610 | 64 MiB |

Several matrices coexist; 64 MiB is not peak process RSS. Primary mean assignment entropy changed 6.595 to 5.280 and predicted foreground probability mass changed 0.713 to 0.672. These are unlabeled execution diagnostics, not segmentation gains. The entropy control was added before any query-truth read in this subtask.

## Result ledger

No accuracy result yet. Required readout: same-batch class-summed complete 1024 mIoU, paired photo-connected-group RandomState(0) 2000-draw bootstrap, folds, batches, episode increases/decreases, recovered FN, removed FP, lost TP and added FP. Report no independent-confirmation claim for these exposed data. Do not infer method value from entropy changes or a successful execution.

## User-requested preparation pause

After the one-episode timing above, the user stopped further CPU tests and requested code preparation before enabling the GPU. No 20-episode run, query-truth read, accuracy calculation, or subsequent synthetic inference was performed by this worker. `additional_controls` exposes the diffusion and entropy controls to the shared runner. The entropy control reuses its own primary logits rather than recomputing cross-image features inside that call; when separately called after `predict`, its primary computation is an explicit extra inference cost. GPU execution and entropy-control runtime remain unverified.

## Measured GPU smoke: four exposed development episodes

Source: `runs/outputs/gpu_smoke_v1/cache/report.json`, `audits.json`, and `episode_metrics.json`. All outputs were frozen before shared evaluation. This batch checks execution and local failure links; four episodes cannot establish efficacy or stable generalization.

Protocol: frozen cached final-layer DINOv3, seed-0 source episodes, complete 1024 masks; four classes, four photo-connected groups. Paired intervals use 2000 connected-photo draws with RandomState(0).

| Arm | mIoU | Gain vs native | 95% interval | Up/down |
|---|---:|---:|---|---|
| structure | 58.437 | -7.581 | [-15.160, -0.002] | 1/3 |
| structure.control | 53.487 | -12.532 | [-18.404, -6.659] | 0/4 |
| structure.diffusion.control | 54.476 | -11.542 | [-16.806, -6.279] | 0/4 |
| structure.entropy.control | 52.660 | -13.358 | [-19.969, -6.747] | 0/4 |

Native: 66.018; RCG: 67.074. Primary minus independent control +4.950 [1.117, 8.784], versus diffusion +3.961 [0.737, 7.186], versus entropy +5.777 [1.204, 10.350]; 4/4 episodes improve over each control. These smoke intervals must not be mistaken for confirmation.

| Fold / batch | Episode | Primary gain vs native |
|---|---|---:|
| 0 / old20 | 0_11_0 | -19.432 |
| 1 / new40 | 1_38_5 | -0.196 |
| 2 / new60 | 2_21_78 | +0.192 |
| 3 / arrived100_exposed | 3_0_75 | -10.888 |

Each fold and batch has only one episode in this smoke.

| Arm | Recovered FN | Removed FP | Lost TP | Added FP |
|---|---:|---:|---:|---:|
| structure | 55,788 | 16,413 | 2,013 | 376,978 |
| structure.control | 58,593 | 13,177 | 671 | 468,401 |
| structure.diffusion.control | 58,150 | 13,618 | 779 | 445,132 |
| structure.entropy.control | 58,492 | 13,203 | 989 | 490,925 |

### Located failure link and next decision

The independent correspondence readout already loses the strong host extent: it creates 468,401 false-positive pixels. Joint compatibility reduces that count by 91,423 (19.5%), while recovering 2,805 fewer true pixels and losing 1,342 more existing true pixels. These are differences of each arm's action counts against native, not direct pixel-set transition counts between arms. The structural iteration corrects part of the excessive expansion, but the primary still introduces 376,978 false pixels against 55,788 recovered true pixels. The dominant failure is over-expansion, not deletion of the host's true target.

This implementation uses the host score only for its shape, so the independent readout discards FoRIS's complete extent inference. The measured signal is therefore that compatibility helps this weak correspondence readout; it is not yet evidence that compatibility improves a strong complete host. Generic sharpening and plain graph diffusion do not explain the four smoke improvements, but stronger matched controls on twenty episodes are still required.

Wait for the running twenty-episode unchanged configuration before modifying the mechanism. A possible subsequent change, if the same failure persists, is a joint correspondence model initialized with each query token's complete host foreground/background prior, while retaining within-role correspondences. Its zero-interaction control must reproduce the host unary exactly, and its matched graph-diffusion control must test whether any gain needs correspondence interaction. This is not yet an implemented or selected result. No RCG gate, mask fusion, parameter sweep, or rerun of the same fixed construction is proposed.

GPU primary measured wall times: 0.165, 0.081, 0.078, 0.074 seconds per episode. No additional encoder forwards.

### Remaining attribution challenge before a mechanism claim

All four primaries reduce mean foreground probability relative to independent transfer: 0.713 to 0.672, 0.311 to 0.272, 0.105 to 0.070, and 0.548 to 0.536. The entropy-matched control does not match this foreground mass. A later fixed diagnostic should compare against independent kernel transfer with one intercept chosen without GT to match the primary's predicted foreground mass or area. If that explains the gain, correspondence-dependent ranking has not been established. This is a proposed same-output-statistic control, not a new candidate or a retrospective query-label cut search.

Agent E also identified a separate raw-score clipping bug in E's own implementation. B's current candidate never uses score values, so that bug does not affect the reported B outputs. Any later host-conditioned B must use the shared evaluation's exact min-max score normalization and distinguish the reproducible pre-CRF unary from complete native CRF output. Zero correspondence interaction could reproduce the former, not automatically the latter.

## Frozen original configuration: all twenty episodes

Source: `runs/outputs/gpu_first20_v1/cache/{report,audits,episode_metrics}.json`; saved fields are available beside the reports. The complete result supersedes smoke-only interpretation: 20 exposed development episodes, 18 classes, 20 connected-photo groups, seed-0 source, full 1024 masks, RandomState(0) 2000-draw photo-group bootstrap.

| Arm | mIoU | Gain vs native | 95% interval | Up/down/tie |
|---|---:|---:|---|---|
| structure | 40.255 | -14.841 | [-25.492, -6.660] | 3/16/1 |
| structure.control | 36.110 | -18.986 | [-30.478, -11.319] | 2/17/1 |
| structure.diffusion.control | 36.847 | -18.249 | [-29.614, -10.546] | 2/17/1 |
| structure.entropy.control | 34.010 | -21.086 | [-33.170, -13.163] | 2/17/1 |

Native: 55.096. Primary gains over independent/diffusion/entropy controls are +4.145/+3.408/+6.245 points; this does not rescue its large complete-native deficit.

| Fold | n | Primary minus native | Primary minus independent |
|---|---:|---:|---:|
| 0 | 5 | -13.863 | +9.548 |
| 1 | 5 | -32.684 | +2.031 |
| 2 | 5 | -8.044 | +5.924 |
| 3 | 5 | -3.217 | +0.513 |

| Batch | n | Primary minus native |
|---|---:|---:|
| arrived100_exposed | 8 | -19.687 |
| new40 | 4 | +6.252 |
| new60 | 4 | -24.066 |
| old20 | 4 | -20.394 |

| Arm | Recovered FN | Removed FP | Lost TP | Added FP |
|---|---:|---:|---:|---:|
| structure | 538,608 | 71,392 | 5,898 | 2,868,955 |
| structure.control | 595,720 | 49,080 | 2,659 | 3,867,040 |
| structure.diffusion.control | 593,627 | 48,861 | 2,754 | 3,607,020 |
| structure.entropy.control | 594,300 | 56,288 | 2,375 | 5,333,424 |

The full result repeats the over-expansion failure: 2,868,955 added false pixels against 538,608 recovered true pixels, with only 5,898 lost true pixels. Structural compatibility removes 998,085 of the independent arm's added false pixels, but complete native remains substantially better. All folds lose to native; the apparently positive new40 batch does not hide the large losses in the other three batches. Thus v1 remains a failed complete method, with a bounded observation that its correspondence feedback improves its weaker independent readout.

## Approved changed inference: native-conditioned joint correspondence v2

Implementation: `src/ics/methods/structure_conditioned.py`. Original `structure.py` and running/frozen v1 outputs remain unchanged. V2 is a change to the inference problem prompted by the measured loss of host scope, not a retuning of v1's temperature, graph size, iteration count, or output threshold. Its results remain unmeasured here. These are still exposed development episodes.

Let `p0` be the exact FoRIS min-max normalized raw score; it is a pre-CRF unary, not calibrated truth probability and not the complete native CRF mask. For each query token construct two distributions `F0(i,j)` and `B0(i,j)` over **all** reference tokens using the same temperature-0.05 cosine likelihood, weighted by reference coverage or its complement and normalized within role. Initialize the two joint assignment matrices as `Jf=p0*F0`, `Jb=(1-p0)*B0`. This initial marginal reproduces `p0` exactly without discarding its query-dependent scope.

Four iterations use the same frozen 12-neighbor feature graphs as v1. For each role, compute `M=Aq @ J @ Ar.T`, normalize its source-token distribution **within that role**, and divide by the corresponding normalized message from the source role prior. The resulting compatibility ratio reweights `F0` or `B0`; its expectation under that original conditional is a role-specific evidence factor `Ef` or `Eb`. Update the role marginal with `logit(p)=logit(p0)+log(Ef)-log(Eb)`, and form the next joint correspondence state from this marginal and the updated conditional distributions. Role probability mass is allowed to change. No per-query foreground marginal is locked after initialization, no mask is blended with another mask, and no RCG output is used.

The label-conditional normalization removes a trivial vote from how much foreground mass was in the query neighborhood. It asks which source identities the neighbors support **given** each role. It does not prove semantic correctness: wrong objects can still be relationally compatible, and a strong but wrong native unary can dominate. The log-evidence expression is an inference construction, not a claim of calibrated Bayesian probabilities or a guaranteed mIoU objective.

Parameter choice before any v2 result: prior and compatibility coefficients are both **1**, giving each factor its natural unit log weight; temperature 0.05, 12 neighbors and four iterations are inherited unchanged from the frozen v1 construction. There is no parameter sweep, fold-specific value or query-label choice. Numerical endpoint clipping uses float32 machine epsilon only in the candidate odds calculation.

Controls:

- `control`: native-prior independent correspondence marginal, exactly the pre-CRF unary. This is explicitly a **no-interaction component control**, not the sole strong same-input method.
- `kernel_control` (`additional_controls['kernel']`): native odds multiplied by the single class-balanced FG/BG kernel likelihood ratio with coefficient 1, using the same q/r/cov evidence but no relation feedback.
- `diffusion_control` (`additional_controls['diffusion']`): same initial two joint matrices, both image graphs and four iterations; averages the initial state with its normalized joint graph message, without conditional likelihood feedback.
- Shared evaluation must retain complete native and RCG and report pre/native differences faithfully.

Validation performed without query truth or heavy feature products: all **20** locally exported score/pre packets gave **zero differing 1024 pixels** for `zero_interaction_control`; the returned field matched the shared min-max normalization exactly (maximum absolute difference 0). This is a renderer/contract check, not a v2 accuracy run. GPU primary, kernel and diffusion outputs are to be run by the parent scheduler; this worker started no remote process.
