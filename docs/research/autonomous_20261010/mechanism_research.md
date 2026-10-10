# Scene-conditioned role reconstruction: evidence, implemented candidate, falsifiable limits

2026-10-10. Branch `codex_m4`. This track implemented exactly one self-contained complete field predictor in `src/ics/methods/autonomous_context_transport.py`. It has not launched inference on real episodes or read query GT. The parent owns the matched cache runner and prediction/scoring freeze. Existing dirty files and sealed experiments remain untouched.

## What the existing evidence actually requires

The verified evidence audit and all five completed parallel reports were read. The central failure is source-to-query task transfer, not merely an inaccurate solver or insufficient foreground score magnitude.

- The source APD-ridge direction has useful difficult-region ordering: on the fixed Deep100 new-region ROI whole/local AUC is .8015/.8667; on PACO100 it is .7641/.6717. The preference reverses. The dataset-difference report finds that reference area/purity and whole/local agreement explain little of this reversal **within** either dataset. Switching views by dataset name or a cross-dataset purity correlation is unsupported.
- A legal reference IoU threshold changes PACO whole ridge from 39.090160 to 24.300485. The FoRIS anchored version reaches 49.227401 on that PACO100; its shared exact global rule subsequently reaches only 62.459822 on COCO200, below FoRIS63.358900 and MEAN64.006821. Better source self-fitting or lawful source calibration is not enough.
- Replacing coverage regression by correctly balanced binary-role regression fixes a genuine target-meaning mismatch but does not create new identity information. Fixed equal whole/local gives Deep34.641645 versus FoRIS29.377542 and fast37.344028; PACO35.666781 versus FoRIS45.563915. The two-domain success condition still fails.
- Existing query-context component filtering has substantial LVIS gains, whereas source-only presence and unfiltered windows fail. This justifies investigating unlabeled query organization. It does not justify permanently locking FoRIS pseudo-labels or calling any query-derived statistic reliable.
- The audited R1-G, local geometry and Gram experiments constrain replacing identity with reference coordinates or generic structure. Reference Fisher separation improving while difficult-query AUC falls directly constrains source fit as a success criterion. The new candidate uses neither correspondence geometry nor a source confidence threshold.
- PACO foreground means the selected part inside a parent crop, not the parent object. Most observed PACO false-positive pixels lie inside the same parent. Any grouping of query tokens can therefore mix legitimate FG and BG; the candidate must be tested for this failure rather than assuming an object cluster is a target.

These are already exposed development cohorts. No number above is new performance produced by this track. The metric boundaries in the originating reports remain: Deep total original-frame I/U; PACO100 observed87 fold/class slots; PACO600 official303 slots including39 missing; COCO200 official40 slots.

## Complete implemented candidate

The reference defines a signed binary task. The query defines an unlabeled scene basis in which that task is reconstructed. Query tokens are never assigned segmentation pseudo-labels during fitting.

Inputs are unit CPU FP32 feature arrays `R[N,D]`, `Q[M,D]` and exact continuous reference patch coverage `c[N]`. The caller supplies one shared APD branch and projection to all arms. The module does not perform implicit normalization or APD, construct DINO, inspect RGB or obtain masks. Default inference needs only whole reference and whole query O24.

**Scene mixture.** Select at most512 deterministic mask-independent candidates; on a64×64 grid they are22×23=506 spatial strata. Initialize at most128 feature centers by deterministic spherical farthest-point selection, remove duplicate initial centers, and perform exactly four spherical Lloyd updates on all query tokens. Recompute the final hard token counts `n_k`, and define

```text
pi_k = (n_k + .5)/(M + .5 K).
A_X[i,k] = softmax_k(x_i dot z_k/.07 + log pi_k).
Psi_X[i,k] = A_X[i,k]/sqrt(pi_k).
```

The query-grid coordinates choose reservoir strata only. They do not impose cross-image geometry, spatial smoothing or a same-object prior. Empty Lloyd components retain their previous unit center; the fixed half-count makes all masses positive. Query mixture mass is an unlabeled feature-distribution model, not a foreground-area estimate or a required class proportion.

**Reference supervision.** Preserve the existing128-per-role cumulative-area quantile sample IDs and their deduplication. For the selected coverage values, `C=sum(c)`, `B=sum(1-c)`, define

```text
a_i = .5 c_i/C + .5 (1-c_i)/B
y_i = (c_i/C - (1-c_i)/B)/(c_i/C + (1-c_i)/B).
```

Fit an unregularized intercept `b` and coefficient `w` by the exact convex objective

```text
sum_i [ .5 c_i/C (Psi_R[i]w+b-1)^2
       +.5 (1-c_i)/B (Psi_R[i]w+b+1)^2 ] + .01 ||w||^2.
```

This equals `sum_i a_i(Psi_R[i]w+b-y_i)^2 + .01||w||^2` plus a known constant. All fit arithmetic is FP64; the dual system has at most256 rows. Coverage remains continuous: a mixed patch contributes to both roles. The reference-mask complement flips every fitted field's sign because it preserves `a` and negates `y`; this is a useful exact semantic property, not a novelty claim.

**Complete field.** Output `f_Q=Psi_Q w+b`, with a strict zero decision downstream. Equivalently, with latent component values `u_k=w_k/sqrt(pi_k)`, `f_Q=A_Q u+b` and the penalty is `.01 sum_k pi_k u_k^2`. Thus the regularizer is the scene-integrated squared latent-role value, rather than a free penalty on every component regardless of its occupancy. There is no minmax, source threshold, FoRIS anchor, class-name switch, query area quota, graph propagation or global veto.

Return schema is frozen as:

```python
fit_predict(reference_features, reference_coverage, query_features,
            query_grid_hw=(64, 64))
# {'fields': {'scene': np.float32[M],
#             'source_kernel': np.float32[M],
#             'support_ridge': np.float32[M]},
#  'diagnostics': JSON_safe_dict}
```

Diagnostics contain fixed sample and landmark IDs, occupancies/empty components, source reference scores and binary-role reconstruction loss, dual condition numbers/solve residuals, role-specific responsibility masses, field range and stage timing. Source reconstruction and responsibility masses are diagnostics of represented evidence, **not** calibrated query confidence.

## Where the added information comes from, and where it cannot come from

`scene` is nonlinear and transductive: every query token participates in forming the scene centers and their mass, which changes both the representation of reference supervision and its readout on the query. It can express several query modes for a whole object and opposing modes for a part and its same-parent background. It tests whether expressing the task in actual query feature organization transfers better than a fixed source direction. This is a different computation from multiplying a FoRIS mask by a reference score or moving a scalar threshold.

The positive claim above is functional capacity, not demonstrated accuracy. Two identical query token vectors receive identical scores. A same-parent distractor indistinguishable in O24 remains indistinguishable. A missing tiny target can be absent from the bounded candidate reservoir; averaging within a mixture can obscure a part; shifted query foreground can reconstruct a reference background better than foreground. Query-only structure is not a new ground-truth identity signal. No inference result can be assumed from these formulas.

The four Lloyd updates are a fixed finite approximation to scene density organization, not an assertion of EM convergence or likelihood optimality. The source field fit is solved exactly for that fixed basis. The candidate does not address new visual observations, thin-token information loss or multi-view selection; the parent should measure this limitation before adding a view adapter.

## Matched controls and falsifiable mechanisms

All three fields are returned in one call and share reference coverage, fit IDs, binary-role loss, lambda, feature preprocessing and signed zero renderer.

1. `source_kernel` builds the identical128-capacity mixture/responsibility/fit procedure from the **unlabeled full reference features** instead of the query. This tests whether the actual query scene basis adds useful transfer beyond a nonlinear source kernel. Mixture counts may differ only when a cloud has duplicate centers; actual counts are disclosed. The maximum capacity and protocol are identical.
2. `support_ridge` fits the identical supervised loss directly in the shared original feature space, with the same sample/weights/intercept/regularization. This is the existing balanced-role ridge direction, reproduced exactly in the synthetic parity check. It tests nonlinear representation versus a known simple discriminant.
3. Full FoRIS and MEAN on the same sealed inputs remain external complete-mask references. Use the same direct renderer and the same one-pass final CRF on each complete candidate arm; direct and CRF results must remain distinct. No finalizer should supply segmentation pseudo-labels to this field fitter.

The parent is preparing Deep100/PACO100/COCO200/LVIS200 cached whole-pair evaluation. All predictions should be sealed before labels enter scoring. On a failure, determine source fit versus query ordering versus readout from these already fixed fields; do not retune mixture size, temperature, regularization or threshold to the query GT.

The scene-transfer hypothesis fails if scene output does not improve complete results beyond source_kernel and support_ridge, or if its improvements do not generalize under one shared rule. Source loss reduction alone does not pass. If gains are explained by nonlinear source fitting, the query organization contribution is unsupported. If only CRF provides gains, separate the finalizer's contribution. If PACO parts regress, inspect how many opposing-role reference responsibility masses collide in each scene component and how many same-parent FP edits result. If Deep thin coverage is unrecovered, compare ordering in omitted-target regions before proposing a new observation.

## Primary-source novelty check

- [FoRIS paper](https://arxiv.org/html/2609.03384v1) and the local official source already combine adaptive positional debiasing, multimodal FG/hard BG evidence, reference votes and query clustering. Merely calling query clustering or background suppression a new mechanism would be incorrect. The candidate replaces that semantic readout with reference-risk reconstruction in an unlabeled query basis; it does not establish originality by replacement alone.
- [INSID3](https://arxiv.org/html/2603.28480v1) establishes the frozen-DINOv3 in-context setting and positional correction. Reusing O24/APD is representation reuse, not a contribution.
- [FROST](https://arxiv.org/html/2606.31136v1) already uses frozen-DINOv3 foreground/background kernel densities, a zero density-ratio decision and source leave-one-out bandwidth selection. A two-role LSE/KDE readout with source calibration is therefore not an original direction. This implementation instead learns a signed function through scene basis reconstruction; its primitives are still classical kernel methods.
- [RePRI](https://openaccess.thecvf.com/content/CVPR2021/papers/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.pdf) already establishes query transduction via supervised source risk, unlabeled query entropy and a predicted-area KL term. Query transduction by itself is not novel. This candidate uses no entropy minimization or predicted foreground proportion constraint.
- [Prototype Mixture Models](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530749.pdf) already estimates multimodal reference prototypes via EM in a learned FSS pipeline. [Query semantic reconstruction](https://link.springer.com/article/10.1007/s00371-023-02817-x) uses query background semantics during training. Therefore neither mixtures nor query background reconstruction alone supports a substantial original contribution.

This is an implemented **mechanism test**, not an Oral-level method claim. The retrieved primary sources do not establish this exact combination as prior art, but that absence in a focused search is not a novelty proof. A substantial contribution would require a distinctly justified transfer principle, causal matched evidence and full-protocol cross-dataset performance beyond these classical controls. The current track supplies the compact runnable test that can falsify the central scene-organization premise first.

## Executed synthetic validation and cost boundary

Synthetic checks were run using the `aidemo` Python runtime, CPU2 threads, without real cached O24, DINO, masks or query GT:

- Existing `balanced_role_ridge.fit_pair` versus returned `support_ridge`: byte-exact field parity.
- Binary-role versus collapsed-risk closure: absolute errors at most2.23e-16; dual solve relative residuals below1e-10; weighted coefficient/intercept stationarity below1e-10.
- Permuting49 synthetic query tokens when all are in the reservoir: all three fields permute exactly. This does not assert invariance to grid-reservoir changes when more than512 candidates are subsampled.
- Constant unit features: one scene component and zero role field. Orthogonal positive/negative role features: all three arms recover the correct signs.
- JSON serialization with `allow_nan=False`: passed.
- A synthetic4096×1024 two-cloud timing fixture: .06868 seconds for all three arms at CPU2 threads; scene/source mixture .01707/.01485, scene/source kernel fit .01067/.00982, support fit .00664 seconds. This is one synthetic runtime observation, not cache-I/O time, real segmentation speed or cold end-to-end inference.

No broad test framework, parameter menu, official benchmark inference or module imports outside numpy/torch were added. Parent freeze/runner owns the real evidence next.
