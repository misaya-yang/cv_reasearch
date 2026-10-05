# Mechanism F: local convex reconstruction

Status: implemented and timed on one real cached episode; no efficacy result
yet. No encoder has been run. Parameters were frozen before reading any query GT.
The user subsequently stopped CPU testing pending GPU availability. No further
real-data or synthetic execution is authorized now. The next activity is a
user-authorized GPU run, not the previously contemplated twenty-case CPU run.

## Gap and construction

The imported exposed DEV241 record identifies 34 heavily contaminated FoRIS
seed cases; making the seed smaller cannot fix the 29 where its highest token
is background. Source-verified candidate selection sometimes changes identity,
but its source-selected cosine cut yields 39.05 versus 58.56 before CRF and
overmerges. A new ranking plus query-side extent rule must be tested jointly.

1. Use exactly the cached frozen DINO query/reference tokens and complete reference mask.
2. Normalize full token vectors; split coordinates using fixed RandomState(0).
3. For each query token, find eight nearest reference FG tokens in one half.
4. Fit a nonnegative convex combination of those tokens in that same half.
5. Measure reconstruction error on the other coordinate half.
6. Repeat independently for reference BG, then exchange the halves and sum errors.
7. Define target evidence as BG residual minus FG residual.
8. Apply a fixed 256-bin query Otsu cut and return a soft field centered on that cut.
9. Bilinearly render the entire field to 1024, align_corners=False, and cut at .5.
10. Compare with uniform weights on the exact same eight neighbors, halves and renderer.

The proposed changed inference is local compositional compatibility rather than
single-token affinity or global covariance membership. Convex combinations
cannot extrapolate through an unrestricted subspace. Intended corrections are
recovery of query parts explained by several reference foreground appearances,
and rejection of lookalikes whose foreground reconstruction does not transfer
across coordinates. Neither effect is observed yet; it must be measured.

DINO coordinates are correlated. The coordinate split is a deterministic stress
test, **not independent statistical validation**, not a domain-invariance proof,
and not new model information. All reference parts may be absent or changed in a
query; a local reconstruction can still select a wrong semantic object. Otsu can
also split one background mode from another when the target is small.

## Prior-failure exclusion and controls

RICE's archived equation audit describes a generalized eigenspace of
discriminant/nuisance covariance. This construction has no generalized eigenproblem,
whitening or subspace projection. It also has no source-IoU cut, candidate ranking,
whole-region MMD, native gate, pseudo-query training or learned class labels.
Unlike the failed full-vector ridge discriminator, it solves local convex
reconstruction problems separately for each token and reference label.

The closest necessary control is uniform reconstruction from the same neighbors:
if it matches the optimized method, the simplex optimizer is unnecessary. Native
FoRIS and RCG remain required complete controls in the shared evaluator. A
successful comparison against this control is not by itself a novelty claim.

Agent A challenged coordinate dependence: applying an orthogonal rotation to
identical embeddings can change a coordinate split without changing the original
cosine geometry. This is a genuine unresolved weakness, not a new source of
semantic information. No rotation/seed sweep is added after seeing outcomes.
Agent E's competing latent query slots share no optimization or labels with this
source-only local reconstruction mechanism.

## Resources and evaluation

Additional forward passes: zero. Feature layer: existing cached layer only.
Both arms use the same reference/query inputs. The optimized arm additionally
runs 32 projected-gradient steps on eight coefficients per class and half.
Query chunks contain 128 tokens, bounding dictionary allocation. Tensor matrix
operations follow the caller's CPU/GPU device; the module loads no encoder.

Static CUDA review: inputs, coordinate indices, reference dictionaries, nearest
neighbor matrix products, batched Gram matrices and simplex iterations all live
on the caller's device. Only the final residual margin is transferred to CPU for
the histogram cut; a final device synchronization bounds reported elapsed time.
CUDA execution itself remains unverified. No module-global model or tensor bank
is created, so one shared GPU runner can call this method without encoder reloads.

The control guarantees the same source pool, coordinate permutation, eight-neighbor
search, rendering and available device/resources. Neighbor indices are computed
before the optimization branch and do not depend on optimized weights. It uses
less arithmetic because it omits the 32 simplex iterations: **exact FLOP/time
matching is not claimed**. Runtime and peak device memory must be reported per
arm. Repeating discarded optimization merely to spend the same time would not
create a stronger control or establish compute-independent mechanism gain.

The module accepts no GT or class identifier. All parameters are fixed here before
real-data results. Missing FG/BG reference classes produce a declared all-BG/all-FG
field, respectively; they do not trigger a native-mask gate. Degenerate counts
must be reported separately. There is no performance-based arm selection.

The first real run must retain every 1024 mask and I/U plus TP-restoration,
FP-deletion, TP-deletion and FP-addition counts relative to native. Evaluate the
available exposed cohort, including batches/folds and connected-photo-group
2,000-draw RandomState(0) bootstrap. Small execution tests do not establish
efficacy. Check photos shared across episodes before any fold selection.

## Failure ledger

- Real cache `/tmp/DEV241_DINO20_export_xknuc38h`, episode `0_11_0`:
  q/r each 4096x1024, reference coverage and score each 64x64. CPU one thread:
  optimized 0.432 seconds, uniform control 0.254 seconds. Patch foreground
  fractions .47925/.54175. Query GT was never accessed. This is an execution
  timing and output-shape check, not efficacy evidence.
- Mean evaluated FG reconstruction error .97095 versus control 1.01454;
  BG 1.04482 versus 1.08494. Lower reconstruction error is not segmentation gain.
- Synthetic contract checks passed: simplex nonnegativity and unit mass,
  deterministic outputs, finite float32 HxW fields bounded to [0,1], all-BG
  reference handling and invariance to changing native score values.
- If fitted residual improves but opposite-coordinate residual worsens, local
  combination fitting does not generalize even within feature coordinates.
- If token evidence improves but final masks deteriorate, local residual ranking
  does not solve query extent; do not retune source cuts or add a native gate.
- If all residual rankings deteriorate, retire the mechanism rather than adding
  dictionary-size or coordinate-seed sweeps.

## GPU smoke evidence: four exposed development episodes

The main runner subsequently completed the authorized GPU smoke. Source:
`runs/outputs/gpu_smoke_v1/cache/{report,audits,episode_metrics}.json`.
The original frozen code is unchanged while the twenty-episode run proceeds.

Complete 1024 class-summed mIoU: native 66.0181, RCG 67.0743, optimized
reconstruction 35.2066, uniform control 41.4243. Reconstruction versus native:
-30.8115 [-32.8979,-28.7251], zero up/four down; versus uniform control:
-6.2176 [-12.1987,-0.2365], one up/three down. These four exposed cases, with
one case per fold, do not estimate a general representation limit or an
independent confirmation effect. Bootstrap has only four photograph groups.

| Episode | Restore TP | Delete FP | Delete TP | Add FP | Failure visible in the output |
|---|---:|---:|---:|---:|---|
| 0_11_0 | 7,586 | 22,395 | 131,758 | 136,826 | Large true-region removal and wrong-region addition |
| 1_38_5 | 5,997 | 2,271 | 1,628 | 153,365 | Nearly all native true pixels retained but large expansion |
| 2_21_78 | 1,853 | 1,671 | 8 | 460,444 | Almost complete true-region retention, catastrophic expansion |
| 3_0_75 | 7,281 | 120,300 | 71,524 | 316,479 | Both substantial deletion and wrong-region addition |
| Sum | 22,717 | 146,637 | 204,918 | 1,067,114 | Net wrong additions dominate |

The uniform control restores 32,637 TP, removes 126,036 FP, removes 83,539 TP,
and adds 930,718 FP. Thus optimizing reconstruction adds 121,379 net TP
deletions and 136,396 FP additions relative to the simpler arm, while all four
episodes show lower mean FG and BG reconstruction error. Better reconstruction
is observably not a better class decision in this construction.

Query Otsu predicts 40.5--50.4% patch foreground in all four cases. In small-target
`2_21_78`, its threshold is -0.2135: it accepts some tokens whose background
reconstruction is *better* than foreground reconstruction. This exposes a real
decision failure: a between-mode query histogram split need not be a target vs
background split. It is not evidence that changing the cut alone will fix the
mechanism. The simultaneous TP deletion/FP addition in two cases also rules out
describing all damage as simple uniform enlargement.

The stored summaries do not distinguish wrong residual ranking from wrong cut,
nor do they localize changed pixels to a boundary band. A twenty-case **diagnostic**
on frozen stored fields can compare residual AUC and a GT-selected cut with the
same quantities for native, and count errors outside a fixed native-boundary
band. Those GT quantities locate the failed link; they cannot select a deployed
cut, prove normal-input recoverability, or justify another threshold sweep.
No further inference, code mutation, or remote task was started for this audit.

## Twenty-case result and retirement of the Otsu construction

Source: `runs/outputs/gpu_first20_v1/cache/{report,audits,episode_metrics}.json`
and the saved `fields/*.npz`. These twenty cases are exposed development data.
Complete 1024 class-summed native is 55.0962; F is 25.9663, gain
**-29.1298 [-39.3323,-22.4521]**, two up/eighteen down. Uniform control is
27.7714, gain -27.3248 [-37.7734,-20.2246], also two up/eighteen down.
F fold gains are -38.559/-40.175/-29.281/-10.421; batch gains are old20
-34.256, new40 -14.709, new60 -38.087, arrived100_exposed -31.592.

| Arm | Restore TP | Delete FP | Delete TP | Add FP |
|---|---:|---:|---:|---:|
| F | 402,956 | 162,119 | 230,137 | 6,815,958 |
| Uniform control | 467,259 | 134,385 | 95,708 | 6,676,486 |

F cut mean/median are -.18344/-.13520, range [-.67971,.07788]; eighteen of
twenty are negative. Predicted patch foreground mean/median are .52253/.49438,
range [.32813,.77759]. Its stored soft-field standard deviation across pixels
averages .15139 (episode range .09970--.23170); fields are not numerically
constant. Control cut mean/median are -.19258/-.14317, nineteen negative;
foreground mean/median .52727/.50818, range [.28882,.79419], mean field standard
deviation .15921. These are descriptive statistics of frozen predictions,
not extra inference or GT-dependent configuration selection.

The severe scope error persists across every batch and fold. The current
source-reconstruction-plus-query-Otsu construction is retired. No neighbor-count,
coordinate-seed, cut or residual-weight sweep follows. This failure does not
exclude frozen DINO mechanisms with a different decision objective.

## F2: fixed-correspondence inverse label inference (implemented, unrun)

The failed link being replaced is the inference from an uncalibrated source
reconstruction margin to a query-histogram target partition. The new method
instead treats query labels as variables constrained jointly by the known
reference coverage and the native continuous score. It does not use F's residual
field, its coordinate split, or Otsu.

Let `A` be a fixed sparse row-stochastic operator: each reference feature is
reconstructed from its eight nearest query features using a simplex mixture.
Normalize the **raw** native score as `(s-min(s))/max(max(s)-min(s),1e-6)` to
obtain `z0`. This prior reproduces the native pre-CRF continuous decision; it is
not the saved post-CRF native mask and is not produced by clipping the raw score.

Let `y` be reference patch coverage. Set diagonal `W` to balance reference FG/BG
coverage mass, with total mass one. When either role is absent use uniform row
mass. Fix the full query prior mass to one. Solve

```
min_{0 <= z <= 1}  (1/2) (A z - y)' W (A z - y)
                 + (1/(2 Nq)) ||z - z0||^2.
```

The label objective has a unique minimizer because of the positive native prior.
That is numerical identifiability, **not semantic identifiability**: a wrong
query object can explain the reference; the source operator is underdetermined.
Feature reconstruction error and source coverage loss do not prove correct query
identity. Unconstrained query tokens remain exactly at their native prior.

Use 32 dictionary projected-gradient steps, 128-reference chunks, then 96
box-projected label steps with the global valid Lipschitz bound
`max(A' W 1) + 1/Nq`. Report the optimization residual; finite iterations are
not claimed to be an exact solve. There is no threshold sweep, area prior,
query graph, candidate selection, RCG gate or output-mask vote. Render `z`
directly with the shared bilinear-1024/.5 finalizer. Zero extra encoder forwards.

The strong simple control uses exactly the same A, W and z0 but substitutes
the independent label-transfer objective

```
(1/2) sum_{r,i} W_r A_ri (z_i-y_r)^2 + (1/(2 Nq)) ||z-z0||^2.
```

Its exact solution is `(A' W y + z0/Nq)/(A' W 1 + 1/Nq)`. This retains the
same correspondence, all reference supervision and native prior while removing
inverse coupling. It is stronger than merely leaving the prior unchanged.
Its arithmetic is cheaper; identical FLOPs are not claimed. Both arms share
the heavy query/reference dictionary construction and all inputs/resources.

Unlike the retired harmonic graph, the main source term squares the **sum** of
query contributions, producing `A' W A` coupling, rather than summing independent
source-query disagreement penalties. Unlike B's native-conditioned correspondence,
A is fixed from features and never updated by source labels or query prior.
Agent B explicitly confirmed this objective is separate from its correspondence
updates and challenged source inverse ambiguity; that risk remains unresolved.

Current checks: static Python AST parse only. No F2 inference, model loading,
remote launch or future-label access occurred during implementation. Main runner
owns the next frozen twenty-case comparison and complete-mask evaluation.

Owned code: `src/ics/methods/reconstruction.py` and
`src/ics/methods/reconstruction_inverse.py`.
