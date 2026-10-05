# D identity with reference-constrained extent inference

Owner: `complete_composition_sol`. Status: design and implementation frozen before
composition outcomes and inspection of the 30 newly included mini50 cases. This is one complete candidate on the existing frozen
DINOv3/reference/query inputs. The parent owns the runner, GPU dispatch and scoring.
No running D recipe is changed. All 241 episodes are DEV.

## Measured failure that the candidate addresses

First20 D is 58.3560 versus complete native 55.0962: +3.2599
[-0.1415, +5.7074], 12/7/1. Concat is 53.7011; RCG is 55.6279.
D gains by batch are old20 +0.3834, new40 -1.1421, new60 +14.6700,
arrived100_exposed +0.0919. This does not establish stable +2.

The complete-mask actions identify both useful and harmful changes:

| Episode | Native / D IoU | D change | Add TP | Delete FP | Delete TP | Add FP |
|---|---:|---:|---:|---:|---:|---:|
| `2_21_78` | 34.936 / 79.099 | +44.163 | 652 | 29,922 | 172 | 711 |
| `3_21_39` | 17.204 / 29.127 | +11.923 | 9,950 | 14,985 | 1,699 | 12,424 |
| `2_38_50` | 26.997 / 24.581 | -2.416 | 76 | 1,550 | 13,437 | 0 |
| `1_5_1` | 55.521 / 51.706 | -3.814 | 870 | 2,133 | 649 | 26,643 |

Sources: `runs/outputs/gpu_first20_v1/forward/{report,episode_metrics}.json`.
These are GT scoring diagnostics, not information used by inference. The evidence
shows correct rejection, missed extent, and false expansion. It does not establish
which internal FoRIS stage caused them or prove that graph connections recognize
the lost target. D's representation changes the complete correspondence/clustering
pipeline; the proposed graph must preserve those useful identity changes while
testing whether coherent query/reference structure improves extent.

Existing RCG already gives 61.0197 versus native 59.0748 on DEV241:
+1.9448 [0.9831, 2.9136], 134/104/3. Its last 21 DEV cases regress -0.6235.
This supports testing a structured inference step and demands RCG as a control;
it does not establish that the proposed cross-image edges work. B's conditioned
correspondence revision and E's joint slots show that repeated compatibility
feedback or sharpened identity can destroy query extent. This candidate therefore
uses one strictly convex anchored inference, without pseudo-label feedback.

## Complete inference in ten lines

1. Input the frozen complete D raw part-4 field, existing debiased last-layer q/r, and reference coverage.
2. Normalize D's field once with the exact FoRIS min/max epsilon to obtain unary s.
3. Keep the real cached last-layer representation for extent geometry; do not fake transition tensors.
4. Build RCG's reciprocal adaptive 20-NN query graph W, globally scaled to mean degree one.
5. Build reciprocal adaptive 10-NN cross-image edges C to reliable reference FG/BG tokens.
6. Keep reference coverage as fixed soft labels; globally scale C to mean query cross-degree one.
7. Solve sum a(z-s)^2 + 16 sum W(z_i-z_j)^2 + sum C(z_i-c_j)^2, with RCG confidence a.
8. Query uncertainty, local extent consistency and source identity jointly permit additions and deletions.
9. Bilinearly render z > 0.5 without re-minmax, then run the same source FoRIS CRF for a complete 1024 mask.
10. Compare D original, original RCG, D-RCG, identical W without C, and identical W/C with mean source labels.

Fixed source labels make the cross-edge energy exactly a per-query
degree-weighted source-label unary. It is not an independently inferred source
state or higher-order correspondence mechanism. The sparse normal equation is
`[diag(a + C1) + 16(diag(W1)-W)] z = a*s + C*c`.
Positive a makes its solution unique. Its maximum principle bounds z by the
range of D unary and reference labels. Neither fact guarantees semantic
correctness, correct area or IoU gain. In particular, confident wrong D regions
and background-biased source matches can still defeat the candidate.

## Parameters and alternatives

Query k=20, cross k=10, source purity=0.9, graph lambda=16, confidence floor=0.1,
CG rtol=1e-7/atol=1e-9/maxiter=300 are inherited from frozen RCG. The new reference
unary coefficient is one, fixed before any result; no scan is scheduled. Reciprocal
cross-image weights use the geometric mean of both directed adaptive weights.
Both source roles retain their actual coverage; no prescribed query foreground
mass is imposed. If a role has no pure token, use its most extreme coverage token
and retain its soft coverage label, recording the fallback.

The no-reference control uses exactly W, a and lambda, but omits C from both sides
of the normal equation. The mean-reference control uses exactly W and C but
replaces every source label with the cross-edge-weighted global source mean. This
preserves the injected global label mass while removing its appearance assignment.
This is one global scalar for the whole image; replacing labels by each query's
own cross-edge-weighted label mean would be algebraically identical to the primary
and is explicitly not an independent control.
The D-RCG control applies the unmodified RCG equations to D's raw score, using
the same cached q/r/cov; composition finalization adds the same source CRF as the
candidate. Original RCG remains its own fixed bilinear-only complete baseline.
The D-unary control is a renderer/native-CRF fidelity audit, not a new method.

If graph-only D or D-RCG is equally good, the reference-cross-edge mechanism is
unnecessary and must not receive credit. Access to D's two-layer readout is
shared by every composition arm. These controls isolate extent inference, while
the original concat and delta-only D comparisons address representation resources.

## Runner contract and first evidence

The parent should overlay stored D fields on an unchanged source manifest, keeping
the original complete D predictions. `composition.predict(q,r,cov,d_raw,...)`
requires `extras['score_origin']='multilayer.transition.part4.raw'`; it rejects a
missing origin instead of silently applying to native score. The regular q/r inputs
must be the existing `feature_export` cache, not raw or fabricated intermediate
layers. The renderer/CRF needs the actual transformed query RGB and existing host.
No extra encoder forward or new natural image is required.

The latest user steering specifies one unified mini50 for screening and optimization,
followed by DEV241. The candidate is frozen before new mini50 query GT is read.
Metadata inspection shows 20/50 cases overlap the already analyzed first20; the
other 30 have not been inspected by this worker. Folds contain 13/13/12/12 cases;
batches contain 10 old20, 10 new40, 10 new60 and 20 arrived100_exposed cases.
First20 provides the failure evidence above and is not an additional selection run.
Mini50 tests the complete method with class-summed I/U,
paired 2,000 photo-connected RandomState(0) bootstrap draws, folds/batches and four
actions relative to both native and D. Freeze the predictions before reading query
GT. Report all arms; no per-episode GT selector or final-mask fusion is allowed.
Mini50 is development screening, and the later DEV241 is also development. Its
selection after inspecting mini50 must not be called independent confirmation.

No composition complete-mask result has yet been measured. Raw D layers from the
ongoing full run would enable a separately declared true-transition graph control;
they are not available in first20 packets and must not be substituted by q/r.

## Local verification receipt

The module ran on a four-token synthetic cache on CPU, with finite bounded fields
for primary, no-reference and global-mean controls. On a separate three-node
system, CG matched an independently constructed dense solution within 1e-7;
the reference FG/BG anchors moved uncertain queries in opposite directions and
the relative residual was 4.95e-15. Missing D provenance and absent source roles
were rejected. A source-finalizer identity stub verified only 1024 boolean output
shape. These checks used no query truth, no real feature matrix product, model
forward or source CRF. They are numerical/contract checks, not efficacy evidence.

Pending parent verification: same D-unary/native-CRF replay equals frozen original
D mask; real CUDA sparse/dense operations; real source CRF; fixed mini50 complete
masks with all controls. The parent was given the exact score overlay and finalizer
contract. The worker started no remote process and edited only its two owned files.
