# Reference neighbor dependence: a genuine interaction probe with exact controls

2026-10-10. Implemented **field/information probe only**, not a segmentation method or graph optimizer. Source: `src/ics/methods/reference_boundary_copula.py`, SHA256 `78eb677a06f663488ef3cceece7f862622dc242e22d1b11b561d1f1042df2488`. The frozen scene-kernel source remains SHA256 `8e7e068d629d744c5e2f8a5fac162e57fa78d4af99609b2318d170c5b8781d06`.

The parent reports that the completed600 scene experiment fails the universal target: scene-CRF loses FoRIS in Deep/PACO/COCO/LVIS, although scene beats its matched source kernel on PACO. This track did not independently score those predictions. The response is not a parameter change to that failed kernel. The present question is whether legal reference role-neighbor **dependence**, after holding single-endpoint evidence fixed, carries extra task information in actual query conflicts.

## Constraints from the already executed work

The existing `RCBM_REVIEW.md`, `RCBM_REVIEW_MATH.md`, `RCBM_REVIEW_HISTORY.md`, original DR02 record and retired implementation were read. Their conclusions remain binding:

- DR02's averaged unit boundary differences produce a linear query direction; its native4 result18.934724 loses prototype28.984250 and average-logistic38.913815, with199,418 added FP against3,349 added TP. This is a real but small old failure, not an experiment of the current shared-neighbor density.
- The exact old DR02 execution source is missing. The available retired code only confirms its mechanism. Its shuffled-group control belongs to DR03, not a marginal-preserving boundary endpoint shuffle.
- Current PACO600 spatial-nearest FG/BG prototype matching loses its independent full-role competition control by4.496143 points. The new probe must preserve the same endpoint competition and cannot remove difficult BG to manufacture a pairing gain.
- The prepared `reference_pair_conflict` was never run to completion. Its independent query correspondences may assign nonzero reference label conflict to identical ambiguous distributions. It is an ambiguity statistic, not validated query boundary identity.
- A fixed pair's additive endpoint cosines, or averaged difference vectors, have zero binary-label interaction. Shared latent aggregation may be non-additive; that fact alone does not show useful transfer.
- R1-G and its mathematical review show why semantic identity cannot be replaced by a numerically consistent geometry. The new probe transfers no reference-to-query coordinates, positions or corresponding scale assumption.

## A unique minimal formula

All features are the caller's same unit FP32 frozen-DINO O24 after a shared APD branch; all mask information is lawful reference coverage. No feature normalization, image encoder, view change, FoRIS pseudo-label or query GT enters the module.

Use actual horizontal and vertical reference token neighbors with **distinct** physical token IDs. With reciprocal adjacency `W_rs` and role masses `c_r^1=c_r`, `c_r^0=1-c_r`, the full source state mass is

```text
Z_yz = sum_(r,s) W_rs c_r^y c_s^z,   y,z in {0,1}.
```

The four masses are retained; their sum must equal the full directed adjacency mass. They are not converted into a forced query boundary prior. An FF edge uses actual two neighboring FG-supported tokens; a BB edge uses actual two neighboring BG-supported tokens. **Reusing a boundary's single FG atom for FF, or its single BG atom for BB, would invent self-pair evidence and is excluded.**

For each state, deterministically sample64 equal-mass quantile occurrences from its weighted physical neighbor distribution. Sampling is reference-only. Repeated occurrences represent source mass and are disclosed; they are not64 independent boundary observations. FF/BB use32 selected undirected neighbors plus their reversals. FB uses64 directed neighbors and BF is exactly its transpose. This keeps query endpoint reversal symmetric without any homologous coordinate assumption.

For state occurrence pairs `(r_k,s_k)` and query edge features `(a,b)`, define

```text
k(a,r) = exp(a dot r/.07)
J_yz(a,b) = (1/64) sum_k k(a,r_k) k(b,s_k)
F_yz(a,b) = [(1/64) sum_k k(a,r_k)] [(1/64) sum_k k(b,s_k)]
L_yz(a,b) = log J_yz(a,b) - log F_yz(a,b).
```

The common spherical-kernel normalization cancels in `J/F`. This is a dependence density ratio. Calling it a copula factor does not introduce a new mathematical model: it is the classical joint/product-of-marginals contrast. Kernel matching stays FP32; stable log-sum-exp and dependence arithmetic are FP64. Neither tiny reference differences nor `unit(rF-rB)` are computed.

The **only binary-label interaction degree of freedom** is

```text
delta = L00 + L11 - L01 - L10.
```

Subtracting each table's row mean and column mean and adding its grand mean leaves exactly

```text
L_interaction(y,z) = delta (2y-1)(2z-1)/4.
V(y,z) = -L_interaction(y,z).
```

Adding any endpoint unary, state-independent constant or row/column gauge to the original2×2 table cannot change `delta`. Thus this field cannot secretly be an additive foreground direction renamed as a relation. If a state has no observed source neighbor mass, its dependence factor is exactly1 (`L=0`), the maximum-entropy no-information extension. All four query roles remain permitted. Missing states and effective source support are explicit diagnostics.

The implemented output is an interaction field on all8,064 undirected4-neighbor query64×64 edges:

```python
probe(R, reference_coverage, Q,
      reference_grid_hw=(64, 64), query_grid_hw=(64, 64))
# edge_index[E,2]
# delta_true[E], delta_fixed_shuffle[E], delta_factorized[E] == 0
# log_copula_true[E,2,2], log_copula_fixed_shuffle[E,2,2]
# diagnostics: JSON-safe construction, state mass, marginals and cost
```

`delta>0` prefers equal roles; `delta<0` prefers opposite roles. It is **not** a probability of a boundary, a role orientation, a predicted part mask, calibrated query confidence or a complete segmentation output.

## Exact information controls

The strong factorized control is the state-conditional product density `F_yz` from the **same** endpoint occurrences and weights. Consequently every state has `L=0` and the factorized `delta` is exactly0, not an approximate shuffled average. This permits independently optimal endpoint explanations rather than comparing against a weak prototype mean.

The fixed shuffle uses seed20261010 once. Every selected occurrence has exact weight1/64, so feature/physical-ID endpoint histograms can be preserved exactly:

- FB retains its FG occurrences and permutes BG occurrences; BF is the transpose of that same shuffled bank.
- FF/BB rematch the identical endpoint stub multiset into symmetric pairs. Each left/right endpoint histogram equals its corresponding true histogram exactly.

The implementation checks equal occurrence count and both endpoint histograms independently. It reuses one canonically ordered factorized denominator for true and shuffle, eliminating arithmetic-order differences in the denominator. There is no post-hoc choice of seed. A shuffle can coincidentally keep true partners; its retained pair-multiset fraction is recorded.

Null rematching may produce same-token pairs. They are explicitly counted and never claimed as observed reference neighbors. The independent product also permits such coincidences. Therefore true-versus-null tests the observed neighbor coupling, including the exclusion of self-pairs, against independent endpoints. If gains concentrate where null self-pair mass is large, the attribution needs a further self-exclusion-matched control before claiming semantic adjacency; this first probe does not conceal that limitation.

The preserved marginal is that of the selected empirical conditional bank. The source state **total mass** comes from all real reference edges. Equal-mass quantile approximation is not claimed to preserve the full unsampled endpoint marginal exactly. Both facts are recorded separately. Budget64 was fixed to limit matching and allow a small information test; it was not selected from query scores.

## What this can add and what it cannot add

Two query candidates can have the same independent foreground/background evidence but differ in whether their adjoining feature modes co-occur in the reference under a task role transition. Joint aggregation binds the two endpoint explanations to one actual reference neighbor occurrence. Independent endpoint maxima or KDEs allow unrelated partners. This is the added observed statistic.

For a part task, the relevant hypothesis is that a true part and its adjoining non-target parent region reproduce role-conditioned neighbor feature combinations that an unrelated same-parent distractor does not. For a whole-object task, same-role adjacency can support internal semantic diversity. Both require actual source/query feature dependence to transfer; one reference context may fail to represent a new scene. No anatomical, material, object identity or part relationship is assumed just because two tokens touch.

If all FG/BG role supports alias to identical frozen feature vectors, the joint/product contrast is zero and cannot invent identity. A missing thin part cannot be recovered from mask coordinates alone. More generally, identical complete query feature neighborhoods receive identical probe observations even when their true labels differ. The current probe does not use new observations or a new layer to break this observational equivalence.

There are two different notions of information. A table's `L01-L10` can be written as an endpoint unary **in the labels**, even if it depends on both observed query features. Such contextual unary information may be useful, but it is not the non-additive label interaction tested here. The pure `delta` field deliberately removes that gauge. It also cannot by itself choose which endpoint is FG when both priors are symmetric: absolute task identity still needs a lawful single-token term. A large nonzero `delta` is insufficient success evidence.

## Minimal real-data acceptance test, before a graph solver

Use the existing fixed Deep100/PACO100 whole inputs and APD branch, and the already frozen query conflict ROIs. Query edges must be selected by these existing legal masks or by the complete grid, before GT scoring. Avoid evaluating only obvious background edges or only GT-selected boundaries.

Freeze a common standalone unary posterior `p_i(y)` for all arms. For example, the full-reference equal-role KDE at the same.07 is a classical strong control, not a new contribution. A well-defined2×2 edge prediction for a **single** edge is

```text
P_arm(y,z | a,b) = normalize[p_i(y) p_j(z)
                                exp(delta_arm(a,b)(2y-1)(2z-1)/4)].
```

This tests pure interaction at unit log-likelihood weight without graph inference or a query-GT threshold. It is not guaranteed to retain `p_i,p_j` as its posterior marginals. The endpoint-density marginals preserved by the shuffle must not be confused with posterior label marginals. If exact fixed label marginals are separately desired,2×2 iterative proportional fitting can construct a coupling with the same cross-ratio; that is a different test and is not implemented.

Seal fields/pair maps/ROIs before GT scoring. Evaluate the whole FF/FB/BF/BB state likelihood and conditional same/opposite-role performance, not FB/BF alone. For mixed query tokens, state evaluation can use the exact product of their continuous coverage masses, interpreted as independently drawn pixels from the two blocks; it is not a pure-token or physical boundary-pixel label. If a physical boundary-face evaluation is used instead, declare that different grain explicitly. Report all missing-source states, repeated-pair effective counts and null self-pair mass.

The role-relation premise requires **true** to improve actual hard-region pair prediction beyond both the fixed shuffle and product-null under the same unary. Higher reference source likelihood, larger interaction magnitude, or improving easy-background edge accuracy does not pass. A full-mask method is warranted only after this independent incremental evidence survives and has a plausible complete inference rule. If the controls explain the benefit, close this precise pairing premise rather than increasing a graph weight.

## Optimization boundary

For the pure interaction energy above,

```text
V00 + V11 - V01 - V10 = -delta.
```

Ordinary binary s-t cuts in a fixed global FG/BG encoding are valid only when every edge has `delta>=0`. Boundary-like repulsive evidence usually has the opposite sign. Sign-balanced graphs can be re-encoded by node label flips, but mixed-sign frustrated cycles need not admit that transformation. Clipping negative interactions changes the tested model and could remove its intended boundary evidence.

[Optimizing Binary MRFs via Extended Roof Duality](https://www.robots.ox.ac.uk/~vilem/QPBOPI.pdf) supports the regularity condition and explains partial optimality for non-submodular objectives. A partial QPBO labeling is not a complete mask without an explicit completion rule. Nothing in this probe claims exact full-graph inference; no Graph Cut, BP, QPBO or another optimizer has been implemented.

## Focused primary-source novelty check

[Nonparametric Scene Parsing via Label Transfer](https://people.csail.mit.edu/celiu/LabelTransfer/) already transfers annotated exemplars through dense alignment and an MRF. Joint scene organization and nonparametric label transfer are established ideas. This probe avoids homologous coordinate alignment, but that difference alone is not originality.

[MSI](https://openaccess.thecvf.com/content/ICCV2023/papers/Moon_MSI_Maximize_Support-Set_Information_for_Few-Shot_Segmentation_ICCV_2023_paper.pdf) identifies information discarded by masking away reference background and models complementary support correlations in a learned pipeline. [DOTGraph](https://openaccess.thecvf.com/content/WACV2026/papers/Biswas_DOTGraph_CLIP-Driven_Feature_Disentanglement_and_Optimal_Transport_based_Graph_Learning_WACV_2026_paper.pdf) uses support object/background graphs and cross-instance transport; [Object-level Correlation](https://openaccess.thecvf.com/content/ICCV2025/html/Wen_Object-level_Correlation_for_Few-Shot_Segmentation_ICCV_2025_paper.html) addresses hard background noise through learned query object organization. The general claims “use background,” “use a graph,” “use correlation” and “query organization prevents distractors” cannot support novelty.

The current stat is a conditional joint/product density contrast, with exact marginal-preserving pairing attribution and all unary gauges removed. Those are classical mathematical primitives applied here to a narrowly defined frozen-DINO task-role hypothesis. The focused search has not proved this exact protocol previously published, and has not proved it original. A substantial contribution requires evidence that this observed dependence is the transferable task condition missing from source marginals, a complete efficient inference method and reproducible matched benchmark gains.

## Executed synthetic checks

Only synthetic fixtures were evaluated; no cached real O24, query GT, masks, source raw mutation or graph inference was used.

- Factorized `delta` is bit-exact zero. True/fixed-shuffle conditional endpoint histograms and64-per-state budgets close exactly.
- Adding arbitrary row/column unary terms leaves `delta` unchanged within3e-14. Query endpoint reversal preserves both true and shuffle interaction within2e-14.
- A2×3 reference has identical full-role foreground/background unary densities after swapping two background feature locations, yet changing their actual FG/BG neighbor partners changes the true interaction by13.592567. The point is functional non-additivity, not performance on a realistic image. True interactions on the four synthetic query edges were[-.693146,+.693146,+.693145,-.693146]; fixed shuffle removed the two strongest boundary effects to numerical zero.
- Constant unit features produce exact zero true/shuffle interactions. A reference containing only one FG/BG neighbor has explicit missing FF/BB independence fallback and zero learned dependence; it does not manufacture same-side source pairs.
- Every returned field is finite; diagnostics serialize with `allow_nan=False`.
- Synthetic4096×1024 reference/query, CPU2 threads: .17189 seconds,8,064 query edges,247 active reference endpoints,1,035,993,088 matching MAC, state-mass closure error0 and factorized maximum absolute interaction0. This is a synthetic head observation, not cold encoder/IO/runtime evidence or a quality result.

The probe source was notified READY and is frozen pending parent review or real-data results. The former kernel remains untouched. There is still no claimed original complete Strong method or ten-dataset victory.
