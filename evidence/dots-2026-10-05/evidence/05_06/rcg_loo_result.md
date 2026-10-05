# RCG self-influence and leave-one-data-term-out support

Exact removal of self-influence gives a small conditional ranking improvement over the original RCG residual. A simple weighted average of the already-solved neighboring values accounts for that improvement. The experiment does not establish a new semantic identity signal and does not justify a new mask gate.

## What was computed

The existing RCG has positive diagonal confidence A, fixed reciprocal query graph W, L=diag(W1)−W, lambda=16, and rank-edited unary u. Set M=A+16L and z=M⁻¹Au. For h_i=a_i(M⁻¹)_ii, deleting only node i's quadratic data term gives

    LOO_i = (z_i − h_i u_i)/(1−h_i)
    LOO_i − u_i = (z_i − u_i)/(1−h_i).

This follows from a rank-one inverse update. It holds with graph, confidence matrix, global ranks and all other unaries frozen. It does not delete the token's influence on graph construction or rank normalization, and is not an independent observation of its identity. Isolated nodes have h_i=1 and no such external support; they are marked undefined.

The residual sign cannot change. Exact self-influence can only reorder residual magnitudes.

## Numerical feasibility and verification

A fixed two-episode trial used no GT. Sixty-four Rademacher probes of the symmetric equivalent smoother estimated h, checked against fixed nodes. Sparse LU made every diagonal entry cheap enough to compute exactly, so the full old120 experiment used exact h rather than Hutchinson estimates.

All 120 graphs were rebuilt in one pass shared with the binary-cut experiment. Original graph edge/component/isolate counts, stored RCG normal equations and continuous-mask parity were checked. No extra model or encoder was used. Full inference fields were frozen before this evaluation; no new100 labels were read.

- Exact diagonal computation: 113.39 seconds total; entire observer: 115.81 seconds.
- Reconstructed z versus stored RCG: maximum absolute error 7.99e−7.
- Maximum relative linear-system residual: 5.53e−15.
- Direct removal and re-solve on four fixed nodes agreed with the formula within 5.6e−17 in the first feasibility case.
- 491,236 valid nodes and 284 isolated nodes; no other numerical exclusions, no exact LOO range violations.
- Hutchinson64 h error, evaluated against all exact diagonals: mean episode MAE .00926, maximum error .16671. Its LOO-value mean episode MAE was .000424. Those stochastic estimates were not used as the tested signal.

## Evaluation target

The target is true-positive versus false-positive pixels inside the actual frozen RCG 1024×1024 prediction. A score is constant within each 64×64 cell, but each cell receives its exact TP and FP pixel masses; mixed cells contribute to both classes, and score ties receive half credit. Coverage is 99.966% of TP mass and 99.924% of FP mass. Four episodes have no positive/negative pair, leaving 116 eligible episodes.

The prespecified primary signal is LOO−u, oriented higher as more graph support than the node's own unary. Its direct comparator is z−u. Other fixed comparators are u, z, original score/confidence, graph degree, and one-hop weighted mean of u. Conditional AUC compares only pairs in the same episode and same fixed score bin (10 equal-width bins); same-z bins and pure/mixed GT-cell strata are secondary diagnostics. Bootstrap resamples photograph-connected groups, 2,000 draws, seed 0. No masks were altered.

## Initial result and necessary simple control

| Existing subset | Conditional LOO residual AUC | Original residual AUC | Difference, 95% CI |
|---|---:|---:|---:|
| Old20 | .605584 | .604519 | +.001066 [−.001615, +.003654] |
| New40 within old120 | .603837 | .601150 | +.002687 [+.000164, +.005074] |
| New60 within old120 | .599921 | .596405 | +.003516 [+.001548, +.005786] |

This small positive result requires a stronger degree control. Added after the primary result, without fitting or choosing a parameter, use h0=a/(a+16 degree). The existing normal equation gives the exact identity

    u + (z−u)/(1−h0) = Wz/degree.

Thus the control is simply the weighted mean of already-solved neighboring RCG values. It requires no inverse diagonal. This mechanism check is explicitly post-result, not relabeled as a preregistered main arm.

| Existing subset | Exact LOO residual | Neighbor-z mean minus u | Exact minus simple, 95% CI |
|---|---:|---:|---:|
| Old20 | .605584 | .605019 | +.000565 [−.000073, +.001192] |
| New40 | .603837 | .604011 | −.000174 [−.000852, +.000434] |
| New60 | .599921 | .600215 | −.000293 [−.001092, +.000347] |
| Pooled120, 116 eligible | .602180 | .602287 | −.000106 [−.000586, +.000300] |

Absolute LOO support also has no advantage over this control: new60 .660026 versus .660792, difference −.000766 [−.003147, +.001183]. Both retain information from the already-smoothed z.

Without score conditioning, pooled residual AUC is .496696 versus original .496801, difference −.000105 [−.000945, +.000723]. Absolute LOO is .735325 versus z .732917, difference +.002409 [−.000278, +.005588]. The large raw AUC belongs predominantly to the existing RCG field rather than a new self-influence cue. Pure-cell and mixed-cell subsets do not establish a consistent exact-LOO advantage either.

## Decision

Keep the mathematical/numerical result and the negative mechanism test. Query graph support can rank reliability conditionally on the original confidence, but exact removal of self-feedback is unnecessary for the observed increment. It has not isolated a new target-versus-distractor identity signal. No confidence gate, masking rule, weighting rule or new decoder was built. This foreground-only diagnostic also cannot establish target recall outside RCG's predicted foreground.

## Reproduce

The portable directory contains the exact observer, portable driver, original primary evaluator, separate degree-control evaluator, and no-GT feasibility script. NumPy, SciPy and CPU PyTorch suffice. Use one BLAS/OMP thread.

1. `rcg_loo_run.py --index OLD120_INDEX --rcg-root RCG_RELEASE --out NEW_FIELDS`
2. `rcg_loo_evaluate.py --index OLD120_INDEX --fields NEW_FIELDS --rcg-root RCG_RELEASE --out primary.json`
3. `rcg_loo_degree_control_evaluate.py --index OLD120_INDEX --fields NEW_FIELDS --rcg-root RCG_RELEASE --out degree_control.json`

Optional `--unary-fields FROZEN_U_A_S_DIRECTORY` reuses those fields and rebuilds only W. The archived run used the same exact observer inside `rcg_binary_cut.py`; its helper SHA and every output SHA are recorded in `cpu_method/rcg_loo120/freeze.json`. Full per-episode metrics are in `report.json` and `degree_control_report.json`; computation checks are in `computation_audit.json`.
