# Independent reference-neighbor copula validation — 2026-10-10

The frozen probe has no material implementation defect in its pair construction, KDE ratio, interaction projection, shuffle marginal preservation or prediction-before-GT boundary. The interaction is mathematically non-additive in the binary labels. Whether it supplies useful task-specific relationship evidence beyond generic feature homogeneity remains an empirical question; this validator did not read query labels or scores.

Sources reviewed:

| Source | SHA256 |
| --- | --- |
| `src/ics/methods/reference_boundary_copula.py` | `78eb677a06f663488ef3cceece7f862622dc242e22d1b11b561d1f1042df2488` |
| `scripts/probe_reference_boundary_copula.py` | `e15f10f3d85dc0dfeb10b94f025f6287987c3ac8bfc81d8d9a80ad25692d47b1` |

Independent fixture and actual replay are `evidence/local/autonomous_20261010/scene_model_validation/copula_no_gt_validation.py`; receipt is `copula_no_gt_validation.json` in that directory. The source module, parent runner and sealed outputs were not changed.

## Mathematical result

For each selected state bank, explicit products of the spherical kernels reproduce `log J-log F` from the implementation to maximum error `3.997e-15`, using identical shared FP32 cosines. Enumerating all64×64 independent endpoint combinations gives an interaction magnitude below `6.662e-16`, confirming that factorization has zero interaction rather than merely a shuffled approximation.

`delta=L00+L11-L01-L10` is invariant to an arbitrary additive row function, column function and constant on each edge. It also removes an antisymmetric `L01=-L10` contribution exactly up to roundoff (`4.441e-16` in the independent fixture). Thus the previous averaged foreground-minus-background direction is not present in this pure label interaction. Removing that direction also discards contextual label-unary information that may depend on both observed query endpoints; the probe deliberately tests only the remaining relation coefficient.

The symmetric same-role banks and transposed cross-role banks make query endpoint reversal preserve both true and shuffled delta (`3.553e-15` maximum error). `V(y,z)=-delta(2y-1)(2z-1)/4` has binary energy difference `V00+V11-V01-V10=-delta`, so the stated attraction/submodularity sign is correct.

The full four-state source coverage masses close to twice the number of physical reference edges. Sampled occurrence weights preserve empirical conditional endpoint marginals, not the full unsampled endpoint distribution; the implementation distinguishes these quantities correctly. Missing source states use a disclosed zero log-dependence extension.

## Actual post-seal replay

After all200 fields were sealed, explicit frozen module and raw-reader loading independently replayed one DeepGlobe and one PACO-Part pair:

| Index | Episode | Saved arrays exact | Reference states | Null self-pairs |
| --- | --- | --- | --- | --- |
| 0 | `dg18-0000` | All7 | Every state64 distinct observed pairs | 0 in every state |
| 100 | `dev_s1/paco_part/0/0` | All7 | Every state64 distinct observed pairs | 1/64 in each cross-role state;0 same-role |

Both raw R/Q O24 arrays had complete input/cache-entry/payload/tensor hash verification; shape4096×1024, dtype FP32. APD selection and continuous reference coverage were independently reproduced. A second probe call matched every returned field exactly. The edge list, both log-copula tables, true/shuffle/factorized deltas and query cosine matched the saved arrays byte-exactly. The factorized delta was bit-exact zero.

For every state in both samples, occurrence budgets, physical pair IDs and left/right endpoint histograms were independently checked against the saved diagnostics. Both true/shuffle marginal histograms were exact. True physical endpoints were distinct. Source masses closed; fixed-shuffle pair multisets changed91%–98% of the occurrences in these samples, so these two nulls did not accidentally preserve most neighbor partners.

Audit hooks prohibited reading both query-mask files, archived baseline predictions and scoring outputs, and prohibited writes under the sealed probe directory. Query-GT/baseline reads and sealed-write attempts were zero. Config, seal, manifest, tasks and both records/fields retained their hashes. No encoder, new mask, CRF or graph solver was used. Replay took1.613 seconds including metadata/hashes and synthetic fixtures; it is not a cold inference latency measurement.

## What the current score can establish

The scorer's continuous patch-area products correctly describe independently drawn pixels from adjacent blocks. They are not physical shared-face boundary labels. The FoRIS-derived regions enter only after the probe field seal, and the whole-grid region remains available.

The exact factorized delta is a valid independence null. Its constant-zero rank AUC is necessarily0.5 when both same/cut masses exist. Comparing raw delta to that constant screens ordering, **not** incremental relationship evidence beyond a common lawful single-token predictor. The runner does not yet execute the common-unary conditional pair-posterior test described in the mechanism document.

True-versus-fixed-shuffle helps isolate observed neighbor coupling because source budgets and empirical endpoint marginals match. It also includes the exclusion of self-pairs in real adjacency. The PACO replay has1/64 null self-pair occurrence in each cross-role bank; the full200 support/null diagnostics should determine whether this artifact matters before attributing any gain to semantic relationships. One finite shuffle is a disclosed realization rather than the exact product density.

Query cosine screens simple generic homogeneity. Even outperforming that scalar control would not rule out generic nonlinear smoothing or a role-independent feature-neighbor statistic. In particular, easy background edges can show strong homogeneity without establishing part-versus-parent identity. The hard-region and conditional tests are therefore necessary evidence for the intended claim.

Before constructing a graph model, compare actual four-state likelihood or proper pair prediction under the same frozen lawful unary for true, shuffle and factorized interactions. If useful effects survive, inspect their dependence on source pair support, null self-pair mass and query cosine. If the controls explain the effects, this pairing premise should be closed without choosing a graph weight from these query labels. These are interpretation and next-test boundaries, not a request to change the already sealed probe.
