# The task as an optimisation problem: the optimum, and what each pipeline does about it

Owner: Claude Code, 2026-10-05. **Derivation and source reading only; no new measurement.** Each statement is
tagged *derived* (algebra), *read* (from the public source), *measured* (an existing record, cited) or *predicted*
(untested, with the prepared readout that decides it). The edit origin is the model's own selection,
`model.raw_nn` ([PLAN](../../../docs/research/PLAN.md)); INSID3, FoRIS and RCG are rows built on it.

## 1. The objective, exactly

Class mIoU is the mean over classes of `J_c = I_c / U_c`, with I and U summed over the class's episodes. Write a
prediction as an edit of the origin O: `S = (O ∪ A) \ D`, additions A outside O, deletions D inside O. With
a_g, a_b the added true and false pixels and d_g, d_b the deleted false and true pixels (*derived*):

    I = I_O + a_g - d_b        U = U_O + a_b - d_g        J' - J = (ΔI - J·ΔU) / (U + ΔU)

- **The gain is positive exactly when `a_g + J·d_g > d_b + J·a_b`.** Per pixel, in units of 1/U_c: an added true
  pixel is worth 1, a deleted false pixel J, a deleted true pixel costs 1, an added false pixel costs J.
- **The two actions are not symmetric.** At J = 0.6 one recovered pixel is worth 1.67 removed false pixels, and a
  wrong addition costs 0.6 of a wrong deletion. An addition pays above purity J/(1+J) (0.375), a deletion above
  1/(1+J) (0.625).
- **Additions and deletions are coupled through one number per class, J.** The sets are disjoint, so each can be
  optimised given J and J updated. As deletions raise J the bar for additions rises (0.31 at J = 0.45, 0.38 at
  0.62, 0.41 at 0.70) and the bar for deletions falls. This is the precise sense in which the task is two
  problems plus their combination.
- **Pixels are priced by 1/U_c.** A class with a small total union weighs each pixel more; pooled pixel purities
  are not the class-macro quantity. Counts are absolute: a false region costs the same wherever in the class it lies.

## 2. The optimal decision

Let p(x) be a calibrated probability that pixel x is target given all label-free evidence. For one class, the set
maximising E[I]/E[U] is (*derived*, Dinkelbach's argument for a ratio of linear functions)

    S* = { x : p(x) / (1 - p(x)) > J* },   J* the value S* attains (a fixed point, reached by iteration)

- The probability cut J*/(1+J*) is **below one half for every J* < 1**: the IoU-optimal mask is larger than the
  most-probable mask. Cutting at one half, or at the middle of a per-image range, is not this rule. (Consistent
  with [CLAIM](../../../docs/research/CLAIM.md): pixelwise Bayes classification is not IoU optimisation.)
- With one cut shared by all classes, the optimal cut is an average of the J_c weighted by each class's pixel
  density at the margin divided by U_c, so it leans to small and hard classes.
- For a region decided as a whole, the same rule applies to its expected true fraction: a group that is 40% target
  is worth including at J = 0.6; pixel-level deletion inside it then adds more.
- Scope: this is the ratio of expected sums, the standard surrogate; it equals expected IoU as counts grow under
  weak dependence. It needs p calibrated, which is the actual difficulty (section 4).

**Odds form.** `odds(x) = prior odds of x's region × product of likelihood ratios of the evidence` (exact under
conditional independence). So a pixel is included when its combined likelihood ratio exceeds `J(1-π)/π`, π being
the target fraction of the region it is proposed from (*derived*).

- A **proposal** raises π; an **auxiliary** supplies the likelihood ratio. That is A + auxiliary. A proposer of
  purity π below break-even is rescued by an auxiliary exactly when the auxiliary's likelihood ratio at its
  confident end exceeds J(1-π)/π (1.8 for π = 0.25, J = 0.6). The criterion is the top-end ratio, not AUC; a
  within-region AUC of 0.48 to 0.52 (*measured*, Astra's five signals) is a ratio of 1 everywhere, i.e. no auxiliary.
- For complete FoRIS (*measured*, pooled, `edit_budget.json`): 2.7% of the pixels outside its mask are target, so
  an addition drawn from all of them needs a ratio above 22; 71.7% inside are target, so a deletion needs a ratio
  below 0.24. **This is why every measured mechanism earns by deleting and none adds above break-even**
  ([edit budget](edit_budget.md)). Additions have to come from regions whose prior is already high (the same
  group as kept tokens, their neighbours), not from the whole background. The same two numbers for the origin are
  in the stage ledger once it runs.

## 3. The two public pipelines in these terms (*read* from `models/insid3.py`, `models/foris.py`)

| Part | INSID3 | FoRIS | Role |
|---|---|---|---|
| Space | always projects off the positional basis for cross-image terms; clusters the raw tokens | projects both images unless the reference mean already matches the query mean (0.8) | representation |
| Evidence | similarity to the reference foreground mean; nearest-token vote | clustered foreground prototypes (log-sum-exp) minus 0.55 × orthogonalised hard-negative background mean; nearest-token vote | likelihood terms |
| Structure | one average-linkage partition (τ = 0.6); cluster mean of the evidence × similarity to the seed cluster × share of candidates | partition on features + colour + position → seed-cluster prior; second partition → per-cluster boost or suppression | smoothing, prior |
| Combination | product, exponents 1 | sum with hand weights 1, -0.55, 0.20, 0.25, penalty, boost | log-odds weights |
| Decision | absolute cut 0.2, token mask upsampled | per-image min-max, cut 0.5 of the upsampled response | cut |
| Finish | CRF in a 10-pixel band | same | boundary |

- Taking logarithms of INSID3's product makes it a weighted sum, as FoRIS is. **Both are points of one family:
  weights over a common set of terms, a scaling, a cut and a renderer.** `stage_bank.FAMILY` and `ANCHORS` define
  it; the origin is a third point. The best point of the family on the fitting folds is at least as good as both
  there by construction (*derived*); whether it holds on the read fold is the measurement.
- Neither cut is the rule of section 2. FoRIS's is relative: every image gets a non-empty mask whose extent is set
  by its largest and smallest response, while pixel values are absolute (section 1). INSID3's is absolute and can
  return nothing.
- The FoRIS stage-1 feature gate multiplies each query token by a positive scalar that the later per-token
  normalisation removes; at the public defaults it does not reach the output (*read*; the parity audit checks it).
- The FoRIS vote normalises the reference tokens along image height, not along channels (*read*; reproduced as is).
- RCG minimises `Σ a_i (z_i - y_i)² + λ zᵀLz`: the posterior mean under a graph prior, i.e. a smoother of the same
  response (*read*). Replacing its reference guide by the plain foreground mean moved the 220-case score by 0.26
  [-0.12, 0.65] (reported in the Astra chat; not checked here): the gain sits in the smoother and the deletion
  budget, not in the particular cross-image evidence.

## 4. Where the optimum should differ from both, by expected leverage (*predicted*)

Readouts are in `outputs/<run>/report.json` of [run_stage_bank.py](../../../scripts/run_stage_bank.py).

1. **The cut.** The first-order value of the pixels at each response level, `Σ_c (true - J_c·false)/U_c`, is
   positive up to the optimal cut and negative beyond it. If it changes sign away from 0.5 (FoRIS) or 0.2
   (INSID3), the cut alone is leaving score. Known: complete FoRIS carries 9.27M false against 5.92M missed pixels
   (*measured*, pooled, `edit_budget.json`), and a truth-area cut of its ranking was worth +9.09 at patch level
   (*measured*, dots 05, privileged). Readouts: `value_by_level`, the threshold lines, the fixed-point rule,
   absolute against per-image scaling.
2. **The weights.** Terms that repeat the same evidence should be down-weighted; a term whose removal does not
   move the score is not a contribution. Readouts: one-at-a-time lines through each anchor; `foris.pre-<term>` in
   the ledger with its added and deleted counts.
3. **Evidence that is independent of reference matching.** More variants of cross-image similarity add little
   once one is in (the Astra result above). What multiplies the odds is evidence of another kind: the query's own
   grouping, spatial connection to kept tokens, colour. Readouts: mixtures of the two pipelines' terms; then the
   [evidence sweep](aux_evidence.md) with the sealed origin.
4. **Additions from high-prior regions.** By section 2 an adder is a region definition with high π plus any
   auxiliary above the required ratio. Readouts: `additions alone` of every stage in the ledger; the sweep's
   addition rows restricted to proposals.
5. **Below token resolution.** With 16-pixel tokens and 1/U_c pricing, small classes are decided at the boundary
   (token-grid ceiling 93.2, *measured*, INSID3 protocol). Not addressed by this batch.

## 5. What this is not

No number here is new. Section 4 is a list of predictions. The family search chooses a handful of constants on
three folds of development classes and reads the fourth, as the public constants were chosen by their authors; the
chosen point is reported for every fold so that it is not mistaken for a fitted readout. Family masks have no CRF.

## 6. Outcome of the first run (stage241_v1, *measured*, development data)

Predictions 1 and 2 of section 4 are not supported: the value of FoRIS's response levels changes sign between 0.525
and 0.575, next to its cut of 0.5; the nested family reads 57.35 against 59.07 for complete FoRIS (-1.73 [-2.80, +0.05]),
threshold only -0.29 [-1.24, +0.47]. From the model origin FoRIS's additions are 55% true and its deletions 88% false,
so both actions earn there (48.78 and 52.77 alone, from 42.90). Numbers: [ledger](../RESULTS.md), [report](stage241_v1/report.md).
