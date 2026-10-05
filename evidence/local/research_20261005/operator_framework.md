# Addition/deletion families as a training-free segmentation framework

Owner: receiving experiment controller, 2026-10-05. The user's objective is a reusable
optimization framework whose complete segmentation output beats strong published methods.
Raw DINO matching is a common accounting origin, not the paper's contribution by itself.
The statements below are derived; empirical method success remains governed by the ledger.

## Representation and exact objective

For an origin O and any binary output S, define A = S minus O and B = O minus S.
Then S = (O union A) minus B, uniquely, with A outside O and B inside O. This covers
binary mask changes regardless of how the underlying inference method was constructed.
It does not, by itself, explain how to choose a correct edit without query labels.

Let A* and B* be available addition/deletion operator families. For selected operators,
count their unions, not sums of their individual pixel counts. State-dependent operators
may edit pixels introduced by earlier operators; record their order and recompute their
effective sets on the current mask. Fixed-origin A and B cannot overlap; sequential
operators can conflict and therefore need an explicit order/conflict rule.

For class c, with intersection I_c, union U_c and J_c = I_c/U_c, let a_g/a_b be added
true/false pixels and b_g/b_b deleted false/true pixels. Exactly,

    I'_c = I_c + a_g - b_b
    U'_c = U_c + a_b - b_g
    J'_c - J_c = (a_g - b_b + J_c*(b_g - a_b)) / (U_c + a_b - b_g)

The objective is the mean of these class ratios. Counts must be class-summed first;
one pooled purity threshold does not determine the sign of class-macro improvement.
Add-only marginal gain is (t - J_c*f)/(U_c + f). Delete-only marginal gain is
(J_c*f - t)/(U_c - f). The t/f counts are for the effective edit after previous edits.
This is the exact meaning of marginal value; separate add/delete mIoU gains are not
generally additive even though their pixel accounting is additive.

## Conditional optimality, and where it stops

For one class and calibrated probabilities p_x, consider the ratio of expected counts,
not expected realized IoU. With unrestricted pixel selection, including x changes the
fractional objective according to p_x - J*(1-p_x). At the optimal ratio J*, the rule is

    p_x/(1-p_x) > J*, equivalently p_x > J*/(1+J*).

The odds threshold is J*, not the probability threshold. Equality permits either choice.
This derivation assumes calibrated membership probabilities, a fixed expected target
area, no operator restrictions/costs and one class ratio. It is not a proof that DINO
cosines are probabilities or that this threshold optimizes class-macro IoU directly.

For a finite attainable family, a ratio solver can maximize I(S) - q*U(S), then update
q = I(S)/U(S). If its inner problem is solved globally and U(S)>0, the standard finite
fractional-programming certificate max_S(I-qU)=0 certifies the optimum of that family.
An approximate greedy inner solve does not inherit that certificate. The class-macro
sum of ratios requires its own optimization; it cannot share one universal q.

**A comparison guarantee with explicit assumptions.** Suppose the attainable family
contains a baseline B, and the label-free surrogate satisfies
abs(J_hat(S)-J(S)) <= epsilon uniformly for every attainable S. If the solver is within
eta of the best surrogate score, then

    J(S_selected) >= J(B) - 2*epsilon - eta.

Proof: J(S_selected) >= J_hat(S_selected)-epsilon >= J_hat(B)-eta-epsilon
>= J(B)-eta-2*epsilon. If its observed surrogate advantage over B exceeds 2*epsilon,
the same uniform bound certifies a strictly positive true advantage. An exact oracle
optimizer has epsilon=eta=0 and weakly dominates every included baseline; strict
superiority still requires a better attainable output. Neither inclusion nor the
decomposition guarantees such an output exists.

For each class, estimated counts in [0,U_hat] and true union at least u_min give
abs(J_hat-J) <= (delta_I+delta_U)/u_min, where the deltas bound the count errors.
A class-macro version averages the corresponding class bounds. Establishing tight
uniform bounds without query labels is unverified; this identifies the evidence and
optimization errors the framework has to reduce, rather than assuming an SOTA proof.

## Counterexamples that determine the solver

1. **Overlap can increase marginal purity and defeat one-step stopping.** Let truth
   T={0,1,2}, origin O={0}, A1={1,3,4,5}, A2={2,3,4,5}. IoU is 1/3 initially and
   remains 1/3 after either single operator, but becomes 1/2 after both. Each operator's
   standalone purity is 1/4; after A1, A2's remaining edit is one true pixel, purity 1.
   A strict positive-gain greedy rule stops before the improving pair.
2. **IoU need not have diminishing marginal returns.** With T={0}, S={0,1,2}, removing
   false pixel 1 improves IoU from 1/3 to 1/2; removing false pixel 2 afterwards improves
   it to 1. The second gain is larger. General IoU edit selection is not a monotone
   submodular maximization problem with an automatic greedy approximation theorem.
3. **Sparsity is not automatic.** If m operators each recover a different omitted true
   pixel and introduce no false pixel, using all m is strictly optimal for every m.

These are finite constructive proofs, not conclusions that a compact framework is
impossible. They require joint-move checks and a stated efficiency objective.
Exact rational execution checks are recorded in [counterexamples](operator_counterexamples.json).

## Operational interpretation

The framework should find a small, useful combination by measured marginal benefit,
overlap and cost. A defensible efficiency target is minimum inference cost among rules
within a fixed tolerance of the best attainable score, with tolerance fixed on DEV.
Report the number of active operators, forwards, wall time and memory alongside accuracy.
Compare one-step greedy with joint moves and the strongest simple same-information rule.
Use exact enumeration on a bounded family to measure an optimization gap when feasible.

Query-label accounting and DEV-fold search are diagnostics/development selection. A
deployable frozen rule must execute using the reference and allowed query evidence only.
Neither oracle purity nor a fitted-per-query rule is a zero-training inference method.
The inference gap is the key empirical question: can label-free evidence rank marginal
edits accurately enough to retain useful joint effects?

The existing600 reevaluation and DEV241 family table are complete. The matched-depth
comparison found joint2=61.5031 versus greedy2=61.6442 and direct complete selection=61.9934.
The one-step comparison's large gain mostly reflects the ability to reconstruct an
existing complete mask. A worst-fitting-fold guard adds only0.00013 over direct selection.
Reference self-calibration failed to transport to query pixels; cross-image calibration
reduced area bias but still localized badly. Combining its mass estimate with RCG ranking
improved the mask to61.0429; two-estimator guarding gave61.0736, without established gain
over direct selection or RCG. These results locate the next gap in conditional edit value,
not a general proof that the available frozen representation cannot improve.
See [family table](pipeline_verified/joint241_v3/family_table.csv) and
[guarded complete comparison](pipeline_verified/calibrated241_v4/report.json).
All timings here refer to selection from cached producer outputs, not end-to-end production.

Public benchmark evaluation must use a frozen final rule and the
official episode, resolution, CRF, encoder and metric contracts. A current 1024 subset
score cannot be subtracted from a published original-resolution score.

Protocol source checked: [FoRIS, Appendix A](https://arxiv.org/html/2609.03384v1#A1).
It specifies 1024 input and original-resolution interpolation/CRF; the exact episode
count and sampling implementation still require checking the official evaluation code.
