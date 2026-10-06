# Theory of selecting addition/deletion compositions

Owner: Root. Started 2026-10-06 02:46 UTC. This note is a derivation, not a new
experiment. Existing experiments and their successors are held by the user's request.
No new model calls, data evaluation, or parameter search are performed for this note.

## 1. The deliverable and the two different selection problems

The deliverable is a complete mask rule in the stated one-reference/frozen-backbone
information budget, with useful improvement over complete FoRIS and the strongest
simple same-information alternative. A/B is a representation and a search space;
its contribution must come from an effective selection rule and a complete result.

There are two selection problems:

* **Fixed recipe:** select one recipe on development data, freeze it, and execute
  exactly that recipe on every new query. Query GT is absent at inference. Development
  labels select a hyperparameter; no new foreground posterior need be learned.
* **Adaptive recipe:** choose a different recipe from reference/query evidence on each
  new query. This requires an identifiable estimate of conditional edit utility.
  A GT development optimizer alone does not supply that estimator.

These are not interchangeable. A globally development-selected composition is an
executable fixed method, even though its development maximum is not fresh confirmation.
Fitting an eta table to query labels is additional supervised calibration and must be
named as such; freezing that table does not remove the fitting.

## 2. Canonical effects and the actual attainable family

For origin O and independently produced complete masks S_i, define

    A_i = S_i \ O,       B_i = O \ S_i.

For selected addition producers L and deletion producers D, canonical fixed-origin
composition is

    C = (O union union_{i in L} A_i) \ union_{j in D} B_j
      = (~O & OR_{i in L} S_i) | (O & AND_{j in D} S_j).

Empty OR is false and empty AND is true. This is an explicit Boolean inference rule.
Each source S_i is included by taking L=D={i}; O itself is included by taking both empty.
Thus a *true-score* exact maximum of this finite family weakly dominates every included
complete source on the same data. Strict superiority requires a better attainable mask.

Include O and every allowed alternative origin in the complete-source bank. If every
member assigns the same bit to a pixel, every canonical composition
preserves that bit. Only producer-disagreement pixels can change. With class GT mass
G_c, unanimous-negative FN mass N_c and unanimous-positive FP mass F_c, even a relaxed
oracle correcting every disagreement pixel has IoU (G_c-N_c)/(G_c+F_c). Therefore this
is an upper bound on the complete-source family. Its remaining capacity loss is not
proof of missing information in the frozen encoder, only of unavailable edits in that bank.

Canonical addition and deletion domains are disjoint. Cross-side pixel conflicts do not
exist in this representation; overlap within each side and IoU denominators still create
interactions. A deletion that removes a pixel introduced by a previous addition is a
state-dependent operator and requires an explicit order. Its result cannot silently be
treated as a fixed-origin proposal.

For unrestricted A and B, any binary mask can be represented relative to any origin.
For a finite producer library, the attainable compositions depend on O. Enumerating
several origins enlarges the family; algebraic universality does not imply that this
finite family covers all training-free methods or all Boolean functions of the sources.

Example: strict C12 is the executable rule

    C12 = (V & G) | (~V & F),

where V=fine16, G=scalar graft and F=fine64. It keeps G's decision inside V and F's
decision outside V. It is a fixed conditional routing rule, with no query-GT branch.

## 3. Exact one-class value, and the complementarity term

For a class, let baseline intersection/union be I,U, with J=I/U and U>0.
An addition contributes t true and f false pixels; a deletion removes d true and e
false pixels. Use effective unions of edits, never sums that double-count overlap.

    J(C) = (I+t-d)/(U+f-e)
    Delta(C) = [t-d + J(e-f)]/(U+f-e).

Define W_A=t-Jf and W_B=Je-d. Then

    Delta(A) = W_A/(U+f)
    Delta(B) = W_B/(U-e)
    Delta(A+B) = (W_A+W_B)/(U+f-e).

Provided all denominators are positive, subtraction gives exactly

    Delta(A+B)-Delta(A)-Delta(B)
      = W_A e/[(U+f-e)(U+f)]
        - W_B f/[(U+f-e)(U-e)].

This separates complementarity from mere pixel overlap. Deletion of false positives
can increase the value of additions through a smaller union. False-positive additions
can dilute deletion gains. The sign is conditional; neither superadditivity nor
subadditivity is universal. For a single class, both W_A and W_B positive guarantees
the combined gain positive. It does not guarantee that the combined gain exceeds
their summed separate gains.

For the same fixed pair A,D, let J_A and J_D be their separate final IoUs. Exact
cross-marginal conditions are

    J(A+D)>J(A) iff J_A e>d,
    J(A+D)>J(D) iff t>J_D f.

The first says deletion must remain useful after addition; the second says addition
must remain useful after deletion. If A,D are individually the optimal single-side
choices and both inequalities hold, the joint family strictly beats both single-side
optima. These are sufficient conditions for strict joint-family improvement, not
universal properties of the available proposals. With f=0 and t>0 the second is automatic;
with d=0 and e>0 the first is automatic. Impure edits can fail the cross-marginal test.

### Class-macro counterexample: two useful parts can hurt together

Class 1: I=50,U=100, truth has 50 pixels and O has 50 TP plus 50 FP. A adds 100 FP;
B deletes the original 50 FP. A gain=-1/4, B gain=+1/2, joint gain=-1/6.

Class 2: I=50,U=100, truth has 98 pixels and O has 50 TP plus 2 FP. A adds 48 TP;
B deletes 49 original TP. A gain=+12/25, B gain=-49/100, joint gain=-1/100.

The equal-class macro gains are

    A: 23/200 = +0.115
    B:  1/200 = +0.005
    A+B: -53/600 = -0.088333...

All masks/counts are feasible and both operators separately improve class-macro IoU.
Their combination makes both class IoUs worse than the baseline. A class-wise vector
of counts/values is necessary; one pooled purity or two positive macro gains is insufficient.

## 4. Same-side overlap determines the marginal proposal value

For adding sets A1,A2, let W(A)=TP(A)-J FP(A), at the fixed baseline J.
The numerator is additive as a signed set measure:

    W(A1 union A2) = W(A1)+W(A2)-W(A1 intersection A2).

When their intersection is mostly false, its W is negative: union pays those false
pixels once rather than twice. This can create complementarity even if both individual
proposals look poor. When the intersection is mostly true, double-counted benefits
disappear. At the updated current mask, both effective edit A2\A1 and J change.

Consequently, marginal purity may rise or fall as more operators are selected. Neither
monotone diminishing returns nor a rule to stop after one nonpositive move is valid
without additional assumptions. Pure false-positive deletion alone also has increasing
IoU increments: 1/3 -> 1/2 -> 1 when removing two false pixels from one true pixel.

At fixed q, signed-coverage interactions satisfy

    Delta_i Delta_j W = -sum_{x in A_i intersection A_j, not yet covered} w_q(x).

Positive-weight overlaps give diminishing returns; negative-weight overlaps give
increasing returns. If every covered weight is nonnegative, the residual is monotone
submodular. General TP-minus-q-FP coverage is a difference of two monotone submodular
functions. The fixed-q residual separates across canonical add/delete domains if there
is no shared budget/cost constraint, although each within-side signed-coverage problem
can remain combinatorial. Fractional IoU and a shared dependency cost restore coupling.

## 5. Probability objectives must be named correctly

For membership probabilities p_x, expected-count ratio is

    R(S) = sum_{x in S} p_x / [|S| + sum_{x not in S} p_x].

No pixel independence is needed to compute these expected counts. For an unrestricted
single-class family, with optimum q*, including x has residual coefficient

    p_x - q*(1-p_x).

The optimal boundary is p_x > q*/(1+q*), or odds p_x/(1-p_x)>q*. It is not p_x>q*.
This is a theorem about R and calibrated probabilities, not about raw DINO cosines,
finite proposal constraints, expected realized IoU, or class-macro optimization.

Expected realized IoU is E[|S intersection T|/|S union T|], generally different from R.
It can depend on the joint label law even with the same membership marginals.

Counterexample with nonempty truth in every outcome: three pixels, p=(1,1/2,2/5).
Pixel 0 is always true. Let t=P(y1=y2=1). Then

    EIoU({0})   = 11/20 + t/3
    EIoU({0,1}) = 41/60 - t/6
    EIoU({0,1,2}) = 19/30.

For t=0, {0,1} is best at 41/60. For t=2/5, {0} is best at 41/60.
The same three marginals therefore do not identify the best expected-IoU mask.
The expected-count ratio instead prefers all three: 19/30 > 5/8 > 10/19.

The benchmark aggregates I and U by class over many episodes before dividing. A
population ratio of class expected counts is the appropriate population analogue of
that benchmark under a stated sampling law; it is not mean per-episode IoU. Class
aggregation and sampling assumptions must remain explicit.

## 6. Solver certificates are conditional on the objective being solved

For one ratio I(C)/U(C) over finite H, U(C)>0, define

    F(q)=max_{C in H} [I(C)-q U(C)].

F(q*)=0 exactly iff q*=max_H I/U. An exact residual maximizer with q updated to its
ratio yields a finite fractional-programming solve. A greedy residual search has no
global certificate. If F(q)>=0 and U>=u_min, the remaining ratio gap is <=F(q)/u_min.

For class-macro IoU, the objective is a *sum of ratios*. Using one q is wrong. Using
each incumbent q_c and globally maximizing the sum of unnormalized residuals is also
not automatically a macro certificate: the actual gain divides each residual by its
own candidate U_c. The class-macro finite family can be solved directly from exact
per-class sufficient statistics; a fractional shortcut needs a separate proof.

## 7. Compression of the finite family

Each pixel has membership signature b=(S_1(x),...,S_m(x)). Every fixed Boolean recipe
is constant on a signature. For class c, retain a foreground mass t_cb and total mass
n_cb for each observed b. Any recipe h then has

    I_c(h) = sum_b t_cb h(b)
    U_c(h) = sum_b t_cb + sum_b (n_cb-t_cb) h(b).

This is an exact sufficient representation, not an approximation. A recipe is evaluated
on at most 2^m signatures, without a new image encoder call. Enumerating all allowed
recipes is legitimate development optimization; GT supplies t_cb and therefore this
is not a no-GT per-query value predictor.

The previous 218,103,808 count is a count of syntactic recipes. Equal masks/signature
functions can be merged. An exact maximizer certifies only the specified finite family.
Repeated model evaluation is unnecessary when those complete source masks already exist.

## 8. Paired uncertainty should focus on edit regions

For baseline B and candidate C, define dI=I_C-I_B and dU=U_C-U_B. Exactly,

    gain(C,B) = [dI-J_B dU]/U_C.

Suppose expected-count estimates are coherent and baseline errors satisfy
|J_B-Jhat_B|<=eps_J, |U_B-Uhat_B|<=eps_U. Let

    E_D = sum_{x in B symmetric_difference C} |p_x-phat_x|.

Then |dI-dIhat|<=E_D and |dU-dUhat|<=E_D. Let ghat denote the estimated gain and
U_C>=u_min>0. By subtracting the two gain fractions,

    |g-ghat| <= [(1+J_B) E_D + eps_J |dUhat|
                 + |ghat|(eps_U+E_D)] / u_min.

Proof: numerator error is bounded by (1+J_B)E_D+eps_J|dUhat|; union error by
eps_U+E_D; |Nhat/Uhat_C|=|ghat|. This is a deterministic sensitivity bound.
It shows why disagreement-specific reliability can be more useful than full-image
absolute-score prediction. It does not make E_D or the baseline error bounds observable
without an assumption linking available evidence to membership.

Common agreement pixels do not disappear from uncertainty entirely: they influence
J_B and U_B. Declaring them irrelevant would be another incorrect shortcut.

## 9. A conditional robust selection rule

If simultaneous membership/error intervals are supplied by a justified evidence model,
form lower bounds L(C,B) on paired gains for each strong baseline B. Include all baselines
in the feasible family. Select the lowest shared inference cost among recipes meeting
a declared performance target and certified gain constraints; fall back to a baseline
when no candidate meets those constraints.

This supplies a complete conditional optimization rule, not a new validated evidence
model. Correlated prior votes, reference self-consistency and graph smoothness cannot
be declared calibrated probabilities merely because they are computable.

## 10. Efficiency and 'less is more'

Count shared inference work in a dependency DAG. For selected sources K, cost is the
cost of union_{k in K} dependencies(k), plus required readout/combination work. Three
outer masks can share an encoder; one composite mask can hide many graph solves.
Outer producer count is not a runtime metric.

For adaptive selection, include every dependency computed to *decide* the recipe,
even if its producer is ultimately rejected. Selected-source closure alone describes
a frozen recipe's production cost. An adaptive selector that first reads all thirteen
complete masks cannot claim to pay only for its three selected masks.

A precise efficiency formulation is

    minimize cost(h), subject to score(h) >= max_H score - tolerance,

or maximize score subject to cost<=budget. The tolerance/budget is explicit. There is
no unconditional sparsity theorem: m disjoint all-true additions require all m operators
for the strict optimum. Compact recipes follow measured redundancy, a cost constraint,
or a chosen tolerance; they do not follow from the A/B decomposition alone.

## 11. What theory can currently predict

The algebra predicts exact interactions once class-wise true or justified estimated
edit masses are supplied. The finite solver predicts the best recipe *for that supplied
objective and family*. It cannot infer unknown semantic correctness from overlap alone.

The current deployable candidate C12 already exists; the unanswered result is stable
superiority over the strongest simple control on fresh data. Existing held-fold strict
result is 62.964699, versus graft +0.120162 [-0.103035,0.319822], unresolved. No theorem
in this note turns that into a success or an original-resolution public SOTA claim.

Next theoretical work in this hour: independently check the proofs, characterize the
partial-identification assumptions, and state the single defensible framework claim.
No experiments resume automatically.

## 12. Population threshold geometry and identification are separate

For a score threshold t, let T(t) be foreground survival and F(t) background survival.
Under known class-conditional score laws invariant across the query population,

    J(t;pi) = pi T(t)/[pi+(1-pi)F(t)].

The distribution of true score values and false score values plus foreground prevalence
determine this population one-query ratio. At an interior optimum,

    f1(t)/f0(t) = [(1-pi)/pi] J(t;pi).

This is a stationary likelihood-ratio boundary. Monotone likelihood ratio is sufficient
for a raw-score threshold to equal the unrestricted Bayes decision. It is not necessary
for weak monotonicity of the optimum *within a fixed nested threshold family*: comparing
two cutoffs gives an affine single-crossing inequality in r=(1-pi)/pi, so a unique optimal
threshold cannot move upward as pi increases, when the two conditional laws stay fixed.
Different score calibrations/laws cannot be compared that way. For the optimal
likelihood-ratio cutoff, write a=pi/(1-pi); its level is max_t T(t)/(a+F(t)), which cannot
increase as a increases. Smooth cutoff sensitivity needs a nonflat local likelihood-ratio
slope, as detailed in the occupancy note.

For the actual class-summed metric, independently maximizing each image's IoU is not
the global class optimum. A class threshold-family oracle instead solves

    max_{t_e} sum_{e:c_e=c} I_e(t_e) / sum_{e:c_e=c} U_e(t_e).

At a given class ratio q_c, each episode independently maximizes I_e(t)-q_c U_e(t).
Exact inner maximization plus ratio updates gives the finite-family class certificate.
Thus a per-image-oracle policy's aggregate score is generally an achieved oracle score,
not automatically the ceiling of the threshold family under class-summed mIoU.

Known F1,F0 identify pi from G=(1-pi)F0+pi F1 whenever F1 differs from F0. In total
variation, TV(G_pi,G_pi')=|pi-pi'| TV(F1,F0). Small foreground/background separation
makes the inverse ill conditioned. If all admissible mixture fits share the same fixed
F1,F0 and lie within epsilon of G, their pi identified-set diameter is at most
2epsilon/TV(F1,F0). Varying component laws require an explicit additional drift allowance.

Approximate distribution invariance is useful for a small pi only if its uncertainty
is small compared with pi*TV(F1,F0). A globally small distribution error can still
overwhelm a tiny foreground population. Estimator failures alone do not prove that
pi is not identifiable from the full allowed images/features.

## 13. Gains and oracle gaps cannot be added across cohorts

On one frozen cohort, an actual nested chain has the exact telescoping identity

    J(final)-J(FoRIS) = [J(RCG)-J(FoRIS)]
                     + [J(fine)-J(RCG)]
                     + [J(final)-J(fine)].

The signed increments include interactions through their current baselines. Numbers
measured on different cohorts or against different baseline/finalizer contracts do not
form this decomposition. The user's +1.41, +0.4, +5.07 and +8.91 cannot simply be summed.

The supplied true-pi formula improvement is an achieved privileged-policy result,
not a theorem that every identifiable pi estimator can recover that many points.
The supplied per-image threshold oracle is a capacity diagnostic, not a numerical
lower bound on unavoidable error or a universal ceiling for finer representations.
No positive number of points is yet proved irreducibly unidentifiable on this dataset.
