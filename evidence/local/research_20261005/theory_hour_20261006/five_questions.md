# Five-question theory synthesis

Owner: Root. Theory-only task starting 2026-10-06 02:46 UTC. No experiment is
authorized by this synthesis. The user's nine supplied observations are taken as
given; this task does not replace their versions or claim new scores.

Status labels: **Exact** means a definition/algebraic theorem; **Conditional**
means a theorem under the named model; **Conjecture** means an empirical mechanism
interpretation that still needs its stated measurement.

## 1. Joint mathematical structure

**Conclusion.** Finite A/B choice is a discrete sum-of-ratios optimization with
coverage, conflict and computation constraints. One-class fractional residuals
are additive in effective pixels, but not generally additive or submodular in
selected overlapping operators. Exact family expansion weakly dominates included
single sources; strict superiority requires complementary, decision-useful errors.

For deletion priority, define effective additions and deletions by

    a = (union selected A \ O) \ union selected D,
    d = O intersection union selected D.

The output is O plus a minus d. The four class-summed edit counts give exactly

    maximize (1/C) sum_c (I0c+aTPc-dTPc)/(U0c+aFPc-dFPc)
    subject to dependency work / memory / input constraints.

The denominator is the edited union. The user's +/-1, +/-J prices are exact
numerator prices at the baseline J, not four finite-edit gains with a fixed old U.

Fixed-q one-class residual prices foreground by +1 and background by -q. Positive
weighted coverage is submodular; signed coverage is a difference of submodular
coverage functions. A two-pixel add/delete example already violates submodularity.
The actual category mean cannot be reduced to one pooled ratio or one universal q.

For a fixed one-class canonical pair with disjoint origin-relative domains, joint beats add alone iff J_A*deletedFP>
deletedTP; joint beats delete alone iff addedTP>J_D*addedFP. Applying both to the
best single-side choices is a sufficient strict-joint advantage condition. Two
positive standalone macro gains do not suffice: the feasible two-class counterexample
in [main derivation](main_theory.md) has A=+.115,D=+.005,joint=-.088333.

**Conditional zero-advantage regime.** In one class, if all complete sources are
nested thresholds of one score, the true posterior is nondecreasing in that score,
and the target is the population ratio of expected counts without a cost constraint,
the best canonical joint composition equals the best available complete single.
A joint mask can keep a lower band while omitting a higher band: if the lower band's
profit at its own ratio is nonpositive, the high-cut single dominates; if positive,
the omitted middle band is valuable and the low-cut single dominates. See the
[nested-family proof](occupancy_nested_family.md). The class-macro/shared-recipe case
can have different dominating endpoints in different classes, so this equality is
not a general benchmark theorem. Expectation of a finite realized IoU ratio differs from
the population expected-count ratio used in this conditional proof.

**Exact solver.** Source-mask membership atoms are sufficient statistics. Subset
and superset zeta tables evaluate each recipe from per-class TP/FP counts without
another feature extraction. L complete sources and all origins give L*4^(L-1)
syntactic recipes; this finite certificate says nothing about all possible algorithms.
Shared costs require ancestor closures, including unselected producers computed to
make an adaptive choice. See [solver proof](solver_theory.md).

**Explains / predicts.** Facts4 and9 are compatible with absent joint advantage:
after the best single source, the other edit can fall below its *updated* threshold.
If no feasible residual correction has positive class-weighted gain, no joint improvement
is possible in that family. An unresolved paired interval does not show this premise holds.

**Minimum check, not executed.** Use the existing aligned4000 complete source masks
and I/U records, plus the existing600 factorial records. Measure the cross-marginal
counts and class-wise overlap signs once, at fixed domain/cost. Algebraic equalities
need no new episodes. Future superiority needs one frozen recipe versus the strongest
single/simple control on fresh groups; sample size follows the paired variance below.

## 2. 'Less is more' and finite selection

**Conclusion.** Near-break-even purity and finite samples alone do not imply negative
expected true gain or a universal maximum number of components. A negative result
needs enough harmful candidates to be mistakenly admitted, an explicit cost, or another
stated adverse effect. Optional-family true optima are nondecreasing as choices expand.

For a fixed context, let a fraction r of candidate edits have true gain +a and the
rest gain -b. A fixed selection gate has true-positive and false-positive inclusion
rates TPR,FPR. Then, exactly under this two-type gain model,

    E[true contribution] = r*a*TPR-(1-r)*b*FPR.

It is negative exactly when false inclusions outweigh accepted useful edits. An
optimistic estimated maximum, a confidence interval crossing zero, or a Hoeffding
error bound is not by itself a negative-expectation theorem.

**Conditional illustrative count law.** For symmetric gains +/-a_j, Gaussian
proxy noise sigma/sqrt(n_eff), accept if the proxy is positive, and r<1/2,

    E[contribution_j] = a_j*[Phi(a_j sqrt(n_eff)/sigma)-(1-r)].

If a_j=A*j^(-nu) and contributions remain additive in this fixed-context/gated
model, the positive-contribution prefix ends at

    j < [A sqrt(n_eff)/(sigma Phi^(-1)(1-r))]^(1/nu).

This is a conditional relation between component number, signal margin and sample
size. It is not a universal IoU bound, nor a rule for forcibly selecting that many
operators. Actual overlaps can raise marginal purity; actual IoU denominators create
nonadditivity. For r>=1/2 or all truly useful edits, this argument provides no finite
accuracy-only cutoff. See [selection proof and limits](quality_theory.md).

**Explains / predicts.** Fact4 can reflect small conditional margins and false
inclusions; fact9 shows useful components need not disappear. If finite-sample selection
is the cause, optimistic DEV inclusion should exceed held-data gain precisely among
near-zero-margin additions. More independent selection data should reduce this effect
under a stationary candidate-error model. Persistent negative oracle margins instead
limit the edit construction, not sample size.

**Minimum check, not executed.** On the existing4000 records, compute fixed-context
class-weighted marginal gains and leave-photo-group/fold stability for the prespecified
components. Estimate valid-edit fraction and proxy error, rather than assume them.
Use the existing frozen1200 only as specified confirmation, with its fitting/exposure
boundary preserved. The observations alone do not supply a numeric optimal k.

## 3. RCG as an estimator and conditional evidence sharing

**Conclusion.** RCG is anchored graph denoising. It may make similar auxiliary
rules redundant, but its energy does not prove conditional sufficiency or semantic
information exhaustion. Finite smoothing is an invertible soft filter for fixed graph
and positive anchors.

The core estimator is exactly

    f = argmin 1/2(f-y)'H(f-y)+lambda/2 f'L f
      = (H+lambda L)^(-1)H y.

H weighs unary anchors; L penalizes disagreement along feature edges. Conditional
Gaussian noise/GMRF assumptions make this a posterior mean/MAP for a continuous
signal. They are not proved by the heuristic anchor scores or by mIoU improvement.

Whitened graph mode l is multiplied by 1/(1+lambda*mu_l). Zero modes survive. In
each graph component, sum H_i*f_i=sum H_i*y_i; output stays in the component's
unary range, and isolated nodes are unchanged. Infinite strength converges to the
component's anchor-weighted mean. Coherent high-score wrong components can survive,
and true low-supported components can be lost. These are exact structural limits.

**Conditional mechanism.** A graph-smooth useful signal with more graph-disagreeing
unary error is one denoising regime, not a necessary condition for IoU improvement.
Under zero-mean whitened noise, squared-risk improvement instead requires the exact
signal/noise energy balance from the estimator note. Graph-only gains and reference-with-graph interaction
in fact2 support this account without establishing its spectral premise. Fact3's whole
false-region deletion can result from lower anchor support or links to low-score neighbors;
the same mechanism can remove true regions. Position and feature graph comparisons must
account for effective smoothing and signal alignment, not just graph degree.

For an extra cue Z and retained RCG state T, full-mask utility sufficiency plus an
optimal decision given T makes Z useless for that utility. If Z changes the best conditional
action, it has positive information value. An implemented nonoptimal threshold can improve
even with no new information. Fine sampling/geometry, changed connectivity, or identifiable
occupancy may escape a coarse scalar readout; this names possible information distinctions,
not new components or guaranteed gains. See [estimator derivations](graph_estimator_theory.md).

**Minimum check, not executed.** Use the existing600 fixed-strength graphs/fields to
measure anchor support, cross-region graph exposure and truth/error mode alignment.
Use the existing4000 failed edits versus the same-domain/same-budget RCG-score control
to measure decision-relevant residual value. Unchanged output/ranking gives an exact
null test; conditional sufficiency needs a specified residual statistic and equivalence
tolerance. A nonsignificant contrast is not proof of no information.

## 4. Occupancy identification and the correct decision geometry

**Conclusion.** The stated budget alone supplies no distribution-free identification
guarantee. With known distinct query foreground/background score laws, pi is identifiable.
Unknown laws, weak separation or allowed drift can prevent reliable inversion. Existing
failed estimators do not prove unidentifiability from full allowed DINO/RGB information.

Known score laws give G_pi=pi F1+(1-pi)F0 and

    TV(G_pi,G_pi')=|pi-pi'| Delta,    Delta=TV(F1,F0).

Distinct components identify the mixture weight; small Delta amplifies drift. Useful
relative accuracy for a tiny object needs component and histogram errors small compared
with pi*Delta. 'Approximately invariant' must be measured on this decision-relevant scale.

For actual measurement, freeze score bins/events or use CDF sup-norm separation;
the same linear-mixture identity gives ||G_pi-G_pi'||_infinity=|pi-pi'|
||F1-F0||_infinity. Raw empirical continuous-score point masses cannot be assigned
a small TV radius against a continuous law: that TV distance is one. Density-level
TV estimation needs its own regularity. Separation, drift and uncertainty must all
be measured in the same declared observation metric.

A strict score-only ambiguity example preserves G while changing pi0 to pi1>pi0:

    F0'=F0; F1'=(pi0/pi1)F1+(1-pi0/pi1)F0.

The foreground-law drift is (1-pi0/pi1)*Delta. If that drift is admissible, the score
histogram cannot distinguish these proportions. This obstruction does not automatically
cover graph, spatial, raw-image or other feature observations.

Three informative, correctly oriented, conditionally independent views can identify a
binary latent model from covariances and a third central moment. Same-encoder views can
satisfy such assumptions in a model, but common appearance/context errors often violate
them. Two unrestricted unknown views generally are insufficient. Three binary views are
just identified; a good fit does not test independence. Four-view restrictions or actual
label-conditional full-joint diagnostics are needed. See [identification proofs](identifiability_theory.md).

For admissible worlds with observable TV v and proportions pi0,pi1, a strict minimax
bound is max-world E|pihat-pi| >= |pi1-pi0|(1-v)/2. If world truths T0,T1 are attainable
and their Jaccard distance is d, IoU regret is bounded below by d*(1-v)/2. Applying it
to current COCO requires legitimate worlds and the full information set. No such numeric
current-dataset irreducible-loss certificate has been established here.

**Threshold consequence, conditional.** With fixed query laws and survivals A,B,

    J(t;pi)=pi A(t)/[pi+(1-pi)B(t)].

A smooth optimum satisfies likelihood_ratio(t)=[(1-pi)/pi]*J*. For the actual
class-summed benchmark it instead uses that class's common J_c*. Known pi does not
erase the objective coupling. Within fixed nested thresholds, a unique optimum moves
weakly down as pi increases; monotone likelihood ratio further connects it to the
unrestricted Bayes rule. Local regret is quadratic in log-odds error under smoothness;
factor2/factor4 are empirical observations, not universal tolerances.

Identifying pi and p(Y|score) identifies score-only threshold utility, not general
spatial/feature-conditioned edit utility. Applying that posterior to arbitrary proposal
memberships additionally needs p(Y|score,membership,context)=p(Y|score), or a justified
richer model. A within-score difference in foreground rates across proposal regions
can reject this extra sufficiency assumption without rejecting pi identification.

Parameter identification is stronger than necessary for a decision: different
admissible pi/utility models may share an optimal mask. Nonidentification of pi implies
unavoidable selection regret only if observationally compatible models have incompatible
optimal actions and a positive utility separation. Reliable action ordering can suffice
without estimating every posterior or absolute score.

**Minimum check, not executed.** On the existing4000 score/GT outputs, measure
foreground/background separation, per-image/stratum drift and the error-to-pi*Delta
ratio, especially below2percent area. On existing600 and then prespecified held data,
check full label-conditional dependence, channel rank/orientation and cross-triple
consistency of already available views; do not train another classifier or invent views
until this premise is established. Facts7/8 distinguish transport/conditioning/decision
failure only after these measurements, not from aggregate scores alone.

## 5. Final form and what can actually be bounded

**Conclusion.** The problem is selection over observable edit atoms, using class-specific
utility and shared inference cost, with an explicit information/estimation uncertainty.
The solver, the utility evidence, and the attainable mask family are different bottlenecks.
The provided observations do not support an unconditional sparse optimum, universal
RCG sufficiency, or a positive numeric irreducible-loss claim on current COCO.

Only a same-cohort nested pipeline has the exact additive gain accounting:

    J(final)-J(FoRIS) = [J(RCG)-J(FoRIS)]
                      +[J(fine)-J(RCG)]+[J(final)-J(fine)].

Thus +1.41, +.4, +5.07 and +8.91 are not independent score budgets to sum. The
supplied new1200 complete +2.49 [1.47,3.47] is a separate measured comparison,
and its six other-fold-GT-fitted cutoffs remain extra label calibration.

| Layer | Evidence | Quantitative boundary |
|---|---|---|
| Already obtained | RCG +1.41 on reported4000; fine about+.4 in its own context; supplied complete new1200 +2.49 | Measured gains, not ceilings; different contexts cannot be added |
| Privileged threshold capacity | 62.33 to71.24 by per-image IoU-optimal cuts | +8.91 is achieved oracle headroom; not the exact class-summed threshold upper bound |
| Privileged occupancy policy | True-pi formula +5.07 | Achieved policy result, not a universal pi-information ceiling or deployable forecast |
| Same-information but not yet estimated | Useful conditional calibration/ranking/geometry if their assumptions hold | No justified numeric available-gain ceiling from the supplied facts |
| Nonidentification / information obstruction | Symbolic two-world lower bounds under explicit model sets | No positive numeric irreducible COCO loss has yet been proved |

An exact fixed-score *class-metric* threshold ceiling is a separate theoretical object:
for each class, maximize sum_i I_i(t_i)/sum_i U_i(t_i); at trial q choose each cut
maximizing I_i-q U_i. This is certifiable from existing cutoff counts without new
encoder work, if later authorized. Finer fields or new rankings have a different ceiling.

**Exact finite-bank capacity bound.** Include the origin and every allowed alternative
origin in the complete-source bank. For canonical edits from that bank, an
all-source-positive pixel is always kept and an all-source-negative pixel is never
added, whatever origin/subsets are selected. Only source-disagreement pixels are
editable. Let F_c be FP in unanimous-positive pixels, N_c be FN in unanimous-negative
pixels, and G_c the GT mass. Even an oracle that freely corrects every disagreement
pixel satisfies

    max_family J_c <= (G_c-N_c)/(G_c+F_c).

This gives a concrete upper bound on the given bank, and a lower bound
(F_c+N_c)/(G_c+F_c) on its capacity loss. It is not an information lower bound for
the encoder: a new complete source may edit those consensus errors. These quantities
can be obtained from existing membership atoms if later authorized; they were not
computed in this theory task.

**Exact signature relaxation.** A fixed global mask recipe assigns one bit to each
complete-source membership signature b. Relax the permitted AND/OR recipes to arbitrary
class-specific functions h_c(b), but keep the unanimous endpoint bits fixed. With GT
atom counts t_cb and f_cb, this gives

    J_signature,c = max_h sum_b t_cb h(b)/(G_c+sum_b f_cb h(b)).

At trial q_c, mandatory endpoints contribute their fixed residual, each free signature
chooses h=1 iff t_cb-q_c f_cb>0, and -q_c G_c remains in the residual. Exact ratio
updates solve this relaxed one-class problem. Therefore

    max_global_recipes macro J <= mean_c J_signature,c
                              <= mean_c (G_c-N_c)/(G_c+F_c) <= 1.

Mixed-GT signatures can make the signature bound strictly tighter than consensus
correction. This is a GT capacity bound on bit-mask readout, not a Bayesian information
bound for the frozen features. For adaptive per-query recipes, the appropriate relaxed
atoms include (episode,b), while all atoms of a class still share q_c.

For a known population law, fixed information Xi, and one fixed feasible action family,
let V_current be deployed expected utility, V_Xi the best allowed Xi-measurable policy,
and V_oracle the best same-family policy allowed the labels. Then

    V_oracle-V_baseline=(V_current-V_baseline)
                       +(V_Xi-V_current)+(V_oracle-V_Xi).

This is a genuine decomposition into obtained gain, unused decision value and a Bayes
information gap, under the declared law. Cross-model statistical nonidentification is
an additional issue and needs its own minimax argument; it cannot be read numerically
from a per-image threshold oracle. Neither V_Xi nor this gap has been measured here.

**Minimal discrimination, not executed.** Use existing4000 counts to distinguish the
exact class-optimal fixed-score ceiling from the reported per-image oracle, then compare
occupancy-law predictions, oracle outcomes and actual frozen fields on the same cases.
Use per-class four-way edits, not raw FP/FN percentages, to attribute the worst10percent.
If the simultaneous *paired-gain* uncertainty bound covers the leading candidate's
advantage, that bound does not certify the winner; more exact recipe optimization alone
does not resolve the evidence error. Large absolute calibration error need not imply
large paired error, and failure of this certificate does not prove that all predictors fail.
A future complete confirmation
tests a frozen result; it does not repair an unidentified evidence model by itself.

## Shared sample-size rule for the proposed falsification checks

Every proposed measurement is unexecuted. Existing4000/600/1200 are named because
they contain the relevant outputs, not because those counts guarantee power.
For a paired class-mIoU contrast, independent units are the prescribed photo-connected
groups; its influence variance sigma_psi^2 must be obtained from matched I/U receipts.
At effect margin delta and target power1-beta, a planning approximation is

    G_req >= (z_.975+z_(1-beta))^2 sigma_psi^2/delta^2.

For confidence half-width h, use G_req>=z_.975^2 sigma_psi^2/h^2. Groups must then
be translated to episodes using the actual class/fold/overlap contract. Equivalence
claims require a prespecified tolerance and an interval contained inside it. A CI
crossing zero is unresolved. No required variance or decision tolerance was newly
estimated during this theory task, so a new numerical episode count would be invented.
