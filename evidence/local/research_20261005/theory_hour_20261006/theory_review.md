# Adversarial review of the one-hour theory

Owner: `/root/theory_adversarial_review`. Theory-only review, 2026-10-06 UTC.
Scope: the supplied COCO-20i 1-shot setting, one complete masked reference,
frozen DINOv3, no training; intersection and union are summed by class before
macro averaging. No model call, experiment, data evaluation, remote operation,
or GPU work was performed. Only this note was written.

Reviewed: `main_theory.md`, `solver_theory.md`,
`occupancy_gain_geometry.md`, `identifiability_theory.md`, and
`graph_estimator_theory.md`; the follow-up audit also covers `five_questions.md`,
`occupancy_finalshape.md`, and `quality_theory.md`.
Numerical observations below are the parent's
supplied facts or another theory note's explicitly retained historical report;
they were not independently remeasured here.

Labels: **Strict** means a mathematical consequence of stated definitions;
**Conditional** means the additional assumptions are necessary to the stated
argument; **Conjecture** means an empirical explanation that remains untested.
Predictions below are implications to assess later, not a proposed new method or
authorization to run a check.

## Verdict

The finite-family accounting and exhaustive solver results are defensible. They give an
exact development optimum for a specified mask family and objective, not a
no-GT adaptive selector or a generalization certificate. The notes correctly
reject a universal sparsity theorem, universal graph sufficiency, a dataset-
specific impossibility claim from failed estimators, and cross-cohort addition
of gains. The proposed **partial-state dominance pruning** has a genuine
counterexample and must be restricted before it can enter an optimum certificate.
The following qualifications must survive the final synthesis:

1. TP/FP-count and dependency-closure dominance safely compares **complete
   recipes**; it does not safely prune partial states with overlapping future actions.
2. The identified-set diameter bound requires common fixed component laws;
   it does not cover unrestricted alternative component/model fits.
3. Identifying prevalence and a posterior conditional on one scalar score
   does not identify arbitrary spatial/feature-conditioned A/B edit utilities.
4. The initial main note overstated the need for monotone likelihood ratio.
   Its 03:08 UTC revision corrects this; retain the weaker nested-family result.

**Readback status, 03:23 UTC:** the solver note now restricts dominance to
complete recipes or continuation-safe states; the final five-question note
states canonical disjoint domains, replaces the absolute-error impossibility
claim with a paired certificate, and removes the smoothness-necessity claim.
The main note now fixes common component laws in its TV bound. The quality
equation's missing plus is repaired. The new consensus and signature-relaxation
bounds in the final framework pass under the bank/cohort conditions below.
The counterexamples are retained as evidence for those restrictions, not as
unresolved defects in the repaired versions.

Two smaller qualifications remain relevant to the final wording: an identified
common best action does not require every utility value to be identified;
strict extra-information value excludes a shared optimal action across Z
outcomes, including ties. These are already correctly stated in the final-shape
and full graph notes, respectively. Preserve those precise statements if their
shorter versions are used in the final response.

## 0. Critical solver correction: count dominance is not partial-state dominance

**Correction target:** `solver_theory.md` section 5 says “Safe state dominance
is classwise TP greater-or-equal and FP smaller-or-equal, together with
dependency-closure inclusion.” This is safe for complete recipes' score/cost
Pareto comparisons. It is false for partial search states that still have
overlapping actions available, even when their remaining actions are identical.

**Strict counterexample.** One class, empty origin, GT `{t1,t2,t3}`, and two
background pixels `f1,f2`. There are three fixed addition producers:

    A = {t1,t3,f1};
    B = {t2,f2};
    C = {t1,t3,f2}.

Suppose a node has already decided the A/B choices and only C remains
undecided. Compare state X choosing A and excluding B with state Y choosing
B and excluding A. X has TP=2/FP=1; Y has TP=1/FP=1. They can have the same
dependency closure: all outputs may be cached, or one common operation may
produce all three masks. Thus X strictly dominates Y by the quoted criterion.

After the same future action C, however,

    X+C = {t1,t3,f1,f2}: IoU = 2/5;
    Y+C = {t1,t2,t3,f2}: IoU = 3/4.

The global optimum of the eight subsets is B+C at `3/4`: A and C each score
`1/2`, B scores `1/4`, A+B and A+B+C each score `3/5`, A+C scores `2/5`, and
empty scores zero. Pruning Y by X deletes the unique optimal subset. Closure
inclusion is satisfied; the defect is unrecorded future overlap, not cost.

**Strict correction.** Retain complete-recipe Pareto pruning, or require a
dominance relation that is preserved under **every common feasible continuation**.
Counts alone do not supply that relation. In the canonical family, pointwise
GT-positive coverage superset and GT-negative coverage subset, together with
compatible remaining choices and cost dominance under continuations, is one
sufficient condition. A stronger compressed state must retain whatever
membership/overlap information establishes continuation dominance. This is a
solver correctness condition, not a proposed new segmentation method.

The section's optimistic numerator/denominator branch upper bound does not
have this defect: its extrema bound every descendant irrespective of overlap.
Exhaustive enumeration and certified arithmetic also remain valid.

**Prediction:** on the symbolic family above, pruning based only on current
TP/FP and closure can return below `3/4`. Any claimed finite-family certificate
must use exhaustive coverage, a valid descendant bound, or continuation-safe
dominance. No code was executed on this example.

## 1. Required correction: raw-threshold comparative statics

**Strict conditional result.** Fix the same foreground/background score laws
and an ordered, nested upper-score threshold family. For `t_H > t_L`, write
`A_H <= A_L` and `B_H <= B_L` for the two survival pairs. With
`r=(1-pi)/pi`,

    J(t_H;r) >= J(t_L;r)
    iff (A_H-A_L) + r(A_H B_L-A_L B_H) >= 0.

The intercept is nonpositive. If the higher cutoff ever strictly wins, the
slope is positive, and it continues to win as background odds increase.
Consequently, unique optimal cutoffs move weakly downward as prevalence
increases. This does **not** require monotone likelihood ratio. Equivalent
masks and tied optima need a fixed canonical cutoff convention.

**Initial correction target:** `main_theory.md` section 12, sentence “A decreasing
raw-score threshold as pi increases additionally requires a monotone
likelihood ratio.” The nested-family argument in `occupancy_gain_geometry.md`
section 2 already contains the correct weaker result.
The main note's revision observed at 03:08 UTC now incorporates that correction;
this finding is resolved in that observed version.

Monotone likelihood ratio is a useful sufficient condition for identifying
the unrestricted Bayes decision with a single upper-score cutoff. It also
supports a smooth derivative argument with a unique interior optimum and
positive likelihood-ratio slope. Even that equivalence at one operating point
only requires its relevant likelihood-ratio superlevel set to be an upper
score tail; global monotonicity is not necessary in every isolated case.

**Prediction:** under fixed laws and canonical unique threshold choices, an
upward optimum cutoff as prevalence increases contradicts the nested-family
model. Across differently calibrated scores or drifting laws it contradicts
nothing. This result does not compare numeric cutoffs from different images
without their common-law premise.

## 2. Required qualification: common component laws in the TV bound

**Strict, for fixed known components.** If

    G_pi = pi F1 + (1-pi) F0,   d = TV(F1,F0) > 0,

then `TV(G_pi,G_pi')=|pi-pi'| d`. If both mixtures are within epsilon of
the same observed law, their proportion separation is at most `2epsilon/d`.

**Correction target:** the phrase “all admissible mixture/model fits” in
`main_theory.md` section 12 is broader than this proof. Every fit must use
the **same** stipulated components, or component uncertainty must appear in
the bound. The transport bounds in `identifiability_theory.md` section 3
correctly keep that extra error term.

**Strict counterexample to the broader reading.** Let the observed score be
binary with positive frequency one half. Both positively oriented models fit
it exactly:

    Model A: pi=1/2, F1=Bernoulli(.9), F0=Bernoulli(.1).
    Model B: pi=1/4, F1=Bernoulli(.8), F0=Bernoulli(.4).

Their residual is zero, their proportions differ, and each model has distinct
components. A zero residual cannot give a zero identified-set diameter across
these unknown-component models.

**Additional finite-sample condition:** empirical point masses and an
absolutely continuous population law have TV distance one. A small estimated
TV radius therefore needs an appropriate discrete observation space or
density-estimation regularity; it does not follow from a raw empirical
continuous-score distribution alone. A fixed-bin/event statement must use
separation in that same observation space. Spatial dependence remains a
separate concentration assumption.

**Prediction:** small fit residual can coexist with a wrong proportion when
template drift is absorbed by the mixture weight. Correct known components
with nonzero separation do identify prevalence; the supplied histogram loss
of -4.08 cannot disprove that conditional theorem.

## 3. Hidden condition: prevalence identification is not full edit calibration

**Strict distinction.** Known `F1,F0,pi` identify `P(Y=1 | score)` and the
population objective of score-only thresholds. To value a mask `C` using
`E[C p(score)]`, one additionally needs `C` measurable with respect to that
score, or a condition such as

    E[Y | score, candidate-membership, allowed context]
        = E[Y | score].

For unrestricted feature/spatial A/B recipes, the needed conditional marginals
are relative to the information that generated and selected those recipes.
Global score calibration alone supplies no such conditional calibration.

**Strict finite counterexample.** There are eight pixels, four with high score
and four with low score. Disjoint candidates P and Q each contain two high and
two low pixels. In world 0, the four true pixels comprise three in P and one in
Q; in world 1 these counts are reversed. Arrange both worlds to have three true
high-score pixels and one true low-score pixel. Both worlds have exactly

    pi=1/2;
    F1(high)=3/4, F0(high)=1/4;
    p(high)=3/4, p(low)=1/4.

The known components are distinct, so prevalence is identified. Each candidate
has the same score-based expected intersection two and expected-count IoU
one third. Yet actual candidate IoUs are `3/5` and `1/7`, with their identities
reversed between worlds. The canonical two-source composition family contains
P, Q, their union and their intersection; its best choice also reverses.

This is a mathematical illustration of missing within-score conditional
structure, not evidence that actual COCO RGB admits two canonical GTs. A
known-component theorem about prevalence must not be relabelled as either an
adaptive A/B selection theorem or a theorem that prevalence is unidentifiable.

**Prediction:** a score can be globally calibrated while residual label
frequencies differ inside the competing proposals' disagreement strata. Such
a difference rejects score-only conditional sufficiency for those utilities,
while leaving prevalence identification intact.

## 4. Metric and oracle claims: the corrections currently pass

**Strict.** Class-summed IoU weights each image by its candidate-dependent union
within class, then gives equal weight to classes. Optimizing each image's own
IoU is not optimizing that benchmark. The two-image example in the occupancy
note is valid: improving one image from `5/10` to `10/16` lowers the same-class
aggregate from `105/110` to `110/116` when a perfect 100-pixel image is present.

Thus the supplied `62.33 -> 71.24` is an achieved privileged-policy score.
If the policy exhaustively maximized each image's fixed-score IoU, it bounds
that image's threshold alternatives individually. Its **aggregate 71.24** is
not a class-summed threshold-family ceiling. The independent-cut class oracle
in the main note's section 12 and occupancy equation (9) is the correct
finite-family ceiling, conditional on allowing those independent cuts and
having no shared parameter, budget or policy constraint.

The class-oracle `q_c` and GT-selected cuts are diagnostic quantities. Their
existence does not make evaluation class identity or its optimum available to
a one-reference inference rule. The note's single-ratio Dinkelbach result and
its warning against a single scalar reduction of a macro sum of ratios pass.

**Prediction:** the exact same-cohort class oracle cannot score below the
per-image oracle if that latter policy is feasible in its family. No such
oracle was computed in this review.

## 5. Finite optimum, small components and near break-even

**Strict.** Including every complete producer in the unconstrained family
makes its exact GT development maximum weakly dominate the best included
producer. With a budget, inclusion applies only to equally feasible actions
and their actual dependency costs. This is a family-inclusion statement.
It supplies neither strict joint gain nor an adaptive no-GT predictor.

Ordinary paired uncertainty for a fixed recipe on a reused cohort does not
automatically account for searching that recipe on the same labels. Held-fold
selection also needs the candidate-library construction and all calibration
exposure disclosed. “Exact optimum” and “selection bias controlled” are
different properties. A mathematical optimum additionally requires correct
rational/tie comparisons; enumerating every recipe in floating arithmetic
alone is not that numerical certificate.

**Conditional finite-sample statement.** Suppose an addition's purity is
estimated from n justified independent Bernoulli validation units. At current
class IoU J, its break-even purity is `p0=J/(1+J)`. A sampling radius of order
`sqrt(log(1/delta)/n)` can make the sign unresolved near p0. Searching a
finite K-family requires simultaneous or otherwise selection-valid uncertainty.
This makes a small gain hard to certify; it does not prove the true gain
negative. This illustrative sampling model is not an assertion that image
pixels are independent or that a one-class threshold certifies macro mIoU.

**Strict counterexample to unconditional sparsity:** m disjoint all-true
additions require all m actions for the strict optimum. A small number of
outer components can hide high dependency cost, and a large number can share
most work. The solver's cost-DAG rule and its rejection of universal
submodular/greedy/sparsity guarantees are sound.

**Prediction:** reducing a valid confidence radius can resolve a sign that was
previously uncertain in either direction. Operator count, near break-even
purity, and a CI crossing zero alone predict neither an automatic negative
result nor reliable superiority of a smaller recipe.

## 6. RCG invertibility: preserve the distinction already in the graph note

**Strict conditional algebra.** For the actual anchored model reported in
`graph_estimator_theory.md`, positive H, fixed symmetric nonnegative graph and
finite lambda give

    f=(H+lambda L)^(-1) H y;
    y=f+lambda H^(-1)Lf.

The core is invertible with exact f, H and L. The solver note's
`(I+lambda L)^(-1)` is its uniform-anchor special case, not the full actual
pipeline. Stored precision, CG residuals, rendering, rank/min-max compression
and a hard mask can lose information or make inversion unstable.

Recovering y is not a theorem that y/H/L contain all semantic evidence in the
frozen features; it is still less a theorem that the current threshold uses
that evidence optimally. A graph may encode information absent from its unary.
For fixed full features, deterministic cues add no information relative to
those features, but may add information relative to a compressed RCG field
or improve an imperfect decision using the same features.

The graph note's conditional squared-error bias/variance model does not imply
an IoU theorem. Its mixed-label infinite-strength result implies an eventually
erroneous constant decision in that component when its limiting mean is away
from the cut; it does not prove an aggregate mIoU downturn from a particular
smaller strength. Gross value of information is nonnegative only for an
optimal selector over the same action set; an extra cue's cost can change
that set under an equal-compute constraint.

**Prediction:** weighted component mass and range conservation follow from the
stated linear system. Monotone score gain over lambda 1/4/16/64, rising TP
deletion, and failed post-RCG auxiliary constructions do not establish a
semantic sufficiency condition. The same-count comparisons limit the tested
selection rules, not all same-encoder evidence.

## 7. Minimax statements: keep the admissible model and observation explicit

**Strict in the unrestricted model.** The identifiability note's disjoint
two-world construction and its `1/2` minimax regret bound are correct for an
otherwise unconstrained reference-to-query label relation. The Le Cam overlap
bound is valid with its pointwise incompatible-action gap assumption.

For the fixed canonical COCO semantic rule on full RGB, freely assigning two
different GTs to identical RGB is not automatically an admissible pair.
A feature-only collision also does not obstruct a selector allowed to read
distinguishable RGB. A dataset-specific lower bound requires two justified
observable laws over the **whole allowed information set**, low distinguishability,
and a latent/action gap. The updated identifiability note explicitly makes
this correction. No supplied result demonstrates a positive irreducibly
unidentifiable number of COCO mIoU points.

Three-view latent identifiability needs the full conditional independence law,
nonzero channel gaps, stable rates and transferable semantic orientation.
Pairwise decorrelation, a common encoder, agreement, or independent augmentation
RNG alone does not establish those premises. Its finite moment bounds are
conditional on justified independent units; the algebraic moment inversions
and the warning about degeneracy pass.

**Prediction:** violation of reconstructed-rate validity or compatible
multi-view moment constraints rejects that latent model. Passing them cannot
certify its semantic foreground interpretation. Estimator failure does not
determine observable TV or a minimax gap on actual COCO.

## 8. Supplied numbers: the defensible attribution boundary

**Strict.** An aligned sequence of complete masks on one frozen cohort admits
telescoping gains. Different cohorts, class sets, finalizers, resolutions or
fitting resources do not. Even on one cohort, true-pi and image-threshold
oracles are alternative interventions; their gains are not additive budgets.
Subtracting `8.91-5.07` would be a difference between those policies if their
contracts match, not a proved ranking or representation gap.

The supplied 4000 RCG `+1.41 [1.08,1.70]` and new1200 RCG+fine+size-cut
`+2.49 [1.47,3.47]` are positive complete effects in their recorded settings.
They do not by themselves establish a stable >=2-point advantage, superiority
to the strongest same-information control, or a strict label-free readout
when size cuts were fitted with other-fold GT. The reported `fine about +.4`
is a marginal result in its own pipeline, not a universal increment.

The historical no-CRF 600 graph/ref/joint rows can support an interaction only
under their matched same-baseline contract. They cannot be treated as additive
stage gains over complete FoRIS4000. The graph note correctly preserves the
historical interaction receipt as a separate reused-development result.

Worst10% FP60%/FN71% identify raw error concentration, not the same fraction of
recoverable macro loss. For class c an edit's exact value has numerator
`dI_c-J_c dU_c` and denominator its final union; class location, true deletion
and new FP matter. The small/large area ratios 2.94/.81 are compatible with
occupancy-dependent calibration, but do not isolate its cause or identify a
working estimator. The factor2/factor4 sensitivity is conditional empirical
behavior of that rule, not a universal tolerance.

**Prediction:** a claimed additive decomposition must reconstruct the exact
final-minus-initial macro score from consecutive matched differences. Without
such identities and resource/cohort alignment, no combined numerical ceiling
or gain forecast follows.

## 9. Follow-up audit of the final five-question framework

The following findings refer to the version read at approximately 03:13 UTC.
Later Root revisions can resolve them without invalidating their counterexamples.

### 9.1 Cross-marginal iff statements need disjoint canonical domains

**Correction target:** `five_questions.md` section 1 starts with general
deletion priority, allowing a selected deletion to cancel a selected addition.
Its later one-class iff conditions are the canonical **disjoint-domain**
conditions; that restriction must be stated there. Effective additions in the
joint mask need not equal additions in the add-only mask in the general family.

**Strict counterexample:** empty origin, GT `{t}`, addition `{t,f}` and
deletion `{f}`. Add-only IoU is `1/2`, joint IoU is one. The inside-origin
deleted TP and FP are both zero, so `J_A*deletedFP > deletedTP` is false despite
the strict improvement. The conflict term is removal of a newly added false
pixel. Restrict the iff to canonical domains or include the conflict counts.

### 9.2 A large calibration error is not a winner-impossibility theorem

**Correction target:** section 5's sentence “If conditional calibration error
already exceeds the prospective joint gain, solving more recipes cannot
provide a reliable winner prediction.” This is not generally implied by a
global calibration error or even failure of one sufficient uniform bound.
It needs a defined comparison-specific uncertainty and a decision criterion.

**Strict counterexamples:** arbitrarily large probability error on common
agreement pixels can coexist with a certified pure-TP addition. That candidate
beats its baseline regardless of the erroneous agreement probabilities. In
the opposite direction, error on one disagreement pixel can be `1/N` of a
whole-image calibration average yet reverse a rare-target decision. Baseline
union/IoU uncertainty must still be accounted for; it does not remove either
counterexample.

**Conditional valid statement:** if simultaneous paired-value radii satisfy
`|g_h-ghat_h| <= epsilon_h`, then
`ghat_h-ghat_k > epsilon_h+epsilon_k` certifies h over k. Failure of that
inequality means **this certificate is inconclusive**, not that no reliable
comparison is possible. An impossibility claim instead needs compatible
models with genuinely incompatible best actions. The main note's
disagreement-specific sensitivity bound is the appropriate accounting model.

### 9.3 Smoothness is a possible regime, not a necessary IoU premise

**Correction target:** section 3's “Improvement needs true signal smoother
than relevant unary error on the feature graph.” In the graph note's zero-mean
spectral squared-error model, the exact improvement condition is

    sum_l [(1-beta_l)^2 theta_l^2
              -(1-beta_l^2) sigma_l^2] < 0.

It depends on energy and signal-to-noise allocation, not an unspecified
smoothness ordering alone. For example, signal energy one entirely in a
mode with mu=2 and noise variance100 entirely in a mode with mu=1 give, at
lambda=1, risk `4/9+25 < 100`, even though the signal has higher normalized
graph frequency than the noise. Such eigenvalues are realizable by a
nonnegative weighted three-node graph. More importantly, that risk condition
does not determine thresholded IoU. Treat a smoother-signal regime as a
conditional mechanism conjecture or sufficient model regime, not a general
necessary condition for observed IoU gain.

An informative Z has strictly positive *optimal* utility value only when no
common T-only action is conditionally optimal across almost all relevant Z
outcomes on a positive-probability T set. A mere change of tie-chosen action,
posterior, or argmax set is insufficient. Equal action-set and compute
conditions remain necessary to that comparison.

### 9.4 One algebraic defect in the quality note

**Correction target:** `quality_theory.md` equation (3) is missing the plus
before `J0,c(dFP,c-aFP,c)`. The correct numerator is

    aTP,c - dTP,c + J0,c*(dFP,c-aFP,c).

The purity expansions, helpful/harmful gate expectation, Gaussian toy cutoff,
optional-family monotonicity and paired influence model otherwise pass under
their stated assumptions. Its sigma_proxy and sigma_pair represent different
noise sources and correctly must not be interchanged. A numerical minimum n
cannot be inferred without the correct variance, sampling unit and effect.

### 9.5 Final-shape result: confirmed with its explicit policy constraints

**Strict under its definitions:** `occupancy_finalshape.md` correctly uses
the expectation of the **finite class-summed benchmark**, rather than quietly
substituting a ratio of expectations. Its same-family chain is

    actual <= Bayes(Xi,F) <= GT-oracle(F) <= 1.

The remaining gap to one has separate extraction regret, privileged-GT value
within F, and family-capacity loss. Including an obtained baseline gain gives
four terms. Alternatively an unrestricted Xi-measurable Bayes ceiling gives
the three-term obtained/decision-regret/Bayes-irreducibility decomposition.
Unrestricted Bayes and the restricted family's GT oracle have no general
ordering; they cannot be inserted into one chain by naming them “ceilings.”

The note explicitly limits its conditional maximum to allowed cohort-wide
actions, and retains a constrained policy supremum for episode-local or
selector-program restrictions. A fixed global development recipe is stricter
than an arbitrary Xi-adaptive Bayes policy; its feasible program class must
remain explicit if the latter is used as a comparison envelope.

The distinction between known-law Bayes ambiguity and cross-model statistical
nonidentification is correct. Exact conditional values need not be identified
to identify a winning action: two compatible models can have utilities
`(.7,.6)` and `(.8,.5)` for two actions, with the same winner and different
values. Conversely, identifying pi need not identify edit rankings. The
main note's “adaptive requires an identifiable estimate of conditional edit
utility” should therefore be understood as requiring justified **decision-
relevant information** for a guarantee, not unique identification of every
numerical utility or a premise every useful adaptive heuristic must satisfy.

### 9.6 Confirmed stronger bound from source signatures

**Strict, on one fixed cohort.** Include every allowed origin in the complete
bank and derive all edits from that bank. For a global Boolean recipe, the
output is one common `h(b)` on source-membership signature b. Let `t_cb,f_cb`
be class-summed GT-positive/negative masses, and `G_c=sum_b t_cb>0`.
Relax the family to arbitrary independent class-wise bit rules `h_c(b)`, while
fixing the unanimous endpoints:

    h_c(0,...,0)=0;  h_c(1,...,1)=1.
    J_c,signature^* = max_h sum_b t_cb h(b)
                            / [G_c + sum_b f_cb h(b)].

At a trial q_c, its exact residual is

    max_h sum_b (t_cb-q_c f_cb) h(b) - q_c G_c.

The free signatures independently take bit one when `t_cb-q_c f_cb>0`.
Mandatory endpoints must remain fixed even when their coefficient has the
opposite sign; ties are handled consistently. Exact residual maximization
and ratio updates certify this finite single-class relaxation. Averaging
those independent class optima gives

    max_global_family macro J
        <= mean_c J_c,signature^*
        <= mean_c [(G_c-N_c)/(G_c+F_c)]
        <= 1.

The last relaxation freely gives every disagreement pixel its GT decision;
it cannot exceed the TP ceiling `G_c-N_c` or improve the FP floor `F_c`.
All sources and all canonical recipes satisfy the first relaxation, but
the shared global recipe and its cost constraints can make that relaxation
unattainable. The source-signature relaxation itself can be strictly below
the consensus bound: an optional signature with both t>0 and f>0 must either
lose those true pixels or retain those false pixels, whereas the consensus
relaxation independently fixes both kinds.

For a per-query adaptive source-subset family, group by `(episode,signature)`
and allow `h_e,b` in the relaxation instead. Its pooled single-class residual
still uses **one class q_c** across episodes. Each image's own IoU-optimal q_e
would optimize a different objective. Every allowed episode's origin must
remain included in its bank, with edits derived from that same bank.

This is a finite-cohort GT-informed capacity bound of a bit readout. It is
neither a prevalence-estimation method nor a Bayes value for the expectation
of a random finite benchmark. Mixed signature labels do not establish an
information obstruction for allowed RGB/full features or an expanded bank.

**Prediction:** exact same-cohort canonical family scores cannot exceed the
signature relaxation, and the latter cannot exceed the consensus relaxation.
Any violation rejects the family/count/metric implementation contract.
No source counts were inspected or evaluated for this proof.

## 10. Do the proposed checks actually falsify their claims?

The checks are unexecuted. They may use GT as a labelled diagnostic after
forecast/field sealing; that does not turn GT into an inference input. Reused
4000/600 are development evidence. No preparation, oracle or expected model
score replaces observed complete-mask I/U on the declared cases.

| Proposed check | What can actually reject it | Boundary / required observable target |
|---|---|---|
| Exact edit accounting, histogram solver, branch bounds | An implementation's final class I/U differs from its exact effective-set accounting or valid exhaustive answer | A mathematical identity is not refuted by data; a failure rejects the implementation or its contract. Singleton marginals alone cannot decide the full overlapping family. The dominance counterexample is already a paper proof of an invalid pruning rule. |
| Finite-sample false-inclusion explanation | Sealed proxy gates show wrong held-data true marginals inconsistent with a prespecified representative, zero-bias/noise model; changing effective selection n fails its stated acceptance/gain prediction | Estimate proxy noise and helpful/harmful types on separate development evidence, then read actual accepted masks' class I/U and four edit counts. Reusing proxy-derived expected gains to define both “truth” and forecast is circular. Leave-fold GT recipe stability concerns development-selection noise; it does not establish independent reference pixels or a no-label per-query proxy. |
| Graph denoising / exposure explanation | A specified continuous residual model or quantitative exposure prediction fails on observed GT/token coverage and retained fixed fields | Conservation/range equalities check the solver without GT. Signal/noise-mode checks can reject the stated estimation model but cannot alone predict IoU. Exposure correlations without a defined approximation/margin tolerance are a mechanism diagnostic, not a sufficiency proof. |
| No useful post-RCG cue information | A predeclared cue changes a decision relevant to the same domain/budget and yields positive held-data complete utility beyond the declared equivalence tolerance | A null chosen rule does not prove full utility sufficiency. Unchanged rank guarantees identical top-k masks only with the same eligible domain, k and tie rule; unchanged rank can still coexist with different threshold/budget masks. Positive posterior or Brier improvement can leave IoU unchanged. |
| Invariant score components / identifiable occupancy prediction | Frozen transported component laws violate a prespecified query-law radius, or sealed pi/cut/gain forecasts violate the derived uncertainty on observed GT outcomes | Fitting the two components anew using each tested query's GT makes the mixture identity tautological. Separation, transport error and pi must refer to the exact scoring operator/observation space. To test a gain forecast, compare actual class I/U, not only the model's expected-count objective. |
| Three-view conditional-independence model | Label-conditional full-joint residuals, invalid reconstructed rates, or additional-view overidentification restrictions exceed the declared radius | Three binary views are generically just identified. A good unlabelled fit cannot certify independence or semantic orientation. Pairwise conditional residuals alone miss higher-order dependence. A failed model does not establish full-information impossibility. |
| Fixed-score threshold headroom | An exact same-cohort class-metric oracle exceeds the alleged per-image-oracle ceiling, or an implementation violates the optimum certificate | Exact class oracle >= feasible per-image oracle is an inclusion theorem. This diagnoses finite threshold capacity; it estimates neither a known-law Bayes gap nor a universal representation ceiling. |
| Frozen method superiority | A prespecified paired class-macro contrast is inconsistent with the declared benefit/equivalence target on the permitted held cohort | Compare complete observed masks with the strongest feasible same-information control. A CI crossing zero is unresolved; label fitting, class coverage and all library/selection exposure remain explicit. A failure limits that frozen recipe, not the optional joint family. |

The shared photo-group influence planning rule is appropriate as a stated
normal approximation. It is a **future planning condition**, not a proof that
existing4000/600/1200 can falsify every row. Proxy-estimation variance,
component-law uncertainty, spectral-model error and complete-mask paired
evaluation variance are different quantities. A numerical sample count or
equivalence verdict needs the corresponding variance/radius and a
prespecified decision margin. No such quantities were re-estimated here.

## Final claim boundary

The theory supports an exact finite-family development solver, conditional
utility/calibration and graph-estimator results, and explicit falsifiable
premises for selection. It does not yet identify an adaptive same-information
selector, guarantee that a sparse joint recipe generalizes, prove RCG semantic
sufficiency, or bound irreducible loss on actual COCO. Those are limits of the
present inference, not impossibility claims about the research objective.
