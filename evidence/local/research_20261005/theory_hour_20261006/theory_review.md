# Adversarial review of the one-hour theory

Owner: `/root/theory_adversarial_review`. Theory-only review, 2026-10-06 UTC.
Scope: the supplied COCO-20i 1-shot setting, one complete masked reference,
frozen DINOv3, no training; intersection and union are summed by class before
macro averaging. No model call, experiment, data evaluation, remote operation,
or GPU work was performed. Only this note was written.

Reviewed: `main_theory.md`, `solver_theory.md`,
`occupancy_gain_geometry.md`, `identifiability_theory.md`, and
`graph_estimator_theory.md`. Numerical observations below are the parent's
supplied facts or another theory note's explicitly retained historical report;
they were not independently remeasured here.

Labels: **Strict** means a mathematical consequence of stated definitions;
**Conditional** means the additional assumptions are necessary to the stated
argument; **Conjecture** means an empirical explanation that remains untested.
Predictions below are implications to assess later, not a proposed new method or
authorization to run a check.

## Verdict

The finite-family accounting and solver results are defensible. They give an
exact development optimum for a specified mask family and objective, not a
no-GT adaptive selector or a generalization certificate. The notes correctly
reject a universal sparsity theorem, universal graph sufficiency, a dataset-
specific impossibility claim from failed estimators, and cross-cohort addition
of gains. Three qualifications must survive the final synthesis:

1. The main note's claim that monotone likelihood ratio is *required* for the
   threshold-family optimum to decrease with prevalence is too strong.
2. The identified-set diameter bound requires common fixed component laws;
   it does not cover unrestricted alternative component/model fits.
3. Identifying prevalence and a posterior conditional on one scalar score
   does not identify arbitrary spatial/feature-conditioned A/B edit utilities.

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

**Correction target:** `main_theory.md` section 12, sentence “A decreasing
raw-score threshold as pi increases additionally requires a monotone
likelihood ratio.” The nested-family argument in `occupancy_gain_geometry.md`
section 2 already contains the correct weaker result.

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
They do not by themselves establish the >=2 lower-bound target, superiority
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

## Final claim boundary

The theory supports an exact finite-family development solver, conditional
utility/calibration and graph-estimator results, and explicit falsifiable
premises for selection. It does not yet identify an adaptive same-information
selector, guarantee that a sparse joint recipe generalizes, prove RCG semantic
sufficiency, or bound irreducible loss on actual COCO. Those are limits of the
present inference, not impossibility claims about the research objective.
