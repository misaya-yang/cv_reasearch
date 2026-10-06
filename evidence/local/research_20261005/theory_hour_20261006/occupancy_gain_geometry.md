# Occupancy, threshold geometry, and attribution boundaries

Theory-only memo, 2026-10-06 UTC. Owner: `/root/occupancy_gain_geometry`.
Scope: one complete masked reference, frozen DINOv3, COCO-20i 1-shot, no training;
the benchmark first sums intersection and union within each class and then averages classes.
No experiment, evaluation, model call, GPU task, remote operation, or empirical recomputation
was performed for this memo. Reported numbers below come from the parent's supplied brief.

Labels used throughout:

- **Strict:** algebraic identities or implications of the definitions.
- **Conditional:** a result that requires the stated distribution, regularity, or sampling assumptions.
- **Conjecture:** an explanation consistent with the supplied observations but not established by them.

## Core conclusions

1. **Conditional:** if the foreground and background conditional score laws are exactly known
   and invariant, the population optimal single-image threshold is determined by foreground
   prevalence. The decision uses likelihood ratio and the optimal IoU, not prevalence alone
   without the conditional laws.
2. **Strict:** class-summed IoU is a different objective from mean single-image IoU. Its optimal
   thresholds share a class-specific optimal IoU and are coupled across images. An oracle
   selected to maximize each image's IoU is not an upper bound on class-summed mIoU.
3. **Conditional:** a smooth, unique optimum gives a locally quadratic loss in log-odds
   prevalence error. Its coefficient depends on score density, likelihood-ratio slope,
   and the operating point. A factor-2 or factor-4 prevalence-error rule is not universal.
4. **Strict:** graph, fine readout, calibration, and ranking headroom cannot be added from
   different cohorts or objectives. Their gains telescope only along a specified complete
   pipeline on the same cohort. No total SOTA ceiling follows from the supplied gains.
5. **Correction to a common sufficient-condition statement:** strict monotone likelihood
   ratio is sufficient for a smooth raw-score cutoff that is Bayes optimal. Weak monotonicity
   of a unique optimum within an invariant *nested threshold family* actually needs less:
   the single-crossing argument below does not require monotone likelihood ratio. Neither
   result compares numeric cutoffs of differently calibrated score fields across images.

## 1. Single-image IoU and its optimal operating point

Choose a uniformly random evaluated pixel in one image. Let its truth be `Y in {0,1}`,
foreground prevalence be `pi = P(Y=1)`, and scalar score be `S`. Predict foreground when
`S >= t`. Define conditional survival functions

\[
A(t)=P(S\ge t\mid Y=1),\qquad B(t)=P(S\ge t\mid Y=0),\qquad
r=\frac{1-\pi}{\pi}.
\]

In plain English, `A` is true-positive rate, `B` is false-positive rate, and `r` is the
background-to-foreground prior odds. Assume `0 < pi < 1`.

**Strict:** intersection and union probabilities give

\[
J(t;\pi)=\frac{\pi A(t)}{\pi+(1-\pi)B(t)}
        =\frac{A(t)}{1+rB(t)}.
\tag{1}
\]

In plain English, every true foreground pixel enters the union, whether detected or missed;
only predicted background pixels add extra union. This is a ratio of population counts,
not generally the expectation of a finite sample's random IoU ratio. For a fixed labeled
image, the same identity is exact with empirical survival functions and empirical prevalence.

**Conditional:** if the conditional laws have densities `f1` and `f0` near an interior
optimum, differentiation gives

\[
\partial_tJ=
\frac{-f_1(t)[1+rB(t)]+rA(t)f_0(t)}{[1+rB(t)]^2}.
\tag{2}
\]

In plain English, raising the cutoff sacrifices true positives and saves false positives;
their marginal balance determines the stationary point.

If `f0(t*) > 0`, every differentiable interior optimum satisfies

\[
L(t^*)\equiv\frac{f_1(t^*)}{f_0(t^*)}=rJ^*,
\qquad
\frac{\pi f_1(t^*)}{(1-\pi)f_0(t^*)}=J^*.
\tag{3}
\]

In plain English, the boundary's likelihood ratio equals prior odds times optimal IoU;
the boundary's posterior foreground-to-background odds equal optimal IoU. This is a
stationarity condition, not a guarantee that every solution is the global maximum.
Boundary optima and atomic score laws require one-sided inequalities or explicit comparisons.

For arbitrary measurable decisions with posterior `eta(x)=P(Y=1|X=x)`, fractional
optimization supplies the stronger global decision rule

\[
D^*(x)=\mathbf 1\!\left\{\eta(x)>q^*\right\},
\qquad q^*=\frac{J^*}{1+J^*}.
\tag{4}
\]

In plain English, the optimal posterior threshold is below one half unless perfect IoU
is achievable. It is self-consistent because the selected mask determines `J*`.
To verify (4), maximize `I(D)-J*U(D)`: including a pixel contributes
`eta-J*(1-eta)`, and the maximum is zero at the optimum. Ties can be included or excluded.
With a raw scalar score, (4) is a single score cutoff when the relevant likelihood-ratio
superlevel set is an upper score tail; global monotone likelihood ratio is a sufficient
condition. Spatial constraints or a CRF can restrict the feasible decisions.

## 2. How the cutoff changes with prevalence

**Conditional on the fixed laws, no differentiability needed:** the critical likelihood
ratio for the threshold family's optimum has the equivalent form

\[
c(\pi)=\frac{1-\pi}{\pi}J^*(\pi)
=\max_t\frac{A(t)}{\pi/(1-\pi)+B(t)}.
\tag{5a}
\]

In plain English, increasing foreground prevalence enlarges the denominator for every
candidate cutoff, so the critical likelihood-ratio value is nonincreasing. This is a
statement about a likelihood-ratio decision price; translating it into a particular raw
score boundary requires the score/likelihood-ratio structure and tie convention below.

**Conditional, sufficient smooth version:** keep both conditional score laws fixed;
assume a unique interior optimum, positive densities, and strictly increasing `L(t)`
near every optimum. Let `z=logit(pi)` and `r=exp(-z)`. Envelope differentiation gives

\[
c(r)=rJ^*(r),\quad
\frac{dc}{dr}=\frac{A(t^*)}{[1+rB(t^*)]^2}>0,\quad
\frac{d\log c}{dz}=-\frac{1}{1+rB(t^*)},\quad
\frac{dt^*}{dz}=-\frac{1}{[\partial_t\log L(t^*)][1+rB(t^*)]}<0.
\tag{5}
\]

In plain English, more foreground permits a lower cutoff on the same calibrated score.
The amount of movement depends on the slope of the likelihood ratio; it is not a universal
linear function of foreground area. This argument does not assert that actual DINO score
laws are invariant across reference/query pairs.

**Conditional, weaker nested-family version:** for two fixed cutoffs `tH > tL`, write
`AH <= AL` and `BH <= BL`. Their comparison has a single crossing in prior odds:

\[
J(t_H;r)\ge J(t_L;r)
\iff (A_H-A_L)+r(A_HB_L-A_LB_H)\ge0.
\tag{6}
\]

In plain English, a higher cutoff can overtake a lower one only as the background odds
increase. Its intercept is nonpositive; if it ever wins, its slope is nonnegative.
Therefore unique maximizing cutoffs move weakly downward as prevalence increases even
without monotone likelihood ratio, when the conditional laws and nested threshold family
are fixed. Equivalent masks, empty score intervals, and arbitrary tie choices need a
canonical cutoff convention. Strict monotone likelihood ratio is a useful sufficient
condition for identifying this family's solution with the unrestricted Bayes decision and
obtaining (5); it is not necessary in every isolated case.
Across different score transformations or conditional laws, no such numeric-cutoff order follows.

**Conditional sufficiency boundary:** known invariant `(F1,F0)` plus prevalence specifies
the entire population objective (1), so prevalence is the only varying population parameter
for a single image in that model. This does not mean prevalence identifies an actual query's
unknown conditional laws, finite-image fluctuations, spatial structure, or class coupling.

## 3. The actual class-summed benchmark couples the images

For image `i` belonging to evaluated target class `c`, let `Ni` be its evaluated pixel count.
Keep its conditional laws and prevalence explicit. The benchmark objective is

\[
I_c(\mathbf t)=\sum_{i\in c}N_i\pi_iA_i(t_i),\qquad
U_c(\mathbf t)=\sum_{i\in c}N_i[\pi_i+(1-\pi_i)B_i(t_i)],\qquad
V(\mathbf t)=\frac1C\sum_cJ_c(\mathbf t),\quad J_c=I_c/U_c.
\tag{7}
\]

In plain English, large unions contribute more within a class, while every evaluated class
gets equal weight in the final average. This is not the mean of per-image IoUs. The formula
is exact for empirical counts; its differentiable population version uses expected counts.

**Conditional:** a differentiable independent threshold for image `i` has stationarity

\[
\frac{\partial J_c}{\partial t_i}
=\frac{N_i[-\pi_i f_{1i}(t_i)U_c+(1-\pi_i)f_{0i}(t_i)I_c]}{U_c^2}=0
\quad\Longrightarrow\quad L_i(t_i^*)=\frac{1-\pi_i}{\pi_i}J_c^*.
\tag{8}
\]

In plain English, replace the image's optimal IoU in (3) by the class's optimal IoU.
All images of one class share the posterior cutoff `qc*=Jc*/(1+Jc*)`, while their
likelihood-ratio cutoffs differ with their priors. Each threshold changes the common `Jc*`;
the class's other images therefore matter even when this image's prevalence is known exactly.
If one common scalar cutoff is imposed on all images, the summed derivative must vanish;
one cannot impose each image's equation (8) separately.

**Conditional sufficiency boundary for the benchmark:** the known conditional laws, full
vector of query prevalences, pixel counts, and evaluated class membership specify (7).
No additional unknown population statistic is needed to solve that idealized objective.
However, a single query's prevalence does not determine its class's common operating point
without the rest of that class's benchmark context. `Jc*` summarizes the metric tradeoff;
it is not an independently supplied pixel-level semantic signal.

**Strict counterexample:** image 1 has 10 true foreground pixels. A conservative threshold
gives `(TP,FP)=(5,0)`, hence IoU `5/10`; a lower threshold gives `(10,6)`, hence IoU `10/16`.
Image 2 of the same class gives `(TP,FP)=(100,0)`. The image-1 oracle selects the lower cutoff,
but the class IoU falls from `105/110` to `110/116`. Thus improving every changed image's
own IoU can worsen the benchmark, and the reported per-image-threshold oracle's benchmark
score is an achieved diagnostic result, not the maximum over all threshold combinations.

**Strict interpretation of the supplied numbers:** `62.33 -> 71.24`, or `+8.91`, establishes
large achieved headroom in that privileged threshold construction. It upper-bounds each
image's fixed-score IoU if all its admissible thresholds were searched. Unless the search
maximized (7) jointly, `71.24` is not an upper bound on the class-summed benchmark.
The true-prevalence formula's `+5.07 [4.27,5.51]` is the result of that particular formula,
not a proof that all prior-based policies are bounded by `+5.07`.

### 3.1 An actual threshold-family benchmark upper bound

**Strict:** for fixed score fields and a finite admissible cutoff set `Ti` in each image,
the class-optimal privileged threshold score can be defined without changing the metric:

\[
J_{c,T}^*=\max_{(t_i\in T_i)_{i\in c}}
\frac{\sum_{i\in c}I_i(t_i)}{\sum_{i\in c}U_i(t_i)},\qquad
H_c(q)=\sum_{i\in c}\max_{t_i\in T_i}[I_i(t_i)-qU_i(t_i)].
\tag{9}
\]

In plain English, trial class IoU `q` makes the inner problem independent across images;
each image selects the cutoff with the best intersection-minus-priced-union value.
Since every union is positive, `Hc(q)` is positive below the optimum, zero at it, and
negative above it. A Dinkelbach update selects those cuts and replaces `q` with their
summed `I/U`. Averaging these class optima is an actual upper bound for that fixed family,
provided no shared policy, common-parameter, or inference-budget constraint couples the cuts.
The per-image IoU oracle is a feasible candidate in (9); hence its `71.24` score is at most
this benchmark optimum, rather than an upper bound on it. This computation is a possible
future GT diagnostic from existing cutoff counts; it was not run during this theory task.

## 4. What prevalence-estimation error can cost

### 4.1 Smooth local regret

**Conditional:** use the same known invariant laws, a unique interior optimum, positive
density, strictly increasing likelihood ratio, and twice continuously differentiable IoU.
Let the estimator change only the prevalence parameter; no score field or conditional law
changes. Put `e=logit(pi_hat)-logit(pi)`, and use the optimal cutoff for `pi_hat` on the true
image population. Taylor expansion around the true optimum gives

\[
J(t^*(\pi);\pi)-J(t^*(\widehat\pi);\pi)
=\frac12K(\pi)\,[t_z^*(\pi)]^2e^2+o(e^2),\qquad
K=-J_{tt}(t^*;\pi)>0.
\tag{10}
\]

In plain English, infinitesimal prior errors cost second order because the true optimum's
first derivative vanishes. This statement is local; a finite factor error can cross a
cutoff switch, leave the regular neighborhood, or encounter an endpoint.

Using (2)-(5), the curvature and coefficient are explicit:

\[
K=\frac{f_0(t^*)L'(t^*)}{1+rB(t^*)},\qquad
J^*-J_{\widehat\pi}
=\underbrace{\frac{f_0(t^*)L(t^*)^2}
{2L'(t^*)[1+rB(t^*)]^3}}_{k(\pi,F_1,F_0)}e^2+o(e^2).
\tag{11}
\]

In plain English, the same log-odds error can be almost harmless in one image and costly
in another. The coefficient contains the density and likelihood-ratio slope at the exact
operating point. A nearly flat likelihood ratio makes the smooth cutoff unstable and can
shrink the range in which this Taylor approximation is useful. Atomic scores, tied masks,
and zero curvature are outside this derivation.

### 4.2 A margin bound without differentiating the optimum

**Conditional:** allow any measurable likelihood-ratio decisions, keep the two conditional
laws exactly fixed, and use `H=log L` as the decision statistic. Define
`a(z)=log[r J*(z)]`; the optimal foreground decision is `H > a(z)`. For any fixed decision
region, `r A/(1+r B)` is increasing in `r`, and its log slope with respect to `log r` lies
between zero and one. Taking the maximum over regions preserves the bounds, so

\[
|a(\widehat z)-a(z)|\le |\widehat z-z|=|e|.
\tag{12}
\]

In plain English, a pure prior error moves the optimal boundary by at most that error on
the log-likelihood-ratio scale. This remains a conditional statement about exactly known
laws; it does not protect a misspecified likelihood-ratio estimate.

**Strict identity:** for the true optimal posterior `eta` and `q*=J*/(1+J*)`, any decision
`D` satisfies

\[
J^*-J(D)
=\frac{(1+J^*)\,\mathbb E[|\eta-q^*|\mathbf1\{D\ne D^*\}]}{U(D)}.
\tag{13}
\]

In plain English, regret is the disagreement's posterior margin, divided by the resulting
union. Verify the identity by subtracting `I(D)-J*U(D)` from its zero value at `D*`;
the integrand is `(eta-J*(1-eta))(D*-D)`, which has the stated sign.

The logistic map has derivative at most one quarter. If, under the true image population,
the posterior obeys the margin condition `P(|eta-q*| <= u) <= C_m u^alpha` for small `u`,
then (12), (13), and `U(D) >= pi` imply, for errors inside that margin regime,

\[
0\le J^*-J_{\widehat\pi}
\le\frac{(1+J^*)C_m}{\pi}\left(\frac{|e|}{4}\right)^{1+\alpha}.
\tag{14}
\]

In plain English, prior errors matter most when much probability lies near the optimal
posterior boundary. A regular density often gives `alpha=1` and a quadratic order; no
positive margin exponent is automatic. The bound can be loose for rare foreground and
should be clipped by the trivial loss bound of one. It measures loss from the population
oracle, rather than gain or loss relative to the current deployed fixed cutoff.

### 4.3 Why factor-2 and factor-4 observations are not general tolerances

**Strict:** if `pi_hat=k pi` and `0 < k pi < 1`, the error entering these bounds is

\[
e=\log k+\log\frac{1-\pi}{1-k\pi}.
\tag{15}
\]

In plain English, multiplicative foreground-area error approximates log-odds error only
for rare foreground. An identical factor error near large occupancy can be much larger
on the relevant scale, and some multiplication factors are not even valid probabilities.

The supplied observations—about `+3.9` for errors within factor 2 and a negative result
above factor 4—are useful empirical sensitivity evidence for that tested rule and cohort.
They are not universal sufficient or necessary error tolerances. Neither (10) nor (14)
turns those factors into a new method's gain forecast without the operating-point density,
margin, score-law drift, and error distribution.

### 4.4 Estimation errors can concentrate where they matter most

**Conditional local accounting:** in an image-wise diagnostic, let `ki` be the coefficient
in (11) and `ei` its log-odds error. Its average leading regret contains

\[
\mathbb E[k_i e_i^2]
=\mathbb E[k_i]\mathbb E[e_i^2]+\operatorname{Cov}(k_i,e_i^2).
\tag{16}
\]

In plain English, a good average area estimate can still perform badly if its large errors
occur on images with sensitive boundaries. Independence is not needed for the identity;
assuming it would discard the covariance term. These image-average weights are not the
benchmark weights; class-summed loss must be computed from the actual class intersections
and unions and its shared optimal cutoffs.

**Conditional extension to score-law drift:** if a smooth cutoff also uses misspecified
conditional laws, its first-order displacement has the form

\[
\delta t\simeq t_z^*e+T_F(\widehat F-F),\qquad
\text{regret}\simeq\tfrac12K[\,t_z^*e+T_F(\widehat F-F)\,]^2.
\tag{17}
\]

In plain English, prior error and conditional-law error can amplify or cancel each other;
their dependence matters. `TF` is the local derivative of the optimum with respect to the
conditional laws, when this derivative exists. A heuristic cutoff that is not itself
optimal has a nonzero first-order gradient and need not even enjoy the quadratic protection.

**Conjecture, not an established cause:** the supplied `mask-area -0.58` despite moderate
prevalence error is compatible with score-dependent estimator errors, score-law drift,
class-objective mismatch, or an inadequate mapping from prevalence to cutoff. It does not
isolate which link failed. Reusing the score field to estimate its own area makes independence
implausible, but does not prove harmful covariance from the result alone.

### 4.5 A local regret formula for the actual coupled class objective

**Conditional:** for a class with independently adjustable cutoffs, assume a unique smooth
interior optimum, positive densities, and `Li' > 0` at its boundaries. Conditional laws and
pixel counts stay fixed; every estimated prior is perturbed by `ei` on the log-odds scale.
Envelope differentiation of the class optimum gives

\[
g_j=\frac{\partial\log J_c^*}{\partial z_j}
=\frac{N_j\pi_j(1-\pi_j)[A_j(t_j^*)-J_c^*(1-B_j(t_j^*))]}{I_c^*},
\qquad
\delta\log J_c^*\simeq\sum_{j\in c}g_je_j.
\tag{24}
\]

In plain English, the class's optimal operating point responds to all of its images'
priors. The sign of an individual `gj` need not be positive: increasing the ground-truth
weight of a difficult image can lower that class's optimum. This is not the single-image
monotonicity claim.

Differentiating the class stationarity relation in (8), with `elli'=d log Li/d ti`, gives

\[
\delta t_i\simeq\frac{\sum_{j\in c}g_je_j-e_i}{\ell_i'}.
\tag{25}
\]

In plain English, the error in an image's cutoff contains both its own prior error and
the shared class-operating-point error. A scalar per-image area error is not sufficient
to predict the benchmark loss.

At the unconstrained interior optimum, the mixed second derivatives between distinct
image cutoffs vanish, while the diagonal curvature is
`-Jc,tii = Ni*pi_i*f0i*Li'/Uc*`. Consequently,

\[
J_c^*-J_c(\widehat{\mathbf t};\boldsymbol\pi)
\simeq\frac{1}{2U_c^*}\sum_{i\in c}
\frac{N_i\pi_i f_{0i}(t_i^*)L_i(t_i^*)^2}{L_i'(t_i^*)}
\left(\sum_{j\in c}g_je_j-e_i\right)^2.
\tag{26}
\]

In plain English, local class loss is a weighted sum of squared cutoff errors, but those
errors are correlated through the shared operating point. Averaging (26) over classes
matches the benchmark objective. For a single-image class, `g=rB/(1+rB)`, and (26)
reduces exactly to (11). Common-cutoff or shared-parameter restrictions alter this local
geometry and invalidate the independent-cutoff curvature formula.

## 5. Interpreting the supplied occupancy and tail evidence

The following are supplied observations, preserved as given. This memo did not read a
fresh metric receipt or recompute any result. Missing fields remain missing.

| Supplied observation | What it supports | What it does not establish |
|---|---|---|
| RCG `62.33 -> 71.24`, per-image best-IoU cutoffs, `+8.91` | An achieved privileged cutoff-policy improvement on the fixed score fields | The exact class-summed threshold-family maximum; deployable gain; a representation ceiling |
| True-prevalence invariance formula `+5.07 [4.27,5.51]` | True area is actionable for this tested rule | Universal optimal use of prevalence; a `+5.07` upper bound; invariance of query conditional laws |
| Error within factor 2 gives about `+3.9`; above factor 4 becomes negative | Sensitivity of that specific area-to-cutoff policy | A distribution-free factor tolerance |
| Small truth occupancy `<2%`: predicted/truth area ratio `2.94`; large occupancy `>30%`: ratio `.81` | A supplied occupancy-dependent calibration pattern | A causal proof that area alone repairs it; conditional-law invariance |
| Worst 10% cases account for 60% of FP and 71% of FN | Raw error concentration and a reason to examine tail reliability | Those fractions of the class-macro score loss, or easy repairability |
| Mixture-histogram estimator `-4.08`; reference-area correlation `.25`; geometry-scale and reference-feature mixture approaches failed, mixture estimates often zero; mask-area method `-.58` | Failure of the tested constructions and evidence against assuming reference/query feature transport | Non-identifiability under every valid model; impossibility of a different same-information estimator |
| Frozen new1200 complete RCG + fine + size-cut versus FoRIS `+2.49 [1.47,3.47]`; fine contribution about `+.4` | A supplied complete-method result in its exact setting | A zero-fitting result if the size cuts used other-fold labels; interchangeability with other1200/4000 cohorts |

The exact seed, resolution, fold gains, and oracle/estimator cohort counts are not all
specified in the parent brief. No values are inferred to fill them. The complete new1200
row has count 1200; its other-fold threshold fitting must remain explicit. Current live
repository rows with similar names and different numbers are not substitutes for this row.

**Strict:** the predicted foreground fraction at cutoff `t` is

\[
\widehat a(t)=\pi A(t)+(1-\pi)B(t),\qquad
\frac{\widehat a(t)}{\pi}=A(t)+\frac{1-\pi}{\pi}B(t).
\tag{18}
\]

In plain English, a nonzero false-positive rate is multiplied by large background odds
for tiny targets, so overprediction can be severe. Large targets can be underpredicted
when true-positive rate is imperfect. This is a sufficient structural explanation of how
the given ratios can arise, not a fit or causal attribution of `2.94` and `.81`.

**Strict:** the value of an edit is class-specific. For an addition with `a` TP and `b` FP,
or a deletion with `d` TP and `f` FP, respectively,

\[
\Delta J_c^{\rm add}=\frac{a-J_cb}{U_c+b},\qquad
\Delta J_c^{\rm del}=\frac{J_cf-d}{U_c-f}.
\tag{19}
\]

In plain English, a false positive in a large-union class and a false positive in a small-union
class do not have equal macro value; deleting true pixels has an explicit cost. Therefore
the worst-case FP/FN percentages identify an error pool, but cannot be read as the same
percentage of recoverable mIoU. Masks with the same area can have opposite edit value.

**Conditional:** if `(F1,F0)` were known and different, a query's exact marginal score law
would identify its prevalence. For an event `E` on which the laws differ,

\[
\pi=\frac{P(S\in E)-P(S\in E\mid Y=0)}
{P(S\in E\mid Y=1)-P(S\in E\mid Y=0)}.
\tag{20}
\]

In plain English, two known distinct components identify their mixture weight. If the
components are unknown or drift between reference and query, a histogram alone need not
identify the weight; a small denominator is also unstable. The often-zero reference-feature
mixture estimates are evidence against that tested transport construction, not a theorem
about every possible query cue. Other agents own the broader identifiability analysis.

## 6. A defensible decomposition of graph, fine-readout, and calibration gains

**Strict:** let `V_D` be the same class-summed metric on one fixed cohort `D`, and let
`M0, M1, M2, M3` be complete masks along a specified pipeline. For example, graph processing,
then fine readout, then a frozen calibration rule may define the successive masks. Then

\[
V_D(M_3)-V_D(M_0)=
[V_D(M_1)-V_D(M_0)]+[V_D(M_2)-V_D(M_1)]+[V_D(M_3)-V_D(M_2)].
\tag{21}
\]

In plain English, the gain can be added only when every term is a consecutive difference
of the same complete pipeline, same cohort, same pixel protocol, and same metric. Component
changes can affect each other's marginal value. Their percentages need not be separate
budgets, and changing the order can change the attribution.

The supplied `+1.41`, true-area formula `+5.07`, per-image cutoff oracle `+8.91`, and fine
readout about `+.4` must not be summed. They are not all consecutive differences, and their
cohort/method boundaries have not all been supplied. Even on one cohort, true-area and
oracle-cutoff gains are alternative interventions on the same fixed score field and can
repair overlapping pixels. The `8.91-5.07` difference, if those rows are truly matched, is
a difference between two cutoff policies; it is not a demonstrated ranking or representation
gap. It can include law misspecification, finite-image variation, and objective mismatch.

**Strict:** for a genuinely matched two-factor complete-mask comparison, interaction is

\[
\Gamma_{GF}=V_D(G,F)-V_D(G,0)-V_D(0,F)+V_D(0,0).
\tag{22}
\]

In plain English, interaction measures what is missing when standalone gains are added.
Adding calibration introduces its main effect, pairwise interactions, and a three-way
interaction. A fixed-order telescoping report is already exact; a factorial or Shapley
allocation is optional when order-independent attribution is wanted. It does not create
new evidence or remove data/resource mismatches.

The relevant roles and limits are:

- **Graph processing:** can denoise, change local agreement, suppress distractors, and
  reorder scores. It can also remove true pixels or change calibration. Its theoretical
  role is not restricted to one mutually exclusive error category.
- **Fine readout:** can increase spatial precision and change ranking at object boundaries
  or between regions. A measured roughly `+.4` is a marginal gain in its tested context,
  rather than a ceiling on fine geometry or an additive amount in every pipeline.
- **Threshold calibration:** chooses an operating point for a given ranking. The `+8.91`
  and `+5.07` results diagnose two distinct privileged policies; only (9), evaluated on
  the same score field and cohort, would bound the threshold family's benchmark score.
- **Remaining ranking/information:** an exact fixed-score threshold optimum cannot recover
  foreground that is below too many background pixels in that ordering. Changing the
  score, combining fields, fine geometry, and contextual inference can change this limit
  while keeping the encoder frozen. Failure of a ranking does not establish failure of
  all information in the frozen representation.

**Conditional population hierarchy:** for a fixed observable representation `Z`, a scalar
score `S(Z)`, and an explicitly defined population objective,

\[
V^*_{\text{thresholds of }S}\le
V^*_{\text{all measurable decisions from }Z}\le1.
\tag{23}
\]

In plain English, restrictions on the readout cannot increase its optimal population value.
The middle quantity is an information-level Bayes envelope, requiring the true conditional
law. The finite-cohort GT oracle is a different object: arbitrary empirical memorization
would make an unrestricted mask oracle trivially perfect. None of the supplied fixed-score
oracle results estimates the middle quantity or supplies a universal total SOTA limit.

**Resource boundary:** a size-cut policy fitted using other-fold GT masks may have no encoder
training, yet it still uses labeled task data to fit a readout. It belongs in an explicitly
label-fitted/cross-fold setting. A cutoff obtained from query GT is a privileged diagnostic.
Neither should silently be promoted to the strict one-reference, no-base-mask-fitting claim.

## 7. Falsifiable predictions and the smallest future comparison

This section proposes measurements for a later, explicitly authorized experiment; none
was run, queued, or restarted. It does not select a new estimator or change the supplied method.

**Conditional prediction A:** if the invariance model is a useful approximation, then after
holding the score construction fixed, cutoffs predicted from true prevalence and known
conditional laws should approach the *joint class-optimal* threshold choices, not necessarily
the per-image best-IoU choices. Large systematic conditional-law differences at the same prior
would directly attack the premise. The correct class-optimal oracle in (9) is the first
diagnostic to obtain from existing cutoff counts; the existing `71.24` is not that quantity.

**Conditional prediction B:** for an actual frozen estimator, equal log-odds area errors should
hurt more on sensitive or high-margin-mass images than on insensitive ones. The complete
class-macro effect and four pixel edit counts decide utility; area MAE or correlation alone
cannot. Failure after conditioning on those quantities would point toward misspecified laws,
an incorrect area-to-cutoff rule, or shared policy constraints.

**Conditional prediction C:** a fine readout that only repairs boundary sampling should earn
its largest incremental benefit where the coarse representation is spatially ambiguous.
If it instead reorders semantically incorrect regions, both its threshold-family optimum
and ranking diagnostics can change. This is a hypothesis about the tested readout, not a
guaranteed additive `+.4` forecast.

The smallest deciding complete-mask comparison is one frozen candidate calibration rule
using its actual authorized estimator, versus the unchanged complete baseline and the
strongest existing complete same-information control, plus a simple policy that uses the
*same area estimate* to choose its score quantile. A method using fitted conditional laws
must give that control the same fitting resources. True-prior and class-threshold oracles
are privileged diagnostic rows alongside the comparison, rather than independent methods.
Use one matched, photo-disjoint confirmation cohort after selection; retain the exact
class set, fold protocol, resolution, and estimator version. Existing score fields can
support this comparison without an additional encoder pass only when their source identity
and readout precisely match the frozen method.

The decision is whether the candidate improves the complete class-summed score beyond its
strongest same-information control, with a sufficiently narrow paired interval, and whether
the improvement survives the tail cases instead of exchanging false removals for missed
targets. A failure of the invariance premise changes the model; a failed estimator changes
the estimator; a failed complete score limits the tested construction. These are separate
outcomes of one comparison, not reasons to schedule a sequence of preliminary efficacy probes.

### 7.1 Episode count must follow paired uncertainty

**Conditional planning approximation:** independent statistical units are photo-connected
groups, or the corresponding prespecified strata; pixels and reused episodes are not assumed
independent. For a method `m`, let group `g` contribute `Igc,m` and `Ugc,m` in class `c`.
With `G` exchangeable groups, its class-macro influence value is

\[
\psi_g^{(m)}=\frac1C\sum_c
\frac{I_{gc}^{(m)}-J_c^{(m)}U_{gc}^{(m)}}{U_c^{(m)}/G},\qquad
\psi_g^\Delta=\psi_g^{(\mathrm{candidate})}-\psi_g^{(\mathrm{control})}.
\tag{27}
\]

In plain English, paired uncertainty depends on how the same connected photo groups change
both methods' class ratios. Their covariance matters. A stratified or nonexchangeable design
needs its own weighted influence values or the prescribed connected-photo bootstrap, rather
than treating all episodes as independent equal-weight IoU observations.

If prior matched DEV receipts supply a relevant paired variance `sigma_psi^2`, a normal
approximation for desired 95% half-width `h` gives

\[
G\gtrsim\frac{z_{.975}^2\sigma_\psi^2}{h^2};
\qquad
G\gtrsim\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\sigma_\psi^2}{\delta^2}
\quad\text{for power }1-\beta\text{ at a paired effect }\delta.
\tag{28}
\]

In plain English, group count follows the paired CI precision or decision margin. The
episode count is obtained from the actual protocol and group structure afterward. If the
decision is a gain above a nonzero target, `delta` means distance from that target, not the
entire gain. Rare-class coverage and exact sampling constraints may require more cases.
The brief supplies no appropriate paired influence variance, so a numerical minimum `n`
cannot be justified here. It must come from matched receipts and the planned metric,
with no optional stopping or repeated use of exposed data as fresh confirmation.

## 8. Independent algebra checks and edge cases

These checks were done by hand on the definitions; no data or numeric experiment was used.

- Quotient differentiation of (1) gives (2); substituting `A'= -f1`, `B'= -f0` into the
  numerator independently gives the same balance in (3).
- Write `h=L(1+rB)-rA`. At stationarity `h=0`, and `ht=L'(1+rB)` because the density
  terms cancel. This gives both `Jtt=-f0 L'/(1+rB)` and the implicit derivative in (5),
  independently checking (11).
- With no signal, `F1=F0`, predicting all pixels maximizes single-image population IoU
  at `pi`; the strict-likelihood-ratio and unique-interior assumptions fail, so the smooth
  sensitivity formula correctly does not apply.
- With perfect separability, many numeric cutoffs can give IoU one; unique-cutoff and
  positive-boundary-density assumptions fail. A raw cutoff can change without any mask loss.
- At `pi=0`, foreground-normalized quantities and log odds are undefined; at `pi=1`, the
  background likelihood ratio is not an identifiable operating statistic. The target
  derivations explicitly exclude those endpoints.
- Equation (6) uses only nested survival pairs, so its weak comparative statics survive
  non-monotone raw-score likelihood ratios. In contrast, unrestricted Bayes foreground
  regions may be disconnected in raw-score space.
- In (9), `Ic(t)-qUc(t)` includes the constant ground-truth union term. Omitting it would
  keep the inner selected cuts for a fixed `q` but give the wrong residual and root.
- The explicit two-image example proves that the image-wise oracle's aggregate score
  need not upper-bound or improve the benchmark, despite improving its changed image.
- Equation (13) follows from a zero optimal fractional residual and signed disagreement;
  the result would not hold with the absolute-margin form for a spatially constrained
  optimizer whose decisions need not agree with the posterior sign.
- At the free class optimum, `Ii'=Jc Ui'` for every image. Differentiating the class ratio
  shows each cross-image second derivative cancels. Its diagonal is
  `(Ii''-Jc Ui'')/Uc=-Ni*pi_i*f0i*Li'/Uc`, checking (26). Coupling remains in (24)-(25).
- The add/delete expressions in (19) keep the unchanged truth in the union and exactly
  match subtraction of the two class fractions. They demonstrate why count concentration
  and area accuracy alone cannot be a universal gain rule.

The usable theoretical result is a precise occupancy-to-operating-point model with an
explicit class-objective correction and conditional regret geometry. Its unknown empirical
links are transport of conditional score laws, trustworthy query occupancy, and actual
complete-mask improvement beyond the strongest matched control. The supplied results
provide evidence about those links, not a proof of a total upper limit or a finished selector.
