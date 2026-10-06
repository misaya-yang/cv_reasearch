# What RCG can absorb: an anchored graph estimator, not a sufficiency theorem

2026-10-06 UTC. Owner: `/root/graph_estimator_theory`.

Scope: the third theory question in the user's one-hour theory-only request. This
note reads code and existing text; it runs no model, GPU/remote job, CPU data
evaluation, sweep, fitting, or training. Proposed checks below are unexecuted
designs, not an experiment queue or authorization to resume one. It proposes no
new inference method.

The target setting is COCO-20i 1-shot: one reference image with its complete
target mask, a frozen DINOv3 encoder, and no training, extra images, class-name
input, mask-pretrained model or base-class mask fitting. Additional views of the
same query still have a compute/input-view cost that must be reported.

## Conclusion and evidence boundary

RCG combines a reference-adjusted unary with a query-affinity prior. Its core is
an anchored graph Laplacian estimator: it can combine mutually supporting weak
evidence and attenuate disagreement with the graph. This explains why another
similarity or grouping rule can have little *incremental decision value* after
RCG. It does **not** prove that all evidence from the frozen encoder has been
used, or that every auxiliary rule must fail.

Three distinct propositions must stay separate:

1. **Exact algebra:** fixing the graph and positive anchor weights, finite-
   strength RCG is an invertible soft filter of its input unary. No nonzero
   spectral mode is exactly discarded in exact arithmetic.
2. **Conditional interpretation:** RCG helps if the useful unary signal is
   smoother on its graph than the errors it needs to suppress. A coherent
   erroneous object may be preserved; a true object weakly supported by anchors
   may be erased. Graph construction is itself a signal-bearing assumption.
3. **Unestablished statistical claim:** “all extra evidence is absorbed” would
   require a conditional sufficiency or decision-optimality result. Neither the
   convex solve nor failures of the tested auxiliary rules establishes it.

The observations supplied for this theory task are taken as given observations,
not re-evaluated here:

| Given observation | What it supports | What it does not establish |
|---|---|---|
| Complete 4000 RCG versus FoRIS: +1.41, paired 95% CI [1.08, 1.70], four positive fold gains | A positive complete-method effect in the reported setting | This note does not independently confirm public canonical protocol or original-resolution alignment; this is not a 4000 stage ablation |
| Historical 600: graph only +2.64 versus no-CRF input; reference correction alone approximately zero; joint +3.29 | Most of this tested gain is associated with the graph; the reference correction has conditional value | +3.29 is not the gain over complete FoRIS, and the two stages are not independent additive contributions |
| On that 600, reference correction with the graph +0.6554 [0.0564, 1.0618]; factorial interaction +0.6604 [0.1968, 1.1834] in the retained verified result | Positive interaction of this correction and this graph on reused development data | A universal law, identity recognition, or superiority to the strongest simple same-information guide |
| Position graph at its best tested strength +1.43; feature graph larger gain | Feature affinities may contain more useful structure than these tested position affinities | Equal mean degree does not equalize spectrum or effective smoothing; one strength cannot reject all position smoothers |
| Strength 1/4/16/64 has monotonically increasing reported mIoU, while erroneous true-target deletion increases | A measured bias/denoising tradeoff on that range | Monotone mIoU, nested masks, or increasing deletion at all strengths or on other data |
| Post-RCG reference two-delete −0.46; query grouping add/delete −0.08/−0.04; tested score adders lose to same-count RCG-score controls | These tested rules have not demonstrated useful incremental selection | Conditional sufficiency of RCG, zero residual information in every same-encoder cue, or failure of all possible constructions |

The per-arm 600 record is
[the verified historical ablation](../pipeline_verified/rcg_ablation_verified600_v1/report.json).
The broader failed-rule observations are supplied by the parent task; no new
source reconciliation or numeric replacement is undertaken in this note.
All reused data remain development evidence. An interval crossing zero is
unresolved, including the reported grouping differences.

## 1. The actual pipeline and the exact estimator

Source: [current RCG code](../../../../src/ics/methods/rcg.py), particularly
lines 51–82. Let normalized query and reference tokens be `q_i, r_j`, with
`n=4096` query nodes. For cross-image cosine `C_ij=q_i^T r_j`, let `F` contain
reference tokens with foreground coverage at least 0.9; the fallback is all
maximally covered tokens. Let `d^q_i,d^r_j` be the row/column means of the top ten
cross-image cosines. The implemented guide is

\[
 g_i^{\rm ref}=\max_{j\in F}(2C_{ij}-d^r_j)-d^q_i.
\]

In plain language, the guide favors a query patch's strong foreground-reference
match while correcting similarities that are common across many patches. This
is a constructed matching statistic, not a posterior foreground probability.

The implementation then uses

\[
 s=\operatorname{minmax}(\text{FoRIS continuous response}),\qquad
 y=s+\alpha[\operatorname{rank}(g^{\rm ref})-
                    \operatorname{rank}(s)],\quad \alpha=0.5.
\]

The guide changes relative ranking, and the original score supplies magnitude.
Rank discards guide magnitude; min–max discards the response's absolute offset
and scale. The corrected unary `y` need not lie in `[0,1]`. It is therefore
incorrect to silently identify `y`, or the solved field, with a calibrated
Bernoulli probability.

The query graph is directed adaptive 20-NN in normalized feature cosine, then
restricted to reciprocal edges with geometric-mean weights. It is divided by
its mean weighted degree. Write its symmetric nonnegative matrix as `W`, degree
matrix as `D`, and Laplacian as `L=D-W`. Define the fidelity matrix

\[
 a_i=\frac{0.1+|2s_i-1|}{\operatorname{mean}_j(0.1+|2s_j-1|)},
 \qquad H=\operatorname{diag}(a_i),\qquad a_i>0.
\]

Patches far from the score's middle are held more strongly to their corrected
unary. Confidence here means confidence of the min–max score, not proven
probability calibration. The code variable named `H` at line 71 is the *whole*
linear-system matrix `diag(a)+lambda*L`; this note reserves `H` for fidelity to
avoid confusing the two matrices.

**Exact, conditional on the constructed `H,L,y`:** for `lambda=16` in the fixed
RCG configuration, the core solves

\[
 f_\lambda=\arg\min_f E_\lambda(f),\qquad
 E_\lambda(f)=\tfrac12(f-y)^TH(f-y)+\tfrac\lambda2 f^TLf,
 \qquad (H+\lambda L)f_\lambda=Hy.
\]

The first term preserves the input evidence, weighted by anchors; the second
penalizes different outputs on feature-neighbor patches. With unordered edges,
`f^T L f = sum_{i<j} W_ij(f_i-f_j)^2`.

Since `H` is positive definite and `L` is positive semidefinite, `H+lambda*L`
is positive definite for every finite `lambda>=0`. The minimizer is unique:

\[
 f_\lambda=P_\lambda y,\qquad
 P_\lambda=(H+\lambda L)^{-1}H.
\]

There is one convex solve. Numerical conjugate-gradient iterations implement
that solve; they are not repeated inference, reference reselection, or a
nonlinear iterative semantic discovery process.

Finally the code returns FP32 `f`, bilinearly upsamples from 64×64 to 1024×1024
with `align_corners=False`, and thresholds at `>0.5`. It applies no second
min–max and no CRF. The complete FoRIS comparison includes its own finalizer.
Thus the *core* is linear in `y` for fixed `H,L`, while the *full pipeline* is
not linear in input features or scores. Rank, graph selection, fidelity,
quantization and hard thresholding cannot be ignored in a pipeline claim.

## 2. Three equivalent interpretations, with explicit assumptions

### 2.1 Penalized least squares: exact

The preceding objective is weighted least squares plus graph roughness. It
does not need a probability model. The graph expresses the hypothesis that
neighboring feature nodes deserve similar foreground responses. This hypothesis
can be right for same-object extent and wrong for semantically different but
visually similar objects.

How to assess the premise later: on sealed development outputs, diagnose GT
coverage variation on feature edges, object/background cut weight, and unary
errors within/between graph components. These are labeled diagnostics, never
inference inputs. The 600 feature/position comparison supports usefulness of
the tested affinity construction, but does not verify semantic homophily on
every edge.

### 2.2 Gaussian MAP / intrinsic GMRF: conditional model

Suppose, for an ideal continuous target `f*`,

\[
 y\mid f^*,H\sim\mathcal N(f^*,H^{-1}),\qquad
 p(f^*\mid L)\propto\exp[-\tfrac\lambda2(f^*)^TLf^*].
\]

The likelihood says reliable anchors have less unary noise. The prior says
large differences across feature edges are unlikely. The posterior is Gaussian
with mean/MAP `P_lambda*y` and precision `H+lambda*L`.

The Laplacian prior is generally **improper** on component-constant modes;
it is an intrinsic GMRF, not an ordinary full-rank Gaussian prior. Positive
anchors make the posterior proper. Also, actual `H,L,y` are all computed from
the observed features/scores. Their independence and the Gaussian noise law
are not established. This is a useful conditional estimation model, not a
claim that RCG implements the calibrated posterior for the binary GT mask.

What would support it later: mean and covariance of unary errors in the graph
basis, conditional on anchor confidence; explicit failure checks for coherent
wrong objects. Existing positive mIoU alone cannot confirm the likelihood.

### 2.3 Random walk with absorption at anchors: exact

At a node with degree `d_i>0`, the normal equation reads

\[
 f_i=k_i y_i+(1-k_i)\sum_j p_{ij}f_j,\qquad
 k_i=\frac{a_i}{a_i+\lambda d_i},\quad p_{ij}=W_{ij}/d_i.
\]

A walk starts at a query patch. It stops there with probability `k_i` and reads
that patch's unary; otherwise it moves along a query-affinity edge and repeats.
`f_i` is the expected unary at the eventual stopping node. Positive anchors
ensure eventual stopping for finite graphs and finite strength. An isolated
node has `k_i=1`, so it immediately reads its own unary.

This formalizes evidence sharing: a patch can borrow many anchors reached
through the feature graph. The process has no knowledge that an anchor is a
true target or a false object. A confident wrong anchor is still an anchor.

## 3. Exact predictions of this core

These results are exact in real arithmetic for fixed symmetric `W>=0`,
positive `H`, and finite `lambda>=0`. Returned FP32/CG fields can differ at
numerical tolerance; no numerical parity test was run for this note.

### 3.1 Component-wise weighted-mean conservation

For every connected graph component `C`, internal edge terms cancel:

\[
 \sum_{i\in C}a_i f_i=\sum_{i\in C}a_i y_i.
\]

The graph redistributes evidence within a component; it cannot change that
component's total anchor-weighted response. This is stronger than global mean
conservation. It does **not** conserve the binary foreground area after
thresholding, unweighted mean, or IoU.

**New exact prediction:** any genuine failure of this identity beyond the
solver/cast tolerance indicates a changed operator, additional normalization,
or numerical error. This is a source/solver diagnostic, not evidence of better
segmentation. Measuring it later needs no GT.

### 3.2 Maximum principle and isolated nodes

`H+lambda*L` is a nonsingular M-matrix. Its inverse is nonnegative and
`P_lambda*1=1`, so every output is a convex combination of unaries in its graph
component:

\[
 \min_{j\in C}y_j\le f_i\le\max_{j\in C}y_j\quad(i\in C),
 \qquad d_i=0\ \Longrightarrow\ f_i=y_i.
\]

RCG cannot create a response outside the component's input range. In particular,
if every unary in a graph component is below `0.5`, the core cannot raise a node
in it above `0.5`; if every unary is strictly above `0.5`, the core cannot delete
a node in it at that threshold. The final pixel mask can mix neighboring grid
values during bilinear rendering, so token statements should not be relabeled
as exact image-component statements. A bilinear pixel supported only by values
below the threshold likewise stays below it.

**New exact prediction:** a disconnected, uniformly positive *false* feature
component is preserved; a disconnected, uniformly negative *true* feature
component is not recovered. This refutes universal false-object deletion or
universal missed-object recovery by smoothing alone. Check graph components
and unary bounds first; GT is needed only to label the component's correctness.

### 3.3 Infinite-strength limit

As `lambda` tends to infinity,

\[
 f_i\longrightarrow\bar y_C^H
   =\frac{\sum_{j\in C}a_jy_j}{\sum_{j\in C}a_j}\quad(i\in C).
\]

An entire connected component ultimately shares one anchor-weighted mean. If
that mean is below the cut, sufficiently strong smoothing deletes its above-cut
patches, whether they are true or false. If it is above the cut, sufficiently
strong smoothing spreads foreground through the component. If it equals the
cut, limiting values alone do not determine finite-strength signs; strict `>`
and numerical effects also matter.

**New conditional prediction:** larger strength eventually fails on a component
whose correct foreground/background labeling is mixed but whose unaries favor
one side. The observed monotone gain at strengths 1–64 cannot be extrapolated
to the limit.

### 3.4 Why whole false components can be removed, and true ones too

For any node subset `S`, not necessarily a graph component,

\[
 \sum_{i\in S}a_i(f_i-y_i)
 =-\lambda\sum_{i\in S,j\notin S}W_{ij}(f_i-f_j).
\]

The subset's mean can be pulled down only through its boundary flux on the
feature graph. When its outside neighbors have lower solved responses, positive
outgoing flux lowers its anchor-weighted mean. Internal smoothing may also
remove local peaks without changing that mean. Hence “false-object deletion”
can result from low average anchor support, high-weight links to low-response
background, or suppression of a localized high-frequency unary bump. It is not
a semantic operation guaranteed by the objective.

A conditional approximation makes this testable. If a subset mixes rapidly
internally, `f_i≈m_S` within it, and its outside boundary response is approximately
`b`, define `rho=lambda*cut_W(S)/sum_{i in S}a_i`. Then

\[
 m_S\approx\frac{\bar y_S^H+\rho b}{1+\rho},\qquad
 m_S<0.5\ \Longleftrightarrow\
 \rho(0.5-b)>\bar y_S^H-0.5\quad(b<0.5).
\]

Low anchor support and strong exposure to low-response neighbors make deletion
more likely. “Small object” alone is not the deciding condition: a small object
can be disconnected and perfectly preserved, while a large one can have strong
cross-label edges. This approximation needs its internal-mixing and outside-
response assumptions checked; it is not a theorem for every image object.

**New prediction:** among similarly positive unaries, newly erased true and
false regions should both concentrate at large graph exposure relative to
anchor mass. If only semantic identity explains deletions after exposure,
anchor mean and initial margin are matched, this simple estimator account is
incomplete. The reported increase of true-target deletion with strength is
consistent with the account; it does not yet validate this stratified prediction.

### 3.5 Lower energy is not higher IoU: counterexample

Consider two nodes, unit edge weight, unit fidelity, and unary `(0.9,0)`. The
first node has solved response

\[
 f_1=0.9\frac{1+\lambda}{1+2\lambda},\qquad
 f_2=0.9\frac{\lambda}{1+2\lambda}.
\]

At strength at least 4, threshold `>0.5` removes node 1. Append an isolated
positive node with unary `0.9`, which always remains selected. If the true labels
are `(1,0,1)`, the initial mask is perfect and smoothing lowers IoU to `1/2`.
For true labels `(0,0,1)`, the same operator and unaries raise IoU from `1/2` to 1.
Both are admissible label configurations for this estimator. The energy uses
no query GT and therefore cannot distinguish them.

This analytical example is not a data experiment. It proves that energy
minimization alone gives no IoU guarantee. Unit fidelity is the existing
uniform-anchor special case; the same qualitative failure occurs with unequal
positive anchors whenever the component-weighted mean is below the threshold.

## 4. Spectral soft filtering and what “absorption” can mean

Let `B=H^{-1/2} L H^{-1/2}=U diag(mu_l) U^T`, with `mu_l>=0`, and use whitened
coordinates `v=H^{1/2}y`, `u=H^{1/2}f`. Then

\[
 u=U\operatorname{diag}(\beta_l)U^Tv,\qquad
 \beta_l=\frac1{1+\lambda\mu_l}.
\]

Component-constant modes have zero eigenvalue and are preserved. Rapid changes
on the feature graph have larger eigenvalues and are attenuated. Frequency here
means graph disagreement, not spatial Fourier frequency; a distant same-object
link can make spatially separated patches graph-smooth.

### 4.1 Finite strength is not information-theoretic destruction

For finite strength, every `beta_l>0` and

\[
 P_\lambda^{-1}=I+\lambda H^{-1}L,\qquad
 y=f_\lambda+\lambda H^{-1}Lf_\lambda.
\]

With exact `f,H,L`, the unary can be recovered exactly. A small coefficient can
make inversion numerically unstable, but it is not zero information. This
claim concerns the core and its retained metadata, not recoverability of full
features or both original guides from a final hard mask.

Information can be discarded elsewhere: max/pure-token selection, rank/min–max,
the reciprocal sparse graph's compression of feature geometry, FP32 storage,
and thresholding. The full features can therefore contain distinctions absent
from the rendered RCG mask even when the linear core is invertible.

### 4.2 Fixed-graph auxiliary unary and an exact no-edit condition

If an auxiliary cue supplies an additive unary perturbation `delta`, leaving
`H,L` fixed, its output perturbation is

\[
 \Delta f=P_\lambda\delta,\qquad
 (U^TH^{1/2}\Delta f)_l=\beta_l(U^TH^{1/2}\delta)_l.
\]

Graph-disagreeing perturbations become smaller, while component-constant ones
pass unchanged. This predicts weak incremental changes for a cue whose useful
part lies in strongly attenuated modes. It does not prove that the useful part
is high-frequency; that is the premise to check.

Let `T_up` be the fixed bilinear renderer. If, at every output pixel `p`,

\[
 |(T_{\rm up}P_\lambda\delta)_p|
 < |(T_{\rm up}f)_p-0.5|,
\]

no pixel changes threshold side. This is an exact sufficient condition for zero
mask gain or loss. More generally, changed pixels must lie where the absolute
initial threshold margin is no greater than the absolute perturbation.
Since both `P_lambda` and bilinear interpolation are positive averaging
operators, `||T_up P_lambda delta||_infinity <= ||delta||_infinity`.

**New prediction:** even an informative cue may have zero measured complete-mask
effect if its surviving perturbation does not reach the low-margin pixels.
Conversely, a small perturbation concentrated there can change many pixels.
Measure surviving perturbation and margins before attributing a null result to
exhausted semantic information. This concerns an explicitly additive fixed-
operator perturbation, not every way of constructing an auxiliary rule.

Repeated application of the same graph filter is not exact absorption:
`P_lambda^2 != P_lambda` in general. It further multiplies each nonconstant mode
by `beta_l`; it may help or harm. Finite RCG is not a projection onto a semantic
subspace.

### 4.3 The graph itself carries evidence

If features change the graph or fidelity as well as the unary, at fixed strength
the exact finite-change identity is

\[
 f'-f=(H'+\lambda L')^{-1}
 [H'(y'-y)+(H'-H)(y-f)-\lambda(L'-L)f].
\]

A cue can affect what should be shared, not only what initial score to inject.
The last two terms are absent from the fixed-unary argument. Changing affinity
geometry can preserve boundaries or reinforce errors. “It uses the same
encoder” is insufficient to assert identical `H,L,y` or identical decision
information. This identity also covers discrete neighbor/reciprocity changes.
Within a differentiable neighborhood/rank regime its first-order form is
`(H+lambda L)df=Hdy+dH(y-f)-lambda dL f`.

The fixed 600 interaction is consistent with reference correction being noisy
alone but useful when graph agreement combines it across patches. It is not an
identification proof of that mechanism. The tested mean guide is a necessary
strong simple comparator; changes in guide strength are not pure matching-rule
ablations.

For the actual correction `v=rank(g_ref)-rank(s)`, fixing the input score and
features fixes `H,L,v`. Its continuous conditional effects are exactly

\[
 f_{\alpha,\lambda}-f_{0,\lambda}=\alpha P_\lambda v,\qquad
 f_{\alpha,\lambda}-f_{0,\lambda}-f_{\alpha,0}+f_{0,0}
 =\alpha(P_\lambda-I)v.
\]

The graph transforms where the reference correction reaches the field; it does
not create a second independent guide contribution. A positive interaction in
thresholded IoU can arise from this changed displacement and the nonlinear
mask/IoU decision. The sign of that metric interaction is not fixed by either
identity. Its measured positive sign on the given 600 remains empirical.

### 4.4 Strength depends on signal, noise and connectivity: conditional result

Assume the whitened unary is `v=theta+epsilon`, where the noise has zero mean,
`theta` is a fixed ideal continuous signal, and `sigma_l^2` is the noise variance
in graph mode `l`; write `theta_l=(U^T theta)_l` and
`sigma_l^2=Var((U^T epsilon)_l)`. The expected squared error in whitened
coordinates is

\[
 R(\lambda)=\sum_l[(1-\beta_l)^2\theta_l^2+
                              \beta_l^2\sigma_l^2].
\]

Smoothing removes variance and introduces bias. An individually optimal mode
shrinkage is `beta_l*=theta_l^2/(theta_l^2+sigma_l^2)` when the denominator is
nonzero, equivalent, for nonzero
signal and eigenvalue, to `lambda_l*=sigma_l^2/(mu_l*theta_l^2)`. One common
strength cannot generally attain all these mode optima.

**New conditional predictions:**

- High-frequency noise with low-frequency signal permits stronger smoothing.
- Low-frequency correlated errors, such as a coherent wrong component, survive
  because those modes are weakly penalized or unpenalized.
- Cross-label edges raise the true signal's graph roughness and can increase
  bias; graph fragmentation preserves more component-constant error modes.
- Graphs with the same mean degree can require different strengths because
  their spectra and signal alignment differ.

These claims concern squared continuous risk under the stated noise model.
They do not predict monotone IoU. Signal/noise mode energies require labeled
diagnostics; approximate zero-mean errors, anchor calibration and graph homophily
must be checked before using this model to explain an empirical strength curve.

## 5. The strict meaning of evidence that is not absorbed

Let `T=(g,Q)` denote the retained RCG field `g` and explicitly stated permitted
context `Q` available to a selector. `Q` must be defined: it may include graph,
anchor, reference and location summaries, but cannot silently mean all raw
features in one claim and a few summaries in another. Let `Z` be an auxiliary
cue and `Y_i` the query foreground indicator.

### 5.1 Marginal posterior information: exact proper-loss result

Define

\[
 \eta_i=\Pr(Y_i=1\mid T),\qquad
 \eta_i'=\Pr(Y_i=1\mid T,Z).
\]

For the optimal Brier predictors,

\[
 R_{\rm Brier}^*(T)-R_{\rm Brier}^*(T,Z)
 =\mathbb E[(\eta_i'-\eta_i)^2]\ge0.
\]

An auxiliary cue is marginally informative exactly when it changes the
conditional posterior on a nonzero-probability set. A conditional-independence
statement `Y_i independent of Z given T` makes this gain zero. The actual RCG
field is not known to equal `eta_i`, so Brier sufficiency cannot be inferred
from its convex objective or thresholded results.

Positive Brier gain need not improve IoU, pixel accuracy or a fixed edit budget.
It may improve calibration while leaving every relevant order or threshold
unchanged. No conditional posterior was estimated in this theory-only task.

### 5.2 Decision sufficiency for class-summed IoU: exact formulation

For the benchmark, write

\[
 J_c(m,Y)=\frac{\sum_{e:\,c(e)=c} I_e(m,Y)}
                   {\sum_{e:\,c(e)=c} U_e(m,Y)},\qquad
 J=\frac1C\sum_c J_c.
\]

Counts are summed within class first. This is not mean episode IoU, a global
pooled pixel ratio, or pixelwise 0/1 loss. Let the allowed action set `A` contain
complete masks or edits with a specified budget/cost. The value of additional
information for this utility is

\[
 \Delta V=\mathbb E\!\left[\max_{a\in A}
                  \mathbb E[J(a,Y)\mid T,Z]\right]
 -\mathbb E\!\left[\max_{a\in A}
                  \mathbb E[J(a,Y)\mid T]\right]\ge0.
\]

An optimal selector can always ignore the new cue. The inequality therefore
applies to optimal selectors, not arbitrary implemented rules; a badly used
informative cue can lower actual mIoU. Strict gain requires that no `T`-only
action is optimal for almost all the relevant `Z` outcomes on a set of `T`
states with positive probability. For the finite family of masks this is also
sufficient: otherwise one common conditional maximizer could be selected
without `Z`. Merely changing the posterior is insufficient.

A sufficient no-gain condition is full-mask conditional sufficiency,
`Y_vector independent of Z given T`, together with decision optimization over
the same action set. A weaker utility-specific sufficient condition is that
`E[J(a,Y)|T,Z]` is a function of `T` for every allowed action `a`.
Marginal pixel sufficiency alone does not establish this for correlated masks.
Even full information sufficiency does not prove an implemented RCG threshold
has achieved the best decision obtainable from `T`.

### 5.3 Same-budget top-k: exactly where posterior ranking is optimal

Fix an eligible domain and exactly `k` additions. For expected true additions,
the optimal action selects the top `k` values of `eta_i`. Deletion selects the
bottom `k` values among current foreground pixels. This needs no conditional
independence because expected true counts are linear. A swap of selected `i`
for unselected `j` improves that expected count exactly when `eta_j>eta_i`.

For an observed class with fixed true count `T_c` and fixed predicted count
`K_c`, the realized class IoU is

\[
 J_c=\frac{I_c}{T_c+K_c-I_c}.
\]

At fixed counts it increases strictly with the *realized* true intersection.
Therefore a matched-budget edit that increases each class's realized
intersection cannot reduce class-macro IoU. This is an accounting fact, not
proof that maximizing expected intersection maximizes expected finite-sample
IoU.

Two conditions do permit a posterior-ranking conclusion for IoU:

1. **Population ratio of expected class counts**, with fixed predicted count:
   `E[I_c]/(E[T_c]+K_c-E[I_c])` is strictly increasing in expected intersection.
   This is the population class-summed functional, not automatically the
   expectation of the finite benchmark ratio.
2. **Conditional independence of pixel labels**, with a fixed count inside
   each class: the exact expected finite class IoU is optimized by top-k
   posterior ranking. Independence is usually a strong, unverified assumption
   for coherent image objects.

The second statement follows by an exact swap. Let a class's prediction include
`i`, exclude `j`, and have total count `k>=1`. Let `a` be the other selected true
pixels and `b` the other unselected true pixels, with `d=k+b`. Always,

\[
 J(\text{swap})-J(\text{original})
 =\frac{a+d+1}{d(d+1)}(Y_j-Y_i).
\]

For correlated labels, the **exact** positive-gain condition is
`E[(a+d+1)/(d(d+1))*(Y_j-Y_i)|T]>0`. Under conditional independence it factors
into a positive coefficient times `eta_j-eta_i`; then top-k is optimal by
successive swaps. Without independence, the coefficient can correlate with
the label difference, so ordinary marginal posterior order can be insufficient.

Counterexample: with probability 0.6 let labels be `(1,0,1)`; with probability
0.4 let them be `(0,1,0)`. Predict one pixel. Pixel 1 has posterior 0.6, above
pixel 2's 0.4, but its expected IoU is `0.6/2=0.3`, below pixel 2's `0.4/1=0.4`.
This is a symbolic distribution, not an experiment. It refutes unconditional
identification of posterior-ranking optimality with expected IoU optimality.

Consequently, beating a same-count RCG-score control demonstrates improved
selection in the tested setting. Losing to it limits that selection rule. A
theorem that no cue can beat it additionally needs posterior order sufficiency
and the stated metric conditions. “RCG score already uses the cue” is not such
a theorem. Match eligible domain, pixel/token unit, per-class/episode counts,
add/delete direction and information/cost before making that comparison.

### 5.4 Same encoder does not imply the same retained information

If full frozen features and permitted reference inputs are `F`, and both
`T=t(F)` and `Z=z(F)` are deterministic, then

\[
 I(Y;Z\mid F)=0\quad\text{but possibly}\quad I(Y;Z\mid T)>0.
\]

An auxiliary statistic can preserve a distinction compressed by the RCG field.
If `Q` already includes all of `F`, it adds no information in the strict
information-theoretic sense; it can still improve an imperfect computation or
decision rule using that information. Distinguish representation sufficiency,
field sufficiency, and decision quality.

A fine local readout can change where coarse evidence reaches pixels, local
affinity weighting, effective resolution and pixel order. Additional shifted
views from the same encoder may also contain sampling detail absent from the
original tokens. It is not generally an additive unary on the same fixed graph,
and it is not covered by the “high-frequency auxiliary is shrunk” argument.
Even a readout using only retained inputs can improve the nonoptimal hard
renderer without new information. This is an explanation of possible value,
not a proposed new readout or an unmeasured gain claim.

## 6. Minimal future checks and data count, not performed

The strongest claim justified now is a conditional mechanism account with
falsifiable predictions. No additional method or run is needed to complete
this theory task. If the user later authorizes a check, use one fixed mechanism
question and the smallest matched complete comparison that can decide it:

| Fixed question | Assumptions / quantities to check | Existing support | New falsifiable prediction | Minimum future design |
|---|---|---|---|---|
| Is the core really the anchored estimator? | Same `H,L,y`; CG/cast residual; component labels | Source equations and retained historical verification | Component-weighted mass, range, isolated-node identities hold to declared numerical tolerance | One source/solver audit on already sealed fields; no parameter search or efficacy conclusion |
| Does graph exposure explain erasure? | Anchor mean, cut weight/anchor mass, initial margins; internal mixing for the approximation | True deletion rises with reported strength | At matched unary support, high exposure to low-response neighbors predicts erasure of true and false regions | Fixed labeled diagnostic on already sealed strengths; complete RCG and no-rank graph control retained; no new inference rule |
| Is strength helping through high-frequency denoising? | Graph signal/noise spectra; mean-zero model limitations | Graph-only gain and strength tradeoff | Improvements concentrate where error is less graph-smooth than signal; coherent component-constant errors persist | Diagnose the existing fixed strengths; freeze frequency bands before held-data readout; no new sweep |
| Is a cue useful after RCG? | Explicit `T`, eligible domain, budget, class counts and attainable conditional selection | Tested post-RCG constructions fail or remain unresolved versus same-count controls | A useful cue must change decision-relevant ordering/budget/geometry, not only correlate with foreground globally | Exactly one already specified cue versus complete RCG and same-domain same-budget RCG-score control; seal before scoring; no cue-family search |
| Is fine readout's value outside coarse smoothing? | Input/view identity, local weights, resolution and renderer, control strength | Possibility follows from operator distinction; no new result asserted | A coarse-field-preserving readout may improve low-margin extent even without changing coarse graph solution | Preserve the supplied fixed readout and strongest simple same-information renderer; complete masks and per-class I/U, not a boundary proxy alone |

Do not choose 20, 50, 600 or 1200 merely because those counts are convenient.
For a complete paired mIoU contrast, let `G` be independent connected-photo
groups and `psi_g` its class-summed-ratio influence contribution in percentage
points. Its variance, `sigma_psi^2`, must come from earlier matched paired I/U
and group structure, not independent pixels or mean episode IoUs. A standard
first-order design for minimum meaningful effect `delta_min`, two-sided level
0.05 and desired power `1-beta` is

\[
 G_{\rm req}\gtrsim
 \frac{(z_{0.975}+z_{1-\beta})^2\sigma_\psi^2}{\delta_{\min}^2}.
\]

More independent photograph groups are needed when paired variability is high
or the desired effect is small. For arm `a`, the ratio influence uses
`(I_{g,c}^a-J_c^a U_{g,c}^a)/mean_g(U_{g,c}^a)` averaged across classes; subtract
the two arms' influence contributions before estimating variance. Missing
classes, uneven class coverage, photograph overlap and changing subgroup
eligibility require additional design constraints. Translate required groups
to episodes using the actual sampling/overlap contract.

For a conditioned region-ranking diagnostic, power is governed by eligible
independent groups and its paired region-selection effect, not all cohort
episodes. A posterior-information test and a complete-mask effect need their
own variance/effect targets. They cannot share a sample-size number by default.
Reported asymmetric bootstrap intervals alone do not supply an exact variance
for a new contrast.

No `sigma_psi` was recomputed here, and no new `delta_min` is selected for the
user. Therefore a numerical future episode count would be invented. A later
authorized design should specify the effect target, estimate its variance from
existing matched records, preserve all classes/folds required by the protocol,
and freeze the rule before fresh confirmation. Base-fold GT calibration remains
labeled/supervised evidence and cannot silently become a training-free
single-reference inference mechanism.

## 7. Independent audit: marginal component value and evidence absorption

Second bounded assignment, 2026-10-06 UTC. The current
[main theory](main_theory.md) was read, especially sections 3–5, 8–9 and 11.
`quality_theory.md` was not present at the single inventory check; its contents
are not audited or inferred here. No repeated wait/read loop or experiment was
performed. This supplement provides decision criteria, not a new inference
method.

### 7.1 The unifying object is conditional utility, not zero marginal information

Let `a` be the complete current recipe, including its other selected components,
and let `a+i` be the feasible result of including component `i`. For retained
state `T` and auxiliary cue `Z`, define

\[
 \mu_i(a\mid T,Z)
 =\mathbb E[J(a+i,Y)-J(a,Y)\mid T,Z].
\]

This is the component's expected change in the actual class-summed objective
after its interactions with the current recipe. It is not the component's
standalone gain at the origin, raw match confidence, mutual information, or
mean foreground purity. For canonical additions it must use only pixels not
already added; for deletions only pixels not already deleted. The other
components also change the relevant IoU denominators and ratios.

The main note's exact deterministic cross-marginal tests support precisely this
state dependence. A negative standalone component may have positive marginal
value in a composition. A positive standalone component may have negative
marginal value after the other components. Neither implies graph posterior
calibration, and neither is contradicted by linearity of the graph core.

For a fixed recipe selected before new queries, if removal of `i` is feasible
and its population expected marginal value at that recipe is strictly negative,
removing it strictly improves expected utility. A true expected-utility optimum
in a family closed under component removal cannot contain such a component.
If a purported optimum does, inspect the objective, estimation error, removal
feasibility or the definition of the marginal. A selected component may still
be harmful on some contexts while beneficial on average. An empirical negative
point estimate on held data does not alone establish a negative population
expectation; uncertainty remains necessary.

This gives a necessary condition at an optimum, not a greedy certificate:
nonpositive single-component additions do not rule out a positive joint move.
The signed overlap and class-macro counterexamples in the main note already
show why that stronger inference would be invalid.

### 7.2 Expected numerator is not expected finite IoU gain

For one realized class and its effective edits, the main note correctly gives

\[
 \Delta_i=\frac{t-d+J_a(e-f)}{U_{a+i}}.
\]

Conditioning on retained inputs leaves those GT counts random in a statistical
model. Therefore the required marginal is
`E[(t-d+J_a(e-f))/U_(a+i)|T,Z]`. Its sign cannot in general be replaced by the
sign of the expected numerator or a ratio of its expected counts. The random
denominator can correlate with the numerator.

A feasible symbolic example: origin `O={p0}` and one fixed addition `A={p1}`.
With probability `1/1000`, the truth is `{p1}`. The addition's numerator is 1
and its IoU gain is `1/2`. Otherwise truth is `{p0}` plus 99 other pixels,
excluding `p1`. Its numerator is `-1/100` and its gain is `-1/10100`. Thus

\[
 \mathbb E[\text{numerator}]=-0.00899<0,\qquad
 \mathbb E[\Delta_i]=\frac{4051}{10100000}>0.
\]

The expected numerator is negative while the expected realized IoU gain is
positive. This is one class, hence also a one-class macro example. It is a
distributional counterexample, not evaluated data. Taking expectations of the
main note's realized identities needs this additional distinction. A population
ratio of expected counts has its own legitimate numerator/ratio algebra; it
must be named as that objective.

### 7.3 Information gain and an imperfect current decision add two gaps

Fix the same finite feasible action family and let
`V_T=E[max_a E[J(a,Y)|T]]` and
`V_TZ=E[max_a E[J(a,Y)|T,Z]]`. For a current `T`-measurable RCG decision `a_g`,

\[
 V_{TZ}-\mathbb E[J(a_g,Y)]
 =\underbrace{V_{TZ}-V_T}_{\text{new information value}}
  +\underbrace{V_T-\mathbb E[J(a_g,Y)]}_{\text{current decision gap}}.
\]

The largest attainable gain can come from additional decision-relevant
information, better use of information already retained, or both. This identity
does not call the graph field a posterior. Zero additional information value
does not eliminate the current decision gap. Conversely, an informative cue
can have zero utility value if the same action is optimal for every cue outcome.

This identity treats the cue as available and holds the action family fixed.
New acquisition cost, extra views, changed resources or new feasible actions
must be handled explicitly; information value alone is not net value at a
compute budget. Actual imperfect auxiliary rules need not achieve either
nonnegative gap and can reduce measured IoU.

Consequently, selected-component negativity and “graph absorption” can be
unified as claims about conditional *action utility*, but not generally as
claims about conditional *marginal information*. Conditional-information and
sufficient-ranking formulations require the extra conditions below.

### 7.4 Strict ineffective classes

These criteria rule out a gain within their stated scope; they do not assume
calibration of the graph field.

1. **Recipe/output redundancy, independent of GT.** If a component's effective
   add/delete sets are empty, or it changes no final mask on any allowed input,
   it cannot change any mask metric. The fixed-operator margin certificate in
   section 4.2 is one such condition. If removing it does not increase shared
   inference cost, the redundant component is weakly dominated.
2. **Exact same-budget selection invariance, independent of GT.** An auxiliary
   score that is a strictly increasing transform of `g`, with the same eligible
   domain, count, unit and tie handling, selects exactly the same top-k pixels.
   It cannot improve that matched-budget mask. It can change a fixed-threshold
   area decision; that is a different action/budget question. Approximate
   correlation with `g` is not this exact invariance condition.
3. **Conditional utility sufficiency plus optimal current action.** If
   `E[J(a,Y)|T,Z]=E[J(a,Y)|T]` for every feasible complete action, and `a_g` is
   already a conditional maximizer, no cue-dependent policy using `Z` can gain.
   Full-mask conditional independence is a sufficient premise. Pixel marginal
   independence alone is insufficient for correlated finite IoU. Current
   conditional optimality is essential; RCG's convex energy does not prove it.
4. **Sufficient ranking at a fixed cut, under the right objective.** For a
   fixed-budget addition set `S_g`, if, for almost every cue outcome, every
   selected pixel's posterior is at least every unselected pixel's posterior,
   `S_g` remains optimal for expected true count. Numerical equality `g=eta`
   is unnecessary: correct ordering across the budget cut is enough. The
   converse applies to deletion. The same statement for finite expected IoU
   additionally needs conditional independence both before and after the cue,
   or another proven joint-law condition. For the population ratio of expected
   counts, fixed per-class counts suffice. No such ordering premise is proved
   for actual RCG by the current auxiliary failures.
5. **Actual marginal dominance.** At a particular feasible recipe, if
   `mu_i(a|T,Z)<=0` for every permitted context, adding `i` cannot improve its
   expected score. If exclusion preserves feasibility and does not increase
   dependency cost, it is dominated in that comparison. To exclude `i` from
   an entire family, this sign must hold at *every* feasible context in which
   it could be added. A sign at the origin or one composition is insufficient.

The first two are the strongest checks available without any semantic
probability premise: the masks are unchanged, regardless of unknown labels.
The other criteria need conditional utility/ordering assumptions and justified
uncertainty. None follows from similarity being used somewhere in RCG.

### 7.5 Classes with possible gain, and what theory can predict first

For new information, strict positive `V_TZ-V_T` is possible exactly when, on a
nonzero-probability set of retained states, cue outcomes require different
optimal feasible actions with no common conditional maximizer. This is the
loss-specific criterion, not simply `I(Y;Z|T)>0`. For expected true count at a
fixed budget, posterior order changes **across the selected/unselected cut**
with a strict utility advantage are sufficient. Changes only within either
side are irrelevant at that budget.

For correlated finite class IoU, use the conditional utility of the attainable
alternative, including the joint label law and denominator. A positive
single-swap weighted condition from section 5.3 is sufficient for that swap's
gain; absence of positive single swaps does not certify absence of a joint
gain. Marginal calibration changes can also matter when the optimal allowed
count changes, even though the ordering is preserved. Thus fixed ranking
sufficiency does not imply budget/complete-mask sufficiency.

For better use of retained information, positive `V_T-E[J(a_g,Y)]` leaves room
for a gain without new conditional information. It can arise through current
renderer or feasible-composition decisions, consistent with the already
specified fine-readout and composition families. This criterion asserts an
opportunity class; it proposes no new rule and no numerical gain.

The useful theory-first prediction order is therefore:

- Rule out exact output/order redundancies from the specified operators first.
- For any remaining component, specify the feasible alternative and *effective*
  residual edit, then which conditional utility premise gives its sign or
  changes selection across the relevant cut.
- Treat graph high-frequency attenuation, anchor/cut exposure and preserved
  component modes as predictions about response geometry. Combine them with a
  semantic conditional-utility premise before predicting an IoU sign.
- Where the allowed evidence still permits both positive and negative
  conditional gains, declare that opportunity unresolved. An algebraic
  possibility alone cannot rank which experiment will succeed. Existing
  evidence may narrow the premise; it does not supply a calibrated graph
  posterior by implication.

If an admissible family of conditional label laws is stated, a strict exclusion
certificate is an upper bound on the component's conditional gain of at most
zero for every law in that family. A positive lower bound supplies a conditional
gain guarantee; an interval crossing zero identifies the unresolved premise.
Without stated restrictions on that family, such bounds will often be vacuous.
This is an identifiability boundary for theory's prediction, not an instruction
to demand certainty before a future authorized experiment.

The main note already separates fixed recipes from adaptive ones and observed
counts from justified membership probabilities. Preserve those distinctions.
The common contribution of the two accounts is **conditional value of attainable
decisions under explicit retained information and cost**, rather than universal
semantic sufficiency of graph smoothing.

## Decision for this theory hour

Retain **anchored graph denoising plus conditional evidence sharing** as the
mechanism supported by the estimator and the given ablations. Reject
**universal evidence exhaustion** as an unsupported claim. The concrete
boundary is conditional information and conditional decision value relative
to an explicitly defined retained state, action budget and class-summed IoU
utility. Existing auxiliary failures narrow their tested constructions; they
do not close that boundary for the whole frozen representation.

The exact estimator identities, symbolic counterexamples and conditional
ranking results above were checked by derivation and source reading only.
Empirical spectral premises, conditional posteriors, sufficiency, predicted
strata and numerical future sample sizes remain unverified. No experiment was
run or automatically scheduled.
