# Finite-sample selection, purity margins, and component count

Owner: rcg_quality_analysis. Pure theoretical derivations for the theory hour.
No experiment, dataset evaluation, GPU action, remote computation, or parameter scan
was performed for this document. Supplied observations are distinguished from assumptions.

## Core conclusions

1. Selection optimism, an interval crossing zero, and a concentration bound do not
   prove negative expected true gain. Negative gain requires a statement about the
   selected components' true benefits, harms, and inclusion probabilities.
2. In a fixed-context helpful/harmful mixture, expected accepted gain is exactly
   \(r a\,\mathrm{TPR}-(1-r)b\,\mathrm{FPR}\). Here r is the helpful-component
   probability; pi is reserved for foreground occupancy. This expression gives a
   necessary and sufficient negative-gain condition.
3. A relation between optimal component count and sample size exists only after a
   model for margins, errors, interactions and the selection policy is specified.
   One explicit Gaussian gated-component model gives a cutoff proportional to
   \(n_{\mathrm{eff}}^{1/(2\nu)}\). It is not a universal theorem about actual
   class-macro IoU or a forced top-k selector.
4. Diminishing but strictly positive true marginal returns alone do not imply a
   finite accuracy-optimal k. Expanding an optional family cannot lower its true
   optimum; enabling every component or using an estimated selector can lower
   realised performance.
5. An unbiased, representative estimate and a transportable variance model are
   substantive assumptions. More reference pixels, more scoring episodes, and more
   independent photographs are not interchangeable sample sizes.

## 1. Purity is a conditional margin, not a universal quality score

For one class, fix a baseline with intersection I, union U>0 and J=I/U.
Suppose an effective addition contains m pixels, of which fraction rho are true.
Then

\[
\Delta_{\rm add}
=\frac{m[(1+J)\rho-J]}{U+m(1-\rho)}.
\]

The break-even purity is theta_add=J/(1+J). With
delta_p=rho-theta_add,

\[
\Delta_{\rm add}
=\frac{m(1+J)\delta_p}{D-m\delta_p},
\qquad D=U+\frac{m}{1+J}.
\tag{1}
\]

For a deletion, define rho as the fraction of deleted pixels that are false.
Then

\[
\Delta_{\rm del}
=\frac{m[(1+J)\rho-1]}{U-m\rho},\qquad
\theta_{\rm del}=\frac1{1+J}.
\tag{2}
\]

Both signs follow exactly from counts. Near the break-even line, the addition
gain is \(h\delta_p+O(\delta_p^2)\), with
\(h=m(1+J)/D\). Deletion has the analogous slope with
\(D=U-m/(1+J)\), provided the resulting union is positive.

Consequently, a purity margin of .01 is not a .01 IoU effect. Its effect also
depends on edit volume, class union, baseline quality and current selection context.
With several classes the exact benchmark change is

\[
\Delta {\rm mIoU}
=\frac1K\sum_c
\frac{a_{TP,c}-d_{TP,c}
J_{0,c}(d_{FP,c}-a_{FP,c})}{U_{1,c}}.
\tag{3}
\]

Pooling purity first discards the class-specific weights and can change the sign.
All sets in (1)--(3) are effective edits after overlap removal, not sums of standalone
proposal counts.

## 2. What does negative expected selected gain require?

### Proposition 1: exact inclusion condition at a fixed context

Let a proposed component have true *benchmark marginal gain* +a with probability r,
and -b with probability 1-r, where a,b>0. The previous mask/context is fixed.
Let an estimated-gain gate accept the component with probability TPR in the
helpful case and FPR in the harmful case. Assume the gate's evaluation noise does
not change the conditional future true gains a and -b.

The unconditional gain of the accepted-or-abstained component is

\[
G=r a\,\mathrm{TPR}-(1-r)b\,\mathrm{FPR}.
\tag{4}
\]

If acceptance has positive probability, its conditional true gain has the same
sign. Thus

\[
G<0\iff r a\,\mathrm{TPR}<(1-r)b\,\mathrm{FPR}.
\tag{5}
\]

**Proof.** Partition on helpful/harmful type and acceptance, and apply the law
of total expectation. Rejection contributes zero. Conditioning on acceptance
only divides (4) by its positive probability.

Equivalently, if r_sel is the posterior helpful probability *after* selection,

\[
E[\Delta\mid{\rm accepted}]
=r_{\rm sel}a-(1-r_{\rm sel})b<0
\iff r_{\rm sel}<\frac b{a+b}.
\tag{6}
\]

When remaining benefits a become small but erroneous deletion/addition losses b
remain substantial, the required precision approaches one. This is a possible
mechanism for harmful tail components; it is not evidence that every tail is harmful.

### Three distinct statements

- **Selection bias:** the selected empirical estimate is systematically optimistic,
  \(E[\widehat\Delta_{\rm sel}-\Delta_{\rm sel}]>0\).
- **Selection uncertainty:** the available evidence cannot confidently determine
  which component or which sign is correct.
- **Negative expected true gain:** the deployment expectation in (4), or another
  explicitly defined population expectation, is below zero.

Neither of the first two implies the third.

**Minimal counterexample.** Two methods each have the same deterministic true gain
+epsilon. Select the larger of two independent noisy unbiased estimates. The maximum
estimate has positive optimism, but whichever method wins still has true gain
+epsilon. If both true gains are zero, maximisation can produce a positive estimate
while true selected gain remains zero, rather than negative.

The independence qualification matters: if selection data and future outcomes
share an unmodelled condition, the selected conditional true gain need not equal
the method's unconditional population gain.

### Proposition 2: a concrete finite-sample mixture

Assume \(\widehat\Delta=\Delta+\epsilon\), with
\(\epsilon\sim N(0,\sigma^2/n_{\rm eff})\), and accept iff
\(\widehat\Delta>0\). The noise is unbiased, its distribution is the same in both
types, and it is independent of future outcomes conditional on type.
Then

\[
G(n_{\rm eff})
=r a\,\Phi(a\sqrt{n_{\rm eff}}/\sigma)
-(1-r)b\,\Phi(-b\sqrt{n_{\rm eff}}/\sigma).
\tag{7}
\]

This is exact under the stated Gaussian model. At high noise it approaches
\([r a-(1-r)b]/2\); at vanishing noise it approaches r a.
Finite-sample negative gain is possible when helpful components are uncommon
or losses outweigh benefits. It is not inevitable at a small n.

For label-free reference-derived proxies, unbiasedness and transfer to query edits
are not established by having a fully labelled reference. If
\(\widehat\Delta=\Delta+d+\epsilon\), a persistent transport bias d can defeat
arbitrarily large n. A one-component counterexample is
\(\Delta=-\epsilon_0,\ d=2\epsilon_0,\ \epsilon=0\): the gate always selects a
harmful component at every sample size.

## 3. Conditional component-count results

### General marginal decomposition

For a specified prefix policy, let q_j be the probability that its j-th accepted
component is harmful, and let a_j and -b_j be its conditional mean true incremental
benchmark gains. At that policy's current context,

\[
g_j=(1-q_j)a_j-q_j b_j.
\tag{8}
\]

The exact expectation of a policy's sequential gain telescopes as the sum of
its expected context-dependent increments. However, changing the prefix or its
gates may change all subsequent q_j,a_j,b_j. Treating these as fixed across policies
is an additional model assumption.

If the marginal law is invariant to the other gates, and g_j has one sign crossing
from positive to negative, the accuracy-optimal prefix ends at the last positive
g_j. This follows by comparing partial sums. Without these assumptions it is not
a theorem about global mask-composition or class-macro IoU.

An explicit conditional upper bound follows if, on a tail,
\(q_j\ge q_0>0,\ b_j\ge b_0>0,\ a_j\le A j^{-\nu}\):

\[
j>
\left(\frac{A(1-q_0)}{b_0q_0}\right)^{1/\nu}
\quad\Longrightarrow\quad g_j<0.
\tag{9}
\]

Using this as an optimal-prefix bound also requires the prefix sign-crossing
assumption. A Hoeffding *upper* bound on error probability cannot supply the
required *lower* bound q_0.

### Proposition 3: an exact gated-prefix toy law

Consider k enabled optional producers. Producer j has a fixed-context true gain
\(+a_j\) with probability r and \(-a_j\) otherwise. It is applied only when its
Gaussian noisy gain estimate is positive. Assume:

1. \(0<r<1/2\), the same for each producer;
2. estimation noise has standard deviation sigma/sqrt(n_eff);
3. the true marginal gain laws do not change when other gates accept or reject;
4. benchmark gains are additive across these producers, or an explicit independent
   utility model is being optimised;
5. \(a_j=A j^{-\nu}\), with A>0 and nu>0, over a finite library of M producers.

The third and fourth assumptions are substantial. They can be exact in a
separable benchmark construction; a small-edit approximation for overlapping
actual IoU must carry its approximation error.

Under these assumptions,

\[
g_j=a_j\left[\Phi(a_j\sqrt{n_{\rm eff}}/\sigma)-(1-r)\right].
\tag{10}
\]

Therefore

\[
g_j>0\iff
a_j>\frac{\sigma z_{1-r}}{\sqrt{n_{\rm eff}}}.
\tag{11}
\]

The exact accuracy-optimal enabled prefix is

\[
k^*=\max\{k\le M:A k^{-\nu}>
\sigma z_{1-r}/\sqrt{n_{\rm eff}}\},
\tag{12}
\]

with k*=0 if the set is empty; equality ties go to the shorter prefix.
It scales as \(n_{\rm eff}^{1/(2\nu)}\) before saturation at M.

**Proof.** In the helpful state, acceptance probability is Phi(a_j/s);
in the harmful state it is Phi(-a_j/s), where s=sigma/sqrt(n_eff).
Equation (4) reduces to (10). Phi is strictly increasing and a_j decreases,
so the signs have one positive-to-negative crossing. Additivity makes expected
prefix gain a partial sum, maximised at that crossing.

Here k counts **enabled producers with a specified acceptance gate**. The actual
number of applied edits is random. This result is not a forced active-top-k theorem.
It does not justify a learned posterior or querying new GT in a zero-query-label
inference method.

For the purity parameter in (1), if \(a_j=h\delta_{p,j}+O(\delta_{p,j}^2)\)
and \(\delta_{p,j}\) decays as \(d j^{-\nu}\), the same scaling is a *local
approximation*. It additionally assumes comparable m/U and J and a specified
noise scale in compatible units. A purity margin alone cannot determine k.

**Counterexamples to a universal sparse optimum.**

- If every effective true increment is positive, all M can be accuracy-optimal,
  however small the increments become.
- In the symmetric Gaussian model with r>=1/2, (10) is strictly positive for
  every a_j>0 and n_eff>0; no finite accuracy cutoff follows.
- With a persistent biased proxy, increasing n can retain a harmful component
  with probability tending to one.
- With proposal overlap, a negative singleton can become useful jointly; the
  fixed marginal law assumed in Proposition 3 then fails.

## 4. Sample size: detection is not negativity

n_eff must describe independent informative units for the relevant estimate.
For benchmark differences, an independent-photograph-group model is appropriate
only if its independence/transport assumptions hold. Pixel count is not this n.
Evaluation sample size narrows uncertainty about a frozen method; it does not
causally change that method's fixed population performance.

For independent groups and a locally regular class-summed metric, the paired
influence contribution is

\[
\psi_g=\frac1K\sum_c\left[
\frac{I_{1,g,c}-J_{1,c}U_{1,g,c}}{\mu U_{1,c}}
-\frac{I_{0,g,c}-J_{0,c}U_{0,g,c}}{\mu U_{0,c}}
\right].
\tag{13}
\]

Here mu U is the population mean group union, and
sigma^2=Var(psi_g), in the same units as the benchmark effect. This accounts for
pairing, group sizes and class-denominator weights; it is not the variance of
unweighted episode IoUs.

With a normal approximation, fixed positive target effect delta_gain, known or
defensibly piloted sigma, two-sided type-I error alpha and target power
1-beta_power, conventional planning gives

\[
n_{\rm eff}\ \gtrsim\
\frac{\sigma^2
[z_{1-\alpha/2}+z_{1-\beta_{\rm power}}]^2}
{\delta_{\rm gain}^2}.
\tag{14}
\]

For an exactly normal estimator with known sigma this is a conservative sufficient
two-sided planning value: the upper rejection tail has power 1-beta_power and the
opposite tail adds a nonnegative amount. It is not the exact minimum two-sided n.
Unknown variance, group dependence, rare-class denominators, multiplicity and
distribution shift require further qualifications. Without sigma and a target
effect in benchmark units, (14) cannot yield a numerical n.

Testing gain above a target gamma_0 uses excess effect delta_gain-gamma_0, not the
full gain. Detecting a purity margin uses its own variance or the slope in (1).

For illustrative independent, bounded representative purity observations,
Hoeffding plus a union bound gives

\[
P\{\max_{j\le M}|\widehat\rho_j-\rho_j|>e_n\}\le\alpha,
\quad e_n=\sqrt{\log(2M/\alpha)/(2n)}.
\tag{15}
\]

Assuming the baseline threshold is known, accepting only
rho_hat-theta>e_n ensures positive true purity margins on that event.
Margins exceeding 2e_n are guaranteed to pass. If delta_p,j=d j^(-nu),
the count of such *uniformly detectable* margins scales as
\((n/\log M)^{1/(2\nu)}\). This is a detection-resolution count, not an
accuracy-optimal k or an upper bound on the number actually accepted.

Equation (15) must not be applied to reference pixels as if independently sampled
from query edits, or to an unweighted episode-IoU mean as if it were class-summed
mIoU. Ratio estimates need count concentration or an appropriate paired influence
analysis.

## 5. Optional family size versus all-on depth

If H_k is the family allowing any subset of k fixed proposals and
H_k is contained in H_(k+1), then for any fixed true objective F,

\[
\max_{S\in H_{k+1}}F(S)\ge \max_{S\in H_k}F(S).
\tag{16}
\]

Proof: every previously feasible recipe remains feasible. The same argument
applies to expected population utility. At a fixed cost budget, containment must
also preserve the old recipe's cost feasibility.

An all-on composition, a greedy selector or an estimated optimum is a particular
policy, not this oracle optimum. Three all-on auxiliaries performing worse than
two does not prove that the enlarged optional family has a worse true optimum.
No claim of universal sparsity follows.

## 6. Supplied facts, conditional interpretation, and falsifiable predictions

The supplied fact 4 reports RCG-relative point losses: reference-similarity
two-stage deletion -.46, graph-cluster addition/deletion -.08/-.04, score-based
addition failures, and losses against matched-count RCG-score controls. The all-on
three-auxiliary composition is worse than two. Confidence intervals were not
supplied here, so these points are not proofs of negative population expectation.

They are consistent with one or more of:

- low helpful probability r and a positive false-inclusion probability;
- small remaining gains a but substantial wrong-edit loss b;
- persistent reference-to-query proxy bias rather than finite-sample noise;
- changes in effective pixel allocation and overlap after previous edits.

Matched edit count localises the failure to allocation/quality rather than merely
edit amount. Under calibrated p and a fixed per-class addition budget, maximising
the sum of added p is optimal for that class's expected-count ratio; under a fixed
deletion budget, minimising the sum of deleted p is optimal. If RCG-score ordering
is monotone in those true conditional probabilities, an auxiliary that selects
inferior pixels cannot beat that matched-count rule. That monotonicity is an
assumption, not a theorem about RCG scores, and global count matching does not
remove class-macro weights.

Fact 9 reports a good fine component, stable around +.4, and a fine plus fitted
area-level combination at +2.49 [1.47,3.47] against FoRIS on frozen new1200,
with group variability. This is compatible with high r or a substantial a for
fine, alongside harmful auxiliary tails. Six levels fitted using other-fold GT
have an explicit labelled calibration setting; they cannot be invoked as proof
that a purely label-free purity estimator is accurate. Neither fact implies that
all components are bad or that sparse recipes are universally optimal.

### Minimal measurement plan — not executed in the theory hour

1. Freeze the reported RCG baseline, the three existing auxiliary recipes, their
   matched-count controls, the two/all-three compositions and the good fine
   component. Do not search another library or alter the main frozen method.
2. From existing600/4000 outputs, later record class-summed I/U and effective
   add/delete TP/FP, edit volume, purity margins and denominator weights. This
   is a nuisance/variance pilot on exposed development data.
3. Compare proxy sign/rank with actual marginal sign both at RCG and after the
   preceding two components. This distinguishes singleton error from interaction.
   Estimate paired photograph-group variance, not a pixel/episode-count substitute.
4. Use the already frozen held1200 outputs for the predeclared paired readout,
   respecting their actual exposure status. Previously inspected labels cannot
   become new confirmation. No additional query GT is used to fit a selector.
5. To study finite-reference uncertainty, predeclare reference-block subsampling
   of the proxy only, preserving the full reference mask for all complete baselines.
   Such within-image samples require a separate dependence model and cannot be
   relabelled independent photographs. This is a proposed diagnostic, not an
   authorised execution or a change to the main recipe.

Predictions: if finite-sample false inclusion dominates under an unbiased proxy,
variance reduction should reduce harmful acceptance in accordance with (7).
If increasing proxy stability leaves a persistent wrong ranking against the
matched-count control, transport bias is implicated instead. If singletons are
positive but their conditional increments become negative after two components,
interaction/context invalidates the fixed-increment model. A genuinely positive
held fine increment falsifies any explanation that assumes no good components.

## Appendix A. Four objectives that must not be conflated

For deterministic GT, J(M,T)=I/U. Class-summed mIoU is
K^(-1) sum_c [(sum_r I_cr)/(sum_r U_cr)], not mean episode IoU and not one pooled
foreground IoU. Within a class it is the union-weighted mean of episode IoUs.

Conditional on allowed information X, let T be random and p_x=P(T_x=1|X).
The ratio of expected counts is

\[
\bar J(M)=\frac{\sum_{x\in M}p_x}
{\sum_xp_x+\sum_{x\in M}(1-p_x)}.
\tag{17}
\]

This does not require label independence. It differs from E[I/U]. Exactly,

\[
\frac{E[I]}{E[U]}-E[J]=\frac{\operatorname{Cov}(J,U)}{E[U]}.
\tag{18}
\]

Thus neither quantity uniformly upper-bounds the other. Deterministic U for
each candidate is a sufficient equality condition.

**Three-pixel ranking reversal.** Let T={a} with probability2/5 and
T={b,c} with probability3/5. For M={a}, expected IoU is2/5 but expected-count
ratio is2/11. For M={b}, both are3/10. The two objectives choose different masks.

**Four-pixel class-macro reversal.** Two classes have truth sizes1 and3.
Candidate A correctly predicts the single first-class pixel and nothing in the
second class: macro1/2, pooled1/4. Candidate B predicts nothing in the first class
and two true second-class pixels: macro1/3, pooled1/2. Pooling reverses the winner.

## Appendix B. Exact ratio certificates and their boundaries

Let H be finite, U(S)>0 for every S, and q*=max_H I(S)/U(S). Then

\[
\max_{S\in H}[I(S)-qU(S)]=0\iff q=q^*.
\tag{19}
\]

Proof: at q*, every residual is nonpositive and a maximiser has zero residual.
Below q*, that maximiser has positive residual; above q*, every residual is
strictly negative. An exact Dinkelbach inner solve and ratio update therefore
reach the optimum on a finite family. Greedy inner maximisation has no such
certificate.

Equation (19) applies also to expected I and expected U, which are deterministic
numbers after specifying p. It does not turn E[I/U] or a general sum of class
ratios into a single fractional program.

For unrestricted pixels and (17), the inner objective has pixel weight
w_x(q)=(1+q)p_x-q. Hence the optimal rule is
p_x>q*/(1+q*), with arbitrary equality ties. Family constraints, budgets,
coupled producer costs and class ratios remove the independent-pixel conclusion.

Class macro can reduce to one ratio under the special condition
U_c(S)=w_c D(S), with fixed w_c>0 common to all candidates. Then its numerator
is K^(-1) sum_c I_c(S)/w_c and its denominator is D(S). General independently
varying class unions do not satisfy this condition.

For fixed-origin A outside P and B inside P, fixed-q utility is signed weighted
coverage on union(A), plus signed weighted coverage on union(B). Add and delete
domains do not overlap, but proposals within a side may overlap, and counting
one producer used on both sides couples costs. Disjoint proposals and uncoupled
cost-free decisions make the inner utility modular. Signed overlapping coverage
does not.

**Five-pixel strict greedy failure.** T={a,b}, P={a,b,x,y,z}.
B1={a,x}, B2={a,y,z}. Baseline IoU is2/5; either deletion alone gives1/4 or1/3,
both worse. Their union leaves{b}, giving1/2. A positive-singleton greedy rule
rejects both despite a strictly better feasible pair. The shared true-pixel
harm is paid once. Four pixels cannot realise this same shared-TP two-deletion
strict-improvement construction: retaining a true pixel requires at least two
initial true pixels, and overcoming the lost true pixel requires at least
three initial false pixels.

Finally, a label-free mask-only statistic cannot universally identify the true
winner when two admissible target worlds share the same observed information
but have different best masks. Any deployed selection certificate must state
the calibration, transport or structural assumptions that eliminate this
ambiguity; a fitted query-GT posterior is a different resource setting.
