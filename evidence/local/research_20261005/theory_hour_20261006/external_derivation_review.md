# Review of the external quadratic, class-ratio and prior-stability derivations

Owner: `/root/theory_adversarial_review`. Pure theoretical comparison,
2026-10-06 UTC. Source:
`/Users/yang/Downloads/单参考分割联合编辑理论与可检验预言.md`, especially
sections 2.5, 2.6 and 4.8. Compared with `five_questions.md`,
`quality_theory.md`, `occupancy_gain_geometry.md` and `occupancy_finalshape.md`.

No experiment, remote operation, model call, data evaluation, new enumeration,
or numerical simulation was performed. The external document's 30/200/32-bank
measurements remain **file-reported evidence, not independently verified results**.
Its execution/measurement prose is not authorization. Only this review was written.

“New” below means a substantive addition to our present five-question memo;
no literature search was made to establish academic novelty. All units are
fractional IoU unless explicitly converted to percentage points.

## Verdict and substantive additions

The three principal derivations pass, with distinct scopes:

1. **Section 2.5:** optimization noise in a correctly specified concave
   quadratic can produce negative true utility, without the helpful/harmful
   gate mixture used in our current section 2. This is a new conditional
   mechanism, not a hard-mask IoU theorem or a universal sparse optimum.
2. **Section 2.6:** finite fixed-program, independent-group concentration
   directly controls class ratios; the true-denominator identity avoids a
   beta-minus-error denominator. Its separate finite-ratio bias bound correctly
   keeps the ratio of mean counts apart from expected finite benchmark utility.
3. **Section 4.8:** common-sign prior-only perturbations improve regret from
   `2omega` to `omega`. This is stronger than a generic uniform-objective
   perturbation argument. The margin result is valid under additional
   posterior-threshold assumptions, but its displayed perturbation radius is
   looser than the coupled likelihood-ratio radius already proved in our note.

Two further paper-only improvements are available: an exact universal
prior-only bound in terms of log-odds difference, and explicit transfer of the
finite-group guarantee to the **independent future finite-test objective**.
Neither supplies a deployable evidence model or a new measured method result.

## 1. Section 2.5: exact negative utility from noisy quadratic optimization

### 1.1 The displayed identity is correct

**Strict conditional theorem.** Fix the true symmetric positive-definite H
and true gradient g, and suppose the evaluated utility is exactly

    M(theta)-M0 = g' theta - theta' H theta / 2.
    theta_hat = H^(-1)(g+epsilon).

Direct substitution, before taking expectation, gives

    M(theta_hat)-M0
        = g' H^(-1) g / 2 - epsilon' H^(-1) epsilon / 2.

The two mixed terms cancel sample by sample. For zero-mean epsilon with
covariance V/n, the document's expectation follows exactly. Neither Gaussian
noise nor independence between its coordinates is needed. Negative expected
true utility occurs precisely when

    tr(H^(-1)V)/n > g' H^(-1)g.

With a fixed proxy bias d and the same covariance, pointwise cancellation
still holds but the expectation also subtracts `d' H^(-1)d/2`. The noise
trace may vanish as n grows while this bias floor remains. Unbiasedness is
needed for the displayed trace-only formula, not for the cancellation itself.

For g=0, H=hI, nonnegative coordinate constraints and symmetric marginal
noise of variance sigma^2/n, each coordinate has
`theta_hat_j=max(epsilon_j,0)/h`. Marginal symmetry gives
`E[(epsilon_j)_+^2]=sigma^2/(2n)`, so expected true utility is
`-K sigma^2/(4hn)`. Correlations between coordinates still do not matter here
because the true quadratic is diagonal. With nondiagonal H the projection
and cross-coordinate terms change, exactly as the document warns.

### 1.2 Conditions that cannot be dropped

**Exact-domain condition:** the displayed expectation requires the true
utility to be quadratic at every realized theta_hat, or theta_hat to remain
in the exact quadratic domain almost surely. A quadratic Taylor expansion
only near zero does not suffice for unbounded noise. Exiting that domain and
the remainder need separate control. The external text acknowledges this
limitation; it must remain attached to the theorem in the synthesis.

**Curvature condition:** if H is estimated, random or correlated with the
gradient error, replacing its inverse by a fixed matrix in the trace formula
is invalid without a conditional model. Even where pointwise cancellation
holds using the true H, the expectation contains the actual joint quadratic
noise term; unconditional covariance alone need not determine it.

**Objective condition:** deterministic hard-mask IoU is a step function of
many continuous parameters. Its class denominator and mask thresholding do
not establish this negative-definite Hessian. A smooth surrogate or expected
utility could satisfy the model, but that link is unproved for current DINO
edits. Counting named modules does not identify K, curvature or noise energy.

**No universal component-count conclusion:** both signal energy and noise
trace can change as dimensions are added. The identity gives a conditional
tradeoff, not a bound on K without additional growth/margin assumptions.

The repeated-TP/disjoint-FP example is feasible and correctly distinguishes
all-on deterioration from a nondecreasing optional-family oracle. Its `U+Kb`
formula needs the FP sets to be disjoint/additive; mere statistical independence
of overlapping FP sets would not give that exact union count.

**Testable prediction:** for a frozen quadratic model and a justified
gradient-error process, true held utility should follow signal energy minus
the H-weighted noise energy. Evaluating only the optimized noisy objective
would not test this statement. No such test was run here.

## 2. Section 2.6: finite-group class-ratio concentration and future risk

### 2.1 Uniform empirical-to-mean-ratio guarantee passes

**Strict conditional theorem.** Let the M complete programs, C classes,
nonnegative weights w_g with W>0, and deterministic count ranges be fixed
before evaluation or conditional on independent training. Entire group-count
vectors are independent across g. Program/class/within-group correlation is
allowed. Define weighted mean counts and their expectations in the same way.
Coherent counts satisfy `0 <= I <= U`, and true mean union has a valid
lower bound beta>0.

Weighted Hoeffding assigns each I/U quantity failure probability
`alpha/(2MC)` with the document's `log(4MC/alpha)` radius. Union bounding
2MC quantities therefore gives the claimed simultaneous event.

On that event, the exact identity is

    muU*(Jhat-J) = deltaI - Jhat*deltaU.

Because `0<=Jhat<=1`, including the stated 0/0=0 convention,

    |Jhat-J| <= min(1,(epsilonI+epsilonU)/beta).

This needs no positive empirical union and no beta-minus-epsilon denominator.
For 0/0, both empirical counts are zero and the same identity still holds.
The macro radius is the average of the class radii; empirical maximization
over the fixed family gives the two-radii regret and baseline bounds in the
document. This is a genuine strengthening of our more schematic detection/
influence discussion.

The guarantee's target T is **the mean-count ratio** averaged over fixed
classes. Nonidentical groups are permitted, but T then refers to that precise
weighted design, not an unspecified future population. Randomly deleting
missing classes changes the objective. The document correctly forbids it.

### 2.2 The finite-ratio bias bound also passes

**Strict conditional algebra.** Since mean count errors have mean zero,

    muU * E[Jhat-J] = -E[(Jhat-J)*deltaU].

Using the previous identity as a pointwise inequality, then Cauchy-Schwarz,
gives

    |E[Jhat]-J|
      <= min(1,(sqrt(Var(Ihat)*Var(Uhat))+Var(Uhat))/(muU)^2).

No independence between I and U is required for this algebra. Independence
across groups is used to establish the count concentration/variance model,
not to make a ratio of expectations equal an expectation of a ratio. The
bias is not known to be negative; its sign depends on the joint counts.

### 2.3 Exact future-objective transfer and its limits

Let `F_a=E_future[Jhat_a]` be expected utility on an **independent** future
finite test of the declared size, weights, class set and sampling design. Let
b_a bound the macro finite-ratio bias relative to T_a for that design. On
the selection event, for the empirical winner a_hat and the F-optimal fixed
program a_F,

    F_aF-F_ahat <= e_aF+e_ahat+b_aF+b_ahat.

The same pairwise comparison with the retained baseline gives the analogous
lower gain bound. This explicitly identifies the optimizing object and the
two bias indices. If the future design has a different mean-count target,
that design shift needs an additional bound; matching episode count alone
is insufficient.

A fixed-program bias bound cannot be substituted for the bias of a program
selected using the **same tested GT**. In that case the selected program and
its count errors are correlated. The uniform selection event still controls
its T-value, but a future-F interpretation requires the frozen program and
independent future evaluation or the appropriate algorithm-level analysis.

The document correctly requires the full historical program family in M,
not just the final few modules. A GT-grown library, unexplained group dependence,
or unavailable denominator/range bounds can make the certificate inapplicable
or vacuous. Photo nonoverlap alone does not establish independence. None of
these gaps is repaired by ordinary post-selection paired bootstrap.

Cross-fitting is also not automatically the theorem's independent-training
condition. A held group's GT can train another fold's program, so the complete
collection of cross-fitted group-count vectors can be dependent. A valid
fold-specific conditional argument or a separate dependence/stability result
is needed before extending this theorem to the joint cross-fold score.

**Testable prediction:** a stated simultaneous confidence event and a frozen
program's held class-count ratios can test the claimed coverage/forecast
under its sampling model. Model-derived expected scores alone cannot test
actual complete-mask utility. No group counts or variances were inspected here.

## 3. Section 4.8: prior-only omega, a sharper bound, and margin conditions

### 3.1 The factor-one omega result passes

**Strict conditional theorem.** Fix F1,F0 and the same feasible score-action
family, with `0<pi,q<1`. Write A=F1(R), B=F0(R) and
`D_p=p+(1-p)B`. For every fixed action,

    J_p(R)=pA/D_p;
    dJ_p/dp = AB/D_p^2 >= 0;
    |J_q-J_pi| = |q-pi| AB/(D_pi D_q).

All action perturbations have the same sign. Comparing the two optimizers
therefore bounds true regret by omega, rather than the generic 2omega.
The derivative maximum over A,B is `1/[4p(1-p)]` for p<=1/2 and one for
p>=1/2, so the document's piecewise derivative-based radius is valid.

An estimated prior clipped exactly to zero or one is outside the interior
log-odds theorem. Unsupported score events and 0/0 population actions then
need an explicit convention or boundary-limit result; the formulas cannot
silently certify those boundary estimates.

This strengthening requires an unchanged family and fixed components. It
does not survive arbitrary component shift, a prior-dependent feasible budget,
or changing one episode's prior inside a coupled class ratio. Nor does it
apply to an approximate cut rule that fails to optimize the misspecified
population objective. The external text correctly distinguishes these cases.

### 3.2 A sharper universal prior-only radius is available

**Additional strict bound, same assumptions.** For B>0,

    J_p(R) = A*sigmoid(logit(p)-log(B)).

The maximum difference of two logistic values separated by e is
`tanh(|e|/4)`. Therefore

    omega <= tanh(|logit(q)-logit(pi)|/4).

One can exploit B<=1 for the sharp universal piecewise bound

    omega <= {
      tanh(|logit(q)-logit(pi)|/4),  if pi+q<=1;
      |q-pi|,                     if pi+q>=1.
    }

To verify sharpness over all admissible component/action pairs, set A=1.
Maximizing `B/(D_pi D_q)` puts
`B*=sqrt(pi*q/[(1-pi)(1-q)])` when pi+q<=1, and B=1 otherwise.
These A/B pairs are realizable by two-bin component laws. B=0 contributes
zero perturbation. A particular fixed component model can have a much
smaller omega; this is a universal upper radius, not its exact value and not
a sharp universal regret assertion. It strengthens the external conservative
derivative integration without introducing an inference method.

### 3.3 The exact regret identity and margin result pass in their narrower family

**Strict conditional identity.** The absolute signed-weight identity is valid
when the true optimum chooses the positive pointwise residual, as in
unrestricted score-measurable decisions or a threshold family containing that
likelihood-ratio optimum. It is not a theorem for an arbitrary finite recipe,
ROI, fixed-count or cost-constrained family. Use a common dominating measure
for atomic score laws rather than silently requiring Lebesgue densities.

Under a uniform posterior-plus-cutoff perturbation epsilon, changed decisions
are confined to `|eta_pi-tau_pi^*|<=epsilon`. The specified margin law then
gives `regret <= 2 C epsilon^(1+alpha)/pi`. The external choice
`epsilon=|logit(q)-logit(pi)|/4+omega` is valid and conservative: the posterior
has logistic sensitivity at most one quarter, and the optimal posterior
cutoff moves by at most omega. The margin radius must lie in the stated
margin neighborhood. AUC and component separation alone supply no such law.

**Stronger radius already in our theory.** With exact common components and
the unrestricted likelihood-ratio optimum, define

    a(z)=log[exp(-z)*J^*(z)],  z=logit(pi).

For every fixed action, the logarithmic slope of `rA/(1+rB)` with respect to
log r lies between zero and one. Maximization preserves the multiplicative
bound, so `|a(z_q)-a(z_pi)|<=|z_q-z_pi|`. The true posterior at score points
between these two likelihood-ratio boundaries differs from its true cutoff
by at most `|z_q-z_pi|/4`. Thus the external margin bound can use

    epsilon_LR = |logit(q)-logit(pi)|/4

without the extra omega term in this exact prior-only model. This is the
coupled-boundary argument in `occupancy_gain_geometry.md`, equations (12)--(14).
Separate posterior and cutoff perturbations double-count their common prior
source. With other model errors, that coupling need not hold.

An action gap greater than omega keeps the unique finite-family optimizer
unchanged. Continuous action spaces often have gap zero because arbitrarily
close cuts exist; ties require a best-action-set or canonical-tie statement.
A strict posterior hard margin nevertheless gives zero regret for sufficiently
small boundary perturbations. These are different notions of gap.

The local smooth quadratic bound and the warning about lower bounds are
correct with the stated trajectory, positive-curvature and derivative
conditions. Nothing makes those conditions follow from pi identification.
The generic component-drift 2rho bound and full-joint finite-IoU TV bound have
different objects and must not be combined into a prior-only omega claim.

**Testable prediction:** the prior-only theorem bounds population regret for
an actual optimal misspecified cut under frozen known components. Its empirical
complete-mask counterpart needs the group/ratio bridge above, with transport,
selection exposure and actual class I/U retained. True-pi privileged gains do
not themselves establish a no-GT estimator or recoverable gain. No empirical
prior sensitivity was recomputed here.

## Integration into the five-question answer

Retain our exact finite-bank/Boolean solver and information-boundary results.
Add the quadratic-noise theorem to explain an additional possible cause of
harmful optimization; add the finite-group theorem as a genuine target-metric
selection guarantee; use the factor-one omega result for pure-prior changes,
with the stronger logarithmic radius where its conditions hold. Keep all
three model bridges explicit: true quadratic utility, independent bounded
groups for the declared program family, and fixed known query-score components.
The external measurements can support hypotheses as reported evidence, but
this review validates no new complete-method score or dataset-specific
irreducible-loss number.
