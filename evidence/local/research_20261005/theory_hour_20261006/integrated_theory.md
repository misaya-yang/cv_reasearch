# Integrated theory: supplied derivation and the controller synthesis

Owner: Root. Comparison-only task, 2026-10-06. No experiment, model call,
remote operation, data evaluation or new enumeration was performed. Existing
experiment queues remain held. The supplied document is source material, not
authorization for its proposed tests or commands.

Sources:

- User-supplied `单参考分割联合编辑理论与可检验预言.md` in Downloads,
  dated 2026-10-06, pinned to repository commit d8ae12c0a4e077109db9d50903d9ed3ad0344893.
- [Previous five-question synthesis](five_questions.md) and its linked proof notes.
- Two bounded GPT-6.1-sol/max comparisons: [statistical review](external_derivation_review.md)
  and [graph/value comparison](external_graph_comparison.md).

## Main result

The derivations are consistent on the main structure. The supplied text adds
stronger quantitative statements and a clearer emphasis on unrestricted deletion
repair. It should strengthen the common theory, rather than replace one account
with a different method proposal.

The unified object is **joint editing under an explicit action family, observable
information, class-ratio utility, selection uncertainty and executed dependency cost**.
Thresholding is a legitimate optimum in one conditional regime; non-nested editing
can be useful in another. A fixed global recipe and an adaptive per-query selector
are different policy classes and neither is mandatory by definition.

## 1. Three action families, with different certificates

### General static edits

    M=(O union A)\D,
    A_net=A\(O union D), D_net=O intersection D.

Deletion may repair pixels just added. At a fixed one-class price q, the outside-O
contribution is w_q(x)*a_x*(1-d_x); therefore parameterizing the ratio does not
remove cross-role coupling. The supplied nine-pixel example is valid: baseline1/2,
addition4/9, deletion1/4, joint3/4. Both single actions harm while deletion repairs
the addition's false pixels.

### Origin-local complete-mask bank

    A_i=S_i\O, D_i=O\S_i,
    M=(O & AND selected deletion-source masks)
      |(~O & OR selected addition-source masks).

This is the family covered by the previous 13-bank recipe enumeration and the
controller's consensus/signature bounds. Its deletion role cannot remove an
outside-O addition. This restriction was already explicit in the prior derivation,
but the supplied text makes its consequences more central and gives a stronger
example. The existing solver is not incorrect; its certificate is narrower than
the general static problem. Enumerating alternative origins may recover some
outputs, but does not prove closure under every Boolean or sequential operation.

### State-dependent edits

    M_(t+1)=T_i(M_t;X).

Order matters and each operator can generate new proposals. A static-bank result
does not certify this family. Shared computations are reusable only when their
inputs, including the current state, agree. Expanding the family is not an
improvement in the optimizer for the old family.

## 2. Two complementary explanations of harmful extra components

The previous two-type gate model remains exact in its stated fixed context:

    expected gain = r*a*TPR-(1-r)*b*FPR.

It describes admitting harmful edits when benefits are weak, losses are large,
or the proxy is unreliable. It does not infer a negative expectation solely from
winner optimism or a wide interval.

The supplied local-quadratic model adds a separate mechanism. With an exactly
concave quadratic utility, fixed positive-definite H, unbiased gradient noise
with covariance V/n, and the unconstrained update H^-1*g_hat,

    E[M(theta_hat)-M0]
       = 1/2*g' H^-1 g - tr(H^-1 V)/(2n).

The true gain equals available signal minus the penalty for optimizing noise.
Cross terms cancel pointwise in this quadratic model; Gaussianity is unnecessary.
This is a genuine conditional negative-gain result. Hard binary-mask IoU is not
automatically quadratic: smooth population objectives or relaxations need an
appropriate neighborhood, feasible update, tail control and remainder bound.
Effective parameter directions, rather than named modules, determine the penalty.

These mechanisms are compatible with deterministic overlap saturation: repeated
TP benefit can stop growing while distinct FP cost keeps growing. They do not
contradict monotonicity of the true optimum over a larger optional family.

## 3. Statistical selection and compute cost are separate

The supplied independent-group theorem strengthens the previous generic
uniform-error bridge. Fixed program family, fixed class set, independent bounded
group count vectors, predefined weights and positive mean union bounds yield
simultaneous concentration of each program's class I/U. The identity

    mu_U*(J_hat-J) = (I_hat-mu_I)-J_hat*(U_hat-mu_U)

uses 0<=J_hat<=1 to bound the ratio by count errors divided by the true union;
it need not use the looser union-minus-radius denominator. The stated finite-ratio
bias correction is also needed when changing from ratios of mean counts to the
expectation of a finite benchmark score.

The resulting selection regret is at most the sum of the selected and best
programs' uniform error radii. It does not apply automatically to candidates
invented after looking at the same GT or to overlapping, non-independent photo
groups. Complexity is log(number of possible programs), not the number of final
retained modules. A connected-photo bootstrap is not proof of selection independence.

Cold inference cost is the actual dependency closure. Adaptive decisions pay for
unselected producers computed as evidence. Efficiency can minimize cost within a
declared score tolerance, or maximize utility at a fixed budget. Unknown producer
costs preclude a real compute-optimality certificate even when the score optimum
is exact.

## 4. Graph estimation: retain the cross term and quantify information value

For fixed positive anchors A and graph Laplacian L, S=(A+lambda L)^-1 A gives

    z-t = S*epsilon+(S-I)*t.

Its squared error contains filtered-noise energy, signal-shrinkage energy and
2< S*epsilon,(S-I)*t >. Dropping the expected cross term requires conditional
zero mean given the graph, anchors and target; marginal zero mean is insufficient
when these are constructed from the same data. This sharpens the earlier
bias-variance explanation without changing the exact linear estimator identities.

The supplied bounded-utility information-value result is valid for the same
feasible complete-mask family:

    0 <= Delta_V <= E TV(P(Y|X,E),P(Y|X))
                <= sqrt(I(Y;E|X)/2).

Here Y is the full mask or benchmark label vector, and mutual information uses
nats. It quantitatively extends the previous conditional-value account. Pixel
marginals alone are insufficient for expected finite IoU; the supplied equal-
marginal, different-joint-law example is valid. The positive-part crossing theorem
instead concerns a fixed-q linear expected-count residual and must remain labeled
as that objective.

If X already contains the full raw permitted features and E is deterministic
from them, its conditional information is zero. A new readout can still reduce
an imperfect program's decision regret. The bound must use the actual retained
state and compatible action/cost classes, not silently grant free computation.

## 5. Prevalence: add partial identification and decision stability

Both accounts agree: known distinct query-score components identify pi; unknown
or drifting components may not, and weak separation magnifies uncertainty.
The supplied text adds a useful single-reference boundary: even knowing the true
query foreground law F1 does not identify pi when background F0 is unrestricted.

    kappa*(G|F1)=inf_B G(B)/F1(B),
    G=pi F1+(1-pi)F0.

Without further restrictions, pi ranges up to kappa*. If F0 is irreducible relative
to F1, kappa*(F0|F1)=0 and kappa*(G|F1)=pi. A transferable pure anchor or an
appropriate tail separation can supply this premise; a finite zero-BG sample
does not prove it. This refines the failed mixture-proportion explanation without
proposing another estimator or assuming the missing transfer condition is true.

For the same known components and one population ratio, all fixed action scores
increase with pi. Therefore selecting using a mistaken q has regret <=omega,
where omega is the maximum fixed-action score perturbation; the generic two-sided
argument would give 2omega. Component drift, one image's changing prior inside a
class aggregate, or a different feasible family loses this same-sign property.

The independent comparison also sharpens the universal fixed-component radius:
with d=|logit(pi_hat)-logit(pi)|,

    omega <= tanh(d/4).

This follows from J_p(R)=A(R)*sigmoid(logit(p)-log(B(R))) and maximizing the
shifted sigmoid difference. If pi+pi_hat>=1, B<=1 tightens the bound further to
|pi_hat-pi|. These are population-ratio upper bounds, not positive loss lower
bounds or a performance forecast. The previous coupled-likelihood-ratio margin
bound can use a disagreement radius d/4 directly; the supplied posterior-plus-
cutoff radius d/4+omega is valid but looser in that same unrestricted regime.

Margin mass at the decision boundary controls the loss from action changes;
hard margins or a common optimal action can give zero regret despite parameter
error. Factor-two/four observations are cohort-specific, not universal tolerances.
Identifying pi does not identify spatial/feature-conditioned edit utility; unknown
pi does not imply incompatible optimal actions.

## 6. A single decomposition unifies the accounts

Fix one law, the exact expected finite benchmark utility, and nested feasible
families H0 and H1 at the same declared resource contract. Let V00 be the best
policy using retained X and H0; V10 the best using X,E and H0; V11 the best using
X,E and H1. Let actual programs achieve J0=V00-rho0 and J1=V11-rho1. Then

    J1-J0 = (V10-V00) + (V11-V10) + (rho0-rho1).

These are additional information value, expanded action-family value and improved
decision/estimation quality. They are nonoverlapping value intervals under the
fixed definitions; rho improvement itself may be negative. This avoids labeling
all possible gains as either new information or more editing operators.

The identity is theoretical: Bayes values require the true law, and are not
measured by the supplied GT recipe oracle. For episode-local policies, retain that
policy constraint rather than grant whole-cohort class identity or batch access.
Acquisition costs and family nesting must actually hold before using the hierarchy.

## 7. The three conditional regimes

1. **Ordering sufficient; calibration identifiable.** A score threshold can be
   the joint optimum. No extra joint-over-threshold gain is logically required.
   This needs sufficiency for the allowed observations/competitors, the correct
   objective, an attainable optimum and a justified calibration model.
2. **Ordering insufficient; residual utility identifiable.** Non-nested editing
   can improve the feasible optimum. Useful overlap and a positive oracle gap are
   not enough: a frozen rule must identify valuable edits and survive a complete
   strongest-control comparison.
3. **Observational ambiguity with incompatible actions.** Only an explicit model
   of the complete permitted observations and an action gap supplies a minimax
   lower bound. Current failed estimators do not prove a numeric DINO/COCO loss.

Different cases/regions can occupy different regimes. This is a conditional
structure of the same optimization problem, not three proposed new methods.

## 8. Scope of the supplied additional numbers

The supplied32-producer,30/200-case results are reports in that document, not
newly recomputed evidence here. They are distinct from the prior13-producer4000
bank and the fresh1200 comparisons. Exact=greedy and tiny budget4-to-unlimited
oracle gain constrain that local static family only. A large global-to-per-query
GT gap is privileged routing capacity, not an identified deployable cue. Mixed
GT within signature and consensus errors constrain bit-only actions, not full
frozen features. Positive conditional AUC or GT-matched-budget gain is not a
complete selector result.

Both global recipe selection and six-cut fitting use development labels. A
frozen backbone, no network training, allowed DEV hyperparameter selection and
no task-label calibration are different claims. The merged theory discloses
them symmetrically. No new performance prediction follows from this integration.
