# Final-shape audit: mutually exclusive value gaps and their identification limits

Theory-only cross-audit, 2026-10-06 UTC. Owner: `/root/occupancy_gain_geometry`.
No experiment, evaluation, numeric simulation, model call, or remote operation was run.
This note supplements `occupancy_gain_geometry.md`; it proposes no additional method.

## Core answer

**Strict:** an ordered ceiling chain gives nonnegative, nonoverlapping *value intervals*
that telescope. It does not partition pixels into disjoint errors or make graph, readout,
occupancy, and representation effects independent.

The requested chain is valid, with explicit definitions:

\[
V_{\rm actual}\le B_{\Xi,\mathcal F}\le O_{\mathcal F}^{\rm GT}\le1.
\tag{1}
\]

In plain English, current performance cannot beat the best policy using the same allowed
information and action family, which cannot beat the family oracle that sees query GT,
which cannot beat a perfect GT mask. These are population expected values under one law
and one benchmark objective. A finite observed dataset is not silently substituted for them.

However, this chain decomposes the **remaining** gap into three parts: extraction regret,
the value of privileged GT within the family, and family-capacity loss. If an already-obtained
gain relative to a baseline is included, the same chain gives **four** parts. Calling its
last parts "statistically unidentifiable" is generally incorrect.

A three-part decomposition including an obtained gain is possible using an unrestricted
same-information Bayes ceiling, but the correct labels are **obtained gain**, **remaining
same-information decision regret**, and **Bayes irreducibility relative to that information**.
The middle part is "identifiable but not yet estimated" only if a separate statistical
identification and resource premise has been established. That premise is currently unproved.
No positive numeric "unidentifiable budget" follows from the supplied results.

## 1. Fix the objective before defining ceilings

Let an entire evaluation cohort be random under one law `P`. Its GT masks are `Y`, and
its evaluated class assignments and class set follow one fixed protocol. For a complete
cohort mask action `D`, the exact benchmark utility and its population expected value are

\[
v(D,Y)=\frac1C\sum_{c=1}^C\frac{I_c(D,Y)}{U_c(D,Y)},\qquad
R_P(\delta)=\mathbb E_P\bigl[v(\delta(\Xi),Y)\bigr].
\tag{2}
\]

In plain English, expectation is taken *after* forming class-summed IoUs. This is generally
different from the mean of per-image IoUs and from ratios of expected class counts.
Assume positive target unions for the evaluated classes, or specify one common empty-class
convention; without that convention even the perfect-mask ceiling needs clarification.

`Xi` is exactly the information authorized at inference: reference image and complete
reference mask, query observations, frozen features, retained scores, geometry, graph
outputs, and fixed producer memberships as appropriate. It excludes query GT, hidden
class-name input, extra masks, and fitted labels unless those resources are explicitly
part of the comparison. It must also specify whether the policy can see the entire cohort
or only a query episode. Benchmark class coupling does not itself authorize batch access
or class-specific deployment rules.

`F(Xi)` is one fixed admissible family of complete cohort actions. For example, it may
contain a frozen producer/composition family with explicit addition/deletion rules and cost
limits. Every action must satisfy the same information and compute contract. The actual
policy `delta0` is measurable with respect to `Xi` and must belong to this family.
If deployment only permits episode-wise policies, that restriction belongs in the policy
class; the conditional-cohort maximization below then needs that constraint too.

Measurability specifies inference-time information, not how a policy was acquired.
A known-law Bayes ceiling is an informational oracle: its definition does not prove that
the optimal rule can be obtained with one reference, no base-mask fitting, and finite
computation. Acquisition or selector-program constraints give another restricted policy
class below that envelope and may invalidate an unrestricted conditional maximization.

## 2. The requested four-level ceiling chain

**Strict definitions:** for a finite family with measurable maximizers, define

\[
\begin{aligned}
A_P&=R_P(\delta_0),\\
B_{\Xi,\mathcal F}(P)
&=\sup_{\delta:\,\delta(\Xi)\in\mathcal F(\Xi)}R_P(\delta)
 =\mathbb E_\Xi\!\left[\max_{D\in\mathcal F(\Xi)}
                  \mathbb E_P[v(D,Y)\mid\Xi]\right],\\
O_{\mathcal F}^{\rm GT}(P)
&=\mathbb E_P\!\left[\max_{D\in\mathcal F(\Xi)}v(D,Y)\right],\\
O_{\rm all}^{\rm GT}(P)&=1.
\end{aligned}
\tag{3}
\]

In plain English, the Bayes policy chooses before seeing query GT, using its conditional
expected benchmark value. The family oracle chooses after seeing GT. The unrestricted GT
oracle simply outputs the truth. The displayed conditional maximum assumes cohort-wide
actions are allowed; for episode-local deployment, retain the constrained supremum instead.

**Strict proof of ordering:** the actual policy is feasible for the first supremum.
For every `Xi`, a maximum of conditional expected utilities is at most the conditional
expectation of their maximum. Finally every IoU is at most one. Thus (1) follows without
any premise about likelihood ratios or prevalence estimators.

**Strict remaining-gap decomposition:**

\[
1-A_P=
\underbrace{B_{\Xi,\mathcal F}-A_P}_{\text{same-family extraction regret}}+
\underbrace{O_{\mathcal F}^{\rm GT}-B_{\Xi,\mathcal F}}_{\text{privileged-GT value within family}}+
\underbrace{1-O_{\mathcal F}^{\rm GT}}_{\text{family capacity loss}}.
\tag{4}
\]

In plain English, each nonnegative difference occupies one consecutive interval of the
ceiling chain, so the accounting cannot double-count a score gain. The middle interval
does not distinguish uncertainty about the unknown law from residual uncertainty about
this query's GT; the last interval can be caused by restricted mask actions even when the
correct GT is fully determined by the available observations.

If baseline `delta_ref` is a feasible policy with value `Aref`, then

\[
1-A_{\rm ref}=
(A_P-A_{\rm ref})+(B_{\Xi,\mathcal F}-A_P)
+(O_{\mathcal F}^{\rm GT}-B_{\Xi,\mathcal F})+(1-O_{\mathcal F}^{\rm GT}).
\tag{5}
\]

In plain English, obtained gain is a fourth interval. It is nonnegative only if the
actual policy improves this baseline under the same population objective; an observed
positive sample difference does not prove the population ordering. Combining the last
two intervals is valid algebra, but naming their sum "unidentifiable" loses the distinction
between information and action-family limitations.

## 3. A defensible three-part answer including obtained gain

Let `Ball,Xi(P)` maximize the same expected benchmark over *all* allowed `Xi`-measurable
complete-mask policies, without the producer-family restriction. Then

\[
1-A_{\rm ref}=
\underbrace{A_P-A_{\rm ref}}_{\text{obtained gain}}+
\underbrace{B_{\rm all,\Xi}-A_P}_{\text{same-information decision regret}}+
\underbrace{1-B_{\rm all,\Xi}}_{\text{Bayes irreducibility relative to }\Xi}.
\tag{6}
\]

In plain English, this is an additive three-part *performance* statement. It makes no claim
that every latent quantity in the middle is statistically identified, computationally
tractable, or estimable from one reference without external fitting.

The middle interval can itself separate restricted-family extraction from family design:

\[
B_{\rm all,\Xi}-A_P=
(B_{\Xi,\mathcal F}-A_P)+(B_{\rm all,\Xi}-B_{\Xi,\mathcal F}).
\tag{7}
\]

In plain English, a poor selector and an insufficient mask family are distinct failures,
and both can remain even when the full observations contain enough information.

**Strict caution:** `Ball,Xi` and `OGT,F` have no general ordering. A family containing one
poor mask can have low GT capacity while an unrestricted observable policy is perfect.
A rich family can contain the exact GT action in every case while the observations do not
determine which action is correct. Therefore a single chain must not insert the unrestricted
Bayes ceiling between the family Bayes ceiling and the family GT oracle without an additional
proved nesting relation.

**Conditional irreducibility statement:** under finite evaluated masks and unrestricted
actions, `Ball,Xi=1` if and only if query GT can be recovered almost surely by an allowed
`Xi`-measurable policy. If the allowed observations leave positive-probability ambiguity
between distinct GT masks, the population optimum is below one. The current benchmark
results do not establish such ambiguity under the full allowed information or quantify it.

## 4. Statistical nonidentifiability is a different object

Let `Theta` be an explicit statistical model class and `Zobs` denote all data actually
available under the resource contract. Two models are observationally equivalent when

\[
\theta\sim\theta'\iff
P_\theta(Z_{\rm obs})=P_{\theta'}(Z_{\rm obs}).
\tag{8}
\]

In plain English, no amount of observation of those authorized data can distinguish the
two models. A target such as prevalence, conditional edit value, or an optimal decision
is identified only when observationally equivalent models agree on that target. Defining
the observations is essential: a query score histogram, full frozen fields, and all allowed
reference/query observations can have different identification properties.

For an observed-data law `Q`, the identified set and conditional action-value intervals are

\[
\Theta(Q)=\{\theta:P_\theta(Z_{\rm obs})=Q\},\quad
\mu_\theta(D,x)=\mathbb E_\theta[v(D,Y)\mid\Xi=x],\quad
\mathcal I_D(x;Q)=
\left[\inf_{\theta\in\Theta(Q)}\mu_\theta(D,x),\;
      \sup_{\theta\in\Theta(Q)}\mu_\theta(D,x)\right].
\tag{9}
\]

In plain English, these intervals describe what the observation law and model assumptions
identify. They are not mIoU residual budgets. All action values being identified is a
sufficient condition for an identified Bayes choice; it is not necessary if every compatible
model already prefers a common action. Nonidentified prevalence can coexist with an
identified optimal decision, and identified prevalence can coexist with unknown rankings,
geometry, or joint class denominators.

Two logical counterexamples separate the concepts:

- **Nonidentifiability without Bayes irreducibility:** compatible models can make query
  truth deterministic from the same observable query under each model, but prescribe
  different truths because reference-to-query transport is unspecified. Each known-law
  Bayes decision is perfect; the observations do not identify which model's decision to use.
- **Bayes irreducibility without model nonidentifiability:** the true conditional law can
  be exactly known while the retained observations leave stochastic or representation-level
  ambiguity about an individual query's truth. The model is identified, but perfect
  instance prediction is unavailable from those observations.

**Strict conclusion:** there is no canonical numeric decomposition into "identified" and
"unidentified" mIoU merely from these model properties. A robust or minimax value could be
defined after choosing a model set and loss, but it would be a different estimand. Its policy
need not dominate the actual policy under each possible true law, so its insertion into an
ordered pointwise ceiling chain requires proof. No such model set or numerical bounds have
been supplied here.

The phrase "identifiable but estimation has not captured it" is justified only when the
relevant conditional action utility or common optimal decision is identified from authorized
observations *and* the available reference/sample resources can estimate it. Knowing the
true law inside a mathematical Bayes definition does not establish either condition.

## 5. What can actually be numeric now

| Object | Numerical status and boundary |
|---|---|
| Actual and reference complete masks on a fixed cohort | Their empirical class-summed values and matched gains are observable from the supplied GT receipts; this note recomputes none |
| Exact GT oracle within a frozen family on that same cohort | In principle computable from complete masks/counts with the benchmark objective; it is retrospective privileged capacity, not an inference policy |
| Per-image-IoU threshold oracle score `71.24` | An achieved family-feasible GT policy value, not the joint class-summed oracle maximum |
| True-prior formula gain `+5.07` | An achieved privileged rule gain, not optimal use of prior information or the Bayes value |
| Population actual value `AP` and population family-oracle expectation | Require a sampling law and uncertainty; one finite-cohort point estimate is not their exact value |
| `BXi,F` and `Ball,Xi` | Depend on the unknown joint conditional law; not numerically identified by the reported oracles |
| Model nonidentifiability and its action-value intervals | Depend on an explicit model class and observed-data law; no dataset-specific positive budget has been established |
| Unrestricted GT ceiling | One under the stated positive-union/perfect-mask convention; this trivial ceiling is not a SOTA forecast |

The ordered population gaps are unknown quantities, not blanks to fill by subtracting
different-cohort graph, fine-readout, and prevalence-oracle gains. A retrospective empirical
oracle can upper-bound a frozen family's *empirical* attainable utility on that cohort;
it does not determine the expected known-law Bayes value for a new query.

## 6. Graph, readout, membership, and the actual problem space

**Strict information boundary:** if graph output, fine processing, and memberships are
deterministic functions of information already included in `Xi`, adding those derived
outputs cannot improve the unrestricted same-information Bayes ceiling. They can improve
current extraction, feasible action family, ranking, spatial precision, or computational
access to useful relations. If the information is instead defined as a coarse stored grid
alone, finer frozen readouts can increase that *restricted-summary* ceiling; this is not
the same claim as creating extra semantic information beyond the full allowed inputs.

**Strict membership boundary:** a fixed mask bank partitions pixels into membership atoms.
The atom structure is observed, but its GT foreground/background counts are hidden at
deployment. Oracle composition uses those hidden counts; a deployable selector needs their
conditional joint utility or an identified ranking of admissible actions. Under objective
(2), expectations of atom counts alone do not generally determine expected IoU because
the class union is random and coupled. The exact known-count Dinkelbach calculation or a
ratio-of-expected-counts approximation must not be silently substituted for (2).

**Interpretation, not an extra method:** prevalence is one operating-point degree of freedom.
Even perfect prevalence does not specify within-image ranking, missing object identity,
fine extent, membership-atom utility, graph damage, or the shared class tradeoff under unknown
conditional laws. Thus prevalence cannot represent all remaining error, and its oracle gap
cannot be assigned exclusively to one of the ceiling intervals above.

**One-sentence final positioning:** the supported problem space is to turn the observable
complementarity of frozen-DINO graph scores, fine readouts, and complete-mask membership
structure into a deployable joint decision that improves class-summed IoU without query GT;
the identification of that decision's conditional utility remains an empirical/modeling
question, while neither a positive Bayes-irreducible budget nor a total SOTA ceiling has
been established.
