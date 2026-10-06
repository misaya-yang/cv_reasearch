# Identifiability of an unlabelled-query selector and foreground proportion

Owner: `/root/rcg_ablation_audit`. Pure theoretical note, 2026-10-06.
No experiment, remote access, GPU work, data evaluation or parameter search was run.
Facts7/8 below are the user's supplied observations, not newly verified results.

## Core conclusions

1. One labelled reference and arbitrarily many deterministic, correlated prior masks do not, by themselves, identify the best complete A/B recipe on an unlabelled query. Two label-generating worlds can preserve every available observation and reverse the best recipe. This is a worst-case statement about the unrestricted model class, not a proof that the present DINO/COCO data cannot support selection.
2. If the *query-score* class-conditional distributions `F1,F0` are known and different, foreground proportion `pi` is uniquely identified by the query histogram. A failed mixture implementation does not negate this theorem. Approximate transport, weak separation and finite effective sample size can make that identifiable inverse unusable.
3. Unknown class-conditional distributions generally do not identify `pi`. Three nondegenerate views can identify a two-class latent model under conditional independence, transferable orientation and stable error rates. Two unrestricted unknown views generally cannot. A shared encoder neither proves nor disproves conditional independence.
4. Agreement, augmentation stability and voting are observations of predictors. They become posterior evidence only through a calibrated generative or discriminative model with explicit assumptions. Common errors can be perfectly consistent.
5. Useful theory is possible: bound component transport and inverse conditioning; obtain a calibrated posterior/risk interval; select a recipe only when its lower score bound exceeds competitors' upper bounds. Otherwise the result is an identified set of plausible winners, rather than an invented certain winner.

## 1. What is observed, and a strict two-world construction

Let `O` contain the full reference RGB and mask, query RGB, frozen encoder and all fixed method parameters. Include every prior mask, continuous field, reference self-evaluation and unlabelled augmentation result that a selector may inspect. Let `H(O)` be the finite family of complete Boolean A/B masks allowed by the fixed construction. A selector is any measurable, possibly randomized function of `O`.

Take a query domain split into two nonempty disjoint sets `P,Q`, and include the two candidates `P,Q` in the family. In both worlds:

- the reference and its complete labels are identical;
- query RGB/features and all fixed prior outputs are identical;
- every reference self-evaluation output is identical;
- transformed outputs are perfectly equivariant, and all augmentation randomness has the same law.

World0 has query truth `T=P`; World1 has query truth `T=Q`. Both are legitimate nonempty same-class queries under an otherwise unrestricted reference-to-query label relation. For example, the reference contains labelled feature types `z_F,z_B`; the query contains two previously unconstrained feature types `z_a,z_b`. World0 labels `z_a` foreground and `z_b` background; World1 reverses them, while both retain the reference labels. The unknown semantic labelling functions agree on the reference and differ on the new query appearance. This does not assert that physical COCO annotations actually possess two interpretations; it specifies the missing bridge assumption.

The admissible model class matters. If the canonical semantic labelling rule is fixed and deterministic on full RGB, arbitrary different labels for the same RGB are not two admissible COCO worlds. A feature-only obstruction would instead require two genuinely different images/latent scenes with different canonical truths but the same allowed feature/prior observation `Xi`. If full RGB is allowed, these images may be distinguishable and that obstruction no longer applies. Such a collision or small observable-TV pair has not been demonstrated on the present dataset. The construction establishes lack of a distribution-free guarantee when the reference-to-query labelling relation is unrestricted, not an empirical irreducible-loss certificate.

The observable distributions are exactly equal. The best candidate has IoU1 in each world, but its identity reverses.

In fact, allowing a selector to return *any* mask `M`, rather than only `P,Q`, does not remove this construction's lower bound. Write `a=|M intersect P|`, `b=|M intersect Q|`, and `c=|M outside (P union Q)|`. Then

```
IoU(M,P) = a / (|P| + b + c) <= a / (a+b+c),
IoU(M,Q) = b / (|Q| + a + c) <= b / (a+b+c).
```

Thus the two IoUs sum to at most1. For an empty `M`, both IoUs are0. The same inequality holds after averaging a randomized selector. Consequently,

```
sup_world E[best-family IoU - selected IoU] >= 1/2.
```

This is a valid unrestricted minimax lower bound. It does **not** show that a fixed COCO cohort has regret1/2, that every DINO representation is insufficient, or that no more restrictive statistical model can identify a useful selector. Perfect reference performance and perfect multiview consistency can coexist with this bound because they do not constrain the new appearance's label relation.

For any two distinct masks `P,Q`, choosing truth `P` versus truth `Q` still reverses their ranking. Their separation may be very small when they are highly correlated. The large disjoint construction proves a worst-case obstruction; it cannot be substituted for a measured lower bound on today's correlated candidates.

## 2. Le Cam: what would be needed for a dataset-specific impossibility claim

Let `P0^O,P1^O` denote the two *observable* laws, and let `v=TV(P0^O,P1^O)`. Suppose the two worlds have incompatible optimal actions and, on the common observable support, every action has sum of its two regrets at least `gamma`. Integrating with the smaller of the two observable densities gives

```
inf_selector sup_world E[regret] >= gamma * (1-v) / 2.
```

For the preceding construction `v=0,gamma=1`. For a binary choice whose wrong choice costs at least `Delta` in either world, use `gamma=Delta`. The overlap argument is also the binary testing bound: average classification error is at least `(1-v)/2`.

A current-dataset lower bound would therefore need an explicitly justified admissible pair of data-generating distributions, their observable TV distance or another bound on distinguishability, and an incompatible latent-label/optimal-action gap. Different GTs with the *same marginal score histogram* are not sufficient if the selector can also inspect RGB, spatial geometry or other feature channels. The observable law must include the selector's complete information set. A failed estimator or a negative aggregate gain establishes none of these ingredients.

For masks, Jaccard distance `d_J(A,B)=1-IoU(A,B)` is a metric. If both world truths `T0,T1` are available candidates, their optimal scores are1 and their regrets are these distances. Triangle inequality supplies `gamma=d_J(T0,T1)`, yielding

```
inf_selector sup_world E[regret] >= d_J(T0,T1) * (1-v) / 2.
```

For truth masks depending on observations, a uniform lower distance on the common observable support can replace the constant distance. Again, admissible worlds and the *entire* allowed `Xi` must be specified before applying this to a real representation or dataset.

For estimating a scalar proportion, two admissible worlds with proportions `pi0,pi1` give the direct absolute-error bound

```
inf_estimator sup_world E|pi_hat-pi| >= |pi1-pi0| * (1-v) / 2,
```

because `|a-pi0|+|a-pi1| >= |pi1-pi0|`. The analogous squared-error bound is `|pi1-pi0|^2*(1-v)/4`. When proportions are positive, the same argument can be applied to `log(pi)` to express multiplicative uncertainty.

## 3. Known class-conditional score laws identify pi

Fix the exact scoring operator, its reference conditioning, normalization, precision and other settings. Let `F1,F0` be the probability laws of this score on query foreground/background, respectively. Suppose these two laws are known and the observed population query law is

```
G_pi = pi F1 + (1-pi) F0,       0 <= pi <= 1.
```

For two proportions,

```
G_pi - G_pi' = (pi-pi') (F1-F0).
```

Therefore the map is one-to-one **if and only if `F1 != F0`**. If they are equal, every proportion has the same score law. Otherwise there is an event `B` with `d_B=F1(B)-F0(B) != 0`, and

```
pi = [G_pi(B)-F0(B)] / d_B.
```

Even a histogram suffices if its bins preserve some nonzero component difference. Coarse binning can destroy identifiability despite differences in the full continuous distributions. With the same known distinct components, exactly identical score histograms cannot correspond to different proportions in that binned model.

Population uniqueness is different from numerical/statistical stability. With `d=TV(F1,F0)`,

```
TV(G_pi,G_pi') = |pi-pi'| d.
```

The inverse's condition number is `1/d`. If `d` is small, apparently minor component mismatch or sampling noise can produce a large proportion error.

### Approximate invariance and a usable error bound

Suppose the frozen component templates are `F1,F0`, but actual query components are `F1q,F0q`, with `TV(Fyq,Fy) <= eta_y`. Set `eta=pi eta_1+(1-pi)eta_0`. If the query distribution estimate is within `epsilon` TV and `pi_hat` minimizes mixture residual TV, triangle inequality gives

```
|pi_hat-pi| <= 2(epsilon+eta) / d.
```

More informatively, let `r_hat=TV(G_hat, G_pi_hat)` be the actual fitted residual. Then

```
|pi_hat-pi| <= [r_hat+epsilon+eta] / d.
```

A small residual alone is insufficient: the templates may be shifted in a direction that another mixture proportion absorbs. A large residual can reject the assumed mixture family, but a small residual cannot certify semantic component transport.

For one fixed event `B`, let labelled component probability errors be `epsilon_1,epsilon_0`, query event-frequency error be `epsilon_q`, and query/template shift at that event be at most `eta_B`. If `|d_B| > epsilon_1+epsilon_0`, the clipped plug-in estimator satisfies

```
|pi_hat-pi| <= [epsilon_q + eta_B + max(epsilon_1,epsilon_0)]
              / [|d_B|-epsilon_1-epsilon_0].
```

The denominator is a directly relevant separation lower bound. Approximate component invariance is useful only relative to this denominator and the precision needed for the proposed decision. A small absolute error can still be a large multiplicative error when `pi` is small.

### Constructive ambiguity under a bounded foreground shift

An exact histogram-level counterexample quantifies why “nearly invariant” needs a relative scale. World0 has

```
G = pi0 F1 + (1-pi0)F0.
```

For any `pi1>pi0` with `pi1<=1`, World1 can have

```
F0' = F0,
F1' = (pi0/pi1)F1 + (1-pi0/pi1)F0.
```

Its query histogram is exactly the same `G`, while

```
TV(F1',F1) = (1-pi0/pi1)*d,       d=TV(F1,F0).
```

The components remain distinct. If the original foreground is ordered above background, the mixture retains that orientation with a smaller gap. Thus a foreground-shift allowance `epsilon_1 >= (1-1/r)*d` admits indistinguishable histogram worlds with occupancy ratio `r=pi1/pi0`. Fourfold ambiguity needs only `.75*d` foreground shift; when component separation is small, this can be small in absolute TV despite a large occupancy error.

For this particular construction and `epsilon_1<d`, the admitted ratios satisfy `r<=1/(1-epsilon_1/d)`. This is not a general sharp identified-set formula allowing arbitrary changes to both components. It is a concrete admissible ambiguity witness. With no observable distinction, any single estimate has worst-world expected absolute log-error at least `log(r)/2`. At `r=4`, the best central log-scale choice reaches a worst-case factor2 error; `r>4` excludes a universal factor2 guarantee. An estimator that fixes World0's templates can instead output `pi0` and be wrong by factor4 in World1.

This lower bound concerns the scalar-score histogram information set. Additional DINO coordinates, geometry or RGB may distinguish the worlds. It cannot be extended to full-DINO/RGB nonidentifiability without constructing their corresponding observable laws. It supplies a premise audit for a failed mixture implementation, not a proposal to rerun that implementation.

Under independent score sampling with known densities, local information per sample is

```
I(pi) = integral (f1-f0)^2 / [pi f1+(1-pi)f0].
```

For `n` independent observations, `KL(G_pi^n || G_pi'^n) <= n*(pi-pi')^2*I(pi')`. Pinsker's inequality bounds observable TV by `|pi-pi'|*sqrt(n I(pi')/2)`. Thus finite samples can be hard to distinguish even though the population mixture is uniquely identified. Spatially correlated pixels cannot simply be counted as `n` independent observations; the joint-law bound or a justified effective sampling model is required.

For a regular known-component iid model, an unbiased estimator obeys the Cramer-Rao bound `Var(pi_hat)>=1/[n I(pi)]`; analogous asymptotic information bounds require local/asymptotic regularity. The corresponding small-error relative/log-proportion variance is approximately at least `1/[n*pi^2*I(pi)]`. This is **not** a universal variance bound for all biased estimators. The preceding two-world/TV risk bounds apply without invoking unbiasedness.

To enter a useful multiplicative-error regime, mixture/template uncertainty must be small relative to `pi*d`, not merely small on a probability scale: the sufficient shift-bound route requires `epsilon+eta << pi*TV(F1,F0)`. Rare foreground is therefore a precision problem even when the existing representation contains occupancy information. The relevant prerequisite measurements are separation/information, template shift and effective sample size. This is a measurement criterion, not a newly proposed proportion estimator.

### Unknown components: a different problem

When `F1,F0` are unrestricted and unknown, the histogram alone generally does not determine `pi`. For a given `G`, choose a bounded signed measure `H` of total mass0 and small enough magnitude to retain nonnegative laws. For any interior `pi`, set

```
F1 = G + (1-pi)H,
F0 = G - pi H.
```

Their mixture is always `G`; for nonzero `H` the components are distinct. Orienting the foreground to higher scores, by choosing the sign of `H`, still leaves different valid proportions. Unknown distributions need additional restrictions, known anchors, transferable labelled calibration or multiple informative independent views. This nonidentifiability result cannot be applied after the two correct components are stipulated as known.

## 4. What facts7/8 do and do not show

**Fact7, supplied:** approximately image-invariant foreground/background score laws; using true proportion in the cut formula yields `+5.07 [4.27,5.51]`; proportion errors within factor2 retain about `+3.9`, while errors above factor4 can turn gain negative.

This supports the utility of occupancy information under the tested readout and the potential value of an accurately calibrated proportion. True query proportion is privileged diagnostic information. The result does not establish an unlabelled proportion estimator. Approximate invariance is compatible with identifiable population mixtures *and* unusable finite-sample inverses: the relevant condition is template/estimation error divided by component separation and by the small proportion scale. The other theory agent owns the precise optimal-cut/macro-IoU formula; this note does not replace that formula.

**Fact8, supplied:** score-histogram mixture gives `-4.08`; reference/query area Spearman is about `.25`; geometry-matched scale fails with multiple objects/occlusion; a reference-positive mixture returns proportion0 on most cases because reference and cross-image features have different distributions; predicted-mask area has seemingly small errors but those errors correlate with score errors and gives `-.58`.

These observations do not prove that a histogram with the *correct known query components* cannot identify proportion. They point to concrete failure links:

- A reference scored against itself is not automatically a sample from the query-score foreground law. Self-matching, cross-image nuisance, feature normalization and reference conditioning change that law. Its use requires an explicit transport assumption or a transport bound.
- A low-separation mixture amplifies small template mismatch. Boundary/clipped estimates at0 are compatible with misspecified components; clipping is not evidence that the true proportion is0.
- Reference area and a geometrically rescaled reference are not substitutes for query occupancy without assumptions about object count, visibility and sampling. The supplied failures reject those constructions' transport assumptions.
- Predicted area is endogenous. At a fixed score cut `t`, `a_pred=pi TPR(t)+(1-pi)FPR(t)`. It identifies proportion only when these rates are known and different. Treating `a_pred` directly as `pi` leaves bias `(1-pi)FPR-pi FNR`, built from the same mistakes that affect IoU. Small marginal area error does not ensure a useful downstream cut when errors concentrate on precisely the difficult cases.

The aggregate gains alone cannot decide whether component shift, weak separation, inaccurate finite-sample fitting or downstream sensitivity dominates. A theory should predict their distinct signatures rather than label every failure “not identifiable.”

## 5. Two versus three views

For a random query pixel let `Y` be its foreground label, and let `A_j` be fixed binary outputs from view `j`. Assume:

1. `0<pi=P(Y=1)<1`;
2. the outputs are conditionally independent **given the true label**, within the declared population/stratum;
3. `a_j=P(A_j=1|Y=1)` and `b_j=P(A_j=1|Y=0)` are stable rates;
4. `Delta_j=a_j-b_j` is nonzero;
5. foreground/background orientation is anchored correctly and transports to the query population.

The model has `E[A_j|Y]=b_j+Delta_j Y`.

### Two unknown views are generally insufficient

Two binary views provide three joint-distribution degrees of freedom, versus five unknown parameters `pi,a1,b1,a2,b2`. A concrete same-observation example is:

```
Observed: E[A1]=E[A2]=.5, Cov(A1,A2)=.04,
          P(11)=P(00)=.29, P(10)=P(01)=.21.

Model0: pi=.5, Delta1=Delta2=.4, b1=b2=.3, a1=a2=.7.
Model1: pi=.25, Delta1=Delta2=sqrt(.04/(.25*.75)),
        b1=b2=.5-.25*Delta, a1=a2=b+Delta.
```

Both models obey conditional independence and positive orientation; Model1 has approximately `b=.38453,a=.84641`. Their observable two-view distributions are identical and their proportions differ. A known full-rank confusion channel, a justified anchor/separability condition or other constraints can make two views sufficient. For example, a known `a1 != b1` gives `pi=(E[A1]-b1)/(a1-b1)` from even one calibrated view.

### Three informative independent views identify the binary latent model

Write `mu_j=E[A_j]`, `c_jk=Cov(A_j,A_k)` and
`t=E[(A1-mu1)(A2-mu2)(A3-mu3)]`. Under the stated model,

```
c_jk = pi(1-pi) Delta_j Delta_k,
t = pi(1-pi)(1-2pi) Delta1 Delta2 Delta3.
```

Assume positive orientation `Delta_j>0`. All three covariances are positive. With

```
rho = t / sqrt(c12*c13*c23),
pi  = [1-rho/sqrt(rho^2+4)]/2,
Delta1 = sqrt(c12*c13 / [c23*pi*(1-pi)]),
b1 = mu1-pi*Delta1,     a1=b1+Delta1,
```

and cyclic versions for views2/3, the parameters are identified. Without semantic orientation there is foreground/background label swapping. Vanishing class mass or vanishing channel gaps destroys nondegeneracy; near-zero quantities make inversion ill-conditioned. Three views being available is not the same as three informative independent channels being available.

Reference labels can orient channels only if their semantic error relation transports. Good reference performance alone does not exclude all channels reversing on a new query appearance. Continuous/categorical views have analogous rank requirements; the binary formulas show the issue without appealing to an unspecified tensor theorem.

A three-binary-view observed joint law has7 degrees of freedom, as does the two-class model's `pi` plus six error rates. It is therefore essentially just identified: a feasible good fit of those three views is not an independent verification of conditional independence. A fourth view provides overidentifying restrictions, or actual-label conditional diagnostics are needed. Related general primary-source framework: [Allman, Matias and Rhodes, identifiability in latent-structure models](https://arxiv.org/abs/0809.5032). The binary moment derivation above is self-contained and does not assume this framework's conditions are satisfied by the encoder.

### Checks, conditioning and common-encoder views

With more than three views, a constant-rate two-class independence model imposes observable constraints. For four distinct indices, `c12*c34=c13*c24`; different nondegenerate triples should give the same prevalence. Invalid reconstructed rates outside `[0,1]` also reject the fitted model. These are useful *falsification* checks, not proofs that the latent class is the desired semantic foreground.

Given actual labels in an independent diagnostic population, check conditional residuals

```
D_jk,y = E[A_j A_k|Y=y] - E[A_j|Y=y] E[A_k|Y=y]
```

and the full conditional joint law against the product law. Pairwise independence alone does not guarantee three-way conditional independence. Heterogeneous background types, object appearance and common encoder/context errors can cause dependence even after conditioning on foreground/background. Conditioning on additional observed nuisance strata may make an independence claim plausible, but changes the model and needs coverage and rank within strata.

The same encoder can, in principle, expose independent noisy channels under a particular generative model. Conversely, different architectures need not have independent errors. Architecture identity is neither a theorem of dependence nor a theorem of independence. Independently sampled augmentations are independent given the *fixed image and augmentation RNG*; this is not the required independence given `Y`, because the image carries common nuisance and systematic error.

For finite-sample moment estimates from `n` justified independent units, the seven raw moments of three binary views each have simultaneous error at most
`e=sqrt(log(14/delta)/(2n))` by Hoeffding/union bound. Covariance error is at most `3e`, and central triple-moment error at most `13e`. A model-mismatch TV allowance can be added to `e` before these transformations. The inverse must propagate these intervals through the displayed equations and abstain if covariance intervals touch0 or parameter/orientation intervals are invalid. Constants in the resulting parameter bound diverge near degeneracy. A plug-in prevalence without its inverse-conditioning interval is not a risk certificate.

## 6. Why consistency/voting is not automatically a posterior

If all views equal the same mask `M`, agreement is1. Truth can be `M` or its complement in the two-world construction, with identical unlabelled agreement. Equivariance can also be exact for either systematically wrong rule. Agreement estimates predictor reproducibility, not semantic correctness.

Under the *identified and correct* independent-channel model, a posterior is available:

```
P(Y=1|A=x) = pi product_j a_j^x_j(1-a_j)^(1-x_j)
             / [pi product_j a_j^x_j(1-a_j)^(1-x_j)
                +(1-pi) product_j b_j^x_j(1-b_j)^(1-x_j)].
```

It uses prevalence, both error rates and the independence assumption; a vote fraction omits them. For example, with three independent channels having `TPR=.95,FPR=.90` and `pi=.01`, unanimous positive outputs have posterior foreground probability only about `.0117`. The unanimous votes carry very little likelihood-ratio evidence.

Multiple deterministic readouts can make existing information computationally accessible and encode different inductive assumptions. They do not manufacture independent label information by being counted several times. A latent agreement factor identified from predictions must still be anchored to semantic foreground; otherwise its “posterior” is a posterior for agreement, not the target class.

## 7. Assumptions that can turn reference evidence into bounded risk

### A. Reference self-evaluation plus transport and valid sampling

A sufficient finite-family theorem requires all of the following:

- a frozen family of `K` candidate functions, with the evaluated reference labels withheld from their support conditioning or with an appropriate uniform-complexity/stability argument;
- `n` independent validation units, or a justified mixing/concentration bound with an effective sample size; thousands of same-image pixels are not automatically independent image-level evidence;
- known/bounded covariate weights `w=dQ_Z/dR_Z <= W` and coverage;
- a bound `eta` on reference-to-query conditional label-law shift;
- if weights are estimated, a bound `epsilon_w` on their integrated error.

For bounded pixel loss, weighted validation gives uniformly over the frozen family

```
|R_query(h)-R_hat_weighted(h)|
    <= W sqrt(log(2K/delta)/(2n)) + eta + epsilon_w.
```

The covariate-shift part can be investigated from unlabelled features. The conditional label shift cannot in general be certified from those marginals. A single reference image permits a useful bound only under explicit within-image sampling and appearance-transfer assumptions. Reference identity matching that includes evaluated pixels' labels in the reference FG/BG sets is not independent self-evaluation.

### B. A fixed-bin posterior calibration model

Partition feature/score space into bins fixed independently of calibration labels. Suppose reference labels are conditionally independent Bernoulli observations within each bin, query/reference label probabilities differ by at most `Delta_b`, and each query bin is covered. With `n_b` independent reference calibration samples,

```
|p_hat_b-p_query,b| <= sqrt(log(2B/delta)/(2n_b)) + Delta_b.
```

An unsupported bin has an uncertainty bound1. Spatial dependence, GT-selected bins, reference-to-query appearance shift and hidden nuisance must be accounted for rather than assumed away. The construction shows a route to calibrated selection from one complete labelled reference under strong but precise assumptions; it does not assert those assumptions hold for today's features.

The bound estimates a bin's conditional mean, not automatically every pixel's posterior. To use it as a pointwise calibration bound, bin membership must be sufficient for the label probability, or a within-bin heterogeneity allowance must be added. Alternatively, bins can be the joint prior-membership states: every Boolean candidate is then constant within a bin, so calibrated *query group means* suffice for its expected TP/union counts. This avoids asserting that an arbitrary score bin provides a full semantic posterior. It still requires reference-to-query group-mean transport and coverage.

### C. Semantic neighbourhood certificates

Another sufficient structural assumption is that opposite semantic labels in the chosen representation are separated by a *known valid* margin `gamma`. A query point within distance `<gamma` of a labelled reference point must have the same label. Covered regions can be certified; uncovered regions remain ambiguous. Empirical separation on one reference is not a lower bound on global semantic separation, so it cannot by itself supply `gamma`.

These are alternative bridges. None requires declaring all DINO representations unusable. Each makes a testable prediction about the regime in which transfer is reliable and the regime in which a bound should become vacuous.

## 8. Turning calibrated uncertainty into complete-mask selection

Let `C_j` be fixed query masks, `p_i` estimated foreground probabilities and `eta_i` the true conditional marginals. Suppose their average absolute calibration error is bounded by `epsilon`. For `N` conditionally independent query labels, define

```
t_hat_j = (1/N) sum_i C_ji p_i,
u_hat_j = (1/N) [sum_i C_ji + sum_i (1-C_ji)p_i].
```

For `K` candidate masks, both normalized realized intersection and union are within
`e=epsilon+sqrt(log(4K/delta)/(2N))` of these estimates simultaneously. A justified dependent-field concentration radius can replace the independent-pixel term. If `u_hat_j-e>0`,

```
|IoU(C_j,T)-t_hat_j/u_hat_j| <= min(1, 2e/(u_hat_j-e)).
```

This follows by bounding numerator/denominator error and using `t_hat_j<=u_hat_j`. Small unions amplify uncertainty. The posterior-count ratio is **not generally the exact expected IoU**: expectation of a ratio differs from a ratio of expectations, and joint label dependence matters. The displayed concentration theorem supplies the missing bridge to the realized mask score.

Uniform score intervals yield a concrete rule: certify a winner only when its lower bound exceeds every competitor's upper bound. Otherwise retain a plausible-winner set. Choosing the highest estimated score has regret at most twice the maximum score radius on that event. This is useful even when it certifies only some recipes/queries; it does not claim certain per-query superiority from unlabelled agreement.

With certified FG/BG pixels and an otherwise unrestricted ambiguous set, complete IoU intervals are also available without pretending ambiguity is probabilistic. Let `f` be known FG count, `t` the mask's known TP, `b` its known FP, `a` predicted ambiguous count and `n` all ambiguous pixels. Subject to the usual nonempty-union convention,

```
IoU_lower = t/(f+b+n),
IoU_upper = (t+a)/(f+b+a).
```

The extrema assign ambiguous predicted pixels and ambiguous unpredicted pixels adversarially. All recipes share the *same* unknown truth. Membership groups can therefore tighten paired bounds by retaining this shared latent assignment, rather than treating each recipe's worst case as a separate independent world. With no certified query labels these intervals can be vacuous, consistently with Section1.

The paper's class-summed I/U macro score needs these bounds applied to each class's pooled intersection/union, followed by averaging. A bound for independent pixel accuracy, one image's IoU or a globally pooled class mixture is not automatically a bound for that macro objective.

## 9. New predictions and the smallest powered follow-up design

Nothing below is run or scheduled during this theory pause.

| Explicit hypothesis | New distinguishing prediction | Minimal future measurement |
|---|---|---|
| Correct known score components, weak inverse separation | Prevalence error/interval width increases as separation or information decreases, even with small component drift | Freeze score operator and histogram/event definitions; seal prevalence predictions on photo-held episodes before GT; then measure separation, coverage and relative/log prevalence error |
| Reference-self component transport fails | Reference templates differ systematically from query components; clipped0 estimates coincide with transport mismatch rather than true absence | Use supplied GT only after predictions are sealed to audit component transport at the predeclared score event/tail; retain reference self-score versus actual cross-image score definitions |
| Three informative independent channels | Nonzero covariance, valid rates, consistent prevalence across triples; labelled conditional residual dependence is small enough for the propagated interval | Fixed three/four views, same episode/pixel alignment, independent-photo uncertainty; check full conditional joint factorization and rank/orientation, not agreement alone |
| Common latent nuisance dominates views | Agreement can remain high while conditional dependence/error increases; more views do not narrow a correctly specified semantic interval as `1/sqrt(K)` | Paired conditional joint-law residuals on previously labelled photo-held data, including foreground and heterogeneous BG; disclose label use |
| A calibrated occupancy estimate can use fact7's tolerance | Held proportion intervals stay inside the useful multiplicative regime; predicted gains exceed a declared complete control with paired uncertainty | One frozen estimator/configuration and the original complete baseline/readout; no post-GT sign flip, bin swap or best-estimator selection |

There is no universal minimal episode count without a separation/effect-size target and dependence/variance estimate. A principled plan specifies the target first:

- For a fixed event with known component gap `|d_B|`, a Hoeffding query proportion interval of half-width `r` requires at most the conservative planning condition `n_eff >= log(2/delta)/(2*d_B^2*r^2)`, before adding template uncertainty/shift. To guarantee a sufficient factor2 regime use `r<=pi/2`; if the transported component errors already exceed `|d_B|r`, additional query pixels cannot repair the bias.
- For conditional-dependence diagnosis, estimate the seven raw moments for each of the two actual labels. Simultaneous raw-moment error is `e=sqrt(log(28/delta)/(2*n_min))`; pairwise residual error is at most `3e`. A predeclared residual of magnitude `d_dep` is guaranteed detectable by an interval excluding0 when `6e<d_dep`, a conservative sufficient condition. Checking pairwise residuals alone is insufficient: retain the triple/full joint law. Effective units must respect photo and spatial dependence.
- For an eventual paired complete-method gain with standard deviation `sigma` across appropriate independent units, the usual approximate80%-power, two-sided5% planning formula is `n_eff≈7.84*sigma^2/MDE^2`. For class-summed macro IoU, use the paired cluster influence/contribution variance for that estimand, not the SD of raw per-episode IoUs. Estimate it from separate development evidence, retain class/connected-photo structure, and freeze the MDE/configuration before confirmation. A nominal token count or an arbitrary20-case smoke cannot supply this power.

These measurements separate an identifiable but ill-conditioned inverse from an incorrect component/independence model. They do not require a new encoder or a new segmentation cue. If templates or view confusion rates are learned using development GT, that supervision must be recorded as an extended setting; it cannot silently become a label-free theorem for a single reference.

## Decision supported by the theory

Keep two claims separate: unrestricted no-assumption selection is impossible in the worst case; constrained, calibrated selection can be identifiable and useful. Facts7/8 make calibrated occupancy a plausible information bottleneck and expose concrete failed assumptions. They do not establish a universal representation limit. The next scientific question is whether the required transport, separation, conditional-independence and semantic-orientation bounds are sufficiently strong for the desired complete-mask gain. No present experiment result is supplied by this theoretical note.
