# External theory comparison: Boolean repair, graph bias and information value

2026-10-06 UTC. Owner: `/root/graph_estimator_theory`.

Bounded task: compare the user's supplied
[external manuscript](</Users/yang/Downloads/单参考分割联合编辑理论与可检验预言.md>)
with [our graph estimator note](graph_estimator_theory.md),
[five-question synthesis](five_questions.md), and the relevant actual
origin-local recipe code. Focus is external sections 1, 3.3 and 3.5.
This is source reading and independent symbolic derivation only. No experiment,
GPU/remote operation, CPU data evaluation, sweep, training, download or execution
of the external document's recommendations was performed.

The external manuscript's pinned-code identity, reported expert/30/200 results
and claimed synthetic checks are source claims. They were not independently
reproduced here. Its execution prose grants no permission for execution.

## Findings at a glance

| Topic | Assessment | Relation to our current account |
|---|---|---|
| General deletions can repair newly added FP | Exact and useful | More explicit capacity distinction and constructive example; consistent with our scoped origin-local bank, not proof its implementation is wrong |
| Net edits are disjoint only after general composition | Exact | Five-question synthesis already uses delete-priority net edits; the actual origin-local solver has the narrower static family |
| Weighted signed coverage and cross-role coupling | Correct with scope distinction | Coverage-separability statements must be restricted to origin-local or a fixed opposite role; general deletion adds `a(1-d)` coupling |
| Graph MAP, whitening, invertibility, maximum principle, anchor flow | Correct; largely equivalent | Already derived in the graph note; external examples sharpen presentation rather than establish a different estimator |
| Exact graph error cross term | Correct and necessary | Our zero-mean risk shorthand needs the explicitly conditional assumption `E[epsilon | H,L,t]=0`; marginal zero mean is insufficient for data-dependent graph/anchors |
| Full-mask TV/conditional-MI value upper bound | Correct, conditional on same feasible action family and normalized bounded utility | A substantive new quantitative bound beyond our qualitative sufficiency/value statements |
| Actual-program Bayes-regret decomposition | Correct | Equivalent extension of our maximum-gain information-value plus current-decision-gap identity; includes the new program's remaining regret |
| Equal pixel marginals, positive complete-IoU value | Correct; hand derivation below | New constructive witness for our existing warning that marginal sufficiency does not establish correlated-mask utility sufficiency |
| Positive-part threshold-crossing identity | Correct for fixed external ratio price and linear expected-count residual | New explicit threshold criterion; does not replace finite expected-IoU or constrained-budget conditions |

No mathematical error was found in the reviewed TV/MI bound, regret identity,
three-pixel counterexample or positive-part identity. The material corrections
are **scope and conditioning**, especially for our abbreviated risk formula.
The claims do not supply measured conditional MI, a calibrated RCG posterior,
or a deployable winner on the real segmentation distribution.

## 1. General Boolean repair versus the implemented origin-local bank

### 1.1 General delete-priority composition: exact

For arbitrary selected unions of addition and deletion proposals,

\[
 M=(O\cup A)\setminus D,\qquad
 A_{\rm net}=A\setminus(O\cup D),\qquad
 D_{\rm net}=O\cap D.
\]

The final net domains are disjoint, but the net addition depends on deletion.
An outside-origin deletion can veto an outside-origin addition. Therefore
truncating every deletion proposal to `O` *before* choosing the composition
changes the attainable family in general. Disjoint final net edits are a
counting normalization; independently disjoint proposal roles are an extra
restriction.

External section 1.3's nine-pixel example is correct. GT has four pixels,
`O` selects two, `A` adds the other two and five FP, and `D` removes one original
TP and those five FP. The scores are respectively

\[
 J(O)=\tfrac12,\quad J(O\cup A)=\tfrac49,\quad
 J(O\setminus D)=\tfrac14,\quad
 J((O\cup A)\setminus D)=\tfrac34.
\]

Both singleton moves are harmful; their combination is useful because deletion
repairs the FP introduced by addition. If the same `D` is clipped to `O`, the
joint score becomes `1/3`, losing that repair. This is a representation and
interaction counterexample, not a new proposed segmentation rule or evidence
that actual proposals realize its semantic pattern.

### 1.2 The actual static origin-local family: exact

For a complete producer bank `S_i`, the implemented recipe uses
`A_i=S_i\O`, `D_i=O\S_i`. Its output is

\[
 M_{\rm local}
 =(O\cap\bigcap_{j\in K_D}S_j)
   \cup(O^c\cap\bigcup_{i\in K_A}S_i).
\]

The current `scripts/run_exact_family_selection.py` contains that same
restriction in `recipe_lut` and `final_mask_one`: its deletion update removes
`origin & ~producer`, and never deletes the pixels added outside `origin`.
This was read, not executed. The solver correctly searches its declared
family. A certificate for that family must not be renamed a certificate for
all general Boolean or state-dependent editing.

Our main note already scopes canonical domains and separately names
state-dependent operators. The five-question synthesis now also defines general
delete-priority net edits. Thus the external point is a valuable explicit
distinction, not a contradiction of those qualified statements. It would
contradict an unqualified statement that *all* joint editing separates add/delete
roles after proposals are defined.

### 1.3 Multiple origins recover some Boolean masks, not arbitrary Boolean logic

Changing the origin to an included producer can recover some cross-role-looking
outputs: taking `O=S1` and the deletion producer `S2` gives `S1 & S2`.
Therefore a fixed-origin example alone does not prove a gap for the union of
all allowed origins.

A stronger representation example is majority of three bits. For any chosen
origin bit `s1`, majority on the `s1=0` slice is `s2 & s3`. The local family on
that slice can only choose `0`, `s2`, `s3` or `s2 | s3`. By symmetry no other
origin repairs this mismatch over all signatures. Majority preserves unanimous
zero and unanimous one, yet is not an origin-local recipe for any of the three
origins. This proves a strict distinction from arbitrary unanimity-preserving
Boolean functions when all needed signatures are possible. It is not a method
proposal, and gives no claim that majority improves segmentation.

Conversely, arbitrary-`D` static composition with a fixed finite proposal
vocabulary is not automatically *all* Boolean functions either. Vocabulary,
roles, available origins, dependency cost and allowed order determine the
actual family. A newly state-dependent producer changes the family as well as
possibly the computation; its result cannot be attributed only to a better
solver of the unchanged bank.

The membership-atom I/U compression remains exact for any declared fixed
Boolean recipe. Enlarging a recipe family changes which functions of atoms
are permitted; it does not invalidate the count representation. Its GT counts
remain development/capacity information, not an inference-time semantic value
predictor.

### 1.4 Fixed-price cross-role coupling: exact

Write node membership as `o,a,d`, and foreground/background signed weight as
`w_q=1_T-q*1_background`. General composition has residual

\[
 I(M)-qU(M)=I(O)-qU(O)
 +\sum_{x\notin O} w_q(x)a_x(1-d_x)
 -\sum_{x\in O} w_q(x)d_x.
\]

Pricing the ratio does not remove the outside-origin product. Its mixed
add/delete difference at an individual outside pixel is `-w_q`: deletion of
an added FP creates a positive repair term, while deletion of an added TP
removes a benefit. The actual sign still depends on semantic correctness.

In origin-local editing the outside deletion indicator is identically zero,
so fixed-price role separation is valid absent shared cost constraints. General
signed coverage need not be modular or monotone/submodular. The synthesis's
short coverage statement should retain this qualification if it follows its
general delete-priority definition. Even in the local subfamily, within-role
overlap and the final class-specific denominators preserve the interactions
already derived in our main note.

## 2. The graph cross term is a genuine assumption check

External section 3.3 sets `S=(H+lambda L)^(-1)H`, `y=t+epsilon`. Exactly,

\[
 Sy-t=S\epsilon+(S-I)t,\qquad
 \|Sy-t\|^2=\|S\epsilon\|^2+\|(S-I)t\|^2
              +2\langle S\epsilon,(S-I)t\rangle.
\]

The first term is surviving unary error; the second is smoothing bias; the
third says whether those two errors reinforce or cancel each other. It cannot
be dropped from an identity. A sufficient assumption for its mean to vanish
is `E[epsilon | H,L,t]=0`; fixing `H,L,t` outside the noise-generating process
and taking zero-mean noise is a special case. Marginal `E[epsilon]=0` does not
suffice when the operator depends on the same noisy observations.

Our section 4.4's spectral risk expression is correct under that fixed-operator
or conditional-zero-mean assumption. Its short phrase “noise has zero mean”
should be read more narrowly, or revised to make the conditioning explicit.
Actual RCG anchors depend on the host score, and its graph depends on query
features, so independence or conditional zero mean is not established.

For nonuniform anchors, the same exact expansion may be written in the
`H`-weighted norm. Whitened operator
`R=H^(1/2) S H^(-1/2)=(I+lambda B)^(-1)` is symmetric, unlike `S` in the ordinary
Euclidean coordinates. Then the modal bias/variance formula requires zero-mean
whitened noise conditional on the realized graph/anchors and ideal target.
No coordinate-independence assumption is needed for the *total squared norm*
once conditional zero mean holds: only each mode's variance enters that sum.

The external guide-increment equation is likewise exact for fixed graph/anchors:

\[
 \|e+Sd\|^2-\|e\|^2=2\langle e,Sd\rangle+\|Sd\|^2.
\]

A nonzero increment improves that squared-error measure precisely when its
negative alignment with the current error outweighs its own squared size.
This is stronger than saying the guide merely has a different frequency.
It is equivalent to our fixed-unary response identity `Delta f=P_lambda*d`,
now with the missing risk comparison made explicit. It does not establish
IoU gain, semantic correctness, or the actual conditional-noise premise.

The external historical 600 stage numbers are claimed source observations in
their reported 1024/pre-CRF setting. They are consistent with interaction,
but cannot identify this cross term from aggregate IoU. Its distinction between
combined-minus-graph uncertainty and full factorial uncertainty is correct;
our retained verified factorial record has its own separately stated CI.
No interval was recomputed or transferred between versions here.

## 3. TV/MI bound: a correct new quantitative upper bound

Let retained old information be `X`, new evidence `E`, and `Y` the whole truth
vector relevant to the utility. For the class-summed benchmark this can be the
whole evaluation-batch mask vector, not just one random pixel. Fix the same
finite feasible action family `F(X)` and `u(a,Y)` in `[0,1]`. Define

\[
 p=P(Y\mid X),\quad q=P(Y\mid X,E),\quad
 V_0=\mathbb E\max_{a\in F(X)}\mathbb E_p u(a,Y),\quad
 V_1=\mathbb E\max_{a\in F(X)}\mathbb E_q u(a,Y).
\]

Ignoring the evidence is allowed, so `V1>=V0` after averaging. For any fixed
action, its expectation changes by at most `TV(q,p)` because the utility span
is at most one. Comparing the maxima gives the external bound:

\[
 0\le V_1-V_0\le\mathbb E\operatorname{TV}(q,p)
 \le\sqrt{I(Y;E\mid X)/2}.
\]

The second inequality is Pinsker with natural logarithms plus Jensen. Use the
convention `TV=one-half L1`. Percentage-point bounds multiply by 100; class
macro's normalization already gives span one, so no extra factor of the number
of classes is needed. The bound can also be capped by one. No pixel
independence is used.

This is a new quantitative addition to our qualitative Bayes-value and
conditional-sufficiency arguments. It proves an **upper** bound: positive MI
has no universal positive utility-gain lower bound. Full-mask conditional
independence makes the upper bound zero. A common conditional maximizing
action across evidence outcomes is a weaker decision-specific zero-value
condition for the finite family.

Its limitations matter:

- Per-pixel MI, AUC or unconditional field MI cannot be substituted for the
  stated *conditional full-mask* MI.
- A batch-level MI may be large, making the bound vacuous; the theorem does
  not show actual RCG residual MI is small.
- The action family and resource budget must remain the same. Free advice or
  evidence computed with extra views can change implementable program classes
  even when a numerical runtime budget is written as the same number.
- The conditional distributions are unknown on this project. No MI or TV was
  estimated by this audit; these formulas do not supply a numeric available-
  gain ceiling from the reported segmentation scores.

### Actual-program regret identity: equivalent and useful

Let the old and new programs have expected utilities `J0,J1`, and respective
regrets `rho0=V0-J0`, `rho1=V1-J1`. Then exactly,

\[
 J_1-J_0=(V_1-V_0)+\rho_0-\rho_1
 \le\sqrt{I(Y;E\mid X)/2}+\rho_0.
\]

Our note previously separated maximum possible gain into information value
plus the current decision gap. This is the same decomposition, now retaining
the new implementation's unused value. An actually harmful auxiliary may have
increased regret; a deterministic derived cue may improve computation and
reduce regret despite zero added information.

If `E=f(X)` for the *full raw allowed input*, its conditional MI is zero by
determinism. This says nothing about sufficiency of a compressed RCG field.
For zero Bayes information value to become a zero gain statement about a
restricted program family, the old programs must be able to simulate the
enriched advice/decision at the same genuine cost. Acquisition/advice-generation
cost must be included. The external manuscript states this qualification
correctly; it should remain in any synthesis.

## 4. Equal pixel marginals can hide positive finite-IoU value

The external three-pixel example is correct. `E0,E1` are equally probable, with

\[
 P_0(\{b\})=1/6,\ P_0(\{a,b\})=1/3,\ P_0(\{c\})=1/2;
 \qquad
 P_1(\{b\})=1/2,\ P_1(\{c\})=1/6,\ P_1(\{a,c\})=1/3.
\]

In both contexts, the foreground marginal vector is `(1/3,1/2,1/2)`. Thus
each individual pixel has zero MI with `E`. The joint foreground patterns
differ. The following fractions were derived symbolically by summing the
three stated outcomes; no enumeration script or experiment was run:

| Predicted mask | Expected IoU given E0 | Given E1 | Without E |
|---|---:|---:|---:|
| empty | 0 | 0 | 0 |
| a | 1/6 | 1/6 | 1/6 |
| b | 1/3 | 1/2 | 5/12 |
| c | 1/2 | 1/3 | 5/12 |
| a,b | 5/12 | 13/36 | 7/18 |
| a,c | 13/36 | 5/12 | 7/18 |
| b,c | 4/9 | 4/9 | 4/9 |
| a,b,c | 4/9 | 4/9 | 4/9 |

The old optimum is `4/9`. With evidence, choose `c` in `E0` and `b` in `E1`,
giving `1/2` and gain `1/18`. Every GT is nonempty. This is one class with one
finite mask per outcome, so it also refutes a universal finite class-IoU claim.
It does not by itself quantify the effect after large class-count aggregation.

Our note already warned that posterior marginals need not identify expected
IoU under correlated masks. This is a stronger constructive witness: *every*
pixel marginal stays the same across evidence outcomes, yet a complete-mask
decision improves. It refutes using zero pixel MI or an unchanged pixel
posterior alone as a proof of full-mask utility sufficiency. It does not
contradict the TV/MI theorem, because full-mask MI is positive.

## 5. Positive-part threshold crossing: correct, with a different objective

For a fixed external ratio price `q>=0`, an unrestricted pixel-selection
residual has foreground coefficient `eta-q*(1-eta)`. Its optimum includes
the pixel when `eta>tau=q/(1+q)`. Let refined posterior be `H=eta_(Z,E)` and
old posterior `mu=E[H|Z]=eta_Z`. Then the gain from refining this *linear
expected-count residual* is

\[
 D_q(Z)=(1+q)\{\mathbb E[(H-\tau)_+\mid Z]-(\mu-\tau)_+\}
 =(1+q)\min\{\mathbb E[(H-\tau)_+\mid Z],
                   \mathbb E[(\tau-H)_+\mid Z]\}.
\]

Write the two nonnegative expectations as `a,b`. Since `mu-tau=a-b`, the
Jensen gap is `a-(a-b)_+=min(a,b)`. Strict gain occurs exactly when the
refined posterior has strictly positive conditional probability on *both*
sides of the decision cut. Merely changing calibration, changing probabilities
within one side, or touching the cut is insufficient.

This is a new explicit expression for our general “optimal action must change”
condition. At the old population ratio optimum, the old maximal residual is
zero. A positive refined residual at that price implies a strictly higher
attainable population expected-count ratio, subject to positive denominators
and the same unrestricted action family. For class-summed population ratios,
the prices are class-specific; shared-recipe or cost constraints can prevent
the coordinatewise positive-part maximizers from being jointly feasible.

It is not a finite expected-IoU theorem. In section 4's example all refined
pixel posteriors equal the old ones, so every such positive-part gap is zero,
but the finite-IoU gain is `1/18`. Nor is it a fixed-count theorem: fixed-count
actions compete through ordering across the selected/unselected budget cut,
as in our top-k results. The external manuscript makes both distinctions.

The manuscript's simpler two-pixel objective-reversal example is also correct:
GT is `{a}` with probability `11/20` and `{a,b}` with probability `9/20`.
Changing prediction from `{a}` to `{a,b}` lowers finite expected IoU from
`31/40` to `29/40`, while raising the expected-count ratio from `20/29` to
`29/40`. That reinforces why an expectation or threshold theorem must retain
its declared objective.

## 6. What should be merged, retained or qualified

**Merge as quantitative/constructive additions:** the full-mask TV/MI upper
bound, actual-program regret identity, the equal-marginals positive-IoU
counterexample, and the exact positive-part gap. Add the cross-role repair
example to make clear why general net normalization does not authorize clipping
deletion proposals before composition.

**Retain as equivalent existing results:** anchored weighted least squares,
intrinsic GMRF caveat, whitening, finite invertibility, maximum principle,
component-flow explanation, posterior-versus-decision sufficiency, same-encoder
determinism versus compressed-state information, and the requirement to report
the real shared dependency cost.

**Qualify or amend our shorthand:** conditional zero mean for the spectral
risk formula; role-separation scope for coverage arguments; exact-family
certificates are for origin-local recipes and permitted origins, not arbitrary
Boolean edits. Keep expected true-count ranking, population expected-count
ratios and finite expected class-IoU separate. A gain upper bound needs both
conditional information and the old program's regret; it cannot prove actual
auxiliary redundancy from a failed rule alone.

**Report external data only as claims in their own scope:** its expert result
is reported on fixed/exposed 30, with improved source-domain fits but a target
complete score of 59.010955 and reported −1.348486 pp [−3.447734,0.885807]
versus H; the interval crosses zero. Its 32-producer/action-budget-four
exact-versus-greedy result concerns that stated static origin-local 200/30/170
panel, and its membership/oracle figures are GT capacity diagnostics. None was
verified, rerun, transported to public4000, or treated as fresh confirmation
by this audit. Source fitting, target semantic transport, attainable mask
capacity and actual deployment gain remain different claims.

The theory-first conclusion is therefore more precise than “RCG absorbs
everything”: exclude exact output/order redundancies; otherwise ask whether
new evidence changes an attainable utility-optimal decision, or merely helps
the current nonoptimal implementation. The external additions provide sharper
criteria and a conditional upper bound, while leaving the actual conditional
semantic law and complete-method superiority unverified. No new inference
method or experiment is proposed by this comparison.
