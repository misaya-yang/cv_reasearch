# Nested-regime theorem: canonical joint edits cannot beat the best available cut

Theory-only final audit, 2026-10-06 UTC. Owner: `/root/occupancy_gain_geometry`.
No data, evaluation, numerical simulation, model call, or remote operation was used.
The counterexamples below are constructed probability/count examples, not dataset results.

## Core theorem and scope

**Conditional theorem:** for one class, suppose all available complete candidates,
including the origin, are upper level sets of the same scalar score `g`. Suppose the
population posterior `eta(g)=P(Y=1|g)` has a nondecreasing version. Optimize the ratio of
population expected intersection and union, with no inference-cost constraint. Then every
canonical fixed-origin union of additions and deletions is weakly dominated by at least
one available single complete threshold. With empty operator selections allowed, the best
canonical joint score equals the best available single-complete score.

This does **not** follow merely from nested masks. Monotone true posterior, the exact
objective, availability of the origin/endpoints, and canonical fixed-origin edits matter.
It does not assert graph sufficiency, empirical rank monotonicity, or equality on a random
finite GT cohort. It also does not generally extend to class-macro optimization with one
shared global cut.

## 1. Definitions and canonical shape

Choose one class's population pixel with score `g` and label `Y`. Let
`eta(g)=P(Y=1|g)` and `pi=E[eta]>0`. Define the available finite threshold bank and origin

\[
M_t=\{g\ge t\},\qquad t\in T,\qquad O=M_{t_O},\quad t_O\in T.
\tag{1}
\]

In plain English, every method chooses an operating point on exactly the same ordering;
the origin itself is an available complete candidate. All tied scores are treated together.

For selected producers `a` and `d`, their canonical edits are `Ma minus O` and `O minus Md`.
Unions of many selected additions and many selected deletions reduce to two extreme cuts:

\[
\alpha=\min(\{t_O\}\cup T_A),\qquad
\beta=\max(\{t_O\}\cup T_D),\qquad
C=\{\alpha\le g<t_O\}\cup\{g\ge\beta\},\qquad
\alpha\le t_O\le\beta.
\tag{2}
\]

In plain English, selected additions supply the lowest available cut, and selected deletions
supply the highest. Their final mask may include a lower band, omit a higher middle band,
and retain an upper tail. This inversion is possible because deletions are only inside
the fixed origin; they do not remove newly added pixels. Both `alpha` and `beta` are cuts
of available complete masks, possibly the origin when one operator union is empty.

Write the lower band, omitted middle band, and upper tail as `L`, `B`, and `H`, respectively.
Then `C=L union H`, `Malpha=L union B union H`, and `Mbeta=H`.

The theorem's population objective is

\[
J(S)=\frac{I(S)}{U(S)},\qquad
I(S)=\mathbb E[\eta(g)\mathbf1_S],\qquad
U(S)=\pi+\mathbb E[(1-\eta(g))\mathbf1_S].
\tag{3}
\]

In plain English, this is expected-count IoU for the class. It is not, in general, the
expectation of a finite cohort's random IoU ratio. Since `pi>0`, every union denominator
is positive, including the empty mask's.

## 2. Proof by the composite's fractional residual

Let `q=J(C)` and define the score-conditional profit and residual

\[
w_q(g)=\eta(g)-q[1-\eta(g)]=(1+q)\eta(g)-q,\qquad
R_q(S)=I(S)-qU(S)=\mathbb E[w_q(g)\mathbf1_S]-q\pi.
\tag{4}
\]

In plain English, profit prices false positives at the composite's own IoU. It is
nondecreasing in `g` because the true posterior is nondecreasing. The composite has zero
residual, and any positive-union mask with nonnegative residual has IoU at least `q`.

Let `WL=E[wq 1L]` and `WB=E[wq 1B]`. There are two exhaustive cases.

**Case 1: `WL <= 0`.** Removing the lower band gives the available upper-threshold mask:

\[
R_q(M_\beta)=R_q(C)-W_L=-W_L\ge0,
\qquad J(M_\beta)\ge J(C).
\tag{5}
\]

In plain English, the lower band's net value is nonpositive, so its available deletion
endpoint is at least as good as the composite. If `WL<0`, the dominance is strict.

**Case 2: `WL > 0`.** The lower band contains positive-profit score values of positive
probability. Every score in the omitted middle band is higher than those values, so
monotonicity implies nonnegative profit throughout that band. Therefore

\[
W_B\ge0,\qquad
R_q(M_\alpha)=R_q(C)+W_B=W_B\ge0,
\qquad J(M_\alpha)\ge J(C).
\tag{6}
\]

In plain English, a valuable lower band makes omission of a higher band unnecessary;
the available addition endpoint is at least as good. If the middle band has positive
probability, its profit is strictly positive and the dominance is strict.

Together (5)-(6) prove the claim for each composite using one of its **existing endpoints**;
the proof does not require searching for a new threshold or assuming the global optimal
cut is in the bank.

Every bank threshold below the origin can be realized with its addition alone, every bank
threshold above the origin with its deletion alone, and the origin with no edits. Hence,
when empty operator selections are allowed,

\[
\max_{C\in\mathcal C_{\rm canonical}}J(C)=\max_{t\in T}J(M_t).
\tag{7}
\]

In plain English, canonical joint search has no population score advantage over the best
available complete method in this nested, monotone-posterior regime.

## 3. Ties, empty sets, and endpoint conditions

- `q>0` is not needed for weak dominance, although it gives the usual posterior threshold
  `q/(1+q)`. If `q=0`, all mask IoUs are nonnegative and the residual proof still applies.
- If `L` is empty or has zero probability, `C=Mbeta` almost surely. If the omitted middle
  band is empty or has zero probability, `C=Malpha` almost surely. These are exact single
  candidates, including `alpha=tO`, `beta=tO`, or all cuts equal.
- Score atoms are allowed. Using one common `g>=t` convention makes the half-open bands
  disjoint and ensures no tied-score block is split. If a constructor splits ties using
  another observable variable, it is not a threshold-only family of this scalar `g`.
- If `pi=0`, IoU for empty union needs a specified convention; (3)-(7) explicitly assume
  positive target mass. `pi=1` is permitted and presents no denominator problem.
- The endpoint complete candidates must be available under the comparison. Excluding the
  origin can remove a dominating endpoint when an operator family contributes no effective
  edit. Origin availability is an assumption, not an extra method created by the proof.
- Sequential redefinition of edits, spatial gating, noncanonical deletion outside the
  origin, additional score-dependent fields, or arbitrary complements do not necessarily
  have shape (2). The theorem does not cover them without a separate proof.

## 4. Counterexamples at the boundaries

### 4.1 Nested candidates alone are insufficient

For three equally weighted score levels `g1<g2<g3`, take deterministic truth at `g1` and
`g3`, and background at `g2`. Available complete masks are all three levels, the upper
two levels (origin), and the highest level. Canonical addition plus deletion selects
`{g1,g3}` and is perfect; the three singles have IoUs `2/3`, `1/3`, and `1/2`.

In plain English, these complete masks are nested, but `eta=(1,0,1)` is not monotone.
The omitted middle band is harmful even when the included lower band is valuable.
Thus "all candidates differ only by cuts" does not by itself prove no joint advantage.

### 4.2 Exact expected finite-cohort IoU is a different objective

Consider one random three-pixel GT mask on `g1<g2<g3`:

\[
P(Y=\{g_1,g_3\})=2/5,\quad
P(Y=\{g_2\})=9/20,\quad
P(Y=\{g_3\})=3/20.
\tag{8}
\]

In plain English, the marginal posteriors are strictly ordered: `eta1=2/5`, `eta2=9/20`,
and `eta3=11/20`. No GT mask is empty, so no empty-union convention is involved.

Nevertheless the canonical two-band mask has higher expected *random-ratio* IoU than
every available upper threshold:

\[
\begin{aligned}
\mathbb E[J(\{g_1,g_3\},Y)]&=19/40,\\
\mathbb E[J(\{g_3\},Y)]&=7/20,\\
\mathbb E[J(\{g_2,g_3\},Y)]&=13/30,\\
\mathbb E[J(\{g_1,g_2,g_3\},Y)]&=7/15.
\end{aligned}
\tag{9}
\]

In plain English, truth-size correlations change the expected random denominator, so
marginal posterior monotonicity alone does not rescue the theorem for this different
objective. These fractions follow directly by substituting the three possible GT masks;
they are a hand-constructed counterexample, not a numerical experiment.
The expected-count theorem remains consistent: its all-pixel upper threshold has
expected-count IoU `7/15`, greater than the composite's expected-count IoU `19/49`.

### 4.3 Class-macro score with one shared cut can favor the composite

Use the same four ordered score levels `g0<g1<g2<g3`; all candidates omit `g0`.
In class 1, one background pixel has score `g0`, and two foreground pixels have scores
`g1,g3`. In class 2, background pixels have scores `g0,g2`, and a foreground pixel has
score `g3`. Both class-specific posteriors have nondecreasing versions on the score line;
levels absent in a class can be filled monotonically.

The shared-cut candidates select all of `g1,g2,g3`, only `g2,g3` (origin), or only `g3`.
Their macro IoUs are `3/4`, `1/2`, and `3/4`. The common canonical two-band mask selects
`g1,g3` and gives macro IoU one. In class 1 the lower complete threshold is best; in class 2
the upper complete threshold is best. No single shared-cut endpoint achieves both.

In plain English, class-dependent tradeoffs defeat the attempt to choose one globally
dominating endpoint. The single-class proof is valid separately, but its winning endpoint
need not be the same in every class.

## 5. What would suffice for a macro extension

For the composite's class scores `qc`, define `wc(g)=eta_c(g)-qc(1-eta_c(g))`, and use
class count measures in `WL,c`, `WB,c`. The exact macro endpoint differences are

\[
V(M_\beta)-V(C)=-\frac1C\sum_c\frac{W_{L,c}}{U_c(M_\beta)},\qquad
V(M_\alpha)-V(C)=\frac1C\sum_c\frac{W_{B,c}}{U_c(M_\alpha)}.
\tag{10}
\]

In plain English, there are distinct class-specific prices and endpoint union weights.
A positive aggregate lower-band value does not automatically imply positive aggregate
omitted-band value with the other endpoint's weights.

**Conditional sufficient extension:** suppose all class profit functions for this composite
share one sign-crossing location on `g`, and are nondecreasing. A stronger sufficient
condition is `wc(g)=ac*w(g)` with `ac>0` and one nondecreasing `w`. If no lower-band score
has positive profit in any class, removing that band weakly improves every class. If one
does, the common crossing implies nonnegative omitted-band profit in every class, so filling
the middle band weakly improves every class. One shared endpoint then dominates.

This alignment must hold for every composite to obtain a whole-family macro equality.
It is stronger than posterior monotonicity separately in each class and is not implied
by graph agreement, nested candidate masks, or a fitted global calibration rule.

## Interpretation for the final research shape

The theorem supplies one rigorous regime in which joint search is unnecessary: the
canonical family only revisits operating points on an already useful common ordering,
and true conditional profit increases along that ordering. A joint gain can arise when
this regime fails through complementary rankings, spatial/member structures, distinct
class tradeoffs, or the exact finite-benchmark objective. Absence of a measured joint gain
does not establish which condition holds. These alternatives locate the question; they
do not introduce a new method or authorize an experiment.
