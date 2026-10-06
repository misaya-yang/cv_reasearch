# Query hierarchy: exact joint capacity and what it establishes

This is a mathematical/code contribution to the supplied query-only hierarchy
direction. The user's600-case run is reported ongoing; Root did not inspect,
modify or duplicate it. No real hierarchy score or reference selector is validated
here. Two GPT-6.1-sol/max agents independently audited the proof and its scope.

## 1. Candidate family and exact metric

Let each query supply a tree of region masks. For episode e, a prediction is the
union of at most K eligible tree nodes, including the option to select none.
Assume actual evaluated masks are laminar: an ancestor contains its descendants,
and incomparable nodes are disjoint. Selecting an ancestor makes its selected
descendants redundant, so some optimum is an antichain.

For node v let t_v be its GT intersection and f_v its false-positive area; let
g_e be the entire GT foreground area, including foreground outside all candidates.
For disjoint selected nodes, intersection and false-positive area are additive.
For a class c the exact metric is

    J_c = sum_e sum_selected t_v / (sum_e g_e + sum_e sum_selected f_v).

In plain terms, select regions jointly, but add all images' pixel counts before
dividing. Maximizing every image's IoU separately is a different objective.

## 2. Exact fractional optimization on the tree

At fixed ratio lambda, assign each eligible node weight

    w_v = t_v - lambda * f_v.

This rewards recovered foreground and charges included background at the current
class IoU. The constant -lambda*sum(g_e) is added only for the class residual.

Define D_v(k) as the largest total weight of an antichain of exactly k eligible
nodes under v. D_v(0)=0. For k>0, choose either v itself (only when k=1 and v is
eligible), or distribute k among its children and add their best states. Invalid
states have value minus infinity. Maximizing D_root(k),0<=k<=K solves each episode.
Repeated child convolutions take O(N*K^2) arithmetic operations per ratio for a
tree of N nodes. There is no need to enumerate all node subsets.

Let F_c(lambda) be the sum of episode optimum weights minus lambda*sum(g_e).
For positive class GT area, its unique zero is the maximum attainable class IoU:
below that optimum some prediction has positive residual; above it every prediction
has negative residual. Independent per-episode caps allow separate episode DPs;
independent classes then maximize macro mIoU separately.

The implementation starts at0 and repeatedly sets lambda to the selected class
intersection/union. Positive residual strictly increases the attainable ratio;
zero residual certifies exact optimality. The finite family guarantees eventual
termination, not a small iteration count. The256-iteration guard raises instead
of returning an uncertified result. Integer counts use lambda=a/b and weights
b*t_v-a*f_v, so ties and residuals are exact rather than tolerance-dependent.

## 3. A concrete metric counterexample

Two images of one class both have GT area10. The first's useful candidate X has
TP1,FP0. The second offers A(TP5,FP0) inside B(TP7,FP5); all other candidates can
have sufficient FP to be inferior. Per-image optimization picks X and A because
0.50>7/15. Their class IoU is6/20=0.30. Joint class optimization picks X and B,
giving8/25=0.32. The code checks this in complete partition trees.

## 4. Strict capacity/selection decomposition

On identical data, resolution and output mapping let O_K be the exact class-mIoU
oracle of this tree family and S_K the deployable reference-conditioned selection.

    100 - S_K = (100 - O_K) + (O_K - S_K).

The first term is candidate approximation loss; the second is selection loss.
For complete baseline B,

    S_K - B = (O_K - B) - (O_K - S_K).

Oracle headroom is available capacity that must pay for selection mistakes.
It is not a prediction of deployable gain. Preserving all old candidate masks and
using an 'at most K' budget makes O_1<=O_2<=O_3<=O_unlimited strictly monotone.
This does not imply that practical selection improves with more candidates.

## 5. Decisive readout for the supplied600 run

- Compare O_1,O_2,O_3 with the same cases' full FoRIS baseline and score-threshold
  oracle; comparing a tree oracle only with a deployable62–63 is insufficient
  to establish superior candidate capacity.
- O_unlimited uses the same tree and counts, with no new encoder call. Low O_3
  and high O_unlimited indicate a node-budget limitation. Low O_unlimited limits
  this tree/leaf partition, not all candidate constructions from frozen DINO.
- Keep node IDs, eligibility, parent/children, full-resolution TP/FP and entire GT
  areas. Require actual mask containment/disjointness; topology and count inequalities
  alone cannot prove them.
- Reference selection, including its choice of hierarchy scale and number of
  disconnected target regions, remains unverified until complete masks are scored.
  A one-class semantic mask can contain more than three instances/components.
- A nonlinear finalizer applied after union changes the candidate family. The
  additive oracle is for direct unions at the evaluated resolution; it must not
  be claimed as the exact oracle of union-plus-CRF without a separate derivation.
- Report paired uncertainty against matched controls. If an interval conditions on
  frozen GT-selected nodes, label it conditional; it is not fresh confirmation.

## Implementation and verification

`src/ics/methods/hierarchy_oracle.py` accepts node counts and eligible-node topology,
returns exact selected antichains, class I/U and macro IoU, and supports K=None for
no cardinality cap. Children may leave residual parent pixels; sibling regions
must still be disjoint. A nonselectable synthetic root supports disjoint forests.
Classes with zero total GT area are rejected until an explicit metric convention
is supplied. Shared cross-image budgets or cross-class constraints are not covered.

`python3 scripts/verify_hierarchy_oracle.py` compared400 exact results against full
enumeration of small trees with varying eligible nodes and K=0/1/2/3/unlimited.
The metric counterexample, macro aggregation and malformed-tree rejection pass.
These are mathematical implementation checks, not model-performance evidence.
No real episodes, remote computation or GPU calls were used.
