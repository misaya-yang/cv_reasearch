# Exact finite add/delete composition: solver and limits

This is a paper-only derivation. No outputs, masks, GT, scripts, remote machine, or GPU were inspected or executed. The candidate bank, objective, and inference cost model are assumed fixed before optimization. The statements below distinguish theorems, conditional predictions, and conjectures; none establishes a new measured method result.

## 1. The family and deletion priority

Let \(P_0\) be an origin mask, \(A_i\) candidate addition masks, and \(B_j\) candidate deletion masks. For selected index sets \(S,T\), define

\[
A_S=\bigcup_{i\in S}A_i,\quad B_T=\bigcup_{j\in T}B_j,\quad
C(S,T)=(P_0\cup A_S)\setminus B_T.
\]

At each pixel, \(c=(p_0\lor a_S)\land\neg b_T\). Thus deletion wins when both actions select the same pixel. Swapping the order to \((P_0\setminus B_T)\cup A_S\) is a different family.

Additions can always be clipped to \(P_0^c\): pixels already in the origin are redundant before deletion. Deletions cannot generally be clipped to \(P_0\), because a deletion outside the origin may cancel a newly added pixel. For example, with \(P_0=\varnothing\) and \(A_1=B_1=\{x\}\), deletion priority returns the empty mask; clipping \(B_1\) to the origin instead returns \(\{x\}\).

Writing \(a=A_S\setminus P_0\), \(b=B_T\cap P_0\), and using GT \(G\), the exact changes are

\[
I=I_0+|a\cap G|-|b\cap G|-|a\cap B_T\cap G|,
\]
\[
U=U_0+|a\cap G^c|-|b\cap G^c|-|a\cap B_T\cap G^c|.
\]

The final terms are genuine cross-action conflicts: deletion cancels an added TP or suppresses an added FP. Requiring \(A_i\subseteq P_0^c\) and \(B_j\subseteq P_0\) removes these terms. This restriction is useful and natural for relative-mask edits, but is not without loss of generality for arbitrary edit masks. Disjoint edit domains are not required for an exact solver.

## 2. Exact solver for general overlapping edits

There are \(m\) addition actions and \(n\) deletion actions. Aggregate pixels into membership atoms. For class \(c\), let

\[
h_{p,c,t}(X,Y)
\]

be the integer count of pixels with origin bit \(p\), GT bit \(t\), addition membership set \(X\subseteq[m]\), and deletion membership set \(Y\subseteq[n]\). Counts are summed over the fixed episodes of that class. Define the two-dimensional subset zeta transform

\[
Z_{p,c,t}(U,V)=\sum_{X\subseteq U,\,Y\subseteq V}h_{p,c,t}(X,Y).
\]

Let \(K_c(S,T,t)\) count predicted foreground pixels of GT type \(t\). Deletion priority gives exactly

\[
K_c(S,T,t)=Z_{1,c,t}([m],[n]\setminus T)
+Z_{0,c,t}([m],[n]\setminus T)
-Z_{0,c,t}([m]\setminus S,[n]\setminus T).
\]

The first term keeps origin pixels not selected by any deletion. The last two terms keep non-origin pixels touched by at least one addition and no deletion. If \(g_c\) is the fixed number of GT-positive pixels, then

\[
I_c=K_c(S,T,1),\qquad U_c=g_c+K_c(S,T,0).
\]

Consequently, after a single atom construction, a recipe requires only constant-many table lookups per class. With \(d=m+n\), zeta preprocessing takes \(O(Cd2^d)\) time and \(O(C2^d)\) memory; enumerating all \(2^d\) recipes takes \(O(C2^d)\) time. The pixel count disappears from recipe evaluation. Sparse atom construction can save storage before the transform, but does not change this dense worst-case bound.

**Proof of exactness.** A pixel survives deletion iff \(Y\cap T=\varnothing\), equivalent to \(Y\subseteq[n]\setminus T\). A non-origin pixel is added iff \(X\cap S\ne\varnothing\); subtracting atoms with \(X\subseteq[m]\setminus S\) implements that condition exactly. The atoms form a disjoint partition, so there is no double counting.

## 3. Faster representation for a bank of complete masks

For a bank \(P_0,P_1,\ldots,P_k\), define the relative actions

\[
A_i=P_i\setminus P_0,\qquad B_i=P_0\setminus P_i.
\]

They have disjoint domains by construction. The complete output is

\[
C(S,T)=\left(P_0\cap\bigcap_{j\in T}P_j\right)
\cup\left(P_0^c\cap\bigcup_{i\in S}P_i\right).
\]

The empty intersection is the whole pixel universe and the empty union is empty. Inside the origin the rule is AND; outside it is OR. Each source can be selected for addition, deletion, both, or neither, so there are \(4^k\) syntactic recipes per origin.

Only the \(k\)-bit source membership pattern \(Q\) is needed, rather than \(2k\) independent edit bits. Let \(h_{p,c}(Q,t)\) be its atom count, and define

\[
Z^\downarrow_{0,c}(W,t)=\sum_{Q\subseteq W}h_{0,c}(Q,t),\quad
Z^\uparrow_{1,c}(T,t)=\sum_{Q\supseteq T}h_{1,c}(Q,t).
\]

Then

\[
a_c(S,t)=H_{0,c}(t)-Z^\downarrow_{0,c}([k]\setminus S,t),\quad
r_c(T,t)=Z^\uparrow_{1,c}(T,t),
\]
\[
I_c=a_c(S,1)+r_c(T,1),\quad U_c=g_c+a_c(S,0)+r_c(T,0).
\]

Preprocessing takes \(O(Ck2^k)\), storage \(O(C2^k)\), and all-pair evaluation \(O(C4^k)\). For \(L\) possible origins, a shared \(L\)-bit membership histogram is sufficient; split it by the chosen origin bit. The total enumeration is \(L4^{L-1}\), excluding that origin's redundant own actions. This is an exact finite-family algorithm, not an assertion of efficient solution for arbitrarily large \(L\).

The following is a concrete exhaustive solver; the bounds below can replace the inner loops with certified branch-and-bound without changing its answer.

```text
construct integer class/truth membership histogram once
construct shared-DAG cost lookup once
best := best feasible single producer (or any feasible recipe)
for each allowed origin o:
    split histogram by origin bit
    compute outside subset-zeta and inside superset-zeta
    tabulate a(S, truth) and r(T, truth) for every subset
    for every S and T:
        Q := {o} union S union T
        reject if DAGcost(Q) exceeds budget
        I := a(S, positive) + r(T, positive)
        U := GTcount + a(S, negative) + r(T, negative)
        compare mean_class(I/U) against best with certified arithmetic
        update best using the predeclared cost/tie rule
return best recipe, integer I/U, dependency closure, optimum certificate
```

## 4. Shared computation is a DAG cost

Let inference steps form a fixed DAG. First cut dependency traversal at available cache vertices: their upstream work is not required merely to read that cached result. A producer leaf \(i\) then requires closure \(D_i\) in this remaining dependency graph. For a selected producer set \(Q\), the additive work cost is

\[
\kappa(Q)=\sum_{v\in\bigcup_{i\in Q}D_i\setminus\text{available cache}}w_v,\qquad w_v\ge0.
\]

This counts a shared encoder or graph construction once, even when several outputs use it. Compound producers retain all their actual upstream dependencies. The number of named output rows is not the number of atomic computations. An additive work cost is also not a measured parallel wall time; optimizing makespan requires a separate scheduling model.

The selected-producer closure is the complete accounting only for a recipe frozen before inference. An adaptive per-query selector also consumes its decision evidence. If it must compute all proposal masks or fields before choosing three, the unselected proposal producers have already incurred cost. Include the selector and every producer needed for its decision evidence in the required DAG closure; counting only the three final masks would understate inference cost. Cached evidence is handled by the same explicitly declared cache boundary, not assumed to be free at deployment.

For each DAG vertex define its requester mask \(R_v=\{i:v\in D_i\}\). Thus

\[
\kappa(Q)=\sum_v w_v\,\mathbf1[Q\cap R_v\ne\varnothing].
\]

Requester sets are obtained by reverse-topological propagation of producer bitsets. Aggregate weights by requester mask as \(W_R\), and compute a subset zeta transform \(Z_W(M)=\sum_{R\subseteq M}W_R\). Then

\[
\kappa(Q)=W_{\rm total}-Z_W([L]\setminus Q).
\]

After \(O(L2^L)\) cost preprocessing, each recipe's cost is a lookup with \(Q=\{o\}\cup S\cup T\). The DAG cost is monotone submodular because it is a nonnegative weighted coverage function. This property of cost does not imply submodularity of segmentation quality.

The solver can return the exact maximum under \(\kappa\le B\), the full score/cost Pareto frontier, or the minimum cost satisfying a fixed score requirement. If algebra removes a producer entirely, its cost should be removed only after that simplification is established; otherwise the selected recipe consumes its closure.

## 5. Certified pruning and exact arithmetic

For the disjoint-domain family, suppose a search node fixes selected actions \(S_c,T_c\), with undecided sets \(S_u,T_u\). Every descendant has

\[
I_c\le a_c(S_c\cup S_u,1)+r_c(T_c,1)=I^{\max}_c,
\]
\[
\mathrm{FP}_c\ge a_c(S_c,0)+r_c(T_c\cup T_u,0)=\mathrm{FP}^{\min}_c.
\]

Hence a valid macro-score upper bound is

\[
\mathrm{UB}=\frac1C\sum_c\frac{I^{\max}_c}{g_c+\mathrm{FP}^{\min}_c}.
\]

The numerator and denominator extremes need not be achievable together; their incompatibility makes the bound optimistic, which is safe for pruning. The cost lower bound is \(\kappa(\{o\}\cup S_c\cup T_c)\). Reject an over-budget node, or a node whose certified upper score is below the incumbent. Equality pruning must respect the declared cost/tie rule. For \(F-\lambda\kappa\), \(\lambda\ge0\), use \(\mathrm{UB}-\lambda\kappa_{\min}\).

Exhausting all unpruned leaves gives an exact optimum certificate. Pruning can be substantial but has no unconditional polynomial bound. With independent unit costs, no deletion, and all additions contained in GT, budgeted IoU maximization is maximum coverage; minimum cost achieving IoU one is set cover. Those special cases already preclude a general polynomial exact guarantee unless P=NP.

Aggregate TP/FP dominance is unsafe for partial states that can still receive further moves, even with equal dependency closures. Let the origin be empty, GT be \(\{t_1,t_2,t_3\}\), and addition masks be A=\(\{t_1,t_3,f_1\}\), B=\(\{t_2,f_2\}\), C=\(\{t_1,t_3,f_2\}\). Suppose the masks are cached, so their remaining required closures and costs are equal. Partial A has TP=2/FP=1 while partial B has TP=1/FP=1. Yet the same continuation C gives \(J(A\cup C)=2/5\) and \(J(B\cup C)=3/4\). Pruning B from its current aggregate counts would discard the better completed recipe: the continuation overlaps A's TPs but supplies B's missing TPs, and overlaps B's FP but adds a new FP to A.

Complete-recipe Pareto dominance is safe: classwise TP greater-or-equal, FP smaller-or-equal, and no higher complete cost imply no lower score and no worse budget feasibility. Partial-state pruning instead requires a continuation-safe proof. A sufficient condition is that both states admit the same remaining moves and, for every feasible continuation Z, the proposed dominating state has classwise no lower TP, no higher FP, and no higher full cost after Z. Full membership/edit signatures can establish such a condition; current aggregate TP/FP counts plus current closure inclusion cannot. The optimistic TP-max/FP-min bounds above remain valid and require no dominance assumption.

Comparing only standalone scalar costs is also unsafe under sharing: action X costs one unit for dependency u, action Y costs two for v, but a partner already requiring v makes X+partner cost three and Y+partner cost two. Preserve dependency signatures and include their interaction with the continuation when proving a partial-state cost comparison, or restrict Pareto comparisons to complete recipes.

The atom counts are integers, so the score is a rational number. Floating-point equality alone is not a mathematical exactness certificate. A practical certified comparison can enclose each ratio using \(b\)-bit dyadic floor/ceiling bounds, refine overlapping intervals, and compare exact rationals for ties. Bounds and pruning must compare upper versus lower enclosures. Fix a tie rule, such as smallest true DAG cost and then lexical recipe. Assume \(g_c>0\), or explicitly specify the empty-class convention before optimization.

## 6. A single IoU is different from a sum of class ratios

The actual objective is

\[
F(S,T)=\frac1C\sum_c\frac{I_c(S,T)}{U_c(S,T)}.
\]

Replacing it by \(\sum_c I_c/\sum_c U_c\) changes the problem. A four-GT-pixel counterexample has class GT sizes three and one and no FP. Candidate X recalls all three pixels in class one and none in class two: macro IoU is \(1/2\), pooled IoU \(3/4\). Candidate Y recalls one class-one pixel and the class-two pixel: macro IoU is \(2/3\), pooled IoU \(1/2\). The objectives select opposite candidates.

For one ratio only, \(\lambda^*=\max I/U\) is characterized by \(\max(I-\lambda^*U)=0\). Starting from any feasible recipe's ratio, finite Dinkelbach updates strictly improve the ratio until the optimum; they terminate because the candidate set is finite, without a general polynomial iteration claim. In the disjoint family without a coupled cost or budget,

\[
I-\lambda U=[a_{\rm TP}(S)-\lambda a_{\rm FP}(S)]
+[r_{\rm TP}(T)-\lambda r_{\rm FP}(T)]-\lambda g,
\]

so each inner solve separates into two \(2^k\) searches. A shared-DAG budget or cost depending on \(S\cup T\) breaks this separability. A sum of class ratios has no corresponding single scalar \(\lambda\) reduction. Generic alternating or greedy optimization therefore provides no global certificate for the macro objective; the atom solver and certified bounds evaluate that objective directly.

## 7. Complementarity, conflict, and counterexamples

For two additions, let \(t_i,f_i\) denote their TP/FP masses outside the origin and \(t_{12},f_{12}\) their intersection masses. With deletion fixed,

\[
I_{12}=I_0+t_1+t_2-t_{12},\quad
U_{12}=U_0+f_1+f_2-f_{12}.
\]

For two deletion actions inside the origin,

\[
I_{12}=I_0-t_1-t_2+t_{12},\quad
U_{12}=U_0-f_1-f_2+f_{12}.
\]

Overlap avoids double counting both desirable and harmful edits. Its sign is not determined by geometry alone. Cross add/delete intersections are the conflict terms in Section 1.

Even with disjoint edit domains, the IoU denominator creates interactions. Starting from intersection \(i\) and union \(u\), a pure TP addition of size \(a\) and pure FP deletion of size \(d\), with \(u>d\), have interaction

\[
J_{AB}-J_A-J_B+J_0=\frac{ad}{u(u-d)}>0.
\]

Conversely, positive interaction is not sufficient for a good method: a pure FP addition of size \(f\) and pure TP deletion of size \(t\) are both harmful but have positive interaction \(tf/[u(u+f)]\). The individual and combined effects must still be evaluated.

**Minimal failure of submodularity (two pixels).** Let GT be \(\{t\}\), background pixel be f, origin be \(\{f\}\), addition A be \(\{t\}\), and deletion B be \(\{f\}\). The scores of empty, A, B, AB are \(0,1/2,0,1\). Submodularity would require \(F(A)+F(B)\ge F(AB)+F(\varnothing)\), but \(1/2<1\). The edit domains are disjoint. Both actions can arise from the same complete producer \(\{t\}\).

**Failure of supermodularity with distinct actions.** Let GT be \(\{t_1,t_2\}\), origin empty, A be \(\{t_1\}\), and B be \(\{t_1,t_2\}\). Scores are \(0,1/2,1,1\); the supermodular inequality fails because \(3/2>1\). With distinct nonnested actions, use three GT pixels and additions \(\{t_1,t_2\}\), \(\{t_2,t_3\}\): scores are \(0,2/3,2/3,1\).

Thus quality is neither submodular nor supermodular in general, and a standard submodular-greedy guarantee does not apply. Adding a missing GT pixel can improve IoU from \(1/2\) to one; adding a background pixel to an already correct one-pixel mask worsens it from one to \(1/2\). “Less is more” is not an unconditional quality theorem. Expanding an at-most-budget candidate family cannot decrease its optimal development score; the optimum at an exact action count can go either way. A lowest-cost maximizer is a defensible tie rule, not proof that a smaller recipe is intrinsically better.

## 8. What joint versus single can and cannot predict

**Strict statement.** If the family includes every single producer directly as an allowed origin with no edits, unconstrained optimal development score is at least the best single score. This is an inclusion fact, not evidence of a positive generalization effect. Under a compute budget, inclusion holds only for singles feasible at that budget. A fixed-origin recipe may require an extra origin dependency, so equal-cost inclusion must be checked rather than assumed.

For comparison with a particular complete single mask, let a recipe change class TP by \(\alpha_c\) and FP by \(\beta_c\). Exactly,

\[
\Delta F=\frac1C\sum_c
\frac{\alpha_c U_c-I_c\beta_c}{U_c(U_c+\beta_c)}.
\]

A strict joint gain requires this sum to be positive. Complementary source errors are necessary for useful corrections but not sufficient: conflicts, erroneous deletions, new FP, the shared global selection, and class weights can cancel them. There is no theorem requiring a joint method to improve on the strongest single.

**Exact cross-marginal conditions for one class.** In the canonical disjoint family, let A add t TP and f FP, and D delete d TP and e FP, starting from i/u. Write

\[
J_A=\frac{i+t}{u+f},\quad J_D=\frac{i-d}{u-e},\quad
J_{AD}=\frac{i+t-d}{u+f-e}.
\]

Assuming positive denominators,

\[
J_{AD}-J_A=\frac{J_Ae-d}{u+f-e},\qquad
J_{AD}-J_D=\frac{t-J_Df}{u+f-e}.
\]

Thus both cross marginals must be positive to beat both separate outputs. Both standalone gains being positive does not ensure this. In fact, standalone A and D positive imply \(J_A>J_0\) and \(d<J_0e\), which makes the first cross marginal positive, but \(t>J_0f\) need not imply \(t>J_Df\). For a feasible example, take i=2, GT size 3, initial FP=2 (u=5), A adding one TP and two FP, and D deleting two FP and no TP. Then \(J_0=2/5\), \(J_A=3/7\), \(J_D=2/3\), and \(J_{AD}=3/5\). Both standalone gains are positive; joint remains below the stronger D output.

If A and D are the best addition-only and deletion-only states for that origin, positive cross marginals prove that this pair beats both branch optima, provided the pair is cost-feasible. This does not by itself prove that it beats every complete producer in the bank. For class-macro IoU, apply the exact signed differences per class and average; two positive macro standalone gains can even produce a negative macro joint gain. A feasible two-class example is: class 1 has GT=50, initial TP=50/FP=50; A adds 100 FP and D deletes the initial 50 FP. Class 2 has GT=98, initial TP=50/FP=2; A adds 48 TP and D deletes 49 original TP. Macro gains are A=23/200, D=1/200, and joint=-53/600. Neither overlapping edit domains nor an order conflict is needed for this failure.

**Strict disagreement ceiling.** For the complete-mask family,

\[
\bigcap_iP_i\subseteq C\subseteq\bigcup_iP_i.
\]

Every composition changes only the source-disagreement region. For each class, a relaxed GT-informed ceiling is

\[
J_c^{\rm ceiling}=
\frac{|G_c\cap\bigcup_iP_i|}
{|G_c|+|G_c^c\cap\bigcap_iP_i|}.
\]

This permits independently correcting every disagreement pixel, whereas a real recipe has shared global actions; it is an upper bound, not generally achievable. Unanimous source errors cannot be repaired by this family. If this ceiling is only \(\varepsilon\) above the strongest single, no finite composition in this family can exceed that single by more than \(\varepsilon\) on the same counted cohort.

**Conditional expectation.** Joint-versus-single may be unresolved when sources agree on the dominant remaining errors, when disagreements contain opposing signed edits, or when the possible gain is small relative to between-photo variability. A confidence interval crossing zero is a statistical statement under its stated sampling procedure; it cannot be deduced from mask algebra alone. Even a exact positive development optimum can have zero or negative population/held-out benefit because the family was selected on that development sample.

**Conjecture, not theorem.** A small recipe will generalize better when additional actions mostly fit noisy disagreement atoms. This needs an explicit noise and sampling model; source count or development Pareto optimality alone does not establish it. Shared DAG cost also makes source count an unreliable regularity proxy.

**Falsifiable predictions and minimum future checks, not work launched here.** After outputs are frozen, changes outside the disagreement envelope falsify the implementation. Classwise signed TP/FP changes and their exact ratio terms decide whether useful complementarity survives harmful edits. A narrow disagreement ceiling limits any possible joint gain; a broad ceiling makes no positive prediction. Compare the frozen global recipe and strongest feasible single on the same held-out identities with the same photo-cluster uncertainty procedure. One such complete comparison suffices for that hypothesis; an unresolved result should remain unresolved rather than trigger a claim that joint improvement is theoretically necessary.

## 9. Optional RCG spectral check: smoothing is not absolute information loss

Assume a fixed symmetric nonnegative graph, combinatorial Laplacian \(L=D-W\succeq0\), and finite \(\lambda\ge0\). The smooth estimator

\[
x=(I+\lambda L)^{-1}y
\]

is the unique minimizer of \(\tfrac12\|x-y\|^2+\tfrac\lambda2x^TLx\). If \(L=U\operatorname{diag}(\mu_j)U^T\), each graph-frequency coefficient is multiplied by \((1+\lambda\mu_j)^{-1}\). Finite \(\lambda\) attenuates rather than annihilates every coefficient; with exact x and known L, \(y=(I+\lambda L)x\). It is therefore incorrect to infer absolute information destruction from smoothing alone.

For a combinatorial Laplacian with a zero eigenvalue, the condition number is \(1+\lambda\mu_{\max}\). Quantization or numerical error can be amplified on inversion, and thresholding is many-to-one. Recoverability from the exact scalar field differs from recoverability from its stored precision or binary mask. Invertibility also retains only y for the given L; it does not prove semantic sufficiency of the underlying evidence.

The node equation is

\[
x_i=\frac{y_i+\lambda\sum_jw_{ij}x_j}{1+\lambda d_i}.
\]

With \(J=(I+\lambda D)^{-1}\), \(K=\lambda JW\) has row sums below one and

\[
(I+\lambda L)^{-1}=\sum_{n=0}^{\infty}K^nJ.
\]

Its entries are nonnegative and rows sum to one. Hence x is a graph-dependent convex mixture of the supplied scalar evidence. This formalizes local evidence absorption through crossing edges, but does not show that smoothing caused every remaining miss. A conditional semantic claim needs the available y/L/features to distinguish the target boundary or object in the first place. If two possible targets yield identical available evidence but different GT, no deterministic postprocessing of that evidence can resolve both. Whether the fixed graph and single-reference evidence are conditionally sufficient is the actual premise; this spectral derivation does not establish it.
