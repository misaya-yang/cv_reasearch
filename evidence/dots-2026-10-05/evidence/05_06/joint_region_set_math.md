# Joint region-set inference: mathematical check before real results

The fixed objective is submodular, including its complement-background term, and generally nonmonotone. It does not collapse to independent region thresholding. Its coverage interaction still does not imply correct object identity.

## Exact objective

Let C1,…,CK partition all query tokens, R_F/R_B be the nonempty labelled reference roles, s_ij=clip(cos(q_i,r_j),−1,1), and v_kj=max_{i∈Ck}s_ij. All denominators are fixed; K=64 in the CPU implementation.

A(S)=[Σ_{i∈union(S)}max_{j∈R_F}s_ij + Σ_{i∉union(S)}max_{j∈R_B}s_ij]/N_Q.

H_F(S)=mean_{j∈R_F}max({−1}∪{v_kj:k∈S}).

H_B(V\S)=mean_{j∈R_B}max({−1}∪{v_kj:k∉S}).

F(S)=A(S)+0.5H_F(S)+0.5H_B(V\S).

The forward and reverse totals have equal weight. Empty facility similarity is−1. Thus each directional total is in[−1,1], and F lies in[−2,2]. This is the CPU normalization; a version with an additional factor0.5 is equivalent for optimization but has a different safe shift.

## Short submodularity proof

Write A(S)=A(∅)+Σ_{k∈S}a_k, with a_k=Σ_{i∈Ck}(max_F s_ij−max_B s_ij)/N_Q. It is modular.

For a fixed reference token, h(S)=max({−1}∪{v_k:k∈S}) has marginal max(v_e−h(S),0). Since h grows with S, the marginal decreases. Hence h and its nonnegative weighted sums are submodular.

For any submodular g, h(S)=g(V\S) is also submodular: apply the submodular inequality to V\S and V\T; their union/complement identities interchange S∩T and S∪T. Consequently the background-complement term remains submodular, and so does F. This says nothing about monotonicity.

More explicitly, adding region e gives its modular a_e plus reference-FG improvement in previously uncovered maxima, minus its reference-BG unique-witness loss from the remaining complement. Either term depends on the other selected regions. Complement competition is essential to this dependence.

## Deterministic double greedy and its scope

Start X=∅,Y=V. In fixed region order, compute a=F(X+e)−F(X) and b=F(Y−e)−F(Y). If a≥b, add e toX; otherwise delete e fromY. Submodularity gives a+b≥0. End with X=Y=S.

Let O_i=(OPT∪X_i)∩Y_i. Its loss in each step is at most the chosen gain in F(X)+F(Y). Summing gives F(OPT)−F(S)≤2F(S)−F(∅)−F(V), hence:

3F(S)≥F(OPT)+F(∅)+F(V).

This is the standard deterministic1/3scheme, not the randomized1/2scheme; see [Buchbinder et al., Sections2–3](https://www.openu.ac.il/personal_sites/moran-feldman/publications/SICOMP2015.pdf). For the raw signed-cosine objective, use G=F+2≥0 if quoting a multiplicative approximation. The algorithm and endpoint bound are invariant to constant shifts. The guarantee concerns this surrogate objective only, never IoU or identity.

Conditions: exact objective marginals; fixed nonnegative weights/denominators; unconstrained subset output; any fixed processing order. A nonempty/cardinality constraint, selected-set normalization, or posthoc label override is not covered by this argument. Naive cached evaluation costs O(K²|R|), after constructing region/reference maxima and modular terms; “linear-time” in the cited theorem counts value-oracle calls, not feature-matching runtime.

## Simplest same-input unary control

Replace max coverage by additive coverage:

H_F^add(S)=−1+(1/K)mean_{j∈R_F}Σ_{k∈S}(v_kj+1),

and use the same expression on the background complement. Keep A and its normalizations unchanged. The exact independent region marginal is:

u_k=a_k+[mean_{R_F}v_kj−mean_{R_B}v_kj]/(2K).

The+1constants cancel between foreground and background increments. Include region k iff u_k≥0, using the same deterministic tie rule. This uses every source token, every region response and the same forward evidence; it removes only competition/saturation. Forward-only sourceNN is an additional useful simpler baseline, but lacks the reverse information.

## Executable tiny exhaustive checks

joint_region_set_mathcheck.py evaluates40synthetic problems with actual unit query/reference vectors and K=2…6. For every problem it enumerates all subsets and checks all submodularity inequalities, exact additive/unary equivalence, the deterministic endpoint bound, and F∈[−2,2]. Maximum floating discrepancy is below7e−16. These are algebra checks, not segmentation evidence.

A valid unit-cosine witness uses r_F=e1,r_B=e2 and q0=q1=(.4,.6,sqrt(.48)),q2=e2, one token per region. Adding region1 to∅ has gain+.63333; after region0 is selected, its gain is−.06667. Identical appearance/unary scores therefore do not imply an independent label decision under the joint objective. It also demonstrates arbitrary choice among redundant witnesses.

The complementary worker supplied a stronger semantic counterexample, verified in the same script. Let r_F=[e1,e2], r_B=b=.8e2+.6e3, and q=[e1,b,b], with synthetic query truth[FG,BG,BG]. The forward unary and additive control both recover the true mask. Yet F({e1})=1.75, whereas F({e1,b})=1.88333; either one-background choice is globally optimal. A missing reference foreground part and a duplicate background witness let coverage manufacture a false-positive part. Even exact optimization cannot correct this objective error.

Therefore the real test can support collective evidence use only if it beats the full-input additive control. A higher facility objective alone cannot establish whole-object binding. No change to the first real configuration is proposed here.

Artifacts: joint_region_set_mathcheck.py, joint_region_set_mathcheck.json, joint_region_set_mathcheck.log. No features, queryGT, model or benchmark data were used.
