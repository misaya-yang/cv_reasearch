# Can collective reference coverage allow genuinely missing parts?

## Answer

Yes, a latent null assignment can allow partial reference matching while retaining genuine set interaction. But the null score must express evidence for absence/non-target, not merely grant an unconstrained escape. The current q/r/cov tuple supplies a cheap candidate null score, not a guarantee that absence is distinguishable from a difficult visible target part. The unit-cosine counterexample makes that gap exact within this model.

This is a derivation only. No new configuration, experiment, prediction, threshold fit or real-data variant was started while the joint run was pending.

## Minimal latent objective

Keep the fixed query-forward term A(S) and background-complement evidence from the existing joint objective. For reference foreground token j let

h_j(S)=max({−1}∪{cos(q_i,r_j):i belongs to a selected region}).

Introduce z_j∈{0,1}, with1meaning this source part has an acceptable visible counterpart and0meaning null/absent. Given a fixed null score ν_j,

max_{z_j}[z_j h_j(S)+(1−z_j)ν_j]=max(h_j(S),ν_j).

After subtracting the irrelevant constantν_j, the contribution is[h_j(S)−ν_j]_+. A symmetric construction can handle absent source-background appearances. Fixedν preserves the collective max and its diminishing returns. It is not a sum of independent query labels: several regions still compete to explain the same source token.

The change is a null hypothesis for each source part, not a new global coverage multiplier. Nonetheless choosingν without evidence would merely hide a new threshold inside the model. In probabilistic languageν must include the unmatched likelihood and visibility prior in the same units as the match score; cosine alone is not a calibrated log likelihood.

Partial assignment with rejection is established matching machinery, not a novelty claim. For example, [SuperGlue](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) predicts matching costs and learns scene/assignment priors from image pairs. Optional matching by itself does not supply those priors here.

## Strongest simple same-input null hypothesis

A parameter-free observable is reference-background confusability:

ν_j=max_{b∈R_B}cos(r_j,r_b).

A selected query token earns additional source-part coverage only when it matches r_j better than that reference part's strongest known background explanation. This uses labelled source-background information already available; it does not invent query-negative labels or target area.

Its identifying assumption is substantive:

For a visible target part, some corresponding query token exceedsν_j; for an absent part, every non-target query token stays at or belowν_j.

Reference-background coverage and foreground/background similarity calibration must transfer across images. The supplied source mask does not certify this condition. Occlusion, appearance change, or a previously unseen background may violate either side. A max over thousands of query candidates also has a different distractor population than a max over this one source background.

## What it fixes in the concrete example

Use r_F=[e1,e2], r_B=b=.8e2+.6e3, and query q=[e1,b,b]. The prior joint objective prefers adding one background b:1.88333versus1.75for the true mask{e1}.

Here ν_e1=0 andν_e2=.8. Selecting b cannot improve e2's partial coverage because its match equals the null score. The null-adjusted objective therefore prefers{e1}:1.95versus1.88333for{e1,b}. This algebra shows how a specified null explanation removes this particular incentive. It is not empirical evidence or proof of target identification.

Now keep exactly the same unit vectors, source labels, region partition and token locations. Hypothesis H0 says e2 is missing and bothb tokens are background. Hypothesis H1 says the firstb token is a difficult visible target part and the otherb is background. Both labelings are compatible with the proposed similarity/coverage model; it has no stated observation linking the firstb to target identity. The same null correction chooses H0in both cases, so it fails on H1.

This is an ambiguity of this permitted feature configuration and model assumptions. It does not establish that real DINO features necessarily realize such collisions, how often they occur, or that richer use of the full cached features cannot distinguish real cases.

## Two apparent solutions that collapse

1. Free deletion with no absence evidence: minimizing nonnegative matching costΣ_j z_j d_j(S) permits z=0for every part. The reverse matching constraint vanishes and the remaining decision is the forward unary.
2. Let the null explanation be the same source token's best match in the query complement:ν_j(S)=h_j(V\S). Then max(h_j(S),h_j(V\S))=h_j(V), a constant. Reverse evidence again disappears. Adding a penalty to null assignments avoids the algebraic collapse, but the penalty requires a justified visibility prior; it is not new identifying information by itself.

## Decision supported

A partial-match formulation can avoid forcing every source part to appear and can remain genuinely collective. The unresolved requirement is calibrated match-versus-null evidence, or another observable target-membership relation that separates a missing part's distractor from a hard positive. The current fields do not establish that requirement. Reference-background confusability is a concrete assumption to name, not a demonstrated solution or an authorized next run.

The existing joint result should be read before selecting another mechanism. A higher objective, a smaller matched subset, or the ability to drop source parts would not by itself resolve this ambiguity.
