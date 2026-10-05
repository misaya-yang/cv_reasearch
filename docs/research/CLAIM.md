# Paper claim and evidence boundary

As of 2026-10-05, **the paper's method claim is unproven**. There is no selected algorithm to present as the
contribution. This page organizes existing evidence; it does not introduce a research direction or launch work.

## The one paper question

The latest user framing is a unified training-free binary-segmentation framework: represent and
select complementary additions from A* and deletions from B*, solve their joint optimization with
explicit marginal value and inference cost, and validate the resulting complete method against
strong published methods. The algebraic decomposition alone is not the method contribution.
Raw matching is the common accounting origin; changing that origin is not the research objective.
FoRIS contributes a particular collection of inference priors, not a required foundation.
The proposal library must include complete alternatives that do not read FoRIS masks or fields.
Reference matching, centering and query-graph agreement also carry explicit assumptions;
their removal controls distinguish which prior supplies any observed benefit.
The optimization mechanism and public-protocol complete result must support the framework claim.
See [framework derivation](../../evidence/local/research_20261005/operator_framework.md) for exact
class-wise accounting, conditional optimality and counterexamples that shape the solver.

Can a complete inference method use one labeled reference and a frozen DINOv3 backbone to identify the right
query objects and recover their extent more reliably than complete FoRIS and the strongest same-information
alternative, without extra images, class-name input, mask-pretrained models or base-class mask fitting?

That is the primary resource setting carried forward from the latest project correction and the supplied
dots record. Supervised readout and SAM3 results remain controls in their own settings. Improving a FoRIS
component or adding a resource does not by itself answer this question. The target is a method paper,
not an analysis paper. Neither acceptance nor experimental success can be guaranteed in advance.

The supplied record uses a practically meaningful target of stable **at least +2 mIoU points against the
same-resolution complete FoRIS**, while also beating a strong simple same-information control. This is a
working performance target, not a sufficient novelty claim, not +2 over every control, and not +2 in every fold.

## What the existing results support

All gains below are percentage points of class mIoU with paired 95% intervals. They are recorded results,
not experiments rerun during this cleanup. Different rows use different data/resolution/resources and are
not a leaderboard. The linked reports retain fold gains and episode increases/decreases where supplied.

| Evidence | Exact comparison and scope | Consequence |
|---|---|---|
| Base-class-fitted `convctx:layers` | COCO-20i 1-shot, CONFIRM600, seed 0, original resolution + FoRIS refinement: 63.325 vs FoRIS 59.783; **+3.542 [1.961, 4.957]**. [Report](../../evidence/local/results/decision_v1/infer_1_confirm/report.json) | A supervised positive result. It does not establish a zero-training method or universal class-independent law. |
| RCG, host-score + reference guide + query graph | COCO-20i, exposed old120 (67 classes), 1024 working pixels: 63.723 vs cached FoRIS native 62.018; **+1.705 [0.453, 3.250]**. Same graph MEAN_CONTROL 63.500; RCG minus control **+0.223 [-0.574, 0.737]**. [Report](../../evidence/dots-2026-10-05/evidence/cloud_results/rcg_verification.json) | A useful candidate/control, not independent confirmation or a demonstrated new identity mechanism. The control changes guide strength too, so it is not a pure CSLS ablation. |
| Complete-candidate `source_contrast_mean` | Same old120: 62.640 vs native 62.018; **+0.622 [-2.817, 3.201]**. Fixed secondary `scalar_contrast`: 63.807; **+1.789 [-1.971, 4.217]**. [Locked comparison](../../evidence/dots-2026-10-05/evidence/01_02/prior120_locked.json) | Do not promote the secondary arm after seeing results. Old120 was repeatedly used for development. |
| Frozen feature-mixture matte | COCO-20i DEV241, seed 0, original resolution: 59.513 vs complete FoRIS 59.122; **+0.391 [-0.093, 1.097]**. Versus same-information delete-only 59.554: **-0.041 [-0.490, 0.451]**. [Report](../../evidence/local/results/frozen_matte_original_v1/dev241.json) | A pre-CRF signal did not establish complete-pipeline superiority. Matte replaces CRF; matte followed by the same CRF was not tested. |

For cloud old120, the record traces episodes to the standard seed-0 source; statistics use RandomState(0),
2,000 connected-photo bootstrap draws. The compact report does not independently encode every episode-generation
field. Its 120 groups and 67 classes differ from full DEV241's 239 groups and 79 classes. Historical intervals
retain their own RNG/estimand; no intervals were recomputed here.
See [protocol](../../evidence/dots-2026-10-05/07_reproduction_and_continuation.md).

The latest local ledger also records stronger SAM3 visual/naming configurations than its early exemplar
reports. Keep their exact configuration and cohort; do not use the old exemplar as the current strongest
SAM3 control. Some latest reports are server-only, and their numbers were not independently checked here.

## What must not become the claim

- GT-selected masks, seeds, names, area or trimaps show privileged capacity, not a deployable selector.
- AUC, source reconstruction, lower graph energy and stable coefficients do not establish final-mask gain.
- A failed rule does not prove the frozen representation is exhausted. Better use of existing information
  remains possible; a new external information source is not a mandatory prerequisite for originality.
- Pixelwise Bayes classification is not IoU optimization; predicted area is not the true foreground prior.
- RePRI supports single-query transductive inference without episodic meta-training. INSID3 performs seed
  selection and region aggregation; it is not merely a fixed-quantile threshold method.

Theory and attribution corrections: [dots 03](../../evidence/dots-2026-10-05/03_novelty_and_contribution.md),
[05](../../evidence/dots-2026-10-05/05_theory_and_identifiability.md), [09](../../evidence/dots-2026-10-05/09_fp_fn_correction_mechanisms.md).

## Current complete-method correction (2026-10-05)

The new D layer-transition candidate did not establish the claim: DEV241 native 59.074825, D 59.192751,
paired +0.117926 [-0.828163, +1.581804], and -1.826908 versus RCG. Its first20 +3.260 was not stable.
RCG reaches +1.944835 on the same DEV241; a delta-only complete readout reaches 60.198027. These are
reused development outputs. Native replay differed by only 25 pixels across two episodes and is reported
under both baselines in the [audit](../../evidence/local/research_20261005/native_replay_audit.md).

Core working hypothesis: reference-conditioned evidence can improve the choice of **edits** to a strong
complete mask—recover true omissions while removing false inclusions—and joint inference can preserve
useful additions while repairing side effects. This is an outcome target, not a mandatory two-mask
architecture or an established contribution. Separate add/delete ablations and four error categories must
show where a deployable mechanism earns its gain. Query-GT error budgets alone do not establish a signal.
For class-summed mIoU, edit value is evaluated from per-class I/U; one pooled 38%/62% purity threshold is
not a universal decision rule.

## Missing evidence

No current candidate establishes the primary claim against the strong complete and same-information controls
in an original-resolution independent confirmation. The later supplied Astra package advances the old
dots export: exposed DEV220 native 58.829934, retained candidate 61.581120, +2.751185
[1.102142, 3.810278]. Versus RCG it gains +0.709731 [-0.704856, 1.334083]; versus same-domain,
same-ranking fixed fraction +0.482531 [-0.521612, 1.098586]. It is a promising complete exploratory
candidate, not an established independent adaptive-budget contribution. Its additional deletion gain
is concentrated in class58; it does not add pixels beyond RCG. Integer-I/U statistics have been locally
recomputed, but the final portable entrypoint has not been run on full241 here. Source, controls and
prepared replay are in the [intake](../../evidence/local/research_20261005/astra_intake.md).
Next work is governed by [PLAN](PLAN.md), not this gap list.

## Latest fixed Astra comparison and600 handoff

Fixed query-mean Astra on allDEV241 reaches62.654343 vs completeFoRIS59.074825:
+3.579518[1.320378,4.824421]. Its improvement overRCG is+1.634674[-.348770,2.523257],
so the full claim against strong controls remains unresolved. Last-step GPU replay matches all241
frozen masks exactly. Source:[verified report](../../evidence/local/research_20261005/pipeline_verified/recheck241/report.json).
Existing600 fixed evaluation is complete:FoRIS60.073377,MEAN_CONTROL61.467653,
fixedAstra60.673832,+.600455[-.631630,1.872440]vsFoRIS. It does not establish the target.
All600 are previously exposed; frozen parameters and600-optimized search rows remain separate.
Source:[fixed600 report](../../evidence/local/research_20261005/pipeline_verified/fixed600/report.json).
The operational transfer is recorded in[HANDOFF](../../HANDOFF.md).
