# RCG anatomy and boundary-evidence audit

2026-10-05. Bounded source/local-report audit. No inference, remote mutation, new method, or parameter selection was performed. Existing worker outputs and shared project records were not edited.

## Conclusion

The original anatomy explains edits from FoRIS **pre-CRF** to RCG; its edit-family table alone cannot explain the complete comparison. The now-completed native16-state supplement resolves the edit-accounting gap: versus complete native, RCG adds14.729M true/17.576M false pixels and deletes12.697M true/28.099M false pixels. The net is2.031M additional intersection and10.523M less union. These pooled counts describe the edits; class-macro attribution still requires the per-class I/U accounting rather than a universal purity threshold.

The completed4000 anatomy does **not** support “remaining errors are concentrated at the boundary”: <=16-pixel errors are25.70% of remaining pixel error, while body errors are54.98%. More densely sampled shifted features are a plausible local repair hypothesis, not established higher effective resolution.

The saved full4000 report gives complete FoRIS 60.931741 and RCG 62.333671: +1.401930 [1.094408, 1.686257], with fold gains +1.264456/+1.157919/+1.376479/+1.808868. RCG versus MEAN is -0.179301 [-0.375652, 0.003145]. These are benchmark-reused results, not fresh confirmation. This report has an empty `corrections_vs_native` map and explicitly says native masks were not independently replayed by that scorer; it supplies scores, not the missing edit attribution.

Evidence: `frozen_public4000_v1/report.json`; `scripts/score_public_frozen_edits_full.py:72-79`; `rcg_native_states4000_v1/report.json` and `states.npz`. The original frozen4000 correction map is empty, but the later native-state supplement now provides the missing counts.

## What the current anatomy actually measures

| Item | Actual definition | Interpretation limit |
|---|---|---|
| Add/delete | `RCG & ~pre`, `pre & ~RCG` | Relative to pre-CRF, not complete native. |
| Add families | Enclosed complement component; otherwise distance to pre <=16; otherwise far | Hole has priority; complement uses 4-connectivity. Three families are disjoint and exhaustive over added pixels. |
| Delete families | Pixels from an 8-connected pre component with >=90% removed; otherwise pre boundary distance <=16; otherwise interior | Almost-whole **predicted component** removal, not necessarily removal of an object or false target. Three families partition deleted pixels. |
| Remaining FN | Entire 8-connected GT component with zero prediction overlap; otherwise GT interior distance <=8 / (8,16] / >16 | One touched pixel makes a GT component cease to be wholly missed. |
| Remaining FP | Entire 8-connected predicted component with zero GT overlap; otherwise GT exterior distance <=8 / (8,16] / >16 | A false blob is not necessarily an annotated other-class object. One GT overlap can merge a large false region into the boundary/body categories. |
| “Object area” | `truth.mean()` | Total target-class semantic area in the image, not individual-instance size. |
| “Parts” | Number of 8-connected GT components | Semantic connected-region proxy; touching instances can merge and one instance can fragment. The value is saved but not used in the current size table. |
| Better/worse episodes | Episode IoU against **pre**, exceeding +/-0.01 | A 1-point dead zone, not the exact up/down/tie count against native. |

The four remaining-error families partition FN and FP separately because whole-component errors take priority over distance. Consequently their “object” and “boundary” counts are not independent axes: a completely missed small component may lie entirely near its GT boundary and still be counted only as object-level. The GT ceilings `fix_object/within8/within16/body` are valid count-based diagnostic counterfactuals; they are not available inference operations. Single-family mIoU gains do not generally add because I/U is nonlinear.

Evidence: `scripts/run_rcg_anatomy.py:118-190`. `pre+add_*` and `pre-del_*` I/U updates correctly change I by true edits and U by false edits. No query GT controls an inference choice in anatomy; GT defines diagnostic truth/area/components/distances and oracle corrections explicitly. The script itself does not check seals, save a manifest, verify packet/prediction hashes, or reject unequal run/root argument counts (`zip` would truncate). The supplemental run should enforce these checks and preserve all4000 sampled draws.

## Existing mechanistic evidence, without extrapolation

The existing reused600 ablation gives pre 60.944388, native 61.628013, graph-only `smooth_only` 63.582277, rank-only 60.939394, and RCG 64.237696. Graph-only minus pre is +2.637889 [1.792333, 3.268409]; RCG minus graph-only is +0.655420 [0.056436, 1.061844]; rank-only minus pre is -0.004994 [-0.584939, 0.279110]. Native finishing contributes +0.683625 [0.555301, 0.977717] over pre on this600. Re-solved RCG field maximum discrepancy is 5.96e-8.

This supports feature-graph agreement as a major source of this tested gain, with a useful conditional contribution from reference ranking. It does not show a standalone rank gain, unique reference identity discovery, fine-boundary recovery, or that a position graph can never work. The position-graph and alternative-lambda rows are exploratory controls, not new selected methods. The full4000 gain cannot be assigned these600 component effects without the corresponding full4000 comparison.

Evidence: `pipeline_verified/rcg_ablate600_v1/report.json` and `report.md`; equations and complete RCG finalizer in `src/ics/methods/rcg.py:17-81`.

## Completed4000 anatomy: observed budgets and side effects

Local verification of the fetched `rcg_anatomy4000_v1/counts.npz` found:

- All4000 per-episode native/RCG I/U pairs exactly match `frozen_public4000_v1/episodes.jsonl` in its preserved order.
- FN and FP family sums exactly equal total FN/FP for native, pre and RCG; maximum discrepancy is zero.
- The six disjoint pre-to-RCG edit families reconstruct every episode's RCG I/U exactly.

The complete readout is pre60.410620, native60.931741, RCG62.333671. Native minus pre is +0.521121 [0.455906, 0.605467] and RCG minus pre is +1.923051 [1.614930, 2.221573]. The latter contains a different finishing comparison than RCG versus native; subtracting native's gain leaves the actual +1.401930 complete gain.

The largest isolated pre-edit family gain is almost-whole predicted-component deletion: +0.799758 [0.625192,0.981792]. Its deleted pixels contain10.817M false and2.308M true pixels. Near deletion contains15.093M false and7.652M true pixels and gains about+0.37 alone. Hole filling contains2.619M true and1.136M false pixels and gains about+0.26; near addition contains7.579M true and8.025M false pixels and gains about+0.31. Far addition and interior deletion have intervals crossing zero. This supports useful cleanup and completion relative to pre, with material side effects; it does not identify semantic instances or isolate complete-native attribution.

| Remaining RCG error family | FN+FP pixels, millions | Share of remaining pixel error | Perfect correction gain over RCG |
|---|---:|---:|---:|
| Whole-component proxy, priority bucket |42.934|19.32%|+5.925803 [5.36,6.60]|
| Within8 Euclidean pixels, excluding object bucket |39.860|17.94%|+6.215432 [6.09,6.51]|
|8 to16 pixels, excluding object bucket |17.236|7.76%|+2.337193 [2.26,2.44]|
|Further than16, excluding object bucket |122.143|54.98%|+16.097566 [14.97,17.17]|

Correcting both boundary buckets together gives a count-only GT oracle of71.108529, +8.774857 over RCG; this supplemental point estimate was recomputed locally from saved counts, and its CI was not recomputed. It is not the sum of the two single-bucket gains. The body oracle is larger and the object budget remains substantial. Object-first allocation also means these fractions are not a complete independent distance histogram: some errors assigned to the object bucket can themselves lie near a GT boundary.

Against complete native, RCG reduces total FN by2.031220M and total FP by10.523149M. However, wholly missed GT-component FN mass increases by4.043553M (5.317816M to9.361369M), while within8 FP increases by2.201430M (17.755683M to19.957113M). The object-FP bucket decreases by7.819979M and body-FP decreases by4.888144M. These aggregates show a real cleanup/recovery tradeoff. They do not count recovered/lost true instances, and dynamic component-family reassignment means bucket deltas alone cannot identify which native pixels were edited. The16-state supplement supplies that missing exact comparison.

The boundary experiment is justified as a bounded question about whether a remaining local error budget is accessible with available feature/RGB evidence. Dominant boundary error, true fine-feature resolution, and actual deployable recovery are not established by these budgets. Body/whole-missed errors must remain visible even if boundary gains are found.

Evidence: `rcg_anatomy4000_v1/report.json`, `report.md`, `counts.npz`; local count reconstruction and metric calculation with `ics.experiment.metric`. All numbers are working-resolution1024,80 classes,4000 preserved draws,2742 connected-photo groups, benchmark reuse.

## Completed native-state supplement and minimum counting contract

The fetched supplement is `rcg_native_states4000_v1/{states.npz,manifest.json,report.json,receipt.json}`. Its actual state encoding is P=1,N=2,R=4,T=8. Independent local checks passed: all4000 histograms sum to1024^2; histograms reconstruct every P/N/R I/U exactly; native/RCG match the previously verified4000 per-draw I/U; and all four native-to-RCG edit totals match the report. No additional predictions were made by this audit.

| Complete native-to-RCG edit | Total pixels, millions | <=8 |8 to16 |>16 |
|---|---:|---:|---:|---:|
|Add TP|14.728544|4.137502|1.672747|8.918295|
|Add FP|17.576193|4.767757|1.455794|11.352642|
|Delete TP|12.697324|3.664281|1.777528|7.255515|
|Delete FP|28.099342|2.650485|1.515806|23.933051|

Unlike the object-first anatomy buckets, these are a pure GT-distance partition. Within16 pixels, net I increases0.368440M and net U increases2.057260M. Further than16, net I increases1.662780M and net U decreases12.580409M. This shows prominent distant-FP cleanup at the pixel-count level; it does not replace class-macro signed-value attribution.

RCG-only edits (P=N differs from R) produce net I+0.289273M/U-11.219219M. Undoing native-only finishing edits (P=R differs from N) produces net I+1.741947M/U+0.696070M. The shared pre edits have6.187112M true and10.807321M false changed pixels but contribute no difference between the final masks. Thus the final gain is not simply the pre-edit table and is not simply “omitting CRF”: the feature readout and differing finish both enter the complete comparison. Their class-macro net values can be calculated from the same ledger without more inference.

For each retained episode let P=pre, N=complete native, R=complete RCG, T=query GT. Save a 16-cell histogram of `(T,P,N,R)`, for example `bincount(8*T + 4*P + 2*N + R, minlength=16)`. Also save those cells by a fixed, common two-sided Euclidean GT-boundary distance partition: <=8, (8,16], >16 working pixels. Set distance to infinity if the opposite GT label is absent. This prevents square-window and image-frame artifacts from masquerading as Euclidean boundary errors. Store episode metadata, original batch-qualified key, class, fold, source hashes, and the unmodified sampled order.

The binary mask-state patterns have exactly four disjoint cases:

| State | Meaning | Contribution to R versus N |
|---|---|---|
| P=N=R | Both unchanged | None |
| P=N, R differs | RCG changes pre; native leaves that pixel unchanged | RCG-only edit |
| P=R, N differs | Native changes pre; RCG leaves that pixel unchanged | Undoing a native-only finishing edit |
| N=R, P differs | RCG and native make the same binary change | None in the final comparison |

Each state is split by T, so the histogram exactly yields final R-versus-N add TP/add FP/delete TP/delete FP, the same four N-versus-P finishing counts, and agreement/overlap footprints. Whenever N and R differ, P must equal one of them; overlapping P-to-N and P-to-R binary edits agree and disappear from the final difference. An exhaustive enumeration of the eight binary `(P,N,R)` states verified this partition locally.

Reconstruct every P/N/R I/U independently. Assert `I_R-I_N=add_TP-delete_TP` and `U_R-U_N=add_FP-delete_FP`, and match per-episode native/RCG I/U to the existing4000 ledger. Report class-summed mIoU, fold/batch gains and paired photo-connected uncertainty. Report the scalar identity `(RCG-pre)-(native-pre)=RCG-native`; do not add isolated family gains or infer causality from pixel purity. The histogram can also yield exact N-only-addition and N-only-deletion counterfactual I/U (union/intersection with R), with their interaction stated rather than silently summed.

If an object-level claim is needed, add only post-hoc semantic-component counts on the same pass: GT components with zero N overlap and >=90% R coverage, GT components newly losing all overlap, and N components with zero GT overlap and >=90% deletion by R. Name these semantic connected regions, retain both count and pixel mass, and keep boundary distance as a separate axis. True instance attribution requires instance IDs, absent from the binary packets. Other-class versus background FP attribution requires the already-existing semantic PNG, not the binary truth mask alone. Neither is required for the minimum CRF attribution ledger.

Reuse path: `outputs/claude_official/root{0..6}/results/extent_v1/run/packets/<key>.npz` provides P/N/T; `outputs/claude_official/run{0..6}/predictions/<key>.npz` provides R. Root's current runtime inspection reports all4000 packets retained and no features in these seven cache directories. No GPU or features are required for this supplement. Local source/reference audit did not independently inspect these remote files.

## Smallest useful fine-feature diagnostic, proposed only

First read out the existing complete subtoken results, including RCG+CRF, feature-guided and RGB-guided controls. No new variants are needed. `run_subtoken.py:80-103` computes four shifted query grids and a 128x128 guide but only saves masks; neither fine features nor affinity matrices nor fine fields persist. Thus the present saved masks/counts/coarse field cannot recover fine-feature separability on CPU. Do not alter the active worker or restart its600 run. A diagnostic can reuse the retained single-reference q/r/cov cache on a predeclared metadata-only cohort of80 episodes,20 per fold, avoiding duplicate query photos where feasible. Root reports another agent is preparing this fixed80 diagnostic after the current job, with about2.5GiB of fine features retained; this is preparation, not evidence that the cache/run/result already exists. This is a mechanism diagnostic, not a cohort sufficient to select a +2-point method. Record missing classes and photo-group uncertainty.

Use exactly the existing four (+/-4,+/-4) query shifts, unchanged frozen encoder/debias setting and geometrical correspondence; no new reference forward. Match coarse/fine representation precision with one fixed cast and report it. Produce all legitimate evidence scores before opening query GT:

1. **Single-reference semantic evidence.** With the existing reference division `cov>=0.5` versus `<0.5`, compute the fixed max-NN foreground-minus-background cosine margin at each fine-cell center. Compare fine descriptors to two fixed coarse controls: bilinear interpolation of the coarse max-NN margin; and max-NN margin from normalized bilinear interpolation of coarse descriptors. Reuse q/r and reference coverage; fit no classifier or linear head. For RGB use transformed query8-cell mean colour and reference16-token mean colour, with margin `min_BG squared_colour_distance - min_FG squared_colour_distance` using the same reference bank division. There is no tau, sigma, threshold, or head search.
2. **Post-hoc readout.** After scores are fixed, query GT identifies target/background fine centers within16 Euclidean pixels of the true boundary. Report both boundary AUROC and the fraction of FG/BG center pairs within the same16-pixel parent cell that are correctly ordered. Compare identical pairs across fine, both coarse controls, and RGB; separate pairs by their fixed8/diagonal separation and report coverage. Bootstrap connected photos, not millions of pixels. An improvement over replicated coarse features alone is insufficient; the interpolated coarse and RGB comparisons matter.
3. **Separate oracle witness diagnosis.** For the same existing valid25 coarse neighbors per fine center, label coarse witnesses with post-hoc GT coverage >=0.9 or <=0.1. For centers with both witness classes present, record `max affinity to same-GT witnesses - max affinity to opposite-GT witnesses`, using cosine for fine/coarse features and negative squared colour distance for RGB. Report witness coverage and same-label/cross-label gaps on matched centers. This asks whether the guide can distinguish local sides when supplied oracle witness labels; it is not one-reference foreground evidence or a deployable selector.
4. **Field availability ceiling.** Separately count whether any neighbor's sealed RCG field value can produce the correct center label: for foreground, `max_j z_j>0.5`; for background, `min_j z_j<=0.5`. Positive normalized guided weights form a convex combination, so a missing correct-side value is a limitation of this neighborhood/field before affinities are considered. The GT-conditioned max/min choice is an oracle diagnostic at128-grid centers, not a final1024-mask ceiling or inference arm.

The decision is precise: oracle witness separation without stronger single-reference margins supports local geometry but not semantic orientation; stronger fine margins than both interpolated coarse and RGB controls support usable conditional fine evidence; no correct-side field values implicate the sealed coarse field/neighborhood; and a mask gain equal to RGB/CRF leaves a fine semantic mechanism unestablished. Only the existing complete-mask comparison establishes actual method benefit.

Four shifted16-pixel patch grids place centers8 pixels apart, but each encoder patch remains16 pixels and Transformer features are contextual. `run_subtoken.py:4-6` therefore establishes denser sampling, not higher effective spatial resolution or a universal “half-token blur” bound. That premise must be measured by the above diagnostics and the complete output.

## Additional interpretation flags

- Subtoken's purported “within8px” band uses a17x17 square convolution with zero padding (`run_subtoken.py:99`), whereas anatomy uses Euclidean EDT. The square includes diagonal distances up to8*sqrt(2) and can flag foreground image-frame pixels because padding is treated as zero. Do not compare the two band totals directly.
- Subtoken loads query truth before forward inference, though visible mask construction does not depend on it. GT then defines error counts and chooses parameters on fitting folds (`112-122`); held-fold inference is GT-independent by visible dataflow, but “query GT is only opened to count” is too strong. No source/prediction seals or manifest are written by this script. A strict diagnostic should score first, then open GT, and save identity/source receipts.
- Ablation CG status is captured but not checked (`run_rcg_anatomy.py:82`); the excellent sealed-RCG parity validates that arm's reconstructed field, not convergence of every exploratory graph/lambda variant.
- Neither GT error ceilings, oracle witness discrimination, dense feature sampling, nor boundary AUROC establishes the missing complete-method contribution by itself.
