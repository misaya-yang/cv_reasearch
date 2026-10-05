# Nonlocal semantic graph: better than local propagation, weaker than direct seed ordering

## Measured result

The one allowed graph replacement improves on the local spatial graph but does not establish a seed-to-whole-object advantage over simpler same-seed scores. Close these two fixed seeded-cut constructions; do not scan k or restart constants.

Old60 contains56oracle-eligible cases (20old20,36new40). Every arm uses the exact same16GT-pure FG and16GT-pure BG seed coordinates as the previous run. This is a privileged capability diagnostic, not a deployable method evaluation.

Graph replacement only: non-spatial cosine16-nearest-neighbor edges, symmetrized by union, with the same exp(−d²/(2median d²)) weight rule. PPR continuation.99, seed labels, conductance objective, tie handling and rendering are unchanged. A unit-seed-mean margin is the additional simple control. All predictions/order fields were frozen before this run's evaluation. Entire score-tie blocks must stay together in every oracle prefix.

| Condition | Cut | NN>0 | Mean>0 | PPR>.5 | PPR best prefix (oracle) | NN best prefix (oracle) | Mean best prefix (oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old20,20cases |32.980|49.811|50.925|29.411|66.410|77.925|78.519|
| New40,36cases |26.811|28.289|29.631|22.037|69.166|75.609|75.397|
| Pooled56,supplementary |28.621|35.887|37.260|23.957|67.272|75.922|75.873|

All values are classmIoU on1024 repeated-token masks beforeCRF. Group bootstrap2000seed0 recomputes classI/U, with per-fold and episode effects retained in report.json. No confirmation/full-pipeline claim.

- Pooled cut−NN:−7.265[−13.099,−1.062],19up37down. Cut−mean:−8.639[−14.739,−2.183].
- Pooled cut−previous local cut:+4.650[+1.083,+8.915]. This is an improvement over the weaker propagation graph, not over the simple control.
- Pooled PPR-bestprefix−NN-bestprefix:−8.650[−11.059,−5.329]. Relative to mean-bestprefix:−8.601[−11.100,−5.211]. Therefore the propagation ordering itself loses usable target ordering, beyond the chosen cut.
- Nonlocal PPR's best prefix67.272does improve the same-case local best prefix47.276by+19.996[+15.431,+25.065]. The causal distinction between spatial and semantic graph connectivity matters, but the simpler NN still reaches75.922.

## Objective mismatch survives this replacement

In all56cases the selected cut is cheaper than the true GT-majority mask under both conductance and normalized cut, although only conductance was actually optimized. Mean conductance is.00505for the selected region versus.07068forGT; mean Ncut is.00735versus.07835. The selected region covers51.30%of the query on average. Thus swapping local for semantic-neighbor connectivity improves ranking capacity without aligning this unsupervised cut objective to the intended target.

Conductance=cut/min(volFG,volBG); Ncut=cut*(1/volFG+1/volBG). This is not a reproduction or refutation of every normalized-cut method.

One case has an unseeded disconnected component,339tokens and0GT-positive pixels. The primary predictor already assigns that component PPR ratio.5. An auxiliary saved ratio initially contained0/0NaN; original fields and source were preserved, and the evaluator applies the same.5definition, without changing any primary mask. The complete.5tie block is indivisible in oracle prefix search. The correction and exact original-source hash are recorded in auxiliary_correctness_repair.json.

## What the strong NN cut gap says

For the same56perfectly pure seed sets, NN>0scores35.887but its GT-bestcut reaches75.922. The saved-margin diagnostic does not fit a new threshold:

-52of56episodes require a positive best threshold; zero is not among the optimal intervals in any case.
- The evaluation-only NN threshold median is.282(IQR.182–.431), but old20andnew40medians differ:.171versus.344. It is not a justified deployable constant.
- Across pixels, actualNN has15,850,722FP and493,572FN. Its GT-bestcut has1,642,408FP and685,258FN. The large gap is chiefly excessive foreground inclusion.
- Seedmean has the same qualitative pattern:52of56positive thresholds and a75.873bestcut ceiling.

Only Fmax−Bmax was saved for this exact query seed bank. Separate FmaxandBmaxwere not. A margin offset can arise from incomplete negative coverage, biased positive coverage, or both. Source-bank packet maxima use different anchors and cannot isolate that cause. Consequently this evidence separates a decision-level mismatch from ordering capacity, but does not identify background representativeness as the cause. NoBGstory, learned threshold or new gate is inferred.

This cheap last decomposition reuses existing IU and score fields; GT area and prefix sizes are algebraically recovered from earlier evaluation records, with no new GT or feature reads.

## Files

- seeded_nonlocal_extent60.py: corrected runnable fixed inference/evaluation.
- seeded_nonlocal_extent60_initial.py: exact initial source retained for provenance.
- seeded_nonlocal_evaluate_frozen.py: auxiliary undefined-component evaluation repair, no graph recomputation.
- seeded_nonlocal_reachable_comparison.py: same-case oracle-reachability and cost summary.
- seeded_frozen_cut_gap.py: saved-score/IU decision-gap diagnostic.
- seeded_nonlocal_extent60/report.json: authoritative result; initial auxiliary-invalid report explicitly named and retained separately.
- seeded_nonlocal_extent60/same_case_reachable_graph_comparison.json
- seeded_nonlocal_extent60/frozen_cut_gap.json
- seeded_nonlocal_extent60/auxiliary_correctness_repair.json

The run took109.22CPU seconds for56pure-seed cases, without new encoders, RGB, models or feature copies. No deployment120follow-up was triggered.
