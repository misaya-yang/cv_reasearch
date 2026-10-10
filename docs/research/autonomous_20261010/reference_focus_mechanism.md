# Actual reference focus: new source features, weak transferable identity

2026-10-10. The fixed complete paired head is rejected. New reference-focus features are real and change source role geometry, but the main query gain is an extent change, with little evidence of stronger target ordering. This supports neither a new complete method nor cross-scale alignment or parameter rescue.

Artifacts: `evidence/local/autonomous_20261010/reference_focus_mechanism/{recipe.json,study.py,reference.json,query_recipe.json,query_saved.py,query.json}`. The reference phase independently validated400 raw requests, every entry/payload/O24 hash and source mask; it read no query raw/GT or prediction fields, and took61.047s at CPU1. The saved-query phase froze its recipe before reading GT, required the completed200 seal, read no query raw, generated no masks and took4.129s. Frozen head SHA remains `37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413`.

## Complete outcome and evidence grain

Parent/benchmark owner completed and independently audited the unchanged CLI1024 head evaluation:

|Fixed complete CRF output|Deep total I/U|PACO87 observed fold/class mIoU|
|---|---:|---:|
|Real reference, equal query views|25.619896|39.056805|
|Derived reference, equal query views|26.979236|35.977374|
|Full FoRIS|29.377542|45.629834|
|Fast9|37.344028|—|
|MEAN|—|46.824561|

PACO real-minus-derived is+3.079431, Deep−1.359340. Global/local mechanism controls reverse preferences as before: real whole/local gives Deep23.079959/26.884548, PACO41.163913/36.003119. The equal default was not changed. These are exposed development cohorts, not ten-dataset or untouched confirmation. Original-frame results remain separate in the benchmark artifact.

## Geometric purity did add feature variation, not a transferable guarantee

The source-mask audit's Deep purity improvement is real geometry: whole high-coverage token median3→focus77 and no-high-coverage references35→2. Its consequence in normalized/APD features is measurable:

|Source metric, mean over100 episodes|Deep derived→real|PACO derived→real|
|---|---:|---:|
|FG kernel effective patterns|17.752→23.873|8.450→13.305|
|Sampled FG/BG normalized kernel overlap|.22343→.19031|.10051→.08593|
|Full physical-source AUC|.957295→.959044|.994189→.993777|
|Full physical-source binary-role loss|.330153→.324028|.136170→.142125|
|Physical heldout source AUC|.927254→.936539|.979512→.976184|

Kernel patterns use a fixed.07 kernel and collapsed source occurrence weights; they measure feature diversity, not independent semantic exemplars. Pixel priors are unchanged by the inverse-overlap construction, and both arms have identical labels/IDs/loss. Thus the feature differences are not a hidden foreground-mass increase.

Source quality predicts query success incorrectly: Deep source separation/heldout performance improves while its query output worsens; PACO source self-fit/heldout performance worsens while its query output improves. This is a new actual-observation version of the old source-Fisher transfer counterexample. Higher source purity and fitting accuracy cannot serve as a query acceptance criterion.

## Paired same-region evidence: useful signal and context drift coexist

Real focus and derived focus represent the same physical512 region at64×64. Mean aligned FG/BG feature cosines are Deep.84455/.85751 and PACO.81219/.78072. Real-minus-derived feature drift has

|Paired reference quantity|Deep|PACO|
|---|---:|---:|
|Common role-mean drift norm|.24355|.38083|
|FG-minus-BG mean drift norm|.21281|.33533|
|FG centered trace correlation|.59281|.62972|
|BG centered trace correlation|.68441|.66158|

Common and role-specific changes are both substantial. The within-role correlations/variance do not identify pure nuisance: appearance, object mode, crop context, encoder coordinates and finer visual response all contribute. Conditional covariance alone does not justify whitening or projecting out the observed difference.

Task projections make this limitation concrete. With a **fixed** source fitted direction, project the aligned feature difference onto that direction; its intercept cancels:

|Fixed direction; focus real-minus-derived projection|Deep FG /BG|PACO FG /BG|
|---|---:|---:|
|Derived-fit direction|−.00662 /−.08781|−.27896 /−.09981|
|Real-fit direction|+.08265 /−.05453|+.00074 /+.01962|

Deep's fixed derived direction gets+.08119 incremental role contrast on the newly measured source features; its real direction gets+.13717. PACO instead gets−.17915 role contrast along the derived direction: **geometrically purer real FG loses response along the coarse-derived task direction**. Refitting restores most of that source shift, but yields almost no positive new source role contrast. Treating all view drift as nuisance or demanding feature invariance would erase role-dependent changes rather than identify the correct target automatically.

The source holdout used two fixed physical64-pixel checkerboard folds inside the focus, removing both whole and fine observations of each held physical block. Outside-whole source remained training support. Cross-representation AUC is

|Train/test representation|Deep|PACO|
|---|---:|---:|
|Real→real|.93654|.97618|
|Real→derived|.92624|.97889|
|Derived→real|.93368|.96525|
|Derived→derived|.92725|.97951|

This prevents duplicate-footprint fitting leakage, but shared image/global DINO context and neighboring appearance remain. It is a correlated same-reference check, not cross-image validation. The main head ignores physical pair correspondence after labels/weights are assigned; it pools feature-role evidence.

## Query ordering versus the learned zero level

Saved FP32 fine128 fields were evaluated with exact FG/BG pixel mass in canonical1024 ROIs; these are node-field ranking/sign diagnostics, not bilinear pixel AUC or complete-mask mIoU.

|Equal-view macro AUC|Deep derived→real|PACO derived→real|
|---|---:|---:|
|All query pixels|.948316→.947390|.905327→.903303|
|FoRIS foreground|.734403→.740592|.749004→.744748|
|Real/derived complete-CRF disagreement|.686825→.700557|.608590→.590047|

Deep has a small conditional ordering gain, while PACO gains complete mIoU despite slightly worse overall/FoRIS-FG ordering and worse disagreement ordering. This is not robust new target discrimination.

The source-fit directions remain nearby, mean cosine.97296/.95841. Their learned real-minus-derived intercepts average+.06921/+.08695. On all query pixels, the equal field changes FG/BG mean scores by

```text
Deep: +.03346 / +.01930
PACO: +.07436 / +.06812.
```

After subtracting the frozen intercept difference, direction-only mean changes are Deep−.03574/−.04991 and PACO−.01259/−.01883. This arithmetic decomposition does not prove pure intercept causation; the direction, image-specific distributions and finalizer also change. It shows that a broad positive shift dominates the cohort mean, rather than a strong separation of new target and distractor responses.

Equal-field FG mass at nonpositive score falls Deep.20388→.18576 and PACO.52083→.45762; BG mass at positive score increases.09097→.10161 and.03844→.05285. PACO recovers extent from a conservative independent head; Deep's thin/rare FG pays for additional background. Within PACO actual/derived disagreement, true FG and BG score shifts are+.08549/+.09054, almost the same, and ranking worsens. No threshold was fitted and no bias-corrected mask was generated.

## What is identifiable and the causal control an alignment claim would need

The existing two measurements change RGB detail sampled by tokenization, image crop context, object scale and encoder coordinates together. The derived interpolation control establishes new encoded feature information, but cannot separately assign its cause to detail or context. Neither source covariance nor a source optimal zero point identifies query nuisance removal.

A cheap reference-only **pairing null** for any future covariance mechanism should keep both feature-role bags fixed and permute only the cross-view mate mapping within exact fine coverage strata. Focus coverage has65 dyadic values for binary8×8 blocks, and all focused physical weights are32, so such a fixed one-seed permutation preserves both weighted role marginals exactly. The current ordinary head must remain unchanged; recomputing quantile fits after permuting feature positions would also change sampling and would be a different control. This tests whether paired covariance contains physical information beyond marginal role distributions. It does not prove that its low-variance directions are task nuisance.

A genuine **new-encode detail/context control**, if later justified, would use the same selected focus FOV/encoder object scale/relative coordinates but pre-bandlimit that crop (fixed512→256→512 RGB) before the same1024 encoder preprocessing. Compare actual focus against this controlled detail loss with unchanged mask and head recipe. This separates finer RGB evidence from crop-context/coordinate change more directly than interpolating old features. No such control has been encoded or launched here; current failed complete output gives no reason to spend another full200 run or implement alignment yet.

No complete focus-derived cross-dataset method is proposed from this evidence. The precise gap is absolute extent plus hard-region target membership under source→query context/appearance shift. Purer source pixels and stronger same-image task separation do not fill it. The current kernel/head/crop recipe remains frozen and rejected as a universal candidate; no dataset-specific selector, scalar rescue or new graph follows.
