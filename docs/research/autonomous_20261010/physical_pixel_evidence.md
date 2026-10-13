# Physical pixel evidence: frozen local appearance transfer diagnosis

**Completed400: local pixel appearance contains transferable Fundus detail, but does not solve PACO target identity.** The exact same fixed probe has near-chance within-patch ranking inside PACO's complete FoRIS foreground. Across all three datasets, FoRIS's existing continuous score has stronger full-query and FoRIS-foreground ranking than the standalone pixel readout. These results justify investigating task-conditioned use of a local cue, not replacing semantics with a universal appearance classifier or claiming a new method.

This is one information diagnosis, not a segmentation candidate or an originality claim. Its source, parameters, all400 episode bindings and synthetic checks were frozen before this track read query ground truth. All400 scores were sealed before query scoring. Inputs are the existing lawful reference RGB/mask and query RGB; no DINO encoder, raw-cache writer, new pretrained decoder, threshold scan or graph was used.

The panel retains the sealed DeepGlobe-road100 and PACO-Part100 development draws and the complete released Fundus seed0 200 draws. Deep remains a custom exploration because the paper's road pairing protocol is unreleased. Repeated photographs and source/query reuse across different episodes are retained and disclosed; each fit receives only that episode's own reference mask.

## What the mathematical ceiling does and does not establish

For a fixed aligned block partition, let block j contain A_j pixels, f_j foreground pixels, and let F=sum_j f_j. A binary decoder constrained to assign one constant label per block, selecting block set S, has

`IoU(S) = sum_{j in S} f_j / (F + sum_{j in S}(A_j-f_j))`.

At the optimal ratio J, including a block improves the ratio precisely when `f_j/(A_j-f_j)>J`, equivalently its foreground coverage exceeds `J/(1+J)`. Consequently sorting blocks by foreground coverage and enumerating all prefixes gives the exact optimum. The implementation also handles unequal block areas in the original output frame. Ties do not invalidate the optimum. Exhaustive2×2 mask/subset fixtures and a nonuniform-area fixture verify the formula.

The64×64 and128×128 partitions correspond to16 and8 canonical RGB pixels at CLI1024. Their ground-truth-only optima are limits for these **block-constant binary decoders**. They are not achievable inference scores and are not upper bounds on DINOv3 O24 information. O24 has a global receptive field; a bilinear or pattern-producing decoder can vary its label inside an encoder block. A low block ceiling establishes a representation-to-readout geometry problem only for the stated decoder class.

## One matched readout, with an explicit transfer control

The fixed12-channel pixel descriptor contains RGB, local contrast, gradient magnitude and local RMS texture, three channels each. RGB is standardized separately per image using unlabeled channel mean/std. Contrast is `z-Gaussian_sigma2(z)`, gradient uses magnitude of Gaussian_sigma1 first derivatives, and texture is the square root of Gaussian_sigma2 smoothed squared contrast. Reflect boundaries and truncate3 are fixed. There is no color-space, sigma, feature-dimension or regularization grid.

One13-parameter logistic readout is fitted from the reference with equal4096 foreground/background physical-area midpoint occurrences, slope penalty lambda=.01 and a free intercept. The training pixels are interiors of even64-pixel checkerboard tiles; odd tiles are the same-image spatial holdout. Each tile excludes an8-pixel perimeter, leaving17 pixels between adjacent training and holdout interiors. Descriptor support is at most12 pixels. Shared global unlabeled image normalization still couples the regions; this is correlated spatial holdout, not independent validation.

The control averages the **same actual12-channel descriptor** inside each16-pixel block before the same fit and query application. Both arms share reference pixel IDs, labels, loss, penalty, parameter count and the normalization estimated from the real descriptor in the unlabeled reference training region. This tests actual within-block appearance variation rather than more parameters, additional reference supervision or a denser resize of the same coarse values. The implementation verifies exact paired identity when the input descriptor is already block constant.

The primary outcome is foreground/background pair concordance **within the same16-pixel block**. The constant control ties every pair, exactly.5. This removes inter-block appearance differences, but a positive pixel score can still be a generic boundary cue. Full-image ranking and ranking within the identical saved complete FoRIS foreground/background regions distinguish that result from task identity transfer. The same quantities on reference spatial holdout expose source discrimination that fails to transfer. Class-balanced logistic NLL is meaningful only for the probe logits; FoRIS's unnormalized score is compared by ranking alone.

No probe threshold is selected and no new binary segmentation output or mIoU is generated. Results cannot by themselves show that adding an appearance term to FoRIS helps: source/query calibration, disagreement resolution and semantic conditioning would remain separate falsifiable requirements. [JAFAR](https://arxiv.org/abs/2506.11136v3) already aligns high-resolution low-level image features with low-resolution semantic features using attention and Spatial Feature Transform. [AnyUp's official implementation](https://github.com/wimmerth/anyup) takes high-resolution RGB and low-resolution encoder features to produce dense features, including encoders unseen during its training. [FeatSharp](https://arxiv.org/abs/2502.16025v2) also targets fine detail in upsampled vision features. Thus ordinary RGB guidance, convolutional detail recovery or semantic alignment alone would not establish a new contribution; none of these models is imported here.

Even dimensionality does not establish that raw O24 lacks pixel information: the cached64×64×1024 array has4,194,304 scalars, versus3,145,728 RGB values at1024². This count proves neither injectivity nor useful label recoverability. The observable failures establish insufficiency of the **tested supervision/readout/transfer rules**, with calibration and identity failures remaining separate from encoder information loss. Positive pixel evidence here would show a lawful measurable cue, not conditional information independence from O24.

## Frozen artifacts

- [Protocol](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/protocol.json): SHA256 `df4a70fd68a5f8612b25d832da73fef5308914fb1d1364e833297734fd2e3d0d`.
- [Freeze receipt](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/FROZEN.json): frozen source SHA256 `6fa3e2861f0b838d91335516a21e0796af078b03199c4cb0b4c2f1e272ef1fc7`, tasks SHA256 `2a5ac31402e0d3cbbe77915677dae3bb5a77b016c4d697e1e485abead836b412`.
- [Synthetic validation](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/synthetic_validation.json): oracle, exact tie-aware ROC, capacity identity, and official PIL bilinear resize parity.
- [Frozen executable](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/frozen/diagnose.py): `infer` followed by `score`; workers1, CPUthreads2. Saved score fields and exact hashes permit independent replay without descriptor duplication.

## Measured transfer, CLI1024

The following AUCs are macro means across the retained episodes, without an independence or cross-dataset generalization claim. The constant appearance control has exactly.5 within-patch AUC in every valid pixel ROI. Foreground/background pair concordance uses actual pixel labels and exact average ranks for ties. All400 source fits had both roles; maximum fitted gradient residual was9.65e-9.

|Fixed12D pixel readout|Fundus200|Deep road100|PACO100|
|---|---:|---:|---:|
|Reference spatial holdout, whole-ROI AUC|.847355|.838981|.849924|
|Query whole-ROI AUC, real pixel|.810805|.644208|.586063|
|Query whole-ROI AUC, constant control|.771020|.678587|.583561|
|Same query whole-ROI AUC, full FoRIS score|.872589|.929907|.892547|
|Reference holdout, within16-patch AUC|.783331|.741904|.693299|
|Query, within16-patch AUC|.757331|.606679|.530218|
|Query FoRIS FG, within16-patch AUC|.777373|.583077|.505864|
|Query FoRIS FG, whole-ROI pixel AUC|.734894|.559947|.505149|
|Same query FoRIS FG, whole-ROI FoRIS score AUC|.784845|.683338|.747676|
|Reference holdout balanced NLL|.508557|.482577|.443121|
|Query whole balanced NLL|.568359|.829363|1.070237|

Fundus/Deep have200/100 valid cases in every listed ROI. PACO has100 whole cases,95 FoRIS-FG cases and96 FoRIS-BG cases because the remaining masks lack one role. The ROI support and FG/BG pair counts are identical between matched arms and when comparing their global ranking with FoRIS. FoRIS's field is bilinearly rendered in FP64 to the same1024 frame, then evaluated on the exact saved complete-FoRIS mask ROI. The Fundus whole-field AUC.872589 independently agrees with the stage-replay audit; it is not a new FoRIS run.

Fundus's within-patch query cue exceeds.5 in196/200 draws, with median.772160; its pair-count-weighted within-patch AUC is.774013 across2,151,773,417 FG/BG pairs. Inside FoRIS FG it also exceeds.5 in196/200 draws, with median.799063. The four adverse cases are retained, with whole-ROI within-patch AUC as low as.335101. These are200 draws from120 query photos and127 reference photos, not200 independent photographs.

Deep is much less stable:67/100 whole queries exceed.5 within a patch, but only59/100 do so inside FoRIS FG. Its pair-count-weighted whole within-patch AUC is.568555, lower than macro.606679; adding real pixel variation also **reduces** whole-query ranking versus the constant descriptor by.034379. Source holdout discrimination therefore fails to establish reliable query identity even when some boundary detail transfers.

PACO has65/100 whole queries above.5 within a patch, but only51/95 inside FoRIS FG. Its reference holdout AUC.849924 collapses to.586063 on whole queries and.505149 inside the semantic foreground. The constant control is similarly poor. This is direct source-to-query appearance-transfer failure under a lawful small classifier, with substantial cases below chance. It cannot be repaired by interpreting stronger source discrimination or fine reference masks as proof of transferable semantics.

The joint12-channel probe does not separately attribute the cue to color, contrast, gradient or texture, and positive within-patch ranking against a constant control does not prove conditional information beyond FoRIS/O24. No new channel/scale/head sweep follows this result. A complete integration would still need a fixed source-only rule that improves full outputs while preserving useful semantic classification when reference appearance fails to transfer.

## Restricted decoder geometry

These are **means of exact per-query GT-oracle IoU limits for block-constant decoders**, in percent; they are not benchmark mIoU, model predictions or attainable inference results.

|CLI1024 geometry|Fundus200|Deep road100|PACO100|
|---|---:|---:|---:|
|64×64 blocks,16 canonical pixels|41.111865|51.229292|90.032995|
|128×128 blocks,8 canonical pixels|58.835072|69.786524|94.507420|
|GT FG mass inside mixed16 blocks|99.632725%|91.765774%|13.042430%|

Fundus almost always places vessel pixels in mixed16-pixel cells. Finer cells materially change this restricted decoder's capacity, whereas PACO already permits a per-query mean90.03% block-constant oracle. Consequently a common resolution-only explanation of the current Fundus/PACO failures is unsupported. These numbers do not compare directly with class/fold-averaged PACO mIoU or pooled Deep I/U.

Original-frame mean limits are Fundus41.084236/58.709050 and PACO91.278746/96.715501 for64/128 blocks, retained separately; Deep originally is1024 and matches CLI. For non-divisible PACO dimensions, the original-frame block boundaries are the exact nearest expansion of the same64/128 partition, with unequal areas handled analytically. `pooled_union_weighted_chosen_episode_oracle` in the JSON is the I/U of individually optimized episode masks; it is **not** the optimum for a globally pooled dataset objective.

## Completion and reproducibility

- [All400 field seal](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/sealed.json): zero query-label attempts, no encoder/cache writes; infer281.850 seconds, one process/two CPU threads.
- [Full results](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/results.json) and [episode metrics](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/episodes.json): all400 frozen reference/query identities, original/CLI geometry, exact ROI ranking, NLL and pair-count denominators. Scoring436.937 seconds. No source or parameters changed after freeze.
- [Descriptive distribution](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/descriptive_distribution.json): natural-chance counts, medians and adverse cases from the completed fixed metrics, with no rule/threshold fitting.

[Independent actual Deep0/PACO100/Fundus200 replay](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/scene_model_validation/physical_pixel_actual_replay.json) passed all four saved score fields, both13-parameter coefficient vectors, fit diagnostics,8192 occurrence IDs/labels and common normalization exactly. An independently reconstructed split has294,912 training and294,912 holdout pixels, zero overlap and no sampled holdout pixels. Direct loss/gradient reconstruction passed. Query-GT/baseline reads and original artifact writes were guarded; all protected hashes stayed unchanged. [Author-run replay](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/physical_pixel_evidence/actual_replay.json) also passes separately and is explicitly labeled as such. Saved actual pixel score fields are ignored NPZ artifacts with all hashes sealed; descriptor arrays are not duplicated.
