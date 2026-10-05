# Existing NN cue on RCG's remaining errors

The declared foreground direction of the existing `fg_max-bg_max` cue does not separate missed true regions from false detections, or deep FN from deep FP, on these fixed post-hoc cohorts. The residual labels are ordered in the opposite direction. This limits the tested cached max-NN cue in this setting; it does not establish that frozen DINO features, every use of reference evidence, or all recovery methods are exhausted.

The CPU-only run retained all4000 public benchmark draws,80 classes and four folds at1024 pixels. This is exposed COCO-20i benchmark reuse, not independent confirmation. Episode seed is not independently encoded in the input rows; bootstrap uses2000 RandomState(0) draws over2742 connected support/query photo groups. Sources are existing packet NN/FoRIS fields and sealed RCG fields/masks, with all native/RCG I/U matching the previous4000 ledger. There were no encoder forwards, parameter searches or inferred masks.

GT-pure tokens have16x16 target coverage>=0.9 or<=0.1. Their region/error membership is read at pixel index16*i+8. Whole regions are8-connected semantic GT components with zero RCG overlap, or8-connected RCG components with zero GT overlap; they are not instances. Deep errors are incorrect token centers more than16 Euclidean pixels from the opposite GT label. The two cohorts overlap and must not be summed.

| Fixed diagnostic | Whole missed/stray regions |Deep FN/FP |
|---|---:|---:|
|Eligible positive/negative tokens |28,063 /137,055|277,338 /360,880|
|Episodes eligible for paired AUROC |223 (209 photo groups)|1,088 (947 photo groups)|
|NN mean episode AUROC [95% CI] |0.253204 [0.215718,0.293471]|0.232535 [0.215564,0.249790]|
|FoRIS minmax-score AUROC [95% CI] |0.087015 [0.060607,0.115471]|0.095483 [0.084175,0.107310]|
|NN minus FoRIS AUROC [paired95% CI] |+0.166188 [0.128023,0.205925]|+0.137053 [0.120266,0.154465]|
|Existing NN>0 rule: pooled recall / FPR |23.30% /72.44%|32.60% /73.43%|
|NN fixed-rule mean episode balanced accuracy |0.342276 [0.314544,0.369776]|0.337447 [0.324744,0.349996]|

NN AUROC by fold0/1/2/3 is0.270550/0.222436/0.281553/0.244091 for whole regions and0.239448/0.255555/0.221728/0.209032 for deep errors. The improvement over the other failed score is real under this diagnostic, but NN remains below0.5 with intervals wholly below0.5. No sign inversion was tested or selected. A GT-defined error cohort cannot become an inference routing condition.

RCG-field AUROC is approximately0 on both cohorts. These cohorts were selected by errors of RCG's own rendered field, so this failure is substantially induced by conditioning; it does not show that the representation has no semantic information.

Of1,720 whole-missed GT regions,690 have an eligible pure token and421 have any token with NN>0:61.01% [55.70%,65.93%] of eligible regions. Of3,894 stray RCG regions,3,680 have an eligible token and2,931 have NN>0:79.65% [76.95%,82.19%]. Thus availability of positive NN evidence alone does not identify the true missing regions. The1,030 missed regions lacking pure tokens contain299,488 pixels; all679 missed regions smaller than256 pixels lack an eligible pure token in this cohort. These are explicit coverage limits, not zero-signal measurements.

The reported whole-missed and stray pixel masses exactly match the previous anatomy's9,361,369 and33,572,981 pixels. Native/RCG I/U parity passed for all4000 rows; stored AUROC eligibility and fixed-rule counts were independently reconstructed. All six reported AUROC intervals were reproduced both by the script's group-weight calculation and an independent direct resampling of group sums, with maximum CI discrepancy zero. This diagnostic does not propose or validate a new recovery component.

Code: `scripts/diagnose_rcg_remaining_cues.py`, SHA256 `e53176e00d3d37b97a510a9439e64097a6d4db354d8e979f43d2f832ec8d7f01`. Immutable remote snapshot: `launch/rcg_remaining_cues4000_v1_e53176e00d3d`. CPU4 producerPID43654/start953563975 completed in170.281 seconds. Existing SciPy runtime was reused; no dependencies were installed, no foreign job was changed, and no feature asset was deleted.
