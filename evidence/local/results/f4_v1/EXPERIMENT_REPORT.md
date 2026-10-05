# Document F4 validation, 2026-10-03

## Measured result

The seven-arm document experiment completed all241 previously examined COCO-20i val2014 episodes (seed0, four folds,79 observed classes). Same frozen DINOv3 ViT-L/16 at1024, complete public FoRIS, original-resolution class-summed IoU. Every complete native mask and original I/U matched the stored baseline exactly. Seven predictions were frozen before each query label was scored. No training, image pool, extra encoder call or downloaded asset was added.

Postprocessing exception, required by the compact specification: G deletes directly from the already-refined native final mask and does not run CRF a second time. The other replacement arms recompute their declared complete host paths. This is the previously validated G control, not a claim that all seven arms have identical postprocessing. B and G have positive point estimates; no new proposed construct passed the expansion criteria.

| Arm | Class mIoU | Paired delta over native, pp | Photo-connected95% interval |
|---|---:|---:|---|
| Complete native |59.1218|0|—|
| B: native anchor projected per FG |59.1984|+0.0765|[-0.6574,+0.6009]|
| C: full001 original hard-bank |57.7479|-1.3740|[-2.7001,-0.2965]|
| D: per-FG002 |58.1378|-0.9840|[-2.2808,+0.1035]|
| E: exposure/slot-matched global bank |58.1854|-0.9364|[-2.2210,+0.0755]|
| F: independent004 Part4 calibration |58.4305|-0.6914|[-1.3910,+0.2760]|
| G: existing query-core deletion |59.5778|+0.4560|[+0.2104,+0.7898]|

C−B=-1.4505[-2.6555,-0.3547]. D−C=+0.3899[+0.1573,+0.7491], but D−E=-0.0476[-0.3733,+0.3079]. No new primary passed the registered expansion gate; registered160 remains unopened. G reproduces an existing useful control, not a new contribution. These are development results, not the standard4×1000 benchmark or independent method confirmation.

## What the failure establishes

C relative to B has38,724 fewer true-positive and61,628 more false-positive original pixels (net). This is not merely excess deletion. Across987,136 stored Stage2 positions, C≤B and D≤C hold within declared FP32 tolerance; SF is bit-exact on241/241. Nevertheless final scores increase at162,401 positions for C−B and121,052 for D−C. Part4 recomputation, per-image normalization/cut and CRF do not preserve raw-score or mask monotonicity. This locates a concrete interaction in the complete intervention; it does not identify an individually wrong BG direction or prove that DINO lacks information.

D/E match independent BG-token exposure and prototype slots. Their actual aggregation counts, effective clustering cuts and unique direction counts differ;320/784 branches have different effective distinct-direction counts. Conditional organization has no established advantage over the matched global control.

The saved changed-pixel GT repair rows in summary.json are arithmetic diagnostics restricted to changed regions, not deployable methods or certificates of reachable information. Per-class error ledgers use U(c-a)+I(b-d); global pixel ratios are not mIoU proofs.

## Execution and evidence

First8 exact smoke:50.08s. Full resumed233 new cases:1161.72s, with the eight-case prefix reused. The outer batch ran1248.81s before shutdown after a later CPU aggregate error. Shared-server quality elapsed time is not formal isolated speed evidence. No full-token feature cache was created.

Evidence: summary.json (all aggregate metrics and comparisons, pointer/SHA to archived complete analysis), failure_ledger241_cpu.json, stage2_monotonicity_first8_cpu.json, source/manifest/freeze receipts and archived per-case I/U records. Source files: scripts/f4_experiment.py, tics/f4_conditional.py, tics/f4_calibration.py. Bootstrap seed4021,2000 photo-connected paired draws.

After the final dependency/PID/open-file check, the used prediction arrays and redundant raw text outputs were retired. Complete text evidence, original I/U, paired results, freeze SHA and source snapshots are recoverable from evidence_retired_20261003.tgz (274 members,13,163,076 bytes; local/remote SHA72badd0b0cd2565cf1d972b1899762a667689785213d4cef1c7d763d9d1a055a). Both copies were validated member by member. The removed paths occupied227,536,896 allocated bytes; concurrent statvfs free space increased227,540,992 bytes. Current SAM3/B dependencies and foreign assets were not touched. Receipt:cleanup_retired_20261003.json. Do not reconstruct this failed feature/prediction set.

## SAM3 continuation: all A/B stages completed

Actual FP32 inference/scoring completed ten fold0 engineering/development cases: visual70.9592 vs paired complete FoRIS66.4329, +4.5263[-16.5854,+25.6399]; true-category text75.1421 is privileged. This is not evidence of a stable gain. Legal semantic-component reference boxes differ from FSS-SAM3's published instance-box protocol.

The aggregate glob incorrectly included before-score recovery snapshots as live shards. CPU repair reads only canonical shard names and preserves all old errors, predictions and scoring. Original GPU source276a53 and repaired CPU aggregateac94e2 are separately recorded. The already-completed ten predictions are reused; remaining A/B stages have prepared27-file preflight and a finite guard. B reuses pooled-class supervised17 constants and is not zero-training. DEV241 and oldCONFIRM600 have already been examined.

All A predictions and scoring are now completed under the frozen v3 continuation. FP32 with TF32 off, legal semantic-component reference box, same 1008-square stitched canvas, original-resolution class-summed mIoU. This is an adaptation of the FSS-SAM3 public route, not an exact reproduction of its instance-box sampling. Both cohorts have been examined before; the 600-case cohort is not a new independent confirmation for this design.

| Cohort | Complete FoRIS | Legal visual SAM3 | Paired visual gain, pp /95% interval | True-category text, privileged |
|---|---:|---:|---|---:|
| Old DEV241, four folds |59.1218|62.5553|+3.4335[-2.7432,+7.0490]|72.5611|
| Old CONFIRM600, four folds |59.7830|61.0908|+1.3078[-2.4489,+3.5690]|73.3759|

Bootstrap:2000 draws over connected support/query photograph groups (239 DEV groups;561 CONFIRM groups), with exact complete episode/SQ coverage and original dimensions. The legal visual result does not establish a stable advantage over FoRIS. GT choice between the two complete outputs reaches70.0012 on600; this is a privileged selection diagnostic and does not establish an observable selector.

The DEV241 text-minus-visual diagnostic is +10.0058pp[+6.815,+14.538]. Observed text edits recover1,084,215 false-negative pixels and add365,198 false-positive pixels; an order-symmetric GT edit decomposition attributes+11.613pp to TP/FN edits and−1.607pp to FP edits. Thus the dominant observed gain is recovery, not background rejection. The reference box contains59.64% foreground on average and excludes18.66% of legal reference foreground on average. However, the partial correlation of box purity with text gain is only.034 after controlling category and reference/query foreground area; matched strata cover only28 episodes. These associations do not identify the cause of the text advantage or prove that full-mask prompting would recover it. Text supplies extra category information and is not a legal method row.

Inference/score source276a53 and repaired CPU aggregationac94e2 remain separately frozen. A_DEV reused its original10-case prefix and predicted231 new cases in the recorded187.165s shard stage; A_CONFIRM600 recorded460.501s. These include declared model/source work and are shared-server elapsed times, not isolated speed evidence. No encoder optimisation, new fitting or threshold search was performed. Compact paired reports are sam3_A_dev_report.json and sam3_A_confirm_report.json.

## Fixed-formula B: old confirmation600

The same17 constants averaged from four label-fitted fold models were used for every fold, without fitting or changing the renderer. Consequently this is pooled-class supervised calibration, not training-free or class-held-out few-shot segmentation. Original FP16 evidence-cache rounding is preserved; the live encoder/renderer is FP32 with TF32 off and complete original FoRIS refinement. All600 native original I/U records match. Query labels were opened only after the entire600 predictions were frozen.

| Original-resolution row | Class mIoU | Paired gain, pp /95% interval |
|---|---:|---|
| Complete native |59.7830|0|
| Registered fixed readout |61.4458|+1.6628[+0.8458,+2.5100]|
| Same readout restricted to deletion |60.7732|+0.9901[+0.2428,+1.8849]|
| Direct affine renderer, same computed logits |61.4458|+1.6628[+0.8458,+2.5100]|

Readout gains per fold:+1.7212,+2.3722,+1.0214,+1.5365. Connected all-role photograph groups561;2000 paired bootstrap draws,seed0.202 episodes improve,152 decline, and31 lose more than10pp. Thus the gain/fold prediction holds, while the prediction of fewer than25 such regressions fails. Net original-pixel edits remove1,196,238 FP and add246,119 FP, lose484,540 TP and recover456,476 FN; these counts supplement rather than replace class-mIoU. Probability clipping affects0 grid positions and direct-affine versus registered rendering changes0 pre-refinement working pixels. The source's retention gate (+1.5pp/all four folds positive) is passed; this retains a supervised component and establishes no novel method or unseen-host generalisation.

The600 prediction worker recorded1087.482s with590 new predictions and10 reused; scoring recorded30.286s. These are scheduling/shared-server records, not an isolated cost comparison. The stronger104k supervised head's previously reported63.33 and patch-level true-share oracles remain separate protocols; they cannot be relabelled as this formula's zero-training advantage or original-resolution oracle. Evidence:formula_B_confirm_report.json and the frozen v3 identity/freeze receipts.

## Final development reading and finite-queue decision

B_DEV241 also completed. Original-resolution complete FoRIS59.1218, fixed readout61.2074, gain+2.0856[+.7964,+3.3046]. Fold gains:+1.9855,+3.0465,−.0420,+3.4188;10 episodes lose more than10pp. Deletion-only60.8129,+1.6911[+.5121,+2.9528]; direct affine has the same final aggregate as the primary, but its pre-refinement rendering differs by1 working pixel in DEV (CONFIRM differs by0). Aggregate equality does not certify final pixelwise identity. All241 original native I/U records match; this is exposed development evidence and does not establish all-fold positivity on both cohorts. The prediction worker recorded439.039s.

The frozen C rule returns COMPLEMENTARY_FAILURES_REVIEW for A and RETAIN_SUPERVISED_FIXED_COMPONENT for B. On600,86 episodes succeed only with FoRIS and67 only with legal visual SAM3 under the registered IoU>.5 definition (153/600). This locates complementarity; it supplies no observable reliable selector. C launches no new stage. F4 remains unexpanded; the160 registered-scope photo-disjoint cases remain unopened. No new zero-training method or solid paper contribution has been established by this batch.

The complete v3 body finished in2341.595s (39.03min), below its2700s cap. This is elapsed execution time, not100% GPU duty or formal exclusive timing; CPU clustering/scoring and GPU encoding alternate. Earlier F4/CPU-failure batch elapsed1248.806s, preserved separately. All canonical final reports, freeze/source identities and per-case original I/U were exported before provider shutdown. Metadata and record archives: sam3_final_metadata.tgz180,992 bytes, formula_final_records.tgz198,785 bytes. Distinct A DEV/CONFIRM scored-record SHA values match their source reports.

AutoDL independently showed E69 xd1lnmgg7n-e1317191 as 已关机 at2026-10-03 18:58:17UTC. The exported guard snapshot predates shutdown and remains GPU_RUNNING as originally captured; final_resource_receipt.json stores the later platform proof. It does not infer billing stop from an SSH disconnect. Earlier E44/F83 sources are stopped; no instance was permanently released and foreign assets were not changed.

Source-level boundary: the visual interface maps the selected semantic component to a box, then [SAM3's fixed ROI encoder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/geometry_encoders.py#L635-L684) reads image features/box roles, not internal support-mask labels. At fixed images and identical selected box(es), mask-interior changes are unidentifiable to this interface. Appearance/context remain in the image features. This mathematical input limitation does not show that restoring mask roles recovers text's gain, and multi-component boxes already appear in [CG-ICS](https://github.com/Kakarot1103/CG-ICS/blob/8da845743f15c18ec490567fbd86dba58c8a388f/main.py#L189-L202). No next algorithm is selected from this observation alone.

Completion audit against the compact F4 specification and PLAN's original A/B/C/D handover: all selected seven-arm cases, both complete A/B cohorts, written C decisions, required reporting/state updates and final platform stop are evidenced. F4's registered no-expansion decision legally leaves160 unopened. C2 requires reporting complementarity and a separate card before any new prompt experiment; it does not append a run to this queue. C4's INSID3 check is still unperformed and is a condition before any future generality claim, not a completed result. No such claim is made. The supervised formula also uses three FoRIS-specific score channels; a future host transfer must define those inputs honestly, rather than inserting dummy values and calling it the same information.

The finite experiment/report/shutdown deliverable is complete. The research-paper objective remains unfinished. No automatic next GPU stage is authorised by this result or by a historical queue. Full100% GPU duty was not established; the operational queue ended and the instance was powered off instead of being kept on for analysis.

## Existing-checkpoint capability check, no GPU

The official builder's default geometry encoder omits a mask encoder, and its strict=False checkpoint loading discards unexpected-key reporting. Therefore source signatures and successful loading alone do not establish an available trained reference-mask pathway. A bounded existing-checkpoint metadata check was performed in no-card mode:1,465 tensors,1,156 detector keys,309 tracker keys,76 detector geometry keys, **zero geometry mask-encoder keys and zero keys containing mask_encoder**. All tensors were read on the meta device; no model was built, no label or inference was evaluated, CUDA remained uninitialised. Elapsed5.212s, peak RSS508,752KiB. Receipt: `../sam3_preparation/condition_keys.json`; checkpoint identity inherits the previously verified official SHA.

This rules out merely enabling an already-weighted reference mask_encoder in this fixed checkpoint/source configuration. Tracker parameters are not evidence of a reference-concept mask pathway. It does not prove that legal mask roles cannot be used by another construction, or that those roles explain the text/visual quality gap. No random mask module was added and no GPU method run followed. The terminal guard and smoke-candidate cleanup receipts were recovered during the same no-card administration; the original pre-shutdown snapshot remains unchanged.
