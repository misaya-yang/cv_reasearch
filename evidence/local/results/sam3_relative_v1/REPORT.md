# SAM3 per-image relative proposal threshold — interim report

**Status (2026-10-04):** `relative_0.7` beats fixed global thresholds on the image-disjoint Fresh915 cohort, but its paired gain over `fallback_top1` is only +0.31 pp [−1.32, +2.40]. A query-only full-resolution second pass was also completed; its DEV/CONFIRM/rest0 results are heterogeneous and do not establish a stable incremental gain. External-pack segmentation has not run. The server was subsequently switched to no-card mode; no external prediction-status file exists. The two recorded run attempts were deferred by foreign GPU processes (details below), so there is no external mIoU or transfer claim.

## Question and protocol

Does selecting SAM3 visual proposals with a per-query threshold, `score >= 0.7 × max_query_score`, improve on the public fixed `score > 0.5` rule, without training or query labels? The coefficient was selected on DEV241 and frozen before reading CONFIRM600 and the fresh subset. The label-based proposal choice is a privileged diagnostic ceiling, not a method.

The measurements are COCO-20i, seed 0, class mIoU at original query resolution. Differences use paired bootstrap over connected support/query photo groups (2,000 draws, seed 0). The fresh 915 episodes exclude images appearing in DEV241 or CONFIRM600; no query annotation was used to choose proposals. Candidate predictions were frozen before scoring query labels.

This is the repository's adapted SAM3 visual-exemplar protocol, not an exact reproduction of the FSS-SAM3 benchmark loader or episode sampler. Its scores must not be placed in a direct leaderboard comparison with published numbers.

## Main results

| Split | Episodes | Fixed 0.5 | Fixed 0.3 | Frozen relative 0.7 | Relative vs 0.5 | Relative vs DEV-selected 0.3 |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 241 | 62.62 | 65.83 (+3.21 [0.82, 8.62]) | 70.84 | +8.22 [5.76, 12.50] | not reported as a paired contrast |
| CONFIRM | 600 | 60.85 | 63.63 (+2.78 [0.93, 6.12]) | 67.12 | +6.27 [4.37, 10.07] | +3.49 [1.29, 6.13] |
| Fresh | 915 | 63.23 | 66.53 (+3.29 [0.57, 5.69]) | 68.76 | +5.53 [3.18, 8.58] | +2.23 [0.70, 4.81] |

The frozen rule was positive on all four fresh folds: +3.77, +6.73, +5.77, and +5.84 points over fixed 0.5. It improved 212 episodes and regressed 35. It beats the DEV-selected 0.3 control by +2.23 points with a positive interval. A stronger diagnostic below shows that this does not establish an advantage over the simple nonempty fallback.

The privileged label-based upper bound was 80.19 on DEV, 81.52 on CONFIRM, and 81.54 on Fresh. It is not available at inference. The best fixed threshold on CONFIRM was 0.4 (64.56); that post-hoc row is not the preselected control and is not the primary paired contrast.

## CPU diagnostic: relative threshold versus fallback_top1

`fallback_top1` uses the same proposal scores and keeps only the top proposal when the fixed 0.5 rule would keep none. This was already included as a label-free control in the evaluation code. I joined the saved per-episode rule I/U sidecars to the frozen score records, then used the same connected-photo-group paired bootstrap as the main table; no image or query label was reopened.

| Split | Episodes | Relative 0.7 | fallback_top1 | Paired difference [95% CI] | Groups | Up / down |
|---|---:|---:|---:|---:|---:|---:|
| DEV | 241 | 70.84 | 68.69 | +2.16 [0.56, 5.05] | 239 | 41 / 23 |
| CONFIRM | 600 | 67.12 | 64.91 | +2.21 [0.09, 4.25] | 561 | 107 / 61 |
| Fresh | 915 | 68.76 | 68.46 | +0.31 [−1.32, +2.40] | 829 | 134 / 67 |

The fresh contrast is unresolved and near zero. `fallback_top1` by itself gains +5.22 [3.16, 7.52] over fixed 0.5 on Fresh, close to relative_0.7's +5.53. Therefore the positive fresh result against fixed thresholds does **not** establish that per-image score scaling adds value beyond the simple rescue of an otherwise empty proposal set. On CONFIRM the relative rule has a small positive contrast against fallback_top1, but that set was exposed to earlier analysis. The strongest current claim is that the frozen relative rule beats fixed global thresholds on these COCO episode sets; the specific scale-normalization mechanism remains uncertain.

## Paired comparison to FoRIS and the full SAM3 visual output

On the same 600 CONFIRM episodes, the saved FoRIS native I/U records join exactly to the SAM3 episode identities: 600/600 support/query image paths match, and FoRIS native I/U agrees with a separate frozen host audit. The relative rule and controls were then bootstrapped over 561 connected support/query photo groups (2,000 paired draws, seed 0).

| SAM3 top-20 rule | SAM3 mIoU | FoRIS mIoU | Paired gain vs FoRIS [95% CI] | Full SAM3 visual mIoU | Paired gain vs full visual [95% CI] | Fold gains vs FoRIS |
|---|---:|---:|---:|---:|---:|---|
| relative 0.7 | 67.12 | 59.78 | +7.34 [+4.83, +10.54] | 61.09 | +6.03 [+4.10, +9.88] | +4.63, +13.10, +5.58, +6.04 |
| fallback top1 | 64.91 | 59.78 | +5.13 [+2.48, +8.29] | 61.09 | +3.82 [+2.72, +6.97] | +4.01, +9.10, +4.19, +3.22 |
| fixed 0.3 | 63.63 | 59.78 | +3.84 [+0.83, +6.88] | 61.09 | +2.54 [+0.68, +5.85] | +2.69, +7.69, +2.38, +2.62 |

This is a paired result on the repository's adapted COCO-20i episode protocol; it is not an exact FSS-SAM3 or FoRIS leaderboard reproduction. The SAM3 rule operates on the saved top-20 proposal bank, while the full visual output unions every proposal passing its native threshold. Those two masks differ in 7/600 episodes. The comparison to FoRIS is same-episode; the relative-vs-full-SAM3 contrast is the more direct SAM3 control. CONFIRM600 has prior analysis exposure, so this is not a newly blinded confirmation.

## CPU diagnostic: where lowering 0.5 to 0.3 helps

I reconstructed predicted area exactly from saved original-resolution fields, `|A| = I + U - |Y|`, using the already-scored episode records. No annotation images were reopened. The two selective strategies choose based only on whether the fixed-0.5 prediction is empty; they are diagnostic controls, not query-label oracles.

| Split | Empty at 0.5 | All episodes switch to 0.3 | Switch only when 0.5 is empty | Switch only when 0.5 is nonempty |
|---|---:|---:|---:|---:|
| DEV241 | 55/241 | +3.21 [+0.82, +8.62] | **+6.21 [+3.96, +10.34]** | −2.76 [−4.35, +0.04] |
| CONFIRM600 | 134/600 | +2.78 [+0.93, +6.12] | **+5.04 [+3.39, +7.68]** | −2.00 [−3.34, −0.15] |
| Fresh915 | 174/915 | +3.29 [+0.57, +5.69] | **+5.00 [+2.87, +7.29]** | −1.40 [−2.83, −0.15] |

All differences are paired class-mIoU points against using fixed 0.5 for every episode, bootstrapped over connected support/query photo groups (2,000 draws, seed 0). On Fresh915, the 0.5 prediction was empty in 19.0% of episodes; changing the threshold only on those cases recovers more than changing it everywhere. Conversely, lowering to 0.3 only on already-nonempty cases hurts on CONFIRM600 and Fresh915. This supports a narrow diagnosis: much of the fixed-threshold gain is empty-output rescue, while indiscriminately adding candidates to nonempty masks is harmful. The split-wise diagnosis is post hoc and not a new independent confirmation.

## Interpretation and limits

- **Measured:** in this adapted COCO-20i SAM3 pipeline, the frozen relative rule beats fixed 0.5 on CONFIRM600 and the 915-episode image-disjoint subset, and beats the DEV-selected fixed 0.3 on both. On the same CONFIRM600 episodes it gains +7.34 [4.83, 10.54] over FoRIS native and +6.03 [4.10, 9.88] over SAM3's full visual output. The CPU decomposition shows 0.5->0.3 gains are concentrated in cases empty at 0.5; applying 0.3 only to already-nonempty predictions is harmful on CONFIRM and Fresh.
- **Not established:** the relative-rule confirmation has prior analysis exposure; fresh915 has no paired FoRIS result. The top-20 rule and full native SAM3 union differ in 7/600 episodes, and this adapted episode/inference protocol is not an exact published benchmark reproduction or leaderboard comparison.
- **Remaining error:** the proposal ledger shows that relative thresholding still keeps both on-target and off-target proposals in many cases. The ongoing backward check measures whether reverse prompting adds a distinct signal; it is not yet a task-level improvement.

## Follow-up queue state

The reverse-prompting diagnostic has now completed. On proposals retained by `relative_0.7`, candidate-level AUC was:

| Split | Forward SAM3 score | Best backward signal | Kept on/off-target proposals |
|---|---:|---:|---:|
| DEV241 | 0.776 | 0.709 | 381 / 91 |
| CONFIRM600 | 0.759 | 0.677 | 944 / 282 |

These AUCs pool proposals and have no episode-clustered interval; they are diagnostic, not task-level gains. On the full candidate set, backward union-IoU had AUC 0.772 on CONFIRM600, but after conditioning on proposals retained by the current rule, the best backward signal falls below the forward score. This does not support backward prompting as the next selector.

At the 2026-10-04 19:52 server-local snapshot, PID 1336 was alive and PID 14964 was running rest shard 2; rest shard 1 had completed with return code 0. GPU utilization was 100% with 7.3 GiB allocated. This was a historical snapshot before the later `sam3_second_boot.sh` run; its outputs must not be confused with the completed run below.

The first proposal-bank queue did not contain a DINO region-feature stage. The later `sam3_second_boot.sh` did: it completed region-feature extraction on rest shards 1 and 2 (2,106 episodes, 339.2 seconds, query labels unopened). A separate CPU run had earlier completed DEV241 features (`regions_dev_cpu.npz`, 241 episodes, 908 seconds, query labels unopened).

The backward smoke completed 10 episodes / 40 passes in 13.64 seconds (0.341 seconds/pass); backward DEV241 completed 1,129 passes in 446.36 seconds (0.395 seconds/pass); backward CONFIRM600 completed 2,823 passes in 957.57 seconds (0.339 seconds/pass).

## Anchor-expand proposal: code-level assessment, no method result yet

The proposed rule chooses a reference-conditioned anchor among SAM3 proposals, then uses cosine similarity between proposal-mean DINOv3 features from the same query image to add or filter regions. This changes proposal-membership inference and differs from the empty-output fallback. It can only select among the frozen top-20 proposals, so it cannot recover a target absent from that proposal bank.

The current `sam3_anchor_expand.py` is a canvas-resolution screen, not the original-resolution full-pipeline score. It searches about 55 configurations on DEV, reports every rule's label-scored result even when a frozen rule is supplied, and lacks a per-episode proposal-count-matched SAM3-score top-k control. Its candidate-level AUC pools correlated proposals and conditions on a correct top-score anchor, so it cannot establish whole-method benefit or robustness to wrong anchors. This mechanism remains unverified until one rule is frozen on DEV and tested on label-unopened episodes against `relative_0.7`, `fallback_top1`, and a same-count score-ranked control at original resolution with paired image-group uncertainty.

An earlier duplicate orchestration attempt was rejected by the GPU guard and produced no model predictions. Its `DEFER_FOREIGN_GPU` lines may be interleaved in the shared log, so rely on per-stage report and prediction-freeze JSON files rather than the log alone.

## External-set continuation requirements

CPU preflight found 299 usable episodes each for PASCAL-Part and PACO-Part and 290 for SUIM. The COCO runner is incompatible with these pack-local class IDs and cyclic folds, so a separate visual-only adapter is prepared at `scripts/sam3_external_relative.py`. Its finite GPU plan is `results/sam3_relative_v1/external_infer_plan.json`; the original CPU preflight verified the frozen official checkpoint, SAM3 source, and all three manifests/pack receipts without loading a model or opening query labels. A second state-file-specific CPU preflight also completed; its receipt is `results/sam3_relative_v1/external_infer_guard_retry2.preflight.json`. The adapter uses the existing support-mask visual exemplar, applies frozen `relative_0.7` without external tuning, and compares absolute 0.5, COCO-DEV-selected absolute 0.3, top-1 fallback, and full visual output. It hashes decoded support/query RGB for paired photo-group uncertainty; query annotations remain closed until requested-pack predictions are frozen. No model or dataset was downloaded.

The external GPU stage did not execute. The guard receipts `results/sam3_relative_v1/external_infer_guard.json` and `results/sam3_relative_v1/external_infer_guard_retry2.json` both record `DEFER_FOREIGN_GPU` (PIDs 15907 and 18685 respectively). The second run followed its matching CPU preflight. The instance was later switched to no-card mode; the final live check found no active SAM3 process and no `results/sam3_external_v1/external_infer_status.json`. Thus no external-pack prediction, paired interval, or transfer result exists.

During the same period, the separate `open_fast` queue produced COCO DEV241 and CONFIRM600 name-conditioned predictions in 456.5 s and 1,149.8 s; both logs report `query_annotation_opened=false`. These are prediction-only diagnostics, not scored external-set results. The later `open_lvis` CONFIRM log ends at 179/600 without a completion receipt. Neither queue supplies an mIoU for the external relative-threshold comparison. Do not describe the curated packs as broad dataset generalization.

An exact-RGB identity pass is now saved for the available transfer packs. It hashes decoded RGB pixels plus dimensions, opens no annotations, and connects episodes sharing an exact support/query image. This recovers exact duplicate photos, but may miss the same source photo if it was transformed differently before packing.

| Dataset pack | Episodes | RGB files | Unique exact RGB | Duplicate copies | Connected photo groups | Largest group |
|---|---:|---:|---:|---:|---:|---:|
| PASCAL-Part | 299 | 598 | 569 | 29 | 270 | 4 |
| PACO-Part | 299 | 598 | 468 | 130 | 196 | 8 |
| SUIM | 290 | 580 | 446 | 134 | 159 | 20 |

The available pack directory is `/root/autodl-tmp/demo9_extent/transfer_v1`; no LVIS or lung pack was present there at the check. These are pack-inventory and dependence-unit counts, not segmentation results. Use `scripts/analyze_external_pack_groups.py`; receipts are in `results/sam3_relative_v1/external_groups/`. The external SAM3 reader still needs a visual-only adapter that uses these group IDs for paired uncertainty and leaves the pack annotations closed until all predictions are frozen.

## Query-only full-resolution second pass

The completed `sam3_second_boot.sh` run tested whether the first pass's top-score query proposal can act as a box prompt for a second SAM3 pass on the full 1008×1008 query image. The selected second-pass masks were resized to the original query-canvas strip and unioned with that first-pass anchor. DEV chose among six frozen retention rules; the chosen rule was `anchor_plus_absolute_0.5`. It was then applied unchanged to CONFIRM600 and rest shard 0. All GPU prediction files report `query_annotation_opened=false`; CPU scoring followed each full prediction freeze.

| Split | Episodes / photo groups | First-pass relative 0.7 mIoU | Frozen second-pass mIoU | Paired gain vs relative 0.7 [95% CI] | Paired gain vs fallback_top1 [95% CI] | Four fold gains vs relative 0.7 |
|---|---:|---:|---:|---:|---:|---|
| DEV | 241 / 239 | 70.845 | 69.430 | −1.415 [−3.658, +1.193] | +0.742 [−0.541, +3.978] | +1.97, −6.09, −6.38, +5.18 |
| CONFIRM | 600 / 561 | 67.122 | 69.286 | +2.165 [+0.320, +3.577] | +4.374 [+2.503, +5.929] | +5.88, +0.95, +1.19, +0.64 |
| rest shard 0 | 1,053 / 945 | 69.133 | 69.686 | +0.553 [−1.052, +2.030] | +0.751 [−0.531, +2.313] | +1.13, −1.45, +1.25, +1.28 |

The rule loses on DEV, gains on CONFIRM600, and is close to zero on rest0. CONFIRM600 was already exposed to previous SAM3 readouts; rest0 had also been scored in the relative-threshold work. Neither is a new blinded confirmation. The measured evidence therefore does not establish a stable task gain, despite the positive CONFIRM interval. The DEV-selected rule was not chosen by its rest0 or CONFIRM result.

The same queue completed the 10-episode smoke (3.4 seconds), DEV241 second pass (84.5 seconds), CONFIRM600 second pass (211.6 seconds), rest0 second pass (373.6 seconds), and rest1/2 region features (2,106 episodes, 339.2 seconds). These are stage elapsed times reported by their scripts and exclude model-load time. All five stage logs ended in `COMPLETED`; all prediction stages state that query annotations were not opened during inference. The region-feature bank is 52 MiB.

One CPU readout attempt for rest0 initially failed because the analyzer mapped the label `rest0` to a nonexistent `sam3_keep_v1/rest0` directory. The path mapping was corrected to the existing `sam3_keep_v1/rest` shard-0 output, then rest0 was scored with the DEV-frozen rule. DEV and CONFIRM were not rescored in that retry. This was an analysis-path error, not a GPU prediction failure.

This experiment is an exploratory COCO-20i adapted-protocol result, not an independent benchmark result or comparison against published SAM3 scores. No external dataset was evaluated in this queue.

## Reproducibility files

- `scripts/sam3_relative_decision.py`
- `results/sam3_relative_v1/dev.json` and `dev.json.episodes.jsonl`
- `results/sam3_relative_v1/confirm.json` and `confirm.json.episodes.jsonl`
- `results/sam3_relative_v1/fresh.json` and `fresh.json.episodes.jsonl`
- `scripts/analyze_sam3_fallback_control.py` and `results/sam3_relative_v1/fallback_control.json`
- `scripts/analyze_sam3_foris_pair.py`, `results/sam3_relative_v1/sam3_confirm_uid_iu.jsonl`, and `results/sam3_relative_v1/foris_paired_confirm.json`
- `scripts/analyze_threshold_reduction.py` and `results/sam3_relative_v1/threshold_reduction.json`
- `scripts/sam3_second_pass.py`, `scripts/sam3_second_boot.sh`, and `scripts/analyze_sam3_second_pass.py`
- `results/sam3_relative_v1/second_pass_analysis_devconfirm.json` and `results/sam3_relative_v1/second_pass_analysis.json`; stage logs were copied from the server and prediction/candidate assets remain under server `results/sam3_keep_v1/{dev,confirm,rest}/second/`
- `results/sam3_relative_v1/regions_rest12.npz` (52 MiB; server feature bank for rest shards 1 and 2)
- `scripts/sam3_external_relative.py` (external visual-only adapter prepared; no inference or scoring receipts yet)
- `results/sam3_relative_v1/external_infer_plan.json` and `results/sam3_relative_v1/external_infer_guard.preflight.json` (CPU preflight completed; external model inference has not started)
- `scripts/sam3_anchor_expand.py` and `scripts/sam3_region_features.py` (proposed follow-up; no task result yet)
- `scripts/analyze_external_pack_groups.py` and exact-RGB group receipts under `results/sam3_relative_v1/external_groups/`
- The per-stage prediction and score receipts under server `results/sam3_keep_v1/`
