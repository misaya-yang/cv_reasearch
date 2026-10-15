# Opus mechanism experiment plan: completion

Date: 2026-10-13. This completes the cached A/B diagnostics and the C label-qualification gate in [the experiment plan](../archive/MECHANISM_EXPERIMENT_PLAN_20261013.md). A/B reused sealed predictions and S-half labels; there were no new DINOv3 calls, no model construction, and no x1013 T scoring.

## A: output-stage and structure diagnostics

The structural results and paired photo-bootstrap intervals are in the [A/B diagnostic report](MECHANISM_DIAGNOSTIC_RESULTS_20261013.md). To add the planned class-pooled CLI-1024 metrics, this table sums intersection/union within each `(fold, class)` before averaging classes. FN/GT and FP/GT are pooled pixel ratios. These differ from the report's per-episode mean IoU.

| S-half domain | Output | class mIoU | FN/GT | FP/GT |
|---|---|---:|---:|---:|
| Fundus, n=108 | FoRIS | 19.01 | 27.91 | 279.13 |
| Fundus, n=108 | A4 | 34.37 | 54.22 | 33.21 |
| Fundus, n=108 | C1 | 34.37 | 54.22 | 33.21 |
| Road, n=62 | FoRIS | 29.49 | 51.75 | 63.61 |
| Road, n=62 | A4 | 29.69 | 50.07 | 68.16 |
| Road, n=62 | C1 | 29.27 | 51.82 | 64.58 |

Fundus C1 and A4 are identical. Road C1 lowers class mIoU by 0.42 pp and lowers centerline quality. The continuous-field skeleton-score diagnostics are reported in the A/B report; they show that many true skeleton locations fall below the fixed output cut, but do not establish that a threshold change alone would recover correct boundaries.

## B: reference-pixel correction and frozen selection metadata

R2 is the sealed 150-Fundus/100-Road comparison. No model was refit. The area metrics below are class-pooled; the structural deltas and photo-bootstrap intervals are in the A/B report.

| Domain | Output | class mIoU | FN/GT | FP/GT |
|---|---|---:|---:|---:|
| Fundus, n=150 | FoRIS | 19.01 | 28.12 | 278.09 |
| Fundus, n=150 | VP_identity / patch_constant | 34.05 | 54.37 | 33.99 |
| Fundus, n=150 | pixel | 36.80 | 51.68 | 31.30 |
| Fundus, n=150 | pixel_permuted | 34.17 | 54.26 | 33.87 |
| Road, n=100 | FoRIS | 29.38 | 54.05 | 56.40 |
| Road, n=100 | VP_identity / patch_constant | 30.28 | 50.32 | 64.08 |
| Road, n=100 | pixel | 30.22 | 50.39 | 64.15 |
| Road, n=100 | pixel_permuted | 30.30 | 50.29 | 64.05 |

The alpha and fit values below are copied from the sealed reference holdout records. The last column gives descriptive `pixel - VP_identity` deltas in percentage points within each selected-alpha group; alpha was selected from reference holdout data, not query GT.

| Domain | Episodes / query photos | Selected alpha counts `(0, .25, 1)` | ΔIoU / ΔclDice at alpha 0 | at .25 | at 1 |
|---|---:|---:|---:|---:|---:|
| Fundus | 150 / 101 | 0 / 1 / 149 | 0 / 0 | +0.19 / −0.38 | +2.64 / −0.15 |
| Road | 100 / 100 | 13 / 36 / 51 | 0 / 0 | +0.13 / −1.45 | −0.11 / −7.91 |

On Road, clDice is negative in every selected-reference-TP quartile (−8.65 to −2.71 pp) and every query mixed-block quartile (−5.73 to −3.00 pp). Fundus has a positive IoU delta in every quartile, but clDice does not change monotonically with reference holdout TP or mixed-block count. These are post-hoc descriptions, not a selector. The machine-readable evidence includes all quartile groups, deltas against the permutation control, and median/p90 fit statistics.

## C: label qualification for relative geometry

| Dataset, S half only | Task rows | Query photos | Side-specific target labels | Same-parent side competitors | Decision |
|---|---:|---:|---:|---:|---|
| PASCAL-Part | 1,091 | 711 | 0 in the current 56 target classes and 0 in matched merged parent annotations | Not represented by the current merged labels | Stop |
| PACO | 1,461 | 393 | `left_button`: 0 target rows; `right_button`: 14 rows across 3 distinct query parents | 2 of those 3 right-button parents have a left-button mask; both pairs touch under 8-connectivity and have 0 overlapping pixels | Stop |

The original [PASCAL-Part taxonomy](https://cs.stanford.edu/~roozbeh/pascal-parts/pascal-parts.html) includes side-specific parts, while the current prepared task labels merge them into names such as `LEG`, `WINGS`, and `ARMHAND`. PACO's [official category list](https://github.com/facebookresearch/paco/blob/main/paco/data/datasets/paco_categories.py) maps categories 2252 and 2313 to the mouse's left and right buttons. Here, only two independent S query photos contain the same-parent competing button masks, and there is no S-half left-button target cohort. That is insufficient for the planned fixed side-identity comparison. The model arm was therefore stopped at its predeclared eligibility gate; no left/right performance claim is made.

The S-only audit hashes and per-parent PACO records are in [compact evidence](../../../evidence/research/x1013_mechanism_completion.json). The reproducible postprocessor is [`complete_mechanism_diagnostics.py`](../../../scripts/x1013/complete_mechanism_diagnostics.py); large inputs and its full output remain under `../cv_data/a/x1013_mechanism_diagnostics/`.

## Decision

A/B are complete. They do not support a topology-recovery claim: Fundus pixel correction raises IoU without a reliable clDice gain and increases fragmentation; Road pixel correction and C1 both lower centerline quality. C is complete as a qualification audit and fails the available-cohort gate, so no relative-geometry model arm should be run on this split. No further experiment is required by this plan.
