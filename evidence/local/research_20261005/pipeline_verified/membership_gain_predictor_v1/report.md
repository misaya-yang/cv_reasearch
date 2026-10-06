# Fixed membership gain forecaster: backtest

This is one research-only posterior estimated from train query-GT, not a training-free segmentation component. All PUBLIC4000 was previously exposed and C12 was globally GT-selected before this diagnostic. Other-three-fold training excludes every photo-connected component touching the held fold. The smoothing is fixed Beta(1,1); unseen patterns use the global train foreground rate. No bins or tuning were added.

| Held fold | Train draws | Pairwise rank accuracy | Absolute-score MAE, pp | Predicted best | Actual best | Regret, pp |
|---|---:|---:|---:|---|---|---:|
| 0 | 2676 | 0.9615 | 0.585659 | strict.family12.frozen | strict.family12.frozen | 0.000000 |
| 1 | 2674 | 0.9487 | 3.938149 | strict.family12.frozen | strict.family12.frozen | 0.000000 |
| 2 | 2750 | 0.7949 | 6.187559 | strict.family12.frozen | strict.family12.frozen | 0.000000 |
| 3 | 2633 | 0.8205 | 0.257006 | strict.family12.frozen | frozen_subtoken4000_v1::fine.rcg64 | 0.198086 |

Gain-sign accuracy versus fine16: 40/48. Global C12 versus graft predicted +0.175811 pp; actual +0.220281 pp.

| DirectMEAN held fold | Predicted vs graft, pp | Actual vs graft, pp |
|---|---:|---:|
| 0 | -0.039692 | +0.010106 |
| 1 | -0.021930 | +0.000240 |
| 2 | -0.023808 | +0.001666 |
| 3 | -0.018691 | +0.000806 |

DirectMEAN had already been scored: every prediction is explicitly retrospective. Its tiny positive graft-relative gain has the wrong predicted sign in all four folds. Broad ranking success therefore does not establish reliable fine-grained winner/gain prediction.

Every fold prediction was saved before this script inspected held GT statistics. Unlabelled membership frequencies were independently reconstructed from all sealed predicted masks, not read from the GT-split histogram. Per-class E[I]/E[U], p-J(1-p) edit accounting, hashes and training keys are retained in each fold/prediction.json. The predicted marginal identity agrees to floating-point roundoff. Frozen eta files remain available for the forthcoming uniform-tau15 arm without refitting; source producer-level fresh600 label fitting remains explicit.
