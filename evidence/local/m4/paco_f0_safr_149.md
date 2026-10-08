# M4 PACO-Part f0: fixed SAFR composition

development, reused PACO-Part f0, 149 episodes / 66 classes. Original-query-size pooled class mIoU. No confirmation scoring.

| Arm | mIoU | Δ FoRIS [95% CI] | Δ RCG [95% CI] |
|---|---:|---:|---:|
| foris.crf | 38.4287 | +0.0000 [+0.0000, +0.0000] | -0.6603 [-1.2569, +0.0191] |
| foris.pre | 37.7524 | -0.6763 [-0.8198, -0.5483] | -1.3366 [-1.9803, -0.6243] |
| mean | 38.7048 | +0.2760 [-0.3738, +0.8913] | -0.3843 [-0.7118, -0.0388] |
| mean.raw_fg_bg | 39.1767 | +0.7479 [-0.0432, +1.3151] | +0.0876 [-0.5421, +0.6250] |
| mean.safr_fg | 38.5040 | +0.0753 [-0.5849, +0.6701] | -0.5850 [-0.8984, -0.2378] |
| mean.safr_fg_bg | 38.8912 | +0.4624 [-0.2743, +0.9868] | -0.1979 [-0.7262, +0.1927] |
| model.raw_mean | 31.5136 | -6.9152 [-9.0904, -5.6123] | -7.5755 [-9.8237, -6.1031] |
| model.raw_nn | 33.8421 | -4.5866 [-6.8296, -3.0909] | -5.2470 [-7.5937, -3.5430] |
| rcg | 39.0891 | +0.6603 [-0.0191, +1.2569] | +0.0000 [+0.0000, +0.0000] |

SAFR FG-BG does not beat RCG or the simpler raw FG-BG control; not advanced to confirmation. The primary arm changes only MEAN’s guide; the raw FG-BG control uses the same graph, constants and finalizer.

Intervals: 100,000 paired bootstrap draws within fold/class, seed 0. Official repeats would retain their sampling weight. This interval is conditional on the given image pool; it does not claim independence of photographs or uncertainty over unseen photographs. Singleton strata are reported in the JSON.

The RSRM default ViT-B attention-branch selection/fusion was adapted to the current frozen ViT-L/24 layers. This experiment is not the complete public RSRM pipeline or the Pro shared-context algorithm.

Prediction seal: `4de7f4dd2502dab51ac16697fa016ff57ecfc5d7d74bf5ec46e3722a552ae56a`. Inputs, code hashes, predictions and timings: `outputs/m4/paco_f0_safr_v1`; baseline source snapshot: `outputs/m4/paco_f0_baselines_v1/source/`.
