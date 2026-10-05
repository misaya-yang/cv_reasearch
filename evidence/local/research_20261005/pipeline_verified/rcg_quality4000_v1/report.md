# RCG baseline-quality analysis: all 4,000 sampled draws

Native 60.931741, RCG 62.333671: **+1.401930 pp, 95% CI [1.094, 1.686]**.
Mean episode gain is +1.188667 pp; this is a different estimand.

All sampled draws, including the natural repeated episode identity, are retained. Query-GT native-quality bins are diagnostic only. This cohort reuses benchmark/development data, not independent confirmation.

Observed gains shrink in the high-quality bins and become negative for native IoU >= .9. The quality-bin and class-frequency composition accounting does not explain the smaller gains in the last two batches; the loss of gain persists within strata. These observations do not identify a causal mechanism.

## Fixed native-quality bins

| Native episode IoU | Draws | Classes | Native macro | RCG macro | Macro gain [95% CI] | Mean episode gain | Error reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| [0,.25) | 566 | 77 | 11.112 | 12.358 | +1.246 [0.708, 4.017] | +2.371 | +2.67% |
| [.25,.5) | 622 | 77 | 38.069 | 40.678 | +2.609 [1.429, 3.580] | +3.039 | +4.90% |
| [.5,.75) | 969 | 80 | 63.493 | 65.095 | +1.602 [0.765, 2.409] | +2.003 | +5.49% |
| [.75,.9) | 1195 | 80 | 83.194 | 83.643 | +0.448 [0.101, 0.732] | +0.095 | +0.57% |
| [.9,1] | 648 | 75 | 93.147 | 92.744 | -0.403 [-0.691, -0.252] | -0.822 | -11.92% |

## Batch composition and paired gains

| Batch | Draws | Native macro | Macro gain [95% CI] | Mean native episode IoU | Mean episode gain | Bins: <.25 / .25-.5 / .5-.75 / .75-.9 / >=.9 |
|---|---:|---:|---:|---:|---:|---|
| official0 | 600 | 61.335 | +1.754 [1.056, 2.394] | 62.687 | +1.278 | 82 / 99 / 147 / 167 / 105 |
| official1 | 600 | 62.585 | +2.087 [1.248, 2.889] | 64.806 | +1.537 | 80 / 74 / 152 / 196 / 98 |
| official2 | 600 | 58.305 | +1.918 [1.030, 2.809] | 61.518 | +1.384 | 97 / 92 / 142 / 182 / 87 |
| official3 | 600 | 60.584 | +1.535 [0.738, 2.355] | 61.419 | +1.158 | 94 / 93 / 152 / 164 / 97 |
| official4 | 600 | 59.812 | +1.618 [0.945, 2.371] | 62.303 | +1.529 | 90 / 95 / 134 / 181 / 100 |
| official5 | 600 | 62.112 | +0.646 [-0.063, 1.369] | 64.230 | +0.591 | 73 / 100 / 141 / 182 / 104 |
| official6 | 400 | 62.429 | +0.396 [-0.144, 1.251] | 62.983 | +0.672 | 50 / 69 / 101 / 123 / 57 |

## Earlier versus last two batches

Earlier = official0–4 (3,000 draws); later = official5–6 (1,000 draws). This comparison is descriptive and was selected after the batch results were visible.

| Group | Native macro | Macro gain [95% CI] | Mean native episode IoU | Mean episode gain | Episode error reduction |
|---|---:|---:|---:|---:|---:|
| earlier | 60.425 | +1.661 [1.301, 2.000] | 62.546 | +1.377 | +3.68% |
| later | 61.852 | +0.618 [0.142, 1.173] | 63.731 | +0.623 | +1.72% |

Later minus earlier macro gain: -1.043 pp, 95% CI [-1.597, -0.342].

| Native IoU bin | Early / late classes | Earlier mean episode gain | Later mean episode gain | Earlier macro gain | Later macro gain |
|---|---:|---:|---:|---:|---:|
| [0,.25) | 74 / 57 | +3.002 | +0.102 | +1.371 | +0.170 |
| [.25,.5) | 77 / 63 | +3.399 | +2.076 | +2.863 | +1.788 |
| [.5,.75) | 80 / 72 | +2.147 | +1.569 | +1.864 | +1.163 |
| [.75,.9) | 80 / 75 | +0.061 | +0.196 | +0.392 | +0.301 |
| [.9,1] | 71 / 56 | -0.726 | -1.114 | -0.372 | -0.621 |

If the earlier group's aggregate episode error-reduction fraction were unchanged, the later group's headroom would correspond to +1.334 pp, versus the observed +0.623. The headroom change accounts for -0.044 pp under that descriptive reference, versus the actual -0.754 pp episode-mean change. This is arithmetic accounting, not a causal prediction.

Symmetric composition accounting concerns **mean episode gain**, not macro mIoU:

| Strata | Shared early / late draws | Later minus earlier | Composition component | Within-stratum component |
|---|---:|---:|---:|---:|
| quality_bin | 3000 / 1000 | -0.754 | +0.013 | -0.766 |
| class | 3000 / 1000 | -0.754 | -0.003 | -0.750 |
| class_by_quality_bin | 2759 / 988 | -0.534 | -0.019 | -0.514 |

Within-class quality standardization retains 2759/3000 earlier and 988/1000 later draws, 80 classes and 316 shared cells. Raw overlap macro gains: +1.303 / +0.588; common-quality gains: +1.251 / +0.554. Minimum shared-cell draw count: 1. These are descriptive weighted I/U ratios; no adjusted inferential CI is asserted.

## Interpretation limits

- Each batch includes the same 80 classes, equally weighted in the primary macro metric. Class-frequency composition can affect the episode mean, but cannot directly change the primary class weights. The sampled quality, object-size and union composition within classes can still vary.
- Same-bin subgroup macro means average the classes observed in each subgroup; early and late class sets can differ. The joint class-quality standardization uses shared support in all 80 classes.
- Error reduction = 100 × sum(episode DeltaIoU) / sum(1 − native episode IoU), excluding exact native IoU = 1. This ratio of sums avoids exploding per-example ratios near perfection. It is not class-macro error reduction and is not a pixel FP/FN correction rate.
- Negative DeltaIoU-versus-native-IoU correlation is mathematically coupled (native appears in both axes), and positive gain is bounded by 1 − native IoU. It cannot establish causation. Binning on query GT also cannot define an inference-time gate.
- The bootstrap uses the full cohort's connected support/query-photo components, retaining cross-batch and cross-bin photo dependence. Classes absent in a bootstrap replicate are omitted from its macro mean.
- Existing I/U counts reproduce the source report. No native-mask replay, encoder or annotation parity is newly established; I/U alone cannot separate recovered TP, deleted FP, erroneous deletions and new FP.
