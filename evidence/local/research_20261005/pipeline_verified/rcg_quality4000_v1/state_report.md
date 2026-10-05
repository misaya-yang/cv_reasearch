# Exact RCG state and quality accounting: 4,000 sampled draws

P = pre-CRF FoRIS mask, N = complete native, R = complete RCG. All results are post-hoc GT diagnostics; all sampled draws and photo dependencies are retained.

The last two batches lose 1.043 pp of RCG gain relative to the earlier five. The largest negative signed changes are less FP-removal benefit (-0.721 pp) and more TP-deletion harm (-0.478 pp). New-FP harm improves (+0.256 pp), partially offsetting them. TP-addition change is small with a CI crossing zero. Native postprocessing gain changes by only +0.053 pp, also with a CI crossing zero.

## Exact stage identity

`(R − N) = (R − P) − (N − P)` uses differences of paired class-summed macro mIoU, not pooled pixel IoU. The native-minus-pre term describes native CRF/postprocessing; the RCG-minus-pre term includes RCG's complete post-pre processing.

| Group | Draws | Pre | Native | RCG | RCG − pre [95% CI] | Native − pre [95% CI] | RCG − native [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 4000 | 60.411 | 60.932 | 62.334 | +1.923 [1.615, 2.222] | +0.521 [0.456, 0.605] | +1.402 [1.094, 1.686] |
| official0 | 600 | 60.774 | 61.335 | 63.089 | +2.315 [1.679, 3.069] | +0.562 [0.441, 0.838] | +1.754 [1.056, 2.394] |
| official1 | 600 | 61.779 | 62.585 | 64.672 | +2.893 [2.108, 3.779] | +0.805 [0.647, 1.113] | +2.087 [1.248, 2.889] |
| official2 | 600 | 57.818 | 58.305 | 60.223 | +2.405 [1.582, 3.309] | +0.487 [0.345, 0.718] | +1.918 [1.030, 2.809] |
| official3 | 600 | 59.965 | 60.584 | 62.119 | +2.154 [1.380, 3.017] | +0.619 [0.460, 0.814] | +1.535 [0.738, 2.355] |
| official4 | 600 | 59.234 | 59.812 | 61.429 | +2.195 [1.570, 2.966] | +0.578 [0.407, 0.844] | +1.618 [0.945, 2.371] |
| official5 | 600 | 61.427 | 62.112 | 62.757 | +1.330 [0.631, 2.079] | +0.685 [0.508, 0.883] | +0.646 [-0.063, 1.369] |
| official6 | 400 | 61.919 | 62.429 | 62.825 | +0.906 [0.381, 1.842] | +0.510 [0.331, 0.801] | +0.396 [-0.144, 1.251] |
| earlier0-4 | 3000 | 59.905 | 60.425 | 62.086 | +2.182 [1.831, 2.548] | +0.520 [0.449, 0.623] | +1.661 [1.301, 2.000] |
| later5-6 | 1000 | 61.279 | 61.852 | 62.471 | +1.191 [0.722, 1.815] | +0.573 [0.469, 0.734] | +0.618 [0.142, 1.173] |

| Stage gain change: later − earlier | Change [95% CI] |
|---|---:|
| RCG_minus_pre | -0.990 [-1.558, -0.252] |
| native_minus_pre | +0.053 [-0.088, 0.226] |
| RCG_minus_native | -1.043 [-1.597, -0.342] |

## Exact class-balanced four-way attribution

For each class, `R − N = (addTP − deleteTP + J_native × (deleteFP − addFP)) / U_RCG`. Sum counts within the group/class before taking each ratio, then average classes ×100. All four terms share the final RCG union; this is an additive identity, not four independently applied masks or causal effects.

| Signed contribution | Earlier | Later | Later − earlier [95% CI] |
|---|---:|---:|---:|
| add_TP | +2.293 | +2.192 | -0.101 [-0.367, 0.245] |
| delete_TP | -1.947 | -2.425 | -0.478 [-0.894, -0.049] |
| delete_FP | +3.171 | +2.450 | -0.721 [-1.140, -0.239] |
| add_FP | -1.855 | -1.599 | +0.256 [0.015, 0.461] |

Each band below uses the same class native IoU and final RCG union as the full identity, so the distance terms sum exactly to the overall attribution. Distances are to the nearest opposite GT label at 1024.

| Band | Δ addTP | Δ deleteTP | Δ deleteFP | Δ addFP | Total late − early [95% CI] |
|---|---:|---:|---:|---:|---:|
| GT_distance_le8 | -0.009 | -0.123 | +0.031 | -0.049 | -0.149 [-0.284, -0.065] |
| GT_distance_8to16 | -0.018 | -0.076 | +0.011 | +0.001 | -0.081 [-0.150, 0.013] |
| GT_distance_over16 | -0.075 | -0.279 | -0.763 | +0.305 | -0.812 [-1.326, -0.163] |

## Opportunity-normalized rates

Rates average class-specific event/opportunity ratios. addTP/nativeFN = recovery; deleteFP/nativeFP = removal; deleteTP/nativeTP = damage; addFP/nativeTN = new false positives. Each distance band uses matching-band opportunities. Zero-opportunity classes are omitted and eligible-class counts are retained in JSON. These rates have different denominators and are not additive mIoU components.

| Event | Earlier all-distance rate | Later all-distance rate |
|---|---:|---:|
| add_TP | 17.962% | 20.130% |
| delete_TP | 3.501% | 4.402% |
| delete_FP | 23.344% | 21.034% |
| add_FP | 0.543% | 0.456% |

| GT-distance band | addTP recovery: earlier / later | deleteTP damage: earlier / later | deleteFP removal: earlier / later | addFP damage: earlier / later |
|---|---:|---:|---:|---:|
| GT_distance_le8 | 20.005% / 19.687% | 9.601% / 10.886% | 15.410% / 16.383% | 11.476% / 11.233% |
| GT_distance_8to16 | 18.747% / 18.245% | 4.269% / 5.118% | 20.207% / 21.809% | 3.055% / 2.771% |
| GT_distance_over16 | 17.501% / 22.238% | 2.250% / 3.012% | 26.332% / 24.958% | 0.374% / 0.289% |

## Fixed native-quality bins

| Native IoU bin | Earlier / later draws | Δ RCG − pre | Δ native − pre | Δ RCG − native | Δ addTP | Δ deleteTP | Δ deleteFP | Δ addFP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [0,.25) | 443 / 123 | -1.323 | -0.122 | -1.201 | +0.026 | -0.441 | -1.049 | +0.263 |
| [.25,.5) | 453 / 169 | -1.262 | -0.187 | -1.075 | +0.312 | -0.455 | -1.005 | +0.072 |
| [.5,.75) | 727 / 242 | -0.353 | +0.348 | -0.701 | -0.684 | -0.692 | +0.616 | +0.060 |
| [.75,.9) | 890 / 305 | -0.060 | +0.030 | -0.091 | +0.005 | -0.391 | +0.290 | +0.005 |
| [.9,1] | 487 / 161 | -0.092 | +0.157 | -0.249 | -0.091 | -0.112 | +0.064 | -0.109 |

Different quality subgroups can contain different classes; their macro contrasts are conditional diagnostics. Distance >16 does not prove an object-level effect or whole-object correction. Stage comparisons cannot isolate token evidence from rendering without the corresponding sealed alternatives. Bootstrap CIs are paired full-cohort connected-photo draws; late/early grouping was chosen after observing the batch readout. This is reused benchmark data, not independent confirmation.
