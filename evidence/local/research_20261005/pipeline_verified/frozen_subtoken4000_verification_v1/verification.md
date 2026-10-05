# Frozen subtoken4000 independent statistical verification

Inference metadata was read from `evidence/local/research_20261005/pipeline_verified/frozen_subtoken4000_v1`; CPU scoring outputs were read from `evidence/local/research_20261005/pipeline_verified/frozen_subtoken4000_scored_v1`. The files retain separate inference and CPU scoring origins, connected by the scorer receipt's seal/config/manifest hashes. The receipt declares inference source `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/frozen_subtoken4000_v1`.

All six-arm class-summed macro scores, stored contrasts/CIs and fold/batch point results match saved I/U with maximum error 0 pp. Global up/down/tie counts match. All4000 prior native/RCG/MEAN and all1200 earlier six-arm I/U match exactly; all sampled draws and natural repeated identities are retained.

| Fine64 minus control | Gain [95% CI], pp | Up / down / tie |
|---|---:|---:|
| native | +1.719876 [+1.256073, +2.113693] | 2198 / 1760 / 42 |
| rcg | +0.317946 [+0.039376, +0.562082] | 2342 / 1565 / 93 |
| mean.control | +0.138645 [-0.184598, +0.422923] | 2269 / 1634 / 97 |
| rcg64.control | +0.353680 [+0.321971, +0.396545] | 2778 / 993 / 229 |
| fine.rcg16.control | -0.057483 [-0.352600, +0.178003] | 1871 / 2030 / 99 |

| Arm | Class-summed macro mIoU |
|---|---:|
| native | 60.931741 |
| rcg | 62.333671 |
| mean.control | 62.512972 |
| rcg64.control | 62.297937 |
| fine.rcg16.control | 62.709100 |
| fine.rcg64 | 62.651617 |

| Subgroup | Draws | Fine64 | vs native | vs coarse64 | vs MEAN | vs fine16 |
|---|---:|---:|---:|---:|---:|---:|
| folds/0 | 1000 | 60.524 | +1.683 [+0.855, +2.602] | +0.416 [+0.360, +0.501] | +0.121 [-0.605, +0.767] | +0.062 [-0.558, +0.607] |
| folds/1 | 1000 | 65.054 | +1.546 [+0.564, +2.493] | +0.306 [+0.234, +0.396] | -0.145 [-0.768, +0.415] | -0.020 [-0.589, +0.489] |
| folds/2 | 1000 | 62.291 | +1.067 [+0.102, +1.806] | +0.313 [+0.224, +0.390] | -0.305 [-0.957, +0.242] | -0.704 [-1.306, -0.205] |
| folds/3 | 1000 | 62.738 | +2.584 [+1.643, +3.483] | +0.380 [+0.323, +0.453] | +0.883 [+0.324, +1.399] | +0.432 [-0.049, +0.892] |
| batchs/official0 | 600 | 63.681 | +2.346 [+1.031, +3.353] | +0.409 [+0.287, +0.619] | +0.750 [-0.382, +1.489] | +0.217 [-0.837, +0.867] |
| batchs/official1 | 600 | 65.148 | +2.563 [+1.199, +3.482] | +0.417 [+0.360, +0.596] | +0.329 [-0.638, +0.998] | -0.032 [-1.069, +0.414] |
| batchs/official2 | 600 | 60.830 | +2.526 [+0.977, +3.451] | +0.472 [+0.361, +0.662] | +0.506 [-0.769, +1.248] | +0.132 [-1.154, +0.684] |
| batchs/official3 | 600 | 62.662 | +2.077 [+0.740, +3.122] | +0.438 [+0.292, +0.619] | +0.376 [-0.700, +0.897] | +0.121 [-0.860, +0.631] |
| batchs/official4 | 600 | 61.805 | +1.993 [+0.526, +2.893] | +0.391 [+0.312, +0.562] | +0.320 [-0.868, +0.872] | -0.004 [-1.210, +0.443] |
| batchs/official5 | 600 | 62.542 | +0.430 [-0.691, +1.513] | +0.350 [+0.298, +0.498] | -0.641 [-1.506, +0.153] | -0.559 [-1.320, +0.007] |
| batchs/official6 | 400 | 61.783 | -0.646 [-2.143, +0.571] | +0.299 [+0.145, +0.593] | -1.292 [-2.743, -0.405] | -1.482 [-2.964, -0.733] |

| Fine64 signed edit contribution vs native | Class-macro contribution [95% CI], pp | Pooled count |
|---|---:|---:|
| add_TP | +2.368 [+2.132, +2.575] | 15790336 |
| add_FP | -1.424 [-1.556, -1.288] | 14686064 |
| delete_TP | -3.759 [-4.112, -3.495] | 19122340 |
| delete_FP | +4.535 [+4.224, +4.915] | 38582389 |

Fold/batch primary contrasts are independently recomputed using the same full-cohort connected-photo RandomState(0)2,000 multiplicities; details and source availability are retained in JSON.

Native-relative per-draw edit identities: all six arms passed. These checks do not replace a mask recount.

Fine64 remains the fixed primary. Evaluate stable>=2 against native and MEAN/coarse64/originalfine16 controls; intervals crossing zero remain unresolved. Full4000 is benchmark/development reuse, not fresh confirmation.
