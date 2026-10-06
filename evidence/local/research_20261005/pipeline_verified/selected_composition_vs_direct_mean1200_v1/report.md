# Complete selected methods versus DirectMEAN: exact original1200

Original1200 order, public batch/e/class/photos and packets are identical; no draw deduplication or recipe substitution. All original six-arm mask/I/U parity passes.

| Arm | Class-summed macro mIoU |
|---|---:|
| native | 61.612802 |
| rcg | 63.598020 |
| mean.control | 63.581592 |
| rcg64.control | 63.690334 |
| fine.rcg16.control | 64.015315 |
| fine.rcg64 | 64.051866 |
| fine.mean16.control | 63.959782 |
| scalar_graft.control | 63.956578 |
| selected.global12 | 64.342116 |
| selected.heldfold12 | 64.214803 |

| Fixed method minus comparator | Gain [paired95% CI], pp |
|---|---:|
| selected.global12 minus fine.mean16.control | +0.382334 [+0.165041, +0.590581] |
| selected.global12 minus native | +2.729315 [+2.115006, +3.320502] |
| selected.global12 minus fine.rcg16.control | +0.326801 [+0.100983, +0.503379] |
| selected.global12 minus fine.rcg64 | +0.290250 [-0.163696, +0.837778] |
| selected.global12 minus scalar_graft.control | +0.385538 [+0.170128, +0.591326] |
| selected.heldfold12 minus fine.mean16.control | +0.255021 [-0.066169, +0.560208] |
| selected.heldfold12 minus native | +2.602001 [+1.956335, +3.256510] |
| selected.heldfold12 minus fine.rcg16.control | +0.199488 [-0.063782, +0.430058] |
| selected.heldfold12 minus fine.rcg64 | +0.162937 [-0.209641, +0.621950] |
| selected.heldfold12 minus scalar_graft.control | +0.258225 [-0.060472, +0.560874] |

Original1200 connected-photo2000 RandomState0 intervals; fold intervals/four-edits/source hashes remain in JSON. Global and heldfold recipes retain their actual4000 selection provenance. Intervals crossing0 remain unresolved; reused1200 is not fresh confirmation or a replacement for the pending4000 DirectMEAN comparison.
