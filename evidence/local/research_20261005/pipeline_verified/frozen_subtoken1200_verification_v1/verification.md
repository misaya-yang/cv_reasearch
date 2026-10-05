# Frozen subtoken1200: independent I/U verification

All six-arm scores, paired2,000 RandomState(0) connected-photo confidence intervals, four-fold/two-batch scores and episode up/down/tie counts reproduce the completed report; maximum difference 0 pp. All1200 native/RCG/MEAN counts also match the previously verified public4000 draws.

| Fine64 minus arm | Gain [95% CI], pp | Up / down / tie |
|---|---:|---:|
| native | +2.439064 [+1.581170, +3.237973] | 665 / 520 / 15 |
| rcg | +0.453846 [-0.059653, +0.905474] | 708 / 462 / 30 |
| mean.control | +0.470274 [-0.104917, +1.001042] | 688 / 482 / 30 |
| rcg64.control | +0.361532 [+0.312316, +0.461490] | 826 / 305 / 69 |
| fine.rcg16.control | +0.036551 [-0.520387, +0.457837] | 571 / 597 / 32 |

Stage arithmetic: fine64−native = (RCG−native) + (coarse64−RCG) + (fine64−coarse64).

+2.439064 = +1.985218 + +0.092315 + +0.361532 pp.

| Pair | Net TP contribution | Net FP contribution | Pooled net TP / FP change |
|---|---:|---:|---:|
| fine.rcg64 minus native | -1.316673 | +3.755738 | -807729 / -7759466 |
| rcg minus native | +0.476163 | +1.509055 | 978082 / -3569007 |
| rcg64.control minus rcg | -2.159782 | +2.252097 | -2377097 / -4285654 |
| fine.rcg64 minus rcg64.control | +0.365483 | -0.003951 | 591286 / 95195 |
| fine.rcg16.control minus rcg | +0.271809 | +0.145486 | 473701 / -128241 |
| fine.rcg64 minus rcg | -1.798039 | +2.251885 | -1785811 / -4190459 |
| fine.rcg64 minus fine.rcg16.control | -2.077206 | +2.113756 | -2259512 / -4062218 |

The exact two-term class-macro identity is `DeltaI/U_final + J_base*(U_base-U_final)/U_final`, averaged across classes. It separates net TP from net FP; it cannot recover all four edit categories.

Saved fine64 edits against RCG: {"add_TP": 1530909, "add_FP": 1484375, "delete_TP": 3316720, "delete_FP": 5674834}. Their net TP/FP changes match I/U sums exactly. Individual categories were not recounted from masks.

Fine64 versus coarse64 yields +591,286 net TP with +95,195 net FP. These are pooled side-effect diagnostics, not class-macro contributions or direct addition/deletion counts.

The full4000 readout keeps fine64 fixed and compares native, coarse64, MEAN and original fine16. A point win whose CI crosses zero does not resolve superiority; 1200's point>=2 with a CI below2 does not resolve stable>=2. Dataset reuse is not fresh confirmation.

Unavailable locally: prediction/packet/field replay, original manifest file, per-draw four-way fine edit counts, raw-DINO origin, GT-distance/object geometry. No such fields are inferred.
