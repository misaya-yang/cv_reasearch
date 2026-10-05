# Frozen complete fine readout on original DEV241

all original241 DEV identities in original order; frozen1200 subset projection, not confirmation

| arm | class mIoU | vs model.raw_nn | vs native | vs rcg | vs mean.control | vs rcg64.control | vs fine.rcg16.control | vs insid3.release_crf.control |
|---|---:|---|---|---|---|---|---|---|
| native | 59.074825 | +16.171 [+13.734,+19.541] |  | -1.945 [-2.914,-0.983] | -1.605 [-2.706,-0.739] | -1.054 [-2.537,+1.158] | -2.426 [-3.554,-1.506] | +4.069 [+0.235,+6.395] |
| rcg | 61.019669 | +18.116 [+15.375,+21.593] | +1.945 [+0.983,+2.914] |  | +0.340 [-0.389,+0.751] | +0.891 [-0.221,+2.700] | -0.481 [-0.771,-0.386] | +6.014 [+2.099,+8.307] |
| mean.control | 60.680117 | +17.776 [+15.222,+21.556] | +1.605 [+0.739,+2.706] | -0.340 [-0.751,+0.389] |  | +0.551 [-0.498,+2.577] | -0.821 [-1.401,-0.133] | +5.674 [+1.938,+8.116] |
| rcg64.control | 60.128680 | +17.225 [+13.847,+20.679] | +1.054 [-1.158,+2.537] | -0.891 [-2.700,+0.221] | -0.551 [-2.577,+0.498] |  | -1.372 [-3.342,-0.302] | +5.123 [+0.442,+7.486] |
| fine.rcg16.control | 61.500661 | +18.597 [+15.871,+22.287] | +2.426 [+1.506,+3.554] | +0.481 [+0.386,+0.771] | +0.821 [+0.133,+1.401] | +1.372 [+0.302,+3.342] |  | +6.495 [+2.596,+8.966] |
| fine.rcg64 | 60.515912 | +17.612 [+14.236,+21.138] | +1.441 [-0.755,+3.052] | -0.504 [-2.354,+0.734] | -0.164 [-2.185,+1.038] | +0.387 [+0.096,+0.754] | -0.985 [-2.977,+0.221] | +5.510 [+0.819,+7.934] |
| model.raw_nn | 42.903888 |  | -16.171 [-19.541,-13.734] | -18.116 [-21.593,-15.375] | -17.776 [-21.556,-15.222] | -17.225 [-20.679,-13.847] | -18.597 [-22.287,-15.871] | -12.102 [-16.917,-9.722] |
| native.forward_replay.control | 59.074810 | +16.171 [+13.734,+19.541] | -0.000 [-0.000,+0.000] | -1.945 [-2.914,-0.983] | -1.605 [-2.706,-0.739] | -1.054 [-2.537,+1.158] | -2.426 [-3.554,-1.506] | +4.069 [+0.235,+6.395] |
| insid3.release_crf.control | 55.006020 | +12.102 [+9.722,+16.917] | -4.069 [-6.395,-0.235] | -6.014 [-8.307,-2.099] | -5.674 [-8.116,-1.938] | -5.123 [-7.486,-0.442] | -6.495 [-8.966,-2.596] |  |

All241 canonical native/RCG masks and old/new truth are bit-identical. Original-forward native replay differs by25pixels in two cases and is shown separately.
All own masks sealed before GT diagnostics; no fitting, new encoder, or efficacy-based subset.

## Edits from actual raw DINO origin

| arm | add TP | add FP | delete TP | delete FP |
|---|---:|---:|---:|---:|
| native | 4137295 | 3355996 | 1576922 | 11744438 |
| rcg | 4255369 | 2874408 | 1413136 | 12386602 |
| mean.control | 4233063 | 2960082 | 1561159 | 12427615 |
| rcg64.control | 4053523 | 2484639 | 1702574 | 12919237 |
| fine.rcg16.control | 4339188 | 2906241 | 1395004 | 12410593 |
| fine.rcg64 | 4127069 | 2503298 | 1658440 | 12917230 |
| model.raw_nn | 0 | 0 | 0 | 0 |
| native.forward_replay.control | 4137300 | 3356013 | 1576922 | 11744435 |
| insid3.release_crf.control | 3672517 | 2441148 | 3002669 | 12667482 |

Native-relative counts are separately retained in report.json and episodes.jsonl. An interval crossing zero is unresolved. This DEV view is bookkeeping of the unchanged frozen1200 recipe, not fresh confirmation.
