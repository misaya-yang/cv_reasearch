# Native replay audit: complete DEV241

State: **native drift observed**. **public replay on discrepant cases still pending**.

Question: quantify the difference between the fixed cached-native packets and the sealed
tapped-native replay, and keep every candidate comparison explicit under both baselines.
Outcome: preserve and reuse the completed cohort; public replay on discrepant cases is
still required to locate the cause. Pixel drift is neither automatic invalidation nor
proof of full native identity. CPU mask analysis only; no encoder/model/GPU execution.

COCO-20i 1-shot, frozen DINOv3, 1024 masks; 241 episodes, 79 classes, 239 connected photograph groups. All cases are DEV, not independent confirmation. Seed 0 follows the current cohort record;
the exported source/RCG episode rows do not independently encode a per-row seed field.
Query GT was used only for class-summed I/U scoring and labeled four-action diagnostics,
after prediction/input seals were checked. No GT-based routing or mask choice occurred.

Drift: **2/241 episodes**, **25 pixels** overall. Maximum per-episode difference is 20 pixels (0.00190735% of 1024²).

Replayed minus cached native is **-0.000015361521 class-mIoU points**, paired 95% CI
[-0.000027228933, +0.000018050573], 1 up / 1 down / 239 ties. The changed pixels are
5 added TP and 20 added FP relative to the fixed cached baseline. The observed numerical
effect does not establish mask identity, and discrepant-case public replay remains pending.

| Complete sealed arm | Class mIoU | Gain vs cached-native [95% CI] | Gain vs replayed-native [95% CI] |
|---|---:|---:|---:|
| cached_native | 59.074825 | — | +0.000015 [-0.000018, +0.000027] |
| replayed_native | 59.074810 | -0.000015 [-0.000027, +0.000018] | — |
| multilayer | 59.192751 | +0.117926 [-0.828163, +1.581804] | +0.117942 [-0.828176, +1.581831] |
| multilayer.control | 55.448749 | -3.626076 [-5.184161, -0.892066] | -3.626061 [-5.184137, -0.892047] |
| multilayer.delta.control | 60.198027 | +1.123202 [-0.708533, +2.633790] | +1.123217 [-0.708518, +2.633805] |
| multilayer.delete_only | 59.610122 | +0.535297 [-0.414135, +1.867456] | +0.535313 [-0.414135, +1.867456] |
| multilayer.add_only | 58.738085 | -0.336740 [-0.779009, +0.234200] | -0.336725 [-0.779002, +0.234226] |
| rcg | 61.019660 | +1.944835 [+0.983141, +2.913596] | +1.944850 [+0.983156, +2.913614] |

The tested D/v1 has an unresolved pooled +0.118 gain, with arrived100 -1.703 and old20
-1.980 below either explicit baseline. It does not establish stable +2. Its complete masks
and retained layers can be reused; this audit launches no additional 241-case inference.

| Arm | Up / down / tie vs cached-native | Up / down / tie vs replayed-native |
|---|---:|---:|
| D multilayer | 131 / 107 / 3 | 131 / 107 / 3 |
| Concat control | 109 / 130 / 2 | 109 / 130 / 2 |
| Delta-only control | 127 / 111 / 3 | 127 / 111 / 3 |
| Fixed delete-only diagnostic | 126 / 109 / 6 | 126 / 109 / 6 |
| Fixed add-only diagnostic | 124 / 110 / 7 | 124 / 110 / 7 |
| RCG | 134 / 104 / 3 | 134 / 104 / 3 |

| Complete candidate/control contrast | Paired gain [95% CI] | Up / down / tie |
|---|---:|---:|
| D minus concat | +3.744002 [+1.456690, +5.354443] | 143 / 96 / 2 |
| D minus delta-only | -1.005276 [-1.824350, +0.611631] | 125 / 113 / 3 |
| D minus RCG | -1.826908 [-3.078843, -0.115249] | 116 / 122 / 3 |

## Folds: fixed scores and explicit baseline gains

| Stratum | n | cached / replayed native | Replay minus cached | D / concat / delta gain vs cached | D / concat / delta gain vs replayed |
|---|---:|---:|---:|---:|---:|
| 0 | 61 | 61.764597 / 61.764597 | +0.000000 | +0.086650 / -8.558278 / -0.277502 | +0.086650 / -8.558278 / -0.277502 |
| 1 | 60 | 61.746920 / 61.746948 | +0.000028 | -1.526341 / -5.314476 / -1.536326 | -1.526368 / -5.314503 / -1.536353 |
| 2 | 60 | 55.354542 / 55.354454 | -0.000088 | +0.400230 / +1.325350 / +4.366633 | +0.400318 / +1.325438 / +4.366721 |
| 3 | 60 | 57.346842 / 57.346842 | +0.000000 | +1.584495 / -1.869049 / +1.982992 | +1.584495 / -1.869049 / +1.982992 |

## Batches: fixed scores and explicit baseline gains

| Stratum | n | cached / replayed native | Replay minus cached | D / concat / delta gain vs cached | D / concat / delta gain vs replayed |
|---|---:|---:|---:|---:|---:|
| arrived100_exposed | 100 | 59.764071 / 59.764071 | +0.000000 | -1.702530 / -3.774504 / -0.516185 | -1.702530 / -3.774504 / -0.516185 |
| new40 | 40 | 58.861616 / 58.861616 | +0.000000 | +2.074827 / -2.822755 / +0.579789 | +2.074827 / -2.822755 / +0.579789 |
| new60 | 60 | 61.845266 / 61.845302 | +0.000036 | +1.292949 / -4.712179 / +1.540937 | +1.292913 / -4.712215 / +1.540901 |
| old20 | 20 | 67.143181 / 67.143181 | +0.000000 | -1.980431 / -5.779350 / -1.726024 | -1.980431 / -5.779350 / -1.726024 |
| remaining21_existing_dev | 21 | 64.493323 / 64.493259 | -0.000064 | +1.131871 / +4.075555 / +3.181637 | +1.131935 / +4.075619 / +3.181702 |

## All discrepant episode keys

Ratios below use the full 1024² image, not the foreground area.

| Key | Fold | Batch | Different pixels | Ratio | Replay minus cached episode IoU (pp) |
|---|---:|---|---:|---:|---:|
| 2_8_26 | 2 | remaining21_existing_dev | 20 | 0.000019073 | -0.001354121 |
| 1_37_1 | 1 | new60 | 5 | 0.000004768 | +0.001628452 |

## Four actions: scoring diagnostics only

| Arm | Fixed baseline | add TP | delete FP | delete TP | add FP |
|---|---|---:|---:|---:|---:|
| replayed_native | cached_native | 5 | 0 | 0 | 20 |
| multilayer | cached_native | 473684 | 1979278 | 768607 | 1229180 |
| multilayer.control | cached_native | 958229 | 3062403 | 2121716 | 3333886 |
| multilayer.delta.control | cached_native | 947304 | 2767410 | 1182300 | 1666005 |
| multilayer.delete_only | cached_native | 3 | 1979278 | 768607 | 0 |
| multilayer.add_only | cached_native | 473686 | 0 | 0 | 1229200 |
| rcg | cached_native | 813596 | 2136149 | 531733 | 1012415 |
| cached_native | cached_native | 0 | 0 | 0 | 0 |
| cached_native | replayed_native | 0 | 20 | 5 | 0 |
| multilayer | replayed_native | 473681 | 1979298 | 768609 | 1229180 |
| multilayer.control | replayed_native | 958229 | 3062414 | 2121721 | 3333877 |
| multilayer.delta.control | replayed_native | 947299 | 2767430 | 1182300 | 1666005 |
| multilayer.delete_only | replayed_native | 0 | 1979298 | 768609 | 0 |
| multilayer.add_only | replayed_native | 473681 | 0 | 0 | 1229180 |
| rcg | replayed_native | 813592 | 2136169 | 531734 | 1012415 |
| replayed_native | replayed_native | 0 | 0 | 0 | 0 |

## Integrity and remaining limits

- Source and RCG manifests, all 241 prediction SHA256 values for each run, and all 241
  original packet SHA256 values were checked. Exact source/RCG key sets and episode
  metadata agree. RCG config, D protocol and D arm-basis hashes were checked.
- Stored first-episode public-versus-tapped audit is reproduced in report.json; this is
  prior-run evidence. This CPU task does not execute any public replay.
- Raw-layer and input-feature SHA256 values are not reread: they are not scoring inputs.
  D raw score packets lack seal entries and are not scoring inputs either.
- Paired uncertainty uses the unchanged 2,000 RandomState(0) connected-photo draws from
  ics.experiment.summarize. The two baseline analyses use exactly equal draws.
- Cached-native and RCG absolute scores, and all bootstrap draws, match the sealed RCG241
  report exactly. Packed popcount I/U, XOR and four-action algebra agree with canonical
  unpacked Boolean counts on three independently generated full-size masks.
- Candidate/control pairwise contrasts, per-class/per-batch action counts, all episode
  I/U counts and full hash receipts are retained in the owned output directory.

Sources:

- root: `/root/autodl-tmp/demo9_extent`
- source: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/gpu_multilayer_dev241_v1`
- rcg: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/rcg241`
- out: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/native_replay_audit_v1/result`

CPU audit runtime: 11.147 seconds.
