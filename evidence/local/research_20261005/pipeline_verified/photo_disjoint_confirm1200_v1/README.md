# Photo-disjoint CONFIRM1200 CPU preparation

The actual immutable remote index is ready at
`/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/photo_disjoint_confirm1200_v1/`.
This preparation ran once on CPU as PID78865/start955406925 and completed in30.895s.
No encoder, GPU, image/mask pixel decoding, downloads, cleanup or method fitting occurred.
The existing strict12 global Boolean recipe is referenced unchanged by freeze SHA
`5fec4eca46825876d40617e8f7c53a38abfd4fc0716305014767cda0477f42c7`.

Actual result:1200 draws,300/fold,2249 unique photographs,zero photograph overlap with
17741 blocked photo IDs. Exposure is the conservative union of758 remote and112 local
paired query/support episode records and explicit COCO role records, including registered
training/development/evaluation/prepared episodes. It does not equate arbitrary image IDs
or entire class/image pools with consumed examples. The native-runtime base_class_index
is a bare class-to-photo pool; actual metric training/inference role manifests are included
separately. The previous broad-regex n0 preparation and all old features were retained.

The serial RandomState(0) choices preserve original class/pool order and distinct-reference
resampling. The first1000 draws/fold match all4000 recorded official-prefix episodes exactly.
Sampling continues at source e=1000 and accepts the first300/fold whose two photo IDs are
unblocked and whose existing JPG/annotation files stat as present. Selected photographs are
not added to the blocked set, so natural repeats are never removed. No query-GT area,
quality score or candidate outcome enters selection. Complete episode repeats happen to be0;
role photographs repeat naturally. The existing split-pickle SHA values and original sampler
source SHA are recorded, without claiming canonical public-pickle identity.

| Fold | Accepted | Last source e | Post1000 source draws | Rejected |
|---|---:|---:|---:|---:|
| 0 | 300 | 4258 | 3259 | 2959 |
| 1 | 300 | 3643 | 2644 | 2344 |
| 2 | 300 | 4003 | 3004 | 2704 |
| 3 | 300 | 4262 | 3263 | 2963 |

All12170 post1000 draws are retained contiguously in source_draws.jsonl, with class,
query/reference IDs, reference-resampling attempts and precise rejection reasons.
Total rejections10970; per-role reasons may overlap. All1200 accepted draw identities and
order match the manifest exactly. Every class is legal for its fold. Each receipt contains
all20 class counts per fold and the class's remaining legal metadata pool/pair count.

Observed classes74; absent32,35,68,70,78,79. Classes70/78 have no remaining legal unexposed
photos. Other absent classes still have legal pairs but none appeared among the fixed first300
accepted draws/fold. There was no class rebalance or episode splicing. Photograph rejection
changes class/image sampling: this is a separate independent confirmation cohort, not the
public first1000/fold80-class SOTA evaluation or an estimate over its unchanged sampling law.

All generated metadata hashes passed a separate read-only check. Exposure parsing has no
unresolved episode-metadata error: two non-UTF8 entries were verified as macOS AppleDouble
._ sidecars by magic00051607; their underlying JSONs were scanned. The original receipt keeps
these raw parse entries, and verification.json documents their resolution without rewriting
any immutable selection artifact.

Manifest SHA:
`21475d39973d448a81140b87db875dc70a47c4f60a53d436ccea8bc8432d3ff6`.
Receipt SHA:
`028e378c2c9eaa519c052395cbcaeccd4a2328b520a5f14276fc572520ec64f9`.
[manifest.json](manifest.json), [receipt.json](receipt.json) and
[verification.json](verification.json) are exact local metadata copies.
The full exposure, ordered pool and source-draw logs remain in the remote owned output.
The immutable CPU snapshot and launch state/log are at
`launch/photo_disjoint_confirm1200_v1/`; GPU confirmation launch belongs to the controller.
