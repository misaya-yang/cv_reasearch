# Status

Updated 2026-10-04. Replace, never append. At most 40 lines; history goes to the ledger in the direction's README.

## Now

- Live direction: `demo_lists/demo9_transductive_ics`. In-context segmentation; frozen matching (FoRIS) plus a
  class-free decision read-out. It holds the project's only confirmed positive result.
- **The one next action:** the GPU session `scripts/decision_boot_pivot.sh` (about 1 h, an estimate; powers
  off at the end). Plan and outcome table: the direction's `PLAN.md`. Owner: Codex.
- Server: `ssh -p 48002 root@connect.westd.seetacloud.com`, directory `/root/autodl-tmp/demo9_extent`.
- After it: read `results/decision_pivot_v1/pivot.txt` and take the decision `PLAN.md` wrote for that outcome.
- Held, not part of this session: the Pro regional batch (40 episodes) and the host reliability diagnostic.

## Measured

COCO-20i 1-shot, seed 0, CONFIRM600, class mIoU at original resolution in the complete pipeline; paired, 95% intervals.

| Method | mIoU | Gain over FoRIS | Column |
|---|---:|---|---|
| FoRIS (public code) | 59.78 | | training-free |
| SAM3, visual exemplar | 61.09 | +1.31 [-2.45, +3.57] | foundation segmenter |
| Fixed 17-constant formula | 61.45 | +1.66 [+0.85, +2.51] | constants fitted on base classes |
| Union of SAM3 and FoRIS masks | 62.83 | +3.05 [+1.82, +4.39] | naive control, two hosts |
| Decision read-out, 104k weights | 63.33 | +3.54 [+1.96, +4.96] | fitted on base classes |
| SAM3 with the true class name | 73.38 | | privileged, diagnostic only |

Published: INSID3 57.6, FoRIS 60.9, UINO-FSS 64.5 (trained), FSS-SAM3 66.1 (here 59.93 on 200 episodes; not reproduced).

## Unverified

- Whether the read-out can be fitted without any annotation (D1) and whether the COCO fit transfers to other
  benchmarks without refitting (T2). The next session reads both.
- The standard 4 x 1000 main table. Not run; not to be run before D1 and T2 are read.
- On a GPU: the pair generator's encoder path, the cache of constructed pairs, FoRIS on the transfer packs.

## Needed from the user

- E58 in GPU mode for the session above.
- A decision on the two held items once the session's result is in.
