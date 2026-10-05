# Native's errors as two goals, and every sealed arm read as two edit operators

Owner: Claude Code, 2026-10-05. Read-only CPU analysis of sealed masks; no encoder, no GPU, no new prediction.
COCO-20i 1-shot, seed 0, DEV241 (79 classes, 239 photograph groups), complete 1024 masks, class-summed mIoU,
paired against the replayed native of `gpu_multilayer_dev241_v1` (59.07), 2000 photograph-group draws, RandomState(0).
Query truth is read throughout: rows marked GT are budgets, not methods. All of it is development data.
Scripts and reports: [edit_budget/](edit_budget/) (inputs: the sealed `predictions/` of `gpu_multilayer_dev241_v1`
and `rcg241`, and `truth`/`native` of the cached packets). Arm scores agree with [the replay audit](native_replay_audit.md).

## What each goal is worth (GT)

A native component with under 10% truth is a whole wrong region; a truth component under 10% found is a whole missed object.

| Edit of native with truth | mIoU | Gain [95% CI] | Pixels | Episodes |
|---|---:|---:|---:|---|
| Delete every false pixel | 81.92 | +22.84 [+19.34, +25.16] | 9.21 M | |
| Delete whole wrong regions | 66.61 | +7.53 [+4.84, +8.92] | 4.20 M | present in 216; over 10% of the mask in 57 |
| Delete false pixels attached to the target | 70.49 | +11.42 [+9.23, +13.98] | 5.07 M | |
| Add every missed pixel | 72.38 | +13.31 [+11.30, +15.71] | 5.89 M | |
| Add whole missed objects | 62.54 | +3.46 [+1.94, +5.05] | 1.30 M | present in 59; over 10% of truth in 24 |
| Complete objects already touched | 68.92 | +9.85 [+8.27, +11.86] | 4.61 M | |

## Every sealed arm as a delete operator and as an add operator

`deletions only` is native ∩ arm; `additions only` is native ∪ arm. Native's pooled pixel IoU is 0.608, so on pooled
pixels deleting helps when more than 62.2% of the deleted pixels are false, and adding helps when more than 37.8% of
the added pixels are true (from I/U: an edit with t true and f false pixels moves IoU up iff t/f is below, resp.
above, IoU). The sign of all eight operators follows this rule.

| Arm | Deletions only | False share of deleted | Additions only | True share of added |
|---|---:|---:|---:|---:|
| RCG | +1.67 [+0.82, +2.64] | 80.1% of 2.67 M | +0.34 [-0.05, +0.70] | 44.6% of 1.83 M |
| Transition only (unit(l24 - l16)) | +1.42 [-0.24, +2.53] | 70.1% of 3.95 M | -0.16 [-0.87, +0.59] | 36.2% of 2.61 M |
| D (l24 with transition) | +0.54 [-0.41, +1.87] | 72.0% of 2.75 M | -0.34 [-0.78, +0.23] | 27.8% of 1.70 M |
| Concat l24, l16 | -0.98 [-2.51, +1.03] | 59.1% of 5.18 M | -2.28 [-3.34, -0.67] | 22.3% of 4.29 M |

Share of each error bucket that an arm repairs (pixels): wrong regions deleted / attached false pixels deleted / missed
objects added / touched objects completed — RCG 31% / 17% / 4% / 17%; D 21% / 21% / 4% / 9%; transition only
36% / 25% / 25% / 14%; concat 35% / 31% / 33% / 12%.

Fixed set rules on the sealed masks (no per-episode choice; many rules were read on the same data, so these are
descriptions, not selections): delete where D and RCG both delete +0.76 [+0.41, +1.19], 144 up / 85 down, 83.4% false;
delete where either deletes +1.48 [+0.24, +3.08]; add where both add +0.23 [+0.11, +0.47], 50.2% true; add where either
adds -0.21 [-0.79, +0.41], 34.7% true; majority of native, D, RCG +0.99 [+0.63, +1.44], 164 up / 73 down. None exceeds RCG.

## What a 20- or 50-episode screen resolves

| Arm | first20 | mini50 | All 241 | Random 50 of 241, fold-stratified (2000 draws): sd; share at or above +2; share at or below 0 |
|---|---:|---:|---:|---|
| D | +3.26 | +1.08 | +0.12 | 1.16; 8%; 36% |
| Transition only | +3.44 | +1.54 | +1.12 | 1.60; 22%; 32% |
| RCG | +0.53 | +1.31 | +1.94 | 1.11; 39%; 5% |

Random 20 of 241: sd 1.7 to 2.5. first20 ranked D above RCG by 2.7 points; on 241 RCG is above D by 1.83 [0.12, 3.08].
Measured cost of the 241 run: four complete arms with layer saving in 1109 s.

## A + B + auxiliary: each arm with its side effect reduced (GT ceilings)

`edit_budget/compose.py`. Additions lie outside native and deletions inside it, so the four pixel counts of two edit
operators add; in mIoU the sum of an arm's two halves is within 0.15 of the arm (RCG 2.01 vs 1.94, transition only 1.26
vs 1.12, D 0.20 vs 0.12). An auxiliary is modelled as removing a share of the side effect and none of the benefit.

| Arm as an adder (cohort) | Recovers, of missed pixels | True share | Gain now | Side effect halved | Removed |
|---|---:|---:|---:|---:|---:|
| RCG (DEV241) | 13.8% | 44.6% | +0.34 | +1.18 | +2.07 |
| Transition only (DEV241) | 16.1% | 36.2% | -0.16 | +0.64 | +1.55 |
| D (DEV241) | 8.0% | 27.8% | -0.34 | +0.27 | +1.04 |
| Concat (DEV241) | 16.3% | 22.3% | -2.28 | -0.68 | +1.78 |
| E, two-slot EM control (mini50; RCG there 8.0%) | 53.9% | 25.2% | -2.44 | +0.52 | +6.89 |
| E, fixed-slot control (mini50) | 35.2% | 14.0% | -6.77 | -2.80 | +7.45 |
| E, latent (mini50) | 19.8% | 20.3% | -2.05 | -0.29 | +3.47 |
| B, kernel control (first20, aggregate counts; RCG there 7.7%) | 69.6% | 25.1% | not computed | | |

| Arm as a deleter (DEV241) | Removes, of false pixels | False share | Gain now | Side effect halved | Removed |
|---|---:|---:|---:|---:|---:|
| RCG | 23.2% | 80.1% | +1.67 | +2.57 | +3.48 |
| Transition only | 30.0% | 70.1% | +1.42 | +2.99 | +4.56 |
| D | 21.5% | 72.0% | +0.54 | +1.71 | +2.89 |
| Concat | 33.2% | 59.1% | -0.98 | +1.92 | +4.82 |

A = transition-only additions with B = RCG deletions (DEV241): no auxiliary +1.32 [+0.22, +2.36]; both side effects
halved +3.14 [+2.20, +4.27]; removed +5.16 [+4.38, +6.62]; only A cleaned +3.35; only B cleaned +3.02. Union of the four
DEV241 arms with an ideal auxiliary: additions recover 30.2% of missed, +3.73 [+3.31, +4.64]; deletions remove 52.4% of
false, +8.81 [+6.95, +10.50]; both +13.21 [+11.50, +15.33].

The one auxiliary measured is agreement between two arms, and it is not selective: on RCG's additions it removes 80% of
the side effect and 75% of the benefit (+0.34 to +0.23); on RCG's deletions 72% and 65% (+1.67 to +0.76).

## Reading

- Observed: the deletion side holds +22.8 and the addition side +13.3; within each, the part attached to an object
  native already found (+11.4, +9.9) is larger than the whole-region part (+7.5, +3.5).
- Observed: every existing arm earns its gain by deleting. No arm adds above break-even except RCG, marginally.
  RCG's +1.94 is +1.67 from deletions.
- Not established: that any signal separates the buckets without truth, or that agreement between operators scales.
- Prepared next: label-free auxiliaries inside each edit set, [aux_evidence.md](aux_evidence.md) (not run).
