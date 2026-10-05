# Complete-hypothesis bank versus verified RCG

GT-assisted inspection finds supplied masks that cover substantial target regions missed by RCG. This proves candidate availability only. It does not prove that the allowed reference/query inputs identify those masks or that the pretrained representation contains a usable semantic selection rule. The existing fixed selectors sometimes select a useful mask, but do not reliably beat RCG or establish a new cross-pose or cross-appearance recognition capability.

## Fixed comparison

All 120 episode identities and support/query photographs match. RCG's supplied masks were independently checked against its fields and local GT before this comparison. This diagnostic uses the single authoritative `hypothesis_selector120` bank: 960 hypotheses, frozen before its GT evaluation. No features were read; no candidate, selector, threshold, fusion, or gate was added. The aborted `complete_hypothesis120` run is excluded.

The predeclared deep-miss definition is RCG target recall at most 0.10. A substantial alternative reaches IoU at least 0.50. Newly recovered target pixels more than 16 working pixels from RCG foreground distinguish target acquisition from immediate boundary movement. These thresholds were written in `hypothesis_vs_rcg_plan.json` before this comparison.

## Capacity and actual selection

| Cohort | RCG | Default | Reference contrast | Scalar contrast | Exact same-bank class-IoU oracle |
|---|---:|---:|---:|---:|---:|
| Old 20 | 70.743 | 65.873 | 66.489 | 69.036 | 74.693 |
| New 40 | 61.482 | 56.086 | 56.777 | 54.691 | 67.089 |
| New 60 | 63.090 | 61.746 | 63.444 | 65.101 | 69.248 |
| Pooled 120, supplementary | 63.723 | 62.189 | 62.640 | 63.807 | 70.141 |

The oracle selects one supplied mask per episode to maximize class-summed IoU. It is not deployable. The per-episode best mask used for the case diagnosis can differ from that exact class-level optimum.

On new 60, the bank's episode-best mask beats RCG in 29 cases and loses in 31; median episode change is −0.350 percentage points. Its positive mean does not imply general dominance. On all 120, the actual reference-contrast selector is −1.083 points versus RCG, 95% photo-group bootstrap interval [−4.668, +1.532]. Scalar contrast is +0.084 [−4.086, +2.658]. Neither establishes superiority. Canonical bootstrap uses 2,000 RandomState(0) photo-group draws after numeric fold/e/c sorting, and recomputes class-summed I/U with absent classes omitted. For a singleton class, its repeat multiplicity cancels by definition. The earlier statistics.json instead used default_rng and an episode-weighted old20 special case: that is a different bootstrap estimand, not an equivalent repair. statistics_contract.json follows the current contract; predictions and point estimates are unchanged.

## Deep misses are a real but small subset

There are four deep-miss cases among 120: one in new40 and three in new60. Two of these four contain a supplied hypothesis with IoU at least 0.50. Both lie in new60; the existing reference/scalar selector finds only one.

| Episode | RCG IoU | Best supplied IoU | Reference contrast IoU | What happens |
|---|---:|---:|---:|---|
| `2_23_62` | 0 | 0.5973, seed 7 | 0 | Good target mask exists but every reported fixed selector misses it. |
| `2_35_78` | 0.0084 | 0.8602, seed 11 | 0.8602 | Reference and scalar contrast successfully select a target RCG largely missed. |
| `1_23_45` | 0 | 0.4771, seed 10 | 0 | Default and mean-dot retain the useful hypothesis; both contrast selectors choose an entirely wrong mask. |
| `3_46_47` | 0 | 0.2579, seed 16 | 0.1645 | Candidate recall reaches 0.9880 but precision is only 0.2587; acquisition remains too broad. |

For `2_23_62`, the best supplied mask adds 12,278 true-positive pixels, all farther than 16 pixels from RCG foreground, and 7,690 false-positive pixels. It removes all 77,408 RCG false-positive pixels. Its target recall is 0.9543. This is a concrete existence proof of target acquisition; the usable identity decision is still absent.

For `2_35_78`, the actually selected mask adds 156,887 true-positive pixels, including 154,272 farther than 16 pixels from RCG foreground. It also adds 18,212 false positives and loses 1,210 true positives. The gain is not just boundary smoothing.

The fifth low-overlap case, `1_46_69`, is different: RCG already has recall 0.9254 but precision 0.0745. The best supplied candidate reaches only 0.1568 IoU, largely by removing false positives. Low IoU alone therefore does not establish a missed target or a wrong semantic category.

Among the 29 new60 cases where the bank's GT-best mask beats RCG, the best masks add 838,364 target pixels; 651,080 (77.7%) lie farther than 16 pixels from RCG foreground. They also add 217,904 false positives and remove 119,864 true positives. This is a GT-selected diagnostic, not an estimate of deployable gains.

## Supported conclusion

Complete-mask target acquisition is a defensible remaining inference problem. The bank already contains two substantial alternatives for four severe RCG misses; one is found by current contrast and one is not. Yet those same selectors discard a useful default hypothesis in another severe case, and pooled performance does not beat RCG reliably. Whether the permitted inputs support a reliable rule for choosing these alternatives remains unresolved. Good-mask existence under GT selection does not prove that only the selector is missing, and failed selectors do not prove the representation lacks the necessary information. This comparison did not inspect RGB, introduce a new pretrained computation, or establish a new recognition capability.

GT components in the JSON are four-connected annotation components, never object instances. All cohorts have already been exposed during this research; they are diagnostic evidence, not fresh confirmation.

## Reproduction

`hypothesis_vs_rcg_diagnostic.py --index INDEX120 --bank HYPOTHESIS_SELECTOR120 --rcg-root RCG_RELEASE --capacity-report BANK_DIAGNOSTIC/report.json --out NEW_OUTPUT_DIRECTORY`

`hypothesis_vs_rcg_statistics_contract.py --diagnostic NEW_OUTPUT_DIRECTORY/report.json --out NEW_OUTPUT_DIRECTORY/statistics_contract.json`

The first command validates saved I/U and all joins, renders only existing selected masks, and writes both an early numeric checkpoint and the full spatial report. NumPy, SciPy, and CPU PyTorch are required. Use one BLAS/OMP thread. The statistics command reads only the completed JSON records.
