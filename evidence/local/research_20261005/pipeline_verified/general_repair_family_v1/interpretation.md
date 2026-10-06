# General repair branch: complete result

The permitted expansion did not improve the existing canonical optimum. The union selected the original canonical recipe globally and on every training-fold selection, in both strict12 and extended13. Every resulting selected global/held-fold mask is bit-identical to its original sealed canonical control.

| Setting | General global | Canonical global | General minus canonical, pp [95% CI] | General held-fold | Canonical held-fold | Held-fold difference, pp [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| strict12 | 62.990095 | 63.064818 | −0.074723 [−0.272749, 0.094675] | 62.900418 | 62.964699 | −0.064280 [−0.170681, 0.030128] |
| extended13 | 63.163237 | 63.238541 | −0.075303 [−0.111908, −0.049856] | 63.019999 | 63.102199 | −0.082200 [−0.230578, 0.056831] |

Global means one DEV-selected fixed recipe. Held-fold means four recipes, each selected using the other three class folds and applied to its held fold. The same 4000 sampled draws, including repeats, are retained. Paired intervals use 2000 RandomState(0) connected-photo draws. Global intervals are conditional on selection and not selection-adjusted. The extended bank retains the supplied sizecut arm's allfresh600 threshold-fitting exposure; held-fold recipe selection does not remove that upstream exposure.

The strict general global optimum is `fine.rcg64 OR conservative_delete`; the extended general global optimum is `conservative_delete OR fixed_size_cut_on_foldtemp_fine16.control`. All ten selected general recipes have an empty negative set. Allowing full-mask negative intersections did not beat these positive-only optima in this bank. This limits the tested all-general policy, not every possible repair rule or the features.

For the selected union, strict held-fold native gain remains +2.032958 [1.657151, 2.387729], but versus scalar graft +0.120162 [−0.103035, 0.319822] is unresolved. Extended global versus strongest complete sizecut is +0.089340 [−0.086422, 0.299029], and extended held-fold versus it is −0.047002 [−0.248330, 0.149315]; superiority remains unresolved.

## Exact finite solver and boundary

The original general policy is `(O OR positive masks) AND all negative masks`. Existing allowed origins are every bank mask. Any reduced nonempty, disjoint positive/negative producer sets U,V have a legal mapping: O is a member of U, positives are U without O, and negatives are V. If original U and V overlap, the output collapses to AND(V), representable by choosing a member of V as O and the remaining members as negatives. Therefore enumeration of nonempty disjoint U,V covers the original all-origin general branch exactly. A three-bit Boolean brute audit compared all 48 original syntactic recipes with 19 reduced recipes and found identical sets of 14 complete truth tables.

For each V, retain membership atoms with all V bits one. A subset-zeta transform over the remaining bits gives foreground counts as the V total minus atoms with every U bit zero. Integer TP/FP then give per-class I/U directly. New enumeration counts are 527345 for strict12 and 1586131 for extended13 per search, ten searches in total; no original canonical search was repeated. Near float64 maxima are compared as exact Fractions under a conservative rounding guard. The inherited canonical search retains its original float64 criterion and its cached selected recipes are rationally recounted.

Scope is **canonical union all-general**, not a family with independently mixed canonical/general policies on each deletion action. Producer count is only an outer output-row proxy; compound upstream dependencies and selection costs must not be interpreted as that count in seconds.

## Evidence

`report.json` and policy reports contain all same-bank controls, fold/batch scores, paired intervals, episode up/down/tie, and four native-relative edit counts with exact class-macro attribution. All 12/13 baseline I/U, complete new-mask I/U versus atoms, four-edit accounting, and original canonical global/held masks passed parity checks. Source membership histograms, packet/mask receipts, allowed-origin contracts, immutable code hashes, and old canonical seals are in `prepared.json`/`result_receipt.json`.

Remote root: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/general_repair_family_v1`. Full draw manifest is `manifest.json`; complete sealed masks are `strict/predictions/<drawkey>.npz` and `extended/predictions/<drawkey>.npz`, with six keys listed in `result_receipt.json`. Local compact files were verified by SHA256 in `fetch_verification.json`. No encoder or GPU was used, no features were deleted, and shared project records were not modified. The first preparation schema failure and its immutable snapshot/log remain preserved in the owned launch directory; the successful v2 only adapts legacy origin-contract metadata.
