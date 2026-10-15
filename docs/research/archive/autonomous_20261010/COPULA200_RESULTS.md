# Copula200 saved-table information decomposition

2026-10-10. **The full table contains contextual label-unary effects that pure
δ discards, but the actual contextual field is weak and does not outperform the
fixed endpoint shuffle or existing DINO readouts consistently.** Full true copula
reduces natural-frequency pair losses while generally worsening four-state-balanced
losses. The evidence does not justify a new graph solver, calibrated confidence,
segmentation mIoU gain or cross-dataset adoption.

This independently analyses the already-sealed Deep100/PACO100 tables and the exact
same first200 scene600 scalar fields. No raw features, encoder, graph solve,
parameter fitting, retuning or segmentation-mask generation occurred. The two
panels were previously exposed; the analysis freeze precedes this auditor's new
query-GT access and is not an untouched preregistration.

## Frozen analysis and verification

The implementation is
`evidence/local/autonomous_20261010/copula_saved_analysis/analysis.py`.
`protocol.json` and the source copy were frozen before this run opened query GT.
The protocol specifies the exact formulas, masks, clipping, aggregation and proper
scores, and includes **626 input-file SHA identities**. Copula and main600 manifests,
task identities, APD branches and first200 episode IDs agree exactly.

- Source SHA: `846380738510a5a5a95fa9f9246b62a007e291e986b1fe1a4eb4aabdb0ae182a`.
- Protocol SHA: `07f21d243829264eb8fdfe34659471f037d020a98adc21e312677b8d15415d1e`.
- Results SHA: `f52befc41b7053621e117cb474fd212d5c9de5ffeab57ad0cc3dd5036fc229e8`.
- Episode ledger SHA: `188797908cb60073e37cdee72d759baac50b4781045e02a7473fb62a70b526e3`.

Six no-image algebra/probability/scoring fixtures passed before the freeze. Actual
analysis completed in10.353 seconds: **400 exact table decompositions,1,000 normalized
pair-probability tables,1,600 endpoint-marginal identities and200 GT array identities**
passed. `results.json` contains all macro means, valid n, edge-pooled metrics, paired
deltas, bank artifacts and limitations; `episodes.jsonl` contains every episode.

## What δ removes

Using BG=0, FG=1 and σ(y)=2y−1, decompose each already-saved table as

```text
g   = (L00 + L01 + L10 + L11)/4
h_i = (L10 + L11 - L00 - L01)/4
h_j = (L01 + L11 - L00 - L10)/4
δ   = L00 + L11 - L01 - L10
L_yz = g + h_i σ(y) + h_j σ(z) + (δ/4)σ(y)σ(z).
```

The scalar g cancels from normalized four-state probabilities. h_i/h_j are unary
in the labels, yet both are functions of the two observed query endpoint features.
Thus δ=0 means conditional label factorization at that pair; it does not mean
that neighbor features carry no identity information. A frozen-module no-GT
capacity example demonstrates this distinction in
`docs/research/autonomous_20261010/reference_boundary_copula_main_effect_addendum.md`.
It is a mathematical counterexample to an overbroad rejection, not actual benefit.

For the node diagnostic, average the appropriate endpoint h over each node's
actual incident4-neighborhood edges, dividing by degree2/3 at boundaries and4 in
the interior. This is a degree-normalized context score, **not** the sum of terms
in a product graph energy, an independent observation or a new segmentation head.
True, fixed shuffle and exactly-zero factorized controls share the recipe.

## Actual node discrimination

Each score remains native64; each node receives its exact foreground/background
pixel mass within canonical1024. FoRIS-FG/BG are the actual saved pixel masks,
not majority-token approximations. Exact score ties receive half AUROC credit.
All methods use the same ROI and valid episodes.

|Dataset / ROI; valid n|True context AUC / AP|Shuffle context AUC / AP|Zero AUC / AP|Support ridge AUC / AP|
|---|---|---|---|---|
|Deep whole;100|.518382 / .066251|.576723 / .076919|.500000 / .042992|.924887 / .442682|
|Deep FoRIS-FG;100|.544479 / .529873|.523922 / .505381|.500000 / .483048|.684069 / .627048|
|Deep FoRIS-BG;100|.517539 / .033860|.553594 / .034612|.500000 / .024810|.895280 / .181318|
|PACO whole;100|.461476 / .191860|.560424 / .237398|.500000 / .153463|.911053 / .715663|
|PACO FoRIS-FG;95|.462517 / .563774|.531817 / .603952|.500000 / .556115|.766019 / .769703|
|PACO FoRIS-BG;96|.519476 / .096688|.549565 / .077316|.500000 / .061775|.874880 / .262250|

Source-kernel AUC is.897194/.660120/.854777 for the three Deep ROIs and
.905883/.738934/.858289 for PACO. Scene AUC is.877172/.619185/.836199 and
.869047/.740623/.708216 respectively. These exactly reproduce the independent
main600 field diagnosis on the same first200 panel. Context does not supply an
identity readout competitive with those saved fields. Deep existing-FG has a small
true-vs-shuffle advantage, which does not transfer to PACO or to whole/BG ranking.

AP's baseline equals ROI prevalence, so a higher AP despite below-.5 AUROC does not
make a globally reliable ranker. No score direction was flipped after observing
GT. These are area-weighted64-node rankings, not original-resolution mIoU.

## Fixed conditional pair probabilities and proper losses

For every model use exactly the same fixed, lawful but **uncalibrated** proxy:

```text
p_i(FG)=clip((saved support_ridge_i+1)/2,1e-6,1-1e-6)
p_i(BG)=1-p_i(FG)
P_yz ∝ p_i(y)p_j(z) exp(L_yz).
```

Models are full true L, full fixed-shuffle L, factorized L=0, true pure-δ and
shuffle pure-δ, with pure-δ table `(δ/4)σ(y)σ(z)`. All probabilities normalize
over00/01/10/11; the δ-only terms are not selected by signs or made attractive.

Query truth uses exact patch coverage c and the soft product
`T_yz=role_c_i(y)role_c_j(z)`. This evaluates two independently sampled pixels from
the two cells; it is **not** the actual joint of spatially aligned neighboring
pixels or a pixel-boundary label. Pair ROIs reproduce the earlier frozen probe:
coarse FoRIS b means token coverage strictly>.5; FG-incident is b_i OR b_j and
border is b_i XOR b_j.

Raw NLL is `−sum_s T_s log(P_s)`. Expected categorical Brier is
`sum_s T_s (sum_t P_t²−2P_s+1)`. The auxiliary soft-target squared error differs
only by truth's irreducible constant and is retained separately. Four-state balance
averages each state's target-mass-weighted proper loss equally; it never changes
P or fits a prior. A per-episode balanced score is NA when a state mass is zero.
Negative differences below mean lower loss than the fixed factorized proxy.

|Full true minus factor0; raw n / balanced n|Raw ΔNLL|Raw ΔBrier|Balanced ΔNLL|Balanced ΔBrier|
|---|---:|---:|---:|---:|
|Deep all;100 /100|−.031872|−.016231|+.039412|+.023315|
|Deep FG-incident;100 /99|−.064775|−.031146|−.005076|−.000636|
|Deep border;100 /98|−.076081|−.036362|−.003084|+.002555|
|PACO all;100 /100|−.053683|−.033278|+.151759|+.083122|
|PACO FG-incident;100 /96|−.000383|−.004743|+.076302|+.045678|
|PACO border;100 /93|−.071589|−.045678|+.097337|+.057338|

|Full true minus shuffle; balanced n|Raw ΔNLL|Raw ΔBrier|Balanced ΔNLL|Balanced ΔBrier|
|---|---:|---:|---:|---:|
|Deep all;100|−.030254|−.015760|+.012749|+.008080|
|Deep FG-incident;99|−.088368|−.038374|−.029176|−.011088|
|Deep border;98|−.092926|−.043487|−.022151|−.007895|
|PACO all;100|−.059147|−.036223|+.133391|+.071933|
|PACO FG-incident;96|−.014009|−.002790|+.040984|+.035181|
|PACO border;93|−.103199|−.053232|+.069158|+.045045|

True pure-δ also improves all-edge natural loss vs factor0 but worsens balanced
loss: Deep ΔNLL−.007657 / balanced+.023020 and ΔBrier−.003317 / balanced+.014410;
PACO−.042081 /+.046603 and−.019336 /+.030473. Pure-δ remains insufficient after a
shared unary conditions the joint; its near-chance cut ranking was not merely an
omitted unary-free test convention.

Natural all-edge state masses are **BB93.10% on Deep and83.25% on PACO**. Dominant
state improvement can lower natural loss while harming rare00/01/10/11-balanced
risk. This weighting explanation is consistent with the measured sign reversal;
the results alone do not uniquely isolate confidence repair, state prior or KDE
estimation error.

Macro means and pooled losses answer different questions. The full true table's
**edge-pooled four-state-balanced loss is worse than factor0 in every ROI on both
datasets**, including Deep FG-incident (+.000940 NLL,+.001619 Brier) and border
(+.003438,+.004474). Therefore the tiny Deep macro improvement vs factor0 is not
stable across these declared aggregation choices. Both sets remain in the artifact;
none is silently chosen to favor a candidate.

## Marginal and self-pair artifacts

All four state banks in all200 episodes contain exactly64 occurrences; no missing
state independence fallback occurred. True pairs always have distinct physical
endpoints. Every shuffle preserves the sampled per-state left/right endpoint
histogram bitwise, confirmed independently1600 times. This does not assert
preservation of all4096 full-reference role marginals or source independence.

Average effective true physical-pair count is64 for BB,63.854 Deep FF,63.181 PACO
FB/BF and64 PACO FF. This avoids mistaking64 repeated occurrences for64 independent
relationships when duplicates arise. Average multiset change is90.625% for BB,
96.594%/96.063% for Deep/PACO FF and98.328%/98.125% for cross states. The unchanged
fraction is retained rather than removed by an outcome-based shuffle.

Shuffle self-pairs are permitted null artifacts: Deep FB/BF mean.03 per64,
PACO FB/BF mean.10 and PACO FF mean.12; BB and Deep FF means are zero. They can
change density contrasts but are not real reference neighbors. No bank, occurrence
budget, state definition or shuffle was repaired after this evaluation.

The support proxy clips an average98.95 Deep and251.79 PACO nodes out of4096, with
maxima1785/2217. This further rules out treating it as a calibrated probability.
The proxy and copulas reuse the same reference, and each state KDE uses its own
selected empirical endpoint marginals. Multiplying the two terms defines a valid
diagnostic distribution; it is not an established Bayesian factorization, a
source-independent likelihood ratio or evidence that all improvements are novel
relational information. Overlapping query edges also share nodes and evidence.

The defensible result is narrow: the mathematical contextual effect survives the
δ projection in principle, but this saved implementation supplies weak identity
ordering and inconsistent balanced conditional pair benefits. Preserve the full
table negative and positive diagnostics; do not advance this probe into a complete
segmentation claim based only on natural-frequency NLL improvement.
