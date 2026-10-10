# One actual reference focus: fixed added-information experiment

2026-10-10. Implemented head `src/ics/methods/reference_focus_head.py`, frozen SHA256 `37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413`. Parent owns actual reference encoding and complete200 inference/scoring. This track ran legal reference-mask and synthetic head checks only; no query GT or real head quality result has been read here.

The completed scene-basis600 and both pure-coupling/context-main-effect copula200 tests did not establish transferable target identity. No graph or parameter rescue is promoted. This experiment tests actual new reference observations, rather than a different weighting of those same failed query fields.

## Actual measured source premise

The published FROST source audit already found that its hard reference downsampler assigns35.487% of aggregate Deep reference foreground pixel mass to BG versus3.633% in PACO. That diagnoses source-label geometry, not DINO semantics or a query performance limit. Current whole/native4 ridge ordering has opposite view preferences in Deep versus PACO. Neither a shared source direction nor source-only threshold currently resolves that transfer.

One extra reference crop is selected using only the same legal mask, while retaining the whole-reference observation and parent context. It is a512×512 box in the canonical1024 reference canvas, with top-left coordinates aligned to16 pixels. Among all such locations, maximize the summed coverage refinement variance

```text
G(B) = sum_(16x16 cells in B) [mean_(four8x8 children) c8^2 - c16^2].
```

Break ties by foreground mass, then row-major position. This fixed rule targets source regions whose known mask granularity can benefit from2× finer encoded footprints. It does not choose by query scores, GT or dataset name. No reference pixel is masked or erased before encoding; the retained whole image supplies other foreground modes and background context.

The corrected actual mask-only200 audit is

`evidence/local/autonomous_20261010/mechanism_research/reference_focus_mask_audit_torch_nearest.json`

SHA256 `d809c7c19b5dc09239ebfb152f79ecc798be0785642e1d7acd1931a9aabc47f1`. Its original pilot manifest hash is `9763fe19263b0a30b00798c4dbc9a07645625a177e83d94facb45a46496c9971`.

|Legal reference geometry,100 each|Deep|PACO|
|---|---:|---:|
|Whole high-coverage(`c>=.9`) token count, median|3|357|
|Focused high-coverage token count, median|77|932.5|
|Whole / focus references with no high-coverage token|35 /2|0 /0|
|Same-box foreground-mass purity, whole→focus median|.51545→.74007|.93585→.96683|
|Foreground mass captured in focus, median|49.465%|64.233%|

All original mask hashes were verified. No query RGB/mask, raw O24, encoder or model was accessed. Pure token means known geometric foreground coverage, **not** a proven pure DINO foreground feature. Reference focus encoding is justified by this observable information opportunity; query gain remains the experimental question.

**Protocol correction preserved.** The initial audit mistakenly used PIL-nearest canonical mask resizing. The existing study and producer use torch-nearest, which differs on non1024 PACO masks. The producer detected the mismatch before DINO encoding. The original PIL receipt remains as history; the corrected torch audit changes18 PACO boxes and0 Deep boxes. Use the corrected hash and boxes exclusively. Correct mask canonicalization is `torch.nn.functional.interpolate(...,mode='nearest',size=(1024,1024))`; RGB uses the existing torchvision square resize. Reference RGB is first loaded with its lawful existing parent crop, canonicalized with the current transform's first resize, then cropped and actually encoded at1024. Cropping raw pre-canonical RGB directly would be a different input protocol.

## Exact paired head and controls

API:

```python
fit_predict(Rwhole_raw, Rfocus_raw, mask1024, box_xyxy,
            Qwhole_raw, native4_raw,
            apply_apd=whole_branch_flag, projection=shared_projection)
```

Each raw array is CPU FP32 `[4096,D]`, actual final O24. The default complete field is `actual.equal`. Other outputs are `actual.global`, `actual.local4`, `derived.global`, `derived.local4`, `derived.equal`; all are signed FP32 `[128,128]` with strict zero downstream decision.

The real arm uses actual focus O24. The derived arm uses the exact same focus box and fine source-mask coverage, but crops the whole-reference raw64 map to32×32 and bilinearly interpolates that raw feature tile to64×64 before normalization/APD. Thus both arms receive new fine mask supervision and the same labels/selection/weights, but only real has new encoded visual evidence. The derived field is not called an actual crop encoding.

All views share the original whole-query APD flag/projection. Reference raw tokens are normalized and projected in the original order. Whole query raw64→128 interpolation occurs before normalization; native4 are normalized/projected and placed in the original NW/NE/SW/SE128 grid. Input strides before normalization preserve the previous study's FP32 arithmetic: a synthetic check caught and removed an unnecessary contiguous copy. There is no per-view branch selection or probability calibration.

### Physical prior preservation and binary-role risk

For a16-aligned focus, each whole token outside the box has physical weight256, each whole token inside128, each focused8×8 token32. Each original pixel outside the box is counted once; inside it contributes half via each observation. Therefore

```text
sum_o area_o = 1024^2
sum_o area_o c_o = reference foreground pixel mass
sum_o area_o(1-c_o) = reference background pixel mass.
```

These equalities are verified independently. Naively concatenating both4096-token arrays at equal weight would instead overcount the selected region; this would confound visual gain with a source prior change.

Select128 midpoint area-mass quantile occurrences from each normalized physical FG/BG role distribution. A mixed observed token can appear in both. Collapse identical physical-view token IDs, with foreground/background occurrence counts `nF,nB`, giving

```text
a_j = (nF_j+nB_j)/256
y_j = (nF_j-nB_j)/(nF_j+nB_j).
```

Fit the exact empirical objective

```text
sum_j [nF_j/256(s_j-1)^2+nB_j/256(s_j+1)^2] + .01 ||w||^2,
s_j = x_j w + b.
```

This is binary-role risk with equal total role weights, not geometric coverage regression. Its collapsed loss has a known additive constant, analytical weighted target mean0, an unpenalized intercept and at most256 FP64 dual rows. Both real/derived fits share all IDs, weights, targets and capacity. Repeated observations/occurrences are quadrature and correlated source evidence, not independent validation or additional human shots. Full physical-source and sampled loss/residuals are diagnostics only.

Apply each independently fitted direction to both fixed query observations. Default field is exactly `.5 global + .5 local4`; no source threshold, query selector, FoRIS anchor, template or graph enters the head. These signed role scores are **not foreground occupancy or calibrated probabilities**.

## Why sensor inversion is not the current method

The lawful mask coverage of physical rectangles obeys a known linear averaging operator. DINO token features do not: global attention, unit normalization, context and crop scale make a token something other than a fixed optical mixture over that rectangle. Training against fractional source labels does not by itself certify transferable occupancy measurements. Balanced role risk explicitly predicts a different target; converting its score to coverage by `(score+1)/2` would repeat the already audited meaning error.

Even with genuinely calibrated query occupancy observations, the current four native quadrants already provide one observation per128-grid latent8×8 cell. The stacked whole/fine measurement operator has full column rank, so inversion need not invent sub-token information. For equal physical-area squared errors with no constraints, a2×2 fine block would have the exact solution

```text
z_k = local_k + .5(whole_block - mean(local_block)).
```

It corrects a block mean while retaining fine contrast. This is classical multi-scale consistency, not new identity or originality. A constrained solution could preserve `[0,1]`, but it would still need true occupancy calibration. Jointly learning an unconstrained query response and mask can admit trivial constant solutions. Forcing feature invariance across views could remove the very scale/context differences already shown useful in Deep/PACO.

[FeatUp](https://arxiv.org/abs/2403.10516) already uses multi-view consistency of frozen-backbone feature observations to learn high-resolution features; [FeatSharp](https://openreview.net/pdf?id=lioemOcq3H) combines tiled features with upsampling and such consistency. Reusing known footprint reconstruction, crop tiling or a frozen encoder alone cannot establish a substantial original contribution. The current experiment therefore isolates real reference information first and does not implement or promote an unsupported sensor inverse.

## Fixed complete-output acceptance

Parent will use the unchanged original200 Deep/PACO order, source manifest and cached whole/four native query inputs. All six fields plus direct/identical-CRF complete masks at CLI1024 and original frames are sealed before query GT enters scoring. Same-batch full FoRIS, MEAN and fast9 remain references. Fixed default real.equal must be evaluated against derived.equal; global/local4 explain observation preference and are not selected per dataset.

The added-visual-information claim requires real to improve actual difficult-query ordering and complete output beyond its same-label derived arm. Source purity, source self-fit loss or merely changing reference FG/BG priors cannot pass. A real-versus-derived gain without beating strong complete baselines is partial progress, not the user's complete result. Compare actual encoder/cache/head/finalizer time separately; one extra R observation plus existing native4 does not justify an unmeasured efficiency claim.

If real focus fails, do not change its box rule, role budget, lambda, view mixture or threshold using these query scores. The precise tested premise should be rejected while preserving its mask-only opportunity. If it succeeds, an ordinary new reference observation is still insufficient paper-level novelty; a distinct complete transfer contribution and matched ten-dataset evidence remain necessary.

## Executed implementation validation

- Real raw focus set equal to the derived raw focus: all three paired fields are byte-exact.
- Physical FG/BG mass closure: exact; role occurrences128/128 and risk weights.5/.5.
- Independent primal solution on256 expanded±1 occurrences agrees with the collapsed dual coefficient within6.67e-16 and bias1.91e-17. Binary loss constants, dual residuals and coefficient/intercept stationarity close.
- Whole/global and native4 query normalization/placement match the previous study byte-exactly, with and without a nontrivial APD projection.
- A synthetic aliased whole-reference representation obtains no signed target from the derived control, while a truly distinct focused representation separates its two roles under the same labels/weights. Constant whole/focus/query observations produce all six fields exactly zero.
- Six fields are finite and correctly shaped; JSON diagnostics serialize with `allow_nan=False`.
- Synthetic seven4096×1024 raw arrays, active projection deleting500 dimensions, CPU2: .21847 seconds for the paired six-field head; source processing/fitting.04747, query processing.11655, readout.02313 seconds. This excludes real encoder/cache/I/O/CRF and is not real end-to-end latency.

After READY, this source is frozen pending coordinated review or result. Older scene and copula modules are unchanged. No real query gain or completed original ten-dataset method is claimed here.
