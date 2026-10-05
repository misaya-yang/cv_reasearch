# Reference-conditioned value residual with complete FoRIS/CRF

Status: fixed construction implemented; local synthetic contracts passed on
2026-10-05. **Actual timm/Eva interception, real-image native parity, execution
cost and complete-mask efficacy are not yet measured for this construction.**
No remote process, model loading, query-label analysis, download, or GPU run was
started by this implementation task. Parent owns the single-model worker and
the fixed stratified mini50 manifest, which includes the previous 20 episodes.
The current D worker must finish its forward/cache work before this queue uses
the encoder. This document does not authorize a competing model process.

## Complete mechanism in ten lines

1. Encode the supplied reference and query together in the native FoRIS order.
2. At block index 18 (`L-6` for ViT-L/24), retain the original attention call.
3. Area-resize the complete reference annotation to the actual patch grid.
4. Compute pre-RoPE input cosine correspondence with fixed temperature 0.1.
5. Normalize correspondence separately over reference foreground/background coverage.
6. Retrieve query-specific foreground and background means of actual projected V.
7. Subtract the background value message from the foreground value message.
8. Set its per-token L2 norm to 10% of the native SDPA output and add it to query patches only.
9. Continue native attention norm/gate/projection, block residual/MLP and remaining frozen blocks.
10. Run the unmodified complete FoRIS score, binarization and CRF; return one 1024 mask.

Parameters are fixed before any real result: `block_from_end=6`,
`temperature=0.1`, `residual_ratio=0.1`, query chunk size 256. The CLI exposes
no layer, temperature or strength sweep. Inference never reads query truth.

## Failed link and source-level non-repetition

The current C20 evidence reports +0.0779 points over native and -0.0024 over its
equal-RMS key-only control. C changes only 0.1385% of all pixels before CRF and
0.1024% after it. Thus a large corrective pre-mask action erased by CRF is not
the measured failure. Its bias `gamma s_i s_j` supplies only one key-ranking
direction, rather than a reference-part value vector. See
[intervention](intervention.md) for the existing measurements and their limits.

The previous negative global-content rewrite was recovered directly from Git,
at commit `3b9f634fd084519c039b436198f273609c6afe34`:

- `demo_lists/demo9_transductive_ics/scripts/global_content_attention.py`,
  `lift_qkv` and `_eva_forward`: patch/patch logits use rotated Q/K, while every
  prefix-involved logit uses unrotated Q/K. It constructs one replaced softmax,
  retains original V and patches every Eva attention block, for both images.
  There is no reference-mask argument or cross-reference value retrieval.
- `demo_lists/demo9_transductive_ics/scripts/native_readout_probe.py`,
  `decode_raw_pair` and the `full_rewrite` loop: rewritten layer readouts are
  combined using support-derived SAFR weights and a FROST-style readout. The
  recorded 29.4929 versus `native_raw` 42.6840 (-13.1911) is its fold-0 first10
  1024 pilot, **not a complete-FoRIS native comparison**. It additionally encodes
  a flipped reference. Its negative result still challenges arbitrary encoder
  intervention; the comparator and resource setting must remain explicit.

Reproduction of this source check requires no checkout or restoration:

```sh
git show 3b9f634:demo_lists/demo9_transductive_ics/scripts/global_content_attention.py
git show 3b9f634:demo_lists/demo9_transductive_ics/scripts/native_readout_probe.py
```

The new operation leaves native logits/softmax/RoPE untouched at every layer,
leaves the reference and prefix outputs directly untouched, and adds a separate
label-conditioned reference V message at one block before complete native
FoRIS/CRF. This is a changed inference operation with previously unused
intermediate correspondence/value information, not new external input. Retaining
native computation does not prove that its accuracy survives nonzero injection.

## Controls and interpretation

The first comparison contains five complete arms on the same fixed mini50:

| Arm | Operation and matched resource |
|---|---|
| `native` | Original complete FoRIS/CRF; SDPA observer returns the exact native result. |
| `reference_value` | Query-specific conditional FG-minus-BG value message. |
| `reference_value.control` | Uniform FG-minus-BG reference mean-V direction, normalized to the **identical per-token native-output norm budget**. It computes and discards the same conditional retrieval, matching input, layer, forward count and retrieval algebra. |
| `reference_value.key_only.control` | Existing C key-only prior at the same layer, using the same mask/reference/query and one forward. Its logit RMS is matched to C, **not** to the new value residual norm; this is an additional strong prior comparison. |
| `reference_value.foreground_only` | Conditional foreground V alone, with the same per-token norm budget and the same FG/BG retrieval computation. This tests whether explicit background subtraction contributes. |

The value control's direction is constant over query tokens; its budget uses
native attention magnitudes and therefore varies by query. The candidate can
retrieve distinct reference-part directions even when scalar FG/BG kernel
densities are close. Near-zero direction vectors receive zero update; their
counts and the uniform-vector norm are recorded, since a degenerate uniform
direction cannot honestly be claimed as norm matched. No episodewise GT-based
switch, threshold, mask fusion, RCG gate or reference selection is present.

This construction can fail if nearest reference parts transfer appearance rather
than target identity, if foreground/background values share the wrong semantic
direction, or if subsequent frozen layers/FoRIS discard useful movement. A
constant-norm intervention also spends its budget on confident correct regions;
the four-action ledger must expose damage along with corrections. These are
specific uncertainties to measure, not observed failures or promises of gain.

## Execution, audit and attribution

Code: `src/ics/methods/reference_value.py` and `scripts/run_reference_value.py`.
The context manager is reusable by the parent's existing single-model worker:

```python
capture = {}
with reference_value(vit, lambda: host._ref_masks[0], arm="reference_value", capture=capture):
    pred, observed, _, _ = run_foris(host, support, support_mask, query)
```

Standalone inference, **only after parent schedules the encoder**:

```sh
python scripts/run_reference_value.py --manifest FIXED_MINI50.json \
  --foris-root EXISTING_FORIS_CHECKOUT --out FRESH_OUTPUT_DIRECTORY
python scripts/score_forward_run.py --out FRESH_OUTPUT_DIRECTORY \
  --cache-root EXISTING_CACHE --rcg-run SEALED_MATCHED_RCG_RUN
```

The runner constructs one host and one frozen DINOv3 instance, sequentially runs
all arms, hashes source/config/manifest and seals complete masks before scoring.
Each deployable arm takes one paired encoder forward, with no added image or
model. Five comparison arms take 5 paired forwards per ordinary episode. The
first adds one untouched-native audit and one full retrieval path with zero
residual: **mini50 totals 252 paired forwards**. RCG scoring joins already sealed
same-identity predictions; its cost is outside this runner.

First-episode observer identity and zero residual must reproduce untouched native
complete masks exactly, scores within `atol=rtol=1e-5`, and raw reference features
within that tolerance. All non-native arms' raw reference features are checked
against native. All value arms additionally require the same pre-injection SDPA
output on every episode, proving that the target receives the same native input
and branch before intervention. Failure leaves the cohort unsealed.

Saved packets contain final packed masks, original score fields, actual native
binarizer pre-masks with explicit shape, and per-token native SDPA norm,
injected-delta norm, projected-attention delta norm, density margin, FG/BG
retrieval entropies and contrast/uniform-direction cosine. They isolate whether
the allocated message moves actual projected attention before later attenuation.
No full N-by-N correspondence matrices or reference/query GT diagnostics are
persisted. Wall time and peak allocated CUDA memory are recorded per arm; no
actual cost is inferred from the synthetic check. Explicit retrieval uses a
256-by-reference-patches kernel, with two corresponding value products; it adds
work beyond native despite requiring no extra encoder forward.

Judge complete class-summed I/U, paired connected-photo 95% CI, folds/batches,
up/down/tie and add-TP/delete-FP/delete-TP/add-FP against native and both controls.
The parent uses mini50 to assess/optimize; only afterward does an unchanged
candidate proceed to existing DEV241. Mini50 and DEV241 are exposed development,
not independent confirmation. A positive crossing-zero interval is unresolved;
a win against native alone does not establish query-specific reference readout.
There is no measured +2-point result for this new construction yet.

## Local observed validation

Command `python3 scripts/run_reference_value.py --self-check` returned
`SYNTHETIC_CONTRACTS_PASSED` with torch 2.8.0. The tiny randomly initialized
attention fixture has two images, two prefixes, four patches and two heads. It
does not instantiate DINOv3 and is not a scientific experiment. Checks passed:
conditional means equal direct weighted sums; reference permutation invariance;
equal per-token candidate/control/foreground norm; conditional versus uniform
non-equivalence; exact native observer and zero residual; exact reference/prefix
outputs; changed query output; empty foreground rejection; exception restoration
of global SDPA and attention method; frozen parameters. CLI help and AST parsing
also passed. Local timm is unavailable, so actual Eva/real-image execution is
explicitly pending on the parent's shared worker.
