# Explicit-origin DEV241 edit suite: offline preparation

Owner: `/root/edit_auxiliary_suite_sol`, updated 2026-10-05. State: **local code/contracts ready;
real inputs, raw-origin producer, full241 inference/scoring and CUDA remain unverified**. No remote
work, GPU, model forward, scientific method execution or download was performed in this preparation.
Only `scripts/run_edit_auxiliary.py`, `src/ics/edit_auxiliary.py` and this record were changed.

## Scientific correction and identity

The common edit origin is now the **explicitly supplied, sealed raw DINOv3 l24 reference-nearest-token
hard mask**, suggested arm `model.raw_nn`. Its producer owns the exact nearest-token/hard-mask
rendering. The source is intended to use normalized encoder taps before FoRIS projection/debiasing.
This suite does not generate that origin or certify its scientific provenance from its arm name.
Its producer's manifest, seal, masks and optional config/protocol receipts are checked before use.

`native` always means complete original cached **FoRIS**, an ordinary method row. FoRIS, RCG, E/B
and other completed methods are allowed combination ingredients; their existing internal algorithms
and inputs are not rewritten. Older edit_budget.md FoRIS error budgets remain historical analyses
of FoRIS, not raw-model error budgets. The v1 host interpretation is superseded; use a fresh v2 output.

The proposal pass remains unchanged and can finish before rawNN is available:

| Alias | Exact function / existing sealed arm |
|---|---|
| E | `latent_native.predict` / `latent_native` |
| Efixed | `latent_native.control` / `latent_native.control` |
| Etwo | `latent_native.two_slot_control` / `latent_native.two_slot_em.control` |
| Bkernel | `structure_conditioned.kernel_control` / `structure_conditioned.kernel.control` |
| delta | Existing sealed `multilayer.delta.control` |
| rcg | Existing sealed `rcg` |

Bkernel is the gpu_second20_v1 control, not original `structure.control`. Its older first20 added
TP/FP counts and E's mini50 results describe edits of FoRIS only; they do not establish raw-origin
coverage. All six complete methods are retained regardless of standalone net score. Cached q/r,
reference cov and FoRIS score are passed to the original functions with unchanged parameters.
The original `ics.experiment.render` probability renderer is unchanged; no image, CRF or weight
asset is added. The packet's `native` is retained as FoRIS identity in proposal files.

## Fixed edits and controls

For explicit origin O and complete proposal P, A=P\O and B=O\P. Reuse the unchanged continuous
`transport_query_witness.predict` FG/BG field h and set H=bilinear1024(h)>0.5. Accepted A=A∩H,
accepted B=B∩~H. Complete mask=(O\accepted_B)∪accepted_A; retain only-add and only-delete.
This thresholded helper is exactly agreement between a proposal and helper in each edit region,
and remains a minimum auxiliary baseline, not an independent new method or established gain.

Modes are `noaux`, `aux`, and `foris_score`. The latter keeps the auxiliary's exact accepted count
inside the same proposal edit region, ranking A by high FoRIS continuous score and B by low score,
with ascending row-major index ties. `foris_score` is ordinary FoRIS evidence, not raw-origin confidence.
Optional origin fields are retained for provenance only; they never redefine the sealed hard mask.

The 12 previously added whole-domain localization controls remain, now named `global_foris_count`:
each raw A/B proposal count is matched across the entire origin background/foreground, ranked by
FoRIS score. They remove proposal localization while preserving its raw edit count; they do not use GT.
There are six same-source A/B pairs and five non-RCG adders paired with RCG deletion. There are
81 fixed edit outputs plus six original method masks plus origin: **88 masks in each final prediction
file**, and **89 scoring rows including separately read FoRIS native**. These are ablations/controls,
not independent new methods. No query-GT routing, per-episode selection or pooled-purity selector is used.

## CLI and two independent protocols

The default `prepare` only checks cached dependencies on CPU. Set the execution device consistently
when preparing a future inference run; prepare does not initialize CUDA. The paths are placeholders
for the authorized execution machine, not remote dispatch commands.

```bash
# Stage 1 needs no rawNN origin and applies no old-host edits.
python scripts/run_edit_auxiliary.py --stage prepare \
  --root SHARED_CACHE_ROOT --d-run SEALED_D241 --rcg-run SEALED_RCG241 \
  --out outputs/edit_auxiliary_dev241_v2 --device cuda
python scripts/run_edit_auxiliary.py --stage infer --proposals-only \
  --root SHARED_CACHE_ROOT --d-run SEALED_D241 --rcg-run SEALED_RCG241 \
  --out outputs/edit_auxiliary_dev241_v2 --device cuda --workers 3

# After rawNN is separately sealed, bind it and reuse the completed proposals.
python scripts/run_edit_auxiliary.py --stage prepare \
  --root SHARED_CACHE_ROOT --d-run SEALED_D241 --rcg-run SEALED_RCG241 \
  --out outputs/edit_auxiliary_dev241_v2 --device cuda \
  --origin RAW_NN_RUN:model.raw_nn
python scripts/run_edit_auxiliary.py --stage infer \
  --root SHARED_CACHE_ROOT --d-run SEALED_D241 --rcg-run SEALED_RCG241 \
  --out outputs/edit_auxiliary_dev241_v2 --device cuda --workers 3 \
  --origin RAW_NN_RUN:model.raw_nn
python scripts/run_edit_auxiliary.py --stage evaluate \
  --out outputs/edit_auxiliary_dev241_v2 --score-workers 2 \
  --origin RAW_NN_RUN:model.raw_nn
```

`--origin` is mandatory for infer without `--proposals-only`, and for evaluate, even if a stored
auxiliary protocol exists. No missing-origin path silently falls back to FoRIS. Default cohort is
`evidence/local/research_20261005/dev241.json`; `--manifest` can explicitly select the same cohort.
Optional `--origin-field RAW_NN_RUN:model.raw_nn` must be supplied consistently at binding, infer
and evaluate if used. Explicit `--device cpu --workers 1` is a model-free CPU alternative.

`protocol.json` freezes the proposal functions, configs/code, input hashes, device and cohort.
`proposal_sealed.json` seals all241 ordinary method masks and new fields before any helper edits.
It includes `mask_directory: "proposals"`, `predictions: {key: file_sha256}`, and hashes of every
field/receipt. `proposal_completion.json` records `PROPOSALS_SEALED_UNSCORED`. A later origin binding
creates `auxiliary_protocol.json` without changing this protocol, seal or proposal artifacts.
The final `sealed.json` binds the auxiliary-protocol hash and origin arm, all final masks/accepted
edits/helper fields, the prior proposal seal and original packet receipts. Evaluation is separate.
Atomic per-episode writes and hash-checked receipts preserve valid work on restart. Changed inputs,
code, origin or sealed files fail closed; unrelated output directories are rejected. There are no
elapsed-time or idle-utilization kill rules. Model-free spawned inference can use 1..3 workers;
independent CPU scoring can use 1..2 workers.

## Exact origin schema

`--origin RUN_OR_MASK_DIR:ARM` accepts a sealed run or its mask directory. With a mask-directory
argument, `manifest.json` and `sealed.json` must be in its parent. A run argument reads `predictions/`
when present, otherwise the run directory itself. The schema is:

- `manifest.json`: list of rows or `{episodes: rows}`; exact same241 `fold,e,c,support,query` identities,
  unique keys/episode identities and reference/query photographs. Batch must agree when both provide it.
- `sealed.json`: `manifest_sha256` and `predictions` mapping every241 key to NPZ-file SHA256. Optional
  `state`, if supplied, is `ALL_PREDICTIONS_SEALED`. Optional `config_sha256`, `protocol_sha256` and
  `arm_bases_sha256` are verified against their standard filenames when present.
- `predictions/<fold>_<e>_<c>.npz`: member ARM is uint8[131072], `np.packbits` row-major 1024² mask,
  default big bit order. `native`, existing proposal arm names and `edit.*` are rejected as origin identities.
- Optional field source: analogous run or `fields/` directory, same cohort identities and seal,
  `fields: {key: file_sha256}`; each NPZ's selected ARM is finite float32[64,64]. No mask/field-renderer
  equality is demanded: the producer's hard-mask definition remains authoritative.

Prepare verifies all241 schema/hash dependencies without indexing query truth. Origin masks/fields
are checked before auxiliary inference, reread against those hashes, and preserved in the final seal.
This adapter does not renormalize l24, remove components, project features or reconstruct the origin.

## Scoring semantics and validation

Independent evaluation keeps full-mask `scores`, paired `contrasts` and `gain_vs_native` comparing
with **FoRIS native** and RCG/ordinary score controls through `ics.experiment.summarize`. The extra
raw-origin row is scored as its own identity and `contrasts_vs_origin` reports complete differences.
Edit counts are explicitly `corrections_vs_origin`, `corrections_vs_origin_by_class` and
`corrections_vs_origin_by_batch`; the shared scorer's historical `corrections_vs_native` key is removed
from this report rather than assigned a new meaning. `origin_arm` is explicit in report/episode records.

Exact per-class I/U, paired95% intervals from 2000 RandomState(0) connected-photograph draws,
fold/batch scores, up/down/tie, add_TP/add_FP/delete_FP/delete_TP, A/B coverage/purity, helper harmful-edit
removal versus beneficial-edit loss and complete-combination attribution are retained. Every saved
mask is checked against exact origin edit I/U and the accepted A/B recipe. Opus <10% overlap,
8-connected GT diagnostic buckets partition **origin** FP into wrong-object/boundary and FN into
whole-object/extent errors. Their true-pixel collateral remains delete_TP, not repaired error mass.

Current local checks passed: syntax/CLI; 20 synthetic Boolean cases including exact I/U, agreement,
same counts/ties and component partitions; synthetic241 dependency preparation; guarded NPZ truth
access; one mocked episode with all89 scoring rows; separate FoRIS/origin identities despite a
mismatching optional origin field; unchanged proposal protocol/artifact when binding an origin;
restart reuse and hash-tamper rejection. CLI rejects missing origin before mutating auxiliary/scoring
outputs, while proposals-only accepts missing origin and reports all243 missing cache/source failures.
Previous two-worker CPU-spawn and 360 count/ranking checks remain interface checks, not efficacy.

**Unverified:** live shared241 assets/source seals, the rawNN producer's scientific/raw-layer provenance,
its actual complete output schema, real method-function execution, CUDA/throughput/memory, full-cohort
seals and scoring, and scientific efficacy. The suite has not run original matching or any full scientific
experiment. Astra remains a separate unchanged candidate/source; its own CPU RCG cannot be silently
replaced by GPU RCG for exact replay. Root owns Astra replay and the raw-origin producer integration.
