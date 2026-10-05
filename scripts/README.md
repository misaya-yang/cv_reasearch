# Baseline entry and resource guard

The retained baseline programs are:

- `run_foris.py`: the existing complete public FoRIS entry on an explicit episode manifest. It saves the
  same score/stage packets and model/original-resolution I/U as the old `foris_dump.py`.
- `experiment_resource_guard.py`: the existing finite-queue/resource guard. Its behavior is unchanged.

The small shared package is in `src/ics/`:

| File | Purpose |
|---|---|
| `data.py` | DINOv3 timm wrapper, deterministic COCO episodes, image/mask loading and I/U helpers, extracted from demo4 |
| `foris.py` | Complete `set_reference / set_target / segment` path and unchanged host configuration, extracted from the extent runner |
| `native_basis.py` | Reuse the recorded native positional basis without changing it |
| `statistics.py` | Existing class-I/U bootstrap with connected-photo groups; preserves its default_rng convention |
| `__init__.py` | Package marker |

Inspect arguments without loading a model:

```bash
python3 scripts/run_foris.py --help
```

A later authorized run needs an existing FoRIS checkout, DINOv3 weights, COCO images/annotations and native
CRF dependencies. None are bundled or downloaded here. Supply `--manifest`, `--out` and optionally
`--foris-root` to override the manifest's existing checkout path. The manifest retains `data_root`,
`annotation_root`, `foris_root`, optional `projection_basis`, and `episodes` with fold/e/c/support/query.
The existing asset store still uses the legacy `DEMO4_CACHE` environment variable.

All statistical conventions remain report-specific; the dots RandomState protocol is not interchangeable
with this helper's default_rng sequence. The cleanup did not rerun the encoder or CRF, so runtime parity
after the file move is not empirically verified. Removed experiment implementations are recoverable from
Git history.

## Prepared six-mechanism batch, 2026-10-05

See [the preparation record](../evidence/local/research_20261005/README.md) for exact inputs, controls,
resource accounting and launch commands. The user has stopped CPU experiments and will enable the GPU.
No prepared command is an instruction to rent hardware or resume a historical queue.

- `inventory_existing_assets.py`: stdlib-only read-only asset/manifest/quota inventory.
- `run_mechanisms.py`: finite cache inference, including RCG, with isolated sealed-prediction scoring.
- `run_intervention.py --include-multilayer`: C/D complete FoRIS arms sharing one frozen model instance.
- `score_forward_run.py`: exact native replay check and common RandomState(0) connected-photo statistics.
- `prepare_mechanism_batch.py`: write a finite resource-guard plan without executing experiments.
- `run_gpu_batch.py`: GPU cache algebra, then one C/D model worker alongside CPU cache scoring.

All new complete-mask summaries use `ics.experiment`, leaving the historical `ics.statistics` convention
unchanged. The new runner saves per-arm timing, fields, packed full masks, input hashes and prediction seals.

## Evidence sweep (Claude Code, prepared 2026-10-05)

See [the record](../evidence/local/research_20261005/aux_evidence.md). It reads sealed proposals and recomputes none.

- `run_aux_evidence.py`: `infer` seals the whole label-free evidence library per episode (one GPU process or CPU
  workers; `--forward-layers` taps more blocks in one extra paired forward); `score` is CPU only: one row per
  proposer, operator and map, placebo calibration, the fold-nested composition and the common report.
- `prepare_aux_evidence_batch.py`: write its two-stage plan for `experiment_pipeline.py` without executing it.
- `src/ics/edit_counts.py` (count algebra, `python -m ics.edit_counts` self-check) and
  `src/ics/methods/aux_evidence.py` (the enumerated library).

## Stage bank: the model origin and both public pipelines (Claude Code, prepared 2026-10-05)

See [the derivation](../evidence/local/research_20261005/objective.md). Not run; synthetic self-checks only.

- `run_stage_bank.py`: `infer` (label-free, CPU workers, no encoder) seals the origin `model.raw_nn`, `model.raw_mean`,
  every stage of INSID3 and FoRIS rebuilt from the saved final-layer tokens, FoRIS with one term removed, and the term
  fields, and compares the rebuilt FoRIS stages with the cached public ones. Its output is the explicit origin of the
  other runners (`OUT:model.raw_nn`). `score` writes every stage as additions and deletions of the origin, searches
  the family that contains both pipelines with the choice nested over folds, and the value of each response level.
- `prepare_stage_bank_batch.py`: write its two-stage plan for `experiment_pipeline.py` without executing it.
- `src/ics/methods/stage_bank.py` (`python -m ics.methods.stage_bank`, `run_stage_bank.py self-check`).
