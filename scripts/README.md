# Baseline entry and resource guard

Only two command-line programs remain:

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
