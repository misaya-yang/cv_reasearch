# Existing complete-mask library

The actual consumer manifests are ready on the owned remote workspace:

```
/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/complete_mask_library_consumer_v3/manifest_public4000.json
/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/complete_mask_library_consumer_v3/manifest_dev241.json
```

| Cohort | Source-arm entries | Distinct ordered mask sequences | Strict setting | Label-fitted extended setting |
|---|---:|---:|---:|---:|
| Public4000 | 14 | 12 | 12 | 12 |
| Original DEV241 | 416 | 186 | 181 | 186 |

These are complete-mask library variants and exact output aliases, not a count
of independent novel research methods. The catalog preserves every source/arm
identity. Exact aliases share `producer_id`; using that producer in both A and
B counts once. No small-K subset was silently substituted for the full library.

`rows` preserves the original cohort order, keys, class/fold/photos and all4000
sampled draws, including the natural repeated identity. Public4000 order was
checked against `frozen_subtoken4000_scored_v1/manifest.json`. Each row has
`packet={path,sha256}` and
`masks[arm_id]={path,key,sha256,packed_array_sha256}`. `sha256` is the entire
sealed prediction NPZ digest; `key` names its packed uint8[131072] complete1024
mask. Consumers load these existing arrays, rather than rerendering fields or
creating a missing mask from counts.

Each arm has `id`, `source_id`, `producer_id`, original NPZ key/source path,
GT/supervision metadata and two selectable eligibility flags:
`strict_eligible` and `label_fitted_extended_eligible`. Strict excludes the
explicit DEV-fold fitted readouts; the user-authorized extension retains them
with their label use disclosed. Query-GT per-example routing/oracle outputs
are excluded from both settings, without erasing their catalog provenance.
The new Opus size-cut masks are pending their worker's actual complete seal;
append them as a separate version. Do not mix crossfold60065.44 provenance
with fresh600-fitted65.86 provenance or describe the latter as label-free.

Actual sources include the canonical native/raw-NN/INSID3 release-CRF/RCG/MEAN/
coarse64/fine16/fine64 outputs; existing241 complete composition, conditional,
directional, joint, calibrated, edit-aux, stage, retained-B/role/jackknife
outputs; and identity-projected complete1200/4000 sources. Public4000 includes
the original six frozen arms, bit-edit transfer, actual official-block Astra/
count-matched deletion variants, and now the sealed scalar residual graft.

Missing assets remain explicit: raw final-layer NN origin has no complete
public4000 mask archive, so `origin_available=false` there. The original raw NN
mask remains unchanged as the DEV241 common origin. Public4000 `frozen.delete_p`
has count-only report/episode/receipt evidence and no sealed complete-mask
archive; it was not invented. No feature directory was restored or copied.

The first successful builder read and checked every selected prediction file's
sealed SHA, mask key/dtype/shape, cohort identity and ordered packed-mask
sequence:107.861 seconds, CPU6 readers, no query GT or features. Its initial
syntax-failed snapshot is retained; it produced no library. Successful
`snapshot_v2` and v1/v2 manifests remain immutable. Consumer v2 added verified
packet references and separate strict/extended eligibility.

Consumer v3 inherited the v2 SHA and checked only the new graft source/masks;
6.120 seconds, without rescanning the previous415 DEV source arms. Graft arm
`mean_fine_residual_transfer_v1` uses the sealed4000 source at
`outputs/mean_fine_residual_transfer4000_v1`; source seal SHA is
`82f9f208f2254cd25aff5a16f0bcd1e90f6af250c45c6efc7ff40275abc5ea8a`.
Its independent statistics check was subsequently completed by the anatomy
worker; library inclusion itself did not compute or alter its predictions.
The older11-method public4000 consumer remains in `complete_mask_library_consumer_v2`
for preserving that exact search, while v3 supports the12-method extension.

Latest manifest SHA256:

```
public4000: e3a7d14f4258c328e53b6e0882c15199c44614f215db1f527d7501059763a161
dev241:     9db261840e9e4e0f333d4c8af84f71d709e8d1da35774e9c1ea562941fda455d
```

The quality worker received both actual consumer paths and hashes for exact
family optimization. Local [compact_index.json](compact_index.json) lists
every arm, source, producer identity and eligibility without duplicating
features or masks. [consumer_v3_receipt.json](consumer_v3_receipt.json) records
the incremental checks; older receipts preserve the original library and
consumer versions. Background-SNR preparation was cancelled before any smoke,
inference or remote launch; its local preparation script was retained only.
