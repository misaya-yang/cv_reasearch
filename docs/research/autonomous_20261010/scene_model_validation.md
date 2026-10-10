# Independent scene-model review — 2026-10-10

No blocking defect was found in the reviewed mathematical objective, matched scene/source procedure, no-GT inference interface, or signed-field renderer. The reviewed implementation can advance to the frozen exploratory mechanism test. These checks establish code and formula consistency; segmentation accuracy, novelty, unseen-data generalization, and full FoRIS protocol superiority remain unvalidated.

Independent evidence: `evidence/local/autonomous_20261010/scene_model_validation/validation.json`; reproducible no-GT fixture: `validate.py` in the same folder. Post-seal actual replay evidence is `actual_repeat_index400_no_gt.json`, produced by `repeat_sealed_no_gt.py` in that folder. The method, parent runner and sealed experiment were not edited.

## Source boundary

| File | SHA256 reviewed and tested |
| --- | --- |
| `src/ics/methods/autonomous_context_transport.py` | `8e7e068d629d744c5e2f8a5fac162e57fa78d4af99609b2318d170c5b8781d06` |
| `scripts/autonomous_scene_reconstruction.py` | `65f43a91133a21fc35fd8849579c311ff48d1577eeddcec2cc3d5e7d699cb2ba` |

Both matched the prepared experiment's frozen copies at validation. All eleven frozen source hashes, manifest/task hashes, the raw-cache profile hash and positional-basis hash passed. The live branch was `codex_m4`, HEAD `40204d66e6b06c5026fe44ff97b0bb7de10643ac`; concurrent worktree changes were left intact.

## Objective and mixture verification

For fixed sampled reference coverage `c`, the code defines `p=c/sum(c)`, `n=(1-c)/sum(1-c)`, `a=(p+n)/2`, and `y=(p-n)/(p+n)`. The exact identity is

```text
.5 sum_i p_i(f_i-1)^2 + .5 sum_i n_i(f_i+1)^2
  = sum_i a_i(f_i-y_i)^2 + sum_i a_i(1-y_i^2).
```

`_ridge` minimizes this risk plus `.01 ||w||^2`, leaving the intercept unpenalized. Its weighted centering, `K + diag(lambda/a)` dual solve, and bias recovery are consistent. Independently solving the original positive/negative objective in an augmented primal system, including an intercept column, agreed on all three representations: maximum coefficient error `4.774e-15`, maximum intercept error `1.849e-14`. Binary/collapsed risk closure error was at most `1.111e-16`.

`A=softmax(cosine/T + log(pi))`, `Psi=A/sqrt(pi)`, and `u=w/sqrt(pi)` give `Psi w=A u` and `||w||^2=sum_k pi_k u_k^2`. The synthetic norm-identity error was at most `8.882e-16`; responsibility row sums and positive occupancy normalization passed. This is a norm on **latent component role values** under the hard-cluster prior, not a claim that the penalty equals the empirical squared final query score.

Repeated method calls produced byte-identical fields in all three arms and did not mutate inputs. Complementing binary-exact continuous reference coverage negated all three fields exactly. That symmetry is conditional on fixed features: the runner's legal foreground-dependent APD selection need not choose the same representation after a raw-mask complement. JSON serialization with `allow_nan=False` passed.

## Matched controls and input identity

`scene` and `source_kernel` share the same spherical mixture procedure, maximum capacity, temperature, occupancy pseudocount, reference samples, binary-role weights, ridge strength and signed-zero rule. They differ by the unlabeled cloud that supplies centers and occupancies. This is a valid contrast for the complete query-distribution representation hypothesis. It does not separately identify the contributions of center geometry and occupancy weighting, and it does not claim fixed identical numerical kernels or regularizers across different clouds.

`support_ridge` shares the same reference objective and fit IDs in the original unit-feature representation. Its independent primal solution passed. Equal numerical lambda does not make raw-feature and mixture-feature function norms identical; interpretation should retain that representation difference.

The prepared panel has DeepGlobe road100, PACO-Part100, COCO200 and LVIS200. Every row matched its corresponding archived baseline manifest on episode ID, reference/query decoded RGB hashes, reference-mask hash, crops and image sizes. LVIS's separate reused complete-FoRIS manifest also matched for every selected row. The validator opened only real input metadata, not real feature arrays, mask pixels or prediction arrays.

The raw-cache interface keys transformed RGB plus the model profile. Its reader verifies the complete payload file hash and each requested O24 array's shape, dtype and tensor hash. `load_inputs` opens only R RGB, the legal R mask and Q RGB, validating their hashes. The fitter imports only NumPy/Torch and receives no dataset name, query annotation, baseline field or baseline mask. Query GT appears only in the separate `score` function after `sealed.json` is checked. Static review found no query-GT or prior-prediction path in inference.

## Final renderer and CRF boundary

The runner interpolates each continuous signed64-grid field to1024 with bilinear `align_corners=False`, applies strict `>0`, and calls the released FoRIS `_finalize_mask` once with the same normalized query image. `cache_host(... resize_to_orig_size=False)` keeps that result at1024; final original-size rendering interpolates the binary result and applies strict `>.5`. The latter matches the released `upsample_mask` helper on an independent synthetic mask. All three candidate arms share this readout; direct and CRF predictions remain separate.

The official narrow-band CRF wrapper receives a binary mask, builds fixed p-core/boundary unaries, uses denormalized query RGB, and changes only the boundary band. A synthetic identity-solver fixture exercised that wrapper and boolean/geometry contract. The later post-seal replay below also executed the existing native lattice backend.

One nonblocking provenance gap remains at `scripts/autonomous_scene_reconstruction.py:139`: config hashes the official FoRIS/data/refinement modules, but does not bind the complete imported `third_party/crf_source/src` Python/C++ tree or native compiled binary. The local CRF wrappers are frozen and hashed. To certify exact finalizer provenance, preserve/hash that external source/backend receipt at output validation or extend the runner before a future freeze. No drift or wrong CRF output was observed in this review; no run needs to be discarded solely on this gap.

The config describes a signed64 field while the method intentionally returns FP32 scores and the runner promotes them to FP64 for interpolation. The field grid is64×64; this wording is not evidence of end-to-end FP64 score precision. It is consistent across all candidate arms and is disclosed by the method's return dtype.

## Executed post-seal actual replay

After the parent sealed all600 predictions, the independent validator replayed index400, LVIS episode `dev_s1/lvis/0/5`, with explicit frozen module loading. The raw reader validated both complete4096×1024 FP32 O24 arrays against their input, cache-entry, payload-file and tensor hashes. The validator independently reproduced the shared legal APD branch, source mask coverage, method call, signed FP64 interpolation and final renderer.

All three returned FP32 fields, promoted to the saved FP64 format, matched the saved fields exactly. A second method call produced byte-identical fields. All12 saved complete outputs (`3 arms × direct/CRF × CLI1024/original`) matched the regenerated masks with **zero differing pixels**.

A Python audit hook blocked the current query-mask file, both archived baseline prediction files and scoring outputs, and prohibited writes anywhere under the sealed experiment root. It recorded zero forbidden-input accesses and zero attempted sealed writes. Seal, config, manifest, task list, episode record, continuous field file and complete prediction file hashes remained unchanged. Encoder calls and raw-feature writes were zero.

The existing `ics_permutohedral_m4.so` and `ics_permutohedral_m4_prepared_v1.so` were loaded directly without compilation or build-receipt updates. The exact frozen M4 wrapper class was extracted via AST; an original FoRIS host used a forbidden encoder stub and native positional-basis reuse. The prepared backend and complete original CRF settings were then used for the final masks. Native binary hashes and all external CRF source hashes from the parent's prelaunch `real_smoke4.json` are preserved in the independent replay receipt; source hashes matched that earlier receipt.

The replay took14.626 seconds including importing modules and reading/hashing the full600-episode seal twice. It is a verification cost, not a candidate inference-latency measurement. The earlier config-level external-CRF provenance limitation remains, but this actual replay now binds the binaries and source files it used.

## Remaining validation

The evidence-validation agent owns all600 scoring and ROI analysis. No query labels or scientific scores were read by this validator. This one-sample repeatability check does not establish accuracy, cross-dataset generalization or novelty. If separately requested after scoring, an independent reducer can recompute class-pooled I/U and fold means from sealed prediction/truth pairs.
