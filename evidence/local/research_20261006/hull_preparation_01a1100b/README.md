# Reference convex-hull candidate: implementation preparation

This is one new candidate, separate from the nine prepared candidates and Pro M1-M5. It has **no measured real DINO segmentation gain**. These checks use synthetic, sparse vectors embedded in 1024 dimensions. No encoder, GPU, download or server job was invoked.

## Fixed inference contract

`reference_hull.predict(q,r,coverage,base,Config())` normalizes frozen tokens and compresses the reference's pure foreground and pure background separately to at most 16 spherical Lloyd modes each. Default purity is 0.9, five Lloyd steps, at least eight samples per class. Missing classes retain every baseline value.

For each query token and class, solve the squared distance to its mode convex hull, allowing nonnegative weights summing to one. Batched Frank-Wolfe starts from the nearest anchor and uses exact line search, for at most 256 updates. With the half-squared objective, the FW gap gives squared-distance bounds `[max(0,cost-2*gap),cost]` in exact arithmetic. They bound numerical optimization error only. Float64 computation and a nominal 1e-7 decision guard do not constitute outward-rounded interval arithmetic or semantic confidence.

For BG-minus-FG squared distance, the interval is `[BG.lower-FG.cost, BG.cost-FG.lower]`. If its lower endpoint exceeds the guard or its upper endpoint is below minus the guard, replace that token's value with `clip(.5+(BG.cost-FG.cost)/8,0,1)`. Otherwise preserve the original baseline value exactly. All fields use the existing float32 bilinear expansion to 1024, `align_corners=False`, and threshold `>0.5`. Bounds apply on the token grid, not after spatial interpolation. Tiny near-half scores can round to half in the float32 finalizer, as visibly happens to the quadratic control in this witness.

Controls are the same compressed-anchor nearest match, original weighted pure-class mean, linear span, affine span, and the existing fixed `reference_quadratic.predict` (default ridge .01). They are controls, not additional claimed methods. Query GT is absent from the inference API.

## Deployment

```sh
python3 scripts/run_reference_hull.py --inputs /bound/input_*.npz --out /new/hull_run --workers 2
python3 scripts/run_reference_hull.py --verify /new/hull_run
```

Each input must have aligned q/r `4096x1024`, cov/base `64x64`, with a frozen existing MEAN baseline. The runner opens only q/r/cov/base even if other keys exist. It preserves repeated inputs as separate occurrences, uses two spawned CPU workers by default and one numerical thread per worker, writes packed complete masks in `predictions/`, continuous fields and token distance intervals in `fields/`, and audited receipts in `receipts/`. It hashes the source, inputs, contract, predictions, fields and receipts. A successful `sealed.json` is written only after all occurrences complete and code hashes remain unchanged; a separate verification checks sealed artifacts. Real GT scoring must happen subsequently and independently. The CLI refuses to overwrite an existing directory. This independent runner does not modify the shared prepared backend.

## Observed checks

`check.py` is rerunnable with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 .../check.py`. Details and saved predictions are in `checks.json`, `witness_outputs.npz` and `semantic_failure_outputs.npz`.

The witness has three rotations θ=0,2π/3,4π/3. FG modes are θ±.4 and BG modes θ+.2. In exact arithmetic both classes have zero mean and second moment .5I. The reference has 96 pure FG tokens balanced over six modes, 3996 pure BG tokens balanced over three modes, and four half-coverage tokens excluded from both classes. Three 8x8 query blocks at θ are designated FG and start at baseline .45; all other query tokens are BG modes and start at .1.

| Readout | Hull | Base / nearest / original mean / linear span / affine span | Quadratic |
|---|---:|---:|---:|
| Token TP / FP / FN | 192 / 0 / 0 | 0 / 0 / 192 | 128 / 2601 / 64 |
| Complete1024 TP / FP / FN | 49152 / 7608 / 0 | 0 / 0 / 49152 | 0 / 0 / 49152 |
| Complete1024 IoU | 86.5962% | 0% | 0% |

The quadratic field differs from .5 by at most 1.872e-8, and its token threshold counts are a numerical near-degeneracy, not stable recovery. The full float32 readout rounds it to half. Hull's interpolation introduces boundary false positives; token perfection must not be reported as complete-mask perfection.

FG FW reaches the 256-step cap with maximum gap .000530764 and all 4096 rows formally unconverged at 1e-8. BG converges in one step. Despite FG optimization error, all 4096 signed intervals are decisive. We do **not** claim global exact optimization. Independent SLSQP checks on 21 small projection problems place their optima within nominal FW bounds to numerical tolerance 1e-9; these checks are numerical corroboration rather than a rigorous floating-point proof.

Shared identical FG/BG/query points leave all 4096 arbitrary baseline values unchanged exactly. A missing reference class also preserves the baseline. Four invalid configurations are rejected.

The semantic failure uses the same reference but gives all query tokens BG-mode appearances, while retaining the same designated true-target blocks with a strong .8 baseline. Hull deletes every true target: token TP 192→0; complete1024 IoU 96.3867%→0%. This is a failure of the appearance assumption despite decisive optimization bounds.

`two_worker_run/` processed two repeated witness inputs in distinct worker PIDs, 0.336 and 0.338 seconds per occurrence locally, with identical outputs. The complete run sealed and verified. `verification.json` records repeat preservation, ignored object-valued GT poison key, and successful detection of a corrupted prediction in a temporary copy. Runtime is synthetic local preparation only and gives no real dataset time estimate.
