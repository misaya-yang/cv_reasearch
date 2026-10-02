# Shared decoder computation: constructive CPU prototypes

Owner scope: computation structure and engineering. All code here is local CPU
research code, independent of the measured four-method implementation. No GPU
job, remote connection, downloaded asset, or trained segmentation score was used.

## Decision and evidence

The most useful first GPU intervention is the **phase-layout upscaler**, supplied
to both the factor candidate and the strongest dense control. The next independent
intervention is **implicit read/write attention**, keeping native dense LayerNorm
as the companion. The companion-free statistic LayerNorm has an exact-real proof
and working random-weight prototype, but has a demonstrated FP32 cancellation
failure; do not bundle it with the safer engineering changes in the first test.

`validate.py --full` tests the same nine seed/grid/token-count/batch configurations
as the imported suite, preserving its constant/spatial dense-prompt modes and
FP32 RNG stream. It tests FP64 and FP32 and all four masks plus four IoU outputs.
The original absolute threshold remains **5e-5**. These are random-weight,
synthetic encoded inputs, not pretrained image or ground-truth accuracy tests.

| Variant | FP32 max mask error | FP32 max IoU error | Checks |
|---|---:|---:|---:|
| Factor + phase layout | 4.02331e-7 | 1.30385e-7 | 18/18 |
| Factor + implicit attention | 4.91738e-7 | 1.63913e-7 | 18/18 |
| Factor + statistic LN | 3.61353e-7 | 1.04308e-7 | 18/18 |
| All three | 4.17233e-7 | 1.19209e-7 | 18/18 |
| Dense auto/auto + phase | 3.12924e-7 | 1.93715e-7 | 18/18 |
| Dense projected/dense + phase | 3.57628e-7 | 7.45058e-8 | 18/18 |
| Dense associated/sparse + phase | 3.12924e-7 | 1.93715e-7 | 18/18 |

Total: 126/126 passed, with no low-resolution binary flips in these fixtures.
FP64 mask errors are below 1.1e-15. Raw entries and a cancellation counterexample
are in `cpu_results.json`; `cpu_validation.log` preserves unaggregated prints.
Historical pretrained errors around 1e-3 remain failures and are not erased by
these CPU results. Diagnostic wall times in JSON are not a performance comparison.
The initial all-spatial stress run is retained separately in
`cpu_spatial_stress_results.json`; it also passed but repeats two configurations,
so the canonical report above uses all nine original distinct fixture modes.

## Closure under LayerNorm and linear rank growth

Write the image state as `X = s B + U V`, where `s` scales each image row, `B` is
common to prompts, and `V` has `R` rows. A write attention head has probabilities
`P_h` and projected sparse values `C_h = value_h O_h`, where `O_h` is the fixed
head block of the output projection. Since each attention row sums to one,

```
P_h C_h = P_h[:, :-1] (C_h[:-1] - C_h[-1]) + 1 C_h[-1].
```

Thus one write adds at most `H(T-1)` varying factors plus a constant. A constant
already occupying the last factor row can be merged with the new constant.
Let `C(z)=z-mean_channels(z)`, and let `rho` be the row inverse standard deviation.
LayerNorm is exactly

```
LN(X) = (rho s) [C(B) gamma] + (rho U) [C(V) gamma] + 1 beta.
```

The base is updated only by fixed centering and affine weights, which can be
cached. Row scale multiplies coefficients; it does not multiply factor rank.
After L layers the current implementation's rank is

```
R_L = L H(T-1) + L + 1.
```

For SAM H=8, L=2, this is 99 at T=7, 131 at T=9, and 259 at T=17.
At T=17 the factor width already exceeds D=256; low rank is not universally
beneficial. An exact per-head schedule can instead bound each write contribution
by `min(T-1, Dh)` using value coordinates before fixed `O_h`, without SVD truncation.
This is a real identity, not a guarantee that extra contractions are faster.
An arbitrary nonlinear image-side MLP would generally destroy this simple closure.

## Companion-free LN with sparse fixed-subspace cross terms

For the pre-normalized factors, centered second moments give

```
var_i = s_i^2 ||Bc_i||^2 / D
      + 2 s_i [u_i (Bc_i Vc^T)] / D
      + u_i (Vc Vc^T / D) u_i^T.
```

This avoids `E[X^2]-E[X]^2`, but the three quadratic terms can still cancel.
A naive implementation computes `B V^T` at cost N R D, comparable to dense
reconstruction. It does not establish savings. A dense zero-padded coefficient
dictionary of width K similarly introduces N K R work and can be worse.

The prototype keeps each factor's **sparse origin coordinates**. Dynamic write
rows of head h are `z_h O_h`; after later affine maps their fixed basis becomes
`M_h = C(...C(O_h) gamma...)`. Cache `Bc M_hc^T`. Then

```
Bc V_segment,c^T = (Bc M_hc^T) z_h^T,
```

whose dynamic inner width is Dh=16 instead of D=256. Constants use concatenated
head bases (width 128), output bias vectors, and preceding LN beta vectors.
`StatisticState.segments` preserves the nonzero coordinate structure; it does
not multiply a dense matrix of padding zeros. No dense image companion persists.
It still constructs the small dynamic R-by-D factor V and its R-by-R Gram matrix.

Approximate leading **MAC counts per image row**, excluding sparse Gram creation,
elementwise reductions, launches, and caches, for T=7 are:

| LN stage | Factors before LN | Current new-write reconstruction | Structured cross + row Gram quadratic |
|---|---:|---:|---:|
| 0 | 49 | 48 * 256 = 12,288 | 897 + 49^2 = 3,298 |
| 1 | 98 | 48 * 256 = 12,288 | 1,795 + 98^2 = 11,399 |

The second stage leaves only a small arithmetic margin before overhead. In
addition, Gram formation costs B R^2 D independently of N, and the prototype
launches multiple small cross contractions. At T=17, the second row-Gram work
alone is 258^2=66,564 MACs versus 128*256=32,768 for the old write reconstruction.
This is a reason to dispatch to a dense/native path for larger T, not to remove
those prompts. These counts are not FLOP-based GPU speed predictions.

For N=4096 the additional stage cross matrices have (129+260)*N FP32 entries,
approximately 6.08 MiB plus two row variance arrays. Per-head cross entries are
views of the all-head cross matrix; do not double-count their storage. Removing
the companion saves B*N*D*4 bytes of persistent state (32 MiB at B=8), while
new Gram/cross temporaries and caching change actual peak memory. Measure peaks.

**Numerical counterexample:** FP32 random `b` scaled by 100, with
`v=-b+0.01*noise`, has direct centered variance 9.62301e-5, but the algebraic
quadratic formula produces **-0.0009765625**, giving nonfinite inverse deviation.
At scale 10,000 it produces 16 instead of 9.59539e-5. The raw record is retained.
The prototype does not clamp, change epsilon, or loosen the threshold. A future
stable or fallback schedule must be explicit and have its own measured cost;
silently clamping is neither numerically exact nor a solved method.

## Implicit attention without projected dense K/V/Q

Per head, after a prior LN, denote cached base projections by `Kb,Vb,Qb`, cached
position projection by `Kpe,Qpe`, and projected factors by `Kv,Vv,Qv`.
Read attention is exactly

```
logits = (Q Kb^T) diag(s) + (Q Kv^T) U^T + Q Kpe^T.
A = softmax(logits / sqrt(Dh)).
output = (A diag(s)) Vb + (A U) Vv + value_bias.
```

The omitted key bias is constant across image positions and cancels in softmax.
Image-to-token write logits are

```
logits = diag(s) Qb K^T + U (Qv K^T) + Qpe K^T + 1 query_bias K^T.
```

Here query bias varies across sparse keys and **cannot** be dropped.
Softmax and the original centered write factors follow without change.
First-layer native-order caches remain intact, to limit avoidable float drift.
The original scale 1/sqrt(Dh) is retained even when a contraction uses factor width.

Projected factor attention spends leading N R H Dh work reconstructing each
projection; implicit contraction uses N R H T work per analogous direction.
At T=7,Dh=16 there is an arithmetic opportunity, but the implicit path exposes
extra small GEMMs and explicit softmax; it can lose SDPA fusion. At T>=Dh the
arithmetic rationale weakens or reverses. The prototype uses explicit attention
for implicit image interactions; backend="sdpa" only affects unchanged first
read/self-attention operations and does not claim FlashAttention.

## Phase layout: eliminate feature spatial reorder before hypernetwork

Both SAM transposed convolutions have kernel=stride=2, no padding, no overlap.
Their output for one parent location is exactly a linear map to four independent
child phases. First convolution produces `[parent, phase1, C1]`; channelwise LN
and GELU act independently on every phase. The second produces
`[parent, phase1, phase2, C2]`; GELU again acts elementwise. Hypernetworks give
four mask coefficient vectors, which can be applied in this layout before any
spatial reordering. Only `[four_masks, parent, phase1, phase2]` is reordered.

The final spatial index is

```
y = 4*parent_y + 2*phase1_y + phase2_y
x = 4*parent_x + 2*phase1_x + phase2_x.
```

This preserves both biases, both GELUs, LN epsilon/affines, all four masks and
IoU. It does not merge through a nonlinearity or move a hypernetwork across GELU.
For each prompt at N4096 FP32, the final 32-channel feature contains 8 MiB while
the four final masks contain 1 MiB. A required reorder therefore acts on 8x
fewer values. The feature values still exist; this is a copy/layout opportunity,
not an 8x end-to-end acceleration claim. Dense controls can use the same rewrite.

## GPU integration and first discriminating experiment

Import from this directory and use these exact cold-cache flags:

```python
# PositionCache is built separately, as in the existing benchmark.
# Factor + phase only:
cache = ResearchCache.build(model, image, dense, position,
                            include_statistics=False, include_phase=True)
out = predict(model, cache, sparse, phase_layout=True)

# Factor + implicit only, native dense LN and original head:
cache = ResearchCache.build(model, image, dense, position,
                            include_statistics=False, include_phase=False)
out = predict(model, cache, sparse, implicit_attention=True)

# Fair strong dense + phase control; select plans by the existing measured routes.
cache = DensePhaseCache.build(model, image, dense, position)
out = dense_phase_predict(model, cache, sparse, order="associated",
                           write_output="sparse", backend="explicit")

# Statistic LN diagnostic only:
cache = ResearchCache.build(model, image, dense, position,
                            include_statistics=True, include_phase=False)
out = predict(model, cache, sparse, statistic_ln=True)
```

For the dense control build `PositionCache` with its matching `shape_plan` to
avoid unused position projections, as the current benchmark already does.
The variance construction is skipped completely when include_statistics=False;
the second-convolution matrix construction is skipped when include_phase=False.
Do not build ResearchCache for a dense control; its factor-only projections and
per-image base/conv projection would bias cold-cache time and memory.

First GPU experiment: N4096,T7,P128,B8, FP32/TF32 off, same prompts/weights,
eager and true Inductor. Compare (1) strongest unchanged baseline, (2) unchanged
factor candidate, (3) strongest dense+phase, (4) factor+phase, then (5) factor
implicit-only. Record cold/first/hot phases and peak memory. A second P1/B1 checks
cache amortization. If a phase improvement holds, interpret it as common
upscaler scheduling; factor+phase must beat dense+phase to support sharing.
If implicit improves on factor under Inductor, combine with phase and recompare.
Profile the explicit softmax, small GEMMs, layouts and memory traffic separately
from timing. Retain all numerical failures under the original 5e-5 gate.

Local reproduction:

```bash
python3 demo_lists/demo1_sam/research/compute_structure/validate.py --full
```

Top-tier potential is a general execution method for fixed shared-context
decoders with sparse writes, exact LN closure and shape-dependent contraction
dispatch, supported across real trained architectures and serving workloads.
This is only a concrete research direction. A 10% synthetic decoder improvement,
CPU identity proof, or a SAM-only public head optimization does not establish
SOTA quality, end-to-end gains, or novelty against newly trained multiplex models.
