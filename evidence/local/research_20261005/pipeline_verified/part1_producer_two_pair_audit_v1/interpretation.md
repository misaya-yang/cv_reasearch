# Two-pair Part1 producer verification

Independent CPU-only review,2026-10-05. **The completed benchmark supports a
faithful, faster q/r cache-reconstruction prefix on its two tested pairs. It
does not measure a faster cold-query complete method.** No GPU rerun, source
change or queue change was performed for this review.

The records cover `public0:0_0_72` and `public0:1_0_73`, different folds/classes
but the same public block. Each has three alternating uninstrumented comparisons
after one warmup per producer, plus one separate instrumented comparison.
Every reported FP16 q/r comparison among prefix, actual full producer and
retained cache has zero differing entries. Gate flags agree and are **true for
both pairs**; the gate-off branch is not empirically covered. Prefix coverage
matches both full coverage and stored coverage exactly. Actual full score has
maxdiff0 against the source; the genuinely recomputed full native mask has
zero differing pixels. The prefix neither runs nor substitutes a native mask.

For both pairs, prefix and full q/r supplied to the fixed CPU lambda16 readout
produce **zero FP32 field differences and zero final-mask differences** against
the independently sealed source, with stored cov/score as inputs. CG residuals
are9.83e-8/7.16e-8 and iteration counts59/56; the source implementation rejects
nonconvergence before returning. This review did not rerun that numerical
computation: it checked its recorded comparisons and independently re-read the
retained FP16 cache/field arrays. q/r byte digests match prefix output digests;
both returned field digests match the retained source field. Feature/packet,
source field/prediction, provider manifest/seal, config/evidence and all recorded
code hashes also match. The evidence embedded in report.json equals evidence.json.

| Pair | Full warmed mean [min,max], seconds | Prefix warmed mean [min,max], seconds | Mean saving |
|---|---|---|---|
| fold0,class72 | 1.416546 [1.408743,1.425820] | 0.509393 [0.508288,0.510980] | 0.907153s,64.04% |
| fold1,class73 | 1.327252 [1.269087,1.437413] | 0.508589 [0.506836,0.511354] | 0.818663s,61.68% |

These are three CUDA-synchronized, **uninstrumented producer-call repeats**,
including source transforms/setters, reference-mask resize, paired encoder,
normalization/gate/projection, output conversion/transfer and cleanup. RGB and
reference-mask file decoding plus cache/source reads happen outside the timer
(separately recorded0.0202/0.0139s). Model/host setup, validation/hashing, CPU
lambda16 solve (about0.83s), fine-feature inference, output sealing/writing,
scoring and4000 throughput are excluded. Do not call these cold latency or
apply the percentages to complete-method runtime. Three repeats/two fixed
pairs are not a representative runtime distribution or an uncertainty interval.

The separately instrumented full calls total1.407551/1.442778s; prefix calls
total0.508147/0.513918s. Their inclusive stage measurements locate substantial
cost in Part3 clustering (0.338889/0.385764s), Part4 correction
(0.340275/0.384241s), and native finalization (0.136917/0.152112s). The shared
encoder is about0.467s in full and0.467/0.471s in prefix; Part1 is about0.0016s.
These profile values explain the tested construction's removed work, but they
are not exclusive stage times. Setters include `_transform`; Part1 includes
`_debias_features`; Part2 includes its stage1/stage2 calls. **Do not sum nested
profile measurements or replace the three uninstrumented repeats with that
sum.** Native finalization includes its source CRF/other finalizer work; it is
not a separately isolated CRF timer.

The crucial deployment boundary is the response source. The prefix returns
q/r and ancillary gate/cov/target information, **no continuous response score**.
The lambda16 test deliberately uses the sealed score/cov. That cached score is
the original FoRIS response after Part2 background suppression, Part3 and
Part4 corrections, tapped before binarization. A cold query using the unchanged
RCG host input must still generate that continuous response. The benchmark
therefore supports avoiding unnecessary *reconstruction* of Part2-4/native
finalization when those packets already exist; it does not establish that
skipping allPart2-4 yields a complete producer. Even though prefix cov is
recomputed and checked, the readout test uses cached cov.

Recorded runtime is Torch2.12.1+cu130 on RTX4080 SUPER, two CPU threads, paired
FP32/no-autocast encoder and FP16 q/r serialization. Actual FoRIS, legacy host
builder, immutable benchmark/readout code and host manifest hashes matched on
review. Basis SHA is
`9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116`.
Model-weight content and all historical runtime packages are not independently
identified by this compact config; direct two-pair numerical agreement is the
stronger evidence here, limited to these pairs. It does not establish parity
over the gate-off branch or all1200/4000, lambda64/fine efficacy, or a full-method
speed claim.

Source records are remote
`outputs/part1_producer_two_pair_v1/{report,evidence,config}.json`. Their digests,
retained-byte checks and exact timing samples are saved in
[verification.json](verification.json). Source implementation:
`scripts/benchmark_part1_producer.py:35-166,175-272`; its current local bytes
match the hashed immutable runtime snapshot. Query GT was never indexed by
this benchmark or this independent review.
