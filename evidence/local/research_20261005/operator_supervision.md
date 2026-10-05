# Operator-family supervision — 2026-10-05, 15:27–15:30 UTC

Read-only review of live outputs and current code. No new experiment, code modification or remote
process mutation was made. Continuous execution remains with chat01a10c9c-32fb-7330-be53-cdb43999fd4f.

## Existing work; do not duplicate

`run_operator_greedy.py` already ran twice. No matching live process was observed. Reports:
- `outputs/greedy241_v1/report.json`, SHA25604d664ff46650a01e477c8c60f826a9094713e4e4ba70862a6a8c1db2a43f962.
- `outputs/greedy241_primitives_v1/report.json`, SHA256436ead5dd6fd93ca18b580b3cc0b8e2bdd8f0b8a1ad9fac78002f549f4179b32.
Copies: `pipeline_verified/operators/full_library_report.json` and `primitives_report.json`.
Logs: `launch/greedy241_v1/stdout.log` and `primitives.log`.
Current remote source SHA2567575ff27cb0ec0440ec046dbf34f43d1013ba127e47f06e7dc06854f593628d5.
The full report lacks the later `offered` field; its exact source revision is not receipt-bound.
Authorship is not declared in PLAN; presence and modification times do not prove the author.
The receiving chat owns any next execution; coordinate source edits with its current author.

Reported DEV241 results (seed0,1024,239 photograph groups; no fresh confirmation): full25-mask library
from raw NN gives61.991777 at two steps, -0.662566[-1.370401,-0.005129] versus fixed Astra62.654343.
From native the two-step result is62.019717, -0.634626[-1.287770,0.055597] versus Astra.
The21-offered primitive library reaches61.035531 from raw/native and RCG, only+0.015862 above RCG.
These step counts are the best observed rows, not independently validated complexity choices.
Paths often intersect and then union the same complete mask, which simply replaces the starting mask.
No new superiority follows from relabeling that replacement as a two-operator construction.

## Inference and evidence boundaries to preserve

The current greedy implementation optimizes class mIoU using fitting-fold GT and reads held-out folds.
It is DEV-supervised method/path selection with frozen-backbone inference; explicitly describe this when
using the term zero-training. This does not require changing the already-authorized fold-selection protocol.
Input key order is checked, but source prediction hashes, photo identities and code versions are not fully
checked. Only I/U paths are saved, not the final held-out mask artifacts. Any chosen deployable recipe must
be frozen and rendered/sealed before its600 use. Counts in greedy() currently aggregate all rows whereas
fit_before/fit_after use the fit subset. Pooled purity/break-even is a descriptive statistic, not the exact
class-macro objective used for operator selection. A best step chosen from the displayed held-out curve is
still DEV selection. Existing report paths should be preserved, not overwritten for these repairs.

## Exact counterexamples; no general greedy/sparsity theorem

For IoU with G={t1,t2,t3}, start S={t1}; let F be four false pixels,
A={t1,t2} union F and B={t1,t3} union F. J(S)=1/3. Each single union has IoU2/7<1/3;
each intersection leaves S unchanged. Thus a strictly improving one-step union/intersection greedy stops,
yet S union A union B has IoU3/7>1/3. This directly applies to the current offered operator algebra.
It proves that failure of this greedy construction does not reject the useful joint-composition framework.
A bounded two-step lookahead is a justified optimizer control; do not infer success before complete results.

Overlap does not force decreasing marginal purity: A={t1,f1}, B={t2,f1} each has purity1/2,
but B minus A is pure foreground. Conversely overlap can remove the true part first. Direction requires
assumptions on conditional overlap, not just overlap area. Probability threshold p>J/(1+J) is equivalent
to odds p/(1-p)>J for a calibrated ratio-of-expectations surrogate; this is not automatically an exact
expected-IoU optimum or a threshold on an uncalibrated matching score.

## Next ownership

Receiving execution chat should reuse these completed results, check the600 fixed comparisons first,
and decide whether a bounded joint-lookahead comparator addresses the failed selection link. Source
chat has no new run queued. Keep any broader family/optimality/SOTA claim separate from these DEV data.
