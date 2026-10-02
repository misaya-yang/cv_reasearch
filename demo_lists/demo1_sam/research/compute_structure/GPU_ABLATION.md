# Phase layout and implicit attention: bounded GPU run

Run from the data-disk task root. Existing output directories are rejected;
raw JSON/log files are never replaced. Manifest updates belong only to the newly
created round. No dependencies are installed.

```bash
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/compute_structure/run_gpu_ablation.py
```

Defaults: compile with real Inductor/fullgraph, P128, microbatch 8 and 128,
N4096/T7/seed0, FP32 with TF32 disabled, warmup10, repetitions20, timeout300s per
job. The 26 jobs run sequentially: first 22 phase/control jobs across both batches,
then four implicit jobs reusing this round's matching controls. Full decoder timing
includes all four masks and IoU; encoder and final resizing are outside scope.

Per batch the phase stage measures:

| Role | Method / route |
|---|---|
| Current practical control | cached_merged explicit and SDPA |
| Merged candidate control | factor_merged explicit |
| Original strong dense control | dense_assoc explicit associated/sparse and projected/dense |
| Original factor control | factor_projected explicit |
| Common phase head | dense_phase explicit auto/auto, associated/sparse, projected/dense |
| SDPA phase representative | cached_phase SDPA |
| Factor phase candidate | factor_phase explicit |

The implicit extension adds factor_implicit explicit and factor_implicit_phase
explicit. Their image interactions are explicit contractions, with no implied
FlashAttention claim. cached_phase only calls SDPA on eligible operations.
The auto route is measured rather than assumed fastest from its arithmetic model.

To run smaller independent rounds:

```bash
# 22 jobs; distinct output directory required.
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/compute_structure/run_gpu_ablation.py --stage phase \
  --output-dir results/gpu_layout_implicit/phase_round

# 16 jobs, including six same-round controls per batch.
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/compute_structure/run_gpu_ablation.py --stage implicit \
  --output-dir results/gpu_layout_implicit/implicit_round

# Print 26 planned jobs without importing torch, touching GPU, or creating files.
python3 research/compute_structure/run_gpu_ablation.py --dry-run
```

Numerical failures retain raw errors under the unchanged 5e-5 gate and skip
timing in that child. Timeouts kill only the launched job's process group and keep
its log/partial JSON. Other cases continue. CUDA OOM is reported without silently
changing batches; rerun every arm at a common smaller batch in a fresh directory.
Only complete groups with matching model SHA and passed numerical checks are
eligible to establish a winner. A successful subprocess with mismatched precision,
seed, dimensions, batch, compiler, method, or warmup settings is rejected.

## Registry/cache review

The current registry matches the independent CPU prototype:

- factor_phase and factor_implicit_phase use ResearchCache with statistics off
  and the second-convolution phase matrix on.
- factor_implicit has both optional cache extensions off.
- dense_phase/cached_phase use DensePhaseCache, without factor projections,
  companion-free statistic caches, or image-dependent factor conv bases.
- Dense PositionCache follows the selected shape plan, avoiding unused PE
  projections on associated routes.
- Merged controls retain MergeCache and their own projection schedule.

`registry_review.json` records one CPU FP32 random fixture across all 13 arms:
13/13 pass, maximum mask error 2.38419e-7, IoU error 1.11759e-7. This verifies
adapter/cache plumbing and is not a GPU measurement. The runner's dry plans,
raw-overwrite refusal, and fabricated numerical-failure preservation were also
checked locally; fabricated harness checks are not model-validation results.

No registry adaptation bug was found. There is a **follow-up comparison gap**:
cached_merged currently lacks the common phase head, while cached_phase uses
unmerged projections. If the phase head improves runtime, combine it with the
fastest merged dense baseline before claiming final net sharing gains. Likewise,
factor_merged_phase may combine complementary improvements. This round decides
whether those combinations are worth implementing; it does not assume they help.
