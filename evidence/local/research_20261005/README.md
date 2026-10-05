# Six complete mechanisms: prepared GPU batch

Status: **code prepared and staged; not launched; no new segmentation result**. The user stopped CPU experiments
and will enable the existing machine's GPU. No rental, extension, download, commit, push or shutdown was
performed or inferred. Code syntax checks are static and do not validate GPU execution.

## Available assets and cohort

Read-only SSH succeeded at `ssh -p 48002 root@connect.westd.seetacloud.com`. Effective no-card quota is
0.5 CPU / 2 GiB, despite host-level 128 CPU / 503 GiB output. The no-card nvidia-smi stub does not establish
GPU availability. Expected 32GB GPU / 12 CPU / 60GB RAM must be rechecked after startup.

[server_inventory.json](server_inventory.json) records existence of all 241 feature/packet, reference/query
RGB and mask paths, the 1,212,347,640-byte DINOv3 weight file, native positional basis, public FoRIS source
and CRF runtime directories. No experiment process was observed. Contents and native runtime parity are
not proved by existence. Shared input locations remain unchanged.

[cohort_audit.json](cohort_audit.json) records 241 unique episode identities and **239 photo-connected groups**.
The shared-photo pairs are `0_6_60 / 0_34_76` and `3_18_47 / 3_38_47`. Old120 support/query identities match
the imported record; new100 matches exported keys, whose cloud photo manifest is not present. Batch labels
are old20/new40/new60/arrived100_exposed/remaining21_existing_dev. All are development data. The 220 exposed
cases and the other already-existing DEV cases are never described as independent confirmation.

Prepared manifests: `smoke4.json` (one per fold across four exposed batches), `first20.json` (five per fold,
four batches), `exposed220.json`, and `dev241.json`. Selection uses metadata/source order only. Do not
silently drop missing inputs or replace failed episodes. A 4- or 20-case run checks execution and gives an
initial effect observation; it cannot establish stable +2 mIoU.

## Independent mechanisms and strongest simple controls

| Owner/module | Changed inference | Simple comparison | Additional encoder work |
|---|---|---|---|
| A `transport` | Each reference appearance competes with fixed opposite-role counterparts | Query-local foreground/background kNN contrast | None; cached last layer |
| B `structure` | Iterated second-order cross-image correspondence compatibility | Independent transfer, same-graph diffusion, entropy-matched independent transfer | None; cached last layer |
| C `intervention` | Reference-conditioned routing inside frozen block 19 (sixth from end) | Equal-RMS key-only routing, native and zero-bias replay audits | One paired forward per arm, no extra natural images |
| D `multilayer` | Layer16-to24 transition descriptors with complete FoRIS | Same-layer concatenation with identical complete readout/CRF | One shared paired layer extraction; one black-image forward and two SVDs once |
| E `latent` | Joint reference-constrained query slot identity/appearance updates | Fixed identity slots and two-slot EM | None; cached last layer |
| F `reconstruction` | Local convex reference explanations tested on other feature coordinates | Uniform combination of the identical neighbors | None; cached last layer |

Each owner read the failure ledger, implemented a complete mechanism and challenged other owners.
Per-mechanism English records are [A](transport.md), [B](structure.md), [C](intervention.md),
[D](multilayer.md), [E](latent.md), [F](reconstruction.md). File A's name preserves ownership, not an OT claim.
No mechanism is selected for the paper. Extra layers/descriptor channels/SVDs are explicitly part of D's
resource setting; gain over native alone cannot identify a mechanism advantage over concatenation.
F's uniform control has less arithmetic, so it is not claimed to have identical FLOPs. B's entropy control
currently repeats the primary computation and records its own total cost. C/D share one model instance.

A/B/E/F return a complete bilinearly rendered 1024 mask (no CRF claim). C/D retain native FoRIS CRF.
All compare with cached complete native, RCG and their own simple controls on the exact same cohort.
The new RCG module preserves the historical equations but moves dense products to GPU; CUDA numerical
parity is unverified and must not be asserted from the historical CPU result. No new parameter fitting occurs.

## Prepared execution

Remote owned code directory: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9`.
Shared assets: `/root/autodl-tmp/demo9_extent`, `/root/demo4_cache`, existing public FoRIS/CRF directories.
No experiment is running. Commands below are for **after the user enables GPU mode**.

```bash
cd /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9
/root/miniconda3/bin/python scripts/experiment_resource_guard.py \
  --plan launch/gpu_smoke_v1/plan.json --state-file launch/gpu_smoke_v1/guard.json --run
```

Both remote plans passed static preflight (21 checked files each); see [staging receipt](staging_receipt.json).
The static preflight receipt must match this exact staged code and plan; an absent/stale receipt refuses
execution. When code changes, regenerate it with the same command plus `--preflight-only` instead of `--run`.
No `--allow-shutdown` is set. The smoke ceiling is 30 minutes, not an estimate or rental authorization.
A separate `launch/gpu_first20_v1/plan.json` has a 60-minute ceiling and is **not automatically run**.
Inspect real time, native parity, errors and first effects before proceeding. Larger manifests are prepared,
but no 220/241 queue is armed. Actual price, GPU time and memory are unknown until a real GPU run.

`run_gpu_batch.py` refuses no-card quotas, checks foreign GPU processes, and executes:

1. A/B/E/F and RCG dense cached-feature algebra on GPU, sequentially in one process with two CPU threads.
2. After its predictions are sealed, CPU scoring runs alongside the C/D GPU worker.
3. One frozen DINOv3 instance runs all C/D/native controls sequentially. C's native and masked-zero audit,
   D's native layer/readout replay, and cached-native identity must agree before valid same-protocol scoring.
4. The forward worker exits; CPU scores its sealed outputs, and the finite batch stops for interpretation.

C/D ordinary episodes use four paired forwards total (public native, key-only, C, shared D extraction).
The first episode uses seven including identity, masked-identity and layer-tap replay audits; the setup
black-image forward is separate. Every arm and setup records actual time; C also records CUDA peak allocation.
The per-process CUDA allocation cap is 85% with foreign jobs excluded. It is not a memory-use prediction.

## Evaluation and persistence

`ics.experiment` and `score_forward_run.py` score full 1024 masks by per-class summed intersection/union,
then macro-average. Inference receives reference coverage and no query GT; all predictions are sealed with
SHA256 before evaluation. Saved records contain packed masks, score fields, inputs/config/code hashes,
per-arm timing and per-episode I/U plus add_TP/delete_FP/delete_TP/add_FP. Reports contain paired gains,
2,000 photo-connected RandomState(0) bootstrap draws, fold/batch scores and gains, up/down/tie and pixel
corrections by episode/class/batch. Missing or changed predictions fail rather than disappearing from the cohort.

Do not promote an oracle or field statistic to a complete-method result. A CI crossing zero is unresolved,
not automatic rejection. Batch degradation remains visible. After the first real results, locate whether a
failure arose in transferred identity, full extent or finalization; do not rename a parameter tweak as a new mechanism.

## What was and was not checked

Before the user's stop-CPU correction, four owners timed one local Mac episode without query truth;
A's initial absolute reference rejection produced all-background output, then changed to relative comparison.
That failed construction is preserved in A's module/record. No cohort score or GPU inference was run.
After the correction, only file preparation, read-only asset inventory and static source checks occurred.
`static_checks.json` is a syntax receipt only. Real model hooks, GPU kernels, sparse CUDA operations, CRF,
all complete-mask gains and the target +2 mIoU remain unverified.
