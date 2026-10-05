# Pending work

Updated 2026-10-05 after the user's GPU-idle correction and supplied Opus/Astra evidence.
**Current instruction: continue method design, improvement, combination and experiments.** The user
sets the common edit origin to direct frozen-DINOv3 reference matching, not FoRIS. The user resumed pipeline work after improving SSH and will enable the existing GPU.
Root owns the isolated `code_resume_v1` / `launch/resume_dev241_20261005_v1` queue; execute only with existing available resources.
Do not rent, extend, download or run heavy work in no-card mode. Completed runs live in the evidence ledger.

## Target and comparison

One fully masked reference and the same frozen DINOv3 should recover missed true targets and remove
false targets. The primary edit origin is raw final-layer cosine nearest-reference-token label transfer;
raw foreground-minus-background mean matching is a second explicitly named definition. FoRIS, INSID3,
RCG and all other mechanisms are comparison rows, not mandatory foundations. Improve the complete
score as far as the evidence supports, retaining the working target of stable >=2 class-summed mIoU
points over complete same-protocol FoRIS at 1024 and improvement over INSID3 and strong controls. Any number of components, iterations
or feedback stages is allowed. A useful proposal is not discarded because its standalone side effects
make net gain negative. Do not silently replace a supplied method with a control or a gating variant.

## Pending ownership

| Work | Owner | Status and authorization |
|---|---|---|
| Resumed DEV241 pipeline | Root | Isolated source snapshot and GPU/CPU plans staged. GPU order: complete E/B proposals, auxiliary evidence, mean-graph complete control, fixed auxiliary edits. CPU scoring is separate so it can overlap later GPU work. Supervisor PID 5561 is verified live in WAITING_GPU_AVAILABLE; 51 source hashes and all three source cohort manifests checked. No new scientific stage started. Preserve other workers and the supplied fixed Astra package. |
| Faithful Astra candidate/controls on DEV241 | Root | Source, identity/statistic audit and launch command prepared locally; execution held. See [intake](../../evidence/local/research_20261005/astra_intake.md). CPU-only; no encoder or GPU required. |
| E corrected/fixed/two-slot and exact B kernel on DEV241; proposal and auxiliary accounting | Root; code by `/root/edit_auxiliary_suite_sol` | Adapting the owned runner to an explicit sealed model-origin mask; keep native=FoRIS and scientific proposal functions unchanged. Proposal-only sealing can feed the sweep independently. [Readiness record](../../evidence/local/research_20261005/edit_auxiliary_ready.md) is being updated. Real preflight/full241 execution remain unverified. |
| Single-encoder parallel forward/CRF pipeline | Root, implementation by `/root/parallel_forward_sol` | CPU contract checks passed. Actual CUDA IPC, serial parity and throughput remain unverified; no inference launch. |
| Public native replay on the two known differing cases | Root | Optional diagnostic, held. Existing 25-pixel total drift is reported under both baselines; it does not block analysis of the completed cohort. |
| Raw-model origin / score-sweep integration and source/statistical checks | Root integrating the user-supplied Opus work | Verify raw l24 provenance, require origin explicitly, preserve FoRIS/INSID3 as comparison methods, fix Astra seal parsing and assess search-size placebo calibration. No new semantic method is selected from preparation. |
| Sweep of every sealed proposal and every enumerated label-free map as additions and deletions of the host mask (A + B + auxiliary), with noise placebos and a fold-nested composition | Claude Code | Code prepared; count self-check and a 20-case local execution smoke passed; not launched, no efficacy result. Reads sealed proposals and recomputes none (the four DEV241 arms now; the suite's E/B proposals and the Astra replay once sealed). 197 maps with the saved layers 16/24, 666 with one extra paired forward per episode; about 14,600 rows with ten proposers. One GPU process for the maps, CPU-only scoring that can run beside other GPU work. Remote execution held. See [record](../../evidence/local/research_20261005/aux_evidence.md). |
| Raw-model origin producer, INSID3 and FoRIS rebuilt stage by stage on it, and the family containing both (weights, scaling, cut, renderer), with the choice nested over folds | Claude Code | Ran on DEV241 as `outputs/stage241_v1` (sealed; origin arm `model.raw_nn` available to the other runners). Family search does not beat complete FoRIS; see the ledger. Closed |

No new combination of Astra and an addition proposal has been tested. Specify the complete operation
and controls before inference; do not describe prepared code or a GT oracle as an achieved method.

## Evaluation contract

Use existing DEV241 for efficacy; 20/50-case sets are execution checks and cannot reliably select +2.
All 220 exposed cases and all reused 241 remain DEV, not independent confirmation. Freeze all requested
predictions before scoring. Parameters are fixed or selected on three folds with the fourth read out;
query GT is only for scoring/labeled diagnostics, never per-example selection. Report class-summed I/U,
paired 95% intervals from 2,000 RandomState(0) photo-connected draws, folds, batches, episode up/down/tie,
add TP/FP and delete TP/FP. Four GT error categories are diagnostic only. A pooled 38%/62% purity
threshold is not a class-macro decision rule. CI crossing zero means unresolved, not automatic rejection.

## Execution preparation

Prepare follow-on work before an authorized run. Use CPU evaluation and model-free algebra alongside
one shared DINO encoder when appropriate; do not keep a GPU active for CPU-only replay. Full241 Astra
runtime is unmeasured, so do not extrapolate a promise from its supplied five-case smoke. New agents
explicitly requested for research use GPT-6.1-sol/xhigh. No downloads, rental, extension, commit, push
or shutdown are authorized. Follow current [SERVER](../harness/SERVER.md) before any resumed remote work.
