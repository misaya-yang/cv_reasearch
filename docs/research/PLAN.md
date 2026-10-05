# Pending work

Updated2026-10-05 at operational handoff. The user assigned continuous goal/execution to receiving
chat `01a10c9c-32fb-7330-be53-cdb43999fd4f`. Source chat performs45-minute supervision, correction and
optimization follow-ups. Immediate priority: finish existing600 fixed-candidate/control evaluation using
the already-running exporter. See [HANDOFF](../../HANDOFF.md). Do not duplicate its encoder work.

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
| Resumed DEV241 pipeline | Receiving chat | Supervisor2161, score watcher2162. Proposals/maps/mean complete; fixed auxiliary inference and score pending after exporter GPU. Adopt live processes; no restart. |
| Existing600 fixed candidates and controls | Receiving chat | Exporter3393 and downstream `launch/confirm600_v1/run.sh` already running. Preserve frozen Astra and DEV-selected picks; separate600 search. Same600 is previously exposed, disjoint fromDEV241 photos. |
| Supplementary fixed600 controls | Receiving chat; prepared by source chat | Dependency watcher8066/start951592385 is live, using `controller_watch_state.json` and adopted parent3391/start951484791. It starts the prepared CPU infer-cache/score after completed exporter report and recheck600 seal. Exact supplied10arms plusnative,4workers,no encoder;55 source hashes and smoke sources verified. |
| A*/B* framework and marginal-value solver | Receiving chat; current greedy script author not identified | Existing full and primitive greedy runs are complete; do not repeat them. Full raw-origin two-step61.9918 is -.6626[-1.3704,-.0051] vsAstra; primitives61.0355 adds only.0159 vsRCG. See [supervision](../../evidence/local/research_20261005/operator_supervision.md). After fixed600 comparison, decide on bounded joint moves versus one-step greedy and strongest same-information controls, with complete sealed masks and per-class marginal/overlap/conflict/cost accounting. [Framework record](../../evidence/local/research_20261005/operator_framework.md) gives the user's framework objective and counterexamples. Coordinate existing source ownership before editing; preserve completed reports. Existing caches; no new encoder for accounting; new solver runtime unmeasured. |
| Public-protocol evaluation preparation | Receiving chat | Freeze the final complete rule before scoring, verify official episode generation/seed/count, original-resolution renderer and CRF, metrics and full FoRIS/INSID3 controls. Prepare streaming inference to avoid all-feature retention; do not launch a second encoder beside existing GPU work. Current600 is prior-exposed reevaluation, not public-protocol SOTA proof. |
| E/B auxiliary accounting onDEV241 | Receiving chat | All241 proposals sealed in `outputs/edit_aux_dev241_resume_v1`; fixed auxiliary stage remains queued. No raw-origin relabeling; native alwaysFoRIS. |
| Single-encoder parallel forward/CRF pipeline | Root, implementation by `/root/parallel_forward_sol` | CPU contract checks passed. Actual CUDA IPC, serial parity and throughput remain unverified; no inference launch. |
| Public native replay on the two known differing cases | Root | Optional diagnostic, held. Existing 25-pixel total drift is reported under both baselines; it does not block analysis of the completed cohort. |
| Raw-model origin / score-sweep integration and source/statistical checks | Root integrating the user-supplied Opus work | Verify raw l24 provenance, require origin explicitly, preserve FoRIS/INSID3 as comparison methods, fix Astra seal parsing and assess search-size placebo calibration. No new semantic method is selected from preparation. |
| Sealed auxiliary sweep onDEV241 | Receiving chat |206maps completed and scored in `outputs/aux_dev241_resume_v1`; record exact report and assess complete score, not map significance. No advantage overRCG established. |
| Raw-model origin producer, INSID3 and FoRIS rebuilt stage by stage on it, and the family containing both (weights, scaling, cut, renderer), with the choice nested over folds | Claude Code | Ran on DEV241 as `outputs/stage241_v1` (sealed; origin arm `model.raw_nn` available to the other runners). Family search does not beat complete FoRIS; see the ledger. Closed |

Astra addition/deletion variants have now been tested onDEV241 in recheck241_v1. Preserve the fixed
Astra identity and distinguish nested/best-in-sample search. No prepared code or GT oracle is a method result.

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
