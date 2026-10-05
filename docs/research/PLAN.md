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
| Seven-layer complete edit evidence | Receiving chat | Following completed600 and bounded joint-search results, test whether new layer information changes edit selection. Existing runner `run_aux_evidence.py --forward-layers 4,8,12,16,20,22,24` uses one encoder and one paired forward per episode plus one black-image setup. Real2-case execution check, then DEV241 complete masks versus same-source unfiltered/simple/noise controls,RCG,mean,Astra. Freeze maps/proposals before CPU fold-nested selection/scoring. Shared rawNN origin, existing sealed proposals; no extra natural images. Same-layer/nearest/prototype/grouping simple maps retained. Cost unknown until smoke; no efficacy claim from map count or extra compute. |
| A*/B* framework and marginal-value solver | Receiving chat; owns new `run_joint_operator.py` only | Existing greedy runs preserved; they do not beatAstra. Next authorized run is an exact bounded<=2-operation search versus one-step and direct complete-mask selection on the same sealed library (stages,RCG/C/Astra,ordinaryE/B/D,mean graph,fields). It addresses the demonstrated greedy joint-move failure; DEV241 with three-fold GT selection/fourth-fold readout excluding shared-photo groups. Freeze candidate recipes/source hashes, save complete selected masks, then independent CPU score. Real2-episode execution/kernel check precedes immediate241. No encoder or extra images; CUDA packed-byte algebra, CPU scoring beside600 controls. Runtime/memory unmeasured until smoke. If it beats direct selection/strong controls, freeze a deployable rule; otherwise identify selection versus attainable-library limitations rather than rename the construction. [Framework](../../evidence/local/research_20261005/operator_framework.md), [prior supervision](../../evidence/local/research_20261005/operator_supervision.md). |
| Public-protocol evaluation preparation | Receiving chat | Official commit1aa02a11 uses1000/fold,4fold,seed0. Paper says original-resolution; current CLI sets resize_to_orig_size=False and resizesGT to1024 predictions. Bind each grid/original renderer to its version and retain both named readouts; see [protocol audit](../../evidence/local/research_20261005/public_protocol_audit.md). Verify sampling/worker RNG, encoder,CRF and full FoRIS/INSID3 controls before a real smoke/full streaming run. Freeze the final rule first; audit all previously inspected photo overlaps. Current600 is prior-exposed reevaluation, not SOTA proof. |
| Single-encoder parallel forward/CRF pipeline | Root, implementation by `/root/parallel_forward_sol` | CPU contract checks passed. Actual CUDA IPC, serial parity and throughput remain unverified; no inference launch. |
| Public native replay on the two known differing cases | Root | Optional diagnostic, held. Existing 25-pixel total drift is reported under both baselines; it does not block analysis of the completed cohort. |
| Raw-model origin / score-sweep integration and source/statistical checks | Root integrating the user-supplied Opus work | Verify raw l24 provenance, require origin explicitly, preserve FoRIS/INSID3 as comparison methods, fix Astra seal parsing and assess search-size placebo calibration. No new semantic method is selected from preparation. |
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
