# Experiment handoff — 2026-10-05

This is an operational snapshot, not another project agreement. Read AGENTS.md and the current
STATUS/CLAIM/PLAN first. The user assigned continuous execution and its goal to chat
`01a10c9c-32fb-7330-be53-cdb43999fd4f` (了解并接管 Astra 实验). The source chat
`01a10ba9-b2d5-7f62-ac5b-79bb8e1ba143` will supervise every 45 minutes, coordinate corrections
and propose evidence-backed next optimization. Do not create two experiment controllers.

## Objective and immediate priority

Preserve one fully annotated reference, the same frozen DINOv3 and the common raw matching origin.
Improve complete masks through recovery and deletion; target stable at least +2 class-summed mIoU
on DEV241 over complete FoRIS and superiority to INSID3 and strong same-information controls.
The user now prioritizes the prepared 600-episode evaluation. No method success is claimed at handoff.
The receiving chat has confirmed that its persistent goal is established and it is now execution owner.
That goal includes bringing the existing
600 queue to complete fixed-candidate/control results and deciding the next actual method improvement.

## Live execution to adopt, not restart

SSH: `ssh -o BatchMode=yes -o ConnectTimeout=10 -p 48002 root@connect.westd.seetacloud.com`.
Root: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9`.
As checked 15:14 UTC: RTX 4080 SUPER 32760 MiB, 12 CPU quota, about62GiB RAM.
Verify process command line and start ticks before any action; boot_id may stay unchanged across
container restarts, so it alone is insufficient. A failed observation is not a failed experiment.

| Work | Live handle / artifacts | State at handoff preparation |
|---|---|---|
| Existing600 cache exporter | PID3393, parent shell3391; `launch/confirm600_v1/run.sh`, `stdout.log`; `outputs/confirm600_root` | At least325/600 logged, first325 native masks bit identical. Running; inspect newest count. |
| Existing600 downstream | Same shell; `run_recheck_sweep.py infer --expected600` -> `sweep --grid wide --suffix _wide`; `outputs/recheck600_v1` | Already queued after export. `outputs/recheck600_v1_frozen.json` / local `evidence/local/research_20261005/recheck600_frozen.json` hold pre600 fixed picks; verify their identity. Separate fixed picks and Astra from tuning on600. |
| Root DEV241 GPU queue | supervisor2161/start951436190; `launch/resume_dev241_20261005_v1/gpu_state_boot2.json` | Proposals,206-map inference,mean_graph complete. Fixed auxiliary edits waiting for exporter GPU. Do not launch a duplicate. |
| Root DEV241 CPU score watch | PID2162; `score_watch_state_boot2.json`, individual scoring logs in same launch directory | Auxiliary evidence score and mean score completed; fixed edit score follows its seal. |
| Duplicate600 end-to-end queue | `launch/frozen600_v1/superseded_before_execution.json` | Stopped owned supervisors4050/4051 before any scientific worker started. Do not resume; existing exporter covers the same600. |

The earlier supervisor5561 was lost on container restart. `gpu_state.json` is stale; the active
DEV241 state is `gpu_state_boot2.json`. Completed workers are retained in `owned_processes`; always
inspect actual /proc identity. Other-owner scripts live at root; root-owned immutable snapshots are
`code_resume_v1` and `code_frozen600_controls_v1`. Never overwrite a running snapshot.

## Cohort and scientific evidence

`evidence/local/research_20261005/existing600.json` matches all600 identities in the active
`demo9_transductive_ics/results/extent_head_t1_isolated_v1/confirm_episodes.json`:150/fold,80classes,
no shared reference/query photographs withDEV241. All required images/annotations exist.
These600 were previously used by other experiments. Call this existing600 reevaluation, not fresh confirmation.
The original shared DEV241 root had zero of these600 features/packets; the live exporter is filling them.
Its preserved complete comparator is `demo9_transductive_ics/results/foris_confirm_v1/packets`.

Verified completed report: `evidence/local/research_20261005/pipeline_verified/recheck241/report.json`.
Astra `external_mean__delete`:62.654343 vs FoRIS59.074825; +3.579518[1.320378,4.824421].
Vs RCG61.019669:+1.634674[-.348770,2.523257]; vs priorC61.841782:+.812561[-.436814,1.636112];
vs same-count RCG deletion:+.862486[-.175978,1.847028]. Fold gains vsFoRIS:4.3252,2.2171,4.2048,3.5706.
Last-step GPU replay is pixel-identical on all241. Control superiority remains unresolved.
`report_wide.json` full nested combination62.893198 is exploratory, not a replacement for the frozen primary.
Auxiliary map scan and mean-graph complete reports are on the server; the receiving chat already inspected
them (auxiliary combination61.09, only about+.07 overRCG). Fetch exact numbers/intervals before recording.

## Fully prepared supplementary fixed controls, no extra encoder

The existing recheck runner computes but discards some original arms and omits MEAN_CONTROL from its main
comparison table. Do not declare all strong controls covered merely because the pipeline is green.
`scripts/run_frozen_cohort.py infer-cache` reuses the completed600 exporter and calls the supplied
`external/astra_emd/external_mean_delete600_modules/run.py::predict_one` unchanged, preserving every
fixed arm:RCG,MEAN_CONTROL,insid3_default,source_contrast_mean,scalar_contrast,RCG_count_matched_delete,
MEAN_count_matched_delete,external_mean__delete,external_mean_delete_sameK_RCG,native_pre_from_score,
plus complete native. Inference never indexes query truth. Its separate scorer checks all seals, reads
query labels only for scoring, retains per-episode I/U and edit counts, and uses the supplied paired
2000 RandomState(0) photograph bootstrap. No threshold fitting or best-row choice on600.

Prepared local and remote `launch/frozen600_controls_v1/plan.json` has two CPU stages: exact fixed controls,
then score. Requires completed600 cache report AND existing recheck600 inference seal, so it will not
compete with the11-worker initial replay. Snapshot `code_frozen600_controls_v1` is separately hashed.
Launch only after these dependencies exist, using fresh state/log names:

```bash
cd /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9
/root/miniconda3/bin/python code_frozen600_controls_v1/scripts/experiment_pipeline.py \
  --plan launch/frozen600_controls_v1/plan.json \
  --state-file launch/frozen600_controls_v1/state.json \
  --log-file launch/frozen600_controls_v1/events.jsonl --run
```

This adds CPU replay for missing controls; it never repeats feature extraction. The cached INSID3 arm is
explicitly the supplied complete rule, not a bitwise claim about the separate official BF16 encoder path.
The existing600 cache lacks raw-origin masks; this runner reports that absence instead of relabeling APD
features as raw DINO. Raw-origin analyses onDEV241 remain available from sealed stage241_v1.

## Validation and known issues / current breakage

- Source parsing, CLI and plan generation passed. All55 frozen600-controls source hashes were verified remotely.
- Two actual cached DEV241 pairs are linked in `outputs/fixed_controls_smoke2_cache_v1` solely for a
  query-label-free execution check. Completed:all11arms sealed,5shared fixed arms pixel-identical on both episodes, separate scorer exits0
  with n=2. Output `outputs/fixed_controls_smoke2_v1`; log `launch/frozen600_controls_v1/smoke2.log`;
  parity receipt `launch/frozen600_controls_v1/smoke2_parity.json`. This verifies execution, not efficacy.
- The new full encoder route in run_frozen_cohort.py is only prepared and NOT tested or run. Its old
  frozen600_v1 queue was superseded. Use infer-cache with the live exporter, not the redundant route.
- In existing recheck output, any nested selection trained on600 or best_in_sample row is600 development.
  Frozen picks and the supplied Astra arm must have their own unchanged names/results.
- Current STATUS/PLAN must be kept synchronized by the receiving owner; timestamps/counts above are snapshots.
- The CPU scorer watch is real and has run scoring. No-card mode gates it below8cores/16GiB. No idle-kill,
  rental, download, shutdown, commit or push is armed by these queues.

## Next actions

1. Receiving goal and ownership are confirmed. Cache export subsequently completed600/600,600 inference
   began and DEV241 fixed auxiliary edits resumed. Revalidate live identities and continue from outputs,
   not from another encoder launch.
2. Complete smoke2 parity and supplemental controls; arrange the prepared CPU-only controls plan after
   recheck600 seals. Record complete600 estimates, paired intervals, folds and edit side effects.
3. Compare fixed Astra and frozen DEV-selected picks with completeFoRIS,RCG,INSID3,mean and same-count
   controls. Identify the actual failing link before another method change. Improve onDEV, not by tuning600.
4. Keep GPU/CPU stages ready and overlap independent work where useful. Never manufacture GPU activity.
5. Source-chat45-minute heartbeat (`cvpr2027-45`) checks evidence, sends corrections/optimization tasks
   to the receiving chat, and stays quiet when nothing meaningful changed. The receiving chat owns execution.

## Files, logs and Git

See REPO_MAP.md for entry points. All work remains in the shared checkout. No handoff commit was made:
AGENTS.md requires an explicit commit request, and the user requested operational handoff/scheduling only.
Existing unrelated edits, reports, imported archives and other workers' scripts were preserved. Existing
ignore rules already cover tensors, archives, caches and secrets; no .gitignore edit was needed.
No preexisting agent_logs/current.md existed to rotate; the initial handoff log records this fact.
