# Additional600 queue audit — 2026-10-05 16:09–16:13 UTC

Read-only supervision requested by execution owner chat01a10c9c-32fb-7330-be53-cdb43999fd4f.
No process was changed and no new experiment was launched. Execution owner is waiting for this existing
GPU exporter before its fixed seven-layer DEV241 queue. Queue authorship/its originating instruction
could not be established from current shared records; do not infer authorization from a shell script.

## Current queue and source

Remote root `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9`.
Parent10003 runs `launch/fresh600_v1/run.sh`; exporter10005/start951826240 was live.
It will run cached recheck inference and a wide GPU search after extraction. These processes were preserved.
Source `scripts/export_confirm_cache.py` SHA256fd1c5240eca356dcab3183c060796374cb9899b9f3f122ed0d067dc52aba684e.
Manifest/run-script copies are in `pipeline_verified/fresh600_audit/`.

## Cohort identity: isolated from two cohorts, not never seen

Manifest SHA256fa3965bf38c59c73e597c50dbf712279c240faf624602d5c7dab3a4f71e590bf.
600 unique episodes,150/fold,1164 distinct support/query photographs. Exact episode identity means
(fold,e,c,support basename,query basename). Actual metadata-only comparison found:

| Compared cohort | Episodes | Shared photos | Additional600 episodes with shared photo | Identical episodes |
|---|---:|---:|---:|---:|
| DEV241 |241|0|0|0|
| Previous confirm600 |600|0|0|0|
| Historical training2400 |2400|1164|600|600|

Historical manifests are under `demo9_transductive_ics/results/extent_head_t1_isolated_v1/`:
`confirm_episodes.json` SHA256248d5c873c7f0e91e35f014cc517c1968dcf835728c8b88bdeea1dfc91ed82ff;
`train_episodes.json` SHA25664fd2f9d5e01bcf049f8d5731baa47672959d3748c00100545024d58182471f3.
DEV241 source SHA2561babdc09d206f0e18cf3a8c5498c13adf87eaf202f42de0bf85defbf37b3c793.

The additional manifest itself says role=training, not_claimed_never_seen=true and first150 kept draws
per fold of train_episodes.json. Its name `fresh600` is not evidence of previously unseen confirmation.
It is valid to report it as another historical-training-pool cohort disjoint from the two named evaluation
cohorts. It cannot by itself establish fresh independent confirmation or an exact public1000/fold protocol.

## Export receipt self-comparison

The active exporter makes --stored optional. This invocation has no --stored. Its new branch creates
stored[native],stored[score],stored[cov] directly from the current run, then compares those values with
the same current arrays. Consequently bit_identical and zero drift in this mode are self-comparisons,
not evidence of reproduction against a previous independent output. This does not invalidate the newly
computed completeFoRIS baseline. Keep this distinct from prior600 export with real stored packets.
The branch also opens query annotation to build truth in the cache. No use of truth for candidate decisions
was observed, but query labels were not unopened. The old copied manifest statement is not a runtime audit.
Do not alter the active exporter to repair wording; attach a receipt qualification and fix the next version.

## Fixed versus searched results and ownership

The queued wide sweep must remain exploratory on this cohort. The new prefix file
`outputs/recheck_fresh600_v1_frozen.json` exists and is byte-identical to the previous600 frozen file:
SHA256e34dcff8a04695ea389ea52fbb5e18b7654ceda342b3842dea395339b0537c1e.
It contains the four original DEV241-selected picks, so their unchanged transfer can be reported
separately. An initial supervision message had not yet located this file; that uncertainty was corrected.
The original fixed Astra arm remains fixed independently of that optional file; do not conflate it with
selected wide-search variants. No source change is authorized by this observation alone.

Receiving chat remains the user's designated execution controller. Record this existing queue as origin
unverified until its launching worker/user instruction is known. Preserve current work; coordinate before
changing its schedule or terminating anything. There is no need to launch another encoder to resolve this.
