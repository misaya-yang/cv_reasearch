# Status
Updated 2026-10-03. One live direction. Update this file in the same change as any state change. Keep it under
80 lines; history belongs in `ARCHIVE.md` and in the direction's own documents.
## Live: demo9, transductive in-context segmentation
Current Codex scope: latest user rejects stopping-only explanation and default new boundary/negative-reference patches. Audit identity ranking from the SAME pair/frozenDINO, zero new base-class training or pool: distinguish current-score weakness from representation information, separately FN/FP and matched-score TP/FP,FN/TN; no probe or GTthreshold is an information upper bound. complete_pair_evidence_v1 COMPLETED:12 old40DEV pairs/3perfold,44.73s extraction/56.61s guard,4,446,084,744bytes nativeFP32 finalraw/source pre/postdebias/block12&24 fullstates/actualnativeSDPA QKV; no pooling or NxN attention matrix. Native old-mask12/12 and first2 unobserved/observed exact. Three-agent review confirmed the capture; ranking benefit remains UNTESTED. Fixed reference-only full-vector readouts are being CPU-checked before one finite cached GPU audit; same-value-coordinate V versus native self-attention aggregation is its mechanism control, not cross-image attention or a method gain. Activecache cap4.5GiB/free37GiB/reserve5GiB, no former1200/19GiB rebuild; delete large tensors after decision. Server E44 weste:13322 KEEP ON, shutdown NOT armed; protect foreign jobs. CPU reader being prepared in parallel; Claude own self_support_replay CPU job was observed healthy on E44, protect it; prepared Claude next GPU batch has priority. Earlier extent241 and exact-level/RGB40 completed with no survivor; content-first10 revalidation duplicated oldE1, not new scientific evidence, closed. No downloads/training/commit/push. Five-minute demo9 heartbeat ACTIVE; overall method goal unfinished.
Historical transductive work adds pseudo references to INSID3 (CVPR 2026 oral); results below are evidence, not a restarted queue.
Entry: `demo_lists/demo9_transductive_ics/HANDOFF.md`, `PLAN.md`, `README.md`.
**Measured on the exact protocol** (fold 0 only, 400 episodes, official masks, cache identical to the released
code; `results/probe_decoder_f0_400.json`, `probe_seed_f0_200.json`): INSID3 55.0; naive self-training 57.0;
current method 59.1 (+4.1); pool with true masks 67.7. A closed-form classifier on the pool does not beat the
nearest-neighbour vote (67.4 with true masks, 56.4 with pseudo masks). Removing the seed-similarity factor
costs 2 to 7 points. Round-trip recall on the labelled reference tracks the true recall of the pseudo masks
(rank correlation 0.72); precision does not (0.33). The first 200 episodes alone gave +1.1, so 200 episodes of
one fold decide nothing. Reading: the decoder and the selection rule are not the bottleneck; the quality of
the pseudo references is (8.6 points). Where the unrealised gain sits (`scripts/analyze_gap.py`): in over-segmentation, not misses. Ten classes
hold half of the gap (skateboard 16 / 15 / 75 for one-shot / current / true masks); true masks help because they
label the co-occurring object as background in place, and pseudo masks all share the same mistake. M1 (collection-level modes) was
refuted the same night (oracle ceiling 54.2). Attribution on fold 0: one well-chosen reference (75.1 with a true
mask, 70.7 oracle choice among 16 pseudo hypotheses) beats pooling 15 true masks (67.7); the gap comes from the
38% of pool images whose one-shot masks are wrong (IoU 0.20), and no label-free score separates right from wrong
well enough (AUC 0.81 to 0.84). All label-free variants stop at 57 to 59. Saved-table and full-token selectors subsequently failed to establish value; current plan overrides that historical next step.
The probe cache `/root/demo9_cache/probe` was deleted after use; rebuilding takes 8 minutes.
**Measured earlier** (COCO-20i, masks rebuilt from the official annotations, one pool seed, baseline not exact):
- Standard episodes, 4 folds x 400, up to 15 unlabeled images per episode: INSID3 55.5 to 61.7. Paired
  difference +6.1, 95% interval [+4.2, +8.2], all four folds positive, 47 of 80 classes up and 12 down.
- Anchors in the same run: naive self-training 58.8; unlabeled images given true masks 67.5.
- FoRIS as a black box (60 episodes per fold, pool of 7): 59.1 to 63.3 with the naive variant, positive on all
  four folds, no interval yet. The round-trip filter gave 61.8, lower than naive.
- Remaining gap, fold 0: pseudo masks 64.9; false positives removed 68.8; misses filled 68.6; true masks
  72.3. Two regimes: low-recall classes and low-precision classes.
- COCO-20i masks: every number above used masks rebuilt from the instance annotations, which count crowd
  regions as foreground. The official masks do not. Per (image, class) pair the two have IoU 0.954 on average;
  6.3% of pairs are below 0.9 (`scripts/check_data.py --sets coco --coco-compare 0`). New caches default to
  the official masks (`scripts/_paths.py`, `COCO_ANN`).
- Reliability filtering against the unfiltered pooled vote, 400 development queries, another session's
  open-pool diagnostic: about +0.70 with clean pools and +0.98 with mixed pools, the latter with an interval
  across 0.
**Inferred:** the gain so far comes from using more (pseudo) references. The part that would count as the
method, reliability selection, is worth about 1 point on current evidence.
**Unverified:**
- Absolute numbers. The main table used singly encoded caches; the baseline row matched the released code in
  only 43% of episodes. The pair-encoded cache matches (57.46 = 57.46 on 100 episodes). Re-run is E1 in the plan.
- Reliable selection against random selection (E1).
- Pools that contain images without the concept. A 30-episode smoke test showed the mutual-agreement rule
  falling below the baseline when half of a small pool was distractors; the round-trip rule held (E3).
- Other datasets, full episode count, several pool seeds.
- Novelty. No direct collision found (training-free in-context segmentation with an unlabeled pool), but the
  simple version is not new: TPA (arXiv 2608.08290) builds a prototype bank from confident predictions on an
  unlabeled pool and fuses it with any host for open-vocabulary segmentation; PPNet (ECCV 2020) and
  uncertainty-aware semi-supervised few-shot segmentation (2022) do it with meta-training; TF-SSD (CVPR 2026)
  segments the common object of an image group without a reference. Naive self-training is therefore a
  control, not a contribution. The plan's E1b is the TPA-style control. See the revision at the top of `PLAN.md`.
## Gates (dates proposed, the user confirms)
| Gate | Passes when | Date | If not |
|---|---|---|---|
| G1 | gain at least +4 with the pair-encoded cache, interval above 0; one rule robust to distractors | 10-06 | under +3: close the direction and return to the demo4 ledger |
| G2 | positive on FoRIS over four folds with an interval; label-cost equivalence measured | 10-10 | it is an INSID3 patch; reassess |
| G3 | positive on at least 4 datasets | 10-20 | COCO only: do not submit to the main conference |
| G4 | full tables and a paper draft | 11-05 | |
## Workstreams
| | Work | Plan items | Owner |
|---|---|---|---|
| A | method: reliability rule, distractors, the two error regimes | E1, E1b, E3, E4 | open-pool diagnostic complete; E0/E1 cache rebuild live; TPA controls being checked |
| B | controls and scale: FoRIS, label cost, full runs with seeds | E2, E6, E5 | unassigned |
| C | data and protocol: official masks, seven more benchmarks | E7 | data preparation in progress (below); loaders and episode lists unassigned |
| D | prior work, paper skeleton | | unassigned |
## Server jobs
- Codex: the finite resume2 queue COMPLETED all 20 remaining stages in 896.5s, then requested provider shutdown; browser independently confirmed STOPPED. Provider shutdown and fresh UI STOPPED confirmed. New user goal permits a prepared next cycle; do not boot paid GPU before complete live CPU preflight. Never restart queue_plan/resume1/resume2 or reconstruct deleted metric tensors.
- Seven bounded candidate pilots completed; reduced native E9/E10 fixtures do not complete the original all-method/cross-domain scientific cards. Local ideal-reach audit:adaptive bound .0231216, hard interior margin .5,1800/10485760 pixels potentially movable; ideal best correction <=+.1841pp on ten held tasks, not a full-FP32 certificate.
- Results: official COCO-20i fold0 development subsets only, not method scores. Standard first10: INSID3 52.4769/FoRIS CRF64.9976; E1 localPG +.0977pp against matched raw42.6840, E2 fullGL -13.1911pp; E4/E5/E6/E7 did not beat their useful simple controls. Do not expand these fixed constructs or sweep parameters.
- E3 held-class/all-role-photo-disjoint 24train/8dev/10infer, ten epochs per arm: native77.8527/protected77.8527/unprotected77.8527/fixed77.8751/FoRIS CRF79.1930. Protected/unprotected selected epoch1, fixed epoch10. Development I/U stayed unchanged for adaptive arms; identical developer masks are NOT proved by those counts. Actual inference packed-mask audit:protected/unprotected0 changed pixels, fixed591 changed pixels; no saved delta/margin audit, do not claim threshold deadzone cause. No training expansion; source-hard-mask margin and near-identity fitted weights are possible limitations, not a proven information-loss claim.
- E10 native brightness .75 five-example stress changed40.0384→54.4173; not a method gain. Fixed strong FoRIS CRF same10 control COMPLETED:64.9976→64.6877, -.3099pp exploratoryCI[-.8494,.2720]; stop expansion. GPU new predictions25s/wholeguard31.5s then provider shutdown; UI STOPPED verified. CPU analysis output valid/0errors, frozen plan expected wrong state-name; reconciled on CPU, no GPU repeat. Instance now STOPPED by latest user request. No brightness/threshold sweep, no downloads. Own 52 temporary tensor files released1,800,941,454bytes; small JSON/log/checkpoints protected. Five-minute heartbeat ACTIVE; finite guard powers off after finish/fault/confirmed zero60s if no foreign CUDA.
- Other owner E0/decoder/seed probe results now present in `/root/autodl-tmp/demo9/results`; last check found no active script PID. Do not rerun its matrices or delete its caches; read completed results for the next decision.
- Claude extent batch COMPLETED 2026-10-03 (241 episodes, 4 folds, seed 0; `results/extent_v1/run/analysis.json`): native 59.12; contrast +1.53 [-1.29, +2.99]; boundary -0.52; signature -2.72; round_trip -12.66; zoom_pair +0.41; oracle_cut +10.52 [+8.49, +12.36]; zoom_oracle +5.96; held-out learned pick recovers 1% of the cut gap. No arm survives: the cut/level family, zoom and single-pair statistics are closed. Cause, from the packets (`scripts/diagnose_extent.py`, `diagnose_extent_regions.py`, `probe_extent_evidence.py` -> `results/extent_v1/diagnosis.json`, `region_diagnosis.json`, `evidence_probe.json`): inside the contested area (native mask united with the target) the FoRIS score ranks target above non-target with probability 0.664 and no stored evidence ranks better; false positives resemble the reference foreground more (0.53) than the missed target parts do (0.44); 69% of false-positive pixels are unannotated, 23% another class absent from the reference; a constant-width shift never helps. The error is a ranking error of the evidence, not a cut.
- Claude evidence audit COMPLETED 2026-10-03 (same 241 episodes, 241/241 masks bit-identical to the extent run, 1200-image pool, 890 s, no shutdown; `scripts/evidence_cache.py`, `evidence_audit.py`, `analyze_evidence_followup.py` -> `results/evidence_v1/audit.json`, `followup.json`). Contested-area ranking probability and held-out read-out gain over the score-only read-out (patch level, exploratory): FoRIS score 0.664; no label-free form passes the gate (0.75 and +3). Label-free: query grouping from FoRIS's confident core 0.724, +1.07 [-0.16, +2.39] (+3.0 on the 146 episodes whose core is at least 90% target, -0.3 on the 37 whose core is mostly wrong); pool competitors 0.60 to 0.62; pool positives claimed by the reference 0.644, +0.05; base-class linear metrics 0.647, base-class classifier 0.418. With labels: grouping from the truly found part 0.807, +9.61 [+6.40, +11.12]; supervised class probe 0.771, +7.20; labelled pool positives 0.734, +7.04; competitors 0.645, +1.88; all-class linear metric 0.654, +0.62. Closed: competitors, linear metrics, pool positives without labels. Every label-free seed is mostly wrong in 15% of episodes. Every evidence, the supervised probe included, ranks missed target parts below wrongly included regions (0.19 to 0.41). Contact sheets: `results/extent_v1/sheets/`. Cache kept on the server at `/root/autodl-tmp/demo9_extent/cache/evidence_v1` (6.3 GB, rebuild 12 min) for replays; delete when the user closes this line. Next step is the user's decision.
- Claude follow-ups on the cache, 2026-10-03 (patch level, exploratory; `scripts/self_support_replay.py`, `analyze_size_prior.py`, `extent_decoder_probe.py` -> `results/self_support_v0/`): re-deciding the uncertain band of the FoRIS mask by similarity to its own confident core, variant chosen on a development half, gives +1.68 [+0.72, +2.27] on the confirmation half; score diffusion on the query graph +0.9. Knowing the target's area in the query recovers +8.67 [+5.94, +9.94] of the +10.19 cut gap and needs a log-area error near 0.35 or less; FoRIS's own area is within 21% in the median episode but more than 1.5x too large in 32%; match geometry (error 1.46) and a held-out regression (0.62) both lose. A class-agnostic convolutional read-out fitted on about 180 episodes per fold overfits (-3.4; score-only control +0.15). Open premise, not yet tested at scale: the size of the target is learnable from base classes.
- 2026-10-02: data prepared with `scripts/get_data.sh` into `/root/autodl-tmp/datasets/ics/` (INSID3's layout); check with `scripts/check_data.py`. Ready: COCO-20i official masks, Pascal-Part, SUIM, chest X-ray, PACO-Part and LVIS-92i annotations.
  Postponed at the user's request until G2: ISIC (11 GB), iSAID (2.9 GB). Not planned: PerMIS (needs a 119 GB archive).
## Resource boundary
Latest user has restarted E44 and explicitly requires KEEPING IT ON while Claude prepares subsequent experiments. Old guard shutdown policy is superseded; own new guard is NOT armed for shutdown. Execute earlier unrun finite comparison, then prepared Claude continuation when available. Protect foreign jobs; no downloads, commits or pushes.
