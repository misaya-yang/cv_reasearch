# Status

Updated 2026-10-05. This file records current state; detailed evidence is linked, not duplicated.

## Now

- **Execution controller: chat `01a10c9c-32fb-7330-be53-cdb43999fd4f`, persistent goal active.**
  Latest user instruction prioritizes useful GPU work with CPU parallelism, independent management,
  and cleanup of poor600attempts after recording results. Reusable features must be retained.
  The latest correction cancels cache reconstruction and returns priority to A*/B* selection.
  Existing600 fixed comparisons are complete.
  Raw DINO matching remains the common DEV accounting origin. [Handoff](../../HANDOFF.md) is a historical snapshot.
- GPU is enabled (32GB,12CPU,about62GiB). Old600 export/recheck and fixed11-arm replay/score are complete.
  Native600/600bit-identical, response/coverage differences0; five shared replay arms match all600pixels.
  MEAN_CONTROL61.467653,+1.394276[.824740,1.907805]vsFoRIS60.073377; fixedAstra60.673832,
  +.600455[-.631630,1.872440]. Target+2 and strong-control superiority remain unestablished.
  [Full fixed results](../../evidence/local/research_20261005/pipeline_verified/fixed600/report.json).
- Historical-training-pool isolated600 export/recheck/sweep are complete. Independent CPU scoring
  adds the omitted MEAN control:FoRIS61.628013,RCG64.237696,MEAN64.264087,Astra63.500125.
  Frozen delete-p64.369663,+2.741650[.661121,3.681100]vsFoRIS;vsMEAN+.105576[-1.414670,1.172892].
  It does not establish strong-control superiority. [Complete600](../../evidence/local/research_20261005/pipeline_verified/fresh600_complete/report.json).
  This600 is historical training-pool reuse,not never-seen confirmation; exporter parity is self-comparison.
  Seven-layer smoke failed from a missing existing CRF path; held without rerun.
- Cleanup exceeded the user's scope:10,067,480,435bytes of reusable old600features were mistakenly
  removed, plus18,836,073bytes of exploratory predictions/fields/counts. Fixed-control masks,
  code,reports,per-episode I/U and source hashes remain. At16:21UTC disk had32GiBfree.
  No reconstruction was launched; the user rejected making restoration the next task.
  [Historical deletion receipt](../../evidence/local/research_20261005/cleanup_existing600_receipt.json).
- A*/B* matched-depth search is complete:greedy2=61.644155,joint2=61.503062; joint-vs-greedy2
  -.141093[-.216928,.020940]. Robust fitting-fold selection61.993517 is only+.000131 over direct
  selection. Exact family counts, overlaps and conflicts are retained;65,536set-accounting checks pass.
  [Result](../../evidence/local/research_20261005/pipeline_verified/joint241_v3/report.json).
- Reference-only A*/B* value estimation completed four fullDEV241 comparisons. Self-calibration37.8934,
  cross-image calibration40.7264,RCG-ranked mass calibration61.0429,two-estimator guarded selection61.0736.
  Guarded output vsFoRIS+1.998822[.567860,3.480652],vsRCG+.053978[-1.017131,1.207666],vsguarded
  direct selection-.099949[-.310364,.205661]. Strong-control/combination advantage remains unestablished.
  [Result](../../evidence/local/research_20261005/pipeline_verified/calibrated241_v4/report.json).
  All predictions sealed before CPU scoring; no query labels in inference and no extra encoder forwards.
  Cached selection time is not complete producer runtime. Main next work is conditional edit value,
  rather than extending the failed global mask-quality estimators.
- Conditional effective edits completedDEV241:61.083596,+2.008770[1.079517,2.986098]vsFoRIS,
  vsRCG+.063926[-.048161,.217392],vssame-countRCG+.009209[-.029547,.107188],vsRCG-value-only
  -.060566[-.155967,.054365]. Strong-control superiority remains unresolved. User requested1200.
  Frozen1200queue22006/start952403608 is active;54missing pair encodes complete104.57s,1146cached
  pair inputs preserved/reused,extra identical encoder released. Method parameters unchanged; GPU
  selector22209/start952414466 then CPU scoring. [Pending execution](../research/PLAN.md).
- Existing public-labelled1200fixed delete-p63.958235vsFoRIS61.612802:+2.345433[1.135312,3.360488].
  VsRCG63.598020:+.360215[-.703107,1.273603],still unresolved; MEAN is being added to the complete
  comparison. Public chain14506/start951952344 now encodes batch2; no peer queue/source changes.
- Another queue's4000public-labelled manifests contain allDEV241,old600 and historical isolated600;
  3999unique episode identities and6722photos. Preserve4000sampled draws; do not silently deduplicate.
  This is benchmark reuse,not4000fresh cases; sampling/worker RNG still needs version verification.
  [Manifest audit](../../evidence/local/research_20261005/public_queue_manifest_audit.json).
- Latest research framing is a reusable A*/B* addition/deletion-family optimization framework, with strong
  complete-method benchmark results as its empirical validation. [Framework record](../../evidence/local/research_20261005/operator_framework.md)
  gives exact marginal accounting and counterexamples to unconditional greedy/sparsity assumptions.
- Latest fixed Astra DEV241 score:62.654343 vsFoRIS59.074825; +3.579518[1.320378,4.824421].
  VsRCG:+1.634674[-.348770,2.523257]; superiority to the strong control remains unresolved.
  Source: [verified report](../../evidence/local/research_20261005/pipeline_verified/recheck241/report.json).
- DEV241 E/B proposals,206-map inference/score,mean graph and89fixed auxiliary rows are complete.
  Best witness-assisted combination50.866779,vsRCG-10.152880[-13.031506,-7.764400]; this construction fails.
  Bounded joint search evaluated5257fixed recipes in23.15sGPU,peak1.36GB; all241 selected masks sealed.
  Joint2=61.503062 vsAstra62.654343,-1.151281[-1.859609,-.194749]. Direct complete selection61.993386
  is stronger. New pairwise intervals are recorded separately with their statistic/RNG identity.
  Old2161/2162/5561 state files are completed/historical and are not active execution handles.
- Astra's supplied complete candidate is retained: exposed DEV220, 1024, +2.751185
  [1.102142, 3.810278] versus native and +0.709731 [-0.704856, 1.334083] versus RCG.
  This earlier candidate is superseded for the current fixed comparison by the query-mean version above.
  It restricts deletion location, sets a reference-BG count and ranks with RCG; it adds no foreground
  beyond RCG. See [intake](../../evidence/local/research_20261005/astra_intake.md).
- **D is not a validated method/base.** Full DEV241, 1024 complete masks: native 59.074825;
  D 59.192751, +0.117926 [-0.828163, +1.581804]. D is -1.826908 [-3.078843, -0.115249]
  versus RCG. Arrived100 loses 1.703 points. The first20 +3.260 was not stable.
- Native replay audit is complete: only two episodes differ (20 and 5 pixels), with class-mIoU drift
  -0.00001536. Both baselines are reported; public replay of those exact cases remains a diagnostic task.
  The drift neither invalidates all predictions nor establishes bitwise identity. See
  [audit](../../evidence/local/research_20261005/native_replay_audit.md).
- RCG full DEV241: 61.019660 versus native 59.074825, +1.944835 [0.983141, 2.913596].
  Delta-only readout is60.198027. The full objective, including strong-control superiority, remains unproven.
- Latest correction: evidence must distinguish wrong-object deletion, boundary leakage removal,
  whole-object recovery and extent completion. Prioritize deployable addition signals; measure separate
  add/delete edits and their combination. GT error categories/oracles are diagnostics, not inference inputs.
- 20/50 cases are execution/gross-failure checks, not a reliable +2-point selector. Meaningful efficacy
  decisions now use the existing DEV241 cohort. All reused data remain DEV, never independent confirmation.
- Existing GPU: reported 32GB, 12 CPU, 62GiB RAM. No artificial duration or round limits. Keep useful work
  prepared ahead, share one DINO encoder, parallelize model-free GPU algebra and CPU evaluation/diagnosis.
  Newly started research agents use GPT-6.1-sol/xhigh. Live work and ownership: [PLAN](../research/PLAN.md).

## Measured / recorded

- The base-class-fitted FoRIS readout has an original-resolution CONFIRM600 positive result; it is supervised.
- Dots' RCG result is on exposed old120 at 1024 working resolution, not original-resolution confirmation.
- Original-resolution DEV241 matte did not establish an advantage over complete FoRIS or delete-only control.
- SAM3 visual/naming results belong to a different resource setting. None establishes the primary DINO claim.
- Numeric comparisons, intervals and sources are in CLAIM and the [ledger](../../evidence/local/RESULTS.md).

## Unverified / held

- The original dots snapshot had new100 unscored and 21 missing. The later Astra package now supplies
  scored220 evidence and a portable replay; its 21 remaining DEV cases have not been run by that package.
- Original-resolution independent confirmation of the cloud candidate is absent.
- The new batch is in [preparation record](../../evidence/local/research_20261005/README.md). Native replay, real encoder hooks and CRF passed the first20 comparison. Full-cohort efficacy remains
  under evaluation; all data are DEV.
- Preparation inventory confirmed endpoint 48002 in no-card mode at 0.5 CPU / 2 GiB, existing DINOv3 weights,
  241 feature/packet pairs and all 241 reference/query image/mask paths. No experiment process was observed.
  That no-card capacity has been superseded by the live GPU inventory above.
- Manifest audit: no duplicate episode identities, 239 photo-connected groups, including two shared-photo
  pairs. Old120 photo identities match; new100 matches exported keys only. All 241 are development data.

## Needed from the user

- No new resource action is currently required. Existing queues run on the enabled machine.
  No rental, download, commit, push or shutdown was performed by this handoff.
