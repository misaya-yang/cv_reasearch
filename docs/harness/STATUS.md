# Status

Updated 2026-10-06. This file records current state; detailed evidence is linked, not duplicated.

## Now

- Resource scheduling failure: the owned CLS GPU stage ended at2026-10-05 23:39:47UTC;
  the next owned GPU stage started at2026-10-06 00:09:15UTC, leaving about29m28s without an
  owned GPU successor. Controller checked missing MEAN continuous inputs too late and spent
  the gap on analysis/code preparation. Current family-composition supervisor56044/start954745049
  runs graft4000 child56169/start954755102; fine-MEAN1200 supervisor56334/start954783172 is
  already validated and waiting for GPU release. CPU reconstruction and mask-only comparison
  proceed separately. Check successor inputs and runnable stages before the active GPU job ends.

- User authorized parallel agents. Controller owns GPU monitoring/dispatch; three analyses
  completed RCG stage,4000quality/batch andcomplete-native edit attribution. OpusCPUanatomy
  andownstate/verification queues have completed. ExistingGPUboundaryjob37422/start953346485
  completed600cases. Fixed80fine/coarse/RGB cue diagnosticcompleted,
  `launch/boundary_features80_v1`,supervisor41485/start953486380,child43130/start953528542. Retain all80FP16finegrids,~2.5GiB; disk had21GiBfree.
  Original rawgraph queue held before inference,CPUvariant neverlaunched; no assets deleted.
- Frozen1200six-armcomparisoncompleted:FoRIS61.612802,RCG63.598020,MEAN63.581592,
  RCG64control63.690334,fine16control64.015315,primaryfine64=64.051866.
  PrimaryvsFoRIS+2.439064[1.581170,3.237973],vsRCG64+.361532[.312316,.461490];
  vsRCG+.453846[-.059653,.905474],MEAN+.470274[-.104917,1.001042],fine16+.036551
  [-.520387,.457837]. Completeobjective remainsunproved. All1200baselineI/Uparityandindependent
  score/CI/foldreconstructionpassed;netTPdamage persists. [Result](../../evidence/local/research_20261005/pipeline_verified/frozen_subtoken1200_v1/report.md).
- Frozen full4000 six-arm readout is complete: native60.931741, RCG62.333671,
  MEAN62.512972, coarse64=62.297937, fine16=62.709100, primary fine64=62.651617.
  Primary-native +1.719876[1.256073,2.113693], primary-MEAN +.138645[-.184598,.422923],
  primary-coarse64 +.353680[.321971,.396545]. The >=2 target and strong-control superiority
  are not established. All4000 baseline and1200 six-arm I/U checks and independent statistics
  passed. [Result](../../evidence/local/research_20261005/pipeline_verified/frozen_subtoken4000_scored_v1/report.json).
- Fixed mask-level MEAN/fine16 A/B transfer completed4000 at62.744357: native
  +1.812616[1.524273,2.098950], MEAN +.231385[.202166,.266264], fine16
  +.035257[-.145128,.218718]. Formula `(M | (V & ~R)) & ~(R & ~V)`; complete masks
  sealed before scoring, no encoder or parameter search. This is a strong composition control.
  [Result](../../evidence/local/research_20261005/pipeline_verified/mean_fine_bit_transfer4000_v1/report.md).
- Latest user correction directs exact selection among existing complete A*/B* families.
  `/root/rcg_ablation_audit` builds aligned libraries; `/root/rcg_quality_analysis` solves all
  allowed add/delete subsets and reports actual complete recipes, scores and cost Pareto rows.
  No new background-SNR or CLS variants are running. GT selection on reused data is development,
  distinct from frozen inference and confirmation. `/root/region_cue_feasibility` integrates the
  supplied Opus fine-readout + size-cut method without silently changing its settings.
- The supplied fresh600 cross-fold combination65.44 uses other-fold labels for size thresholds.
  The separate `rcg2_frozen.json` uses all fresh600 to fit six thresholds and reports in-sample65.86;
  its fixed readout uses tau=.15 in every fold. Existing own4000 fields use tau=.07 on folds0/3,
  so exact full4000 reuse requires resolving that source difference. These versions remain separate.
- Fixed scalar residual-transfer4000 completed at62.844537: native
  +1.912796[1.618404,2.210338], MEAN +.331565[.303193,.372805], fine16
  +.135437[-.048326,.334529]. All original six-arm I/U and independent score/CI/fold/batch reconstruction passed
  exactly; relative to fixed bit transfer +.100180[.079995,.128147]. Formula `upsample(MEAN)+fine16-upsamp(RCG)`, coefficient1/no clipping.
  All4000 masks were sealed before score. Target>=2 and fine16 superiority remain unproved.
- Read-only complete-mask library construction completed107.861s CPU6:4000 has13 source
  arms/11 distinct ordered mask sequences; DEV241 has415 source arms/185 distinct sequences.
  Exact4000 A*/B* selection runs before the much larger DEV library. Original rawNN4000 and
  count-only delete-p4000 masks are absent and are not fabricated. Label-fitted supplied methods
  are included in a separately marked extended setting.
- Fixed direct-MEAN fine1200 is validated and waiting for GPU availability. Its retained kernels
  permit later scalar-family reuse. The supplied uniform-tau15 size-cut method requires2000
  missing fields; a fused producer successor is being prepared to avoid repeated four-shift
  encoding. A separately named control applies the fixed fresh600 thresholds to all existing
  fold-temperature fine16 fields, preserving their source tau; it is not the supplied primary.
  Existing peer Opus stream evaluates fixed fresh600-trained settings on groupsB/C,600 draws
  each; its queue is read-only and must not be duplicated or changed.
- Original DEV241 raw-origin accounting completed without new inference. Frozen fine64=60.515912,
  native=59.074825, raw NN=42.903888 and complete INSID3 CRF=55.006020. Fine64-native
  +1.441087[-.754819,3.052193]; fine64-coarse64 +.387231[.095634,.753852]. RCG/MEAN/fine16
  superiority remains unresolved. All241 identities, pixel truth, six-arm I/U and raw/native edit
  closure passed independent checks. [Accounting](../../evidence/local/research_20261005/pipeline_verified/frozen_fine_raw_dev241_v1/interpretation.md).
- Fixed region-mean cue diagnostic completed CPU-only on DEV241. Only18 episodes support the
  paired missed-versus-stray comparison: raw AUROC .3944[.1680,.6072], cached prototype .2556
  [.0733,.4556], stored NN .3272[.1176,.5556]. No positive semantic discrimination is established;
  stop this fixed mean-prototype recovery branch. All197 eligible region descriptors are retained.
  Fixed object-crop final-CLS completed367views/85episodes/197regions,114 encoder calls,
  153.29s encode/peak2,080,604,672bytes. Primary18episodes/71regions AUROC .5920[.3889,.7878],
  versus raw region mean +.1975[-.0714,.4704]: unresolved. Extra crops, reference erasure and
  privileged GT geometry make this a construction test, not a pure CLS ablation or method.
  [CLS result](../../evidence/local/research_20261005/pipeline_verified/object_cls_dev241_v1/report.md).
  Concurrent GPU mean rose71.8->95.8percent while main slowed2.448->4.002s/newpair; no free
  throughput gain is claimed. Short fixed test ended; main recovered~2.39s/newpair.
  [Resource audit](../../evidence/local/research_20261005/pipeline_verified/stream4000_resource_audit_v1/report.md).
  The unchanged1200 CLS diagnostic is complete:77 paired episodes/296 regions,
  AUROC .433959[.340099,.527200], versus NN +.190674[.090203,.300785]. This does not
  establish useful target discrimination; stop this construction. All1400 descriptors and
  source/statistical checks passed. Natural-RGB reference-FG intervention completed: AUROC
  .494905[.400891,.585473], delta+.060946[-.032513,.150309], unresolved. No further CLS
  variants or expansion are queued. [CLS1200](../../evidence/local/research_20261005/pipeline_verified/object_cls_dev1200_v1/report.md).
  Part1-only producer operational parity/timing completed on two pairs:~62-64percent saved
  for the feature-only producer. Complete cold RCG still needs its host score; this is not an
  end-to-end method speedup. [Audit](../../evidence/local/research_20261005/pipeline_verified/part1_producer_two_pair_audit_v1/interpretation.md).
- Fixed600matchedcoarsecontrolcompleted:coarseguide64.273121,fine64.625380;
  fineadvantage+.352259[.325705,.544086],allsource600I/Uexact. GPUbatch4+CPU6readers/2writers,
  19.92sinference/4.16sGPUcompute/10.87sCPUscore. Sourcelegacysealinput-keyfailureinfirstattempt
  fixedinnewv2snapshot;failedoutputretained,CUDAfinalrenderer matchedsource. No featuresdeleted.
  [Control](../../evidence/local/research_20261005/pipeline_verified/subtoken_coarse_control600_v2/report.md).
  Fixed1200fine-readout/strongRCG64comparison completed with exactλ16field/mask replay;
  the expanded frozen4000 stream is now active as recorded above.
- Fine80boundarydiagnosticcompleted111sGPU+12sCPU;retain2.5GiBfinefeatures. Finevsbilinearcoarse
  NNmargin AUROC+.0010[-.0041,.0056],parentrank-.0591[-.0859,-.0335]. Togetherwith600matched
  control,thissupportsvalueinthetestedquery-affinityreadout,notimprovedreferenceNNsemanticmargin.
  [Boundary report](../../evidence/local/research_20261005/pipeline_verified/boundary_features80_v1/report.md).
- CPU4existing-reference-cue4000diagnosticcompleted170s. SignedNNmarginAUROC.253[.216,.293]
  onwhole-missed-vs-strayregions(223eligibleepisodes),.233[.216,.250]ondeepFN/FP(1088).
  No positiveevidenceforrecoveringtheseerrorsusingthatcue;don'tinvertposthocGTconditionalcueor
  repeatNN-margin/prototype variants. Conditionaldiagnosticdoesnotruleoutallrepresentation.
  [Cue result](../../evidence/local/research_20261005/pipeline_verified/rcg_remaining_cues4000_v1/interpretation.md).
- Historical600fixed ablation is now independently verified:600RCGfields andmasks exact,
  allCGstatuses/residuals pass; allpredictions sealedbeforeGT. Pre60.944388,rerank60.939394,
  smooth63.582277,both64.237696. Smoothvs pre+2.637889[1.792333,3.268409];
  rerankwithgraph+.655420[.056436,1.061844];factorialinteraction+.660413[.196835,1.183400].
  Graph supplies most of thetestedgain;reranking isconditional. Not a4000ablation/confirmation.
  [Verified result](../../evidence/local/research_20261005/pipeline_verified/rcg_ablation_verified600_v1/report.json).
- Complete4000RCGvsnative+1.401930[1.094,1.686]. Last1000gain+.618vsfirst3000+1.661;
  difference-1.043[-1.597,-.342],notexplainedbyquality/classmixalone. Exactclass-balanced
  loss:FPremoval-.721andTPdeletion-.478,offsetpartlybylessnewFP+.256;GTdistance>16
  accounts-.812[-1.326,-.163]. NativeCRFgainchange+.053[-.088,.226].
  [Quality](../../evidence/local/research_20261005/pipeline_verified/rcg_quality4000_v1/report.md),
  [exact attribution](../../evidence/local/research_20261005/pipeline_verified/rcg_quality4000_v1/state_report.md).
  RemainingRCGerrors:body54.98percent,GTboundary<=16pixels25.70percent,untouched/straysemantic
  components19.32percent. Whole-missedGTcomponentFNmassincreases4.044Mvsnative;boundarydominance
  isnot supported. GTcomponent/area binsaresemanticproxies,notinstances;alltheseareGTdiagnostics.
  Raw-DINO matching remains the common DEVorigin;4000raw features areunavailable.
- Full4000 fixed-component readout completed: FoRIS60.931741,RCG62.333671,MEAN62.512972,
  Astra61.853251,delete-p62.328852. Delete-p vsFoRIS+1.397111[.783637,1.970674],
  vsRCG-.004819[-.568074,.523499],vsMEAN-.184120[-.751011,.351018]. Target remains unmet.
  [Result](../../evidence/local/research_20261005/pipeline_verified/frozen_public4000_v1/report.json).
  All sampled4000draws retained; benchmark reuse,not fresh confirmation. Serial seed0 episode
  identities match official sampling logic on existing metadata; canonical metadata/encoder parity
  remain unverified. No new full4000complete candidate result is asserted.
- Spatial BG jackknife completedDEV241:63.248866,vsoriginalB+sameextent-.034775
  [-.237678,.183151],vsoriginal-margin same-count-.001509[-.020362,.003954].
  It does not establish a useful extra mechanism. Stop this prior-dependent stability branch.
  [Result](../../evidence/local/research_20261005/pipeline_verified/spatial_jackknife_dev241_v1/report.json).

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
  Frozen1200queue22006 completed:63.679854vsFoRIS61.612802,+2.067053[1.480652,2.622434];vsRCG
  +.081835[.011417,.162600],butvsRCG-value-only-.081893[-.150924,-.016964]. Retire the extra
  two-estimator gate; the full objective remains unmet.54gap pairs replayed exactly,1146cached
  inputs preserved/reused; no parameter updates. [1200report](../../evidence/local/research_20261005/pipeline_verified/conditional1200_v1/report.json).
  Direction-specific rule is a newDEVconstruction:61.168023vsFoRIS+2.093198[1.161610,3.060837],
  vsRCG+.148354[.017192,.310205],vsRCG-value-only+.023861[-.029879,.102806]. Its1200development
  re-evaluation completed:63.760280,+2.147479[1.561716,2.702933]vsFoRIS and+.162261[.083281,.256631]
  vsRCG,but-.001468[-.038359,.037012]vsRCG-value-only. Joint andgreedy remain equivalent.
  [Directional1200](../../evidence/local/research_20261005/pipeline_verified/directional1200_dev_v1/report.json).
  This1200was already read and is not confirmation. All arm scores were independently reconstructed;
  actual directional source hashes are verified separately from the copied wrapper's incomplete source list.
  Strong-control DEV241comparison completed:pixel optimum61.108318,p1greedy61.147027,single-family61.166854.
  Directionaljointbeats pixel optimum+.059706[.009841,.154251],butvsbest single-family+.001170
  [-.067942,.084219]. Original six output arms are bit-identical. Joint advantage remains unresolved.
  Same fixed control expansion on1200completed:pixel63.718897,p1greedy63.761862,single-family63.797241.
  Jointvspixel+.041384[.003448,.085630],butvssingle-family-.036960[-.080854,.006758].
  Method-family constraints help this imperfect value field; joint choice superiority remains unestablished.
  [1200strong controls](../../evidence/local/research_20261005/pipeline_verified/directional_controls1200_dev_v1/report.json).
  No extra encoder forwards. The unrestricted expected-count solver passed80exhaustive cases.
- Existing public-labelled1200fixed delete-p63.958235vsFoRIS61.612802:+2.345433[1.135312,3.360488].
  VsRCG63.598020:+.360215[-.703107,1.273603],still unresolved; MEAN is being added to the complete
  comparison. CompleteMEAN1200=63.581592;delete-pvsMEAN+.376643[-.648147,1.314328],still unresolved.
  Public chain14506/start951952344 continues its4000stream; no peer queue/source changes.
- Frozen delete-p expanded unchanged to1800/450perfold:62.628810vsFoRIS60.484167,
  +2.144643[1.221813,2.958440];vsRCG+.341148[-.465580,1.085701],vsMEAN+.210531[-.596419,.993175].
  Strong-control superiority remains unresolved. All prior1200counts and complete RCG/MEAN/Astra masks
  independently match existing annotations; no encoder or feature recreation.
  [1800report](../../evidence/local/research_20261005/pipeline_verified/frozen_public1800_v1/report.json).
- Complete INSID3 logic comparison onDEV241 finished:bilinear54.408042,640CRF55.006020;
  versusFoRIS59.074825,CRFgain-4.068805[-6.395323,-.234831]. Released0c165a10logic,
  common timm DINOv3-L weights,paired BF16 andnative FP32basis. Hub numerical parity is not asserted.
  [Report](../../evidence/local/research_20261005/pipeline_verified/insid3_complete241_v2/report.json).
- CompleteA/Bcomposition1200=64.119318,+2.506516[1.348066,3.435200]vsFoRIS and+.161083
  [.017346,.276713]vsretainedB. VsRCG+.521298[-.465123,1.318251],vsMEAN+.537726[-.422437,1.389626];
  vsunionquota+.032162[-.017208,.127733],vspixel+.002634[-.001900,.007321]. Strong-control superiority
  andstable>=2points remain unestablished. All1200frozen-deletion/same-count I/U match exactly;full edits retained.
  [Completecomposition](../../evidence/local/research_20261005/pipeline_verified/composed1200_dev_v1/report.json).
  Unchangedmethod2400/600perfold completed onblocks0/1/4/5:63.426455vsFoRIS61.635100,
  +1.791355[1.098103,2.489658];vsRCG+.286291[-.347554,.878681],vsMEAN+.149259[-.493105,.752387].
  New1200alone62.032075vsFoRIS61.016582,+1.015494;vsRCG+.020356,vsMEAN-.270258.
  Targetandstrong-controlsuperiority remain unmet. Parentseals andall2400originaldeletion/countI/Uverified;
  no featuredeletion/recreation. Original1200cache protected. [2400report](../../evidence/local/research_20261005/pipeline_verified/composed2400_v1/report.json).
- NewcompositionDEV241raw-originreadout completed:62.936227vsFoRIS+3.861402[1.609918,5.162354],
  vsRCG+1.916558[.037169,2.840281],butvsB-.186110[-.480213,.156544]. Native/RCGreplaymaskdifference0.
  [Raw-originreport](../../evidence/local/research_20261005/pipeline_verified/composed_dev241_v1/report.json).
  GTdiagnostics:restoration-.348848,extent+.161303;wholeGTregionrecovery beyondRCG is not established.
  Reference-NN BGfilterreducesbanktargetmass7.45percentto5.38percent,butcompleteDEV241method62.807903
  loses tooriginalB63.122337 andsame-sizeBGcomposition63.017747. BGpurityalone is insufficient.
  Role-prototype construction completed:62.306604vsoriginalB63.122337,-.815733[-1.258972,-.289823].
  Bonly62.400503,query-onlysplitB62.667190,unsplitreferenceguardB62.883290;oldB+newA63.035282.
  Maxprototype splitting/referenceguard andwhole-query high-margin addition didnot improve this construction.
  [Roleprototype result](../../evidence/local/research_20261005/pipeline_verified/role_prototypes_dev241_v1/report.json).
  No encoder orfeaturedeletion. Do not extend this margin/role grid without a changed failed link.
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
