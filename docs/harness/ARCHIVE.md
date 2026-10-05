# Closed directions — historical evidence

These are dated decisions about tested constructions, not permanent research prohibitions or a pending plan.
Current scope and rules are in root AGENTS.md and docs/research/CLAIM.md. Reopening a direction requires a
current decision; historical authorization does not carry over. Most retired sources are in commit b9efc2b;
never-committed work was preserved in refs/backup/pre-cleanup-20261004. Cleanup removals are reflected in
Git history. Local experiment history is in evidence/local/RESULTS.md. All demo paths below are historical.

## Lessons that recur

- A gain over the baseline vanished against a properly trained control with the same information (demo2, demo1).
- A problem assumed from intuition did not exist in the benchmark (demo7).
- Oracle headroom was real but the available evidence could not reach it (demo4 rules, demo8, CPU probes).
- Verified algebra did not become task gain (GIC).
- Signs seen on 60 to 300 samples flipped at full scale (demo2, demo4).
- Probes on 40 episodes were read as failures although their intervals (3 to 9 points) could not see the
  2 to 4 point effects this project has confirmed (demo9).
- Preparation grew in place of a real run: a first run completed zero cases on a missing path, and a prepared
  fit would have trained for one epoch (demo9).

## Entries

**demo1_sam: exact shared-state rewrite of the SAM mask decoder.** `demo_lists/demo1_sam`
- Closed 2026-10-01. Fair benchmark: 35.9 ms dense against 33.6 ms (6.5%); under 2% end to end because the
  encoder takes 87 ms. The earlier 1.47x came from an upsampler layout the baseline can also use. Sparse or
  output-sensitive decoding also failed: per-prompt image state deviates 37 to 46% far from the object.
- Kept: nothing for the paper. The tested decoder rewrite and quality-selector recipes did not establish a paper contribution.

**demo2_where_what_decoding: segmenter decides where, frozen recogniser decides what.** `demo_lists/demo2_where_what_decoding`
- Closed 2026-10-02. ADE20K: Mask2Former Swin-T 47.99 to 54.31, but a pixel-level ensemble with a properly
  trained per-patch head reaches 54.57. SegFormer-B5 -0.49, EoMT-L +0.28 against the same control.
- Kept: the diagnosis that 45 to 53% of error pixels predict a class absent from the image; one model's ADE
  val mIoU has a bootstrap spread near 0.9.
- Direction screening built on it (`demo_lists/DIRECTION_SCREENING_20261002.md`): the loss sits in large
  regions named wrongly among near-synonym classes, and 52 of 150 classes have under 30 validation images.
  Nine post-hoc ideas for closed-set segmentation were screened out. Its area-based gains are lower bounds, not
  upper bounds on what a direction can gain.

**demo3_conditional_grouped: conditional grouped operators in a diffusion model.** `demo_lists/demo3_conditional_grouped`
- Paused at the user's request; outside the paper's field. Noise-prediction MSE improves by 0.3 to 2.7% at
  some time steps and degrades at others; no FID or GenEval result exists.
- Do not resume without the user. The server folder (22 GB) has a `GPU_PAUSED` marker.

**demo4 single-pair selection rules** (shared helpers now in `src/ics/data.py`). `demo_lists/demo4_incontext_seg/README.md`
- INSID3 on COCO-20i: 56.3 reproduced; oracle selection among its clusters 82.1. Fourteen rule families over
  one (reference, query) pair failed (best 55.4 against 56.1 over four folds); supervised scorers reach 58 to
  60. Removing INSID3's coverage factor costs 9.4 points.
- The measured rule families did not solve the selection problem. This does not exclude all single-pair inference; the old pivot to an image pool was later closed.

**demo5_resolution_attention: fixed attention correction across input resolutions.** `demo_lists/demo5_resolution_attention`
- Closed. The correction beats naive high-resolution inference but loses to temperature scaling and to native
  512 input; at low resolution it hurts.

**demo6_instance_evidence: object identity probes on DAVIS.** `demo_lists/demo6_instance_evidence`
- Closed as a probe. Local propagation already has a high identity hit rate; no mask-IoU result and no
  comparison with propagation or memory methods was made.

**demo7_shared_state_ad: one imaging state per image in an augmented memory bank (anomaly detection).** `demo_lists/demo7_shared_state_ad`
- Closed 2026-10-02, the day it was opened. On MVTec AD 2 the baseline without augmentation loses nothing
  under lighting change: pixel AP varies 1 to 2 points across conditions in three categories, and a shared
  state differs from a free per-patch state by at most 0.6. Real exposure change is only x0.87 to x1.13.
- Numbers and method: git history at commit `b9efc2b`,
  `.claude/skills/measure-first-research/references/worked-examples.md`, case B.

**demo8_local_verification and the M-TAP / G-MDN plan: a learned local verifier for in-context segmentation.**
`demo_lists/demo8_local_verification`, `demo_lists/research_decision_20261001`, `docs/research`, `PROJECT.md`, `.agents/teamwork`
- Closed 2026-10-02 at the user's request. The verifier (final and intermediate layers, first seed) scored
  below the strong baseline on the test fold. CPU probes before it: a scalar statistical selector on 4000
  episodes with classes and images held out reached 55.75 against 56.06; graph propagation 56.26; transferring
  completion strength from the reference 39.38 against 54.89.
- The planning documents (architecture, theorems, numerical self-tests) were generated without a positive
  experiment and were deleted.
- Kept: FoRIS reproduced at 60.5 (github.com/Xi-Mu-Yu/FoRIS, revision `1aa02a1`; a checkout is on the server
  used by historical demo9 runs; current availability unverified). The identity J(C) = mu(C) / (|C| + mu(outside C)) and the rule "adding region D raises
  IoU iff its foreground share exceeds J / (1 + J)" are correct and free to use. COCO-20i class folds share
  images across folds (`docs/reference/coco20i-image-overlap-audit-2026-10-02.json`); this matters for any
  trained component.

**GIC: correlated-error pose refinement.** `research/gic_validation` (part of it was never committed)
- Closed 2026-10-02. ScanNet, 1260 pairs from 84 held-out scenes: AUC@5 33.75 native against 30.86, scene-level
  interval [-3.80, -1.98]. MegaDepth 1500 pairs: 69.96 against 67.89. A single update already hurts (-2.12).
- All numerical checks of the algebra had passed. The tested solver did not help; a different explanation would need evidence that real
  match residuals have the assumed tangential-normal coupling.
