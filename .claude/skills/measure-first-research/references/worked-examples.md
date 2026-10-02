# Worked examples

Five cases from one project (CVPR 2027 submission work, October 2026). Each shows a step of `SKILL.md` being
followed or skipped, with the measured numbers.

## A. Finding a method from a ledger (in-context segmentation)

Arena: training-free in-context segmentation. Baseline INSID3 (CVPR 2026 oral, frozen DINOv3, public code).

1. **Reproduce.** COCO-20i one-shot, 4 folds x 1000 episodes: 56.3 mIoU locally, 57.6 in the paper.
2. **Ledger.** Oracle choice among INSID3's own clusters: 82.1. Only adding missed foreground clusters: 71.5.
   Only removing false positives: 64.3. The object is a single cluster in 32% of episodes. So about 26 points
   are lost in choosing regions, mostly by missing parts of the object. The features are not the limit.
3. **Rules on the same evidence.** Fourteen families of selection rules over one (reference, query) pair were
   tried: bidirectional nearest-neighbour scores over tree nodes, optimal transport, persistence, cycle
   checks, other layers, whitening and more. The best one gave +2.2 on fold 0 and 55.4 against 56.1 over four
   folds. The single-fold gain was noise.
4. **Is the information there?** A supervised scorer over all available statistics reached 58 to 60 of a
   possible 82; a second one by another session, with classes and images held out, did not beat the baseline
   (55.8 against 56.1). The evidence in one pair is the limit, not the decision rule. Rule writing stopped.
5. **Where is more evidence?** Five labelled references give +6 to +9, so more references help. The only
   free source of references is the test batch itself: other unlabeled images of the same concept.
6. **Fifteen-minute test with all anchors** (fold 0, 150 episodes, 4 unlabeled images per episode):
   baseline 57.1; unlabeled images used naively 59.3; filtered by a round-trip check 60.8; filtered with
   ground truth 61.6; five labelled references 63.4. The idea sits between its lower and upper bound, so it
   is worth an afternoon.
7. **Hardening.** Standard episodes, 4 folds x 400, up to 15 unlabeled images: 55.5 to 61.7, paired difference
   +6.1 with interval [+4.2, +8.2], 47 classes up and 12 down of 80. Naive self-training 58.8. Unlabeled
   images given true masks 67.5. As a black box on a stronger base method (FoRIS): 59.1 to 63.3.
8. **Diagnosis of the remaining gap.** Pseudo masks 64.9; false positives removed with ground truth 68.8;
   misses filled 68.6; true masks 72.3. Two regimes: classes with low recall (person: precision .94, recall
   .54) and classes with low precision (skateboard: .42, .95). One rule cannot serve both, which explains why
   five further trust rules all landed near 66.

What made it work: the ledger said where the points were, the learned scorer said rules were pointless, and
the labelled five-shot number said how much the new source could be worth before any method existed.

## B. Killing a direction in hours (shared imaging state for anomaly detection)

Idea: a memory bank augmented with random gains lets each patch choose its own imaging state; forcing one
state per image should recover accuracy under lighting change. Baseline SuperADD on MVTec AD 2.

- **Premise.** Lighting change must hurt the baseline. One row tests it: no augmentation, scored per lighting
  condition.
- **Prediction written first.** Augmented and plain banks within 1 point; shared and free state within 0.5.
- **Result** (three categories, pixel AP for regular, overexposed, underexposed, extra light): vial 52.0,
  52.4, 53.1, 53.9; fruit_jelly 45.2, 46.4, 45.2, 46.5; sheet_metal 65.0, 65.5, 65.3, 70.0. Shared against
  free state differs by at most 0.6. The real exposure change in the data is only x0.87 to x1.13.
- **Verdict.** The problem does not exist on this benchmark. Stopped the same day; data and weights deleted.

The mistake on the way: the first version queued a 2.5-hour run of five categories with the full backbone to
answer this yes/no question. The replacement used 48 reference images and a smaller backbone, and printed one
line per category within minutes. Three categories were enough.

## C. A claim removed by a strong control (region-level decoding for semantic segmentation)

Idea: a segmenter decides where the regions are, a frozen recogniser on region-pooled features decides what
they are.

- The gain over the baseline was real: Mask2Former Swin-T 47.99 to 54.31 on ADE20K.
- The control with the same information, a pixel-level ensemble with a per-patch head on the same encoder,
  reached 54.57 once trained properly. In the pilot that head had been trained for 3 epochs and scored 41.5
  instead of 49.1, which produced an apparent margin of +0.8 to +3.0.
- The core assumption, "pooling inside a region gives cleaner evidence", had never been tested alone. It
  takes 20 minutes: the recogniser is 87.6% accurate on ground-truth regions and 66 to 71% on predicted ones.
- After the decisive control failed, breadth runs on more models and datasets continued for two hours. The
  cause analysis afterwards needed no GPU; the per-class statistics were already on disk.

Skipped: testing the assumption first (step 4), predicting a number (step 5), a strong control (step 8),
stopping on a failed test (step 7).

## D. Correct algebra, negative task result (correlated-error pose refinement)

Recorded by another session in the same project.

- The derivation and solver passed 22 linear-model checks and 9 geometry checks. An 8-pair interface run
  looked positive.
- On 1260 ScanNet pairs from 84 held-out scenes: AUC@5 33.75 native, 30.86 with the method, scene-level
  interval of the difference [-3.80, -1.98]. MegaDepth 1500 pairs: 69.96 against 67.89.
- A one-update diagnostic gave -2.12, so the first step already hurt. It was not drift over iterations.

Numerical checks verify the implementation of a model. They say nothing about whether real data follow the
model. The assumption to measure first was whether real match residuals have the correlation structure the
method exploits.

## E. A fast loop that was not exact (cached features under mixed precision)

- The cached reimplementation of the baseline scored 55.97 where the released code scored 57.46 on the same
  100 episodes. Only 43% of the masks were identical.
- Cause: the encoder runs in bfloat16. Encoding an image alone or in a (reference, query) batch changes
  features by about 0.01, which is enough to change an agglomerative clustering.
- Fix: cache features from the same batch composition as the released code. Result 57.46 against 57.46, with
  97% identical masks.

Paired differences measured on the old cache stay valid because every row used the same features. Absolute
numbers do not, and were scheduled for a re-run. The check that found this was item-level identity, not the
mean.

## Smaller lessons from the same project

- **Protocol variance.** With one fixed reference per class, two random draws gave 62.0 and 51.3 on the same
  fold. The main protocol was changed to standard paired episodes.
- **Thresholds.** A consensus threshold tuned on one fold was unstable on the others. Rules without a
  threshold (keep the top half) transferred.
- **Speed claims.** A 1.47x decoder speed-up shrank to 6.5% once the baseline was given the same upsampler
  layout, and to under 2% end to end.
- **Planning without measurement.** A multi-agent planning run produced 150 KB of architecture and theorem
  text on top of a verifier idea whose first experiment was negative. None of it changed a decision.
