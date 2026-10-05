# Astra portable candidate: intake and next comparison

Source supplied by the user on 2026-10-05: `/Users/yang/Downloads/portable_mask_composition`.
The unchanged [snapshot](astra_portable/README.md) contains all 37 manifest-listed files plus its
delivery manifest. The [intake audit](astra_intake_audit.json) verifies every listed SHA and Python syntax.
No new prediction, remote access, GPU job, model loading or download was performed for this intake.

## Verified existing result

COCO-20i, one labeled reference, frozen DINOv3 cached features, exposed DEV220, 1024 working raster.
Episode identities come from the existing seed-0 DEV241 manifest; all 220 support/query photo identities
match. The 220 episodes contain 76 classes and 218 connected photograph groups. Class mIoU sums I/U
within each class before averaging; intervals use 2,000 RandomState(0) connected-photo resamples.
These are existing predictions, not fresh confirmation or an original-resolution result.

| Candidate minus control | Gain, 95% CI (points) |
|---|---|
| Native complete cached FoRIS | +2.751185 [1.102142, 3.810278] |
| RCG | +0.709731 [-0.704856, 1.334083] |
| MEAN_CONTROL | +1.021665 [-0.531043, 1.503544] |
| Same-domain, same-ranking cross-fitted fixed fraction | +0.482531 [-0.521612, 1.098586] |
| Same K, global RCG ranking | +1.180505 [0.050090, 1.661947] |

Native=58.829934, candidate=61.581120. Candidate/native fold gains are
+1.7697/+2.4103/+4.0771/+2.7663; episode up/down/tie=124/93/3.
Old120 gain=+2.027300 [0.533954, 3.762815]; exposed new100 gain=+1.534530
[-0.555521, 2.977023]. Combined class mIoU is not a weighted average of batch mIoUs.
Full statistics and sensitivity: [local recomputation](astra_statistics_verified.json).

All 2,000 bootstrap draws were additionally checked by literal repeated-episode expansion;
maximum score difference was 6.40e-14 points. System NumPy emitted matrix-multiply warnings, but
the saved results were finite and matched that independent expansion. Native integer I/U matches
our shared 220 cases exactly. RCG I/U differs slightly in 25 cases across the supplied CPU and our
GPU implementations: aggregate local-minus-Astra=-0.00005083 points. This is not mask parity evidence;
the faithful replay must retain Astra's RCG implementation rather than silently substituting ours.

## What the algorithm actually contributes

Let R be RCG, S the source-contrast complete hypothesis and T the scalar-contrast complete hypothesis.
The editable domain is E=R & ~(S|T). It protects RCG pixels supported by either complete hypothesis.
The original packet's reference background-minus-foreground cosine maxima determine
K=count(E & upsample(bg_max-fg_max)>0). The candidate deletes exactly K pixels with the lowest
continuous RCG score inside E. Ties are stable in row-major order.

This separates three questions: where deletion is allowed, how many pixels to delete, and how to rank
those pixels. In the delivered experiment the retained candidate began as a count-matched control;
preserve that exploratory provenance. The background signal sets the count, not a new pixel ranking.
The fixed-fraction control tests adaptive count; the global same-K control tests deletion location.
Neither comparison can be replaced by an agreement-only mask without changing the supplied method.

This is useful concrete evidence for an auxiliary mechanism: a complete hypothesis can protect useful
foreground while existing confidence orders the edits. It is not evidence that every auxiliary should
be a gate, or that high-recall additions should be discarded. The incremental candidate only deletes
from RCG; it does not solve additional missing-object recovery. Removing class58 leaves the candidate
at +2.0978 versus native but only +0.1608 versus RCG and +0.0148 versus fixed fraction. The native gain
is promising; the extra adaptive mechanism remains uncertain and concentrated, not disproven.

The same auxiliary accounting requested for our own mechanisms can be recovered from the supplied
frozen report. Deleting the entire E would remove 1,243,040 TP and 1,967,841 FP; the retained candidate
removes 234,151 TP and 728,794 FP. Thus budget/ranking prevent **81.16% of harmful deletions**, while
also losing **62.96% of beneficial deletions**; retained deletion purity is 75.68%. Candidate counts
were independently recovered from the supplied RCG/candidate integer I/U using the verified deletion-only
definition. These are pooled pixel rates, not class-mIoU thresholds or proof of stable gain.
See [accounting](astra_auxiliary_accounting.json). This directly answers the side-effect question rather
than judging an auxiliary by harmful-edit suppression alone.

## Prepared next comparison (held)

1. Replay the supplied algorithm unchanged on DEV241, including its RCG, MEAN, explicit-BG,
   FG same-K, fixed-fraction and global same-K controls. Preserve packet maxima; do not silently
   recompute them from quantized features. Freeze all predictions before scoring.
2. Check the shared220 as a numerical reproduction stratum and report all241 for efficacy. All21
   additional cases are also DEV. Fixed-fraction controls depend on the cohort: the 241 fraction
   differs legitimately from the 220 fraction; label both rather than demand false parity.
3. Use the prepared E/kernel proposal suite on all241 to measure addition coverage and deletion
   coverage. Report auxiliary harmful-edit removal together with beneficial-edit loss, followed
   by exact complete-mask class I/U. A later combination with Astra must be explicitly specified
   and sealed; no combination has been implemented or tested by this intake.

Runnable unchanged-source command: [launch_astra_dev241.sh](../../../scripts/launch_astra_dev241.sh).
It needs only an existing Python environment and the feature/packet cache, not RGB, model weights,
new DINO forwards, CRF or GPU. Full241 runtime and resource use are unmeasured. Do not retain a paid
GPU merely for this CPU replay; current authorization is local preparation only. The script is not
submitted to any queue. Original-source 5-case parity is supplied evidence, not a new run here.
