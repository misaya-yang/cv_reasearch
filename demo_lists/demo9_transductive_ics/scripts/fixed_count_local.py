#!/usr/bin/env python3
"""What does "keep FoRIS's number of patches, choose them again with another evidence" give? Files on this machine only.

  python scripts/fixed_count_local.py --run results/extent_v1/run --maps results/evidence_v1/maps \
      --out results/self_support_v0/fixed_count.json

The 12-pair readout audit (results/extent_v1/complete_pair_evidence_v1) reports the ranking probability (AUC) of a
reference-only readout inside 0.05-wide bins of the FoRIS score: missed target against true background, and found
target against wrongly included background. This script computes the same two numbers on all 241 episodes for the
evidence saved here, and the patch-level class mIoU of the construction proposed from that audit: the mask keeps
FoRIS's own count and its positions are chosen again, either everywhere ("full") or only where the FoRIS score is
within +-w of its cut ("band w"). Rows marked LABELS use query labels or labels of other images: they are bounds.
Two more bounds do not keep the count: every wrongly included patch removed, and every missed patch added.

Card, written 2026-10-03 before the run.
  Assumption: evidence that ranks errors correctly inside FoRIS score bins becomes mIoU once the count is kept and
    the positions are chosen again.
  Prediction: last-layer nearest-neighbour margin: conditional AUC 0.60 (missed) and 0.50 (wrongly included), as on
    the 12 pairs (0.615, 0.487); full re-rank -8 to -12; band 0.1 between -0.5 and +0.5. The supervised class
    classifier (LABELS): full re-rank +2 to +4. Perfect choice at FoRIS's count (LABELS): +14 to +18.
  Match: conditional AUC measures added information, not a usable ranking; a readout has to be combined with the
    FoRIS score, and the count caps what any re-choice can give.
  Mismatch: the label-free margin gains >= +2 in a band: the readout family is usable as it is; run the 12-pair
    readouts (final-layer mean margin first, block 12 second) on all 241 episodes as the main line.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_extent import load, miou
from evidence_audit import auc, normalise

BANDS = (0.1, 0.2, 0.3)
EVIDENCE = dict(
    foris_score=lambda z, m, sn, tf: sn,
    nn_margin_last_layer=lambda z, m, sn, tf: m["fg"] - m["bg"],
    foris_stage2=lambda z, m, sn, tf: normalise(z["s2"].astype(np.float32)).ravel(),
    foris_stage3=lambda z, m, sn, tf: normalise(z["s3"].astype(np.float32)).ravel(),
    query_core_grouping=lambda z, m, sn, tf: m["grp_core"] - m["grp_out"],
    LABELS_class_classifier=lambda z, m, sn, tf: m["probe_class"],
    LABELS_truly_found_part=lambda z, m, sn, tf: m["grp_true"] - m["grp_out"],
    LABELS_perfect=lambda z, m, sn, tf: tf)


def conditional(v, sn, pos, neg, width=0.05):
    """Mean AUC of v over the FoRIS-score bins that hold both groups; None when no bin does."""
    b = np.minimum((sn / width).astype(int), int(round(1 / width)) - 1)
    got = [auc(v, pos & (b == k), neg & (b == k)) for k in np.unique(b[pos | neg])]
    got = [x for x in got if np.isfinite(x)]
    return float(np.mean(got)) if got else None


def choose(v, sn, count, w):
    """`count` patches: those FoRIS scores above 0.5 + w, then the best by v among the ones within +-w."""
    if w is None:
        keep, free = np.zeros_like(sn, bool), np.ones_like(sn, bool)
    else:
        keep, free = sn > 0.5 + w, np.abs(sn - 0.5) <= w
    need = count - int(keep.sum())
    out = keep.copy()
    if need > 0:
        idx = np.flatnonzero(free)
        out[idx[np.argsort(-v[idx], kind="stable")[:need]]] = True
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    cls = np.array([r["c"] for r in recs])
    forms = ["full"] + ["band %.1f" % w for w in BANDS]
    iu = {(k, f): np.zeros((n, 2)) for k in EVIDENCE for f in forms}
    base, ranks = np.zeros((n, 2)), {k: dict(missed=[], included=[], missed_all=[], included_all=[]) for k in EVIDENCE}
    moved = {(k, f): [] for k in EVIDENCE for f in forms}
    free = {k: np.zeros((n, 2)) for k in ("LABELS_remove_only", "LABELS_add_only")}  # the count is not kept here
    tally = dict(wrongly_included=0, missed=0, count_above_1p5_truth=0, count_below_truth_over_1p5=0)
    for j, r in enumerate(recs):
        name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
        z, mm = np.load(a.run / "packets" / (name + ".npz")), np.load(a.maps / (name + ".npz"))
        m = {k: mm[k].astype(np.float32) for k in mm.files}
        tf = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).ravel()
        sn = normalise(z["score"].astype(np.float32)).ravel()
        pred = sn > 0.5
        count = int(pred.sum())
        score = lambda mask: (float(tf[mask].sum()), float(tf.sum() + mask.sum() - tf[mask].sum()))
        base[j] = score(pred)
        true = tf > 0.5
        free["LABELS_remove_only"][j], free["LABELS_add_only"][j] = score(pred & true), score(pred | true)
        tally["wrongly_included"] += int((pred & ~true).sum())
        tally["missed"] += int((true & ~pred).sum())
        tally["count_above_1p5_truth"] += int(count > 1.5 * max(int(true.sum()), 1))
        tally["count_below_truth_over_1p5"] += int(count < max(int(true.sum()), 1) / 1.5)
        fg, bg = tf >= 0.9, tf <= 0.1  # pure patches only, as in the 12-pair audit
        for k, fn in EVIDENCE.items():
            v = np.nan_to_num(fn(z, m, sn, tf).astype(np.float64), nan=-1e9)
            for tag, pos, neg in (("missed", fg & ~pred, bg & ~pred), ("included", fg & pred, bg & pred)):
                c = conditional(v, sn, pos, neg)
                if c is not None:
                    ranks[k][tag].append(c)
                u = auc(v, pos, neg)
                if np.isfinite(u):
                    ranks[k][tag + "_all"].append(u)
            for f, w in zip(forms, (None,) + BANDS):
                mask = choose(v, sn, count, w)
                iu[k, f][j] = score(mask)
                moved[k, f].append(float((mask & ~pred).sum()) / max(count, 1))
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    ref = miou(base[:, 0], base[:, 1], cls)[0]
    out = dict(state="ANALYSED", episodes=n, scope="patch level, no refinement; exploratory", foris_patch_miou=float(ref),
               note="count = FoRIS's own number of patches (score above 0.5); conditional AUC inside 0.05-wide FoRIS-score bins, pure patches",
               rows={})
    for k in EVIDENCE:
        row = {t: dict(mean=float(np.mean(v)) if v else None, episodes=len(v)) for t, v in ranks[k].items()}
        for f in forms:
            t = iu[k, f]
            d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
            row[f] = dict(over_foris=float(miou(t[:, 0], t[:, 1], cls)[0] - ref), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                          share_of_mask_moved=float(np.mean(moved[k, f])))
        out["rows"][k] = row
        print("%-26s cond missed %s (%3d)  included %s (%3d) | all %.3f %.3f | " % (
            k, *("%.3f" % row[t]["mean"] if row[t]["mean"] is not None else "  -  " for t in ("missed",)), row["missed"]["episodes"],
            *("%.3f" % row[t]["mean"] if row[t]["mean"] is not None else "  -  " for t in ("included",)), row["included"]["episodes"],
            row["missed_all"]["mean"], row["included_all"]["mean"])
            + "  ".join("%s %+6.2f [%+.2f,%+.2f]" % (f, row[f]["over_foris"], *row[f]["ci95"]) for f in forms))
    out["count_not_kept"], out["patches"] = {}, tally
    for k, t in free.items():
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
        out["count_not_kept"][k] = dict(over_foris=float(miou(t[:, 0], t[:, 1], cls)[0] - ref), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])
        print("%-26s count not kept  %+6.2f [%+.2f,%+.2f]" % (k, out["count_not_kept"][k]["over_foris"], *out["count_not_kept"][k]["ci95"]))
    print(tally)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
