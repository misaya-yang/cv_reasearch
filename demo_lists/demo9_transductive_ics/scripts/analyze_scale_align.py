#!/usr/bin/env python3
"""Read the stream of scripts/scale_align_experiment.py against the prediction written before the run.

  python scripts/analyze_scale_align.py --run results/scale_align_v0/run [--out results/scale_align_v0/analysis.json]

Class mIoU at original resolution; paired bootstraps over episodes. Works on a partial stream.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze_extent import compare, load

ARMS = ("reference_centric", "align_mask", "align_core", "align_oracle")
# Written 2026-10-03 before the run, from results/self_support_v0/scale_mismatch.json (241 episodes, FoRIS 59.12):
# -9.5 IoU points per doubling of scale mismatch with query size held fixed (interval -15.5 to -3.5); 82 episodes
# mismatched by 2x or more average 51.5 against 64.9; a crop from the true box gave +18.8 where the query object is
# under 0.35x of the reference object and nothing where it is over 2.8x (that side was never tested).
CARD = dict(
    assumption="FoRIS loses accuracy because it compares two objects of different scale; the loss is caused by the "
               "mismatch, so enlarging the smaller object until both are equal restores it",
    prediction="align_oracle >= +3.0 overall with the interval above 0, >= +8 on episodes whose true mismatch is 2x "
               "or more, within 1 where the true ratio is inside 0.7 to 1.4; a label-free arm recovers at least half "
               "of align_oracle; reference_centric within 1.5 of native",
    match="comparing at matched scale is a component of the method: next estimate the ratio in two rounds, add the "
          "query-side trimming of scripts/self_support_replay.py, run 1000 episodes per fold and INSID3 as second host",
    mismatch="align_oracle < +1.5: mismatched episodes are hard for another reason (small targets, odd references), "
             "drop scale alignment; align_oracle high and label-free low: the first-pass size is the limiter, run "
             "one more round from the aligned result before anything else; reference_centric as good as align_*: "
             "the gain is object-centric cropping, not scale, simplify to that")
GATE = dict(gain=2.0, folds=3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    rho = np.array([0.5 * math.log2(max(r["component"]["truth"], 1) / max(r["component"]["reference"], 1)) for r in recs])
    fresh = np.array([not r["dev40"] for r in recs])
    sub = lambda m: [r for r, k in zip(recs, m) if k]
    table = lambda rows: {k: compare(rows, k) for k in ARMS} if len(rows) >= 2 else {}
    out = dict(state="ANALYSED", episodes=len(recs), fresh=int(fresh.sum()), card=CARD, gate=GATE,
               scope="class mIoU at original resolution; paired bootstrap over episodes, 2000 draws, seed 0",
               all=table(recs), fresh_only=table(sub(fresh)),
               mismatch_2x_or_more=dict(episodes=int((np.abs(rho) >= 1).sum()), **table(sub(np.abs(rho) >= 1))),
               matched_within_1p4x=dict(episodes=int((np.abs(rho) < 0.5).sum()), **table(sub(np.abs(rho) < 0.5))),
               query_object_smaller=dict(episodes=int((rho <= -0.5).sum()), **table(sub(rho <= -0.5))),
               query_object_larger=dict(episodes=int((rho >= 0.5).sum()), **table(sub(rho >= 0.5))),
               by_fold={int(f): table([r for r in recs if r["fold"] == f]) for f in sorted({r["fold"] for r in recs})})
    est = {}
    for k in ("mask", "core"):
        e = np.array([0.5 * math.log2(max(r["component"][k], 1) / max(r["component"]["reference"], 1)) for r in recs])
        est[k] = dict(correlation_with_true_log_ratio=float(np.corrcoef(e, rho)[0, 1]), error_std_in_doublings=float(np.std(e - rho)),
                      same_action_as_true_ratio=float(np.mean([r["plan"]["align_" + k]["action"] == r["plan"]["align_oracle"]["action"] for r in recs])))
    out["ratio_estimates"] = est
    out["actions"] = {k: {x: int(sum(r["plan"][k]["action"] == x for r in recs)) for x in ("none", "reference", "query", "failed")} for k in ARMS}
    orc = out["all"].get("align_oracle", {})
    folds_up = lambda k: int(sum(t.get(k, {}).get("gain", 0) > 0 for t in out["by_fold"].values()))
    out["verdict"] = dict(
        true_scale_gain=orc.get("gain"), premise_holds=bool(orc and orc["gain"] >= 3.0 and orc["ci95"][0] > 0),
        premise_refuted=bool(orc and orc["gain"] < 1.5),
        survivors=[k for k in ("align_mask", "align_core") if out["fresh_only"].get(k) and out["fresh_only"][k]["gain"] >= GATE["gain"]
                   and out["fresh_only"][k]["ci95"][0] > 0 and folds_up(k) >= GATE["folds"]])
    for k in ARMS:
        r = out["all"].get(k)
        if r:
            print("%-18s %6.2f  %+6.2f [%+.2f, %+.2f]  mismatched %+6.2f  matched %+6.2f" % (
                k, r["miou"], r["gain"], *r["ci95"], out["mismatch_2x_or_more"][k]["gain"], out["matched_within_1p4x"][k]["gain"]))
    print(json.dumps(dict(state="ANALYSED", episodes=len(recs), verdict=out["verdict"], ratio_estimates=est)))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
