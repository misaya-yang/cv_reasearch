#!/usr/bin/env python3
"""Read the stream of scripts/extent_experiment.py and check each arm against the prediction written before the run.

  python scripts/analyze_extent.py --run results/extent_v1/run [--out results/extent_v1/analysis.json]

Works on a partial stream. Class mIoU is the few-shot convention (per class, summed intersection over summed union);
intervals are paired bootstraps over episodes.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SMALL = 0.03  # target area below 3% of the frame
# The cards. Numbers were fixed on 2026-10-03 from the 40-task ledger (results/native_membership_v1/causal_v3/
# extent_cut_ledger.json): native 65.1, score-only contrast cut +3.9 at patch level, oracle cut +10.2, small targets
# 53.8 against 72.7 for the rest.
CARDS = dict(
    boundary=dict(
        assumption="the stop is readable from the query's own adjacent-patch affinity along the reference-ranked chain",
        prediction="gain >= +3.0 with the interval above 0, at least +1.0 over contrast, inverted rule below native",
        match="query structure carries the stop: run all folds, a second host (INSID3), part benchmarks",
        mismatch="not above contrast: last-layer adjacency adds nothing to the score's own shape; inverted not worse: "
                 "the statistic is uninformative, drop it"),
    round_trip=dict(
        assumption="the reference's exclusion, applied by sending a candidate region back, tells where to stop",
        prediction="+1 to +2 overall; >= +3 where native under-extends, <= +1 where it over-extends "
                   "(round trip tracked recall 0.72 and precision 0.33 in the earlier probe)",
        match="reference verification repairs misses only: pair it with a rule for the over-extension side",
        mismatch="<= 0 everywhere: backward nearest neighbours are too noisy at this granularity, drop it; "
                 ">= +3 on over-extension: the reference's exclusion is stronger than measured, make it primary"),
    signature=dict(
        assumption="the reference mask says which kind of adjacency is the target's edge",
        prediction="gain >= +3.0 and not below boundary where native over-extends",
        match="the boundary type is transferable: this is the arm to carry to part benchmarks",
        mismatch="below contrast: pair correspondences across images are no more reliable than single ones"),
    zoom_pair=dict(
        assumption="extent errors on small targets come from observing them at the wrong scale",
        prediction="zoom_oracle >= +5.0; zoom_pair >= +2.5 overall, >= +6 on small targets, within 0.5 on the rest, "
                   "and not below zoom_query",
        match="re-observation at object scale is a component; next, choose the cut inside the crop",
        mismatch="zoom_oracle < +2: scale is not the limiter, drop zoom; oracle high but zoom_pair low: the crop box "
                 "from the first pass is the limiter, take it from the core of the chain"),
    learned=dict(
        assumption="the per-level statistics saved here contain the stopping information",
        prediction="a held-out-fold learned pick recovers >= 50% of the oracle-cut gap before refinement",
        match="the information is there: a rule can reach it, keep working on the cut family",
        mismatch="< 25%: these statistics do not carry the stop; stop writing cut rules, add observations"),
)


def load(run):
    return [json.loads(l) for l in open(Path(run) / "episodes.jsonl") if l.strip()]


def miou(i, u, cls, w=None):
    """Class mIoU for [n] intersections/unions and class ids; w [B, n] gives bootstrap replicates."""
    ids = np.unique(cls)
    m = (cls[:, None] == ids[None]).astype(float)
    if w is None:
        w = np.ones((1, len(cls)))
    with np.errstate(all="ignore"):
        si, su = (w * i) @ m, (w * u) @ m
        r = np.where(su > 0, si / su, np.nan)
    return 100 * np.nanmean(r, 1)


def compare(recs, arm, base="native", key="original_iu", draws=2000, seed=0):
    get = lambda a, j: np.array([r[key][a][j] for r in recs], float)
    cls = np.array([r["c"] for r in recs])
    ia, ua, ib, ub = get(arm, 0), get(arm, 1), get(base, 0), get(base, 1)
    w = np.random.default_rng(seed).multinomial(len(recs), np.ones(len(recs)) / len(recs), size=draws).astype(float)
    d = miou(ia, ua, cls, w) - miou(ib, ub, cls, w)
    e = 100 * (ia / np.maximum(ua, 1) - ib / np.maximum(ub, 1))
    return dict(miou=float(miou(ia, ua, cls)[0]), gain=float(miou(ia, ua, cls)[0] - miou(ib, ub, cls)[0]),
                ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])], up=int((e > 1).sum()), down=int((e < -1).sum()))


def table(recs, arms):
    return {a: compare(recs, a) for a in arms} if len(recs) >= 2 else {}


def ratio(r):
    """Native predicted area over true area at model size."""
    i, u = r["iu"]["native"]
    t = r["area"] * (r["grid"][0] * 16) * (r["grid"][1] * 16)
    return (u - t + i) / max(t, 1)


def cut_family(recs, seed=0):
    """Every selection rule before refinement, from the stored per-level intersections and unions."""
    rng = np.random.default_rng(seed)
    cls = np.array([r["c"] for r in recs])
    mid = min(range(len(recs[0]["levels"]["I"])), key=lambda k: abs(0.2 + 0.0125 * k - 0.5))  # FoRIS's own level

    def at(pick):
        i = np.array([r["levels"]["I"][k] for r, k in zip(recs, pick)], float)
        u = np.array([r["levels"]["U"][k] for r, k in zip(recs, pick)], float)
        return float(miou(i, u, cls)[0])

    def picked(name):  # the level the runner chose (ties resolved there); None means it kept FoRIS's own level
        return [mid if r["pick"].get(name) is None else int(round((r["pick"][name] - 0.2) / 0.0125)) for r in recs]
    valid = [np.nonzero(r["levels"]["valid"])[0] for r in recs]
    oracle = [int(np.argmax(np.array(r["levels"]["I"]) / np.maximum(r["levels"]["U"], 1))) for r in recs]
    rows = dict(native=at([mid] * len(recs)), contrast=at(picked("contrast")), boundary=at(picked("boundary")),
                boundary_inverted=at(picked("boundary_inverted")), round_trip=at(picked("round_trip")),
                signature=at(picked("signature")),
                random_level=at([int(rng.choice(v)) if len(v) else mid for v in valid]), oracle=at(oracle))
    rows["learned"] = learned(recs, mid, at)
    return rows


def learned(recs, mid, at):
    """Measuring device: gradient boosting on the label-free per-level statistics, tested on a held-out fold."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    folds = sorted({r["fold"] for r in recs})
    if len(folds) < 2 or len(recs) < 40:
        return None
    names = ("contrast", "boundary", "round_trip", "signature")

    def feats(r):
        L = r["levels"]
        n = len(L["I"])
        cols = [0.2 + 0.0125 * np.arange(n), np.array(L["area"], float) / (r["grid"][0] * r["grid"][1])]
        for k in names:
            v = np.array([np.nan if x is None else x for x in L[k]], float)
            ok = np.isfinite(v)
            rank = np.full(n, np.nan)
            if ok.sum() > 1:
                rank[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
            cols += [v, rank, v - v[mid] if np.isfinite(v[mid]) else np.full(n, np.nan)]
        return np.stack(cols, 1), np.array(L["I"], float) / np.maximum(L["U"], 1)
    data = [feats(r) for r in recs]
    pick = [mid] * len(recs)
    for f in folds:
        held = {x for r in recs if r["fold"] == f for x in (r["support"], r["query"])}
        tr = [j for j, r in enumerate(recs) if r["fold"] != f and r["support"] not in held and r["query"] not in held]
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0)
        model.fit(np.concatenate([data[j][0] for j in tr]), np.concatenate([data[j][1] for j in tr]))
        for j, r in enumerate(recs):
            if r["fold"] == f:
                p = model.predict(data[j][0])
                ok = np.array(r["levels"]["valid"])
                pick[j] = int(np.where(ok, p, -np.inf).argmax()) if ok.any() else mid
    return at(pick)


def verdicts(out):
    """Compare with the cards. Only statements the numbers support; None when the row is not available."""
    a, s, cut = out["all"], out["by_native_error"], out["cut_before_refinement"]
    v = {}
    if "boundary" in a:
        b, c = a["boundary"], a["contrast"]
        v["boundary"] = dict(gain=b["gain"], ci95=b["ci95"], over_contrast=b["gain"] - c["gain"],
                             inverted_below_native=cut["boundary_inverted"] < cut["native"],
                             matches=bool(b["gain"] >= 3 and b["ci95"][0] > 0 and b["gain"] - c["gain"] >= 1
                                          and cut["boundary_inverted"] < cut["native"]))
        under, over = s.get("under_extended", {}).get("round_trip"), s.get("over_extended", {}).get("round_trip")
        v["round_trip"] = dict(gain=a["round_trip"]["gain"], under=under and under["gain"], over=over and over["gain"],
                               matches=bool(1 <= a["round_trip"]["gain"] and under and under["gain"] >= 3))
        ov = s.get("over_extended", {})
        v["signature"] = dict(gain=a["signature"]["gain"],
                              matches=bool(a["signature"]["gain"] >= 3 and (not ov or ov["signature"]["gain"] >= ov["boundary"]["gain"])))
        z, sm, lg = a["zoom_pair"], out["small"].get("zoom_pair"), out["large"].get("zoom_pair")
        v["zoom_pair"] = dict(gain=z["gain"], oracle=a["zoom_oracle"]["gain"], small=sm and sm["gain"], large=lg and lg["gain"],
                              query_only=a["zoom_query"]["gain"],
                              matches=bool(a["zoom_oracle"]["gain"] >= 5 and z["gain"] >= 2.5 and sm and sm["gain"] >= 6
                                           and lg and abs(lg["gain"]) <= 0.5 and z["gain"] >= a["zoom_query"]["gain"]))
    if cut.get("learned") is not None and cut["oracle"] > cut["native"]:
        share = (cut["learned"] - cut["native"]) / (cut["oracle"] - cut["native"])
        v["learned"] = dict(share_of_oracle_gap=share, matches=bool(share >= 0.5), refuted=bool(share < 0.25))
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    recs = load(a.run)
    arms = [k for k in recs[0]["original_iu"] if k != "native"]
    sub = lambda f: [r for r in recs if f(r)]
    out = dict(state="ANALYSED", episodes=len(recs), fresh=len(sub(lambda r: not r["dev40"])),
               native=float(miou(*[np.array([r["original_iu"]["native"][j] for r in recs], float) for j in (0, 1)],
                                 np.array([r["c"] for r in recs]))[0]),
               all=table(recs, arms), fresh_only=table(sub(lambda r: not r["dev40"]), arms),
               dev40=table(sub(lambda r: r["dev40"]), arms),
               small=table(sub(lambda r: r["area"] < SMALL), arms), large=table(sub(lambda r: r["area"] >= SMALL), arms),
               by_fold={str(f): table(sub(lambda r: r["fold"] == f), arms) for f in sorted({r["fold"] for r in recs})},
               by_native_error=dict(over_extended=table(sub(lambda r: ratio(r) > 1.25), arms),
                                    under_extended=table(sub(lambda r: ratio(r) < 0.8), arms)),
               counts=dict(small=len(sub(lambda r: r["area"] < SMALL)), over_extended=len(sub(lambda r: ratio(r) > 1.25)),
                           under_extended=len(sub(lambda r: ratio(r) < 0.8)),
                           zoomed=len(sub(lambda r: bool(r["zoom"].get("zoom_pair"))))),
               cut_before_refinement=cut_family(recs), cards=CARDS,
               scope="class mIoU at original resolution unless stated; paired bootstrap over episodes, 2000 draws, seed 0")
    out["verdicts"] = verdicts(out)
    out["survivors"] = [k for k in arms if k not in ("oracle_cut", "zoom_oracle") and out["fresh_only"].get(k)
                        and out["fresh_only"][k]["gain"] >= 2 and out["fresh_only"][k]["ci95"][0] > 0
                        and sum(t.get(k, {}).get("gain", 0) > 0 for t in out["by_fold"].values()) >= 3]
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    f = lambda t, k: "%6.2f %+6.2f [%+5.2f,%+5.2f] %3d/%-3d" % (t[k]["miou"], t[k]["gain"], *t[k]["ci95"], t[k]["up"], t[k]["down"]) if k in t else "-"
    print("episodes %d (fresh %d)  native %.2f" % (out["episodes"], out["fresh"], out["native"]))
    print("%-12s %-38s %-8s %-8s %-8s %-8s" % ("arm", "all: mIoU gain [95% CI] up/down", "fresh", "small", "large", "over"))
    g = lambda t, k: "%+6.2f" % t[k]["gain"] if k in t else "   -  "
    for k in arms:
        print("%-12s %-38s %-8s %-8s %-8s %-8s" % (k, f(out["all"], k), g(out["fresh_only"], k), g(out["small"], k),
                                                 g(out["large"], k), g(out["by_native_error"]["over_extended"], k)))
    print("cut family before refinement:", {k: None if v is None else round(v, 2) for k, v in out["cut_before_refinement"].items()})
    print("verdicts:", json.dumps(out["verdicts"]))
    print("survivors (fresh gain >= +2, interval above 0, positive in >= 3 folds):", out["survivors"])


if __name__ == "__main__":
    main()
