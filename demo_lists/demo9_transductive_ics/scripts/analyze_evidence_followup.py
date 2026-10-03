#!/usr/bin/env python3
"""Why the label-free forms of the evidence audit fall short of their upper bounds. CPU only, from saved maps.

  python scripts/analyze_evidence_followup.py --run results/extent_v1/run --audit results/evidence_v1 \
      --out results/evidence_v1/followup.json

Three questions: which error each evidence can order (found against wrongly included, missed against wrongly
included, missed against correctly rejected); how the grouping evidence depends on the purity of its seed; how pure
any label-free seed is. Uses query labels; exploratory, patch level.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from analyze_extent import SMALL, load, miou
from evidence_audit import EVIDENCE, auc, normalise

KEYS = ("reference_margin", "competitors_oracle", "positives_claimed", "positives_oracle", "metric_probe_all",
        "grouping_core", "grouping_true", "class_probe")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    area = np.array([r["area"] for r in recs])
    sn, tf, ev, seeds = [], [], {k: [] for k in KEYS}, {}
    yy, xx = np.divmod(np.arange(4096), 64)
    for r in recs:
        name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
        z = np.load(a.run / "packets" / (name + ".npz"))
        m = {k: v.astype(np.float32) for k, v in np.load(a.audit / "maps" / (name + ".npz")).items()}
        s = normalise(z["score"].astype(np.float32)).ravel()
        t = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).ravel()
        sn.append(s)
        tf.append(t)
        for k in KEYS:
            ev[k].append(EVIDENCE[k][2](m))
        cov, fwd, back = z["cov"].ravel(), z["fwd_idx"].astype(int), z["back_idx"].astype(int)
        there = back[fwd]
        cyc = (cov[fwd] >= 0.5) & (np.hypot(yy[there] - yy, xx[there] - xx) <= 2)
        top = lambda v, k: np.isin(np.arange(4096), np.argsort(v)[-k:])
        for k, mask in dict(score_at_least_0p8=s >= 0.8, score_at_least_0p9=s >= 0.9, top16_by_score=top(s, 16),
                            cycle_consistent=cyc, margin_top16=top(z["fg_max"] - z["bg_max"], 16)).items():
            if mask.any():
                seeds.setdefault(k, []).append([float((t > 0.5)[mask].mean()), float((mask & (t > 0.5)).sum() / max((t > 0.5).sum(), 1))])
    sn, tf = np.stack(sn), np.stack(tf)
    truth, native = tf > 0.5, sn > 0.5
    out = dict(state="ANALYSED", episodes=n, scope="patch level; uses query labels; exploratory")

    # 1. Which error each evidence can order.
    def split(v):
        rows = [(auc(v[j], native[j] & truth[j], native[j] & ~truth[j]), auc(v[j], ~native[j] & truth[j], native[j] & ~truth[j]),
                 auc(v[j], ~native[j] & truth[j], ~native[j] & ~truth[j])) for j in range(n)]
        return dict(zip(("found_above_wrongly_included", "missed_above_wrongly_included", "missed_above_correctly_rejected"),
                        (float(x) for x in np.nanmean(np.array(rows, float), 0))))
    out["ordering"] = dict(foris_score=split(sn), **{k: split(np.stack(ev[k])) for k in KEYS})

    # 2. The grouping evidence against the purity of its seed.
    def iu(pred):
        i = (tf * pred).sum(1)
        return np.stack([i, tf.sum(1) + pred.sum(1) - i], 1)

    def held(cols, keep=None, start=None):
        D = np.stack(cols, 2)
        keep = np.ones(n, bool) if keep is None else keep
        pred = np.zeros(sn.shape, bool) if start is None else start.copy()
        rng = np.random.default_rng(0)
        for f in np.unique(folds):
            shut = {x for r in recs if r["fold"] == f for x in (r["support"], r["query"])}
            tr = np.array([j for j, r in enumerate(recs) if keep[j] and r["fold"] != f and r["support"] not in shut and r["query"] not in shut])
            te = np.nonzero((folds == f) & keep)[0]
            xs, ys = D[tr].reshape(-1, D.shape[-1]), truth[tr].ravel()
            pick = rng.choice(len(ys), 80000, replace=False)
            mu, sd = xs[pick].mean(0), xs[pick].std(0) + 1e-6
            model = LogisticRegression(C=1.0, max_iter=200).fit((xs[pick] - mu) / sd, ys[pick])
            pred[te] = model.predict_proba((D[te].reshape(-1, D.shape[-1]) - mu) / sd)[:, 1].reshape(len(te), -1) > 0.5
        return pred
    control = held([sn])
    C = iu(control)
    score = lambda t, m: float(miou(t[m, 0], t[m, 1], cls[m])[0])
    core = sn >= 0.8
    purity = np.array([truth[j][core[j]].mean() if core[j].sum() >= 4 else np.nan for j in range(n)])
    strata = dict(all=np.ones(n, bool), seed_purity_at_least_0p9=purity >= 0.9, seed_purity_0p5_to_0p9=(purity >= 0.5) & (purity < 0.9),
                  seed_purity_below_0p5=purity < 0.5, small_target=area < SMALL, large_target=area >= SMALL)
    out["strata_sizes"] = {k: int(v.sum()) for k, v in strata.items()}
    out["by_stratum"], out["error_change"] = {}, {}
    contested = truth | native
    for k in ("grouping_core", "grouping_true", "class_probe", "positives_oracle"):
        v = np.stack(ev[k])
        ok = np.isfinite(v).all(1)
        pred = held([sn, np.where(ok[:, None], v, 0)], ok, control)
        P = iu(pred)
        out["by_stratum"][k] = {s: dict(contested_auc=float(np.nanmean([auc(v[j], contested[j] & truth[j], contested[j] & ~truth[j]) for j in np.nonzero(m)[0]])),
                                        gain_over_score_only=score(P, m) - score(C, m)) for s, m in strata.items()}
        fp0, fn0, fp1, fn1 = (control & ~truth).sum(), (~control & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
        out["error_change"][k] = dict(false_positive_patches=[int(fp0), int(fp1)], missed_patches=[int(fn0), int(fn1)])

    # 3. How pure a label-free seed is.
    out["seed"] = {k: dict(episodes=len(v), mean_purity=float(np.mean([x[0] for x in v])),
                           share_purity_at_least_0p9=float(np.mean([x[0] >= 0.9 for x in v])),
                           share_purity_below_0p5=float(np.mean([x[0] < 0.5 for x in v])),
                           mean_target_coverage=float(np.mean([x[1] for x in v]))) for k, v in seeds.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
