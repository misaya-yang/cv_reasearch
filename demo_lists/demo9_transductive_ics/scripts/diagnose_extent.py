#!/usr/bin/env python3
"""Find the cause of the extent_v1 outcome from its saved stream and packets. CPU only, no new inference.

  python scripts/diagnose_extent.py --run results/extent_v1/run --out results/extent_v1/diagnosis.json

Everything here uses query labels: it sizes where the error is and what a label-free rule would have to know. Rules
whose constants are chosen here are fitted on three folds and scored on the fourth. Masks are at model size
(1024 x 1024), so the native row differs slightly from the original-resolution row of analyze_extent.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from analyze_extent import SMALL, load, miou

SHIFTS = (-48, -32, -24, -16, -12, -8, -4, 0, 4, 8, 12, 16, 24, 32, 48)  # pixels at model size; negative erodes
LEVEL = lambda k: 0.2 + 0.0125 * k
RULES = ("contrast", "boundary", "round_trip", "signature")
RAW = np.linspace(-0.2, 1.2, 141)  # absolute thresholds for the unnormalised FoRIS score


def bits(z, k):
    return np.unpackbits(z[k])[:1 << 20].reshape(1024, 1024).astype(bool)


def iu(p, t):
    return [int((p & t).sum()), int((p | t).sum())]


def ledger(p, t):
    """Counterfactual masks: each repairs one kind of error with the truth and leaves the rest."""
    lp, lt = ndi.label(p)[0], ndi.label(t)[0]
    both = p & t
    located = np.isin(lp, np.unique(lp[both])) & p  # predicted components that overlap the target
    found = np.isin(lt, np.unique(lt[both])) & t  # target components the prediction overlaps
    return dict(native=iu(p, t), drop_attached_false_positive=iu(p & ~(located & ~t), t),
                fill_inside_located_target=iu(p | found, t), drop_wrong_components=iu(p & located, t),
                fill_missed_components=iu(p | (t & ~found), t),
                fix_located_components_only=iu((p & ~located) | found, t))


def shifted(p, t):
    """The native mask moved inwards or outwards by a constant width."""
    if not p.any():
        return {r: iu(p, t) for r in SHIFTS}
    inside, outside = ndi.distance_transform_edt(p), ndi.distance_transform_edt(~p)
    return {r: iu(inside > -r if r < 0 else (p if r == 0 else outside <= r), t) for r in SHIFTS}


def band(p, t):
    """How far the errors lie from the other mask's edge."""
    fp, fn = p & ~t, t & ~p
    out = dict(fp=int(fp.sum()), fn=int(fn.sum()))
    if t.any() and fp.any():
        d = ndi.distance_transform_edt(~t)[fp]
        out.update(fp16=int((d <= 16).sum()), fp32=int((d <= 32).sum()))
    if p.any() and fn.any():
        d = ndi.distance_transform_edt(~p)[fn]
        out.update(fn16=int((d <= 16).sum()), fn32=int((d <= 32).sum()))
    return out


def held_out(recs, table, key, bins=1):
    """Pick one column of `table` [n, k, 2] per bin of the label-free `key`, on the other folds; apply to the fold."""
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    pick = np.zeros(len(recs), int)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        edges = np.quantile(key[tr], np.linspace(0, 1, bins + 1)[1:-1]) if bins > 1 else np.array([])
        b = np.searchsorted(edges, key)
        for j in range(bins):
            m = tr & (b == j)
            if not m.any():
                continue
            score = [miou(table[m, k, 0], table[m, k, 1], cls[m])[0] for k in range(table.shape[1])]
            pick[te & (b == j)] = int(np.argmax(score))
    return pick


def paired(table, pick, base, cls, draws=2000, seed=0):
    n = len(cls)
    a, b = table[np.arange(n), pick], table[np.arange(n), base]
    w = np.random.default_rng(seed).multinomial(n, np.ones(n) / n, size=draws).astype(float)
    d = miou(a[:, 0], a[:, 1], cls, w) - miou(b[:, 0], b[:, 1], cls, w)
    return dict(miou=float(miou(a[:, 0], a[:, 1], cls)[0]),
                gain=float(miou(a[:, 0], a[:, 1], cls)[0] - miou(b[:, 0], b[:, 1], cls)[0]),
                ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])


def learner_in_sample(recs, mid):
    """The measuring device of analyze_extent.learned, scored on its own training episodes."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    def feats(r):
        L = r["levels"]
        n = len(L["I"])
        cols = [LEVEL(np.arange(n)), np.array(L["area"], float) / (r["grid"][0] * r["grid"][1])]
        for k in RULES:
            v = np.array([np.nan if x is None else x for x in L[k]], float)
            ok = np.isfinite(v)
            rank = np.full(n, np.nan)
            if ok.sum() > 1:
                rank[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
            cols += [v, rank, v - v[mid] if np.isfinite(v[mid]) else np.full(n, np.nan)]
        return np.stack(cols, 1), np.array(L["I"], float) / np.maximum(L["U"], 1)
    data = [feats(r) for r in recs]
    model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0)
    model.fit(np.concatenate([d[0] for d in data]), np.concatenate([d[1] for d in data]))
    return [int(np.where(np.array(r["levels"]["valid"]), model.predict(d[0]), -np.inf).argmax())
            if any(r["levels"]["valid"]) else mid for r, d in zip(recs, data)]


def auc(pos, neg):
    """Probability that a target patch outranks a non-target patch (ties count half)."""
    if not len(pos) or not len(neg):
        return np.nan
    order = np.concatenate([pos, neg]).argsort(kind="stable")
    v = np.concatenate([pos, neg])[order]
    rank = np.empty(len(v))
    rank[order] = np.arange(1, len(v) + 1)
    for x in np.unique(v[1:][v[1:] == v[:-1]]):  # average the ranks of ties
        m = np.concatenate([pos, neg]) == x
        rank[m] = rank[m].mean()
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evidence(z):
    """Patch level. Which stored evidence separates target from non-target inside the contested area (the native
    mask united with the target), and whether the reference holds a better competitor for the false positives."""
    t = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).ravel() > 0.5
    raw = z["score"].astype(float).ravel()
    sn = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-6)
    pm = sn > 0.5
    fg, bg = z["fg_max"].astype(float), z["bg_max"].astype(float)
    norm = lambda x: (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    feats = dict(foris_score=sn, stage2=norm(z["s2"].astype(float).ravel()), stage3=norm(z["s3"].astype(float).ravel()),
                 foreground_similarity=fg, minus_background_similarity=-bg, margin=fg - bg,
                 nearest_reference_label=z["cov"].astype(float).ravel()[z["fwd_idx"].astype(int)])
    con = pm | t
    out = {"auc_" + k: auc(v[con & t], v[con & ~t]) for k, v in feats.items()}
    for name, m in (("fp", pm & ~t), ("tp", pm & t), ("fn", ~pm & t), ("tn", ~pm & ~t)):
        out["n_" + name] = int(m.sum())
        out["bg_wins_" + name] = int((bg[m] > fg[m]).sum())
        out["bg_sim_" + name] = float(bg[m].sum())
        out["fg_sim_" + name] = float(fg[m].sum())
    # The raw score against a threshold on its own scale: I and U for a grid of absolute thresholds.
    tf = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).ravel()
    out["raw"] = [[float(tf[raw > x].sum()), float(tf.sum() + (raw > x).sum() - tf[raw > x].sum())] for x in RAW]
    out["mid"] = [float(tf[pm].sum()), float(tf.sum() + pm.sum() - tf[pm].sum())]
    out["raw_oracle"] = float(RAW[int(np.argmax([i / max(u, 1e-6) for i, u in out["raw"]]))])
    out["raw_min"], out["raw_max"] = float(raw.min()), float(raw.max())
    return out


def box_iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    recs = [r for r in load(a.run) if (a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))).exists()]
    n = len(recs)
    cls = np.array([r["c"] for r in recs])
    area = np.array([r["area"] for r in recs])
    evi, led, shf, bnd, key = [], [], [], [], dict(reference_area=[], score_max=[], score_range=[])
    for r in recs:
        z = np.load(a.run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))
        pm, t = bits(z, "native"), bits(z, "truth")
        evi.append(evidence(z))
        led.append(ledger(pm, t))
        shf.append(shifted(pm, t))
        bnd.append(band(pm, t))
        key["reference_area"].append(float(z["cov"].mean()))
        key["score_max"].append(float(z["score"].max()))
        key["score_range"].append(float(z["score"].max() - z["score"].min()))
    key = {k: np.array(v) for k, v in key.items()}
    out = dict(state="DIAGNOSED", episodes=n, scope="model-size masks; uses query labels; exploratory, not a method score")

    # 1. Where the error is.
    col = lambda rows, k: np.array([x[k] for x in rows], float)
    native = col(led, "native")
    nat = float(miou(native[:, 0], native[:, 1], cls)[0])
    out["ledger"] = {k: float(miou(col(led, k)[:, 0], col(led, k)[:, 1], cls)[0]) for k in led[0]}
    e_iou = native[:, 0] / np.maximum(native[:, 1], 1)
    out["native_episode_iou"] = dict(zero=int((e_iou == 0).sum()), below_0p1=int((e_iou < 0.1).sum()),
                                     below_0p5=int((e_iou < 0.5).sum()), median=float(np.median(e_iou)))
    tot = lambda k: float(sum(b.get(k, 0) for b in bnd))
    out["error_distance"] = dict(false_positive_within_16px_of_target=tot("fp16") / max(tot("fp"), 1),
                                 false_positive_within_32px=tot("fp32") / max(tot("fp"), 1),
                                 false_negative_within_16px_of_prediction=tot("fn16") / max(tot("fn"), 1),
                                 false_negative_within_32px=tot("fn32") / max(tot("fn"), 1),
                                 false_positive_share_of_error=tot("fp") / max(tot("fp") + tot("fn"), 1))
    # Episodes where the prediction does not overlap the target cannot be repaired by any cut of that component.
    located = e_iou > 0.1
    perfect = native.copy()
    perfect[located, 0] = perfect[located, 1] = 1
    out["ledger"]["all_located_episodes_perfect"] = float(miou(perfect[:, 0], perfect[:, 1], cls)[0])

    # 2. A constant-width shift of the native mask.
    S = np.array([[s[r] for r in SHIFTS] for s in shf], float)  # [n, shifts, 2]
    zero = SHIFTS.index(0)
    small = area < SMALL
    curve = lambda m: {str(r): float(miou(S[m, j, 0], S[m, j, 1], cls[m])[0]) for j, r in enumerate(SHIFTS)}
    best = (S[:, :, 0] / np.maximum(S[:, :, 1], 1)).argmax(1)
    pred_area = np.array([(x["native"][1] - r["area"] * (1 << 20) + x["native"][0]) / (1 << 20) for x, r in zip(led, recs)])
    out["shift"] = dict(all=curve(np.ones(n, bool)), small=curve(small), large=curve(~small),
                        oracle_per_episode=paired(S, best, zero, cls),
                        held_out_one_constant=paired(S, held_out(recs, S, pred_area), zero, cls),
                        held_out_by_predicted_area_3bins=paired(S, held_out(recs, S, pred_area, 3), zero, cls),
                        oracle_shift_median_small=float(np.median(np.array(SHIFTS)[best][small])),
                        oracle_shift_median_large=float(np.median(np.array(SHIFTS)[best][~small])))

    # 3. The level: what the oracle chooses and what would predict it.
    L = np.array([np.stack([r["levels"]["I"], r["levels"]["U"]], 1) for r in recs], float)  # [n, levels, 2]
    nl = L.shape[1]
    mid = int(np.argmin(np.abs(LEVEL(np.arange(nl)) - 0.5)))
    liou = L[:, :, 0] / np.maximum(L[:, :, 1], 1)
    star = liou.argmax(1)
    matters = liou.max(1) - liou[:, mid] > 0.02
    mid_area = np.array([r["levels"]["area"][mid] / (r["grid"][0] * r["grid"][1]) for r in recs])
    feats = dict(true_area=area, predicted_area_at_midpoint=mid_area, reference_area=key["reference_area"],
                 predicted_over_reference_area=mid_area / np.maximum(key["reference_area"], 1e-6),
                 raw_score_max=key["score_max"], raw_score_range=key["score_range"])
    out["oracle_level"] = dict(
        quartiles=[float(x) for x in np.quantile(LEVEL(star), [0.25, 0.5, 0.75])], episodes_where_level_matters=int(matters.sum()),
        tighter=int((LEVEL(star) > 0.55).sum()), looser=int((LEVEL(star) < 0.45).sum()),
        spearman_with={k: float(spearmanr(v[matters], star[matters])[0]) for k, v in feats.items()})
    pick_from = lambda vals: np.array([[int(np.argmin(np.abs(LEVEL(np.arange(nl)) - v))) for v in vals]] * n)

    def restricted(vals):  # oracle among a few levels
        idx = pick_from(vals)
        return paired(L, idx[np.arange(n), liou[np.arange(n)[:, None], idx].argmax(1)], mid, cls)
    out["level_rules"] = dict(
        native=float(miou(L[:, mid, 0], L[:, mid, 1], cls)[0]), oracle=paired(L, star, mid, cls),
        oracle_among_0p5_0p65=restricted((0.5, 0.65)), oracle_among_0p35_0p5=restricted((0.35, 0.5)),
        oracle_among_0p35_0p5_0p65=restricted((0.35, 0.5, 0.65)),
        held_out_one_constant=paired(L, held_out(recs, L, mid_area), mid, cls),
        held_out_by_predicted_area_3bins=paired(L, held_out(recs, L, mid_area, 3), mid, cls),
        held_out_by_reference_area_3bins=paired(L, held_out(recs, L, key["reference_area"], 3), mid, cls),
        held_out_by_area_ratio_3bins=paired(L, held_out(recs, L, feats["predicted_over_reference_area"], 3), mid, cls),
        held_out_by_true_area_3bins_uses_labels=paired(L, held_out(recs, L, area, 3), mid, cls),
        learner_in_sample=paired(L, np.array(learner_in_sample(recs, mid)), mid, cls))
    out["rule_direction"] = {}
    for k in RULES:
        pk = np.array([mid if r["pick"].get(k) is None else int(round((r["pick"][k] - 0.2) / 0.0125)) for r in recs])
        moved = matters & (np.abs(star - mid) >= 4) & (pk != mid)
        out["rule_direction"][k] = dict(mean_pick_minus_oracle=float((LEVEL(pk) - LEVEL(star)).mean()),
                                        mean_pick=float(LEVEL(pk).mean()),
                                        spearman_with_oracle=float(spearmanr(pk[matters], star[matters])[0]),
                                        same_direction_as_oracle=float((np.sign(pk - mid) == np.sign(star - mid))[moved].mean()),
                                        episodes=int(moved.sum()))

    # 4. Zoom: is the crop box of the first pass the limiter?
    ok = lambda r, k: r["zoom"][k][0]["box"] if r["zoom"].get(k) and r["zoom"][k][0].get("state") == "ok" else None
    O = {k: np.array([r["original_iu"][k] for r in recs], float) for k in ("native", "zoom_pair", "zoom_query", "zoom_oracle")}
    bi = np.array([box_iou(ok(r, "zoom_pair"), ok(r, "zoom_oracle")) if ok(r, "zoom_pair") and ok(r, "zoom_oracle") else np.nan
                   for r in recs])

    def gains(m):
        g = lambda k: float(miou(O[k][m, 0], O[k][m, 1], cls[m])[0] - miou(O["native"][m, 0], O["native"][m, 1], cls[m])[0])
        return dict(episodes=int(m.sum()), zoom_pair=g("zoom_pair"), zoom_query=g("zoom_query"), zoom_oracle=g("zoom_oracle"))
    both = np.isfinite(bi)
    out["zoom"] = dict(pair_zoomed=int(sum(ok(r, "zoom_pair") is not None for r in recs)),
                       oracle_zoomed=int(sum(ok(r, "zoom_oracle") is not None for r in recs)), both=int(both.sum()),
                       box_iou_quartiles=[float(x) for x in np.quantile(bi[both], [0.25, 0.5, 0.75])] if both.any() else None,
                       boxes_agree_iou_ge_0p6=gains(both & (np.nan_to_num(bi) >= 0.6)),
                       boxes_differ_iou_lt_0p6=gains(both & (np.nan_to_num(bi) < 0.6)),
                       oracle_only=gains(np.array([ok(r, "zoom_oracle") is not None and ok(r, "zoom_pair") is None for r in recs])))
    out["native_model_size"] = nat

    # 5. Evidence inside the contested area, competitors in the reference, and the score's own scale.
    out["contested_auc_mean"] = {k[4:]: float(np.nanmean([e[k] for e in evi])) for k in evi[0] if k.startswith("auc_")}
    tot = lambda k: float(sum(e[k] for e in evi))
    out["reference_competitor"] = {m: dict(patches=int(tot("n_" + m)), background_beats_foreground=tot("bg_wins_" + m) / max(tot("n_" + m), 1),
                                           mean_background_similarity=tot("bg_sim_" + m) / max(tot("n_" + m), 1),
                                           mean_foreground_similarity=tot("fg_sim_" + m) / max(tot("n_" + m), 1))
                                   for m in ("tp", "fp", "fn", "tn")}
    R = np.array([[e["mid"]] + e["raw"] for e in evi])  # [n, 1 + raw thresholds, 2]; column 0 is FoRIS's own cut
    rmin, rmax, ro = (np.array([e[k] for e in evi]) for k in ("raw_min", "raw_max", "raw_oracle"))
    out["raw_scale"] = dict(
        patch_level_native=float(miou(R[:, 0, 0], R[:, 0, 1], cls)[0]),
        held_out_one_absolute_threshold=paired(R, held_out(recs, R[:, 1:], pred_area) + 1, 0, cls),
        raw_min_quartiles=[float(x) for x in np.quantile(rmin, [0.25, 0.5, 0.75])],
        raw_max_quartiles=[float(x) for x in np.quantile(rmax, [0.25, 0.5, 0.75])],
        oracle_absolute_threshold_quartiles=[float(x) for x in np.quantile(ro, [0.25, 0.5, 0.75])],
        oracle_normalised_threshold_quartiles=[float(x) for x in np.quantile((ro - rmin) / np.maximum(rmax - rmin, 1e-6), [0.25, 0.5, 0.75])])
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
