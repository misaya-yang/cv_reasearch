#!/usr/bin/env python3
"""Which new evidence could repair FoRIS's ranking inside the contested area? A replay on cached features.

Measured before this audit (results/extent_v1/diagnosis.json): inside the contested area (native mask united with
the target) the FoRIS score ranks a target patch above a non-target patch with probability 0.664; nothing stored from
the same pair ranks better. A rule that re-reads the same evidence cannot help, so this audit asks which *source* of
new evidence would, before any method is built. Every source has a label-free form and the same form given labels
(its upper bound); `class_probe` is the ceiling of the frozen features and is available to no method.

  competitors  what else a query patch could be: patches of other images (all / not claimed by the reference /
               from images without the class), how common the patch is across images, a base-class classifier
  positives    more views of the target: pool patches the reference claims / pool patches of the class
  metric       a linear metric fitted on base classes (the other three folds) / on all classes
  grouping     the query's own similarity to what FoRIS is sure of / to the truly found part

  python scripts/evidence_audit.py --cache cache/evidence_v1 --run results/extent_v1/run --out results/evidence_v1
  python scripts/evidence_audit.py --analyse --run results/extent_v1/run --out results/evidence_v1   # CPU, seconds
  python scripts/evidence_audit.py --selfcheck [--out FILE]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_extent import load, miou  # noqa: E402

KINDS = ("lda_base", "probe_base", "lda_all", "probe_all")
PURE = 0.9
# name: (source, form, formula over the saved maps). Higher means more target-like.
EVIDENCE = dict(
    reference_margin=("reference", "baseline", lambda m: m["fg"] - m["bg"]),
    competitors_all=("competitors", "naive", lambda m: m["fg"] - np.maximum(m["bg"], m["neg_all"])),
    competitors_unclaimed=("competitors", "label_free", lambda m: m["fg"] - np.maximum(m["bg"], m["neg_unclaimed"])),
    commonness=("competitors", "label_free", lambda m: -m["common"]),
    base_class_claim=("competitors", "base_labels", lambda m: -m["base_claim"]),
    competitors_oracle=("competitors", "oracle", lambda m: m["fg"] - np.maximum(m["bg"], m["neg_oracle"])),
    commonness_oracle=("competitors", "oracle", lambda m: -m["common_oracle"]),
    positives_claimed=("positives", "label_free",
                       lambda m: np.maximum(m["fg"], m["pos_claimed"]) - np.maximum(m["bg"], m["neg_unclaimed"])),
    positives_oracle=("positives", "oracle",
                      lambda m: np.maximum(m["fg"], m["pos_oracle"]) - np.maximum(m["bg"], m["neg_oracle"])),
    metric_lda_base=("metric", "base_labels", lambda m: m["fg_lda_base"] - m["bg_lda_base"]),
    metric_probe_base=("metric", "base_labels", lambda m: m["fg_probe_base"] - m["bg_probe_base"]),
    metric_lda_all=("metric", "oracle", lambda m: m["fg_lda_all"] - m["bg_lda_all"]),
    metric_probe_all=("metric", "oracle", lambda m: m["fg_probe_all"] - m["bg_probe_all"]),
    grouping_core=("grouping", "label_free", lambda m: m["grp_core"] - m["grp_out"]),
    grouping_true=("grouping", "oracle", lambda m: m["grp_true"] - m["grp_out"]),
    class_probe=("ceiling", "oracle", lambda m: m["probe_class"]))
# The cards, written 2026-10-03 before the run. Baseline: FoRIS score 0.664, reference margin 0.627 (contested AUC).
CARDS = dict(
    ceiling=dict(
        assumption="the frozen last-layer patch features can tell target from non-target inside the contested area",
        prediction="class_probe contested AUC 0.85 to 0.92",
        match="the answer is in the features; what is missing is knowledge of the class, not resolution",
        mismatch="below 0.75: last-layer patches cannot resolve the contested area; stop all work on last-layer "
                 "evidence and measure other layers or a higher resolution instead"),
    competitors=dict(
        assumption="false positives are regions with no competitor in the reference; other images can supply one",
        prediction="oracle 0.70 to 0.76, label-free forms 0.64 to 0.70, unfiltered below the reference margin; "
                   "competitors do not recover the misses, so the oracle stays below positives_oracle",
        match="competitors alone are not enough; they are a component only if the oracle logistic gain is >= +3",
        mismatch="oracle below 0.68: supplying competitors does not reorder the contested area, close the source; "
                 "a label-free form >= 0.75: build FoRIS with an extended background set next"),
    positives=dict(
        assumption="misses are parts the single reference does not show; more views of the class supply them",
        prediction="oracle 0.82 to 0.90; pool patches claimed by the reference 0.64 to 0.70",
        match="the gain needs correct labels on the pool; the reference's own claim is too noisy, as measured "
              "before on INSID3 (pool with true masks +12.7, pseudo masks +2 to +4)",
        mismatch="claimed form >= 0.75: the pool can be labelled well enough at patch level, build it; "
                 "oracle below 0.75: even a labelled pool does not help, close the transductive line for good"),
    metric=dict(
        assumption="cosine similarity mixes class with nuisance; base classes can teach which directions matter",
        prediction="base-class forms 0.63 to 0.69; all-class forms above them by 0.05 to 0.12",
        match="a metric learned on base classes does not transfer enough: training a small head is not the route",
        mismatch="a base-class form >= 0.75 with logistic gain >= +3: training on base classes is the mechanism; "
                 "the next run fits it on train2014 and plugs it into FoRIS unchanged"),
    grouping=dict(
        assumption="the missed part belongs to the same object as the found part and looks like it in the query",
        prediction="true found part 0.74 to 0.82; FoRIS's own confident core 0.64 to 0.70",
        match="grouping helps only when the seed is right; FoRIS already uses its own seed this way",
        mismatch="core form >= 0.75: propagate from the core inside the query, build it; "
                 "true-part form below 0.70: the object is not coherent in these features, close the source"))
GATE = dict(label_free_auc=0.75, label_free_gain=3.0, oracle_auc=0.80, oracle_gain=5.0)


def normalise(x):
    x = x - x.min()
    return x / max(float(x.max()), 1e-6)


def auc(v, pos, neg):
    from scipy.stats import rankdata
    if not pos.any() or not neg.any() or not np.isfinite(v[pos | neg]).all():
        return np.nan
    r = rankdata(np.concatenate([v[pos], v[neg]]))
    n = int(pos.sum())
    return float((r[:n].sum() - n * (n + 1) / 2) / (n * int(neg.sum())))


# ------------------------------------------------------------------ replay (torch)

def fit_lda(X, y, classes, shrink=0.1, least=20):
    """Whitening by the pooled within-class covariance of the given classes."""
    import torch
    C, n, kept = X.shape[1], 0, 0
    sw = torch.zeros(C, C, dtype=torch.float64, device=X.device)
    for c in classes:
        m = y == c
        if int(m.sum()) < least:
            continue
        xc = X[m].double()
        xc = xc - xc.mean(0)
        sw += xc.T @ xc
        n += len(xc)
        kept += 1
    if kept < 2:
        return None
    sw /= n
    ev, vec = torch.linalg.eigh(sw + shrink * sw.diagonal().mean() * torch.eye(C, dtype=torch.float64, device=X.device))
    return dict(W=((vec * ev.rsqrt()) @ vec.T).float(), classes=kept)


def fit_probe(X, y, classes, steps=300, lr=0.05, scale=20.0, least=20, seed=0):
    """Linear classifier over label 0 (not annotated) and the given classes; balanced by inverse root frequency."""
    import torch
    ids = [0] + [c for c in classes if int((y == c).sum()) >= least]
    if len(ids) < 3 or int((y == 0).sum()) < least:
        return None
    remap = torch.full((256,), -1, dtype=torch.long, device=X.device)
    remap[torch.tensor(ids, device=X.device)] = torch.arange(len(ids), device=X.device)
    t = remap[y]
    X, t = X[t >= 0], t[t >= 0]
    wgt = torch.bincount(t, minlength=len(ids)).float().rsqrt()
    torch.manual_seed(seed)
    with torch.enable_grad():
        W = torch.zeros(len(ids), X.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(len(ids), device=X.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(scale * (X @ W.T) + b, t, weight=wgt)
            loss.backward()
            opt.step()
        acc = float(((X @ W.T) * scale + b).argmax(1).eq(t).float().mean())
    return dict(W=W.detach(), b=b.detach(), ids=ids, scale=scale, train_accuracy=acc, classes=len(ids) - 1)


def fit_metrics(pool, dev):
    """Per feature space: base-class metrics for each fold (its 20 classes withheld) and all-class metrics."""
    import torch
    lab = pool["label"].reshape(-1).long().to(dev)
    pure = (pool["purity"].reshape(-1).float() >= PURE).to(dev)
    per = pool["label"].shape[1]
    out, info = {}, {}
    for space in ("raw", "deb"):
        X = pool[space].reshape(-1, pool[space].shape[-1]).to(dev).float()
        for tag in (0, 1, 2, 3, "all"):
            novel = set() if tag == "all" else {tag + 4 * v + 1 for v in range(20)}
            classes = [c for c in range(1, 81) if c not in novel]
            clean = torch.tensor([not (novel & set(p)) for p in pool["present"]], device=dev).repeat_interleave(per)
            use = pure & ((lab == 0) & clean | (lab > 0) & (lab <= 80))
            kind = "all" if tag == "all" else "base"
            out[space, "lda_" + kind, tag] = fit_lda(X[use], lab[use], classes)
            out[space, "probe_" + kind, tag] = fit_probe(X[use], lab[use], classes)
            for k in ("lda_", "probe_"):
                m = out[space, k + kind, tag]
                info["%s/%s%s/%s" % (space, k, kind, tag)] = None if m is None else {
                    x: m[x] for x in ("classes", "train_accuracy") if x in m}
    return out, info


def reduce_pool(q, r, fgm, P, lab_c, pure, has_c, images=64):
    """Maxima of query-to-pool similarity over subsets of pool patches, and per pool image. P: [N, per, C]."""
    import torch
    n, per = P.shape[:2]
    low = lambda: torch.full((q.shape[0],), -1.0, device=q.device)
    red = dict(neg_all=low(), neg_oracle=low(), neg_unclaimed=low(), pos_oracle=low(), pos_claimed=low())
    per_image = torch.empty(q.shape[0], n, device=q.device)
    for s in range(0, n, images):
        X = P[s:s + images].reshape(-1, P.shape[-1]).float()
        ref = X @ r.T
        claimed = (ref[:, fgm].amax(1) if fgm.any() else low()[:1]) > (ref[:, ~fgm].amax(1) if (~fgm).any() else low()[:1])
        S = q @ X.T
        per_image[:, s:s + images] = S.view(q.shape[0], -1, per).amax(2)
        subsets = dict(neg_all=torch.ones_like(claimed), neg_oracle=~has_c[s:s + images].repeat_interleave(per),
                       neg_unclaimed=~claimed, pos_oracle=(lab_c[s:s + images] & pure[s:s + images]).reshape(-1),
                       pos_claimed=claimed)
        for k, m in subsets.items():
            if m.any():
                red[k] = torch.maximum(red[k], S[:, m].amax(1))
    return red, per_image


def replay(row, feat, z, pool, metrics, dev, near):
    import torch
    import torch.nn.functional as F
    q, r = feat["q"].to(dev).float(), feat["r"].to(dev).float()
    space, c, fold = ("deb" if feat["debiased"] else "raw"), row["c"], row["fold"]
    cov = torch.from_numpy(z["cov"]).to(dev).flatten()
    fgm = cov >= 0.5
    edge = lambda s: (s[:, fgm].amax(1) if fgm.any() else torch.zeros(len(s), device=dev),
                      s[:, ~fgm].amax(1) if (~fgm).any() else torch.zeros(len(s), device=dev))
    m = {}
    m["fg"], m["bg"] = edge(q @ r.T)
    drift = float((m["fg"].cpu() - torch.from_numpy(z["fg_max"].astype(np.float32))).abs().max())
    has_c = torch.tensor([c + 1 in p for p in pool["present"]], device=dev)
    red, per_image = reduce_pool(q, r, fgm, pool[space], pool["label_dev"] == c + 1, pool["pure_dev"], has_c)
    m.update(red)
    beats = per_image > m["fg"][:, None]
    m["common"] = beats.float().mean(1)
    m["common_oracle"] = beats[:, ~has_c].float().mean(1) if (~has_c).any() else torch.full_like(m["common"], float("nan"))
    # Grouping inside the query; a patch never votes for itself or its close neighbours.
    sn = torch.from_numpy(normalise(z["score"].astype(np.float32))).to(dev).flatten()
    order = sn.argsort()
    core, outm = sn >= 0.8, sn <= 0.2
    if int(core.sum()) < 4:
        core = torch.zeros_like(core).index_fill(0, order[-16:], True)
    if int(outm.sum()) < 4:
        outm = torch.zeros_like(outm).index_fill(0, order[:16], True)
    G = (q @ q.T).masked_fill(near, -1.0)
    side = int(round(sn.numel() ** 0.5))
    ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
    truth = torch.from_numpy(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)) > 0.5).to(dev).flatten()
    m["grp_core"], m["grp_out"] = G[:, core].amax(1), G[:, outm].amax(1)
    found = (sn > 0.5) & truth
    m["grp_true"] = G[:, found].amax(1) if found.any() else torch.full_like(sn, float("nan"))
    nan = torch.full_like(sn, float("nan"))
    for kind in KINDS:
        met = metrics.get((space, kind, "all" if kind.endswith("all") else fold))
        if met is None:
            m["fg_" + kind] = m["bg_" + kind] = nan
        else:
            qw, rw = F.normalize(q @ met["W"].T, dim=1), F.normalize(r @ met["W"].T, dim=1)
            m["fg_" + kind], m["bg_" + kind] = edge(qw @ rw.T)
    pa, pb = metrics.get((space, "probe_all", "all")), metrics.get((space, "probe_base", fold))
    logp = lambda p: F.log_softmax(p["scale"] * (q @ p["W"].T) + p["b"], 1)
    m["probe_class"] = logp(pa)[:, pa["ids"].index(c + 1)] if pa is not None and c + 1 in pa["ids"] else nan
    m["base_claim"] = 1 - logp(pb)[:, 0].exp() if pb is not None else nan
    return {k: v.float().cpu().numpy().astype(np.float16) for k, v in m.items()}, drift, space


def run(a):
    import torch
    out = Path(a.out)
    if (out / "maps").exists() and any((out / "maps").iterdir()):
        raise SystemExit("fresh output directory required")
    (out / "maps").mkdir(parents=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        torch.set_num_threads(a.threads)
    recs = load(a.run)[:a.limit]
    report = dict(state="RUNNING", episodes=0, device=dev)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        (out / "maps_report.json").write_text(json.dumps(report, indent=1))
    save()
    try:
        pool = torch.load(Path(a.cache) / "pool.pt", map_location="cpu", weights_only=False)
        metrics, report["metrics"] = fit_metrics(pool, dev)
        for k in ("raw", "deb"):
            pool[k] = pool[k].to(dev)
        pool["label_dev"], pool["pure_dev"] = pool["label"].to(dev).long(), (pool["purity"].float() >= PURE).to(dev)
        counts = torch.bincount(pool["label"][pool["purity"].float() >= PURE].long().flatten(), minlength=81)
        report.update(pool_images=len(pool["names"]), pool_patches_per_image=int(pool["label"].shape[1]),
                      pure_patches_not_annotated=int(counts[0]), pure_patches_per_class_min=int(counts[1:81].min()),
                      pure_patches_per_class_median=float(counts[1:81].float().median()))
        save()
        near, drift, spaces = None, 0.0, {}
        with torch.no_grad():
            for row in recs:
                name = "%d_%d_%d" % (row["fold"], row["e"], row["c"])
                feat = torch.load(Path(a.cache) / "feat" / (name + ".pt"), map_location="cpu", weights_only=False)
                z = np.load(Path(a.run) / "packets" / (name + ".npz"))
                n = feat["q"].shape[0]
                if near is None or near.shape[0] != n:
                    side = int(round(n ** 0.5))
                    y, x = torch.arange(n, device=dev) // side, torch.arange(n, device=dev) % side
                    near = ((y[:, None] - y[None]).abs() <= 2) & ((x[:, None] - x[None]).abs() <= 2)
                maps, d, space = replay(row, feat, z, pool, metrics, dev, near)
                if d > 0.02:
                    raise RuntimeError("cached features do not reproduce the stored similarity on %s: %.4f" % (name, d))
                drift = max(drift, d)
                spaces[space] = spaces.get(space, 0) + 1
                np.savez_compressed(out / "maps" / (name + ".npz"), **maps)
                report.update(episodes=report["episodes"] + 1, max_similarity_drift=drift, spaces=spaces)
                if report["episodes"] % 20 == 0:
                    save()
                    print(json.dumps(dict(n=report["episodes"], s=round(time.monotonic() - start, 1))), flush=True)
        report["state"] = "COMPLETED"
        save()
        print(json.dumps({k: v for k, v in report.items() if k != "metrics"}), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


# ------------------------------------------------------------------ analysis (numpy, seconds)

def analyse(a):
    from sklearn.linear_model import LogisticRegression
    out = Path(a.out)
    recs = [r for r in load(a.run) if (out / "maps" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))).exists()]
    n = len(recs)
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    sn, tf, ev = [], [], {k: [] for k in EVIDENCE}
    for r in recs:
        name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
        z = np.load(Path(a.run) / "packets" / (name + ".npz"))
        m = {k: v.astype(np.float32) for k, v in np.load(out / "maps" / (name + ".npz")).items()}
        s = normalise(z["score"].astype(np.float32)).ravel()
        side = int(round(len(s) ** 0.5))
        ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
        sn.append(s)
        tf.append(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)).ravel())
        for k, (_, _, fn) in EVIDENCE.items():
            ev[k].append(fn(m))
    sn, tf = np.stack(sn), np.stack(tf)
    truth, native = tf > 0.5, sn > 0.5
    contested = truth | native

    def iu(pred):
        i = (tf * pred).sum(1)
        return np.stack([i, tf.sum(1) + pred.sum(1) - i], 1)
    base = iu(native)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    score = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])

    def held_out(cols, keep=None, start=None):
        """Logistic read-out of the given per-patch columns, fitted on the other folds (classes unseen). Episodes
        outside `keep` (evidence undefined there) are neither fitted on nor changed from `start`."""
        D = np.stack(cols, 2)  # [n, patches, d]
        keep = np.ones(n, bool) if keep is None else keep
        pred = np.zeros(sn.shape, bool) if start is None else start.copy()
        rng = np.random.default_rng(0)
        for f in np.unique(folds):
            held = {x for r in recs if r["fold"] == f for x in (r["support"], r["query"])}
            tr = np.array([j for j, r in enumerate(recs) if keep[j] and r["fold"] != f and r["support"] not in held
                           and r["query"] not in held], int)
            te = np.nonzero((folds == f) & keep)[0]
            if not len(te):
                continue
            if not len(tr):
                return None
            xs, ys = D[tr].reshape(-1, D.shape[-1]), truth[tr].ravel()
            pick = rng.choice(len(ys), min(a.train_rows, len(ys)), replace=False)
            mu, sd = xs[pick].mean(0), xs[pick].std(0) + 1e-6
            if len(np.unique(ys[pick])) < 2:
                return None
            model = LogisticRegression(C=1.0, max_iter=200).fit((xs[pick] - mu) / sd, ys[pick])
            pred[te] = model.predict_proba((D[te].reshape(-1, D.shape[-1]) - mu) / sd)[:, 1].reshape(len(te), -1) > 0.5
        return pred

    def paired(t, ref):
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(ref[:, 0], ref[:, 1], cls, w)
        return dict(gain=score(t) - score(ref), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                    folds_up=int(sum(score(t, folds == f) > score(ref, folds == f) for f in np.unique(folds))))
    control_mask = held_out([sn])
    control = None if control_mask is None else iu(control_mask)
    rows = dict(foris_score=dict(source="reference", form="baseline",
                                 contested_auc=float(np.nanmean([auc(sn[j], contested[j] & truth[j], contested[j] & ~truth[j]) for j in range(n)]))))
    for k, (source, form, _) in EVIDENCE.items():
        v = np.stack(ev[k])
        ok = np.isfinite(v).all(1)
        row = dict(source=source, form=form, episodes=int(ok.sum()),
                   contested_auc=float(np.nanmean([auc(v[j], contested[j] & truth[j], contested[j] & ~truth[j]) for j in range(n)]))
                   if ok.any() else None)
        if ok.sum() >= 40 and control is not None:
            t = held_out([sn, np.where(ok[:, None], v, 0.0)], ok, control_mask)
            if t is not None:
                t = iu(t)
                row.update(with_score_over_score_only=paired(t, control), with_score_over_native=paired(t, base), patch_miou=score(t))
        rows[k] = row
    verdict = {}
    for source in CARDS:
        of = lambda forms, key: [r[key] for r in rows.values() if r["source"] == source and r["form"] in forms and r.get(key) is not None]
        gain = lambda forms: [r["with_score_over_score_only"] for r in rows.values()
                              if r["source"] == source and r["form"] in forms and "with_score_over_score_only" in r]
        free, orc = ("label_free", "base_labels"), ("oracle",)
        verdict[source] = dict(
            best_label_free_auc=max(of(free, "contested_auc"), default=None), best_oracle_auc=max(of(orc, "contested_auc"), default=None),
            label_free_passes=any(g["gain"] >= GATE["label_free_gain"] and g["ci95"][0] > 0 for g in gain(free))
            and max(of(free, "contested_auc"), default=0) >= GATE["label_free_auc"],
            oracle_alive=max(of(orc, "contested_auc"), default=0) >= GATE["oracle_auc"]
            or any(g["gain"] >= GATE["oracle_gain"] for g in gain(orc)))
    res = dict(state="ANALYSED", episodes=n, scope="patch level, no refinement; contested area = native mask united with the target; "
               "read-outs fitted on three folds and scored on the fourth; exploratory, not a method score",
               native_patch_miou=score(base), score_only_readout=None if control is None else dict(patch_miou=score(control), **paired(control, base)),
               evidence=rows, gate=GATE, verdict=verdict, cards=CARDS)
    (out / "audit.json").write_text(json.dumps(res, indent=1))
    for k, r in rows.items():
        g = r.get("with_score_over_score_only")
        print("%-24s %-12s %-11s auc %s  gain %s" % (k, r["source"], r["form"], "  n/a" if r.get("contested_auc") is None else "%.3f" % r["contested_auc"],
                                                     "n/a" if g is None else "%+.2f [%+.2f, %+.2f]" % (g["gain"], *g["ci95"])))
    print(json.dumps(dict(state="ANALYSED", native_patch_miou=res["native_patch_miou"], verdict=verdict)))


def synthetic(root, episodes=48, side=16, dim=32, images=300, per=32, seed=0):
    """A small world with the structure the audit asks about: every image shifts each class by its own offset, so
    one reference is an imperfect view of the class and a pool of other images carries what it lacks."""
    import torch
    rng = np.random.default_rng(seed)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    centre = unit(rng.normal(size=(90, dim)))  # 80 classes, then 10 kinds of unannotated background

    def image(concepts):
        """Features of a grid whose cells carry the given concept ids, with one offset per concept in this image."""
        off = {k: 0.7 * unit(rng.normal(size=dim)) for k in np.unique(concepts)}
        return unit(np.stack([centre[k] + off[k] for k in concepts.ravel()]) + 0.25 * rng.normal(size=(concepts.size, dim)) / dim ** 0.5 * 3)
    (root / "cache/feat").mkdir(parents=True)
    (root / "run/packets").mkdir(parents=True)
    rows = []
    for e in range(episodes):
        fold = e % 4
        c = fold + 4 * int(rng.integers(20))
        other = int(rng.choice([k for k in range(80) if k != c]))
        qc = np.full((side, side), 80 + int(rng.integers(10)))
        qc[3:9, 2:8], qc[3:9, 8:13] = c, other  # the target next to another class
        rc = np.full((side, side), 80 + int(rng.integers(10)))
        rc[5:12, 4:11] = c
        q, r = image(qc), image(rc)
        cov = (rc == c).astype(np.float32)
        sim = q @ r.T
        fg, bg = sim[:, cov.ravel() > 0.5].max(1), sim[:, cov.ravel() < 0.5].max(1)
        name = "%d_%d_%d" % (fold, e, c)
        np.savez(root / "run/packets" / (name + ".npz"), score=(fg - bg + 0.05 * rng.normal(size=fg.shape)).reshape(side, side).astype(np.float32),
                 cov=cov, fg_max=fg.astype(np.float32), truth=np.packbits(np.kron(qc == c, np.ones((16, 16), bool))))
        torch.save(dict(q=torch.from_numpy(q).half(), r=torch.from_numpy(r).half(), debiased=False), root / "cache/feat" / (name + ".pt"))
        rows.append(dict(fold=fold, e=e, c=c, support="s%d" % e, query="q%d" % e))
    (root / "run/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    feats, labels, present = [], [], []
    for _ in range(images):
        pc = np.full(per, 80 + int(rng.integers(10)))
        a, b = rng.choice(80, 2, replace=False)
        pc[:per // 4], pc[per // 4:per // 2] = a, b
        feats.append(image(pc))
        labels.append(np.where(pc < 80, pc + 1, 0))
        present.append(sorted({0, int(a) + 1, int(b) + 1}))
    x = torch.from_numpy(np.stack(feats)).half()
    torch.save(dict(raw=x, deb=x, label=torch.from_numpy(np.stack(labels)).to(torch.uint8), purity=torch.ones(images, per).half(),
                    present=present, names=["p%d" % i for i in range(images)], seed=seed), root / "cache/pool.pt")


def selfcheck(save=None):
    """Chunked pool reductions against brute force, the rank statistic against counting, the fits on separable data,
    then the whole replay and analysis on a synthetic world."""
    import tempfile
    import torch
    torch.manual_seed(0)
    g = lambda *s: torch.nn.functional.normalize(torch.randn(*s), dim=-1)
    q, r, P = g(40, 8), g(30, 8), g(9, 5, 8)
    fgm = torch.arange(30) < 7
    lab_c, pure, has_c = torch.rand(9, 5) < 0.3, torch.rand(9, 5) < 0.8, torch.tensor([1, 0, 0, 1, 0, 0, 0, 1, 0]).bool()
    red, per_image = reduce_pool(q, r, fgm, P, lab_c, pure, has_c, images=4)
    X = P.reshape(-1, 8)
    S, ref = q @ X.T, X @ r.T
    claimed = ref[:, fgm].amax(1) > ref[:, ~fgm].amax(1)
    brute = dict(neg_all=S.amax(1), neg_oracle=S[:, ~has_c.repeat_interleave(5)].amax(1), neg_unclaimed=S[:, ~claimed].amax(1),
                 pos_oracle=S[:, (lab_c & pure).reshape(-1)].amax(1), pos_claimed=S[:, claimed].amax(1))
    out = {"pool_" + k: bool(torch.allclose(red[k], v, atol=1e-6)) for k, v in brute.items()}
    out["pool_per_image"] = bool(torch.allclose(per_image, S.view(40, 9, 5).amax(2), atol=1e-6))
    v, pos = np.array([0.1, 0.4, 0.4, 0.8, 0.9, 0.2]), np.array([0, 1, 0, 1, 1, 0], bool)
    count = np.mean([(x > y) + 0.5 * (x == y) for x in v[pos] for y in v[~pos]])
    out["auc_counts_pairs"] = bool(abs(auc(v, pos, ~pos) - count) < 1e-12)
    centres = g(4, 8)
    y = torch.arange(4).repeat_interleave(60)
    Xc = torch.nn.functional.normalize(centres[y] + 0.05 * torch.randn(240, 8), dim=1)
    probe = fit_probe(Xc, y, [1, 2, 3], steps=200)
    out["probe_separates"] = bool(probe is not None and probe["train_accuracy"] > 0.98)
    lda = fit_lda(Xc, y, [1, 2, 3])
    out["lda_whitens"] = bool(lda is not None and torch.isfinite(lda["W"]).all() and lda["W"].shape == (8, 8))
    out["too_few_classes_declines"] = fit_lda(Xc, y, [1]) is None and fit_probe(Xc, y, [1]) is None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        synthetic(root)
        a = argparse.Namespace(cache=str(root / "cache"), run=str(root / "run"), out=str(root / "audit"), device="cpu",
                               threads=1, limit=None, train_rows=20000)
        run(a)
        analyse(a)
        res = json.loads((root / "audit/audit.json").read_text())
    ev = res["evidence"]
    out["synthetic_all_rows_scored"] = all(r.get("contested_auc") is not None for r in ev.values()) and all(
        "with_score_over_score_only" in r for k, r in ev.items() if k != "foris_score")
    out["synthetic_ceiling_above_reference"] = bool(ev["class_probe"]["contested_auc"] > ev["reference_margin"]["contested_auc"] + 0.05)
    out["synthetic_labelled_pool_above_reference"] = bool(ev["positives_oracle"]["contested_auc"] > ev["reference_margin"]["contested_auc"])
    out["synthetic_oracle_gate_reads_it"] = bool(res["verdict"]["positives"]["oracle_alive"])
    out.update(checks=len(out), passed=int(sum(out.values())), state="PASSED" if all(out.values()) else "FAILED")
    print(json.dumps(out, indent=1))
    if save:
        Path(save).write_text(json.dumps(out, indent=1))
    raise SystemExit(0 if out["state"] == "PASSED" else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--run", help="finished extent run: episodes.jsonl and packets")
    p.add_argument("--out")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--device")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int)
    p.add_argument("--train-rows", type=int, default=80000)
    a = p.parse_args()
    selfcheck(a.out) if a.selfcheck else analyse(a) if a.analyse else run(a)


if __name__ == "__main__":
    main()
