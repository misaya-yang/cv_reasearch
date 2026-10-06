#!/usr/bin/env python3
"""Deleting unreliable connected components of the RCG mask: ceiling and fold-nested rules (CPU, token level).

Role: RELIABILITY. The mask at 0.5 is a union of connected components; a component should be kept only if its
precision exceeds the break-even J/(1+J) of the pooled class IoU (J the current score). The ceiling uses truth. Each
rule reads one statistic of the component from the fields (no truth) and deletes below a threshold; the threshold, or
the logistic weights, are fitted on the other folds without the photographs shared with the held fold.

  python scripts/score_components.py outputs/claude_region_a outputs/claude_region_b
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ics.experiment import photo_groups

rows, T = [], {}
for j, d in enumerate(sys.argv[1:]):
    rows += [dict(r, c=r["c"] + 1000 * j * int(os.environ.get("SEPARATE", "0")), key="%d/%s" % (j, r["key"])) for r in json.load(open(d + "/rows.json"))]; Z = np.load(d + "/tokens.npz")
    for k in ("truth", "rcg", "cos", "score"):
        T.setdefault(k, []).append(Z[k].astype(np.float32))
T = {k: np.concatenate(v) for k, v in T.items()}; N = len(rows); cls = np.array([r["c"] for r in rows]); fold = np.array([r["fold"] for r in rows]); grp = photo_groups(rows)
t, z = T["truth"], T["rcg"]; m = z > .5
smin, smax = T["score"].min(1, keepdims=True), T["score"].max(1, keepdims=True); sn = (T["score"] - smin) / np.maximum(smax - smin, 1e-9)
C = np.unique(cls); oh = np.zeros((N, len(C))); oh[np.arange(N), np.searchsorted(C, cls)] = 1; gid = np.unique(grp, return_inverse=True)[1]; G = gid.max() + 1


def miou(I, U, ix=None):
    ix = np.ones(N, bool) if ix is None else ix
    return float(np.mean([I[ix & (cls == c)].sum() / max(U[ix & (cls == c)].sum(), 1e-9) for c in np.unique(cls[ix])]) * 100)


def boot(a, b):
    rs = np.random.RandomState(0); out = []
    gs = lambda x: (lambda o: (np.add.at(o, gid, oh[:, :, None] * np.stack(x, 1)[:, None, :]), o)[1])(np.zeros((G, len(C), 2)))
    X, Y = gs(a), gs(b)
    for _ in range(2000):
        s = rs.randint(0, G, G); x, y = X[s].sum(0), Y[s].sum(0); ok = y[:, 1] > 0; out.append((np.mean(x[ok, 0] / np.maximum(x[ok, 1], 1e-9)) - np.mean(y[ok, 0] / y[ok, 1])) * 100)
    return miou(*a) - miou(*b), np.percentile(out, [2.5, 97.5])


comp = []                                           # episode, tokens, true area, false area, then the statistics
for i in range(N):
    lab, k = ndimage.label(m[i].reshape(64, 64)); lab = lab.flatten(); tot = max(m[i].sum(), 1); zmax = z[i][m[i]].max() if m[i].any() else 1
    for c in range(1, k + 1):
        s = lab == c; comp.append((i, s.sum(), t[i][s].sum(), (1 - t[i][s]).sum(), z[i][s].mean(), z[i][s].max(), z[i][s].max() / zmax, s.sum() / tot, np.log(s.sum()), sn[i][s].mean(), T["cos"][i][s].mean(), T["cos"][i][s].max()))
comp = np.array(comp); ep = comp[:, 0].astype(int); FEATS = dict(mean_field=4, max_field=5, max_field_rel=6, share_of_mask=7, log_size=8, mean_foris=9, mean_cos=10, max_cos=11)
base = (np.bincount(ep, comp[:, 2], N), t.sum(1) + np.bincount(ep, comp[:, 3], N)); J = miou(*base) / 100; be = J / (1 + J)


def apply(keep):
    return np.bincount(ep, comp[:, 2] * keep, N), t.sum(1) + np.bincount(ep, comp[:, 3] * keep, N)


def nested(feature_fn):
    keep = np.ones(len(comp), bool); picks = []
    for f in np.unique(fold):
        held = fold == f; fit = ~held & ~np.isin(grp, grp[held]); keep[held[ep]], pick = feature_fn(fit[ep], held[ep], fit); picks.append(pick)
    return apply(keep), picks


def threshold_rule(col):
    def fn(fit_c, held_c, fit_e):
        v = comp[:, col]; best = max(np.quantile(v[fit_c], np.linspace(0, .9, 46)), key=lambda th: miou(*apply(~fit_c | (v >= th)), fit_e)); return v[held_c] >= best, round(float(best), 3)
    return fn


def logistic(cols):
    def fn(fit_c, held_c, fit_e):
        X = comp[:, cols]; mu, sd = X[fit_c].mean(0), X[fit_c].std(0) + 1e-9; X = np.c_[(X - mu) / sd, np.ones(len(X))]; y = (comp[:, 2] / (comp[:, 2] + comp[:, 3]) >= be).astype(float); wgt = (comp[:, 2] + comp[:, 3]); w = np.zeros(X.shape[1])
        for _ in range(300):                                           # area-weighted logistic regression, plain gradient steps
            p = 1 / (1 + np.exp(-X[fit_c] @ w)); w -= .5 * (X[fit_c].T @ ((p - y[fit_c]) * wgt[fit_c]) / wgt[fit_c].sum() + 1e-3 * w)
        s = X @ w; best = max(np.quantile(s[fit_c], np.linspace(0, .9, 46)), key=lambda th: miou(*apply(~fit_c | (s >= th)), fit_e)); return s[held_c] >= best, [round(float(x), 2) for x in w]
    return fn


print("## reliability role: delete connected components of the RCG mask, %d episodes, %d components, token-level class mIoU" % (N, len(comp)))
print("break-even precision %.2f; components below it: %d holding %.0f%% of the false area" % (be, (comp[:, 2] / (comp[:, 2] + comp[:, 3]) < be).sum(), 100 * comp[comp[:, 2] / (comp[:, 2] + comp[:, 3]) < be, 3].sum() / comp[:, 3].sum()))
print("| rule | mIoU | vs RCG at 0.5 | folds | fitted |"); print("|---|---:|---|---|---|")
line = lambda name, r, picks="": print("| %s | %.2f | %+.2f [%+.2f, %+.2f] | %s | %s |" % (name, miou(*r), boot(r, base)[0], *boot(r, base)[1], " / ".join("%+.2f" % (miou(*r, fold == f) - miou(*base, fold == f)) for f in np.unique(fold)), picks))
line("RCG at 0.5 (control)", base); line("ceiling: delete below break-even (truth)", apply(comp[:, 2] / (comp[:, 2] + comp[:, 3]) >= be))
line("control: keep the largest component only", apply(comp[:, 7] >= np.array([comp[ep == i, 7].max() for i in range(N)])[ep]) if len(comp) else base)
for name, col in FEATS.items():
    r, picks = nested(threshold_rule(col)); line("delete if %s below threshold" % name, r, picks)
r, picks = nested(logistic([4, 5, 6, 7, 8])); line("logistic on field statistics and size", r, "")
r, picks = nested(logistic([4, 5, 6, 7, 8, 9, 10, 11])); line("logistic on all statistics", r, "")
