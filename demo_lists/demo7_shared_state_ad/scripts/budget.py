#!/usr/bin/env python3
"""CPU step: where does the baseline lose on MVTec AD 2 test_public?  Works on the tables written by bank.py.

For one score map (default: bank B = SuperADD augmentation) it reports, per category:
  - pixel F1 at the calibrated threshold, at the best global threshold, and with a per-image best threshold
  - oracles at the best global threshold: silence good images; drop false-positive blobs that touch no defect;
    drop false positives within a few pixels of a defect (halo); fill every defect that was touched; all defects found
  - how the missed defect area splits into defects that were missed entirely vs. partly covered, and by defect size
Pixels are at 1/4 of the input resolution (the evaluation resolution of the baseline); one ViT token covers 6.4 such pixels.
"""
import argparse, os, sys, json
import numpy as np, torch
import torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE

ap = argparse.ArgumentParser()
ap.add_argument("--cats", default="vial")
ap.add_argument("--tag", default="")
ap.add_argument("--map", default="dB")
ap.add_argument("--cond", default="all", help="all | regular | shifted")
ap.add_argument("--halo", type=int, default=6)
a = ap.parse_args()


def f1_of(pred, gt):
    tp = float((pred & gt).sum()); return 2 * tp / max(1.0, float(pred.sum() + gt.sum()))


def best_thr(s, g, n=400):
    """best global threshold on a quantile grid of the positive-class scores and the top normal scores"""
    cand = np.unique(np.quantile(s, np.linspace(0.90, 0.99999, n)))
    best = (0, cand[0])
    gs = g.sum()
    for t in cand:
        p = s > t
        tp = (p & g).sum()
        f = 2 * tp / max(1, p.sum() + gs)
        if f > best[0]: best = (f, t)
    return best


rows = []
for cat in a.cats.split(","):
    R = torch.load(f"{CACHE}/results/{cat}{a.tag}.pt", weights_only=False)
    meta, gt = R["meta"], R["gt"].numpy()
    size = gt.shape[1:]
    s = F.interpolate(R[a.map].mean(1)[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0].numpy()
    kind = np.array([m["kind"] for m in meta]); cond = np.array([m["cond"] for m in meta])
    test = np.isin(kind, ["good", "bad"])
    if a.cond == "regular": test &= cond == "regular"
    if a.cond == "shifted": test &= cond != "regular"
    calib = ~np.isin(kind, ["good", "bad"])
    thr_cal = float(np.percentile(s[calib], 95) * 1.421)
    S, G, K = s[test], gt[test], kind[test]
    f_cal = f1_of(S > thr_cal, G)
    f_best, t = best_thr(S.ravel(), G.ravel())
    P = S > t
    # per-image best threshold on bad images (the challenge's local "SegF1")
    per_img = []
    for i in np.where(K == "bad")[0]:
        if G[i].any(): per_img.append(best_thr(S[i].ravel(), G[i].ravel(), 100)[0])
    # oracles at the best global threshold
    P_good0 = P.copy(); P_good0[K == "good"] = False
    P_noiso = P.copy(); P_nohalo = P.copy(); P_fill = P.copy(); P_all = P.copy()
    miss_all = miss_part = 0.0; n_comp = n_missed = 0
    sizes, covs = [], []
    fp_iso = fp_halo = fp_far_bad = fp_good = 0.0
    for i in range(len(S)):
        lab, n = ndimage.label(P[i])
        if n:
            touch = np.unique(lab[G[i]]) if G[i].any() else np.array([], int)
            iso = ~np.isin(lab, touch) & P[i]
            P_noiso[i] &= ~iso
            if K[i] == "good": fp_good += iso.sum()
            else: fp_iso += iso.sum()
        if G[i].any():
            near = ndimage.binary_dilation(G[i], iterations=a.halo) & ~G[i]
            fp_halo += (P[i] & near).sum()
            P_nohalo[i] &= ~near
            gl, gn = ndimage.label(G[i])
            for c in range(1, gn + 1):
                m = gl == c; cov = float(P[i][m].mean()); n_comp += 1
                sizes.append(int(m.sum())); covs.append(cov)
                if cov == 0: n_missed += 1; miss_all += m.sum()
                else: miss_part += (m & ~P[i]).sum(); P_fill[i] |= m
            P_all[i] |= G[i]
    fp = float((P & ~G).sum()); fn = float((~P & G).sum()); gsum = float(G.sum())
    sizes, covs = np.array(sizes), np.array(covs)
    tok = 6.4 ** 2
    bins = [(0, tok), (tok, 4 * tok), (4 * tok, 16 * tok), (16 * tok, 1e12)]
    row = dict(cat=cat, f_cal=f_cal, f_best=f_best, f_img=float(np.mean(per_img)), good0=f1_of(P_good0, G), noiso=f1_of(P_noiso, G),
               nohalo=f1_of(P_nohalo, G), fill=f1_of(P_fill, G), allrec=f1_of(P_all, G), noiso_fill=f1_of(P_noiso | P_fill & ~P | (P_fill & P_noiso), G),
               fp_share=fp / max(1.0, fp + fn), fp_good=fp_good / max(1.0, fp), fp_iso=fp_iso / max(1.0, fp), fp_halo=fp_halo / max(1.0, fp),
               fn_missed=miss_all / max(1.0, fn), fn_part=miss_part / max(1.0, fn), n_comp=n_comp, missed_comp=n_missed / max(1, n_comp),
               size_share=[float(sizes[(sizes >= lo) & (sizes < hi)].sum() / max(1, sizes.sum())) for lo, hi in bins],
               size_cov=[float((sizes * covs)[(sizes >= lo) & (sizes < hi)].sum() / max(1, sizes[(sizes >= lo) & (sizes < hi)].sum())) for lo, hi in bins],
               size_n=[int(((sizes >= lo) & (sizes < hi)).sum()) for lo, hi in bins],
               defect_px=gsum / G.size)
    rows.append(row)

pc = lambda v: f"{100 * v:6.1f}"
print(f"map {a.map}, conditions: {a.cond}; F1 in %, pixels at 1/4 resolution")
print(f"{'category':12s} cal.thr  best.thr per-img | oracles at best thr: good=0  no-iso-FP no-halo  fill-hit  all-GT  iso+fill")
for r in rows:
    print(f"{r['cat']:12s} {pc(r['f_cal'])}  {pc(r['f_best'])}   {pc(r['f_img'])}  |                       {pc(r['good0'])}  {pc(r['noiso'])}   {pc(r['nohalo'])}  {pc(r['fill'])}  {pc(r['allrec'])}  {pc(r['noiso_fill'])}")
if len(rows) > 1:
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{'mean':12s} {pc(m('f_cal'))}  {pc(m('f_best'))}   {pc(m('f_img'))}  |                       {pc(m('good0'))}  {pc(m('noiso'))}   {pc(m('nohalo'))}  {pc(m('fill'))}  {pc(m('allrec'))}  {pc(m('noiso_fill'))}")
print("\nerror pixels at the best threshold")
print(f"{'category':12s} FP share | of FP: in good imgs  isolated blobs in bad imgs  halo(<= {a.halo}px) | of FN: defects missed entirely  partly | defects  missed(count)  defect px share")
for r in rows:
    print(f"{r['cat']:12s} {pc(r['fp_share'])}   |        {pc(r['fp_good'])}        {pc(r['fp_iso'])}                     {pc(r['fp_halo'])}        |        {pc(r['fn_missed'])}               {pc(r['fn_part'])} | {r['n_comp']:6d}   {pc(r['missed_comp'])}       {100 * r['defect_px']:.3f}")
print("\ndefect size (area in tokens: <1, 1-4, 4-16, >16): share of defect area | pixel recall | count")
for r in rows:
    print(f"{r['cat']:12s} " + " ".join(pc(v) for v in r["size_share"]) + "  |  " + " ".join(pc(v) for v in r["size_cov"]) + "  |  " + " ".join(f"{v:5d}" for v in r["size_n"]))
