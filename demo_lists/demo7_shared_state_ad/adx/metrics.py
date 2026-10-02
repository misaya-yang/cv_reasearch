"""Pixel-level metrics on pooled images. scores, gt: lists of 2-D numpy arrays (float32 / bool)."""
import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def pooled(scores, gts):
    return np.concatenate([s.ravel() for s in scores]), np.concatenate([g.ravel() for g in gts])


def au_pro(scores, gts, max_fpr=0.05, steps=200):
    """area under the per-region-overlap curve up to max_fpr, normalised to [0,1]"""
    s, g = pooled(scores, gts)
    neg = np.sort(s[~g])
    if len(neg) == 0 or g.sum() == 0: return float("nan")
    fprs = np.linspace(0, max_fpr, steps)
    # threshold t_k such that the share of normal pixels with score > t_k equals fpr_k
    thr = neg[np.clip(len(neg) - 1 - np.floor(fprs * len(neg)).astype(int), 0, len(neg) - 1)]
    pro = []
    for sc, gt in zip(scores, gts):
        if not gt.any(): continue
        lab, n = ndimage.label(gt)
        for c in range(1, n + 1):
            v = np.sort(sc[lab == c])
            pro.append(1.0 - np.searchsorted(v, thr, side="right") / len(v))
    pro = np.mean(pro, 0)
    return float(np.trapezoid(pro, fprs) / max_fpr)


def seg_metrics(scores, gts, thr=None):
    s, g = pooled(scores, gts)
    out = {}
    if g.sum() == 0 or g.all():
        return dict(ap=float("nan"), auroc05=float("nan"), aupro=float("nan"), f1max=float("nan"), f1=float("nan"),
                    fpr=float((s[~g] > thr).mean()) if thr is not None else float("nan"))
    out["ap"] = float(average_precision_score(g, s))
    out["auroc05"] = float(roc_auc_score(g, s, max_fpr=0.05))
    out["aupro"] = au_pro(scores, gts)
    # best F1 over a global threshold (hindsight)
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(g[order]); k = np.arange(1, len(s) + 1)
    f1 = 2 * tp / (k + g.sum())
    out["f1max"] = float(f1.max())
    if thr is not None:
        p = s > thr
        tpc = float((p & g).sum())
        out["f1"] = 2 * tpc / (p.sum() + g.sum()) if p.sum() + g.sum() > 0 else 0.0
        out["fpr"] = float((p & ~g).sum() / max(1, (~g).sum()))
        out["rec"] = tpc / g.sum()
    return out
