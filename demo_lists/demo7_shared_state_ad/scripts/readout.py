#!/usr/bin/env python3
"""CPU step: read-outs of the saved distance tables and their metrics per lighting condition.

Read-outs (D = distance table [layer, patch, gain] of the paired grid; A, B = banks without / with SuperADD augmentation):
  A        no augmentation                       B     SuperADD augmentation (the baseline)
  An       no augmentation, query rescaled to the prototype mean intensity (simple normalisation control)
  G1       paired grid, gain-1 slice only (checks that random identities are as good as bank A)
  P        mean_l min_g D   every patch and layer picks its own gain
  C        min_g mean_l D   every patch picks one gain for all layers
  S        mean_l D[g^]     one gain per image, g^ = argmin_g of the mean over the lowest 75% of patch distances
  S-cond   one gain per lighting condition, chosen on the good test images of that condition (state known)
  S-hind   one gain per image chosen in hindsight with the ground truth (upper bound of any image-level choice)
Thresholds: SuperADD rule (1.421 x 95th percentile) on held-out normal images (train part + validation), per read-out.
"""
import argparse, os, sys, json
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE
from adx.metrics import seg_metrics
from sklearn.metrics import average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--cats", default="vial")
ap.add_argument("--tag", default="")
ap.add_argument("--keep", type=float, default=0.75)
ap.add_argument("--json", default="")
a = ap.parse_args()


def up(x, size):
    """(n, h, w) token maps -> (n, H/4, W/4) as in the baseline"""
    return F.interpolate(x[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]


def group_of(c):
    return "shift" if c.startswith("shift") else c


GROUPS = ["regular", "overexposed", "underexposed", "shift", "all"]
table = {}
for cat in a.cats.split(","):
    R = torch.load(f"{CACHE}/results/{cat}{a.tag}.pt", weights_only=False)
    meta, gt, gains = R["meta"], R["gt"].numpy(), R["gains"]
    n = len(meta); size = gt.shape[1:]
    D = R["D"]                                   # (n, L, h, w, G)
    kind = np.array([m["kind"] for m in meta]); cond = np.array([m["cond"] for m in meta])
    test = np.isin(kind, ["good", "bad"]); calib = ~test
    Dm = D.mean(1)                               # (n, h, w, G) mean over layers
    # image-level state estimate: trimmed mean of patch distances per gain
    flat = Dm.reshape(n, -1, len(gains))
    k = int(a.keep * flat.shape[1])
    trim = flat.sort(1).values[:, :k].mean(1)    # (n, G)
    g_hat = trim.argmin(1)
    read = {
        "A": R["dA"].mean(1), "B": R["dB"].mean(1), "An": R["dAn"].mean(1), "G1": Dm[..., gains.index(1.0)],
        "P": D.min(-1).values.mean(1), "C": Dm.min(-1).values,
        "S": Dm[torch.arange(n), :, :, g_hat],
    }
    # state known per lighting condition: the gain with the lowest trimmed distance on that condition's good images
    g_cond = g_hat.clone()
    cond_gain = {}
    for c in np.unique(cond[test]):
        sel = torch.from_numpy((cond == c) & (kind == "good"))
        gc = int(trim[sel].mean(0).argmin()); cond_gain[c] = gains[gc]
        g_cond[torch.from_numpy((cond == c) & test)] = gc
    g_cond[torch.from_numpy(calib)] = gains.index(1.0)
    read["S-cond"] = Dm[torch.arange(n), :, :, g_cond]
    ups = {k_: up(v.float(), size).numpy() for k_, v in read.items()}
    # hindsight: per bad image the gain with the best per-image AP; per good image the gain with the lowest maximum
    allg = np.stack([up(Dm[..., gi].float(), size).numpy() for gi in range(len(gains))], -1)   # (n, H, W, G)
    g_hind = g_hat.numpy().copy()
    for i in np.where(test)[0]:
        if gt[i].any():
            g_hind[i] = int(np.argmax([average_precision_score(gt[i].ravel(), allg[i, ..., gi].ravel()) for gi in range(len(gains))]))
        else:
            g_hind[i] = int(np.argmin([allg[i, ..., gi].max() for gi in range(len(gains))]))
    ups["S-hind"] = allg[np.arange(n), :, :, g_hind]
    del allg
    res = {}
    for name, s in ups.items():
        thr = float(np.percentile(s[calib], 95) * 1.421)
        res[name] = {}
        for grp in GROUPS:
            sel = test & (np.array([group_of(c) for c in cond]) == grp if grp != "all" else True)
            if sel.sum() == 0: continue
            res[name][grp] = seg_metrics(list(s[sel]), list(gt[sel]), thr)
    table[cat] = res
    gh = np.array(gains)[g_hat.numpy()]
    print(f"\n=== {cat}  ({R['backbone']}, K={R['K']}, grid {R['grid']}, gains {gains})")
    print("   gain per condition chosen on good images:", cond_gain)
    for c in np.unique(cond[test]):
        v = gh[test & (cond == c)]
        print(f"   estimated gain, {c:13s}: " + " ".join(f"{g}:{(v == g).sum()}" for g in gains))
    print("   estimated gain, calibration : " + " ".join(f"{g}:{(gh[calib] == g).sum()}" for g in gains))
    for metric in ["ap", "aupro", "f1", "fpr", "f1max"]:
        print(f"   -- {metric}")
        print("      " + " " * 8 + "".join(f"{g:>13s}" for g in GROUPS))
        for name in ups:
            print("      " + f"{name:8s}" + "".join(f"{100 * res[name].get(g, {}).get(metric, float('nan')):13.2f}" for g in GROUPS))

if len(table) > 1:
    print("\n=== mean over categories")
    for metric in ["ap", "aupro", "f1", "fpr", "f1max"]:
        print(f"   -- {metric}")
        print("      " + " " * 8 + "".join(f"{g:>13s}" for g in GROUPS))
        for name in next(iter(table.values())):
            row = [np.nanmean([100 * table[c][name].get(g, {}).get(metric, np.nan) for c in table]) for g in GROUPS]
            print("      " + f"{name:8s}" + "".join(f"{v:13.2f}" for v in row))
if a.json:
    json.dump(table, open(a.json, "w"), indent=1)
