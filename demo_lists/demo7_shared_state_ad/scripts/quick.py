#!/usr/bin/env python3
"""Minutes-scale test of the two questions that decide the shared-state idea, one category at a time:
  (H2) does a lighting change hurt the memory bank at all?        A (no augmentation) per lighting condition
  (H4) is there anything for an image-shared state to recover?    P (free per patch) vs S-cond / S-hind (one gain per image)

Small on purpose: M prototype images (all their tokens, no subsampling), a few gains, a subset of test scenes.
Pixel AP and best-threshold F1 are computed on the GPU at 1/4 input resolution. One line per category is appended
to results/quick<tag>.jsonl as soon as that category is done.
"""
import argparse, json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE, load_backbone, load_image, load_mask, cond_of, list_split, mask_path, extract, nn_dist

ap = argparse.ArgumentParser()
ap.add_argument("--cats", default="vial,fruit_jelly,sheet_metal,wallplugs,can")
ap.add_argument("--backbone", default="large")
ap.add_argument("--M", type=int, default=48)
ap.add_argument("--gains", default="0.87,1,1.13")
ap.add_argument("--scenes", type=int, default=10, help="bad test scenes per category (all their lighting conditions)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
gains = [float(g) for g in a.gains.split(",")]; g1 = gains.index(1.0)
model, layers = load_backbone(a.backbone); L = len(layers)
out_path = f"{CACHE}/results/quick{a.tag}.jsonl"


def ap_f1(s, g):
    """pixel average precision and best-threshold F1 of pooled scores (GPU)"""
    s, g = s.reshape(-1), g.reshape(-1)
    o = torch.argsort(s, descending=True); gs = g[o].double()
    tp = torch.cumsum(gs, 0); k = torch.arange(1, len(gs) + 1, device=s.device, dtype=torch.double)
    apv = float((tp / k * gs).sum() / gs.sum())
    return 100 * apv, 100 * float((2 * tp / (k + gs.sum())).max())


def grp(c): return "shift" if c.startswith("shift") else c


for cat in a.cats.split(","):
    t0 = time.time(); rng = np.random.default_rng(0)
    sp = list_split(cat)
    proto = [sp["proto"][i] for i in np.linspace(0, len(sp["proto"]) - 1, a.M).astype(int)]
    img_gain = rng.uniform(0.8, 1.2, size=len(proto))
    bank = {k: [[] for _ in layers] for k in ["B"] + gains}
    for i, p in enumerate(proto):
        img = load_image(p).cuda()
        for key, g in [("B", float(img_gain[i]))] + [(g, g) for g in gains]:
            f = extract(model, layers, img, g)
            for l in range(L): bank[key][l].append(f[l].reshape(-1, f[l].shape[-1]).half())
    bank = {k: [torch.cat(v) for v in b] for k, b in bank.items()}
    C = bank["B"][0].shape[1]
    scenes = sorted(set(cond_of(p)[0] for p in sp["bad"]))[:a.scenes]
    queries = [(p, "good") for p in sp["good"]] + [(p, "bad") for p in sp["bad"] if cond_of(p)[0] in scenes]
    S = {k: [] for k in ["A", "B", "P", "S-hind"]}; Dg = []; gts, conds, kinds = [], [], []
    for p, kind in queries:
        img = load_image(p).cuda(); H, W = img.shape[-2:]
        f = extract(model, layers, img, 1.0); ht, wt = f[0].shape[:2]
        D = torch.stack([torch.stack([nn_dist(f[l].reshape(-1, C), bank[g][l])[0] / C for g in gains], -1) for l in range(L)])  # (L, hw, G)
        dB = torch.stack([nn_dist(f[l].reshape(-1, C), bank["B"][l])[0] / C for l in range(L)])
        up = lambda m: F.interpolate(m.reshape(1, 1, ht, wt), size=(H // 4, W // 4), mode="bilinear", align_corners=False)[0, 0]
        if kind == "bad":
            gt = F.interpolate(load_mask(mask_path(p)).float()[None, None].cuda(), size=(H // 4, W // 4), mode="area")[0, 0] >= 0.5
        else:
            gt = torch.zeros(H // 4, W // 4, dtype=torch.bool, device="cuda")
        Dm = D.mean(0)                                             # (hw, G)
        per_g = torch.stack([up(Dm[:, gi]) for gi in range(len(gains))])
        S["A"].append(up(Dm[:, g1])); S["B"].append(up(dB.mean(0))); S["P"].append(up(D.min(-1).values.mean(0)))
        Dg.append(per_g); gts.append(gt); conds.append(cond_of(p)[1]); kinds.append(kind)
        # hindsight: the gain with the best per-image AP (bad) or the lowest maximum (good)
        if gt.any(): gi = int(np.argmax([ap_f1(per_g[k], gt)[0] for k in range(len(gains))]))
        else: gi = int(per_g.flatten(1).max(1).values.argmin())
        S["S-hind"].append(per_g[gi])
    conds = np.array(conds); kinds = np.array(kinds); groups = np.array([grp(c) for c in conds])
    # one gain per lighting condition: lowest mean (lowest-75%) distance on the good images of that condition
    trim = torch.stack([d.flatten(1).sort(1).values[:, :int(0.75 * d[0].numel())].mean(1) for d in Dg])   # (n, G)
    g_img = trim.argmin(1).cpu().numpy(); g_cond = np.zeros(len(queries), int); chosen = {}
    for c in np.unique(conds):
        gi = int(trim[torch.from_numpy((conds == c) & (kinds == "good")).cuda()].mean(0).argmin()); g_cond[conds == c] = gi; chosen[str(c)] = gains[gi]
    S["S-cond"] = [Dg[i][g_cond[i]] for i in range(len(queries))]
    S["S-est"] = [Dg[i][g_img[i]] for i in range(len(queries))]
    row = dict(cat=cat, backbone=a.backbone, M=a.M, gains=gains, n_queries=len(queries), chosen=chosen, sec=0, res={})
    names = ["A", "B", "P", "S-est", "S-cond", "S-hind"]
    print(f"\n=== {cat}: {len(proto)} prototype images, {len(queries)} queries, gain per condition {chosen}", flush=True)
    print("   pixel AP / best-threshold F1    " + "".join(f"{g:>16s}" for g in ["regular", "overexposed", "underexposed", "shift", "all"]))
    for nme in names:
        line = f"   {nme:8s}"
        for g in ["regular", "overexposed", "underexposed", "shift", "all"]:
            sel = np.where(groups == g)[0] if g != "all" else np.arange(len(queries))
            if len(sel) == 0: line += f"{'-':>16s}"; continue
            apv, f1 = ap_f1(torch.cat([S[nme][i].reshape(-1) for i in sel]), torch.cat([gts[i].reshape(-1) for i in sel]))
            row["res"].setdefault(nme, {})[g] = (apv, f1); line += f"{apv:9.2f}/{f1:6.2f}"
        print(line, flush=True)
    row["sec"] = time.time() - t0
    print(f"   ({row['sec']:.0f} s)", flush=True)
    with open(out_path, "a") as fh: fh.write(json.dumps(row) + "\n")
    del bank, S, Dg, gts; torch.cuda.empty_cache()
