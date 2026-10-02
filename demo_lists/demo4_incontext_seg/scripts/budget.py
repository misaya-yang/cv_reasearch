"""Error budget of INSID3 on COCO-20i (one shot): is the loss in the grouping of the target or in choosing the groups?

  python scripts/budget.py --fold 0 --n 1000 --out results/budget_coco_f0
All variants are built at feature resolution (64 x 64) and scored exactly like INSID3's output (bilinear upsampling to 1024, threshold 0.5).
  insid3            the method as released (cross-checked against models/insid3.py on the first --check episodes)
  grid              every patch labelled by its ground-truth majority               -> ceiling of the 64 x 64 grid
  clus_major        INSID3's clusters, each labelled by its ground-truth majority   -> ceiling of the grouping
  clus_best         INSID3's clusters, the subset with the best IoU for this image
  seed_only         only the seed cluster
  oracle_seed       seed replaced by the cluster holding the most foreground, INSID3's aggregation rule unchanged
  drop_fp / add_fn  INSID3's choice minus wrongly included clusters / plus the missed ones
  thr_*             INSID3 with another aggregation threshold
Per-cluster evidence is saved to <out>.tables.pt for offline work on the selection rule.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--check", type=int, default=20); ap.add_argument("--tau", type=float, default=0.6)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(tau=a.tau); tf = model._transform; S = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n); M = Meter(); rows, tables = [], []; t0 = time.time(); tt = dict(feat=0.0, clus=0.0)
for e, (c, tn, rn) in enumerate(eps):
    if e < a.start: continue
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S, S), mode="nearest")[0, 0] > 0.5
    d = dissect(model, ri, rm, ti); h, K, lab, area = d["h"], d["K"], d["lab"], d["area"]
    if e < a.check:                                                           # same result as the released implementation?
        ref = model.predict_mask(ri, rm, ti); mine = up(d["pred"], S); assert bool((ref == mine).all()), (e, int((ref != mine).sum()))
    g = F.avg_pool2d(gt.float()[None, None], S // h)[0, 0]; oh = F.one_hot(lab.reshape(-1), K).float()
    pur = (oh.T @ g.reshape(-1)) / area; fgmass = pur * area; major = pur > 0.5
    V = {"insid3": d["pred"], "grid": g > 0.5, "clus_major": major[lab]}
    o = pur.argsort(descending=True); inter = fgmass[o].cumsum(0); iou = inter / (area[o].cumsum(0) + fgmass.sum() - inter); kb = int(iou.argmax())
    best = torch.zeros(K, dtype=torch.bool, device=DEV); best[o[:kb + 1]] = True; V["clus_best"] = best[lab]
    seed = d["seed"]; V["seed_only"] = (lab == seed) if seed >= 0 else d["pred"]
    s2 = int(fgmass.argmax()); aw = d["aw"].clone(); aw[s2] = 1.0; cross = (oh.T @ d["sim_fwd"].reshape(-1)) / area
    V["oracle_seed"] = ((cross * (d["Po"] @ d["Po"][s2]) * aw) > model.merge_threshold)[lab]
    V["drop_fp"] = d["pred"] & major[lab]; V["add_fn"] = d["pred"] | major[lab]
    for t in (0.1, 0.15, 0.3, 0.4): V[f"thr_{t}"] = (d["combined"] > t)[lab]
    for k, v in V.items(): M.add(k, c, *iu(up(v, S), gt))
    i_, u_ = iu(up(V["insid3"], S), gt)
    rows.append(dict(e=e, c=c, K=K, seed=seed, seed_pur=float(pur[seed]) if seed >= 0 else -1.0, fg=float(g.mean()), iou=i_ / max(u_, 1), empty=int(d["pred"].sum() == 0),
                     cand_prec=float(g[d["cand"]].mean()) if d["cand"].any() else -1.0, cand_rec=float((g * d["cand"]).sum() / g.sum().clamp(min=1e-6)),
                     n_sel=int((d["combined"] > model.merge_threshold).sum()), n_major=int(major.sum()), iou_major=(lambda x: x[0] / max(x[1], 1))(iu(up(V["clus_major"], S), gt))))
    mf = d["m"].reshape(-1); Sd = d["S"]; So = d["T"] @ d["R"].T; pool = lambda v: (oh.T @ v) / area
    nb = (~mf).any()
    tables.append(dict(e=e, c=c, K=K, seed=seed, area=area.cpu(), pur=pur.half().cpu(), cross=d["cross"].half().cpu(), intra=d["intra"].half().cpu(), aw=d["aw"].half().cpu(),
                       combined=d["combined"].half().cpu(), back=pool(d["back"].reshape(-1).float()).half().cpu(),
                       fgmax=pool(Sd[:, mf].max(1).values).half().cpu(), bgmax=(pool(Sd[:, ~mf].max(1).values) if nb else torch.zeros(K, device=DEV)).half().cpu(),
                       fgmax_o=pool(So[:, mf].max(1).values).half().cpu(), bgmax_o=(pool(So[:, ~mf].max(1).values) if nb else torch.zeros(K, device=DEV)).half().cpu(),
                       Pd=d["Pd"].half().cpu(), Po=d["Po"].half().cpu(), proto=d["proto"].half().cpu(), lab=lab.to(torch.int16).cpu(), g=(g * 255).round().to(torch.uint8).cpu(),
                       ref_fg=int(mf.sum())))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in ("insid3", "clus_major", "grid")), flush=True)
res = dict(fold=a.fold, n=len(rows), tau=a.tau, miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0),
           seed_is_fg=float(np.mean([r["seed_pur"] > 0.5 for r in rows])), empty=float(np.mean([r["empty"] for r in rows])), mean_K=float(np.mean([r["K"] for r in rows])),
           per_class={k: {str(c): v for c, v in M.d[k].items()} for k in M.names()})
print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
json.dump(dict(res, rows=rows), open(a.out + ".json", "w")); torch.save(tables, a.out + ".tables.pt")
