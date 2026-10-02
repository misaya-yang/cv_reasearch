"""Label-free filtering of pseudo references by a round trip to the labelled reference.  (COCO-20i, one labelled reference)

For every unlabeled image k: INSID3 predicts its mask p_k from the labelled reference, then (image k, p_k) is used as the
reference to predict the labelled reference image; the IoU of that prediction with the TRUE reference mask is the cycle
score (it needs no label of image k). Pseudo references are kept when the cycle score exceeds a threshold.
  python scripts/pseudo_shots2.py --fold 0 --n 150 --shots 5 --out results/pseudo5cyc_f0_150
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from utils.data import load_image, load_mask

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=150); ap.add_argument("--shots", type=int, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=a.shots); M = Meter(); t0 = time.time(); rec = []
THR = [0.3, 0.5, 0.7]
for e, (c, tn, rns) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); ti, _ = load_image(timg, tf, DEV); refs = [coco_load(base, r, c) for r in rns]
    ri = torch.cat([load_image(im, tf, DEV)[0] for im, _ in refs]); rm = torch.cat([load_mask(m, S_, DEV) for _, m in refs])
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    M.add("1shot", c, *iu(model.predict_mask(ri[:1], rm[:1], ti), gt))
    M.add("Kshot_gt", c, *iu(model.predict_mask(ri, rm, ti), gt))
    pm = rm.clone(); true_iou, cyc = [1.0], [1.0]
    for k in range(1, len(rns)):
        p = model.predict_mask(ri[:1], rm[:1], ri[k:k + 1]); i, u = iu(p, rm[k] > 0.5); true_iou.append(i / max(u, 1.0)); pm[k] = p.to(rm.dtype)
        if p.any():
            b = model.predict_mask(ri[k:k + 1], pm[k:k + 1], ri[:1]); i, u = iu(b, rm[0] > 0.5); cyc.append(i / max(u, 1.0))
        else: cyc.append(0.0)
    true_iou, cyc = np.array(true_iou), np.array(cyc); rec.append((true_iou[1:], cyc[1:]))
    sets = {"pseudo_all": [k for k in range(len(rns)) if k == 0 or pm[k].any()], "pseudo_orc": list(np.where(true_iou > 0.5)[0])}
    for t in THR: sets[f"pseudo_cyc{t}"] = list(np.where(cyc > t)[0])
    sets["pseudo_cyc_top2"] = [0] + [int(k) for k in np.argsort(-cyc[1:])[:2] + 1 if cyc[k] > 0.3]
    for name, ks in sets.items():
        M.add(name, c, *iu(model.predict_mask(ri[ks], pm[ks], ti), gt))
    if (e + 1) % 25 == 0:
        T = np.concatenate([r[0] for r in rec]); Cy = np.concatenate([r[1] for r in rec])
        print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()),
              f"| corr(cycle, true IoU) {np.corrcoef(T, Cy)[0, 1]:.2f}; precision of cyc>0.5 at true>0.5: {np.mean(T[Cy > 0.5] > 0.5):.2f} (kept {np.mean(Cy > 0.5):.2f}, base rate {np.mean(T > 0.5):.2f})", flush=True)
res = dict(fold=a.fold, n=a.n, shots=a.shots, miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0)); print(json.dumps(res)); json.dump(res, open(a.out + ".json", "w"))
np.savez(a.out + ".npz", true=np.stack([r[0] for r in rec]), cyc=np.stack([r[1] for r in rec]))
