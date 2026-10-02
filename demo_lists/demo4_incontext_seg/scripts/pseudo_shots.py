"""Can unlabeled images of the same concept stand in for extra labelled references?  (COCO-20i, one labelled reference)

Each episode has one labelled reference and K-1 further images known to contain the class (the other references of the
K-shot episode list, masks hidden). INSID3 first segments those images from the labelled reference; its predictions are
then used as if they were reference masks.
  1shot        the labelled reference only
  Kshot_gt     all K references with their true masks (upper bound: what real extra labels are worth)
  pseudo       labelled reference + all predicted masks
  pseudo_orc   labelled reference + predicted masks whose IoU with the hidden truth exceeds 0.5 (upper bound of any filter)
  python scripts/pseudo_shots.py --fold 0 --n 150 --shots 5 --out results/pseudo5_f0_150
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from utils.data import load_image, load_mask

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=150); ap.add_argument("--shots", type=int, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=a.shots); M = Meter(); t0 = time.time(); pious = []
for e, (c, tn, rns) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); ti, _ = load_image(timg, tf, DEV); refs = [coco_load(base, r, c) for r in rns]
    ri = torch.cat([load_image(im, tf, DEV)[0] for im, _ in refs]); rm = torch.cat([load_mask(m, S_, DEV) for _, m in refs])
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    M.add("1shot", c, *iu(model.predict_mask(ri[:1], rm[:1], ti), gt))
    M.add("Kshot_gt", c, *iu(model.predict_mask(ri, rm, ti), gt))
    keep, keep_orc = [0], [0]; pm = rm.clone()
    for k in range(1, len(rns)):
        p = model.predict_mask(ri[:1], rm[:1], ri[k:k + 1]); g = rm[k] > 0.5
        i, u = iu(p, g); pious.append(i / max(u, 1.0)); pm[k] = p.to(rm.dtype)
        if p.any(): keep.append(k)
        if i / max(u, 1.0) > 0.5: keep_orc.append(k)
    M.add("pseudo", c, *iu(model.predict_mask(ri[keep], pm[keep], ti), gt))
    M.add("pseudo_orc", c, *iu(model.predict_mask(ri[keep_orc], pm[keep_orc], ti), gt))
    if (e + 1) % 25 == 0: print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()), f"| pseudo-mask IoU mean {np.mean(pious):.2f}, >0.5: {np.mean(np.array(pious) > 0.5):.2f}", flush=True)
res = dict(fold=a.fold, n=a.n, shots=a.shots, miou={k: round(M.miou(k), 2) for k in M.names()}, pseudo_iou_mean=float(np.mean(pious)), seconds=round(time.time() - t0)); print(json.dumps(res)); json.dump(res, open(a.out + ".json", "w"))
