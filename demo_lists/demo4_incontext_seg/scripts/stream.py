"""Transductive in-context segmentation on a stream of unlabeled images of one concept.  (COCO-20i)

Per class: one labelled reference and N unlabeled images that contain the class. Round 1 segments every image from the
labelled reference alone. The round-trip score of image j is the IoU between the true reference mask and the mask obtained
on the reference image when (image j, its predicted mask) is used as the reference. Round r+1 segments image j from the
labelled reference plus the m other images with the highest round-trip scores (above --thr), using their round-r masks.
Controls: all = every other image above the threshold; oracle = the m other images with the best true IoU;
gt = labelled reference + m other images with their TRUE masks (what m extra annotations are worth).
  python scripts/stream.py --fold 0 --N 12 --m 4 --out results/stream_f0
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from utils.data import load_image, load_mask

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--N", type=int, default=12); ap.add_argument("--m", type=int, default=4)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--classes", type=int, default=20); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size
base = f"{CACHE}/data/COCO2014"; meta = pickle.load(open(f"{base}/splits/val/fold{a.fold}.pkl", "rb")); class_ids = [a.fold + 4 * v for v in range(20)][:a.classes]
rng = np.random.default_rng(a.seed); M = Meter(); t0 = time.time(); log = []


def rt(img_j, mask_j, ri0, rm0):
    """round-trip score of a pseudo reference"""
    if not mask_j.any(): return 0.0
    i, u = iu(model.predict_mask(img_j, mask_j, ri0), rm0[0] > 0.5); return i / max(u, 1.0)


for c in class_ids:
    names = [str(x) for x in rng.choice(meta[c], size=a.N + 1, replace=False)]
    ims, gts = [], []
    for nme in names:
        img, m = coco_load(base, nme, c); ims.append(load_image(img, tf, DEV)[0]); gts.append(load_mask(m, S_, DEV))
    ims = torch.cat(ims); gts = torch.cat(gts); G = gts > 0.5; ri0, rm0 = ims[:1], gts[:1]; J = range(1, a.N + 1)
    P = gts.clone(); score = np.ones(a.N + 1); true = np.ones(a.N + 1)
    for j in J:                                                               # round 1: labelled reference only
        p = model.predict_mask(ri0, rm0, ims[j:j + 1]); P[j] = p.to(gts.dtype); i, u = iu(p, G[j]); M.add("round1", c, i, u); true[j] = i / max(u, 1.0)
        score[j] = rt(ims[j:j + 1], P[j:j + 1], ri0, rm0)
    for j in J:                                                               # controls built on round-1 masks
        others = np.array([k for k in J if k != j])
        ks = [0] + [int(k) for k in others if score[k] > a.thr]; M.add("all>thr", c, *iu(model.predict_mask(ims[ks], P[ks], ims[j:j + 1]), G[j]))
        ks = [0] + [int(k) for k in others[np.argsort(-true[others])][:a.m]]; M.add("oracle_top", c, *iu(model.predict_mask(ims[ks], P[ks], ims[j:j + 1]), G[j]))
        ks = [0] + [int(k) for k in others[:a.m]]; M.add("gt_shots", c, *iu(model.predict_mask(ims[ks], gts[ks], ims[j:j + 1]), G[j]))
    corr1 = float(np.corrcoef(score[1:], true[1:])[0, 1])
    for r in range(2, a.rounds + 1):                                          # rounds 2..: top-m by round-trip score
        Pn = P.clone()
        for j in J:
            others = np.array([k for k in J if k != j]); top = others[np.argsort(-score[others])][:a.m]
            ks = [0] + [int(k) for k in top if score[k] > a.thr]
            p = model.predict_mask(ims[ks], P[ks], ims[j:j + 1]); Pn[j] = p.to(gts.dtype); M.add(f"round{r}", c, *iu(p, G[j]))
        P = Pn
        if r < a.rounds:
            for j in J: score[j] = rt(ims[j:j + 1], P[j:j + 1], ri0, rm0)
    print(f"class {c:2d} done {time.time() - t0:.0f}s | running mIoU: " + " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()) + f" | corr(round-trip, true) {corr1:.2f}", flush=True)
res = dict(fold=a.fold, N=a.N, m=a.m, thr=a.thr, miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0)); print(json.dumps(res)); json.dump(res, open(a.out + ".json", "w"))
