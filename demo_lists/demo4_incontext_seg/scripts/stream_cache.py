"""Cache what the stream experiments need: per class one labelled reference and N unlabeled images of that class, with their
L2-normalised DINOv3 features (float16), INSID3 cluster labels, and masks. Also checks icx/fast.py against the released code.
  python scripts/stream_cache.py --fold 0 --N 16 --seed 0
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.fast import ClassSet
from utils.data import load_image, load_mask, downsample_mask
from utils.clustering import agglomerative_clustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--N", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--classes", type=int, default=20)
a = ap.parse_args(); out = f"{CACHE}/stream/f{a.fold}_s{a.seed}_N{a.N}"; os.makedirs(out, exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size; torch.save(model.positional_basis.float().cpu(), f"{out}/U.pt")
base = f"{CACHE}/data/COCO2014"; meta = pickle.load(open(f"{base}/splits/val/fold{a.fold}.pkl", "rb")); class_ids = [a.fold + 4 * v for v in range(20)][:a.classes]
rng = np.random.default_rng(a.seed); t0 = time.time()
for ci, c in enumerate(class_ids):
    names = [str(x) for x in rng.choice(meta[c], size=min(a.N + 1, len(meta[c])), replace=False)]
    fn, lab, g64, gb, ims, gts = [], [], [], [], [], []
    for nme in names:
        img, m = coco_load(base, nme, c); x = load_image(img, tf, DEV)[0]; g = load_mask(m, S_, DEV)
        f = F.normalize(model._extract_features(x.unsqueeze(0)).float(), p=2, dim=2)[0, 0]; C, h, w = f.shape; f = f.reshape(C, -1).T
        fn.append(f.half().cpu()); lab.append(agglomerative_clustering(f, model.tau).short().cpu())
        g64.append(downsample_mask(g.unsqueeze(1), h, w).cpu()); gb.append(np.packbits(g[0].cpu().numpy().reshape(-1)))
        if ci == 0 and len(ims) < 4: ims.append(x); gts.append(g)
    torch.save(dict(c=c, names=names, fn=torch.stack(fn), lab=torch.stack(lab), gt64=torch.stack(g64), gt_bits=np.stack(gb), S=S_), f"{out}/class{c}.pt")
    print(f"class {c}: {len(names)} images, {time.time() - t0:.0f}s", flush=True)
    if ci == 0:                                                               # exactness check on the first class
        cs = ClassSet(f"{out}/class{c}.pt", model.positional_basis.float()); ims = torch.cat(ims); gts = torch.cat(gts)
        for refs, t in (([0], 1), ([0], 2), ([0, 1], 3), ([0, 1, 2], 3)):
            p_ref = model.predict_mask(ims[refs], gts[refs], ims[t:t + 1]); p_fast = cs.up(cs.predict(t, refs, [cs.gt64[j] for j in refs]))
            i, u = iu(p_ref, p_fast); print(f"  check refs {refs} -> target {t}: IoU(released, cached) = {i / max(u, 1):.4f}; released IoU {iu(p_ref, gts[t] > 0.5)[0] / max(iu(p_ref, gts[t] > 0.5)[1], 1):.3f}", flush=True)
print("saved", out)
