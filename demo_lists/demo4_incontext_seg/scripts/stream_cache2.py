"""Compact cache for the stream experiments (about half the size of stream_cache.py's).
Per class: image 0 = labelled reference, images 1..N = unlabeled images of the class, then D distractors (images of other
classes of the fold that do not contain the class). Stored: debiased features as coordinates in an orthonormal basis of the
non-positional subspace (float16; dot products are unchanged), INSID3 cluster labels, cluster prototypes of the original
features, masks.   python scripts/stream_cache2.py --fold 0 --N 48 --D 16 --seed 0
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.fast import ClassSet, cluster_protos
from utils.data import load_image, load_mask, downsample_mask
from utils.clustering import agglomerative_clustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--N", type=int, default=48); ap.add_argument("--D", type=int, default=16); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--classes", type=int, default=20)
a = ap.parse_args(); out = f"{CACHE}/stream/f{a.fold}_s{a.seed}_N{a.N}_D{a.D}"; os.makedirs(out, exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size; U = model.positional_basis.float()
Q = torch.linalg.svd(U, full_matrices=True)[0][:, U.shape[1]:].contiguous()            # (C, C - svd) basis of the complement
torch.save(U.cpu(), f"{out}/U.pt")
base = f"{CACHE}/data/COCO2014"; meta = pickle.load(open(f"{base}/splits/val/fold{a.fold}.pkl", "rb")); class_ids = [a.fold + 4 * v for v in range(20)][:a.classes]
rng = np.random.default_rng(a.seed); t0 = time.time()
for ci, c in enumerate(class_ids):
    names = [str(x) for x in rng.choice(meta[c], size=min(a.N + 1, len(meta[c])), replace=False)]; n_real = len(names)
    others = [str(x) for c2 in class_ids if c2 != c for x in rng.choice(meta[c2], size=4, replace=False)]; rng.shuffle(others)
    for nme in others:
        if len(names) - n_real >= a.D: break
        if not bool(coco_load(base, nme, c)[1].any()) and nme not in names: names.append(nme)
    fq, lab, Po, g64, gb, ims, gts = [], [], [], [], [], [], []
    for nme in names:
        img, m = coco_load(base, nme, c); x = load_image(img, tf, DEV)[0]; g = load_mask(m, S_, DEV)
        f = F.normalize(model._extract_features(x.unsqueeze(0)).float(), p=2, dim=2)[0, 0]; C, h, w = f.shape; f = f.reshape(C, -1).T
        l = agglomerative_clustering(f, model.tau); fd = F.normalize(f - (f @ U) @ U.T, dim=-1)
        fq.append((fd @ Q).half().cpu()); lab.append(l.short().cpu()); Po.append(cluster_protos(f, l, int(l.max()) + 1)[0].half().cpu())
        g64.append((downsample_mask(g.unsqueeze(1), h, w) if g.any() else torch.zeros(h, w, dtype=torch.bool, device=DEV)).cpu()); gb.append(np.packbits(g[0].cpu().numpy().reshape(-1)))
        if ci == 0 and len(ims) < 4: ims.append(x); gts.append(g)
    torch.save(dict(c=c, names=names, n_real=n_real, fq=torch.stack(fq), lab=torch.stack(lab), Po=Po, gt64=torch.stack(g64), gt_bits=np.stack(gb), S=S_), f"{out}/class{c}.pt")
    print(f"class {c}: {n_real} images + {len(names) - n_real} distractors, {time.time() - t0:.0f}s", flush=True)
    if ci == 0:
        cs = ClassSet(f"{out}/class{c}.pt", U); ims = torch.cat(ims); gts = torch.cat(gts)
        for refs, t in (([0], 1), ([0], 2), ([0, 1], 3), ([0, 1, 2], 3)):
            p_ref = model.predict_mask(ims[refs], gts[refs], ims[t:t + 1]); p_fast = cs.up(cs.predict(t, refs, [cs.gt64[j] for j in refs]))
            i, u = iu(p_ref, p_fast); print(f"  check refs {refs} -> target {t}: IoU(released, cached) = {i / max(u, 1):.4f}", flush=True)
print("saved", out)
