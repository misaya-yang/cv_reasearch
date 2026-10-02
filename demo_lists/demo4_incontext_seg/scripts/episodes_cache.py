"""Compact feature cache for the first n STANDARD COCO-20i episodes of a fold (the list INSID3's loader draws with seed 0).
Entry 2e = reference of episode e, entry 2e+1 = its query; masks are for the episode's class.
  python scripts/episodes_cache.py --fold 0 --n 400
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.fast import cluster_protos
from utils.data import load_image, load_mask, downsample_mask
from utils.clustering import agglomerative_clustering
ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=400)
a = ap.parse_args(); out = f"{CACHE}/stream/episodes_f{a.fold}_n{a.n}.pt"
model = build_model(); tf = model._transform; S_ = model.image_size; U = model.positional_basis.float()
Q = torch.linalg.svd(U, full_matrices=True)[0][:, U.shape[1]:].contiguous()
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=1); t0 = time.time(); fq, lab, Po, g64, gb, names, cls = [], [], [], [], [], [], []
for e, (c, tn, rns) in enumerate(eps):
    for nme in (rns[0], tn):
        img, m = coco_load(base, nme, c); x = load_image(img, tf, DEV)[0]; g = load_mask(m, S_, DEV)
        f = F.normalize(model._extract_features(x.unsqueeze(0)).float(), p=2, dim=2)[0, 0]; C, h, w = f.shape; f = f.reshape(C, -1).T
        l = agglomerative_clustering(f, model.tau); fd = F.normalize(f - (f @ U) @ U.T, dim=-1)
        fq.append((fd @ Q).half().cpu()); lab.append(l.short().cpu()); Po.append(cluster_protos(f, l, int(l.max()) + 1)[0].half().cpu())
        g64.append(downsample_mask(g.unsqueeze(1), h, w).cpu()); gb.append(np.packbits(g[0].cpu().numpy().reshape(-1))); names.append(nme); cls.append(c)
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", flush=True)
torch.save(dict(fold=a.fold, n=a.n, names=names, cls=cls, fq=torch.stack(fq), lab=torch.stack(lab), Po=Po, gt64=torch.stack(g64), gt_bits=np.stack(gb), S=S_), out)
print("saved", out)
