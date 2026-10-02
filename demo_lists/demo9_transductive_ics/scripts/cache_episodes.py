#!/usr/bin/env python3
"""GPU, ~1 s per episode: feature cache for standard COCO-20i episodes (the list INSID3's loader draws with seed 0).

Entry 2e = reference of episode e, entry 2e+1 = its query; masks are for the episode's class. Stored per entry: debiased
DINOv3 features as float16 coordinates in the non-positional subspace (dot products unchanged, 4.3 MB), INSID3 cluster
labels, cluster prototypes of the original features, masks, and the classes present in the image (to pick distractors).
Reference and query of an episode are encoded as one batch, as the released code does, so the cached one-shot prediction is
identical to the released INSID3 output; the first --check episodes verify this and print both mIoUs.
  python scripts/cache_episodes.py --fold 0 --n 400          -> $DEMO9_CACHE/episodes_f0_n400.pt (3.6 GB; delete after use)
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DEMO4, CACHE9, COCO_ANN
sys.path.insert(0, DEMO4)
from icx.common import *            # build_model, coco_episodes, coco_load, iu, DEV, torch, F, np, time   (adds the INSID3 checkout to sys.path)
from utils.data import load_image, load_mask, downsample_mask
from utils.clustering import agglomerative_clustering
from tics.imageset import ImageSet, cluster_protos

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=400); ap.add_argument("--check", type=int, default=100, help="episodes on which the cached predictor is compared with the released INSID3 code")
a = ap.parse_args(); os.makedirs(CACHE9, exist_ok=True); out = f"{CACHE9}/episodes_f{a.fold}_n{a.n}.pt"
model = build_model(); tf = model._transform; S_ = model.image_size; U = model.positional_basis.float()
Q = torch.linalg.svd(U, full_matrices=True)[0][:, U.shape[1]:].contiguous()            # basis of the complement of the positional subspace
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=1); t0 = time.time(); M = Meter(); agree = []; E = {k: [] for k in "fq lab Po gt64 gt_bits names cls present".split()}
for e, (c, tn, rns) in enumerate(eps):
    ims, gts, anns = [], [], []
    for nme in (rns[0], tn):
        img = Image.open(os.path.join(base, nme)).convert("RGB"); ann = torch.from_numpy(np.array(Image.open(os.path.join(COCO_ANN, nme[:-4] + ".png"))))
        ims.append(load_image(img, tf, DEV)[0]); gts.append(load_mask(ann == c + 1, S_, DEV)); anns.append(ann)
    # reference and query are encoded TOGETHER, exactly as the released predict_mask does: the encoder runs in bfloat16 and its
    # output depends slightly on the batch, which is enough to change INSID3's clustering (see HANDOFF.md, pitfalls)
    fpair = F.normalize(model._extract_features(torch.cat(ims).unsqueeze(0)).float(), p=2, dim=2)[0]
    for f, g, ann, nme in zip(fpair, gts, anns, (rns[0], tn)):
        C, h, w = f.shape; f = f.reshape(C, -1).T
        l = agglomerative_clustering(f, model.tau); fd = F.normalize(f - (f @ U) @ U.T, dim=-1)
        E["fq"].append((fd @ Q).half().cpu()); E["lab"].append(l.short().cpu()); E["Po"].append(cluster_protos(f, l, int(l.max()) + 1)[0].half().cpu())
        E["gt64"].append(downsample_mask(g.unsqueeze(1), h, w).cpu()); E["gt_bits"].append(np.packbits(g[0].cpu().numpy().reshape(-1)))
        E["names"].append(nme); E["cls"].append(c); E["present"].append(sorted(int(v) - 1 for v in ann.unique().tolist() if 0 < v <= 80))
    if e < a.check:                                                                  # cached predictor vs the released code on the same episode
        s = ImageSet(dict(c=c, names=E["names"][-2:], fq=torch.stack(E["fq"][-2:]), lab=torch.stack(E["lab"][-2:]), Po=E["Po"][-2:], gt64=torch.stack(E["gt64"][-2:]), gt_bits=np.stack(E["gt_bits"][-2:]), S=S_))
        pr = model.predict_mask(ims[0], gts[0], ims[1]); pc = s.up(s.predict(1, [0], [s.gt64[0]])); g = gts[1][0]
        M.add("released", c, *iu(pr, g)); M.add("cached", c, *iu(pc, g)); i, u = iu(pr, pc); agree.append(i / max(u, 1.0))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", flush=True)
if agree:
    print(f"check on {len(agree)} episodes: mIoU released {M.miou('released'):.2f} / cached {M.miou('cached'):.2f}; identical masks in {np.mean(np.array(agree) > 0.999):.0%}, mean IoU between the two {np.mean(agree):.3f}")
    print("  (expected: identical in 100%; reference and query are encoded as a pair like the released code)")
torch.save(dict(fold=a.fold, n=a.n, names=E["names"], cls=E["cls"], present=E["present"], fq=torch.stack(E["fq"]), lab=torch.stack(E["lab"]), Po=E["Po"],
                gt64=torch.stack(E["gt64"]), gt_bits=np.stack(E["gt_bits"]), S=S_), out)
print("saved", out, "| masks:", COCO_ANN)
