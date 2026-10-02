"""Can the correspondence evidence itself be made cleaner?  Variants of the similarity used for the bidirectional F1 rule.

  deb            INSID3's debiased features (the evidence used so far)
  deb{k}         the same with k positional components removed instead of 500
  cen            deb + subtract each image's own mean feature, renormalise
  jcen           deb + subtract the mean over both images
  wht{l}         deb + whitening with the covariance of both images' patches (shrinkage l)
  +flip          reference foreground/background patches from the mirrored reference added
  +zoom          reference patches from a crop around the reference mask (25% margin, resized to full input) added
  knn5           back = majority of the 5 nearest reference patches (deb)
Saved: per-leaf back / fwd / bfwd for every variant; the target tree is INSID3's.

  python scripts/ev2.py --fold 0 --n 300 --out results/ev2_f0_300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from utils.data import load_image, load_mask, downsample_mask
from sklearn.cluster import AgglomerativeClustering
import torchvision.transforms.functional as TF

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(svd=700); tf = model._transform; S_ = model.image_size; U = model.positional_basis                  # 700 components; prefixes give smaller bases
proj = lambda X, k=500: F.normalize(X - (X @ U[:, :k]) @ U[:, :k].T, dim=1)

def enc(imgs):
    f = F.normalize(model._extract_features(imgs.unsqueeze(0)).float(), p=2, dim=2)[0]; C = f.shape[1]; return [x.reshape(C, -1).T for x in f]
def whiten(Xs, lam):
    X = torch.cat(Xs); mu = X.mean(0, keepdim=True); Xc = X - mu; cov = Xc.T @ Xc / len(X); ev, V = torch.linalg.eigh(cov); W = V @ torch.diag((ev.clamp(min=0) + lam * ev.mean()).rsqrt()) @ V.T
    return [F.normalize((x - mu) @ W, dim=1) for x in Xs]

eps, class_ids, base = coco_episodes(a.fold, a.n); out = []; t0 = time.time()
for e, (c, tn, rn) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    # extra reference views: mirror, and a crop around the mask
    ys, xs = torch.where(rm[0]); y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1; my, mx = int(0.25 * (y1 - y0)) + 8, int(0.25 * (x1 - x0)) + 8
    y0, y1, x0, x1 = max(0, y0 - my), min(S_, y1 + my), max(0, x0 - mx), min(S_, x1 + mx)
    rz = F.interpolate(ri[:, :, y0:y1, x0:x1], size=(S_, S_), mode="bilinear", align_corners=False); mz = F.interpolate(rm[None, :, y0:y1, x0:x1].float(), size=(S_, S_), mode="nearest")[0] > 0.5
    R, Tt, Rf, Rz = enc(torch.cat([ri, ti, ri.flip(-1), rz], 0)); h = int(math.sqrt(len(Tt))); n = h * h
    D = (1.0 - (Tt @ Tt.T).clamp(-1, 1)).cpu().numpy(); ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)
    g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1); dm = lambda m: downsample_mask(m.unsqueeze(1), h, h).reshape(-1)
    mf = dm(rm); mff = dm(rm.flip(-1)); mfz = dm(mz); ev = {}
    def maps(name, Tx, refs, k=1):
        """refs: list of (features, fg mask). Nearest neighbours over the union of the reference views."""
        Rx = torch.cat([r for r, _ in refs]); m = torch.cat([q for _, q in refs]); Sm = Tx @ Rx.T
        if k == 1: back = m[Sm.argmax(1)]
        else: back = m[Sm.topk(k, dim=1).indices].float().mean(1) > 0.5
        nr = Sm.argmax(0); ev[name + "/back"] = back.cpu().numpy(); ev[name + "/fwd"] = torch.bincount(nr[m], minlength=n).to(torch.int16).cpu().numpy(); ev[name + "/bfwd"] = torch.bincount(nr[~m], minlength=n).to(torch.int16).cpu().numpy()
        ev[name + "/nfg"] = int(m.sum())
    Td, Rd, Rfd, Rzd = (proj(x) for x in (Tt, R, Rf, Rz))
    maps("deb", Td, [(Rd, mf)]); maps("knn5", Td, [(Rd, mf)], k=5)
    for k in (100, 300, 700): maps(f"deb{k}", proj(Tt, k), [(proj(R, k), mf)])
    maps("cen", F.normalize(Td - Td.mean(0, keepdim=True), dim=1), [(F.normalize(Rd - Rd.mean(0, keepdim=True), dim=1), mf)])
    jm = torch.cat([Td, Rd]).mean(0, keepdim=True); maps("jcen", F.normalize(Td - jm, dim=1), [(F.normalize(Rd - jm, dim=1), mf)])
    for lam in (0.1, 1.0):
        Tw, Rw = whiten([Td, Rd], lam); maps(f"wht{lam}", Tw, [(Rw, mf)])
    maps("deb+flip", Td, [(Rd, mf), (Rfd, mff)]); maps("deb+zoom", Td, [(Rd, mf), (Rzd, mfz)]); maps("deb+flip+zoom", Td, [(Rd, mf), (Rfd, mff), (Rzd, mfz)]); maps("zoom only", Td, [(Rzd, mfz)])
    out.append(dict(e=e, c=c, children=ac.children_.astype(np.int16), g=g.half().cpu().numpy(), ev=ev))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", flush=True)
torch.save(out, a.out + ".ev2.pt"); print("done", round(time.time() - t0))
