"""GPU pass, second version: both cluster trees (target and reference), nearest-neighbour indices in both directions, and
distribution distances between the reference foreground and every target tree node.  (COCO-20i, one shot)

  python scripts/leaves2.py --fold 0 --n 300 --out results/l2_f0_300
With the saved indices every hard-nearest-neighbour statistic of any target region, in either direction and on either tree, can be
recomputed offline. Per target node: MMD^2 between its patches and the reference foreground patches, for a linear kernel (debiased
and original features) and Gaussian kernels exp(k (x.y - 1)) approximated with random Fourier features.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=0)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n); M = Meter(); out = []; t0 = time.time()
gen = torch.Generator(device="cpu").manual_seed(0); Dr = 2048; KS = (10.0, 30.0)
W = {k: (torch.randn(Dr, 1024, generator=gen) * math.sqrt(k)).to(DEV) for k in KS}; B = (torch.rand(Dr, generator=gen) * 2 * math.pi).to(DEV)
rff = lambda X, k: math.sqrt(2.0 / Dr) * torch.cos(X @ W[k].T + B)

def tree(X):
    D = (1.0 - (X @ X.T).clamp(-1, 1)).cpu().numpy()
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)

for e, (c, tn, rn) in enumerate(eps):
    if e < a.start: continue
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    imgs = torch.cat([ri, ti], 0).unsqueeze(0); f = F.normalize(model._extract_features(imgs).float(), p=2, dim=2); C = f.shape[2]
    Rt, Tt = f[0, 0].reshape(C, -1).T, f[0, 1].reshape(C, -1).T
    ac = tree(Tt); acr = tree(Rt); labels = torch.from_numpy(ac.labels_).long().to(DEV); d = dissect(model, ri, rm, ti, labels=labels); h = d["h"]; n = h * h
    g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1); mf = d["m"].reshape(-1)
    rg = F.avg_pool2d(rm.float()[None], S_ // h)[0].reshape(-1)                                          # soft reference mask per patch
    Sd = d["S"]; So = Tt @ Rt.T; rec = dict(e=e, c=c, children=ac.children_.astype(np.int16), dist=ac.distances_.astype(np.float16), g=g.half().cpu().numpy(),
        r_children=acr.children_.astype(np.int16), r_dist=acr.distances_.astype(np.float16), r_mask=mf.cpu().numpy(), r_g=rg.half().cpu().numpy(),
        insid3=d["pred"].reshape(-1).cpu().numpy(), lab=labels.to(torch.int16).cpu().numpy(), seed=d["seed"],
        sim=d["sim_fwd"].reshape(-1).half().cpu().numpy())
    for sfx, Sm in (("", Sd), ("_o", So)):
        v, i = Sm.max(1); rec["nn_t" + sfx] = i.to(torch.int16).cpu().numpy(); rec["nn_t_s" + sfx] = v.half().cpu().numpy()
        v, i = Sm.max(0); rec["nn_r" + sfx] = i.to(torch.int16).cpu().numpy(); rec["nn_r_s" + sfx] = v.half().cpu().numpy()
    # distribution distances: mean embeddings of the reference foreground and of every target node
    feats = torch.cat([d["Td"], Tt, rff(d["Td"], KS[0]), rff(d["Td"], KS[1])], 1).cpu().numpy()          # (n, 1024 + 1024 + 2048 + 2048)
    reff = torch.cat([d["Rd"][mf].mean(0), Rt[mf].mean(0), rff(d["Rd"][mf], KS[0]).mean(0), rff(d["Rd"][mf], KS[1]).mean(0)]).cpu().numpy()
    X = np.concatenate([feats, np.zeros((n - 1, feats.shape[1]), np.float32)]); area = np.concatenate([np.ones(n), np.zeros(n - 1)])
    for i, (x, y) in enumerate(ac.children_): X[n + i] = X[x] + X[y]; area[n + i] = area[x] + area[y]
    Mu = X / area[:, None]; sl = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 6144)]
    for name, (lo, hi) in zip(("mmd_lin_d", "mmd_lin_o", "mmd_rff10", "mmd_rff30"), sl): rec[name] = ((Mu[:, lo:hi] - reff[None, lo:hi]) ** 2).sum(1).astype(np.float16)
    rec["coh_d"] = np.linalg.norm(Mu[:, :1024], axis=1).astype(np.float16); rec["cos_d"] = ((Mu[:, :1024] @ reff[:1024]) / (np.linalg.norm(Mu[:, :1024], axis=1) * np.linalg.norm(reff[:1024]) + 1e-9)).astype(np.float16)
    rec["ref_coh_d"] = float(np.linalg.norm(reff[:1024])); rec["ref_coh_o"] = float(np.linalg.norm(reff[1024:2048]))
    M.add("insid3", c, *iu(up(d["pred"], S_), gt)); out.append(rec)
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s insid3 {M.miou('insid3'):.1f}", flush=True)
print(json.dumps(dict(fold=a.fold, n=len(out), insid3=round(M.miou("insid3"), 2), seconds=round(time.time() - t0))))
torch.save(out, a.out + ".l2.pt")
