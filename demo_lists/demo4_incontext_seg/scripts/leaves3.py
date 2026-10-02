"""GPU pass, third version: the reference tree as a set of labelled regions.

Every node of the reference tree is a region whose IoU with the reference mask is known. Every node of the target tree gets an IoU
estimate by nearest-neighbour / kernel regression from the reference nodes, on pooled (debiased, L2-normalised) region features.
  tr1 / tr5      IoU of the most similar reference node / similarity-weighted mean over the 5 most similar
  trs            similarity to the most similar reference node
  krr{k}         kernel ridge regression of IoU on reference nodes, kernel exp(k (cos - 1))
Also saved: both trees, nearest-neighbour patch indices in both directions (as in leaves2.py).

  python scripts/leaves3.py --fold 0 --n 300 --out results/l3_f0_300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--min-area", type=int, default=1)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n); M = Meter(); out = []; t0 = time.time()

def tree(X):
    D = (1.0 - (X @ X.T).clamp(-1, 1)).cpu().numpy()
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)

def node_sums(children, leaf):
    """leaf (n, D) numpy -> (2n-1, D) subtree sums."""
    n = leaf.shape[0]; X = np.concatenate([leaf, np.zeros((n - 1, leaf.shape[1]), np.float32)])
    for i, (x, y) in enumerate(children): X[n + i] = X[x] + X[y]
    return X

for e, (c, tn, rn) in enumerate(eps):
    if e < a.start: continue
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    imgs = torch.cat([ri, ti], 0).unsqueeze(0); f = F.normalize(model._extract_features(imgs).float(), p=2, dim=2); C = f.shape[2]
    Rt, Tt = f[0, 0].reshape(C, -1).T, f[0, 1].reshape(C, -1).T
    ac = tree(Tt); acr = tree(Rt); labels = torch.from_numpy(ac.labels_).long().to(DEV); d = dissect(model, ri, rm, ti, labels=labels); h = d["h"]; n = h * h
    g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1); mf = d["m"].reshape(-1); rg = F.avg_pool2d(rm.float()[None], S_ // h)[0].reshape(-1)
    Sd = d["S"]; rec = dict(e=e, c=c, children=ac.children_.astype(np.int16), dist=ac.distances_.astype(np.float16), g=g.half().cpu().numpy(),
        r_children=acr.children_.astype(np.int16), r_dist=acr.distances_.astype(np.float16), r_mask=mf.cpu().numpy(), r_g=rg.half().cpu().numpy(),
        insid3=d["pred"].reshape(-1).cpu().numpy(), lab=labels.to(torch.int16).cpu().numpy(), seed=d["seed"], sim=d["sim_fwd"].reshape(-1).half().cpu().numpy())
    v, i = Sd.max(1); rec["nn_t"] = i.to(torch.int16).cpu().numpy(); rec["nn_t_s"] = v.half().cpu().numpy(); v, i = Sd.max(0); rec["nn_r"] = i.to(torch.int16).cpu().numpy(); rec["nn_r_s"] = v.half().cpu().numpy()
    # soft evidence maps (temperature 30) and pooled-feature statistics, kept for learned / combined scoring
    Pj = torch.softmax(30 * Sd, 1); Pi = torch.softmax(30 * Sd, 0); Dm = Pj * Pi
    rec["ev"] = {k: v.half().cpu().numpy() for k, v in dict(q30=Pj[:, mf].sum(1), mF30=Pi[:, mf].sum(1), mB30=Pi[:, ~mf].sum(1), dF30=Dm[:, mf].sum(1), dB30=Dm[:, ~mf].sum(1),
                 fgmax=Sd[:, mf].max(1).values, bgmax=(Sd[:, ~mf].max(1).values if (~mf).any() else torch.zeros(n, device=DEV))).items()}
    # labelled reference regions -> IoU estimate for target regions
    rgn = rg.cpu().numpy(); RS = node_sums(acr.children_, np.concatenate([d["Rd"].cpu().numpy(), np.ones((n, 1), np.float32), rgn[:, None]], 1)); r_area, r_fg = RS[:, -2], RS[:, -1]
    r_iou = torch.from_numpy(r_fg / (r_area + rgn.sum() - r_fg + 1e-9)).to(DEV)
    TS = node_sums(ac.children_, np.concatenate([d["Td"].cpu().numpy(), np.ones((n, 1), np.float32)], 1))
    t_area = TS[:, -1]; rec["coh_d"] = (np.linalg.norm(TS[:, :C], axis=1) / t_area).astype(np.float16); pr_ = d["proto"].cpu().numpy()
    rec["cos_d"] = ((TS[:, :C] @ pr_) / (np.linalg.norm(TS[:, :C], axis=1) + 1e-9)).astype(np.float16)
    PR = F.normalize(torch.from_numpy(RS[:, :C]).to(DEV), dim=1); PT = F.normalize(torch.from_numpy(TS[:, :C]).to(DEV), dim=1); keep = torch.from_numpy(r_area >= a.min_area).to(DEV)
    Cn = PT @ PR.T; Cn[:, ~keep] = -1                                             # (target nodes, reference nodes)
    v5, i5 = Cn.topk(5, dim=1); w5 = torch.softmax(50 * v5, 1); rec["tr1"] = r_iou[i5[:, 0]].half().cpu().numpy(); rec["tr5"] = (w5 * r_iou[i5]).sum(1).half().cpu().numpy(); rec["trs"] = v5[:, 0].half().cpu().numpy()
    rec["tr1_area"] = (torch.from_numpy(r_area).to(DEV)[i5[:, 0]] / n).half().cpu().numpy(); rec["tr1_idx"] = i5[:, 0].to(torch.int16).cpu().numpy()
    Krr = PR @ PR.T
    for k in (10.0, 30.0):
        alpha = torch.linalg.solve(torch.exp(k * (Krr - 1)) + 1e-2 * torch.eye(Krr.shape[0], device=DEV), r_iou.float()); rec[f"krr{int(k)}"] = (torch.exp(k * (Cn.clamp(min=-1) - 1)) @ alpha).half().cpu().numpy()
    rec["r_iou_max"] = float(r_iou.max())
    M.add("insid3", c, *iu(up(d["pred"], S_), gt)); out.append(rec)
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s insid3 {M.miou('insid3'):.1f}", flush=True)
print(json.dumps(dict(fold=a.fold, n=len(out), insid3=round(M.miou("insid3"), 2), seconds=round(time.time() - t0))))
torch.save(out, a.out + ".l3.pt")
