"""One GPU pass that saves, per episode, the target cluster tree and patch-level evidence maps, so that region-scoring rules
can be developed offline on CPU.  (COCO-20i, one shot, the episodes INSID3 draws)

  python scripts/leaves.py --fold 0 --n 300 --out results/leaves_f0_300
Saved per episode: tree (children, merge distances), soft ground truth per patch, INSID3's own prediction, and evidence maps on the
target grid. Notation: S = cosine similarity between target and reference patches; FG/BG = reference mask at patch level.
  nn_*      hard nearest neighbours: back = nearest reference patch is FG; fwd / bfwd = number of reference FG / BG patches whose nearest target patch is this one
  q{k}      soft version of back: share of softmax_j(k S) mass on reference FG
  mF{k}, mB{k}   reference FG / BG mass arriving under softmax_i(k S)   (reference -> target)
  dF{k}, dB{k}   the same under the dual softmax  softmax_j(k S) * softmax_i(k S)   (mutual)
  oF{k}, oB{k}   the same under entropic optimal transport with uniform marginals (Sinkhorn, 1 / epsilon = k)
A suffix _o means the features before INSID3's positional debiasing.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--kappa", default="30,100")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); KS = [float(k) for k in a.kappa.split(",")]
model = build_model(); tf = model._transform; S_ = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n); M = Meter(); out = []; t0 = time.time()

def sinkhorn(S, k, iters=50):
    K = torch.exp(k * (S - S.max())); nt, nr = S.shape; u = torch.ones(nt, device=S.device) / nt; v = torch.ones(nr, device=S.device) / nr
    for _ in range(iters):
        u = (1.0 / nt) / (K @ v + 1e-30); v = (1.0 / nr) / (K.T @ u + 1e-30)
    return u[:, None] * K * v[None]                                           # (Nt, Nr), rows sum to 1/Nt, columns to 1/Nr

for e, (c, tn, rn) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    imgs = torch.cat([ri, ti], 0).unsqueeze(0); f = F.normalize(model._extract_features(imgs).float(), p=2, dim=2); Tt = f[0, 1].reshape(f.shape[2], -1).T
    D = (1.0 - (Tt @ Tt.T).clamp(-1, 1)).cpu().numpy()
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)
    labels = torch.from_numpy(ac.labels_).long().to(DEV); d = dissect(model, ri, rm, ti, labels=labels); h = d["h"]; n = h * h
    g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1); mf = d["m"].reshape(-1); nb = bool((~mf).any())
    ev = {}
    for sfx, Sm in (("", d["S"]), ("_o", d["T"] @ d["R"].T)):
        nn_t = Sm.argmax(1); ev["nn_back" + sfx] = mf[nn_t].float(); nn_r = Sm.argmax(0)
        ev["nn_fwd" + sfx] = torch.bincount(nn_r[mf], minlength=n).float(); ev["nn_bfwd" + sfx] = torch.bincount(nn_r[~mf], minlength=n).float()
        for k in KS:
            Pj = torch.softmax(k * Sm, 1); Pi = torch.softmax(k * Sm, 0); Dm = Pj * Pi; ks = f"{int(k)}{sfx}"
            ev["q" + ks] = Pj[:, mf].sum(1); ev["mF" + ks] = Pi[:, mf].sum(1); ev["mB" + ks] = Pi[:, ~mf].sum(1); ev["dF" + ks] = Dm[:, mf].sum(1); ev["dB" + ks] = Dm[:, ~mf].sum(1)
            if sfx == "":
                O = sinkhorn(Sm, k) * n; ev["oF" + ks] = O[:, mf].sum(1); ev["oB" + ks] = O[:, ~mf].sum(1)
    ev["sim"] = d["sim_fwd"].reshape(-1); ev["fgmax"] = d["S"][:, mf].max(1).values; ev["bgmax"] = d["S"][:, ~mf].max(1).values if nb else torch.zeros(n, device=DEV)
    M.add("insid3", c, *iu(up(d["pred"], S_), gt))
    out.append(dict(e=e, c=c, children=ac.children_.astype(np.int16), dist=ac.distances_.astype(np.float16), g=g.half().cpu().numpy(), n_reffg=int(mf.sum()), n_refbg=int((~mf).sum()),
                    insid3=d["pred"].reshape(-1).cpu().numpy(), lab=labels.to(torch.int16).cpu().numpy(), seed=d["seed"], gt_area=float(gt.sum()),
                    ev={k: v.half().cpu().numpy() for k, v in ev.items()}))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s insid3 {M.miou('insid3'):.1f}", flush=True)
print(json.dumps(dict(fold=a.fold, n=len(out), insid3=round(M.miou("insid3"), 2), seconds=round(time.time() - t0))))
torch.save(out, a.out + ".leaves.pt")
