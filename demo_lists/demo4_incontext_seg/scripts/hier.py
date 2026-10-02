"""Does the target's cluster tree contain the whole object as one node?  (COCO-20i, one shot)

INSID3 cuts the average-linkage tree of the target patches at one fixed level and then picks clusters. Here the full tree is kept:
  anc_best     the ancestor of INSID3's seed cluster (the seed itself included) with the best IoU      -> ceiling of "choose the level"
  node_best    the single tree node with the best IoU                                                  -> ceiling of "choose one node"
  anc_cand     the ancestor that best overlaps INSID3's own candidate mask (no ground truth used)
  cand         the candidate mask itself
  node_cos     the tree node whose pooled (debiased) feature is closest to the pooled reference foreground (no ground truth used)
Per-node evidence for every tree node is saved to <out>.nodes.pt for offline work on the node-scoring rule.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S = model.image_size
eps, class_ids, base = coco_episodes(a.fold, a.n); M = Meter(); tabs = []; t0 = time.time()
for e, (c, tn, rn) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S, S), mode="nearest")[0, 0] > 0.5
    # the same clustering call as INSID3, keeping the tree
    imgs = torch.cat([ri, ti], 0).unsqueeze(0); f = F.normalize(model._extract_features(imgs).float(), p=2, dim=2); Tt = f[0, 1].reshape(f.shape[2], -1).T
    D = (1.0 - (Tt @ Tt.T).clamp(-1, 1)).cpu().numpy()
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)
    labels = torch.from_numpy(ac.labels_).long().to(DEV); d = dissect(model, ri, rm, ti, labels=labels); h, n = d["h"], d["h"] * d["w"]
    g = F.avg_pool2d(gt.float()[None, None], S // h)[0, 0].reshape(-1).cpu().numpy(); G = g.sum()
    mf = d["m"].reshape(-1); fwdnn = d["S"][:, mf].argmax(0).cpu().numpy() if mf.any() else np.zeros(0, int)       # nearest target patch of every reference foreground patch
    bwdnn = d["S"][:, ~mf].argmax(0).cpu().numpy() if (~mf).any() else np.zeros(0, int)                             # nearest target patch of every reference background patch
    leaf = dict(area=np.ones(n), fg=g, cand=d["cand"].reshape(-1).float().cpu().numpy(), back=d["back"].reshape(-1).float().cpu().numpy(),
                sim=d["sim_fwd"].reshape(-1).cpu().numpy(), fwd=np.bincount(fwdnn, minlength=n).astype(float), bfwd=np.bincount(bwdnn, minlength=n).astype(float),
                fgmax=d["S"][:, mf].max(1).values.cpu().numpy(), bgmax=(d["S"][:, ~mf].max(1).values.cpu().numpy() if (~mf).any() else np.zeros(n)))
    Fd = np.concatenate([d["Td"].cpu().numpy(), np.zeros((n - 1, d["Td"].shape[1]), np.float32)]); Fo = np.concatenate([d["T"].cpu().numpy(), np.zeros((n - 1, d["T"].shape[1]), np.float32)])
    ch = ac.children_; N = 2 * n - 1; parent = np.full(N, -1); X = {k: np.concatenate([v, np.zeros(n - 1)]) for k, v in leaf.items()}
    for i, (x, y) in enumerate(ch):
        parent[x] = parent[y] = n + i
        for k in X: X[k][n + i] = X[k][x] + X[k][y]
        Fd[n + i] = Fd[x] + Fd[y]; Fo[n + i] = Fo[x] + Fo[y]
    pr = d["proto"].cpu().numpy(); nd = np.linalg.norm(Fd, axis=1) + 1e-9; cosd = (Fd @ pr) / nd; coh_d = nd / X["area"]; coh_o = np.linalg.norm(Fo, axis=1) / X["area"]
    dist = np.concatenate([np.zeros(n), ac.distances_]); iou = X["fg"] / (X["area"] + G - X["fg"] + 1e-9)
    def leaves(node):
        out, st = [], [node]
        while st:
            v = st.pop()
            if v < n: out.append(v)
            else: st.extend(ch[v - n])
        return np.array(out)
    def mask_of(node):
        m = torch.zeros(n, dtype=torch.bool, device=DEV); m[torch.from_numpy(leaves(node)).to(DEV)] = True; return m.reshape(h, h)
    V = {"insid3": d["pred"], "cand": d["cand"], "node_best": mask_of(int(iou.argmax())), "node_cos": mask_of(int(cosd.argmax())),
         "node_cos_a8": mask_of(int(np.where(X["area"] >= 8, cosd, -9).argmax()))}
    seed = d["seed"]; rec = dict(e=e, c=c, seed=seed, G=float(G), n_cand=float(leaf["cand"].sum()), n_reffg=int(mf.sum()), n_refbg=int((~mf).sum()),
                                 parent=parent.astype(np.int16), dist=dist.astype(np.float16), cos=cosd.astype(np.float16), coh_d=coh_d.astype(np.float16), coh_o=coh_o.astype(np.float16),
                                 **{k: X[k].astype(np.float32) for k in X})
    if seed >= 0:
        sa = float(d["area"][seed]); v = int((labels == seed).nonzero()[0])
        while X["area"][v] < sa - 0.5: v = parent[v]
        chain = []
        while v != -1: chain.append(v); v = parent[v]
        chain = np.array(chain); ci = iou[chain]; V["anc_best"] = mask_of(int(chain[ci.argmax()]))
        nc = leaf["cand"].sum(); piou = X["cand"][chain] / (X["area"][chain] + nc - X["cand"][chain] + 1e-9); V["anc_cand"] = mask_of(int(chain[piou.argmax()]))
        rec.update(chain=chain.astype(np.int16), seed_node=int(chain[0]))
    else:
        V["anc_best"] = V["anc_cand"] = d["pred"]
    for k, v in V.items(): M.add(k, c, *iu(up(v, S), gt))
    tabs.append(rec)
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()), flush=True)
res = dict(fold=a.fold, n=len(tabs), miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0))
print(json.dumps(res, indent=1)); json.dump(res, open(a.out + ".json", "w")); torch.save(tabs, a.out + ".nodes.pt")
