"""Does the bidirectional F1 rule use additional reference images better than INSID3 does?  (COCO-20i, --shots references per episode)

INSID3 is run through its released predict_mask (prototype averaged over shots, backward matching by majority vote).
The F1 rule treats all reference patches as one labelled set: precision = share of node patches whose nearest reference patch (over
all references) is foreground; recall = share of reference foreground patches (all references) whose nearest target patch is in the node.

  python scripts/shots.py --fold 0 --n 300 --shots 5 --out results/shots5_f0_300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import up
from icx.offline2 import accumulate, f1, leaves_of
from utils.data import load_image, load_mask, downsample_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--shots", type=int, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size; U = model.positional_basis; proj = lambda X: F.normalize(X - (X @ U) @ U.T, dim=-1)
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=a.shots); M = Meter(); t0 = time.time()
for e, (c, tn, rns) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); ti, _ = load_image(timg, tf, DEV); refs = [coco_load(base, r, c) for r in rns]
    ri = torch.cat([load_image(im, tf, DEV)[0] for im, _ in refs]); rm = torch.cat([load_mask(m, S_, DEV) for _, m in refs])
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    pred = model.predict_mask(ri, rm, ti); M.add("insid3", c, *iu(pred, gt))
    f = F.normalize(model._extract_features(torch.cat([ri, ti], 0).unsqueeze(0)).float(), p=2, dim=2)[0]; C = f.shape[1]; Tt = f[-1].reshape(C, -1).T; h = int(math.sqrt(len(Tt))); n = h * h
    R = torch.cat([x.reshape(C, -1).T for x in f[:-1]]); mf = torch.cat([downsample_mask(rm[s:s + 1].unsqueeze(1), h, h).reshape(-1) for s in range(len(rns))])
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit((1.0 - (Tt @ Tt.T).clamp(-1, 1)).cpu().numpy())
    ch = ac.children_.astype(np.int64); g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1).cpu().numpy()
    Sd = proj(Tt) @ proj(R).T; back = mf[Sd.argmax(1)].float().cpu().numpy(); fwd = torch.bincount(Sd.argmax(0)[mf], minlength=n).float().cpu().numpy()
    X = accumulate(ch, np.stack([np.ones(n, np.float32), g, back, fwd])); Fs = f1(X[2] / X[0], X[3] / max(int(mf.sum()), 1)); io = X[1] / (X[0] + X[1, -1] - X[1] + 1e-9)
    for name, k in (("f1_node", int(Fs.argmax())), ("oracle_node", int(io.argmax()))):
        m = torch.zeros(n, dtype=torch.bool, device=DEV); m[torch.from_numpy(leaves_of(ch, n, k)).to(DEV)] = True; M.add(name, c, *iu(up(m.reshape(h, h), S_), gt))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()), flush=True)
res = dict(fold=a.fold, n=a.n, shots=a.shots, miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0)); print(json.dumps(res)); json.dump(res, open(a.out + ".json", "w"))
