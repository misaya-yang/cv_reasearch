"""Is the transductive step a plug-in? FoRIS (github.com/Xi-Mu-Yu/FoRIS, commit 1aa02a1) as the black-box base method on the
standard COCO-20i episodes, with up to M unlabeled images of the same class (queries of other episodes) as the pool.
Only FoRIS's own predict() is used (its multi-reference rule included); encoder features and clusterings are memoised.
  1shot    FoRIS on (reference, query)
  naive    FoRIS with its own one-shot masks of all pool images as extra references
  rt r2/r3 pseudo references kept when their round-trip score (IoU on the labelled reference) exceeds 0.5; 2 further rounds
  true     pool images with their true masks (ceiling)
  python scripts/foris_stream.py --fold 0 --n 400 --limit 80 --out results/foris_stream_f0
"""
import argparse, json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
FORIS = "/root/autodl-tmp/demo8_local_verification/foris_source"
sys.path[:0] = ["/root/demo4_cache/env", FORIS]
import numpy as np, torch, torch.nn.functional as F
import models.foris as foris_mod
from models.foris import FoRIS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.common import TimmDINOv3, coco_episodes, coco_load, iu, Meter
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=400); ap.add_argument("--limit", type=int, default=80)
ap.add_argument("--M", type=int, default=15); ap.add_argument("--res", type=int, default=1024); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=5)
a = ap.parse_args(); DEV = "cuda"; torch.cuda.set_per_process_memory_fraction(0.45)

_orig_cluster = foris_mod.agglomerative_clustering; _memo = {}
def cached_cluster(x, tau=0.6):
    key = (tuple(x.shape), round(float(x.sum()), 3), round(float(x[0].sum()), 4), round(float(x[-1].sum()), 4), tau)
    if key not in _memo: _memo[key] = _orig_cluster(x, tau=tau)
    return _memo[key]
foris_mod.agglomerative_clustering = cached_cluster

class Cached(FoRIS):
    feats = None; order = None
    def _extract_features(self, imgs):
        return torch.stack([self.feats[i] for i in self.order])[None]

model = Cached(TimmDINOv3().eval(), image_size=a.res, mask_refiner="bilinear", resize_to_orig_size=False)
for p in model.parameters(): p.requires_grad = False
model = model.to(DEV).eval(); tf = model._transform if hasattr(model, "_transform") else None
from utils.data import build_transform
tf = tf or build_transform(a.res)
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=1); cls = [c for c, _, _ in eps]; qn = [t for _, t, _ in eps]; rn = [r[0] for _, _, r in eps]
rng = np.random.default_rng(0); M = Meter(); t0 = time.time()

def load(name, c):
    img, m = coco_load(base, name, c); x = tf(img).to(DEV)
    g = F.interpolate(m[None, None].float().to(DEV), size=(a.res, a.res), mode="nearest")[0, 0] > 0.5
    return x, g

@torch.inference_mode()
def pred(t, refs, masks):
    keep = [i for i, m in enumerate(masks) if m.any()]
    if not keep: return torch.zeros(a.res, a.res, dtype=torch.bool, device=DEV)
    refs = [refs[i] for i in keep]; masks = [masks[i] for i in keep]
    model.order = refs + [t]
    try: return model.predict(X[refs], torch.stack(masks), X[t])
    except RuntimeError as ex:
        if "No foreground" in str(ex): return torch.zeros(a.res, a.res, dtype=torch.bool, device=DEV)
        raise

iou = lambda p, q: float((p & q).sum()) / max(float((p | q).sum()), 1.0)
for e in range(min(a.limit, len(eps))):
    c = cls[e]
    cand = [o for o in range(len(eps)) if o != e and cls[o] == c and qn[o] not in (rn[e], qn[e])]
    seen = set(); cand = [o for o in cand if not (qn[o] in seen or seen.add(qn[o]))]
    pool = [int(o) for o in rng.permutation(cand)[:a.M]]
    data = [load(rn[e], c), load(qn[e], c)] + [load(qn[o], c) for o in pool]
    X = torch.stack([d[0] for d in data]); G = [d[1] for d in data]; n = len(data); J = list(range(1, n)); q = 1; _memo.clear()
    with torch.inference_mode():
        fm = model.encoder.get_intermediate_layers(X, n=1, reshape=True)[0] if n <= 9 else torch.cat([model.encoder.get_intermediate_layers(X[s:s + 6], n=1, reshape=True)[0] for s in range(0, n, 6)])
    model.feats = fm
    P1 = {0: G[0]}; P1.update({j: pred(j, [0], [G[0]]) for j in J}); M.add("1shot", c, *iu(P1[q], G[q])); t1 = time.time()
    o = [x for x in J if x != q]
    M.add("naive", c, *iu(pred(q, [0] + o, [G[0]] + [P1[x] for x in o]), G[q]))
    M.add("true", c, *iu(pred(q, [0] + o, [G[0]] + [G[x] for x in o]), G[q]))
    trust = {y: (iou(pred(0, [y], [P1[y]]), G[0]) > 0.5 if P1[y].any() else False) for y in o}
    oo = [x for x in o if trust[x]]; M.add("rt r2", c, *iu(pred(q, [0] + oo, [G[0]] + [P1[x] for x in oo]), G[q]))
    if (e + 1) % a.every == 0: print(e + 1, f"{time.time() - t0:.0f}s", " ".join(f"{k} {M.miou(k):.1f}" for k in M.names()), flush=True)
res = dict(fold=a.fold, episodes=min(a.limit, len(eps)), miou={k: round(M.miou(k), 2) for k in M.names()}, seconds=round(time.time() - t0)); print(json.dumps(res)); json.dump(res, open(a.out + ".json", "w"))
