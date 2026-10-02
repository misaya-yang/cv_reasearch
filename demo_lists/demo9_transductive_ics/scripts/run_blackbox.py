#!/usr/bin/env python3
"""GPU, ~10-40 s per episode: the transductive step around a BLACK-BOX base method (default FoRIS, commit 1aa02a1).

Only the base method's own predict(ref_images, ref_masks, target) is called; encoder features and clusterings are memoised
so that repeated calls on the same images are cheap. Standard COCO-20i episodes; pool = up to --M queries of other episodes
of the same class. Rows: 1shot | naive (all pseudo references, base method's own multi-reference rule) | true (ceiling)
and, per --trust entry, references filtered by roundtrip (score > 0.5) or agree (top half by mutual agreement; N^2 calls).
  python scripts/run_blackbox.py --fold 0 --limit 60 --M 7 --trust roundtrip,agree --out results/e2_foris_f0.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DEMO4
FORIS = os.environ.get("FORIS_ROOT", "/root/autodl-tmp/demo8_local_verification/foris_source")     # read-only checkout of another session
sys.path[:0] = ["/root/demo4_cache/env", FORIS]
import numpy as np, torch, torch.nn.functional as F
import models.foris as foris_mod                      # must be imported before icx.common puts INSID3's `models` on the path
from models.foris import FoRIS
from utils.data import build_transform
sys.path.append(DEMO4)
from icx.common import TimmDINOv3, coco_episodes, coco_load

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=400, help="episode list length that defines the pools")
ap.add_argument("--limit", type=int, default=60); ap.add_argument("--M", type=int, default=7); ap.add_argument("--res", type=int, default=1024); ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--trust", default="roundtrip,agree"); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=5)
a = ap.parse_args(); DEV = "cuda"

_orig = foris_mod.agglomerative_clustering; _memo = {}
def _cached_cluster(x, tau=0.6):
    key = (tuple(x.shape), round(float(x.sum()), 3), round(float(x[0].sum()), 4), round(float(x[-1].sum()), 4), tau)
    if key not in _memo: _memo[key] = _orig(x, tau=tau)
    return _memo[key]
foris_mod.agglomerative_clustering = _cached_cluster

class Cached(FoRIS):
    feats = None; order = None
    def _extract_features(self, imgs): return torch.stack([self.feats[i] for i in self.order])[None]

model = Cached(TimmDINOv3().eval(), image_size=a.res, mask_refiner="bilinear", resize_to_orig_size=False)
for p in model.parameters(): p.requires_grad = False
model = model.to(DEV).eval(); tf = build_transform(a.res)
eps, class_ids, base = coco_episodes(a.fold, a.n, shot=1); cls = [c for c, _, _ in eps]; qn = [t for _, t, _ in eps]; rn = [r[0] for _, _, r in eps]
rng = np.random.default_rng(0); t0 = time.time(); recs = []; iou = lambda p, q: float((p & q).sum()) / max(float((p | q).sum()), 1.0)
iu = lambda p, g: (float((p & g).sum()), float((p | g).sum()))

def load(name, c):
    img, m = coco_load(base, name, c); return tf(img).to(DEV), F.interpolate(m[None, None].float().to(DEV), size=(a.res, a.res), mode="nearest")[0, 0] > 0.5

@torch.inference_mode()
def pred(t, refs, masks):
    keep = [i for i, m in enumerate(masks) if m.any()]
    if not keep: return torch.zeros(a.res, a.res, dtype=torch.bool, device=DEV)
    refs = [refs[i] for i in keep]; model.order = refs + [t]
    try: return model.predict(X[refs], torch.stack([masks[i] for i in keep]), X[t])
    except RuntimeError as ex:
        if "No foreground" in str(ex): return torch.zeros(a.res, a.res, dtype=torch.bool, device=DEV)
        raise

def miou(key):
    acc = {}
    for r in recs: s = acc.setdefault(r["c"], [0.0, 0.0]); s[0] += r["iu"][key][0]; s[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()]))

for e in range(min(a.limit, len(eps))):
    c = cls[e]; cand = [o for o in range(len(eps)) if o != e and cls[o] == c and qn[o] not in (rn[e], qn[e])]
    seen = set(); cand = [o for o in cand if not (qn[o] in seen or seen.add(qn[o]))]; pool = [int(o) for o in rng.permutation(cand)[:a.M]]
    data = [load(rn[e], c), load(qn[e], c)] + [load(qn[o], c) for o in pool]; X = torch.stack([d[0] for d in data]); G = [d[1] for d in data]
    n = len(data); J = list(range(1, n)); q = 1; _memo.clear()
    with torch.inference_mode(): model.feats = torch.cat([model.encoder.get_intermediate_layers(X[s:s + 6], n=1, reshape=True)[0] for s in range(0, n, 6)])
    P1 = {0: G[0]}; P1.update({j: pred(j, [0], [G[0]]) for j in J}); o = [x for x in J if x != q]; r = dict(e=e, c=int(c), pool=len(pool), iu={})
    r["iu"]["1shot"] = iu(P1[q], G[q]); r["iu"]["naive"] = iu(pred(q, [0] + o, [G[0]] + [P1[x] for x in o]), G[q]); r["iu"]["true"] = iu(pred(q, [0] + o, [G[0]] + [G[x] for x in o]), G[q])
    for trust in [t for t in a.trust.split(",") if t]:
        P = dict(P1)
        for rd in range(2, a.rounds + 1):
            if trust == "roundtrip": ok = {y: (iou(pred(0, [y], [P[y]]), G[0]) > 0.5 if P[y].any() else False) for y in J}
            else:
                ag = {y: (np.mean([iou(pred(x, [y], [P[y]]), P[x]) for x in J if x != y]) if P[y].any() and len(J) > 1 else 0.0) for y in J}; med = np.median(list(ag.values())); ok = {y: ag[y] >= med and P[y].any() for y in J}
            Pn = dict(P)
            for j in (J if rd < a.rounds else [q]):                           # pool masks are only needed if another round follows
                oo = [x for x in J if x != j and ok[x]]; Pn[j] = pred(j, [0] + oo, [G[0]] + [P[x] for x in oo])
            P = Pn
        r["iu"][trust] = iu(P[q], G[q])
    recs.append(r)
    if (e + 1) % a.every == 0: print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in recs[0]["iu"]), flush=True)
res = dict(fold=a.fold, args=vars(a), episodes=len(recs), miou={k: miou(k) for k in recs[0]["iu"]}, records=recs); print(json.dumps(res["miou"]))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
