#!/usr/bin/env python3
"""Diagnostic: does INSID3's single-seed aggregation suppress parts that have cross-image support?

INSID3 keeps a cluster when  cross * intra * coverage > 0.2, where intra is the similarity of the cluster to ONE seed cluster.
A part that does not look like the seed (wheel against car body) is pushed down even when references support it.
Aggregation variants, each with the pool's TRUE masks (ceiling, diagnostic) and with its one-shot pseudo masks:
  insid3    the released rule (pooled vote for the candidates, as in tics.ImageSet.predict)
  nointra   the same without the seed-similarity factor: cross * coverage > 0.2
  cover     a cluster is kept when at least half of its patches are candidates (no seed at all), plus the seed cluster
  local     insid3, plus every cluster whose patches mostly match the foreground of some reference more strongly than the
            background of the LABELLED reference (unsegmented areas of pseudo references are not used as background)
  python scripts/probe_seed.py --file $DEMO9_CACHE/episodes_f0_n400.pt --limit 200 --out results/probe_seed_f0_200.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
import numpy as np, torch, torch.nn.functional as F
from tics import ImageSet, one_shot

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--k", type=int, default=5); ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=50)
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); t0 = time.time(); recs = []; MODES = ["insid3", "nointra", "cover", "local"]


@torch.no_grad()
def predict_modes(s, t, refs, masks, k):
    """the four aggregation rules on the same prototype and the same candidate set"""
    keep = [i for i, m in enumerate(masks) if m.any()]; refs = [refs[i] for i in keep]; masks = [masks[i] for i in keep]
    proto = F.normalize(sum(s.fd[j][m].mean(0) for j, m in zip(refs, masks)) / len(refs), dim=0)
    sim_fwd = s.fd[t] @ proto; fwd = sim_fwd > 0
    if fwd.sum() == 0: fwd = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
    V = torch.stack([s.nnv(t, j)[0] for j in refs]); L = torch.stack([m[s.nnv(t, j)[1]].float() for j, m in zip(refs, masks)])
    kk = min(k, len(refs)); top = V.topk(kk, dim=0).indices; back = L.gather(0, top).mean(0) > (0.5 if kk > 1 else 0.0)
    cand = fwd & back; zero = torch.zeros(s.P, dtype=torch.bool, device=cand.device)
    if cand.sum() == 0: return {m: zero for m in MODES}
    lab, K, Pd, Po, area = s.lab[t], s.K[t], s.Pd[t], s.Po[t], s.area[t]; oh = F.one_hot(lab, K).float()
    cnt = torch.bincount(lab[cand], minlength=K).float(); aw = cnt / area
    cs = Pd @ proto; seed = int(torch.where(cnt > 0, cs, torch.full_like(cs, -9)).argmax())
    cov = aw.clone(); aw[seed] = 1.0; cross = (oh.T @ sim_fwd) / area; intra = Po @ Po[seed]
    out = {"insid3": (cross * intra * aw > s.merge)[lab], "nointra": (cross * aw > s.merge)[lab]}
    kc = cov >= 0.5; kc[seed] = True; out["cover"] = kc[lab]
    # local support: best match to the foreground of any reference against best match to the background of the labelled reference
    sfg = torch.full((s.P,), -2.0, device=cand.device)
    for j, m in zip(refs, masks): sfg = torch.maximum(sfg, (s.fd[t] @ s.fd[j][m].T).max(1).values)
    g0 = masks[0] if refs[0] == 0 else None
    sbg = (s.fd[t] @ s.fd[0][~g0].T).max(1).values if g0 is not None and (~g0).any() else torch.full((s.P,), -2.0, device=cand.device)
    sup = ((oh.T @ (sfg > sbg).float()) / area) >= 0.5
    out["local"] = out["insid3"] | sup[lab]
    return out


def miou(key):
    acc = {}
    for r in recs:
        if key in r["iu"]: s_ = acc.setdefault(r["c"], [0.0, 0.0]); s_[0] += r["iu"][key][0]; s_[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()])) if acc else float("nan")


def prf(key):
    tp = sum(r["pr"][key][0] for r in recs); pp = sum(r["pr"][key][1] for r in recs); gg = sum(r["pr"][key][2] for r in recs)
    return tp / max(pp, 1), tp / max(gg, 1)


for e in range(nE if not a.limit else min(a.limit, nE)):
    c = int(cls[2 * e]); ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o + 1 for o in range(nE) if o != e and cls[2 * o] == c and names[2 * o + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; idx = [ref, qry] + pool
    s = ImageSet(dict(c=c, names=[names[i] for i in idx], fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=D["gt64"][idx], gt_bits=D["gt_bits"][idx], S=D["S"]))
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; o = [x for x in J if x != q]; P1 = one_shot(s, J); gq = s.gt64[q]
    r = dict(e=e, c=c, pool=len(pool), iu={}, pr={})
    def put(k, m): r["iu"][k] = s.iu(m, q); r["pr"][k] = (float((m & gq).sum()), float(m.sum()), float(gq.sum()))
    put("1shot", P1[q])
    for tag, ms in (("1shot", [g0]), ("true", [g0] + [s.gt64[x] for x in o]), ("p1", [g0] + [P1[x] for x in o])):
        rf = [0] if tag == "1shot" else [0] + o
        for mode, m in predict_modes(s, q, rf, ms, 1 if tag == "1shot" else a.k).items(): put(f"{tag}:{mode}", m)
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in recs[0]["iu"]), flush=True)

keys = list(recs[0]["iu"]); res = dict(file=a.file, args=vars(a), episodes=len(recs), miou={k: miou(k) for k in keys},
                                       patch_precision_recall={k: [round(v, 3) for v in prf(k)] for k in keys}, records=recs)
print(json.dumps({k: round(v, 2) for k, v in res["miou"].items()})); print("patch precision / recall of the query masks:", json.dumps(res["patch_precision_recall"]))
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
