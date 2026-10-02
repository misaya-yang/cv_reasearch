#!/usr/bin/env python3
"""Diagnostic (uses ground truth everywhere): what the errors are made of and which information would remove them.

Rows scored on the episode's query (pool = up to --M other queries of the class, same sampling as probe_decoder.py):
  1shot | pool (unfiltered pooled vote with one-shot masks of the pool) | true (pool with true masks)
  D5  single references: best1-true (oracle choice of ONE pool image with its true mask), best1-pseudo (oracle choice among the one-shot
      result and the 15 single-pseudo-reference results), medoid-pseudo / vote-pseudo (label-free consensus among those candidates)
  D7  which pool images matter (easy = one-shot mask of the pool image has IoU >= 0.5 with its truth):
      pool-easy (pseudo masks of easy images only = perfect reliability selection), true-easy (true masks of easy images only),
      fix-hard (pseudo for easy, TRUE for hard), fix-easy (TRUE for easy, pseudo for hard)
  D6  curr-rt: sequential propagation, images labelled in order of round-trip score, each from the gold and all images labelled before it
  D11 modes<t>: region prototypes of all images clustered together (average linkage, cosine distance t); a mode is labelled from the
      pool's true masks (oracle), or from the gold image only (gold: unknown modes are background; veto: one-shot minus gold-background
      modes; add: one-shot plus gold-foreground modes; gold+1shot: both)
Error taxonomy at patch level (64x64) with the full COCO class map of the image:
  false positives by true label (person / another annotated class / unannotated), attached to the object or in a separate component;
  misses inside a partly found instance or whole missed instances. Also for the one-shot masks of the pool images.
D4  novelty: similarity of query patches to their nearest gold patch, for true positives, false positives, true negatives, misses.
--viz writes contact sheets (gold+mask | query+truth | 1shot | pool | true) for the listed classes.
"""
import argparse, json, os, sys, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths
from _paths import COCO_ANN
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from tics import ImageSet, one_shot

ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True); ap.add_argument("--M", type=int, default=15); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--thetas", default="0.3,0.45,0.6"); ap.add_argument("--out", default=""); ap.add_argument("--every", type=int, default=50)
ap.add_argument("--viz", default="", help="comma list of class ids"); ap.add_argument("--viz-n", type=int, default=4); ap.add_argument("--viz-dir", default="results/viz_errors")
a = ap.parse_args(); D = torch.load(a.file, weights_only=False); cls = np.array(D["cls"]); names = D["names"]; nE = len(names) // 2
rng = np.random.default_rng(0); t0 = time.time(); recs = []; DEV = "cuda"; H = 64; CM = {}; thetas = [float(x) for x in a.thetas.split(",")]
IMG = os.path.join(os.environ.get("DEMO4_CACHE", "/root/demo4_cache"), "data/COCO2014"); viz_cls = [int(x) for x in a.viz.split(",") if x]; viz_rows = collections.defaultdict(list)
EIGHT = np.ones((3, 3), int)


def cmap(name):
    if name not in CM: CM[name] = np.array(Image.open(os.path.join(COCO_ANN, name[:-4] + ".png")).resize((H, H), Image.NEAREST))
    return CM[name]


def taxonomy(p, g, cm, c):
    """[tp, fp on person, fp on another class, fp unannotated, fp on the class itself (grid mismatch), fp attached, fp detached,
        fn inside partly found instances, fn in wholly missed instances, instances, wholly missed instances]"""
    p = p.cpu().numpy().reshape(H, H); g = g.cpu().numpy().reshape(H, H); fp = p & ~g; fn = g & ~p
    same = cm == c + 1; person = (cm == 1) & (c != 0); other = (cm > 0) & (cm <= 80) & ~same & ~person; unl = ~(same | person | other)
    lp, _ = ndimage.label(p, structure=EIGHT); ids = np.unique(lp[p & g]); att = np.isin(lp, ids[ids > 0])
    lg, m = ndimage.label(g, structure=EIGHT); hid = np.unique(lg[p & g]); hid = hid[hid > 0]; hit = np.isin(lg, hid)
    return [int((p & g).sum()), int((fp & person).sum()), int((fp & other).sum()), int((fp & unl).sum()), int((fp & same).sum()),
            int((fp & att).sum()), int((fp & ~att).sum()), int((fn & hit).sum()), int((fn & ~hit).sum()), int(m), int(m - len(hid))]


def modes(s, q, p1q, out, stats):
    X, img, ar, fg = [], [], [], []
    for i in range(s.n):
        oh = F.one_hot(s.lab[i], s.K[i]).float()
        if i == 0:        # gold image: split every cluster by the true mask, so a small object inside a larger cluster keeps its own region
            g = s.gt64[0].float()
            for part, w in ((1.0, g), (0.0, 1 - g)):
                a_ = oh.T @ w; keep = a_ > 0
                if keep.any():
                    X.append(F.normalize((oh * w[:, None]).T @ s.fd[0], dim=1)[keep]); img += [0] * int(keep.sum()); ar.append(a_[keep]); fg.append(torch.full((int(keep.sum()),), part, device=DEV))
        else:
            X.append(s.Pd[i]); img += [i] * s.K[i]; ar.append(s.area[i]); fg.append((oh.T @ s.gt64[i].float()) / s.area[i])
    X = torch.cat(X).cpu().numpy(); img = np.array(img); ar = torch.cat(ar).cpu().numpy(); fg = torch.cat(fg).cpu().numpy()
    Z = linkage(X, "average", metric="cosine"); isq = img == q; isg = img == 0; labq = s.lab[q]
    p1c = np.array([bool(p1q[labq == k].any()) for k in range(s.K[q])])
    for th in thetas:
        mode = fcluster(Z, t=th, criterion="distance"); nm = mode.max() + 1
        F_pool = np.bincount(mode[~isq], weights=(ar * fg)[~isq], minlength=nm); T_pool = np.bincount(mode[~isq], weights=ar[~isq], minlength=nm)
        F_gold = np.bincount(mode[isg], weights=(ar * fg)[isg], minlength=nm); T_gold = np.bincount(mode[isg], weights=ar[isg], minlength=nm)
        orc = (T_pool > 0) & (F_pool >= 0.5 * T_pool); gF = (T_gold > 0) & (F_gold >= 0.5 * T_gold); gB = (T_gold > 0) & ~gF
        mq = mode[isq]; dec = lambda v: torch.from_numpy(v).to(DEV)[labq]
        out[f"modes{th:g}:oracle"] = dec(orc[mq]); out[f"modes{th:g}:gold"] = dec(gF[mq]); out[f"modes{th:g}:veto"] = dec(p1c & ~gB[mq])
        out[f"modes{th:g}:add"] = dec(p1c | gF[mq]); out[f"modes{th:g}:gold+1shot"] = dec(gF[mq] | (p1c & ~gB[mq] & ~gF[mq]))
        aq, fq_ = ar[isq], fg[isq]; gt_a = aq * fq_; fp_a = aq * (1 - fq_) * p1c; lab3 = np.where(gF[mq], 0, np.where(gB[mq], 1, 2))
        st = stats.setdefault(f"{th:g}", np.zeros(9))
        st += np.array([gt_a[lab3 == k].sum() for k in range(3)] + [fp_a[lab3 == k].sum() for k in range(3)] + [nm - 1, (np.bincount(mode)[1:] == 1).sum(), 1])


def miou(key):
    acc = {}
    for r in recs:
        if key in r["iu"]: s_ = acc.setdefault(r["c"], [0.0, 0.0]); s_[0] += r["iu"][key][0]; s_[1] += r["iu"][key][1]
    return 100 * float(np.mean([i / max(u, 1.0) for i, u in acc.values()])) if acc else float("nan")


def panel(name, mask, color):
    im = Image.open(os.path.join(IMG, name)).convert("RGB").resize((224, 224)); m = np.array(Image.fromarray(mask.cpu().numpy().reshape(H, H).astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)) > 0
    x = np.array(im).astype(float); x[m] = 0.45 * x[m] + 0.55 * np.array(color); return x.astype(np.uint8)


mstats = {}; nov = collections.defaultdict(lambda: np.zeros(3))
for e in range(nE if not a.limit else min(a.limit, nE)):
    c = int(cls[2 * e]); ref, qry = 2 * e, 2 * e + 1
    cand = [2 * o_ + 1 for o_ in range(nE) if o_ != e and cls[2 * o_] == c and names[2 * o_ + 1] not in (names[ref], names[qry])]
    seen = set(); cand = [x for x in cand if not (names[x] in seen or seen.add(names[x]))]
    pool = [int(x) for x in rng.permutation(cand)[:a.M]]; idx = [ref, qry] + pool; nm_ = [names[i] for i in idx]
    s = ImageSet(dict(c=c, names=nm_, fq=D["fq"][idx], lab=D["lab"][idx], Po=[D["Po"][i] for i in idx], gt64=D["gt64"][idx], gt_bits=D["gt_bits"][idx], S=D["S"]))
    J = list(range(1, s.n)); q = 1; g0 = s.gt64[0]; gq = s.gt64[q]; o = [x for x in J if x != q]; P1 = one_shot(s, J)
    r = dict(e=e, c=c, pool=len(pool), iu={}, tax={}); M = {}
    def put(k, m): r["iu"][k] = s.iu(m, q); M[k] = m
    put("1shot", P1[q]); put("pool", s.predict(q, [0] + o, [g0] + [P1[x] for x in o], backward="pooled", k=5))
    put("true", s.predict(q, [0] + o, [g0] + [s.gt64[x] for x in o], backward="pooled", k=5))
    # D5 single references and candidate selection
    ct = {y: s.predict(q, [y], [s.gt64[y]]) for y in o}; cp = [P1[q]] + [s.predict(q, [y], [P1[y]]) for y in o if P1[y].any()]
    if ct: put("best1-true", max(ct.values(), key=lambda m: s.iou(m, gq))); r["mean1_true"] = float(np.mean([s.iou(m, gq) for m in ct.values()]))
    put("best1-pseudo", max(cp, key=lambda m: s.iou(m, gq)))
    A = np.array([[s.iou(x, y) for y in cp] for x in cp]); put("medoid-pseudo", cp[int(A.mean(1).argmax())]); put("vote-pseudo", torch.stack(cp).float().mean(0) >= 0.5)
    # D7 easy / hard pool images
    qual = {y: s.iou(P1[y], s.gt64[y]) for y in o}; easy = [y for y in o if qual[y] >= 0.5]; r["n_easy"] = len(easy); r["n_pool"] = len(o)
    put("pool-easy", s.predict(q, [0] + easy, [g0] + [P1[y] for y in easy], backward="pooled", k=5))
    put("true-easy", s.predict(q, [0] + easy, [g0] + [s.gt64[y] for y in easy], backward="pooled", k=5))
    put("fix-hard", s.predict(q, [0] + o, [g0] + [P1[y] if y in easy else s.gt64[y] for y in o], backward="pooled", k=5))
    put("fix-easy", s.predict(q, [0] + o, [g0] + [s.gt64[y] if y in easy else P1[y] for y in o], backward="pooled", k=5))
    # D6 sequential propagation in order of round-trip score
    rt = {y: (s.iou(s.predict(0, [y], [P1[y]]), g0) if P1[y].any() else 0.0) for y in J}; L = [0]; Mk = {0: g0}
    for y in sorted(J, key=lambda y: -rt[y]): Mk[y] = s.predict(y, L, [Mk[l] for l in L], backward="pooled", k=5); L.append(y)
    put("curr-rt", s.predict(q, [l for l in L if l != q], [Mk[l] for l in L if l != q], backward="pooled", k=5))
    # D11 collection-level modes
    mo = {}; modes(s, q, P1[q], mo, mstats)
    for k_, m in mo.items(): put(k_, m)
    # taxonomy and novelty
    cmq = cmap(nm_[q]); r["grid_iou"] = float(((cmq == c + 1) & gq.cpu().numpy().reshape(H, H)).sum() / max(((cmq == c + 1) | gq.cpu().numpy().reshape(H, H)).sum(), 1))
    for k_ in ("1shot", "pool", "true"): r["tax"][k_] = taxonomy(M[k_], gq, cmq, c)
    r["tax"]["poolmask"] = np.sum([taxonomy(P1[y], s.gt64[y], cmap(nm_[y]), c) for y in o], 0).tolist() if o else [0] * 11
    S_ = s.fd[q] @ s.fd[0].T; sf = S_[:, g0].max(1).values if g0.any() else torch.full((s.P,), -1.0, device=DEV); sb = S_[:, ~g0].max(1).values; p = P1[q]
    for nme, sel in (("TP", p & gq), ("FP", p & ~gq), ("TN", ~p & ~gq), ("FN", ~p & gq)):
        if sel.any(): nov[nme] += np.array([float(sel.sum()), float(torch.maximum(sf, sb)[sel].sum()), float((sf - sb)[sel].sum())])
    if c in viz_cls and len(viz_rows[c]) < a.viz_n:
        viz_rows[c].append(np.concatenate([panel(nm_[0], g0, (0, 255, 0)), panel(nm_[q], gq, (0, 255, 0))] + [panel(nm_[q], M[k_], (255, 0, 0)) for k_ in ("1shot", "pool", "true")], 1))
    recs.append(r); del s
    if (e + 1) % a.every == 0:
        torch.cuda.empty_cache(); print(e + 1, f"{time.time() - t0:.0f}s", " | ".join(f"{k} {miou(k):.1f}" for k in ("1shot", "pool", "true", "best1-true", "fix-hard", "curr-rt")), flush=True)

keys = [k for k in recs[0]["iu"] if all(k in r["iu"] for r in recs)]; res = dict(file=a.file, args=vars(a), episodes=len(recs), miou={k: miou(k) for k in keys}, records=recs)
print("\nmIoU:", json.dumps({k: round(v, 1) for k, v in res["miou"].items()}))
print(f"easy pool images (one-shot mask IoU >= 0.5): {np.sum([r['n_easy'] for r in recs]) / np.sum([r['n_pool'] for r in recs]):.2f} of the pool; "
      f"mean IoU of a single true-mask reference {np.mean([r['mean1_true'] for r in recs if 'mean1_true' in r]):.3f}; class-map / gt64 grid agreement {np.mean([r['grid_iou'] for r in recs]):.3f}")
T = lambda rows, k: np.sum([r["tax"][k] for r in rows], 0).astype(float)
def show(rows, k, tag):
    t = T(rows, k); fp = t[1:5].sum(); fn = t[7:9].sum()
    print(f"  {tag:22s} P {t[0] / max(t[0] + fp, 1):.2f} R {t[0] / max(t[0] + fn, 1):.2f} | FP: person {t[1] / max(fp, 1):.2f} other class {t[2] / max(fp, 1):.2f} unannotated {t[3] / max(fp, 1):.2f} grid {t[4] / max(fp, 1):.2f}"
          f" ; attached {t[5] / max(fp, 1):.2f} detached {t[6] / max(fp, 1):.2f} | FN: part of a found instance {t[7] / max(fn, 1):.2f} whole instance {t[8] / max(fn, 1):.2f} ({t[10]:.0f} of {t[9]:.0f} instances missed)")
print("\nerror taxonomy, all episodes (patch counts pooled):")
for k in ("1shot", "pool", "true", "poolmask"): show(recs, k, k)
byc = collections.defaultdict(list)
for r in recs: byc[r["c"]].append(r)
print("per class, one-shot and true-pool query masks:")
for c_, v in sorted(byc.items()): show(v, "1shot", f"class {c_} 1shot"); show(v, "true", f"class {c_} true")
res["taxonomy"] = {k: T(recs, k).tolist() for k in ("1shot", "pool", "true", "poolmask")}; res["taxonomy_by_class"] = {str(c_): {k: T(v, k).tolist() for k in ("1shot", "pool", "true", "poolmask")} for c_, v in byc.items()}
print("\nnovelty (one-shot query patches): mean similarity to the nearest gold patch, mean (nearest gold FG - nearest gold BG)")
for k, v in nov.items(): print(f"  {k}: n {v[0]:.0f}  nearest {v[1] / v[0]:.3f}  margin {v[2] / v[0]:+.3f}")
res["novelty"] = {k: v.tolist() for k, v in nov.items()}
print("\ncollection modes: share of the query's TRUE area in modes labelled by the gold as FG / BG / unknown ; share of the one-shot FALSE-POSITIVE area in FG / BG / unknown modes ; modes per episode, singleton share")
for th, st in mstats.items(): print(f"  t={th}: truth {st[0] / st[:3].sum():.2f} / {st[1] / st[:3].sum():.2f} / {st[2] / st[:3].sum():.2f} ; false positives {st[3] / st[3:6].sum():.2f} / {st[4] / st[3:6].sum():.2f} / {st[5] / st[3:6].sum():.2f} ; {st[6] / st[8]:.0f} modes, {st[7] / max(st[6], 1):.2f} singletons")
res["mode_stats"] = {k: v.tolist() for k, v in mstats.items()}
if a.out: os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"))
if viz_rows:
    os.makedirs(a.viz_dir, exist_ok=True)
    for c_, rows in viz_rows.items(): Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.viz_dir, f"class{c_}.jpg"), quality=80); print("viz", c_, len(rows))
