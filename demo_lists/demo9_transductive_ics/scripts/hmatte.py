"""Boundary from finer tokens of the query (bank: scripts/hires_bank.py; arms: scripts/matte_common.py). Label-free arms
(grid x band x mix) are read against FoRIS after its CRF at model resolution; the reported gain is nested over folds (the arm
for a fold is chosen on the other three). Arms marked (label) are diagnostics. Writes the gate file for scripts/hires_pipe.py.
  BANK=/path/to/hires_dev THREADS=10 python hmatte.py [limit]      verdict PASS: nested gain >= +2.0, interval above 0"""
from cl_common import *
from matte_common import G, alpha, from64, refine, to_px
BANK = Path(os.environ["BANK"]); LEVELS = [i / 20 for i in range(1, 20)]
ARMS = [(g, b, m) for g in ("s8", "z2") for b in (1, 2, 3) for m in (0, 1)]
name_of = lambda a: "%s | band %d (x 8 px), %s" % (a[0], a[1], "mean of alpha and FoRIS score" if a[2] else "alpha")
R = {}; pur = []; used = []
def put(k, m, truth): R.setdefault(k, []).append(iu(m, truth))
lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
for r in rows[:lim]:
    key = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
    if not (BANK / (key + ".npz")).exists(): continue
    z, bk = np.load(packets / (key + ".npz")), np.load(BANK / (key + ".npz")); used.append(r)
    s = normalise(torch.from_numpy(z["score"]))
    bits = lambda k: torch.from_numpy(np.unpackbits(z[k])[:1024 * 1024].reshape(1024, 1024).astype(bool))
    truth, pre, nat = bits("truth"), bits("pre"), bits("native")
    cov64 = F.adaptive_avg_pool2d(truth[None, None].float(), 64)[0, 0].flatten(); pur.append(float((cov64[s.flatten().topk(41).indices] >= .5).float().mean()))
    put("FoRIS before CRF", pre, truth); put("FoRIS after CRF", nat, truth)
    for name, off in (("s8", 0), ("z2", 4)):
        q = F.normalize(torch.from_numpy(bk[name]).float(), dim=-1); mu, hard = torch.from_numpy(bk["mu_" + name]).float(), torch.from_numpy(bk["hard_" + name]).float()
        sc = from64(s, off)
        hb = hard - (hard @ mu) * mu / (mu @ mu).clamp_min(1e-8); fine = (q @ F.normalize(mu, dim=0) - .55 * (q @ F.normalize(hb, dim=0))).view(G, G)
        u = to_px(normalise(fine), off); R.setdefault("%s | fine one-vector score, best cut (label)" % name, []).append(max((iu(u > l, truth) for l in LEVELS), key=lambda v: v[0] / max(v[1], 1)))
        put("%s | FoRIS score + fine one-vector score, cut at half the range" % name, to_px(normalise(sc + normalise(fine)), off) > .5, truth)
        for a in ARMS:
            if a[0] == name: put(name_of(a), refine(q, s, off, a[1], a[2]), truth)
        cov = (F.avg_pool2d(F.pad(truth[None, None].float(), (4, 4, 4, 4)), 8, 8)[0, 0, :G, :G] if off == 0 else F.avg_pool2d(truth[None, None].float(), 8, 8)[0, 0]).numpy()
        a = alpha(q, cov >= .999, cov <= .001, (cov > .001) & (cov < .999))
        put("%s | truth trimap, alpha (label)" % name, to_px(a if a is not None else cov, off) > .5, truth)
        put("%s | truth trimap, true coverage (label)" % name, to_px(cov, off) > .5, truth)
    if len(used) % 20 == 0: print(len(used), flush=True)
pur = np.array(pur); cls, groups, fold = np.array([r["c"] for r in used]), photo_groups(used), np.array([r["fold"] for r in used]); base = np.array(R["FoRIS after CRF"], float)
GR = (("pure", pur >= .9), ("mid", (pur >= .5) & (pur < .9)), ("bad", pur < .5)); print("episodes %d; gain against FoRIS after its CRF (model resolution)" % len(used))
gain = {}
for k, v in R.items():
    v = np.array(v, float); io = v[:, 0] / np.maximum(v[:, 1], 1); p = paired(v, base, cls, groups, 500); gain[k] = p
    print("%-66s %6.2f %+6.2f [%+5.2f,%+5.2f] up %3d down %3d | %s | folds %s" % (k, p["miou"], p["gain"], p["ci95"][0], p["ci95"][1], p["up"], p["down"], "  ".join("%s %.3f" % (n, io[m].mean()) for n, m in GR if m.any()),
          " ".join("%+.1f" % (class_miou(v[fold == f], cls[fold == f]) - class_miou(base[fold == f], cls[fold == f])) for f in range(4) if (fold == f).any())))
# nested over folds: the arm used on a fold is the best one on the other folds
nested = base.copy(); picks = {}
for f in np.unique(fold):
    o = fold != f
    if not o.any(): continue
    best = max(ARMS, key=lambda a: class_miou(np.array(R[name_of(a)], float)[o], cls[o]) - class_miou(base[o], cls[o]))
    nested[fold == f] = np.array(R[name_of(best)], float)[fold == f]; picks[int(f)] = name_of(best)
pn = paired(nested, base, cls, groups, 2000); top = max(ARMS, key=lambda a: gain[name_of(a)]["gain"])
verdict = "PASS" if pn["gain"] >= 2.0 and pn["ci95"][0] > 0 else "FAIL"
out = dict(state="COMPLETED", episodes=len(used), control="FoRIS after its CRF, model resolution", verdict=verdict, nested=pn, picks=picks,
           frozen=dict(grid=top[0], band=top[1], mix=top[2]), frozen_gain=gain[name_of(top)]["gain"], frozen_ci95=gain[name_of(top)]["ci95"],
           arms={name_of(a): dict(gain=gain[name_of(a)]["gain"], ci95=gain[name_of(a)]["ci95"]) for a in ARMS},
           diagnostics={k: gain[k]["miou"] for k in gain if "(label)" in k})
(Path(os.environ.get("GATE", str(L / "hmatte_gate.json")))).write_text(json.dumps(out, indent=1))
print("nested over folds: %+.2f [%+.2f, %+.2f]; arm frozen on all episodes: %s (%+.2f); verdict %s" % (pn["gain"], pn["ci95"][0], pn["ci95"][1], name_of(top), gain[name_of(top)]["gain"], verdict))
