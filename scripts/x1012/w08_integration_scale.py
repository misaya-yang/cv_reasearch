"""W08: read x1013 T0/T2-T5 on the selection half and ask what selects the integration scale.

Motivation: T4/T5 variants help large targets and hurt small ones (and the sharpening variants do
the opposite), so none passes the plan's retention rule.  The open question is whether a quantity
available at inference can pick the operator: the reference mask's area, or the candidate's own
predicted area.  Offline; reads only the per-episode I/U written by scripts/x1013 (third integer of
each arm = predicted area) and, for the last table, the T5 query clusters and query GT (diagnostic).

    python scripts/x1012/w08_integration_scale.py [S|T|ALL]      # T is reserved for plan T9
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
A = HERE.parents[2] / "cv_data/a"
D = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road")
SPL = sys.argv[1] if len(sys.argv) > 1 else "S"
E = {}
for t in ("T0", "T2", "T3", "T4", "T5"):
    for l in gzip.open(A / f"x1013_{t}/iu.jsonl.gz", "rt"):
        r = json.loads(l); E.setdefault(r["index"], {}).update(r)
E = [e for e in E.values() if SPL == "ALL" or e["split"] == SPL]
ARMS = [k for k in E[0] if isinstance(E[0][k], list)]
PX = 1024 * 1024
rng = np.random.default_rng(20261013)
iou = lambda e, a: 100 * e[a][0] / max(e[a][1], 1)


def miou(L, pick, w=None):
    """(fold, class)-pooled I/U, slot-equal mean; pick(e) names the arm used for episode e."""
    I, U = {}, {}
    for j, e in enumerate(L):
        k, a, c = (e["fold"], e["cls"]), pick(e), 1 if w is None else w[j]
        I[k] = I.get(k, 0) + c * e[a][0]; U[k] = U.get(k, 0) + c * e[a][1]
    return float(np.mean([100 * I[k] / U[k] for k in I if U[k] > 0]))


def table(ref, rules, B=300, title=""):
    print(f"\n## {title}: class-pooled diff vs {ref} (* = photo-bootstrap 95% CI excludes 0)")
    print(f"{'rule':30s}" + "".join(f"{d[:6]:>9s}" for d in D) + f"{'mean':>8s}")
    out = {n: [] for n in rules}
    for d in D:
        L = [e for e in E if e["dataset"] == d]
        ph = {k: i for i, k in enumerate(sorted({e["photo"] for e in L}))}
        p = np.array([ph[e["photo"]] for e in L])
        W = [np.bincount(rng.integers(0, len(ph), len(ph)), minlength=len(ph))[p] for _ in range(B)]
        base, bw = miou(L, lambda e: ref), [miou(L, lambda e: ref, w) for w in W]
        for n, pick in rules.items():
            lo, hi = np.percentile([miou(L, pick, w) - b for w, b in zip(W, bw)], [2.5, 97.5])
            out[n].append((miou(L, pick) - base, hi < 0 or lo > 0))
    for n in rules:
        print(f"{n:30s}" + "".join(f"{v:+8.2f}{'*' if s else ' '}" for v, s in out[n]) + f"{np.mean([v for v, _ in out[n]]):+8.2f}")


arm = lambda a: (lambda e: a)
gate = lambda a, key, th: (lambda e: a if e["rho"] <= .9 and key(e) > th else "candidate")
pred, gt, ref = (lambda e: e["candidate"][2] / PX), (lambda e: e["gt"] / PX), (lambda e: e["reference_fraction"])
print("split", SPL, "n", len(E))
table("foris", {a: arm(a) for a in ARMS if a not in ("foris", "A4")}, title="plan arms")
table("candidate", {a: arm(a) for a in ARMS if a not in ("foris", "candidate", "A4")}, title="plan arms")
table("foris", {"candidate": arm("candidate"), "F1b if predicted area >25%": gate("F1b", pred, .25),
                "F6a if predicted area >25%": gate("F6a", pred, .25), "F1b if reference area >25%": gate("F1b", ref, .25),
                "F1b if GT area >25% (oracle)": gate("F1b", gt, .25)}, title="selector for region-level integration")

ds = np.array([e["dataset"] for e in E]); keep = (ds != "lung") & (np.array([e["rho"] for e in E]) <= .9)
SH = ("foris", "A0", "A7", "F6b", "F6a", "F1b")
diff = {a: np.array([iou(e, a) - iou(e, "candidate") for e in E]) for a in SH}
BINS = [("<1%", 0, .01), ("1-3%", .01, .03), ("3-10%", .03, .1), ("10-25%", .1, .25), (">25%", .25, 2)]
for name, f in (("QUERY GT area (diagnostic)", gt), ("REFERENCE area (legal)", ref), ("candidate PREDICTED area (legal)", pred)):
    v = np.array([f(e) for e in E])
    print(f"\n## per-episode IoU diff vs candidate by {name}; rho<=.9, Lung excluded\n{'bin':>8s}{'n':>6s}" + "".join(f"{a:>8s}" for a in SH))
    for nm, lo, hi in BINS:
        m = (v >= lo) & (v < hi) & keep
        print(f"{nm:>8s}{m.sum():6d}" + "".join(f"{diff[a][m].mean():+8.2f}" for a in SH))
    g = np.array([gt(e) for e in E])
    print("  spearman with GT area: " + " ".join(f"{d[:6]} {spearmanr(v[ds == d], g[ds == d])[0]:.2f}" for d in D))

# How well do the T5 feature-only query clusters respect the target? (query GT, diagnostic)
import torch
sys.path.insert(0, str(HERE)); import lab, w02_scale_eval as w
lab.DEV = torch.device("cpu"); torch.set_num_threads(2)
rows = {str(b["index"]): b for b in w.rows_all(0)}
print(f"\n## T5 clusters (100 random episodes per domain)\n{'domain':12s}{'K med':>7s}{'union-oracle IoU':>18s}{'largest cluster':>17s}")
for d in D:
    L = [e for e in E if e["dataset"] == d and e["rho"] <= .9]
    L = [L[i] for i in np.random.default_rng(0).permutation(len(L))[:100]]
    K, orc, big = [], [], []
    for e in L:
        i = e["index"]
        lb = np.load(A / f"x1013_T5/clusters/{int(i):06d}.npz" if i.isdigit() else A / f"x1013_T5/clusters/{i}.npz")["labels"].astype(int)
        y = lab.coverage64(lab.mask1024(rows[i]["row"]["query_mask_path"])).flatten().numpy()
        tot, hit = np.bincount(lb).astype(float), np.bincount(lb, weights=y)
        o = np.argsort(-hit / tot); I, P = np.cumsum(hit[o]), np.cumsum(tot[o])
        K.append(len(tot)); orc.append(100 * (I / (P + y.sum() - I)).max()); big.append(tot.max() / 4096)
    if L:
        print(f"{d:12s}{np.median(K):7.0f}{np.mean(orc):18.1f}{np.mean(big):17.2f}")
