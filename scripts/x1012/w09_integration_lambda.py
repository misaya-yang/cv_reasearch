"""W09: is the integration length of our own graph solve the knob behind the large-target deficit?

Motivation (W08): every tested variant lies on one axis -- region-level integration helps large
targets and hurts small ones -- and the candidate trails FoRIS only where the target covers more
than 25% of the query.  The candidate's solve (diag a + lam L) f = a s has used lam = 16 since B4;
lam sets the integration length and has never been varied.  Hard feature clusters (T5) fail on Lung.

Pre-stated predictions, selection half only, rho <= .9 (the fallback branch does not use the solve):
  H1  dose-response: larger lam raises IoU for GT area > 25% and lowers it for GT area < 3%.
  H2  origin of the level law: at larger lam the gap between the natural (min-max > .5) and the
      GT-optimal level shrinks on large targets and corr(t*, log area) weakens.
  H3  the first-pass predicted area (legal) can pick lam.
If H1 fails, lam is not the integration knob and the solve is left alone.  If H1 holds but H3 fails,
the selector must come from elsewhere.  Offline: reads the x1013 T0 field package (primary, log_e,
Laplacian), the cached FoRIS score and query GT (scoring / diagnostic).  No encoder call.

    python scripts/x1012/w09_integration_lambda.py produce OUT [workers] [S|T|ALL]
    python scripts/x1012/w09_integration_lambda.py report  OUT
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
A = HERE.parents[2] / "cv_data/a"
LAMS = (0, 4, 16, 64, 256, 1024)
D = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road")


def solve(primary, log_e, L, lam):
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from scipy.special import expit
    s = primary.ravel().astype(np.float64)
    s = (s - s.min()) / max(s.max() - s.min(), 1e-12)
    if lam == 0:
        return s
    a = (.1 + np.abs(2 * s - 1)) * expit(log_e.astype(np.float64) - np.median(log_e))
    a = a / a.mean()
    f, status = cg(sparse.diags(a) + lam * L, a * s, x0=s, rtol=1e-7, atol=1e-9, maxiter=2000)
    assert status == 0, status
    return f


def curve(field64, gt):
    """Natural, GT-optimal and equal-area operating points of minmax(up(field)) at 1/256 resolution."""
    import torch, lab
    u = lab.minmax(lab.up(torch.as_tensor(np.asarray(field64, np.float32)).reshape(64, 64))).numpy()
    k = np.minimum((u * 256).astype(np.int64), 255).ravel()
    g = gt.ravel()
    P = np.cumsum(np.bincount(k, minlength=256)[::-1])[::-1].astype(np.float64)      # pixels with bin >= j
    I = np.cumsum(np.bincount(k[g], minlength=256)[::-1])[::-1].astype(np.float64)
    G = float(g.sum()); iou = I / (P + G - I)
    j, e = int(iou[1:].argmax()) + 1, int(np.abs(P - G).argmin())
    return {"nat": [int(I[128]), int(P[128] + G - I[128]), int(P[128])], "opt": float(100 * iou[j]), "t": j / 256,
            "eq": [int(I[e]), int(P[e] + G - I[e])]}


def one(b):
    import torch, lab
    from scipy import sparse
    lab.DEV = torch.device("cpu"); torch.set_num_threads(1)
    i = str(b["index"])
    z = np.load(A / "x1013_T0/fields" / ((f"{int(i):06d}" if i.isdigit() else i) + ".npz"))
    if float(z["reference_rho"][0]) > .9:
        return None
    gt = lab.mask1024(b["row"]["query_mask_path"]).cpu().numpy().astype(bool)
    L = sparse.csr_matrix((z["L_data"], z["L_indices"], z["L_indptr"]), shape=(4096, 4096))
    r = {"index": i, "gt": int(gt.sum())}
    for lam in LAMS:
        f = solve(z["primary"], z["log_e"], L, lam)
        if lam == 16:
            r["identity"] = float(np.abs(f - z["VP_UN"].ravel()).max())
        r[f"l{lam}"] = curve(f, gt)
    try:
        r["foris"] = curve(np.load(b["score64"]["path"])[b["score64"]["key"]], gt)
    except Exception:
        pass
    return r


def load_meta(split="S"):
    M = {}
    for l in gzip.open(A / "x1013_T0/iu.jsonl.gz", "rt"):
        e = json.loads(l)
        if split == "ALL" or e["split"] == split:
            M[e["index"]] = e
    return M


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "produce":
        import w02_scale_eval as w
        out.mkdir(parents=True, exist_ok=True)
        M = load_meta(sys.argv[4] if len(sys.argv) > 4 else "S")
        rows = [b for b in w.rows_all(0) if str(b["index"]) in M]
        with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3,
                                 mp_context=multiprocessing.get_context("spawn")) as pool, (out / "episodes.jsonl").open("w") as fh:
            for n, rec in enumerate(pool.map(one, rows, chunksize=8), 1):
                if rec:
                    fh.write(json.dumps(rec) + "\n")
                if n % 500 == 0:
                    print(n, len(rows), flush=True)
        print("done", flush=True)
        sys.exit()

    M = load_meta("ALL")
    E = [dict(M[r["index"]], **r) for r in map(json.loads, (out / "episodes.jsonl").read_text().splitlines())]
    PX = 1024 * 1024
    print("n", len(E), "max |lam16 - stored VP_UN|", max(e["identity"] for e in E),
          "lam16 natural I/U equals T0 candidate in", np.mean([e["l16"]["nat"][:2] == e["candidate"][:2] for e in E]).round(3))
    F = [f"l{l}" for l in LAMS] + ["foris"]
    E = [e for e in E if "foris" in e]
    ds = np.array([e["dataset"] for e in E]); gf = np.array([e["gt"] / PX for e in E])
    pf = np.array([e["l16"]["nat"][2] / PX for e in E])
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    V = {(f, k): np.array([iou(e[f][k]) if k != "opt" else e[f]["opt"] for e in E]) for f in F for k in ("nat", "opt", "eq")}
    T_ = {f: np.array([e[f]["t"] for e in E]) for f in F}
    BINS = [("<1%", 0, .01), ("1-3%", .01, .03), ("3-10%", .03, .1), ("10-25%", .1, .25), ("25-50%", .25, .5), (">50%", .5, 2)]

    def bins(v, k, title, mask=None):
        mask = np.ones(len(E), bool) if mask is None else mask
        print(f"\n## {title}\n{'bin':>8s}{'n':>6s}" + "".join(f"{f:>8s}" for f in F))
        for nm, lo, hi in BINS:
            m = (v >= lo) & (v < hi) & mask
            if m.sum() >= 25:
                print(f"{nm:>8s}{m.sum():6d}" + "".join(f"{V[(f, k)][m].mean():8.1f}" for f in F))

    nl = ds != "lung"
    bins(gf, "nat", "H1 natural (min-max > .5) per-episode IoU by GT area, Lung excluded", nl)
    bins(gf, "eq", "ranking at the true area (top-k, diagnostic) by GT area, Lung excluded", nl)
    bins(gf, "opt", "GT-optimal level IoU by GT area, Lung excluded", nl)
    bins(pf, "nat", "H3 natural IoU by first-pass (lam 16) PREDICTED area, Lung excluded", nl)
    for d in ("lung", "coco", "isic", "suim"):
        bins(pf, "nat", f"{d}: natural IoU by first-pass predicted area", ds == d)
    print("\n## H2 level law: corr(t*, log GT area) and mean (optimal - natural) on GT area > 25%")
    big = gf > .25
    for f in F:
        c = np.mean([np.corrcoef(T_[f][ds == d], np.log(gf[ds == d]))[0, 1] for d in D if (ds == d).sum() > 30])
        print(f"{f:>8s}  mean within-domain corr {c:+.2f}   gap on large {np.mean(V[(f, 'opt')][big] - V[(f, 'nat')][big]):5.1f}"
              f"   sd t* {T_[f].std():.2f}")

    def miou(L, pick):
        I, U = {}, {}
        for e in L:
            k = (e["fold"], e["cls"]); x = pick(e)
            I[k] = I.get(k, 0) + x[0]; U[k] = U.get(k, 0) + x[1]
        return float(np.mean([100 * I[k] / U[k] for k in I if U[k] > 0]))

    # rho > .9 episodes are absent here; every rule keeps the candidate's fallback there, so differences among rules are unaffected.
    step = lambda lo, hi, th: (lambda e: e[f"l{hi}"]["nat"] if e["l16"]["nat"][2] / PX > th else e[f"l{lo}"]["nat"])
    rules = {"FoRIS (complete)": lambda e: e["foris_final"], **{f"lam {l}": (lambda e, l=l: e[f"l{l}"]["nat"]) for l in LAMS},
             "lam 64 if pred>25%": step(16, 64, .25), "lam 256 if pred>25%": step(16, 256, .25),
             "lam 1024 if pred>25%": step(16, 1024, .25), "lam 256 if pred>10%": step(16, 256, .10),
             "GT-oracle lam (diagnostic)": lambda e: max((e[f"l{l}"]["nat"] for l in LAMS), key=iou)}
    for e in E:
        e["foris_final"] = M[e["index"]]["foris"]
    print(f"\n## class-pooled mIoU, rho <= .9 episodes of this split\n{'rule':28s}" + "".join(f"{d[:6]:>8s}" for d in D) + "    mean")
    tab = {}
    for n, pick in rules.items():
        v = tab[n] = [miou([e for e in E if e["dataset"] == d], pick) if (ds == d).any() else np.nan for d in D]
        print(f"{n:28s}" + "".join(f"{x:8.2f}" for x in v) + f"{np.nanmean(v):8.2f}")
    best = np.array([tab[f"lam {l}"] for l in LAMS])
    print(f"{'best lam per domain (tuned)':28s}" + "".join(f"{x:8.2f}" for x in best.max(0)) + f"{np.nanmean(best.max(0)):8.2f}")
    print(f"{'  its lam':28s}" + "".join(f"{LAMS[i]:8d}" for i in best.argmax(0)))
