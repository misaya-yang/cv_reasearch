"""W10: does the candidate's ranking, cut at FoRIS's extent, beat FoRIS on final masks in every domain?

Motivation: W07 (dev1360, per-episode) and W09 (selection half, every size tier) show the candidate's
field ranks pixels better than the FoRIS score at equal area, while its own cut (min-max > .5) is
worse than FoRIS's on targets above 25% of the image -- the tier that holds most GT pixels.  W09
closed the solve's integration length as a way to fix the cut with a legal selector.  The direct
consequence to test at scale and in the class-pooled metric: keep our ranking, take the extent from
complete FoRIS (k = area of its final mask).  This consumes the FoRIS output, so it is a
composition, not an independent method.

Pre-stated arms, rho <= .9 only (rho > .9 keeps the candidate's fallback, as in T0):
  U@F     top-k of minmax(up(VP_UN)) with k = area of the complete FoRIS mask
  U@gm    k = sqrt(area_FoRIS * area_candidate)   (two extent estimates averaged in log space)
  U@Fraw  k = area of FoRIS without CRF (min-max > .5 of its score)
  U@gmraw k = sqrt(area_FoRIS-without-CRF * area_candidate)
Prediction: U@F >= FoRIS in all ten domains on the selection half.  If it fails in the large-target
domains (COCO, ISIC, SUIM), the equal-area ranking advantage does not survive pooling and the
composition is dropped.

    python scripts/x1012/w10_area_transfer.py produce OUT [workers] [S|T|ALL]
    python scripts/x1012/w10_area_transfer.py report  OUT
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
A = HERE.parents[2] / "cv_data/a"
D = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road")
META = {}


def topk(u, k, gt):
    k = int(min(max(k, 1), u.size - 1))
    m = u >= np.partition(u.ravel(), u.size - k)[u.size - k]
    return [int((m & gt).sum()), int((m | gt).sum()), int(m.sum())]


def one(job):
    b, e = job
    import torch, lab
    lab.DEV = torch.device("cpu"); torch.set_num_threads(1)
    i = str(b["index"])
    z = np.load(A / "x1013_T0/fields" / ((f"{int(i):06d}" if i.isdigit() else i) + ".npz"))
    if float(z["reference_rho"][0]) > .9:
        return None
    gt = lab.mask1024(b["row"]["query_mask_path"]).cpu().numpy().astype(bool)
    mm = lambda f: lab.minmax(lab.up(torch.as_tensor(np.asarray(f, np.float32)).reshape(64, 64))).numpy()
    u, aF, aU = mm(z["VP_UN"]), e["foris"][2], e["candidate"][2]
    r = {"index": i, "U@F": topk(u, aF, gt), "U@gm": topk(u, (aF * max(aU, 1)) ** .5, gt)}
    try:
        raw = mm(np.load(b["score64"]["path"])[b["score64"]["key"]]) > .5
        r["Fraw"] = [int((raw & gt).sum()), int((raw | gt).sum()), int(raw.sum())]
        r["U@Fraw"] = topk(u, r["Fraw"][2], gt)
        r["U@gmraw"] = topk(u, (r["Fraw"][2] * max(aU, 1)) ** .5, gt)
    except Exception:
        pass
    return r


def load_meta(split):
    return {e["index"]: e for e in map(json.loads, gzip.open(A / "x1013_T0/iu.jsonl.gz", "rt")) if split == "ALL" or e["split"] == split}


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "produce":
        import w02_scale_eval as w
        out.mkdir(parents=True, exist_ok=True)
        M = load_meta(sys.argv[4] if len(sys.argv) > 4 else "S")
        jobs = [(b, M[str(b["index"])]) for b in w.rows_all(0) if str(b["index"]) in M]
        with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 3,
                                 mp_context=multiprocessing.get_context("spawn")) as pool, (out / "episodes.jsonl").open("w") as fh:
            for n, rec in enumerate(pool.map(one, jobs, chunksize=8), 1):
                if rec:
                    fh.write(json.dumps(rec) + "\n")
                if n % 1000 == 0:
                    print(n, len(jobs), flush=True)
        print("done", flush=True)
        sys.exit()

    R = {r["index"]: r for r in map(json.loads, (out / "episodes.jsonl").read_text().splitlines())}
    M = load_meta("ALL")
    split = {M[i]["split"] for i in R}
    E = [dict(e, **R.get(i, {})) for i, e in M.items() if e["split"] in split]        # rho > .9 episodes: candidate only
    rng = np.random.default_rng(20261013)
    PX = 1024 * 1024
    arm = lambda a: (lambda e: e[a] if a in e else e["candidate"])
    big = lambda a, key, th: (lambda e: e[a] if a in e and e[key][2] / PX > th else e["candidate"])
    rules = {"candidate": arm("candidate"), "U@F": arm("U@F"), "U@gm": arm("U@gm"), "U@Fraw": arm("U@Fraw"), "U@gmraw": arm("U@gmraw"),
             "FoRIS without CRF": lambda e: e.get("Fraw", e["foris"]),
             "U@F if FoRIS area >25%": big("U@F", "foris", .25), "U@F if FoRIS area >10%": big("U@F", "foris", .10),
             "U@F if FoRIS area > ours": lambda e: e["U@F"] if "U@F" in e and e["foris"][2] > e["candidate"][2] else e["candidate"]}

    def miou(L, pick, w=None):
        I, U = {}, {}
        for j, e in enumerate(L):
            k, x, c = (e["fold"], e["cls"]), pick(e), 1 if w is None else w[j]
            I[k] = I.get(k, 0) + c * x[0]; U[k] = U.get(k, 0) + c * x[1]
        return float(np.mean([100 * I[k] / U[k] for k in I if U[k] > 0]))

    print("split", split, "n", len(E), "of which rho<=.9", len(R))
    for refname, ref in (("complete FoRIS", lambda e: e["foris"]), ("FoRIS without CRF", rules["FoRIS without CRF"])):
        print(f"\n## class-pooled diff vs {refname} (* = photo-bootstrap 95% CI excludes 0)\n{'rule':26s}"
              + "".join(f"{d[:6]:>9s}" for d in D) + "    mean  neg*")
        out_ = {n: [] for n in rules}
        for d in D:
            L = [e for e in E if e["dataset"] == d]
            ph = {k: i for i, k in enumerate(sorted({e["photo"] for e in L}))}
            p = np.array([ph[e["photo"]] for e in L])
            W = [np.bincount(rng.integers(0, len(ph), len(ph)), minlength=len(ph))[p] for _ in range(300)]
            base, bw = miou(L, ref), [miou(L, ref, w) for w in W]
            for n, pick in rules.items():
                lo, hi = np.percentile([miou(L, pick, w) - x for w, x in zip(W, bw)], [2.5, 97.5])
                out_[n].append((miou(L, pick) - base, -1 if hi < 0 else int(lo > 0)))
        for n in rules:
            print(f"{n:26s}" + "".join(f"{v:+8.2f}{'*' if s else ' '}" for v, s in out_[n])
                  + f"{np.mean([v for v, _ in out_[n]]):+8.2f}{sum(s < 0 for _, s in out_[n]):5d}")
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    Rr = [e for e in E if "U@F" in e]
    v = np.array([e["foris"][2] / PX for e in Rr])
    print(f"\n## per-episode IoU by FoRIS predicted area (legal for a composition), rho <= .9\n{'bin':>8s}{'n':>6s}{'FoRIS':>8s}{'cand':>8s}{'U@F':>8s}{'U@gm':>8s}")
    for nm, lo, hi in [("<1%", 0, .01), ("1-3%", .01, .03), ("3-10%", .03, .1), ("10-25%", .1, .25), ("25-50%", .25, .5), (">50%", .5, 2)]:
        S = [e for e, x in zip(Rr, v) if lo <= x < hi]
        print(f"{nm:>8s}{len(S):6d}" + "".join(f"{np.mean([iou(e[a]) for e in S]):8.1f}" for a in ("foris", "candidate", "U@F", "U@gm")))
