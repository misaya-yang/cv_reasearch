"""W03: in which task regime does the frozen independent candidate gain over complete FoRIS?

Reads the W02 per-episode I/U (13,613 exposed pairs) and adds geometry descriptors of the
reference mask (legal at inference) and of the query GT (diagnostic only): area fraction,
width estimate 4*mean(distance transform) in 1024-frame pixels, connected components.
No encoder call, no new prediction.

    python scripts/x1012/w03_reference_regime.py produce OUT [workers]
    python scripts/x1012/w03_reference_regime.py report  OUT
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EVID = HERE.parents[1] / "evidence/research/w02_scale13613_iu.jsonl.gz"
DOMS = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road")


def geom(m):
    from scipy import ndimage
    m = m.astype(bool)
    a = int(m.sum())
    if a == 0:
        return 0.0, 0.0, 0
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    c = np.pad(m[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], 1)      # image border counts as background
    dt = ndimage.distance_transform_edt(c)
    n = ndimage.label(c, structure=np.ones((3, 3)))[1]
    return a / m.size, float(4 * dt[c].mean()), int(n)


def one(b):
    import lab
    r = geom(lab.mask1024(b["row"]["reference_mask_path"]).cpu().numpy())
    q = geom(lab.mask1024(b["row"]["query_mask_path"]).cpu().numpy())
    return {"index": str(b["index"]), "name": str(b["row"]["global_class_id"]),
            "ra": r[0], "rw": r[1], "rn": r[2], "qa": q[0], "qw": q[1], "qn": q[2]}


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "produce":
        import w02_scale_eval as w
        out.mkdir(parents=True, exist_ok=True)
        rows = w.rows_all(0)
        with ProcessPoolExecutor(max_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6,
                                 mp_context=multiprocessing.get_context("spawn")) as pool, (out / "geom.jsonl").open("w") as fh:
            for i, rec in enumerate(pool.map(one, rows, chunksize=16), 1):
                fh.write(json.dumps(rec) + "\n")
                if i % 2000 == 0:
                    print(i, len(rows), flush=True)
        print("done", flush=True)
        sys.exit()

    G = {json.loads(l)["index"]: json.loads(l) for l in (out / "geom.jsonl").read_text().splitlines()}
    E = [dict(json.loads(l)) for l in gzip.open(EVID, "rt")]
    for e in E:
        e.update(G[e["index"]])
    rng = np.random.default_rng(20261013)
    iou = lambda e, a: 100 * e[a][0] / max(e[a][1], 1)

    def miou(L, arm):
        I, U = {}, {}
        for e in L:
            k = (e["dataset"], e["fold"], e["cls"])
            I[k] = I.get(k, 0) + e[arm][0]; U[k] = U.get(k, 0) + e[arm][1]
        return float(np.mean([100 * I[k] / max(U[k], 1) for k in I]))

    def cell(L, n=300):
        """n, class-pooled FoRIS, class-pooled diff, photo-bootstrap CI, per-episode mean diff."""
        if len(L) < 30:
            return f"{len(L):6d}" + " " * 44
        photos = {}
        for e in L:
            photos.setdefault(e["photo"], []).append(e)
        keys = list(photos)
        d = []
        for _ in range(n):
            S = [e for k in rng.choice(len(keys), len(keys)) for e in photos[keys[k]]]
            d.append(miou(S, "un50") - miou(S, "foris"))
        lo, hi = np.percentile(d, [2.5, 97.5])
        f = miou(L, "foris")
        ep = np.mean([iou(e, "un50") - iou(e, "foris") for e in L])
        return f"{len(L):6d}{f:7.1f}{miou(L, 'un50') - f:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{ep:+7.2f}"

    head = f"{'n':>6s}{'FoRIS':>7s}{'d pool':>7s}{'95% CI':>16s}{'d epis':>7s}"

    def table(title, bins, key, groups):
        print(f"\n=== {title}\n{'':22s}" + head)
        for gname, doms in groups:
            for lab_, lo, hi in bins:
                L = [e for e in E if e["dataset"] in doms and lo <= key(e) < hi]
                if L:
                    print(f"{gname:12s}{lab_:10s}" + cell(L))

    GROUPS = [("part", ("pascal_part", "paco_part")), ("object", ("coco", "lvis")), ("scene", ("suim",)),
              ("aerial", ("isaid", "deepglobe_road")), ("medical", ("isic", "lung", "fundus")), ("ALL", DOMS)]
    EACH = [(d, (d,)) for d in DOMS] + [("ALL", DOMS), ("ALL-fundus", tuple(d for d in DOMS if d != "fundus"))]
    W = [("<8px", 0, 8), ("8-16", 8, 16), ("16-32", 16, 32), ("32-64", 32, 64), ("64-128", 64, 128), (">128", 128, 1e9)]
    A = [("<0.5%", 0, .005), ("0.5-2%", .005, .02), ("2-8%", .02, .08), ("8-20%", .08, .2), (">20%", .2, 2)]
    table("by task family", [("all", 0, 9)], lambda e: 1, GROUPS)
    table("by REFERENCE width estimate (legal at inference)", W, lambda e: e["rw"], EACH)
    table("by REFERENCE area fraction (legal at inference)", A, lambda e: e["ra"], EACH)
    table("by QUERY GT width estimate (diagnostic)", W, lambda e: e["qw"], EACH)

    print("\n=== class level: classes ranked by median reference width (class property), pooled over domains with class names")
    cls = {}
    for e in E:
        cls.setdefault((e["dataset"], e["name"]), []).append(e)
    for d in ("lvis", "paco_part", "pascal_part", "coco", "isaid", "suim"):
        C = [(np.median([e["rw"] for e in L]), k, L) for k, L in cls.items() if k[0] == d and len(L) >= 4]
        for lab_, lo, hi in W:
            S = [e for w_, k, L in C if lo <= w_ < hi for e in L]
            nc = sum(lo <= w_ < hi for w_, k, L in C)
            if S:
                print(f"{d:12s}{lab_:7s}{nc:4d}cls" + cell(S))
    if len(sys.argv) > 3:
        json.dump(E, open(sys.argv[3], "w"))
