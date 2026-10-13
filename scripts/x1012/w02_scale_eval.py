"""W02: large-n development scoreboard of the frozen independent candidate against complete FoRIS.

Candidate = R1 VP_UN exactly as frozen in r5_candidate.infer (imported unchanged, no pixel
geometry): min-max > .5 and the R5 final > .475, rho > .9 keeps the raw-T fallback.
Pairs = every cached explore / confirm pair that already has a complete FoRIS mask.
All these pairs are exposed; this is a measurement of the true per-domain gap and of how the
loss is distributed over episodes, not a confirmation.  No encoder call; GT is read only to score.

    python scripts/x1012/w02_scale_eval.py produce OUT [cap_per_domain] [workers]
    python scripts/x1012/w02_scale_eval.py report  OUT
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ASSETS = HERE.parents[1].parent / "cv_data"
DOMS = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid", "isic", "lung", "fundus", "deepglobe_road")


def rows_all(cap):
    P0 = ASSETS / "a/x1012_P0"
    dev = {str(b["index"]) for b in json.loads((ASSETS / "a/x1012_R1/manifest.json").read_text())}
    out = []
    for split in ("explore", "confirm"):
        for b in json.loads((P0 / f"{split}.json").read_text()):
            if "B" in b.get("old_arms", {}):
                b["_split"] = "dev1360" if str(b["index"]) in dev else split
                out.append(b)
    keep = []
    for d in DOMS:
        L = [b for b in out if b["dataset"] == d]
        if cap and len(L) > cap:                      # deterministic stride, labels not consulted
            L = [L[i] for i in (np.arange(cap) * len(L) / cap).astype(int)]
        keep += L
    return keep


def init(out):
    global rc, lab, torch
    import torch
    import r5_candidate as rc
    rc.initialize(out, "cpu", 2, "midrange", False)
    lab = rc.lab


def one(b):
    t = time.monotonic()
    pred, fields, detail = rc.infer(b)                # guarded: query GT cannot be opened here
    un50 = torch.from_numpy(rc.pc.unpack(pred["UN_identity"]).copy())
    rho = float(fields["reference_rho"][0])
    un475 = un50 if rho > .9 else lab.minmax(lab.up(torch.from_numpy(fields["VP_UN"]))) > .475
    gt = lab.mask1024(b["row"]["query_mask_path"])
    ig = lab.ignore1024(b)
    keep = None if ig is None else ~ig
    cut = lambda m: m if keep is None else m & keep
    g = cut(gt)
    rec = {"index": str(b["index"]), "dataset": b["dataset"], "split": b["_split"], "fold": b["row"]["fold"],
           "cls": b["row"]["global_class_id"], "photo": b["row"].get("query_photo_id") or b["row"].get("query_path"),
           "rho": rho, "gt": int(g.sum()), "sec": round(time.monotonic() - t, 3)}
    arms = {"foris": lab.arm_cli(b, "B"), "un50": un50, "un475": un475}
    if "G" in b["old_arms"]:
        arms["mean"] = lab.arm_cli(b, "G")
    for k, m in arms.items():
        m = cut(m)
        rec[k] = [int((m & g).sum()), int((m | g).sum()), int(m.sum())]
    return rec


if __name__ == "__main__":
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "produce":
        cap = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = "2"
        import r5_candidate as rc
        out.mkdir(parents=True, exist_ok=True)
        rc.prepare_runtime(out)
        rows = rows_all(cap)
        done = {}
        log = out / "episodes.jsonl"
        if log.exists():
            done = {json.loads(l)["index"] for l in log.read_text().splitlines() if l.strip()}
        rows = [b for b in rows if str(b["index"]) not in done]
        print("to run", len(rows), "already", len(done), flush=True)
        t0 = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init, initargs=(str(out),)) as pool, log.open("a") as fh:
            for i, rec in enumerate(pool.map(one, rows, chunksize=4), 1):
                fh.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    fh.flush()
                    print(i, len(rows), round(time.monotonic() - t0, 1), flush=True)
        print("done", round(time.monotonic() - t0, 1), flush=True)
    else:
        E = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines() if l.strip()]
        rng = np.random.default_rng(20261013)

        def miou(L, arm):
            I, U = {}, {}
            for e in L:
                k = (e["fold"], e["cls"])
                I[k] = I.get(k, 0) + e[arm][0]; U[k] = U.get(k, 0) + e[arm][1]
            return float(np.mean([100 * I[k] / max(U[k], 1) for k in I]))

        def boot(L, a, b, n=500):
            photos = {}
            for e in L:
                photos.setdefault(e["photo"], []).append(e)
            keys = list(photos)
            d = []
            for _ in range(n):
                S = [e for k in rng.choice(len(keys), len(keys)) for e in photos[keys[k]]]
                d.append(miou(S, a) - miou(S, b))
            return np.percentile(d, [2.5, 97.5])

        for scope in ("all", "not dev1360"):
            print(f"\n=== scope: {scope}; class-pooled mIoU, CLI-1024; paired photo bootstrap 95% CI")
            print(f"{'domain':15s}{'n':>6s}{'FoRIS':>8s}{'MEAN':>8s}{'UN .5':>8s}{'UN .475':>8s}{'d(.5)':>8s}{'95% CI':>18s}{'d(.475)':>9s}{'95% CI':>18s}")
            for d in DOMS:
                L = [e for e in E if e["dataset"] == d and (scope == "all" or e["split"] != "dev1360")]
                if not L:
                    continue
                f = miou(L, "foris")
                M = [e for e in L if "mean" in e]
                mm = f"{miou(M, 'mean') - miou(M, 'foris'):+8.2f}" if len(M) > 0.5 * len(L) else "       -"
                c5, c4 = boot(L, "un50", "foris"), boot(L, "un475", "foris")
                print(f"{d:15s}{len(L):6d}{f:8.2f}{mm}{miou(L, 'un50'):8.2f}{miou(L, 'un475'):8.2f}{miou(L, 'un50') - f:+8.2f}"
                      f"   [{c5[0]:+6.2f},{c5[1]:+6.2f}]{miou(L, 'un475') - f:+9.2f}   [{c4[0]:+6.2f},{c4[1]:+6.2f}]")
        print("\n=== by GT area (per-episode mean IoU, UN .5 minus FoRIS) and loss concentration")
        print(f"{'domain':15s}{'<2% n':>7s}{'d':>7s}{'2-20% n':>9s}{'d':>7s}{'>20% n':>8s}{'d':>7s}{'worse/better':>14s}{'top5% share of extra error px':>32s}")
        for d in DOMS:
            L = [e for e in E if e["dataset"] == d]
            if not L:
                continue
            iou = lambda e, a: 100 * e[a][0] / max(e[a][1], 1)
            cells = []
            for lo, hi in ((0, .02), (.02, .2), (.2, 2)):
                S = [e for e in L if lo <= e["gt"] / 1048576 < hi]
                cells.append(f"{len(S):7d}{(np.mean([iou(e, 'un50') - iou(e, 'foris') for e in S]) if S else float('nan')):+7.1f}")
            err = lambda e, a: e[a][1] - e[a][0]
            ex = np.array([err(e, "un50") - err(e, "foris") for e in L], float)
            pos = ex[ex > 0]
            top = np.sort(pos)[::-1][: max(1, int(0.05 * len(L)))].sum() / max(pos.sum(), 1)
            print(f"{d:15s}{cells[0]}{'  ' + cells[1]}{' ' + cells[2]}{int((ex > 0).sum()):>8d}/{int((ex < 0).sum()):<5d}{top:>20.2f}   net {ex.sum():+.3g}")
