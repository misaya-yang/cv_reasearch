#!/usr/bin/env python3
"""Why RCG gains over FoRIS: which of its parts carries the gain, what it changes, and what error remains (CPU).

  ablate   on a cohort with cached features: the readout re-solved with one part removed or replaced
             foris_pre.control   the FoRIS score, bilinear, > 0.5 (no CRF): the input RCG starts from
             rerank_only         reference rank correction, no graph            smooth_only    graph, no rank correction
             rcg                 both (checked against the sealed field)        uniform_anchor every token anchored equally
             spatial_graph       the same solve on a 5 x 5 position graph instead of the feature graph
             spatial_smooth_only, spatial_lam1 / spatial_lam0.25 (weaker position smoothing), lam1 / lam4 / lam64, smooth_lam64, alpha1
  anatomy  on sealed runs (no features needed): every pixel RCG changes relative to the FoRIS pre-CRF mask, sorted into
             added:   hole (enclosed by the mask) / near (within 16 px of it) / far
             deleted: component (a connected part removed almost whole) / near (within 16 px of the edge) / interior
           each scored alone on the FoRIS mask; the error RCG and FoRIS leave, sorted into object-level and boundary-level;
           gains by object size, by how good FoRIS already was, and by batch.

Both modes read truth only to count. Nothing here is a method or a selection.

  python scripts/run_rcg_anatomy.py ablate --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_rcg_ablate_fresh600
  python scripts/run_rcg_anatomy.py anatomy --runs outputs/claude_official/run{0..6} --roots outputs/claude_official/root{0..6} --out outputs/claude_rcg_anatomy4000
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
VARIANTS = dict(rerank_only=(.5, 0., "feat", "conf"), smooth_only=(0., 16., "feat", "conf"), rcg=(.5, 16., "feat", "conf"), uniform_anchor=(.5, 16., "feat", "flat"),
                spatial_graph=(.5, 16., "grid", "conf"), spatial_smooth_only=(0., 16., "grid", "conf"), lam1=(.5, 1., "feat", "conf"), lam4=(.5, 4., "feat", "conf"),
                lam64=(.5, 64., "feat", "conf"), alpha1=(1., 16., "feat", "conf"), smooth_lam64=(0., 64., "feat", "conf"),
                spatial_lam1=(.5, 1., "grid", "conf"), **{"spatial_lam0.25": (.5, .25, "grid", "conf")})
ADD, DEL = ("hole", "near", "far"), ("component", "near", "interior")
LEFT = ("object", "within8", "within16", "body")


def bits(x):
    return np.unpackbits(x).reshape(1024, 1024).astype(bool)


def ablate_one(job):
    import torch
    import torch.nn.functional as F
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from scipy.stats import rankdata
    from ics.experiment import packet
    root, run, row = job; torch.set_num_threads(1)
    feat = torch.load(Path(root) / "cache/evidence_v1/feat" / (row["key"] + ".pt"))
    q, r = (F.normalize(feat[k].float(), dim=1) for k in ("q", "r"))
    with np.load(packet(root, row), allow_pickle=False) as z:
        score, cov, truth, native, pre = z["score"], z["cov"], bits(z["truth"]), bits(z["native"]), bits(z["pre"])
    with np.load(Path(run) / "fields" / (row["key"] + ".npz"), allow_pickle=False) as z:
        sealed = z["rcg"]
    rank = lambda x: ((rankdata(x.ravel(), method="average") - .5) / x.size).astype(np.float32)
    render = lambda f: F.interpolate(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32).reshape(1, 1, 64, 64)), (1024, 1024), mode="bilinear", align_corners=False)[0, 0].numpy() > .5
    fi = np.flatnonzero(cov.ravel() >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T; dq = sim.topk(10, dim=1).values.mean(1); dr = sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, fi] - dr[fi][None, :]).max(1).values - dq).numpy()
    s = np.asarray(score, np.float32); s = ((s - s.min()) / max(float(s.max() - s.min()), 1e-6)).ravel()
    sim = q @ q.T; sim.fill_diagonal_(-2); values, idx = sim.topk(20, dim=1); dist = (1 - values).clamp_min(0)
    w = sparse.csr_matrix((torch.exp(-dist / dist[:, -1:].clamp_min(1e-6)).numpy().ravel(), (np.repeat(np.arange(4096), 20), idx.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T); w.data = np.sqrt(w.data)
    yy, xx = np.divmod(np.arange(4096), 64); rows_, cols_ = [], []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ok = ((dy, dx) != (0, 0)) & (yy + dy >= 0) & (yy + dy < 64) & (xx + dx >= 0) & (xx + dx < 64)
            rows_.append(np.flatnonzero(ok)); cols_.append((yy + dy)[ok] * 64 + (xx + dx)[ok])
    g = sparse.csr_matrix((np.ones(sum(map(len, rows_))), (np.concatenate(rows_), np.concatenate(cols_))), shape=(4096, 4096))
    lap = {}
    for name, m in (("feat", w), ("grid", g)):
        m = m / max(float(np.asarray(m.sum(1)).mean()), 1e-8); lap[name] = sparse.diags(np.asarray(m.sum(1)).ravel()) - m
    conf = .1 + np.abs(2 * s - 1); conf = (conf / conf.mean()).astype(np.float64)
    masks, parity = {"native": native, "foris_pre.control": render(s), "foris_pre_sealed": pre}, 0.
    for name, (alpha, lam, graph, anchor) in VARIANTS.items():
        y = (s + alpha * (rank(guide) - rank(s))).astype(np.float64); a = conf if anchor == "conf" else np.ones(4096)
        if lam:
            y, status = cg(sparse.diags(a) + lam * lap[graph], a * y, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
        masks[name] = render(y)
        if name == "rcg":
            parity = float(np.abs(y.reshape(64, 64) - sealed).max())
    base = masks["foris_pre.control"]
    return ({k: (int((m & truth).sum()), int((m | truth).sum())) for k, m in masks.items()},
            {k: (int((m & ~base & truth).sum()), int((m & ~base & ~truth).sum()), int((base & ~m & ~truth).sum()), int((base & ~m & truth).sum())) for k, m in masks.items()}, parity)


def report(rows, arrays, extra, out, title, notes):
    from ics.experiment import summarize
    corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for k in arrays}
    s, _ = summarize(rows, arrays, corr)
    for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        s.pop(k, None)
    s.update(extra); out.mkdir(parents=True, exist_ok=True); (out / "report.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
    g = lambda k, b: "" if b not in s["contrasts"].get(k, {}) else "%+.2f [%+.2f, %+.2f]" % (s["contrasts"][k][b]["gain"], *s["contrasts"][k][b]["ci95"])
    bases = [b for b in ("native", "foris_pre.control", "rcg") if b in arrays]
    L = ["# " + title, "", "| arm | mIoU | " + " | ".join("vs " + b for b in bases) + " |", "|---|---:|" + "---|" * len(bases)]
    L += ["| %s | %.2f | %s |" % (k, s["scores"][k], " | ".join(g(k, b) for b in bases)) for k in arrays]
    (out / "report.md").write_text("\n".join(L + [""] + notes) + "\n"); print("\n".join(L + [""] + notes), flush=True)


def ablate(a):
    rows = json.loads((a.run / "manifest.json").read_text())[:a.limit]; begin = time.monotonic()
    with Pool(a.workers) as pool:
        got = pool.map(ablate_one, [(str(a.root), str(a.run), r) for r in rows], chunksize=4)
    names = list(got[0][0]); arrays = {k: np.array([g[0][k] for g in got], np.int64) for k in names}
    edits = {k: np.array([g[1][k] for g in got], np.int64).sum(0) for k in names}; parity = max(g[2] for g in got)
    notes = ["Edits against the FoRIS pre-CRF mask (million pixels): added true / added false / deleted false / deleted true", ""]
    notes += ["- %s: %.2f / %.2f / %.2f / %.2f" % (k, *(edits[k] / 1e6)) for k in names]
    notes += ["", "Largest difference between the re-solved RCG field and the sealed one: %.2g. Elapsed %.0f s." % (parity, time.monotonic() - begin)]
    report(rows, arrays, dict(edits_vs_pre={k: v.tolist() for k, v in edits.items()}, rcg_field_parity=parity), a.out, "RCG with one part removed or replaced: %d episodes, class mIoU at 1024" % len(rows), notes)
    print("ANATOMY_DONE", flush=True)


def anatomy_one(job):
    from scipy import ndimage as nd
    run, root, row = job
    with np.load(Path(root) / "results/extent_v1/run/packets" / (row["key"] + ".npz"), allow_pickle=False) as z:
        truth, native, pre = bits(z["truth"]), bits(z["native"]), bits(z["pre"])
    with np.load(Path(run) / "predictions" / (row["key"] + ".npz"), allow_pickle=False) as z:
        rcg = bits(z["RCG"])
    eight = np.ones((3, 3), bool); cnt = lambda m: (int((m & truth).sum()), int((m & ~truth).sum()))
    added, deleted = rcg & ~pre, pre & ~rcg; out = {}
    lab, _ = nd.label(~pre); edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    hole = ~pre & ~np.isin(lab, edge); near = nd.distance_transform_edt(~pre) <= 16 if pre.any() else np.zeros_like(pre)
    for k, m in zip(ADD, (added & hole, added & ~hole & near, added & ~hole & ~near)):
        out["add:" + k] = cnt(m)
    lab, n = nd.label(pre, eight)
    if n:
        size = np.bincount(lab.ravel(), minlength=n + 1); gone = np.bincount(lab[deleted], minlength=n + 1)
        whole = (gone >= .9 * size)[lab] & pre; rim = nd.distance_transform_edt(pre) <= 16
    else:
        whole = rim = np.zeros_like(pre)
    for k, m in zip(DEL, (deleted & whole, deleted & ~whole & rim, deleted & ~whole & ~rim)):
        out["del:" + k] = cnt(m)
    tl, tn = nd.label(truth, eight); d_in = nd.distance_transform_edt(truth) if truth.any() else np.full(truth.shape, np.inf); d_out = nd.distance_transform_edt(~truth)
    for name, m in (("native", native), ("pre", pre), ("rcg", rcg)):
        out["iu:" + name] = (int((m & truth).sum()), int((m | truth).sum()))
        hit = np.bincount(tl[m], minlength=tn + 1) > 0; hit[0] = True; lost = ~hit[tl]                       # truth parts the mask never touches
        ml, mn = nd.label(m, eight); ok = np.bincount(ml[truth], minlength=mn + 1) > 0; ok[0] = True; stray = ~ok[ml]   # mask parts that touch no truth
        miss, false = truth & ~m, m & ~truth
        out["missed:" + name] = [int(x.sum()) for x in (miss & lost, miss & ~lost & (d_in <= 8), miss & ~lost & (d_in > 8) & (d_in <= 16), miss & ~lost & (d_in > 16))]
        out["false:" + name] = [int(x.sum()) for x in (false & stray, false & ~stray & (d_out <= 8), false & ~stray & (d_out > 8) & (d_out <= 16), false & ~stray & (d_out > 16))]
    out["area"] = float(truth.mean()); out["parts"] = int(tn)
    return out


def anatomy(a):
    rows, jobs = [], []
    for b, (run, root) in enumerate(zip(a.runs, a.roots)):
        for r in json.loads((run / "manifest.json").read_text()):
            rows.append(dict(r, batch=str(b))); jobs.append((str(run), str(root), r))
    begin = time.monotonic()
    with Pool(a.workers) as pool:
        got = pool.map(anatomy_one, jobs, chunksize=8)
    A = {k: np.array([g[k] for g in got]) for k in got[0]}; a.out.mkdir(parents=True, exist_ok=True); np.savez_compressed(a.out / "counts.npz", **A)
    classes = np.array([r["c"] for r in rows]); batch = np.array([int(r["batch"]) for r in rows]); pre = A["iu:pre"].astype(np.int64)
    arrays = {"native": A["iu:native"], "foris_pre.control": pre, "rcg": A["iu:rcg"]}
    for k in ADD:                                            # one family of RCG's edits applied alone to the FoRIS mask
        arrays["pre+add_" + k] = pre + np.stack([A["add:" + k][:, 0], A["add:" + k][:, 1]], 1)
    for k in DEL:
        arrays["pre-del_" + k] = pre - np.stack([A["del:" + k][:, 0], A["del:" + k][:, 1]], 1)
    for name in ("pre", "rcg"):                              # GT ceilings: one family of the remaining error corrected alone
        iu = A["iu:" + name].astype(np.int64)
        for j, k in enumerate(LEFT):
            arrays["%s|fix_%s" % (name, k)] = iu + np.stack([A["missed:" + name][:, j], -A["false:" + name][:, j]], 1)

    def miou(x, ix):
        return float(np.mean([x[ix][classes[ix] == c, 0].sum() / max(x[ix][classes[ix] == c, 1].sum(), 1) for c in np.unique(classes[ix])]) * 100) if ix.any() else float("nan")
    notes = ["## What RCG changes against the FoRIS pre-CRF mask (million pixels: true / false; share true)", ""]
    for tag, names in (("add", ADD), ("del", DEL)):
        for k in names:
            t, f = A[tag + ":" + k].sum(0); notes.append("- %s %s: %.2f / %.2f (%.0f%% true)" % ("added" if tag == "add" else "deleted", k, t / 1e6, f / 1e6, 100 * t / max(t + f, 1)))
    notes += ["", "## Error left (million pixels): object-level / within 8 px of the true boundary / 8 to 16 px / further", ""]
    for name in ("native", "pre", "rcg"):
        notes.append("- %s: missed %s; false %s" % (name, " / ".join("%.1f" % (v / 1e6) for v in A["missed:" + name].sum(0)), " / ".join("%.1f" % (v / 1e6) for v in A["false:" + name].sum(0))))
    iou = pre[:, 0] / np.maximum(pre[:, 1], 1); tables = {}
    groups = {"object area (share of the image)": (A["area"], (0, .02, .1, .3, 1.01)), "FoRIS pre-CRF IoU on the episode": (iou, (0, 1e-9, .3, .6, .8, 1.01)), "batch": (batch, tuple(range(8)))}
    for title, (v, edges) in groups.items():
        notes += ["", "## By " + title, "", "| bin | n | FoRIS | FoRIS pre-CRF | RCG | RCG - pre | episodes better / worse |", "|---|---:|---:|---:|---:|---:|---|"]; tables[title] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            ix = (v >= lo) & (v < hi)
            if not ix.any():
                continue
            ri = A["iu:rcg"][ix, 0] / np.maximum(A["iu:rcg"][ix, 1], 1); row = [miou(arrays[k], ix) for k in ("native", "foris_pre.control", "rcg")]
            tables[title].append(dict(lo=lo, hi=hi, n=int(ix.sum()), scores=row, better=int((ri > iou[ix] + .01).sum()), worse=int((ri < iou[ix] - .01).sum())))
            notes.append("| [%g, %g) | %d | %.2f | %.2f | %.2f | %+.2f | %d / %d |" % (lo, hi, ix.sum(), *row, row[2] - row[1], tables[title][-1]["better"], tables[title][-1]["worse"]))
    notes += ["", "Elapsed %.0f s." % (time.monotonic() - begin)]
    report(rows, arrays, dict(tables=tables), a.out, "Anatomy of RCG against FoRIS: %d episodes, class mIoU at 1024" % len(rows), notes)
    print("ANATOMY_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("ablate"); s.add_argument("--root", type=Path, required=True); s.add_argument("--run", type=Path, required=True); s.add_argument("--limit", type=int)
    t = sub.add_parser("anatomy"); t.add_argument("--runs", type=Path, nargs="+", required=True); t.add_argument("--roots", type=Path, nargs="+", required=True)
    for x in (s, t):
        x.add_argument("--out", type=Path, required=True); x.add_argument("--workers", type=int, default=5)
    a = p.parse_args(); dict(ablate=ablate, anatomy=anatomy)[a.mode](a)


if __name__ == "__main__":
    main()
