#!/usr/bin/env python3
"""An adder and a deleter from the query's own grouping, applied to the RCG mask.

Query tokens are grouped by average-linkage clustering of the cached features (cosine, threshold tau), as INSID3 and
FoRIS group them. A group mostly inside the mask is completed (its remaining tokens are added); a group mostly
outside loses the tokens it has inside. Additions lie outside R and deletions inside R, so every pair is exact from
counts. Controls: the same number of pixels added from the highest, or deleted from the lowest, RCG scores.
`infer` reads no truth (CPU pool); `score` (GPU) nests the choice over folds. Development data.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TAUS = (.5, .6, .7)
ADD_COVER = (.3, .4, .5, .6, .7, .8)
ADD_FIELD = (-1., .2, .3, .4)                 # minimum mean RCG score of the tokens to be added
DEL_COVER = (.05, .1, .2, .3, .4)
DEL_FIELD = (2., .7, .6)                      # the tokens removed must have a mean RCG score below this


def one(job):
    key, feat_path, out = job
    import torch
    from sklearn.cluster import AgglomerativeClustering
    torch.set_num_threads(1); begin = time.perf_counter()
    q = torch.load(feat_path, map_location="cpu", weights_only=True)["q"].float().numpy()
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    dist = 1 - np.clip(q @ q.T, -1, 1); np.fill_diagonal(dist, 0); labels = {}
    for tau in TAUS:
        labels[f"tau{tau:g}"] = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1 - tau).fit_predict(dist).astype(np.int16)
    np.savez_compressed(Path(out) / "labels" / f"{key}.npz", **labels)
    return key, {k: int(v.max()) + 1 for k, v in labels.items()}, time.perf_counter() - begin


def infer(a):
    import multiprocessing as mp
    rows = json.loads((a.recheck / "manifest.json").read_text())
    a.out.mkdir(parents=True, exist_ok=False); (a.out / "labels").mkdir()
    jobs = [(r["key"], str(a.root / "cache/evidence_v1/feat" / f"{r['key']}.pt"), str(a.out)) for r in rows]; audit = {}
    with mp.get_context("spawn").Pool(a.workers) as p:
        for n, (key, counts, sec) in enumerate(p.imap_unordered(one, jobs), 1):
            audit[key] = counts
            if n % 25 == 0 or n == len(jobs):
                print(json.dumps(dict(completed=n, total=len(jobs), seconds=round(sec, 1))), flush=True)
    (a.out / "manifest.json").write_text(json.dumps(rows)); (a.out / "audit.json").write_text(json.dumps(audit))
    (a.out / "sealed.json").write_text(json.dumps(dict(state="ALL_PREDICTIONS_SEALED", taus=TAUS, query_gt_in_inference=False)))
    print("ALL_PREDICTIONS_SEALED", flush=True)


def score(a):
    import torch
    import torch.nn.functional as F
    from ics.experiment import metric, packet, photo_groups, summarize, unpack
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = json.loads((a.out / "manifest.json").read_text()); n = len(rows)
    adds = list(itertools.product(TAUS, ADD_COVER, ADD_FIELD)); dels = list(itertools.product(TAUS, DEL_COVER, DEL_FIELD))
    base, native = np.zeros((n, 2), np.int64), np.zeros((n, 2), np.int64)
    A, Ac, D, Dc = (np.zeros((len(x), n, 2), np.int64) for x in (adds, adds, dels, dels))
    up = lambda x: F.interpolate(x.reshape(1, 1, 64, 64).float(), (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
    for i, row in enumerate(rows):
        key = row["key"]
        with np.load(a.recheck / "predictions" / f"{key}.npz", allow_pickle=False) as z:
            R = torch.from_numpy(unpack(z["RCG"])).to(dev)
        with np.load(a.recheck / "fields" / f"{key}.npz", allow_pickle=False) as z:
            field = torch.from_numpy(z["rcg"]).to(dev).reshape(4096)
        with np.load(packet(a.root, row), allow_pickle=False) as z:
            t = torch.from_numpy(unpack(z["truth"])).to(dev); nat = torch.from_numpy(unpack(z["native"])).to(dev)
        with np.load(a.out / "labels" / f"{key}.npz", allow_pickle=False) as z:
            labels = {tau: torch.from_numpy(z[f"tau{tau:g}"].astype(np.int64)).to(dev) for tau in TAUS}
        base[i] = int((R & t).sum()), int((R | t).sum()); native[i] = int((nat & t).sum()), int((nat | t).sum())
        cover = R.reshape(64, 16, 64, 16).float().mean((1, 3)).reshape(4096); inside = cover >= .5; zup = up(field)
        out_sorted = torch.cat([torch.zeros(1, device=dev), t[~R][torch.argsort(-zup[~R], stable=True)].float().cumsum(0)])
        in_sorted = torch.cat([torch.zeros(1, device=dev), t[R][torch.argsort(zup[R], stable=True)].float().cumsum(0)])
        stats = {}
        for tau, lab in labels.items():
            k = int(lab.max()) + 1; size = torch.bincount(lab, minlength=k).float()
            frac = torch.bincount(lab, weights=cover, minlength=k) / size
            out_mean = torch.bincount(lab, weights=field * (~inside), minlength=k) / torch.bincount(lab, weights=(~inside).float(), minlength=k).clamp_min(1)
            in_mean = torch.bincount(lab, weights=field * inside, minlength=k) / torch.bincount(lab, weights=inside.float(), minlength=k).clamp_min(1)
            stats[tau] = (lab, frac, out_mean, in_mean)
        for j, (tau, c, fmin) in enumerate(adds):
            lab, frac, out_mean, _ = stats[tau]; new = (up(((frac >= c) & (out_mean > fmin))[lab] & ~inside) > .5) & ~R; k = int(new.sum())
            A[j, i] = int((new & t).sum()), int((new & ~t).sum()); ct = int(out_sorted[k]); Ac[j, i] = ct, k - ct
        for j, (tau, c, fmax) in enumerate(dels):
            lab, frac, _, in_mean = stats[tau]; gone = (up(((frac <= c) & (in_mean < fmax))[lab] & inside) > .5) & R; k = int(gone.sum())
            D[j, i] = int((gone & t).sum()), int((gone & ~t).sum()); ct = int(in_sorted[k]); Dc[j, i] = ct, k - ct
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(scored=i + 1, total=n)), flush=True)
    np.savez_compressed(a.out / "counts.npz", A=A, Ac=Ac, D=D, Dc=Dc, base=base, native=native)
    classes, folds, groups = np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows]), photo_groups(rows)
    zero = np.zeros((1, n, 2), np.int64); A_, Ac_, D_, Dc_ = (np.concatenate([zero, x]) for x in (A, Ac, D, Dc))
    aname = ["none"] + [f"add[tau{t:g},cover>={c:g},field>{f:g}]" for t, c, f in adds]; dname = ["none"] + [f"del[tau{t:g},cover<={c:g},field<{f:g}]" for t, c, f in dels]
    onehot = (classes[:, None] == np.unique(classes)[None]).astype(np.float64)
    def table(dd, aa, ix):
        M = onehot[ix]; M = M[:, M.sum(0) > 0]; b = base[ix].astype(np.float64)
        I = (b[:, 0] @ M)[None, None] - (dd[:, ix, 0] @ M)[:, None] + (aa[:, ix, 0] @ M)[None]
        U = (b[:, 1] @ M)[None, None] - (dd[:, ix, 1] @ M)[:, None] + (aa[:, ix, 1] @ M)[None]
        return 100 * (I / np.maximum(U, 1)).mean(-1)
    compose = lambda dd, aa, d, j: np.stack([base[:, 0] - dd[d, :, 0] + aa[j, :, 0], base[:, 1] - dd[d, :, 1] + aa[j, :, 1]], 1)
    def nested(dd, aa, d_ok, a_ok):
        out, chosen = np.zeros((n, 2), np.int64), {}
        for f in sorted(set(folds)):
            read = folds == f; fit = ~read & ~np.isin(groups, groups[read]); tab = table(dd[d_ok], aa[a_ok], fit)
            d, j = np.unravel_index(int(tab.argmax()), tab.shape); d, j = d_ok[d], a_ok[j]
            out[read] = compose(dd, aa, d, j)[read]; chosen[str(f)] = [dname[d], aname[j]]
        return out, chosen
    everything = np.ones(n, bool); all_d, all_a = np.arange(len(dname)), np.arange(len(aname)); full = table(D_, A_, everything); ctl = table(Dc_, Ac_, everything)
    arrays = {"native": native, "rcg": base}; selection = {}
    for name, (dd, aa, d_ok, a_ok) in {"add.nested": (D_, A_, all_d[:1], all_a), "delete.nested": (D_, A_, all_d, all_a[:1]), "both.nested": (D_, A_, all_d, all_a),
                                      "both.same_count.control": (Dc_, Ac_, all_d, all_a)}.items():
        arrays[name], selection[name] = nested(dd, aa, d_ok, a_ok)
    bd, ba = np.unravel_index(int(full.argmax()), full.shape); arrays["best_in_sample"] = compose(D_, A_, bd, ba)
    corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for k in arrays}
    s, _ = summarize(rows, arrays, corr)
    for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        s.pop(k, None)
    pool = lambda x: [int(v) for v in x.sum(0)]
    out = dict(n=n, summary=s, selection=selection, best_in_sample=[dname[bd], aname[ba], float(full[bd, ba])],
               additions=[dict(name=aname[j], miou=float(full[0, j]), control=float(ctl[0, j]), true_false=pool(A_[j])) for j in all_a],
               deletions=[dict(name=dname[d], miou=float(full[d, 0]), control=float(ctl[d, 0]), true_false=pool(D_[d])) for d in all_d])
    (a.out / "report.json").write_text(json.dumps(out, indent=2) + "\n")
    L = [f"# Grouping-based adder and deleter on RCG: {n} episodes, RCG {s['scores']['rcg']:.2f}, FoRIS {s['scores']['native']:.2f}", "", "| arm | mIoU | vs native | vs rcg |", "|---|---:|---|---|"]
    for k in arrays:
        c = s["contrasts"][k]; f = lambda b: f"{c[b]['gain']:+.2f} [{c[b]['ci95'][0]:+.2f}, {c[b]['ci95'][1]:+.2f}]" if b in c else ""
        L.append(f"| {k} | {s['scores'][k]:.2f} | {f('native')} | {f('rcg')} |")
    L += ["", f"Chosen: {json.dumps(selection)}", "", f"Best in sample: {out['best_in_sample']}", "", "## Additions alone (no selection)", "", "| variant | mIoU | same-count control | added true | added false | purity |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(out["additions"], key=lambda r: -r["miou"])[:25]:
        L.append(f"| {r['name']} | {r['miou']:.2f} | {r['control']:.2f} | {r['true_false'][0]:,} | {r['true_false'][1]:,} | {r['true_false'][0] / max(sum(r['true_false']), 1):.2f} |")
    L += ["", "## Deletions alone (no selection)", "", "| variant | mIoU | same-count control | removed true | removed false | share false |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(out["deletions"], key=lambda r: -r["miou"])[:25]:
        L.append(f"| {r['name']} | {r['miou']:.2f} | {r['control']:.2f} | {r['true_false'][0]:,} | {r['true_false'][1]:,} | {r['true_false'][1] / max(sum(r['true_false']), 1):.2f} |")
    (a.out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:48]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("infer", "score")); p.add_argument("--root", type=Path, required=True); p.add_argument("--recheck", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(); (infer if a.mode == "infer" else score)(a)


if __name__ == "__main__":
    main()
