#!/usr/bin/env python3
"""The Astra `external_mean__delete` candidate replayed exactly, then swept: which query tokens outside the mask serve
as background evidence, and whether the same evidence can also add.

  R, S, T, C and the candidate come unchanged from the delivered modules (--astra); `infer` reads no truth (CPU pool).
  `sweep` (GPU) rebuilds the candidate's last step from tokens and varies it. A deletion variant removes pixels of C;
  an addition variant adds pixels outside R; the two never overlap, so every pair is exact from counts.
    deletion  bank of background tokens x leave-out window x margin threshold
    addition  bank x reach from the mask x margin threshold x minimum RCG score
    banks     astra (top-K outside by RCG score, K = tokens removed by C), top fractions, all, far (lower half),
              split:* (the top candidates that are closer to the far background than to the foreground mean; the
              others are treated as recoverable target instead of background)
  Controls: the same number of deletions taken from the lowest RCG scores of C; the same number of additions taken
  from the highest RCG scores of the same zone. Selection is nested over folds; all episodes are development data.
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

SAVED = ("RCG", "RCG_count_matched_delete", "conservative_delete", "external_mean__delete", "external_mean_delete_sameK_RCG", "MEAN_CONTROL")
FRACTIONS = (.02, .05, .1, .2)
DEL_BANKS = ("astra", "p", "all", "far") + tuple(f"top{f:g}" for f in FRACTIONS) + ("split:astra", "split:p", "split:top0.05", "split:top0.1")
DEL_WINDOWS = (0, 5, 9)
DEL_T = (-.05, 0., .05)
ADD_BANKS = ("far", "all", "split:astra", "split:top0.1")
ADD_REACH = (1, 2, 3, 0)                                   # 0 = anywhere
ADD_T = (0., .05, .1, .15, .2)
ADD_Z = (-1., .3, .4)
ASTRA = ("astra", 5, 0.)
PROPOSED = (("split:astra", 5, 0.), ("far", 2, .1, -1.))   # declared before any result of this sweep


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def one(job):
    key, feat_path, packet_path, astra, out = job
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, astra)
    import adaptive_deletion
    import components
    import query_background
    begin = time.perf_counter()
    feat = torch.load(feat_path, map_location="cpu", weights_only=True)
    q, r = feat["q"].float().numpy(), feat["r"].float().numpy()
    with np.load(packet_path, allow_pickle=False) as p:                      # truth, native and pre are never indexed
        cov, score, fg, bg = p["cov"].copy(), p["score"].copy(), p["fg_max"].copy(), p["bg_max"].copy()
    base, fields, _ = components.predict_all(q, r, cov, score)
    added, budget = adaptive_deletion.predict(base, fields, fg, bg); base["RCG_count_matched_delete"] = added["RCG_count_matched_delete"]
    qb, qi = query_background.predict(q, base, fields); base.update(qb)
    np.savez_compressed(Path(out) / "predictions" / f"{key}.npz", **{k: base[k] for k in SAVED})
    np.savez_compressed(Path(out) / "fields" / f"{key}.npz", rcg=np.asarray(fields["RCG_field"], np.float32).reshape(64, 64))
    return key, dict(K=budget["explicit_deletions"], recheck=qi, seconds=time.perf_counter() - begin)


def infer(a):
    import multiprocessing as mp
    from ics.experiment import load_rows, packet, sha
    rows = load_rows(a.manifest)
    if len(rows) != a.expected:
        raise ValueError(f"{len(rows)} episodes, expected {a.expected}")
    a.out.mkdir(parents=True, exist_ok=False); (a.out / "predictions").mkdir(); (a.out / "fields").mkdir()
    jobs = [(r["key"], str(a.root / "cache/evidence_v1/feat" / f"{r['key']}.pt"), str(packet(a.root, r)), str(a.astra.resolve()), str(a.out)) for r in rows]
    audit = {}
    with mp.get_context("spawn").Pool(a.workers) as p:
        for n, (key, info) in enumerate(p.imap_unordered(one, jobs), 1):
            audit[key] = info; print(json.dumps(dict(completed=n, total=len(jobs), key=key, seconds=round(info["seconds"], 1))), flush=True)
    write_json(a.out / "manifest.json", rows); write_json(a.out / "audit.json", audit)
    write_json(a.out / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED", manifest_sha256=sha(a.out / "manifest.json"), arms=list(SAVED),
               predictions={r["key"]: sha(a.out / "predictions" / f"{r['key']}.npz") for r in rows},
               fields={r["key"]: sha(a.out / "fields" / f"{r['key']}.npz") for r in rows},
               astra_sources={p.name: sha(p) for p in sorted(a.astra.glob("*.py"))}, query_gt_in_inference=False))
    print("ALL_PREDICTIONS_SEALED", flush=True)


def sweep(a):
    global DEL_BANKS, DEL_WINDOWS, DEL_T, ADD_BANKS, ADD_REACH, ADD_T, ADD_Z
    if a.grid == "wide":
        DEL_BANKS = DEL_BANKS + ("p0.5", "p2", "p4"); DEL_WINDOWS = (0, 5); DEL_T = (-.1, -.05, -.02, 0., .02, .05)
        ADD_BANKS = ("all", "far", "p", "top0.2"); ADD_REACH = (1, 2, 4, 0); ADD_T = (.1, .15, .2, .25, .3, .35, .4, .5); ADD_Z = (-1., .3, .4, .45, .48)
    import torch
    import torch.nn.functional as F
    from ics.experiment import metric, packet, photo_groups, sha, summarize, unpack
    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    seal = json.loads((a.out / "sealed.json").read_text()); rows = json.loads((a.out / "manifest.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(a.out / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Not sealed")
    dels = list(itertools.product(DEL_BANKS, DEL_WINDOWS, DEL_T)); adds = list(itertools.product(ADD_BANKS, ADD_REACH, ADD_T, ADD_Z))
    n = len(rows); fixed = ("native", "rcg", "c", "astra", "astra_sameK", "mean")
    iu = {k: np.zeros((n, 2), np.int64) for k in fixed}
    d_rem, d_ctl = np.zeros((len(dels), n, 2), np.int64), np.zeros((len(dels), n, 2), np.int64)     # removed true, removed false
    a_add, a_ctl = np.zeros((len(adds), n, 2), np.int64), np.zeros((len(adds), n, 2), np.int64)     # added true, added false
    replay = np.zeros(n, np.int64)
    t2g = lambda m: torch.from_numpy(m).to(dev)
    up = lambda x: F.interpolate(x.reshape(-1, 1, 64, 64).float(), (1024, 1024), mode="bilinear", align_corners=False)[:, 0]
    cover = lambda m: m.reshape(64, 16, 64, 16).float().mean((1, 3)).reshape(4096)
    def box(x, w):
        t = x.reshape(64, 64, -1).permute(2, 0, 1)[None]
        return F.avg_pool2d(t, w, 1, w // 2, divisor_override=1)[0].permute(1, 2, 0).reshape(4096, -1)
    def centroid(q, bank, w):
        b = bank.double(); sums = q * b[:, None]; remain = sums.sum(0)[None].expand(4096, -1); count = b.sum().expand(4096)
        if w:
            remain = remain - box(sums, w); count = count - box(b, w)[:, 0]
        norm = remain.norm(dim=1); valid = (count > .5) & (norm > 1e-12)
        return ((q * remain).sum(1) / norm.clamp_min(1e-12)).float(), valid
    def top(order, k):
        m = torch.zeros(4096, dtype=torch.bool, device=dev); m[order[:int(k)]] = True; return m
    begin = time.perf_counter()
    for i, row in enumerate(rows):
        key = row["key"]
        for group in ("predictions", "fields"):
            if sha(a.out / group / f"{key}.npz") != seal[group][key]:
                raise ValueError(f"Sealed {group} changed: {key}")
        with np.load(a.out / "predictions" / f"{key}.npz", allow_pickle=False) as p:
            m = {k: t2g(unpack(p[k])) for k in SAVED}
        with np.load(a.out / "fields" / f"{key}.npz", allow_pickle=False) as p:
            z = t2g(p["rcg"]).reshape(4096)
        with np.load(packet(a.root, row), allow_pickle=False) as p:
            t, native = t2g(unpack(p["truth"])), t2g(unpack(p["native"]))
        q = torch.load(a.root / "cache/evidence_v1/feat" / f"{key}.pt", map_location="cpu", weights_only=True)["q"].float().to(dev)
        q = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)).double()
        R, C, D = m["RCG"], m["RCG_count_matched_delete"], m["conservative_delete"]
        for name, mask in (("native", native), ("rcg", R), ("c", C), ("astra", m["external_mean__delete"]), ("astra_sameK", m["external_mean_delete_sameK_RCG"]), ("mean", m["MEAN_CONTROL"])):
            iu[name][i] = int((mask & t).sum()), int((mask | t).sum())
        P = cover(D) > .5; inside = cover(R & ~C); covR = cover(R); outside = covR == 0
        ids_out = torch.nonzero(outside)[:, 0]; order = ids_out[torch.argsort(-z[ids_out], stable=True)]
        ids_in = torch.nonzero(inside > .5)[:, 0]; k_astra = min(len(ids_in), len(ids_out))
        NI = torch.zeros(4096, dtype=torch.bool, device=dev); NI[ids_in[torch.argsort(-inside[ids_in], stable=True)[:k_astra]]] = True
        banks = {"astra": top(order, k_astra), "p": top(order, min(int(P.sum()), len(order))), "all": outside.clone(),
                 "far": top(order.flip(0), len(order) // 2)}
        for f in FRACTIONS:
            banks[f"top{f:g}"] = top(order, round(len(order) * f))
        for mult in (.5, 2, 4):
            banks[f"p{mult:g}"] = top(order, min(int(int(P.sum()) * mult), len(order)))
        pos5, vpos5 = centroid(q, P, 5); far5, vfar5 = centroid(q, banks["far"], 5)
        like_target = vpos5 & vfar5 & (pos5 > far5)
        for name in ("astra", "p", "top0.05", "top0.1"):
            banks["split:" + name] = banks[name] & ~like_target
        zup = up(z)[0]; has_p = bool(P.any())
        # deletions inside C, with the same-count lowest-RCG control
        zc = zup[C]; sorted_truth = torch.cat([torch.zeros(1, device=dev), t[C][torch.argsort(zc, stable=True)].float().cumsum(0)])
        cache = {}
        for (bank, w) in itertools.product(DEL_BANKS, DEL_WINDOWS):
            pm, vp = centroid(q, P, w); om, vo = centroid(q, banks[bank], w); valid = vp & vo
            if bank == "astra":
                valid = valid & centroid(q, NI, w)[1]
            if not has_p or not bool(banks[bank].any()):
                valid = torch.zeros_like(valid)
            cache[bank, w] = (up(torch.where(valid, pm - om, torch.zeros_like(pm)))[0], up(valid.float())[0] >= 1 - 1e-6)
        for j, (bank, w, thr) in enumerate(dels):
            margin, eligible = cache[bank, w]; gone = C & eligible & (margin < thr); k = int(gone.sum())
            d_rem[j, i] = int((gone & t).sum()), int((gone & ~t).sum()); ct = int(sorted_truth[k]); d_ctl[j, i] = ct, k - ct
            if (bank, w, thr) == ASTRA:
                replay[i] = int(((C & ~gone) != m["external_mean__delete"]).sum())
        # additions outside R, with the same-count highest-RCG control of the same zone
        tokC = (cover(C) > .5).reshape(1, 1, 64, 64).float(); zones, zone_truth = {}, {}
        for reach in ADD_REACH:
            zone = ~R & (up(F.max_pool2d(tokC, 2 * reach + 1, 1, reach))[0] > .5) if reach else ~R
            zones[reach] = zone; zz = zup[zone]
            zone_truth[reach] = torch.cat([torch.zeros(1, device=dev), t[zone][torch.argsort(-zz, stable=True)].float().cumsum(0)])
        acache = {}
        for bank in ADD_BANKS:
            om, vo = centroid(q, banks[bank], 5); valid = vpos5 & vo & has_p & bool(banks[bank].any())
            acache[bank] = up(torch.where(valid, pos5 - om, torch.full_like(pos5, -1.)))[0]
        for j, (bank, reach, thr, zmin) in enumerate(adds):
            new = zones[reach] & (acache[bank] > thr) & (zup > zmin); k = int(new.sum())
            a_add[j, i] = int((new & t).sum()), int((new & ~t).sum()); ct = int(zone_truth[reach][k]); a_ctl[j, i] = ct, k - ct
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(json.dumps(dict(swept=i + 1, total=n, seconds=round(time.perf_counter() - begin, 1))), flush=True)
    np.savez_compressed(a.out / f"sweep_counts{a.suffix}.npz", d_rem=d_rem, d_ctl=d_ctl, a_add=a_add, a_ctl=a_ctl, **{"iu_" + k: v for k, v in iu.items()})
    # composition: I = I_C - removed true + added true, U = U_C - removed false + added false
    classes, folds, groups = np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows]), photo_groups(rows)
    zero = np.zeros((1, n, 2), np.int64)
    D_, Dc, A_, Ac = (np.concatenate([zero, x]) for x in (d_rem, d_ctl, a_add, a_ctl))          # index 0 = no edit
    dname = ["none"] + [f"del[{b},w{w},t{t:g}]" for b, w, t in dels]; aname = ["none"] + [f"add[{b},r{r},t{t:g},z{zm:g}]" for b, r, t, zm in adds]
    onehot = (classes[:, None] == np.unique(classes)[None]).astype(np.float64)
    def table(dd, aa, ix):
        """Class mIoU of every (deletion, addition) pair on the episodes ix. -> [Nd, Na]"""
        M = onehot[ix]; M = M[:, M.sum(0) > 0]; base = iu["c"][ix].astype(np.float64)
        I = (base[:, 0] @ M)[None, None] - (dd[:, ix, 0] @ M)[:, None] + (aa[:, ix, 0] @ M)[None]
        U = (base[:, 1] @ M)[None, None] - (dd[:, ix, 1] @ M)[:, None] + (aa[:, ix, 1] @ M)[None]
        return 100 * (I / np.maximum(U, 1)).mean(-1)
    def compose(dd, aa, d, j):
        return np.stack([iu["c"][:, 0] - dd[d, :, 0] + aa[j, :, 0], iu["c"][:, 1] - dd[d, :, 1] + aa[j, :, 1]], 1)
    everything = np.ones(n, bool); full = table(D_, A_, everything); full_ctl = table(Dc, Ac, everything)
    def nested(dd, aa, d_ok, a_ok):
        out, chosen = np.zeros((n, 2), np.int64), {}
        for f in sorted(set(folds)):
            read = folds == f; fit = ~read & ~np.isin(groups, groups[read]); tab = table(dd[d_ok], aa[a_ok], fit)
            d, j = np.unravel_index(int(tab.argmax()), tab.shape); d, j = d_ok[d], a_ok[j]
            out[read] = compose(dd, aa, d, j)[read]; chosen[str(f)] = [dname[d], aname[j]]
        return out, chosen
    all_d, all_a = np.arange(len(dname)), np.arange(len(aname)); di = lambda v: 1 + dels.index(v); ai = lambda v: 1 + adds.index(v)
    arrays = {"native": iu["native"], "rcg": iu["rcg"], "c.control": iu["c"], "astra.control": iu["astra"], "astra_sameK.control": iu["astra_sameK"],
              "astra.replayed_on_gpu": compose(D_, A_, di(ASTRA), 0), "proposed.fixed": compose(D_, A_, di(PROPOSED[0]), ai(PROPOSED[1])),
              "proposed.delete_only": compose(D_, A_, di(PROPOSED[0]), 0), "proposed.same_count.control": compose(Dc, Ac, di(PROPOSED[0]), ai(PROPOSED[1]))}
    selection = {}
    for name, (dd, aa, d_ok, a_ok) in {"delete.nested": (D_, A_, all_d, all_a[:1]), "add_on_astra.nested": (D_, A_, np.array([di(ASTRA)]), all_a),
                                      "add_on_c.nested": (D_, A_, all_d[:1], all_a), "full.nested": (D_, A_, all_d, all_a),
                                      "full.same_count.control": (Dc, Ac, all_d, all_a)}.items():
        arrays[name], selection[name] = nested(dd, aa, d_ok, a_ok)
    frozen_path = a.out.parent / (a.out.name + "_frozen.json")       # pairs fixed on another cohort before this one was read
    frozen = json.loads(frozen_path.read_text())["picks"] if frozen_path.exists() else {}
    for name, (dn, an) in frozen.items():
        arrays["frozen." + name] = compose(D_, A_, dname.index(dn), aname.index(an))
        arrays["frozen." + name + ".same_count.control"] = compose(Dc, Ac, dname.index(dn), aname.index(an))
    bd, ba = np.unravel_index(int(full.argmax()), full.shape); arrays["best_in_sample"] = compose(D_, A_, bd, ba)
    def report_for(ix, tag):
        sub = [r for r, keep in zip(rows, ix) if keep]
        corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in sub] for k in arrays}
        s, _ = summarize(sub, {k: v[ix] for k, v in arrays.items()}, corr)
        for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
            s.pop(k, None)
        s["cohort"] = tag; return s
    out = dict(n=n, device=str(dev), exposure="development; not independent confirmation", astra_replay_pixels_differing=dict(total=int(replay.sum()), episodes=int((replay > 0).sum())),
               frozen=frozen, selection=selection, best_in_sample=[dname[bd], aname[ba], float(full[bd, ba])], proposed=[dname[di(PROPOSED[0])], aname[ai(PROPOSED[1])]],
               dev=report_for(everything, "all"))
    if a.subset:
        keys = {f"{int(r['fold'])}_{int(r['e'])}_{int(r['c'])}" for r in json.loads(a.subset.read_text())}
        out["subset"] = report_for(np.array([r["key"] in keys for r in rows]), a.subset.name)
    pool = lambda x: [int(v) for v in x.sum(0)]
    out["deletions"] = [dict(name=dname[d], miou=float(full[d, 0]), control=float(full_ctl[d, 0]), removed_true_false=pool(D_[d]), control_removed_true_false=pool(Dc[d])) for d in all_d]
    out["additions_on_astra"] = [dict(name=aname[j], miou=float(full[di(ASTRA), j]), control=float(table(D_[[di(ASTRA)]], Ac, everything)[0, j]),
                                      added_true_false=pool(A_[j]), control_added_true_false=pool(Ac[j])) for j in all_a]
    write_json(a.out / f"report{a.suffix}.json", out)
    L = [f"# Recheck sweep: {n} episodes, {len(dels)} deletion x {len(adds)} addition variants", "",
         f"GPU replay of the candidate's last step differs from the delivered mask in {int(replay.sum())} pixels over {int((replay > 0).sum())} episodes.", ""]
    for tag in ("dev", "subset"):
        if tag not in out: continue
        s = out[tag]; L += [f"## {s['cohort']} (n={s['n']})", "", "| arm | mIoU | vs native | vs rcg | vs C | vs astra candidate |", "|---|---:|---|---|---|---|"]
        for k in arrays:
            c = s["contrasts"][k]; f = lambda b: f"{c[b]['gain']:+.2f} [{c[b]['ci95'][0]:+.2f}, {c[b]['ci95'][1]:+.2f}]" if b in c else ""
            L.append(f"| {k} | {s['scores'][k]:.2f} | {f('native')} | {f('rcg')} | {f('c.control')} | {f('astra.control')} |")
        L += ["", "Folds vs native: " + "; ".join(f"{k}: " + ", ".join(f"{x} {v['gain_vs_native'][x]:+.2f}" for x in ("astra.control", "proposed.fixed", "full.nested")) for k, v in s["folds"].items()), ""]
    L += [f"Chosen: {json.dumps(selection)}", "", f"Best in sample (optimistic): {out['best_in_sample']}", "", "## Deletion variants alone (all episodes, no selection)", "",
          "| variant | mIoU | same-count control | removed true | removed false |", "|---|---:|---:|---:|---:|"]
    for r in sorted(out["deletions"], key=lambda r: -r["miou"])[:40]:
        L.append(f"| {r['name']} | {r['miou']:.2f} | {r['control']:.2f} | {r['removed_true_false'][0]:,} | {r['removed_true_false'][1]:,} |")
    L += ["", "## Addition variants on the Astra candidate (all episodes, no selection)", "", "| variant | mIoU | same-count control | added true | added false |", "|---|---:|---:|---:|---:|"]
    for r in sorted(out["additions_on_astra"], key=lambda r: -r["miou"])[:40]:
        L.append(f"| {r['name']} | {r['miou']:.2f} | {r['control']:.2f} | {r['added_true_false'][0]:,} | {r['added_true_false'][1]:,} |")
    (a.out / f"report{a.suffix}.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:30]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("infer", "sweep"):
        s = sub.add_parser(name); s.add_argument("--root", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    s = sub.choices["infer"]; s.add_argument("--manifest", type=Path, required=True); s.add_argument("--astra", type=Path, required=True)
    s.add_argument("--workers", type=int, default=11); s.add_argument("--expected", type=int, default=241)
    s = sub.choices["sweep"]; s.add_argument("--subset", type=Path); s.add_argument("--device", default="auto")
    s.add_argument("--grid", choices=("first", "wide"), default="first"); s.add_argument("--suffix", default="")
    a = p.parse_args(); (infer if a.mode == "infer" else sweep)(a)


if __name__ == "__main__":
    main()
