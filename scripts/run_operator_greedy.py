#!/usr/bin/env python3
"""Additions and deletions as operators: which few of the available masks, applied in which order, compose best (GPU).

Library: every sealed mask of the stage bank (the model origin, each stage of INSID3 and FoRIS), complete FoRIS, RCG,
the restricted deletion C, the Astra candidate, and level sets of the RCG field and of the FoRIS score. An operator is
`add M` (S | M) or `delete M` (S & M). From a start mask the best operator by class mIoU on the fitting episodes is
applied, repeatedly. Each step is recorded with its exact four counts, so its marginal purity is read against the
break-even of the mask it is applied to. The path is chosen on three folds and read on the fourth after every step:
the score as a function of the number of components. All episodes are development data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LEVELS = (.3, .4, .6, .7)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--stage", type=Path, required=True); p.add_argument("--recheck", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--starts", default="model.raw_nn,native,insid3.final"); p.add_argument("--steps", type=int, default=8); p.add_argument("--exclude", default="", help="masks kept for comparison but not offered as operators")
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from ics.experiment import metric, packet, photo_groups, summarize, unpack
    dev = torch.device("cuda")
    rows = json.loads((a.stage / "manifest.json").read_text()); other = json.loads((a.recheck / "manifest.json").read_text())
    if [r["key"] for r in rows] != [r["key"] for r in other]:
        raise ValueError("The two sealed runs cover different episodes")
    stage_arms = [k for k in json.loads((a.stage / "sealed.json").read_text())["arms"] if "pre-" not in k and "@never" not in k]
    rename = {"RCG": "rcg", "RCG_count_matched_delete": "c", "conservative_delete": "rcg&hypotheses", "external_mean__delete": "astra", "MEAN_CONTROL": "mean_control"}
    n = len(rows); lib = {}; truth = torch.zeros((n, 1024, 1024), dtype=torch.bool, device=dev)
    put = lambda name, i, m: lib.setdefault(name, torch.zeros((n, 1024, 1024), dtype=torch.bool, device=dev)).__setitem__(i, m)
    up = lambda x: F.interpolate(torch.from_numpy(np.asarray(x, np.float32)).to(dev)[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
    for i, row in enumerate(rows):
        key = row["key"]
        with np.load(a.stage / "predictions" / f"{key}.npz", allow_pickle=False) as z:
            for k in stage_arms:
                put(k, i, torch.from_numpy(unpack(z[k])).to(dev))
        with np.load(a.recheck / "predictions" / f"{key}.npz", allow_pickle=False) as z:
            for k, name in rename.items():
                put(name, i, torch.from_numpy(unpack(z[k])).to(dev))
        with np.load(a.recheck / "fields" / f"{key}.npz", allow_pickle=False) as z:
            field = up(z["rcg"])
        with np.load(packet(a.root, row), allow_pickle=False) as z:
            truth[i] = torch.from_numpy(unpack(z["truth"])).to(dev); put("native", i, torch.from_numpy(unpack(z["native"])).to(dev))
            s = z["score"].astype(np.float32); score = up((s - s.min()) / max(float(s.max() - s.min()), 1e-6))
        for t in LEVELS:
            put(f"rcg_field>{t:g}", i, field > t); put(f"foris_score>{t:g}", i, score > t)
    names = sorted(lib); offered = [k for k in names if k not in set(a.exclude.split(","))]; classes, folds, groups = np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows]), photo_groups(rows)
    area = truth.flatten(1).sum(1)
    def iu(S):
        inter = (S & truth).flatten(1).sum(1); return torch.stack([inter, S.flatten(1).sum(1) + area - inter], 1).cpu().numpy()
    alone = {k: iu(lib[k]) for k in names}
    def greedy(start, fit):
        S = lib[start].clone(); cur = iu(S); path, per_step = [], [cur.copy()]
        for _ in range(a.steps):
            best = None
            for k in offered:
                for op in ("add", "delete"):
                    cand = iu(S | lib[k] if op == "add" else S & lib[k]); score = metric(cand[fit], classes[fit])
                    if best is None or score > best[0]:
                        best = (score, op, k, cand)
            if best[0] <= metric(cur[fit], classes[fit]) + 1e-9:
                break
            score, op, k, cand = best; new = S | lib[k] if op == "add" else S & lib[k]; plus, minus = new & ~S, S & ~new
            counts = [int((plus & truth).sum()), int((plus & ~truth).sum()), int((minus & ~truth).sum()), int((minus & truth).sum())]
            j = float(cur[:, 0].sum() / max(cur[:, 1].sum(), 1)); good, bad = (counts[0], counts[1]) if op == "add" else (counts[2], counts[3])
            path.append(dict(op=op, mask=k, fit_before=metric(cur[fit], classes[fit]), fit_after=score, add_TP=counts[0], add_FP=counts[1], delete_FP=counts[2], delete_TP=counts[3],
                             purity=good / max(good + bad, 1), pooled_break_even=(j / (1 + j) if op == "add" else 1 / (1 + j))))
            S, cur = new, cand; per_step.append(cur.copy())
        while len(per_step) <= a.steps:
            per_step.append(per_step[-1])
        return path, per_step
    everything = np.ones(n, bool); out = dict(n=n, library=names, offered=offered, steps=a.steps, exposure="development; not independent confirmation",
                                              alone={k: metric(v, classes) for k, v in alone.items()}, starts={})
    for start in a.starts.split(","):
        path, per_step = greedy(start, everything)
        nested = [np.zeros((n, 2), np.int64) for _ in range(a.steps + 1)]; fold_paths = {}
        for f in sorted(set(folds)):
            read = folds == f; fit = ~read & ~np.isin(groups, groups[read]); fp, fs = greedy(start, fit)
            fold_paths[str(f)] = [f"{s['op']} {s['mask']}" for s in fp]
            for k in range(a.steps + 1):
                nested[k][read] = fs[k][read]
        arrays = {"native": alone["native"], "rcg": alone["rcg"], "astra.control": alone["astra"], "start.control": alone[start]}
        arrays.update({f"nested.{k}_steps": nested[k] for k in range(1, a.steps + 1)})
        corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for k in arrays}
        s, _ = summarize(rows, arrays, corr)
        for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
            s.pop(k, None)
        out["starts"][start] = dict(in_sample_path=path, in_sample_scores=[metric(v, classes) for v in per_step], fold_paths=fold_paths, nested=s)
        print(json.dumps(dict(start=start, path=[f"{x['op']} {x['mask']}" for x in path], nested=[round(s["scores"][f"nested.{k}_steps"], 2) for k in range(1, a.steps + 1)])), flush=True)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "report.json").write_text(json.dumps(out, indent=2) + "\n")
    L = [f"# Operators chosen greedily: {n} episodes, {len(names)} masks, add or delete", ""]
    for start, r in out["starts"].items():
        L += [f"## Start: {start} ({out['alone'][start]:.2f})", "", "In sample (optimistic):", "", "| step | operator | mIoU | added true | added false | deleted false | deleted true | purity | break-even |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        L += [f"| {i + 1} | {x['op']} {x['mask']} | {x['fit_after']:.2f} | {x['add_TP']:,} | {x['add_FP']:,} | {x['delete_FP']:,} | {x['delete_TP']:,} | {x['purity']:.3f} | {x['pooled_break_even']:.3f} |" for i, x in enumerate(r["in_sample_path"])]
        L += ["", "Path chosen on three folds, read on the fourth, after k steps:", "", "| k | mIoU | vs native | vs astra candidate | vs start |", "|---:|---:|---|---|---|"]
        for k in range(1, a.steps + 1):
            c = r["nested"]["contrasts"][f"nested.{k}_steps"]; f = lambda b: f"{c[b]['gain']:+.2f} [{c[b]['ci95'][0]:+.2f}, {c[b]['ci95'][1]:+.2f}]"
            L.append(f"| {k} | {r['nested']['scores'][f'nested.{k}_steps']:.2f} | {f('native')} | {f('astra.control')} | {f('start.control')} |")
        L += ["", "Fold paths: " + json.dumps(r["fold_paths"]), ""]
    L += ["## Every mask alone", "", "| mask | mIoU |", "|---|---:|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(out["alone"].items(), key=lambda kv: -kv[1])]
    (a.out / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
