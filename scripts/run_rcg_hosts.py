#!/usr/bin/env python3
"""RCG with other scores as its input: is the gain tied to FoRIS, or does the same readout lift the model's own score?

For every available 64 x 64 score of the stage bank (the model's nearest-token and mean margins, each FoRIS stage
response, the INSID3 terms) the delivered `rcg_readout.predict` is run unchanged on the cached features; the host
alone is the same score scaled per image and cut at 0.5, as RCG's own renderer does. Development data (DEV241).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def one(job):
    key, feat_path, packet_path, field_path, astra, names = job
    import torch
    torch.set_num_threads(1); sys.path.insert(0, astra)
    import rcg_readout as rcg
    feat = torch.load(feat_path, map_location="cpu", weights_only=True); q, r = feat["q"].float().numpy(), feat["r"].float().numpy()
    with np.load(packet_path, allow_pickle=False) as p:
        cov, truth, native, cached = p["cov"].copy(), rcg.unpack(p["truth"]), rcg.unpack(p["native"]), p["score"].copy()
    with np.load(field_path, allow_pickle=False) as p:
        fields = {k: p[k].astype(np.float32).reshape(64, 64) for k in names}
    fields["foris.cached_score"] = cached.astype(np.float32); out = {"native": [int((native & truth).sum()), int((native | truth).sum())]}
    for k, f in fields.items():
        if not np.isfinite(f).all() or float(f.max() - f.min()) < 1e-9:
            host = z = np.zeros((1024, 1024), bool)
        else:
            host = rcg.mask_from_field(rcg.minmax(f)); z = rcg.mask_from_field(rcg.predict(q, r, cov, f)[0])
        out[k] = [int((host & truth).sum()), int((host | truth).sum())]; out[k + " + RCG"] = [int((z & truth).sum()), int((z | truth).sum())]
    return key, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--stage", type=Path, required=True); p.add_argument("--astra", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    import multiprocessing as mp
    from ics.experiment import packet, summarize
    rows = json.loads((a.stage / "manifest.json").read_text())
    with np.load(a.stage / "fields" / f"{rows[0]['key']}.npz") as z:
        names = [k for k in z.files if z[k].size == 4096 and "@never" not in k and "pre-" not in k]
    jobs = [(r["key"], str(a.root / "cache/evidence_v1/feat" / f"{r['key']}.pt"), str(packet(a.root, r)), str(a.stage / "fields" / f"{r['key']}.npz"), str(a.astra.resolve()), names) for r in rows]
    got = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for n, (key, out) in enumerate(pool.imap_unordered(one, jobs), 1):
            got[key] = out
            if n % 40 == 0 or n == len(jobs):
                print(json.dumps(dict(completed=n, total=len(jobs))), flush=True)
    arms = list(got[rows[0]["key"]]); arrays = {k: np.array([got[r["key"]][k] for r in rows], np.int64) for k in arms}
    arrays = {(k + ".control" if k != "native" and not k.endswith("+ RCG") else k): v for k, v in arrays.items()}
    corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for k in arrays}
    s, _ = summarize(rows, arrays, corr)
    for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        s.pop(k, None)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "report.json").write_text(json.dumps(s, indent=2) + "\n")
    L = [f"# RCG on other input scores: {len(rows)} episodes (DEV241), complete FoRIS {s['scores']['native']:.2f}", "", "| input score | alone | with RCG | RCG gain over its own input | with RCG vs complete FoRIS |", "|---|---:|---:|---|---|"]
    for k in sorted([k for k in arms if k != "native" and not k.endswith("+ RCG")], key=lambda k: -s["scores"][k + " + RCG"]):
        c = s["contrasts"][k + " + RCG"]; g, v = c[k + ".control"], c["native"]
        L.append(f"| {k} | {s['scores'][k + '.control']:.2f} | {s['scores'][k + ' + RCG']:.2f} | {g['gain']:+.2f} [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}] | {v['gain']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] |")
    (a.out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
