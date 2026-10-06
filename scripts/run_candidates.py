#!/usr/bin/env python3
"""Candidates from the query alone: how good is the best region of a hierarchy built without the reference? (GPU)

From zero, a method is a candidate set of query regions and a rule that selects among them with the reference.
FoRIS's candidates are the level sets of a similarity score; INSID3's are flat token clusters. Here the candidates are
all nodes of one complete spatially-connected agglomerative hierarchy of the query's tokens (no threshold, no
reference). Stored per node: size, truth mass, and the sums of the reference cosine and of the RCG field, so that the
ceiling (best node, best union of a few nodes; reads truth) and truth-free selections can be scored without the GPU.

  python scripts/run_candidates.py --episodes .../train_episodes.json --start 0 --count 75 --out outputs/claude_candidates_a
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LINKS = ("ward", "average")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, required=True); p.add_argument("--start", type=int, required=True); p.add_argument("--count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--limit", type=int)
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent")); p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.image import grid_to_graph
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host, run_foris
    from run_rcg2 import solve
    man = json.loads(a.episodes.read_text()); rows = []
    for f in sorted({e["fold"] for e in man["episodes"]}):
        rows += [dict(e, key="%d_%d_%d" % (e["fold"], e["e"], e["c"])) for e in man["episodes"] if e["fold"] == f][a.start:a.start + a.count]
    rows = rows[:a.limit]; data, ann = Path(man["data_root"]), Path(man["annotation_root"]); dev = torch.device("cuda"); N = len(rows)
    a.out.mkdir(parents=True, exist_ok=False); grid = grid_to_graph(64, 64)
    tok = {k: np.zeros((N, 4096), np.float16) for k in ("truth", "score", "rcg", "cos", "native")}
    kids = {l: np.zeros((N, 4095, 2), np.int16) for l in LINKS}; node = {l: np.zeros((N, 8191, 4), np.float32) for l in LINKS}      # size, truth, cosine sum, field sum
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(int(os.environ.get("RCG_THREADS", "4")))
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            mask, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"]; q, r = F.normalize(deb[0, -1].flatten(1).T.float(), dim=1), F.normalize(deb[0, 0].flatten(1).T.float(), dim=1)
            score = got["score"].float(); cov_t = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]; fg = cov_t.flatten() > .5
            z0 = solve(q.half().float().cpu(), r.half().float().cpu(), cov_t.cpu().numpy(), score.cpu().numpy().astype(np.float32), 16., "cpu")
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            tt = F.avg_pool2d(F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest"), 16).flatten()
            mu_r = F.normalize(r[fg].mean(0), dim=0) if fg.any() else F.normalize(r.mean(0), dim=0); cos = q @ mu_r
            for k, v in dict(truth=tt, score=score.flatten(), rcg=torch.from_numpy(z0).flatten(), cos=cos, native=F.avg_pool2d(mask[None, None].float(), 16).flatten()).items():
                tok[k][n] = v.float().cpu().numpy()
            x = q.cpu().numpy(); leaf = np.stack([np.ones(4096), tok["truth"][n].astype(np.float64), cos.cpu().numpy().astype(np.float64), z0.flatten().astype(np.float64)], 1)
            for l in LINKS:
                ch = AgglomerativeClustering(n_clusters=1, linkage=l, connectivity=grid, compute_full_tree=True, **({} if l == "ward" else dict(metric="cosine"))).fit(x).children_
                s = np.zeros((8191, 4)); s[:4096] = leaf
                for j, (u, v) in enumerate(ch):
                    s[4096 + j] = s[u] + s[v]
                kids[l][n], node[l][n] = ch, s
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=N, seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "tokens.npz", **tok, **{"kids:" + l: kids[l] for l in LINKS}, **{"node:" + l: node[l] for l in LINKS})
    print("CANDIDATES_DONE", flush=True)


if __name__ == "__main__":
    main()
