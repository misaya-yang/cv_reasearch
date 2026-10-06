#!/usr/bin/env python3
"""RCG with its three measured auxiliaries, alone and together: per-level counts for every field (GPU), scored offline.

Fields per episode (all label-free):
  lam16 / lam64            the RCG solve at the sealed strength (read from the sealed run) and at four times the strength
  *.fine[t]                the field upsampled to 128 x 128 with shifted-grid encoder features as the guide (sigma 1.25, tau t)
Each field is cut at LEVELS on the 1024 grid; intersection, union and mask size are stored, so the level rule
(fixed 0.5, or one level per bin of the observable mask size, fitted on the other folds) is applied offline.

  python scripts/run_rcg2.py --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_rcg2_fresh600
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LEVELS = np.round(np.arange(0.30, 0.7001, 0.0125), 4)
TAUS, SIGMA, WINDOW = (0.07, 0.15), 1.25, 2


def solve(q, r, cov, score, lam, dev):
    """The RCG readout (src/ics/methods/rcg.py) with the smoothing strength as an argument."""
    import torch
    from scipy import sparse
    from scipy.sparse.linalg import cg
    from scipy.stats import rankdata
    rank = lambda x: ((rankdata(x.ravel(), method="average") - .5) / x.size).astype(np.float32)
    fi = np.flatnonzero(cov.ravel() >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov.ravel() == cov.max())
    sim = q @ r.T; dq = sim.topk(10, dim=1).values.mean(1); dr = sim.topk(10, dim=0).values.mean(0)
    guide = ((2 * sim[:, fi] - dr[fi][None, :]).max(1).values - dq).cpu().numpy()
    s = np.asarray(score, np.float32); s = ((s - s.min()) / max(float(s.max() - s.min()), 1e-6)).ravel()
    y = (s + .5 * (rank(guide) - rank(s))).astype(np.float64)
    sim = q @ q.T; sim.fill_diagonal_(-2); values, idx = sim.topk(20, dim=1); dist = (1 - values).clamp_min(0)
    w = sparse.csr_matrix((torch.exp(-dist / dist[:, -1:].clamp_min(1e-6)).cpu().numpy().ravel(), (np.repeat(np.arange(4096), 20), idx.cpu().numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T); w.data = np.sqrt(w.data); w = w / max(float(np.asarray(w.sum(1)).mean()), 1e-8)
    a = .1 + np.abs(2 * s - 1); a = (a / a.mean()).astype(np.float64)
    z, status = cg(sparse.diags(a) + lam * (sparse.diags(np.asarray(w.sum(1)).ravel()) - w), a * y, x0=y, rtol=1e-7, atol=1e-9, maxiter=600)
    return z.reshape(64, 64).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent")); p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4"); p.add_argument("--limit", type=int)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host
    man = json.loads(a.host_manifest.read_text()); rows = json.loads((a.run / "manifest.json").read_text())[:a.limit]
    data = Path(man["data_root"]); dev = torch.device("cuda"); lv = torch.from_numpy(LEVELS).float().to(dev)[:, None, None]
    a.out.mkdir(parents=True, exist_ok=False)
    names = [f + x for f in ("lam16", "lam64") for x in [""] + [".fine[t%g]" % t for t in TAUS]]
    I, U = ({k: np.zeros((len(rows), len(LEVELS)), np.int64) for k in names} for _ in range(2)); T = np.zeros(len(rows), np.int64); parity = 0.
    fi = torch.arange(128, device=dev); base = ((fi * 8 + 4 - 8).float() / 16).round().long(); off = torch.arange(-WINDOW, WINDOW + 1, device=dev)
    ti = base[:, None] + off[None]; dist = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16; ok1 = (ti >= 0) & (ti < 64); ti = ti.clamp(0, 63); K = len(off)
    nb = (ti[:, None, :, None].expand(128, 128, K, K) * 64 + ti[None, :, None, :].expand(128, 128, K, K)).reshape(-1, K * K)
    ks = torch.exp(-(dist[:, None, :, None] ** 2 + dist[None, :, None, :] ** 2).reshape(-1, K * K) / (2 * SIGMA ** 2)) * (ok1[:, None, :, None] & ok1[None, :, None, :]).reshape(-1, K * K)
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda")
        for n, row in enumerate(rows):
            feat = torch.load(a.root / "cache/evidence_v1/feat" / (row["key"] + ".pt")); q, r = (F.normalize(feat[k].to(dev).float(), dim=1) for k in ("q", "r"))
            with np.load(a.root / "results/extent_v1/run/packets" / (row["key"] + ".npz"), allow_pickle=False) as z:
                truth = torch.from_numpy(np.unpackbits(z["truth"]).reshape(1024, 1024).astype(bool)).to(dev); cov, score = z["cov"], z["score"]
            with np.load(a.run / "fields" / (row["key"] + ".npz"), allow_pickle=False) as z:
                sealed = z["rcg"].astype(np.float32)
            if n < 20:
                parity = max(parity, float(np.abs(solve(q, r, cov, score, 16., dev) - sealed).max()))
            fields = dict(lam16=torch.from_numpy(sealed).to(dev), lam64=torch.from_numpy(solve(q, r, cov, score, 64., dev)).to(dev))
            tgt = host._transform(Image.open(data / row["query"]).convert("RGB")).to(dev); pad = F.pad(tgt[None], (4, 4, 4, 4), mode="reflect")[0]
            fine = torch.zeros(128, 128, q.shape[1], device=dev)
            for ay, ax in itertools.product((0, 1), (0, 1)):
                sy, sx = (-4, 4)[ay], (-4, 4)[ax]
                f = F.normalize(host._extract_features(pad[None, None, :, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]), p=2, dim=2)
                if bool(feat["debiased"]):
                    f = host._debias_features(f)
                fine[ay::2, ax::2] = F.normalize(f[0, 0], dim=0).permute(1, 2, 0)
            cos = torch.einsum("pc,pkc->pk", fine.reshape(128 * 128, -1), q[nb])
            maps = {}
            for k, f in fields.items():
                maps[k] = f[None, None]; zq = f.flatten()[nb]
                for t in TAUS:
                    w = ks * torch.exp((cos - 1) / t); maps["%s.fine[t%g]" % (k, t)] = ((w * zq).sum(1) / w.sum(1).clamp_min(1e-12)).reshape(1, 1, 128, 128)
            for k, f in maps.items():
                m = F.interpolate(f, (1024, 1024), mode="bilinear", align_corners=False)[0] > lv
                I[k][n] = (m & truth).flatten(1).sum(1).cpu().numpy(); U[k][n] = (m | truth).flatten(1).sum(1).cpu().numpy()
            T[n] = int(truth.sum())
            if (n + 1) % 50 == 0:
                print(json.dumps(dict(n=n + 1, total=len(rows), seconds=round(time.monotonic() - begin, 1), parity=parity)), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "counts.npz", levels=LEVELS, truth=T, parity=parity, **{"I:" + k: v for k, v in I.items()}, **{"U:" + k: v for k, v in U.items()})
    print("RCG2_DONE parity %.2g" % parity, flush=True)


if __name__ == "__main__":
    main()
