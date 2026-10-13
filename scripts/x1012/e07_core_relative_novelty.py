"""Novelty measured against the target's own explained-ness instead of the whole query.

e06 showed a global novelty penalty helps whole-object sets and hurts part sets.  Here a token is penalised
only if the reference explains it WORSE than it explains the confidently matched core of the target:
  core    tokens whose region-averaged closed-set posterior is >= 0.8 of its maximum
  V1      c * min(1, E / median_core E)                         (kernel-mass ratio)
  V2      c * clip((s - s_med) / (s_core - s_med), 0, 1)        (best-match similarity, linear)
  V3      D_F / (E + median_core E)                             (open-set posterior with core-level novelty mass)
No FoRIS input.  Single rule: region average, bilinear, min-max > 0.5.
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N) + lab.ext_subset(N)
DOMS = tuple(dict.fromkeys(b["dataset"] for b in rows))
M = lab.defaultdict(lab.Meter)
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b); ig = None if ig is None else ig.to(DEV)
    g = gt if ig is None else gt & ~ig
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    smax = sim.max(1).values
    k = torch.exp((sim - sim.max()) / 0.05)
    DF, DB = k @ y, k @ (1 - y)
    E = DF + DB
    c = DF / E.clamp_min(1e-30)
    C = lab.kmeans(lab.unit(q), 64, iters=6)
    oh = torch.nn.functional.one_hot((lab.unit(q) @ C.T).argmax(1), 64).float()
    reg = lambda v: oh @ ((oh.T @ v) / oh.sum(0).clamp_min(1))
    rc = reg(c)
    core = rc >= 0.8 * rc.max()
    Ec, sc, sm = E[core].median(), smax[core].median(), smax.median()

    def add(name, field):
        S = lab.up(field.reshape(64, 64))
        if ig is not None:
            S = torch.where(ig, S.min(), S)
        M[name].add(b, *lab.iu(lab.minmax(S) > 0.5, g))

    p = lab.foris_cli(b).to(DEV)
    M["FoRIS complete (CRF)"].add(b, *lab.iu(p if ig is None else p & ~ig, g))
    add("closed-set c", rc)
    add("global novelty: c * minmax(best-match sim)", reg(c * lab.minmax(smax)))
    add("V1 core-relative kernel mass", reg(c * (E / Ec).clamp(max=1)))
    add("V2 core-relative similarity", reg(c * ((smax - sm) / (sc - sm).clamp_min(1e-3)).clamp(0, 1)))
    add("V3 open-set, core-level novelty mass", reg(DF / (E + Ec)))
    lab.free()

print(f"class-pooled mIoU (no CRF on the independent rows), n<={N}/domain; last column = sets at or above FoRIS")
print(f"{'method':46s}" + "".join(f"{d[:8]:>9s}" for d in DOMS))
base = M["FoRIS complete (CRF)"].row(DOMS)
for n, m in M.items():
    r = m.row(DOMS)
    print(f"{n:46s}" + "".join(f"{r[d][0]:9.1f}" for d in DOMS) + f"   {sum(r[d][0] >= base[d][0] for d in DOMS)}/10")
