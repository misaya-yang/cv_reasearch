"""Are the independent readout's false positives 'unexplained' query regions, and does an open-set term remove them?

No FoRIS input anywhere.  k_ij = exp(sim_ij / 0.05) on centred, positionally projected O24.
  D_F, D_B   kernel mass of query token j on reference FG / BG;  E = D_F + D_B  (how well the reference explains j)
  c          closed-set posterior D_F / E                      (the coverage transfer used so far)
  p_kappa    open-set posterior D_F / (E + kappa * median_j E) (a token explained far below the query's typical
             level is pushed to background, whatever its closed-set posterior)
Diagnostic: among tokens the closed-set readout calls FG, can E / best-match similarity tell true from false?
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N) + lab.ext_subset(N)
DOMS = tuple(dict.fromkeys(b["dataset"] for b in rows))
M = lab.defaultdict(lab.Meter)
A = lab.defaultdict(lambda: lab.defaultdict(list))


def auc(score, pos):
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 5 or n0 < 5:
        return None
    r = torch.empty_like(score); r[score.argsort()] = torch.arange(1, len(score) + 1, device=score.device, dtype=score.dtype)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


for kk, b in enumerate(rows):
    d = b["dataset"]
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

    def add(name, field, rule="mm"):
        S = lab.up(field.reshape(64, 64))
        if ig is not None:
            S = torch.where(ig, S.min(), S)
        M[name].add(b, *lab.iu((lab.minmax(S) > 0.5) if rule == "mm" else (S > 0.5), g))

    p = lab.foris_cli(b).to(DEV)
    M["FoRIS complete (CRF)"].add(b, *lab.iu(p if ig is None else p & ~ig, g))
    add("closed-set c | region, minmax", reg(c))
    add("closed-set c | region, > 0.5", reg(c), "abs")
    for kap in (0.1, 0.3, 1.0, 3.0):
        pk = DF / (E + kap * E.median())
        add(f"open-set kappa={kap} | region, minmax", reg(pk))
        add(f"open-set kappa={kap} | region, > 0.5", reg(pk), "abs")
    add("c * minmax(best-match sim) | region, minmax", reg(c * lab.minmax(smax)))
    add("c * minmax(log E) | region, minmax", reg(c * lab.minmax(torch.log(E.clamp_min(1e-30)))))
    # diagnostic on tokens that the closed-set readout calls FG
    cg = lab.coverage64(g).flatten() > 0.5
    called = lab.minmax(reg(c)) > 0.5
    if called.sum() > 20:
        for nm, v in (("log E", torch.log(E.clamp_min(1e-30))), ("best-match sim", smax), ("closed-set c", c)):
            a = auc(v[called].float(), cg[called])
            if a is not None:
                A[nm][d].append(a)
        A["share of called tokens that are false"][d].append(float((~cg[called]).float().mean()))
    lab.free()
    if kk % 300 == 0:
        print(kk, len(rows), flush=True)

print(f"\nclass-pooled mIoU (no CRF on the independent rows), n<={N}/domain")
print(f"{'method':44s}" + "".join(f"{d[:8]:>9s}" for d in DOMS))
for n, m in M.items():
    r = m.row(DOMS)
    print(f"{n:44s}" + "".join(f"{r[d][0]:9.1f}" for d in DOMS))
print("\nwithin tokens called FG by the closed-set readout: AUROC of each quantity for true vs false (0.5 = no information)")
for nm in A:
    print(f"{nm:44s}" + "".join(f"{np.mean(A[nm][d]):9.2f}" if A[nm][d] else "        -" for d in DOMS))
