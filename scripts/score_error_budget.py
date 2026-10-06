"""Error budget of a stored score field on the public 4000: episodes whose best cut stays below IoU 0.5 against
extent errors of the rest. Reads per-level counts only (`counts.npz`, `rows.json`); no model, no GPU.

    python scripts/score_error_budget.py evidence/local/research_20261005/cut_levels4000
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]); Z = np.load(D / "counts.npz"); rows = json.loads((D / "rows.json").read_text())
lev = Z["levels"]; k0 = int(np.argmin(abs(lev - .5))); cls = np.array([r["c"] for r in rows]); T = Z["truth"].astype(float); N = len(rows); a = np.arange(N)
miou = lambda I, U: np.mean([I[cls == c].sum() / max(U[cls == c].sum(), 1e-9) for c in np.unique(cls)]) * 100
for name in ("foris", "rcg"):
    I, U = Z[f"I_{name}"].astype(float), Z[f"U_{name}"].astype(float); iou = I / np.maximum(U, 1); best, kb = iou.max(1), iou.argmax(1)
    I0, U0 = I[:, k0], U[:, k0]; base = miou(I0, U0); miss = best < .5; P, M = U0 - T, T - I0

    def swap(ix, Ib, Ub):
        i, u = I0.copy(), U0.copy(); i[ix], u[ix] = Ib[ix], Ub[ix]; return miou(i, u) - base
    print(f"{name} at 0.5: {base:.2f}; not found {miss.sum()} ({miss.mean() * 100:.1f} %); not found made perfect {swap(miss, T, T):+.2f}; "
          f"best cut on found {swap(~miss, I[a, kb], U[a, kb]):+.2f}; found made perfect {swap(~miss, T, T):+.2f}; "
          f"share of all false-positive pixels from not found {P[miss].sum() / P.sum():.3f}")
    for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, 1.01)):
        ix = (T / 1024 ** 2 >= lo) & (T / 1024 ** 2 < hi)
        print(f"  object share {lo}-{hi}: n={ix.sum()}, not found {miss[ix].mean() * 100:.0f} %, IoU at 0.5 {iou[ix, k0].mean():.3f}, best cut {best[ix].mean():.3f}, "
              f"false-positive / truth area {P[ix].sum() / T[ix].sum():.2f}, missed / truth area {M[ix].sum() / T[ix].sum():.2f}")
