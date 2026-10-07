"""Where reference tokens land in the failing episodes: stored fresh600 token record, no model, no GPU.

For each episode whose best RCG cut stays below IoU 0.5 (64x64 tokens), compare the true object tokens with the tokens
RCG wrongly keeps at 0.5: share of tokens that are the nearest query token of a reference object token (back_fg) or of
a reference background token (back_bg). Hard nearest-token landing, a proxy for a soft reference-to-query matrix.
"""
import numpy as np
Z = np.load("/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/claude_order_fresh600/tokens.npz")
t = Z["truth"].astype(np.float32) > .5; r = Z["rcg"].astype(np.float32); bf, bb = Z["back_fg"], Z["back_bg"]
lev = np.linspace(.1, .9, 33); N = len(t)
best = np.array([max(((r[i] > l) & t[i]).sum() / max(((r[i] > l) | t[i]).sum(), 1) for l in lev) for i in range(N)])
rows = []
for i in np.where(best < .5)[0]:
    wrong = (r[i] > .5) & ~t[i]; rest = ~(r[i] > .5) & ~t[i]
    if t[i].sum() < 1 or wrong.sum() < 1: continue
    rows.append((best[i], bf[i][t[i]].mean(), bf[i][wrong].mean(), bb[i][t[i]].mean(), bb[i][wrong].mean(), bb[i][rest].mean(), bf[i][rest].mean()))
R = np.array(rows); print("failing", int((best < .5).sum()), "comparable", len(R))
for name, ix in (("all", np.ones(len(R), bool)), ("best<0.2", R[:, 0] < .2), ("0.2-0.5", R[:, 0] >= .2)):
    x = R[ix]
    print(f"{name} n={len(x)} | ref-object hits: object {x[:,1].mean():.3f} wrong {x[:,2].mean():.3f} rest {x[:,6].mean():.3f}"
          f" | ref-background hits: object {x[:,3].mean():.3f} wrong {x[:,4].mean():.3f} rest {x[:,5].mean():.3f}"
          f" | wrong has more background hits than object in {int((x[:,4] > x[:,3]).sum())}/{len(x)}"
          f" | net (obj hits - bg hits) higher on object in {int(((x[:,1]-x[:,3]) > (x[:,2]-x[:,4])).sum())}/{len(x)}")
