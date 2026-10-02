"""Region-level accuracy of cross-fitted recognisers on predicted regions, by region size and purity.
  python scripts/region_report.py rf/ade20k_val_X_SRC.pt rec/X_SRC_cf_KINDS.pt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *
P = lambda p: p if p.startswith("/") else f"{CACHE}/{p}"
d = torch.load(P(sys.argv[1])); R = [Recogniser(P(sys.argv[2]) + f".f{k}") for k in (0, 1)]
m = d["kind"] == 1; x, y, pc, area, pur, img = d["x"][m].float().to(DEV), d["y"][m], d["pc"][m], d["area"][m], d["purity"][m], d["img"][m].long()
r = torch.zeros(len(y), R[0].C)
for k in (0, 1): r[img % 2 == k] = R[k](x[(img % 2 == k).to(DEV)]).cpu()
top = r.argmax(1); seg_ok = (pc == y).float(); rec_ok = (top == y).float(); own = r[torch.arange(len(y)), pc]
def row(name, s):
    if s.sum() == 0: return
    print(f"{name:26s} n={int(s.sum()):6d} share_of_area={float(area[s].sum() / area.sum()):.3f} seg_acc={float(seg_ok[s].mean()):.3f} rec_acc={float(rec_ok[s].mean()):.3f} "
          f"rec_fixes_wrong={float(rec_ok[s & (seg_ok == 0)].mean()):.3f} rec_breaks_right={float(1 - rec_ok[s & (seg_ok == 1)].mean()):.3f} purity={float(pur[s].mean()):.3f}")
row("all", torch.ones_like(m[m]))
for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 2)): row(f"area in [{lo:g},{hi:g})", (area >= lo) & (area < hi))
for lo, hi in ((0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)): row(f"purity in [{lo:g},{hi:g})", (pur >= lo) & (pur < hi))
# what a perfect region-level decision would need: is the true class among the recogniser's top-k?
for k in (1, 2, 3, 5): print(f"top-{k} recall of majority class: {float((r.topk(k, 1).indices == y[:, None]).any(1).float().mean()):.3f}")
wrong = seg_ok == 0
print("of regions the segmenter mislabels: recogniser agrees with segmenter", f"{float((top == pc)[wrong].float().mean()):.3f}", "recogniser right", f"{float(rec_ok[wrong].mean()):.3f}",
      "third class", f"{float(((top != pc) & (top != y))[wrong].float().mean()):.3f}")
